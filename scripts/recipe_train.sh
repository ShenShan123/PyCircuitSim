#!/usr/bin/env bash
# V6.6.1 recipe campaign — train DirectNet recipe variants for the
# recipe-comparison accuracy study (charge-Sobolev / Sobolev / EKV / ...).
#
# Every checkpoint is saved as  tsmc{X}_dn_{recipe}_{size}_{dev}_best.pt  via
# --exp-name, so:
#   * the `tsmc{X}_dn_` prefix keeps the parser's per-tech LOCAL-vocab scope
#     detection (Rule 16) — a different prefix would silently fall back to the
#     universal vocab and corrupt the tech_code at inference;
#   * the production `tsmc{X}_dn_{size}_{dev}` (clean) checkpoints are NEVER
#     clobbered — the clean recipe is the control and stays on disk untouched.
#
# Everything else mirrors the proven clean recipe used by the V6.6.0 benchmark:
#     --apply-filter off --swa-mode ema --seed 42
# plus the per-recipe addendum from the RECIPES map below. Capacity (--size) and
# the recipe are the only variables vs the clean control.
#
# The 4090s sit near-idle on these tiny nets (data-loader bound, _NUM_WORKERS=8),
# so NSTREAMS jobs run concurrently, GPU round-robin (job index % NGPU).
# Resumable: a job whose _best.pt exists is skipped unless $1 == --force.
#
# Usage:
#   RECIPES="csob sob ekv" SIZES="large xl" NSTREAMS=9 bash scripts/recipe_train.sh
#   GPUS="0 1 2" TECHS="tsmc5 tsmc7 tsmc12 tsmc16" DEVS="nmos pmos" ...
#
# V6.8 — MODEL env selects the architecture: MODEL=direct (default, byte-
# identical legacy behavior, `_dn_` stems) or MODEL=transformer (BSIMAR,
# `_tf_` stems; the parser's LEVEL=74 preempt cascade keys off `tsmc{X}_tf_`
# for the same Rule-16 local-vocab scope detection). The recipe map is
# shared; ekv/monotonic recipes stay DirectNet-only (the CLI rejects them).
# `clean` trains WITHOUT --exp-name so it lands on the production slot
# `tsmc{X}_{dn,tf}_{size}_{dev}` that the resolver preempt reads.
# EXTRA_ARGS env appends to every job (e.g. EXTRA_ARGS="--amp").
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: [ENV=VALUE ...] bash scripts/recipe_train.sh [--force]

Environment:
  MODEL        direct | transformer (default: direct)
  RECIPES      space-separated recipes (use clean for clean checkpoints)
  TECHS        space-separated lowercase technology names
  SIZES        space-separated small | medium | large | xl
  DEVS         space-separated nmos | pmos
  GPUS         space-separated physical GPU IDs
  NSTREAMS     maximum concurrent training jobs
  TRAIN_OMP    CPU threads per training job
  EXTRA_ARGS   extra neural_network.cli.train arguments
  BSIMAR_CHECKPOINT_DIR  checkpoint output/selection directory
  RECIPE_TRAIN_LOG_DIR   training-log directory
EOF
  exit 0
fi
CKPT="${BSIMAR_CHECKPOINT_DIR:-$ROOT/external_compact_models/neural_network/checkpoints}"
DS="$ROOT/external_compact_models/neural_network/data/datasets"
LOGDIR="${RECIPE_TRAIN_LOG_DIR:-$ROOT/results/recipe_bench/train_logs}"
mkdir -p "$CKPT" "$LOGDIR"
read -r -a GPU_IDS <<< "${GPUS:-0 1 2}"
NGPU=${#GPU_IDS[@]}
NSTREAMS="${NSTREAMS:-9}"
MODEL="${MODEL:-direct}"
case "$MODEL" in
  direct)      TAG="dn" ;;
  transformer) TAG="tf" ;;
  *) echo "[train] UNKNOWN MODEL=$MODEL (direct|transformer)"; exit 1 ;;
esac
export MODEL
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

required_sidecars_exist () {
  local stem="$1"
  [ -f "$CKPT/${stem}_norm.npz" ] || return 1
  case "$TAG" in
    dn) return 0 ;;
    tf) [ -f "$CKPT/${stem}_config.npz" ] ;;
  esac
}

# ── recipe → extra train args. 'clean' is the control (no addendum); it is
#    normally NOT retrained here (the production tsmc{X}_dn_{size} ckpts serve
#    as the clean control), but it is defined for completeness / re-baselining.
recipe_args () {
  case "$1" in
    clean) echo "" ;;
    csob)  echo "--charge-sobolev" ;;                       # cap dQ/dV consistency
    sob)   echo "--sobolev --sobolev-corridor-only" ;;      # id dI/dV slope (corridor)
    sobf)  echo "--sobolev" ;;                              # id dI/dV slope (full)
    ekv)   echo "--ekv-core" ;;                             # EKV analytic backbone
    ekvhi) echo "--ekv-core --ekv-lam-lo 0.12" ;;           # EKV + raised r_o floor (opamp)
    s7)    echo "--seed 7" ;;                               # clean recipe, seed 7 (opamp basin probe)
    s17)   echo "--seed 17" ;;                              # clean recipe, seed 17
    s123)  echo "--seed 123" ;;                             # clean recipe, seed 123
    cs7)   echo "--charge-sobolev --seed 7" ;;             # csob + seed 7 (smooth-axis winner + basin probe)
    e3)    echo "--loss-preset e3" ;;                       # down-weight non-load-bearing
    csobekv) echo "--charge-sobolev --ekv-core" ;;          # combo
    # ── V6.6.2 uniform-recipe-beyond-13/16 (docs/plans/2026-07-01) ──
    #   inv_trip = sample_class 7 (per-tech Vth-centered trip band, already in
    #   every dataset). Upweighting it (LR-neutral, renormalized to unit mean)
    #   tightens the id-value + dQ/dV surface at the ~0.65V moderate-inversion
    #   switching edge where the tsmc5 ring under-drives.
    invtrip)   echo "--class-weights inv_trip=2.0" ;;                                     # from-scratch bake-off arm (800ep, seed42)
    invtripft) echo "--class-weights inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;; # curriculum warm-start (init-from injected in _one)
    # ── V6.6.2 trajectory-corridor (the real tsmc5-ring lever; V6.5.5-proven) ──
    #   Trains on the traj_corridor-augmented {tech}_cor_{dev}.npz (harvested bias
    #   tube along each tech's OWN ground-truth ring/opamp/SC/SRAM trajectories).
    #   weight=3.0 matches V6.4.7-S12 (commit d61049a). --data injected in _one.
    cor|corr|corro)     echo "--class-weights traj_corridor=3.0" ;;                                     # from-scratch corridor (full|ring+SC|ring-only)
    corft|corrft|corroft) echo "--class-weights traj_corridor=3.0 --lr 3e-4 --epochs 120 --patience 40" ;; # curriculum (full|ring+SC|ring-only)
    # ring-only curriculum at HALF weight (1.5): opens the ring (margin) while the
    # gentler perturbation preserves the delicate opamp OP that weight=3.0 railed.
    corro15) echo "--class-weights traj_corridor=1.5 --lr 3e-4 --epochs 120 --patience 40" ;;
    # ── V6.6.2 COMBO — ring-only corridor + inv_trip (the untested cross-wall pair) ──
    #   The corridor opens tsmc5-ring but DRIFTS an opamp (tsmc5@w3.0, tsmc12@w1.5);
    #   inv_trip (class 7) is the proven opamp-MARGIN holder (invtripft: tsmc12-opamp
    #   6.25→0.35, tsmc5-opamp 2.10→0.67). Combining them on the corro dataset (which
    #   carries BOTH class 7 and class 12) should open the ring while the inv_trip
    #   anchor holds the opamp the ring-corridor would otherwise drift. curriculum
    #   warm-start (init-from own clean large), ring-only corro data injected in _one.
    crit15) echo "--class-weights traj_corridor=1.5,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    crit30) echo "--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    # crit30f = crit30 rerun to COMPLETION (the original crit30 stragglers were
    # killed at heterogeneous epochs 30-92, so the on-disk crit30 artifact is not
    # a uniformly-executed recipe; crit30f is the honest-contract validation).
    crit30f) echo "--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    crit20) echo "--class-weights traj_corridor=2.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    # Round-2: STRONGER inv_trip opamp-anchor at the w1.5 corridor that opens the
    # ring. crit15 left tsmc12-opamp MULTISTABLE (bare corro15 killed it dead at
    # 100%); more inv_trip weight should tip its (existing) high-gain basin to a
    # deterministic pass → 15/16 target.
    crit15m) echo "--class-weights traj_corridor=1.5,inv_trip=3.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    crit15h) echo "--class-weights traj_corridor=1.5,inv_trip=4.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    # Gentler corridor (w1.0) + inv_trip: perturbs tsmc12-opamp less; ring may
    # sit nearer the 5% edge — margin-gated.
    crit10) echo "--class-weights traj_corridor=1.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    # ── V6.6.7 15/16 candidates (plan §5 P2a + the corroft/crit30 anchor asymmetry) ──
    #   csobcrit: csob base (the only large recipe holding the tsmc16-opamp basin)
    #   + the proven crit30 curriculum (holds tsmc5+tsmc12). charge-sobolev stays ON
    #   during the fine-tune so the cap surface that banks tsmc16 is preserved.
    #   init-from csob (injected in _one), corro data.
    csobcrit) echo "--charge-sobolev --class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    #   crit30a1: corridor w3.0 with a HALF anchor — at large, anchor 0 (corroft)
    #   holds {tsmc16-opamp} and anchor 2.0 (crit30) holds {tsmc5,tsmc12}; the
    #   intermediate probes the 5+12+16 triple.
    crit30a1) echo "--class-weights traj_corridor=3.0,inv_trip=1.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    *) echo "__UNKNOWN__" ;;
  esac
}

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  recipe="$2"; tech="$3"; size="$4"; dev="$5"; gpu="$6"; force="${7:-}"
  if [ "$recipe" = "clean" ]; then
    # clean = the production slot (default save_prefix, no --exp-name).
    name="${tech}_${TAG}_${size}_${dev}"
  else
    name="${tech}_${TAG}_${recipe}_${size}_${dev}"
  fi
  ckpt="$CKPT/${name}_best.pt"
  log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ -f "$ckpt.complete" ] \
      && required_sidecars_exist "$name" && [ "$force" != "--force" ]; then
    echo "[train] SKIP existing $name"; exit 0
  fi
  if [ -f "$ckpt" ] && [ "$force" != "--force" ]; then
    echo "[train] RETRAIN incomplete $name (checkpoint, sidecar, or completion marker missing)"
  fi
  extra="$(recipe_args "$recipe")"
  if [ "$extra" = "__UNKNOWN__" ]; then echo "[train] UNKNOWN recipe $recipe"; exit 1; fi
  # Curriculum (warm-start) recipes — the "*ft" family fine-tunes from its OWN
  # clean same-size checkpoint. Locality (init-from) is THE basin-preserving
  # lever (plan §2/§4). Injected here (not in recipe_args) because it is
  # per-(tech,dev) yet MECHANICALLY DETERMINED (same rule everywhere) → stays
  # inside the honest-uniform contract (plan §10).
  case "$recipe" in
    csobcrit) initstem="${tech}_${TAG}_csob_${size}_${dev}"
         if [ ! -f "$CKPT/${initstem}_best.pt" ]; then
           echo "[train] MISSING init-from ckpt $CKPT/${initstem}_best.pt"; exit 1; fi
         extra="$extra --init-from ${initstem}" ;;
    *ft|corro15|crit*)
         # own CLEAN same-size base. Post-V6.6.4 the production DirectNet
         # large slots carry crit30f, so at large the DN clean base is the
         # v660clean archive — init-from production would silently stack
         # curricula. The Transformer production slots ARE the clean bases.
         initstem="${tech}_${TAG}_${size}_${dev}"
         if [ "$TAG" = "dn" ] && [ "$size" = "large" ] && [ -f "$CKPT/${tech}_dn_v660clean_large_${dev}_best.pt" ]; then
           initstem="${tech}_dn_v660clean_large_${dev}"
         fi
         if [ ! -f "$CKPT/${initstem}_best.pt" ]; then
           echo "[train] MISSING init-from ckpt $CKPT/${initstem}_best.pt"; exit 1; fi
         extra="$extra --init-from ${initstem}" ;;
  esac
  # V6.6.2 corridor recipes ("cor", "corft") train on the traj_corridor-augmented
  # dataset {tech}_cor_{dev}.npz (from v6_4_7_s12_append_corridors.py). Uniform:
  # every (tech,dev) trains on its OWN corridor set — mechanical, not hand-picked.
  case "$recipe" in
    crit*|csobcrit) cordata="$DS/${tech}_corro_${dev}.npz"   # combo always uses ring-only corridor data
          if [ ! -f "$cordata" ]; then echo "[train] MISSING corridor dataset $cordata"; exit 1; fi
          extra="$extra --data ${cordata}" ;;
    cor*) corvar="${recipe/ft/}"   # cor->cor, corft->cor, corr->corr, corrft->corr
          corvar="${corvar%%[0-9]*}"  # strip weight suffix: corro15->corro
          cordata="$DS/${tech}_${corvar}_${dev}.npz"
          if [ ! -f "$cordata" ]; then echo "[train] MISSING corridor dataset $cordata"; exit 1; fi
          extra="$extra --data ${cordata}" ;;
  esac
  extra="$extra ${EXTRA_ARGS:-}"
  expname=""
  if [ "$recipe" != "clean" ]; then
    expname="--exp-name ${tech}_${TAG}_${recipe}_${size}"
  fi
  echo "[train] START $name on GPU$gpu  (model: $MODEL, extra: ${extra:-<none>})"
  # V6.8: pin CPU threads per job. Un-pinned, each torch process spawns a
  # near-full-core OpenMP pool; at NSTREAMS=6-9 concurrent jobs the box hit
  # loadavg ~400/192 with GPUs starved at 56-78% util. TRAIN_OMP=4 default.
  rm -f "$ckpt.complete"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="${TRAIN_OMP:-4}" MKL_NUM_THREADS="${TRAIN_OMP:-4}" \
    conda run --no-capture-output -n pycircuitsim python -u -m neural_network.cli.train \
    --model "$MODEL" --size "$size" --device-type "$dev" --tech-scope "$tech" \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite \
    $expname $extra \
    > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ] && required_sidecars_exist "$name"; then
    if ! touch "$ckpt.complete"; then
      echo "[train] LIFECYCLE ERROR $name: could not create completion marker" >&2
      exit 3
    fi
    echo "[train] DONE $name"
  else
    rm -f "$ckpt.complete"
    # audit B3 — a failed train produces NO artifact, so there is nothing for a
    # downstream gate to judge; exit 3 (never 255: xargs aborts the run on 255)
    # so the dispatcher fails loudly instead of printing FAIL and returning 0.
    echo "[train] FAIL $name (rc=$rc, see $log)"
    exit 3
  fi
  exit 0
fi

# ---- dispatcher ----
if [ "$#" -gt 1 ]; then
  echo "[train] UNKNOWN arguments: $*" >&2
  echo "Usage: [ENV=VALUE ...] bash scripts/recipe_train.sh [--force]" >&2
  exit 2
fi
FORCE="${1:-}"
if [ -n "$FORCE" ] && [ "$FORCE" != "--force" ]; then
  echo "[train] UNKNOWN argument: $FORCE (expected --force or --help)" >&2
  exit 2
fi
read -r -a recipes <<< "${RECIPES:-csob sob ekv}"
read -r -a techs   <<< "${TECHS:-tsmc5 tsmc7 tsmc12 tsmc16}"
read -r -a sizes   <<< "${SIZES:-large xl}"
read -r -a devs    <<< "${DEVS:-nmos pmos}"

missing=0
for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  [ -f "$DS/${tech}_${dev}.npz" ] || { echo "[train] MISSING dataset $DS/${tech}_${dev}.npz"; missing=1; }
done; done
[ "$missing" -eq 0 ] || { echo "[train] ABORT: datasets incomplete"; exit 1; }

# size-major (large first → opamp signal first), then recipe, tech, dev.
# trailing field ALWAYS non-empty (noforce placeholder) so `xargs -L1` never
# joins lines (the benchmark trailing-blank bug).
#
# BALANCED round-robin (V6.8.1): the GPU index advances ONLY for jobs that will
# actually train. An already-`.complete` job would SKIP inside _one but, with a
# naive `i % NGPU` over ALL jobs, it still consumes its GPU slot — so on a resume
# where e.g. every tsmc7-*-nmos is done, one GPU is starved while the others run
# 3 jobs each (observed 3:3:1 on the reboot-resume of the xl clean wave). Skipping
# completed jobs before assignment keeps the survivors evenly spread. A fresh run
# (no .complete) skips nothing → byte-identical to the old assignment. GPU choice
# never affects results (seeds are pinned), so this only rebalances wall-clock.
lines=(); i=0
for size in "${sizes[@]}"; do for recipe in "${recipes[@]}"; do for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  if [ "$recipe" = clean ]; then _nm="${tech}_${TAG}_${size}_${dev}"; else _nm="${tech}_${TAG}_${recipe}_${size}_${dev}"; fi
  if [ -f "$CKPT/${_nm}_best.pt" ] \
      && [ -f "$CKPT/${_nm}_best.pt.complete" ] \
      && required_sidecars_exist "$_nm" \
      && [ "$FORCE" != "--force" ]; then continue; fi
  lines+=("$recipe $tech $size $dev ${GPU_IDS[$((i % NGPU))]} ${FORCE:-noforce}")
  i=$((i+1))
done; done; done; done
if [ "${#lines[@]}" -eq 0 ]; then echo "[train] nothing to do — all requested jobs already .complete"; exit 0; fi

echo "[train] launching ${#lines[@]} jobs, $NSTREAMS concurrent across $NGPU GPUs (${GPU_IDS[*]})"
# audit B3 — capture xargs BEFORE the trailing echo, which would otherwise
# overwrite $? with 0 and report a clean sweep over a wave of failed trainings.
printf '%s\n' "${lines[@]}" | xargs -P "$NSTREAMS" -L1 "$SELF" _one
xrc=$?
if [ "$xrc" -ne 0 ]; then
  echo "[train] INFRASTRUCTURE FAILURE: some jobs produced no checkpoint (xargs rc=$xrc, see $LOGDIR)" >&2
  exit 1
fi
echo "[train] ALL RECIPE TRAINING COMPLETE"
