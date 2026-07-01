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
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
DS="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/recipe_bench/train_logs"
mkdir -p "$LOGDIR"
read -r -a GPU_IDS <<< "${GPUS:-0 1 2}"
NGPU=${#GPU_IDS[@]}
NSTREAMS="${NSTREAMS:-9}"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

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
    *) echo "__UNKNOWN__" ;;
  esac
}

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  recipe="$2"; tech="$3"; size="$4"; dev="$5"; gpu="$6"; force="${7:-}"
  name="${tech}_dn_${recipe}_${size}_${dev}"
  ckpt="$CKPT/${name}_best.pt"
  log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ "$force" != "--force" ]; then echo "[train] SKIP existing $name"; exit 0; fi
  extra="$(recipe_args "$recipe")"
  if [ "$extra" = "__UNKNOWN__" ]; then echo "[train] UNKNOWN recipe $recipe"; exit 1; fi
  echo "[train] START $name on GPU$gpu  (extra: ${extra:-<none>})"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size "$size" --device-type "$dev" --tech-scope "$tech" \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite \
    --exp-name "${tech}_dn_${recipe}_${size}" $extra \
    > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then echo "[train] DONE $name"; else echo "[train] FAIL $name (rc=$rc, see $log)"; fi
  exit 0
fi

# ---- dispatcher ----
FORCE="${1:-}"
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
lines=(); i=0
for size in "${sizes[@]}"; do for recipe in "${recipes[@]}"; do for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  lines+=("$recipe $tech $size $dev ${GPU_IDS[$((i % NGPU))]} ${FORCE:-noforce}")
  i=$((i+1))
done; done; done; done

echo "[train] launching ${#lines[@]} jobs, $NSTREAMS concurrent across $NGPU GPUs (${GPU_IDS[*]})"
printf '%s\n' "${lines[@]}" | xargs -P "$NSTREAMS" -L1 "$SELF" _one
echo "[train] ALL RECIPE TRAINING COMPLETE"
