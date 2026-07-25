#!/usr/bin/env bash
# V6.7.0 universal-transfer campaign — train universal-scope DirectNet
# checkpoints (plan docs/plans/2026-07-02-universal-nn-tsmc5-transfer.md §5).
#
# Every checkpoint is saved as  u716_dn_{recipe}_{size}_{dev}_best.pt  via
# --exp-name. The stem deliberately does NOT start with `tsmc{X}_dn_` (parser
# scope detection → universal vocab, correct tech_code for every tech) and
# does NOT match any resolver fallback name → production per-tech resolution
# is untouched; these checkpoints are reachable only via explicit env pin.
#
# Base recipes (clean, csob) train on uni716_{dev}.npz; curriculum recipes
# (corroft, crit30u) fine-tune from the u716 clean same-size base on
# uni716_corro_{dev}.npz — never from another curriculum (crit30f-stacking
# trap, recipe_train.sh:133 analog).
#
# Completed runs get a `<ckpt>.complete` marker; a bare _best.pt is a killed
# run. Resumable: a job whose _best.pt exists is skipped unless $1 == --force.
#
# Usage:
#   RECIPES="clean csob" bash scripts/uni_train.sh              # wave 1
#   RECIPES="corroft crit30u" bash scripts/uni_train.sh         # wave 2
#   GPUS="1 2" NSTREAMS=4 SIZE=large DEVS="nmos pmos" ...       # defaults
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
DS="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/uni_bench/train_logs"
mkdir -p "$LOGDIR"
read -r -a GPU_IDS <<< "${GPUS:-1 2}"
NGPU=${#GPU_IDS[@]}
NSTREAMS="${NSTREAMS:-4}"
SIZE="${SIZE:-large}"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

recipe_args () {
  case "$1" in
    clean)   echo "" ;;
    csob)    echo "--charge-sobolev" ;;
    corroft) echo "--class-weights traj_corridor=3.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    crit30u) echo "--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40" ;;
    *) echo "__UNKNOWN__" ;;
  esac
}

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  recipe="$2"; dev="$3"; gpu="$4"; force="${5:-}"
  name="u716_dn_${recipe}_${SIZE}_${dev}"
  ckpt="$CKPT/${name}_best.pt"
  log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ "$force" != "--force" ]; then
    [ -f "$ckpt.complete" ] || echo "[uni-train] WARN $name exists WITHOUT completion marker (killed run? --force to retrain)"
    echo "[uni-train] SKIP existing $name"; exit 0
  fi
  extra="$(recipe_args "$recipe")"
  if [ "$extra" = "__UNKNOWN__" ]; then echo "[uni-train] UNKNOWN recipe $recipe"; exit 1; fi
  case "$recipe" in
    corroft|crit30u)
      data="$DS/uni716_corro_${dev}.npz"
      initstem="u716_dn_clean_${SIZE}_${dev}"
      if [ ! -f "$CKPT/${initstem}_best.pt" ]; then
        echo "[uni-train] MISSING init-from ckpt ${initstem}_best.pt"; exit 1; fi
      if [ ! -f "$CKPT/${initstem}_best.pt.complete" ]; then
        echo "[uni-train] ABORT: init-from ${initstem} has no .complete marker (killed run?)"; exit 1; fi
      extra="$extra --init-from ${initstem}" ;;
    *)
      data="$DS/uni716_${dev}.npz" ;;
  esac
  if [ ! -f "$data" ]; then echo "[uni-train] MISSING dataset $data"; exit 1; fi
  echo "[uni-train] START $name on GPU$gpu  (data: $(basename "$data"), extra: ${extra:-<none>})"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size "$SIZE" --device-type "$dev" --tech-scope universal \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite \
    --exp-name "u716_dn_${recipe}_${SIZE}" --data "$data" $extra \
    > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then
    touch "$ckpt.complete"
    echo "[uni-train] DONE $name"
  else
    # audit B3 — a failed train produces NO artifact, so there is nothing for a
    # downstream gate to judge; exit 3 (never 255: xargs aborts the run on 255)
    # so the dispatcher fails loudly instead of printing FAIL and returning 0.
    echo "[uni-train] FAIL $name (rc=$rc, see $log)"
    exit 3
  fi
  exit 0
fi

# ---- dispatcher ----
FORCE="${1:-}"
read -r -a recipes <<< "${RECIPES:-clean csob}"
read -r -a devs    <<< "${DEVS:-nmos pmos}"

# trailing field ALWAYS non-empty so `xargs -L1` never joins lines
# (the benchmark trailing-blank bug).
lines=(); i=0
for recipe in "${recipes[@]}"; do for dev in "${devs[@]}"; do
  lines+=("$recipe $dev ${GPU_IDS[$((i % NGPU))]} ${FORCE:-noforce}")
  i=$((i+1))
done; done

echo "[uni-train] launching ${#lines[@]} jobs (size=$SIZE), $NSTREAMS concurrent across $NGPU GPUs (${GPU_IDS[*]})"
# audit B3 — capture xargs BEFORE the trailing echo, which would otherwise
# overwrite $? with 0 and report a clean sweep over a wave of failed trainings.
printf '%s\n' "${lines[@]}" | SIZE="$SIZE" xargs -P "$NSTREAMS" -L1 "$SELF" _one
xrc=$?
if [ "$xrc" -ne 0 ]; then
  echo "[uni-train] INFRASTRUCTURE FAILURE: some jobs produced no checkpoint (xargs rc=$xrc, see $LOGDIR)" >&2
  exit 1
fi
echo "[uni-train] ALL UNIVERSAL TRAINING COMPLETE"
