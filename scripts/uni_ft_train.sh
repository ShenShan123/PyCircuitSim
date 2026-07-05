#!/usr/bin/env bash
# V6.7.0 universal-transfer campaign — Phase 3: TSMC5 fine-tune tiers
# (plan docs/plans/2026-07-02-universal-nn-tsmc5-transfer.md §7).
#
# Fine-tunes the universal CLEAN base (u716_dn_clean_${SIZE}_${dev}) on the
# stratified TSMC5 tiers tsmc5ft_n{N}_{dev}.npz (+ the full corro file as the
# "nfull" tier). Checkpoints: u716f5_{plain,crit}_n{N}_${SIZE}_{dev}. Stays at
# --tech-scope universal so the 18-row embedding matches --init-from (vocab
# hard-fail otherwise); TSMC5 rows carry universal codes 0-3 — the fine-tune
# trains those previously-unused embedding rows.
#
# plain: no class weights.  crit: --class-weights traj_corridor=3.0,inv_trip=2.0.
# Completed runs get a `<ckpt>.complete` marker; existing _best.pt is skipped
# unless $1 == --force (same conventions as uni_train.sh).
#
# Usage:
#   bash scripts/uni_ft_train.sh                       # all 24 runs
#   TIERS="2000 10000" FTRECIPES="plain" ...           # subset
#   GPUS="1 2" NSTREAMS=4 SIZE=large DEVS="nmos pmos"  # defaults
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

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  ftrecipe="$2"; tier="$3"; dev="$4"; gpu="$5"; force="${6:-}"
  name="u716f5_${ftrecipe}_n${tier}_${SIZE}_${dev}"
  ckpt="$CKPT/${name}_best.pt"
  log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ "$force" != "--force" ]; then
    [ -f "$ckpt.complete" ] || echo "[uni-ft] WARN $name exists WITHOUT completion marker (killed run? --force to retrain)"
    echo "[uni-ft] SKIP existing $name"; exit 0
  fi
  case "$ftrecipe" in
    plain) extra="" ;;
    crit)  extra="--class-weights traj_corridor=3.0,inv_trip=2.0" ;;
    *) echo "[uni-ft] UNKNOWN ftrecipe $ftrecipe"; exit 1 ;;
  esac
  if [ "$tier" = "full" ]; then
    data="$DS/tsmc5_corro_${dev}.npz"
  else
    data="$DS/tsmc5ft_n${tier}_${dev}.npz"
  fi
  initstem="u716_dn_clean_${SIZE}_${dev}"
  if [ ! -f "$CKPT/${initstem}_best.pt" ]; then
    echo "[uni-ft] MISSING init-from ckpt ${initstem}_best.pt"; exit 1; fi
  if [ ! -f "$CKPT/${initstem}_best.pt.complete" ]; then
    echo "[uni-ft] ABORT: init-from ${initstem} has no .complete marker (killed run?)"; exit 1; fi
  if [ ! -f "$data" ]; then echo "[uni-ft] MISSING dataset $data"; exit 1; fi
  echo "[uni-ft] START $name on GPU$gpu  (data: $(basename "$data"), extra: ${extra:-<none>})"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size "$SIZE" --device-type "$dev" --tech-scope universal \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite \
    --lr 3e-4 --epochs 120 --patience 40 \
    --init-from "$initstem" \
    --exp-name "u716f5_${ftrecipe}_n${tier}_${SIZE}" --data "$data" $extra \
    > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then
    touch "$ckpt.complete"
    echo "[uni-ft] DONE $name"
  else
    echo "[uni-ft] FAIL $name (rc=$rc, see $log)"
  fi
  exit 0
fi

# ---- dispatcher ----
FORCE="${1:-}"
read -r -a ftrecipes <<< "${FTRECIPES:-plain crit}"
read -r -a tiers    <<< "${TIERS:-2000 10000 50000 200000 1000000 full}"
read -r -a devs     <<< "${DEVS:-nmos pmos}"

# trailing field ALWAYS non-empty so `xargs -L1` never joins lines
lines=(); i=0
for ftrecipe in "${ftrecipes[@]}"; do for tier in "${tiers[@]}"; do for dev in "${devs[@]}"; do
  lines+=("$ftrecipe $tier $dev ${GPU_IDS[$((i % NGPU))]} ${FORCE:-noforce}")
  i=$((i+1))
done; done; done

echo "[uni-ft] launching ${#lines[@]} jobs (size=$SIZE), $NSTREAMS concurrent across $NGPU GPUs (${GPU_IDS[*]})"
printf '%s\n' "${lines[@]}" | SIZE="$SIZE" xargs -P "$NSTREAMS" -L1 "$SELF" _one
echo "[uni-ft] ALL FINE-TUNE TRAINING COMPLETE"
