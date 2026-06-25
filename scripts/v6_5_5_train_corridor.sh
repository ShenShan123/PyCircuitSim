#!/usr/bin/env bash
# V6.5.5 — train the diagnostic-routed corridor checkpoints.
#   tsmc5  ring-edge corridor  (2a, NMOS conduction under-drive)
#   tsmc7  opamp trip-OP corridor (2b, value-surface gain->0)
# Proven V6.4.7-S12 recipe: --size medium --apply-filter off --swa-mode ema
#   --class-weights traj_corridor=W  (the corridor is ~0.3-1% of rows; up-weight).
# Checkpoints land in external_compact_models/bsimar/checkpoints/ as
#   tsmc{5,7}_dn_cor{ring,opamp}_s{seed}_{dev}_best.pt  (the tsmc{X}_dn_ prefix is
#   REQUIRED so the inference resolver detects per-tech local vocab).
#
# Usage:  bash scripts/v6_5_5_train_corridor.sh [W] [gpu list]
set -u
W="${1:-3}"
read -r -a GPUS <<<"${2:-0 1 2}"
SIZE="${3:-medium}"
SZT=""; [ "$SIZE" = "large" ] && SZT="L"   # exp-name suffix so sizes don't collide
NG="${#GPUS[@]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_5_5/train_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

SEEDS=(42 17 7)
# job spec: "tech expname_tag dev seed"
jobs=()
for seed in "${SEEDS[@]}"; do
  for dev in nmos pmos; do
    jobs+=("tsmc5 corring  $dev $seed")
    jobs+=("tsmc7 coropamp $dev $seed")
  done
done

echo "[v6.5.5] W=$W  ${#jobs[@]} jobs across GPUs ${GPUS[*]}"

run_queue() {
  local gpu="$1"; shift
  for job in "$@"; do
    read -r tech tag dev seed <<<"$job"
    local name="${tech}_dn_${tag}${SZT}_s${seed}"
    local log="$LOGDIR/${name}_${dev}.log"
    echo "[gpu$gpu] START $name $dev $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
      PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
      conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
        --model direct --size "$SIZE" \
        --device-type "$dev" --tech-scope "$tech" --cuda --overwrite \
        --data "$DATA/${tech}_cor_${dev}.npz" \
        --apply-filter off --swa-mode ema \
        --class-weights "traj_corridor=${W}" \
        --seed "$seed" \
        --exp-name "${tech}_dn_${tag}${SZT}_s${seed}" \
        >"$log" 2>&1 \
      && echo "[gpu$gpu] DONE  $name $dev $(date +%H:%M:%S)" \
      || echo "[gpu$gpu] FAIL  $name $dev (see $log)"
  done
}

# round-robin jobs onto GPU queues; 2 concurrent queues per GPU
declare -a Q
nq=$((NG * 2))
for ((i=0; i<${#jobs[@]}; i++)); do
  qi=$((i % nq)); Q[$qi]+="${jobs[$i]}|"
done
pids=()
for ((qi=0; qi<nq; qi++)); do
  gpu="${GPUS[$((qi % NG))]}"
  IFS='|' read -r -a qjobs <<<"${Q[$qi]}"
  run_queue "$gpu" "${qjobs[@]}" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[v6.5.5] all training done $(date +%H:%M:%S)"
