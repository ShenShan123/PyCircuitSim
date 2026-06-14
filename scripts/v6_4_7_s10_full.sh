#!/usr/bin/env bash
# V6.4.7 S10 (P4) — full Sobolev retrain arm at the screen-winning config.
#
# Runs from-scratch Sobolev retrains for (tech x dev x seed) so each cell has
# >=4 seeds (user ruling), same medium recipe as control-v2 + the winning
# Sobolev flags. Checkpoints: v6_4_7_s10p4_s<seed>_<tech>_<dev> (inert until
# promoted). A/B vs control-v2 (same seeds) isolates the Sobolev term.
#
# Usage:
#   bash scripts/v6_4_7_s10_full.sh "<lam>" "<extra flags>" "<techs csv>" "<seeds csv>"
# e.g.
#   bash scripts/v6_4_7_s10_full.sh 0.1 "--sobolev-corridor-only" \
#        tsmc7,tsmc16 42,17,7,31
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s10_full_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

LAM="${1:?need lam}"
EXTRA="${2:-}"
TECHS_CSV="${3:-tsmc7}"
SEEDS_CSV="${4:-42,17,7,31}"
GPUS=(0 1 2)

IFS=',' read -ra TECHS <<<"$TECHS_CSV"
IFS=',' read -ra SEEDS <<<"$SEEDS_CSV"
DEVS=(nmos pmos)

jobs=()
for tech in "${TECHS[@]}"; do
  for dev in "${DEVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("$tech|$dev|$seed")
    done
  done
done
echo "[s10-full] ${#jobs[@]} jobs (λ=$LAM $EXTRA) across GPUs ${GPUS[*]}"

run_queue() {
  local gpu="$1"; shift
  local failed=0
  for job in "$@"; do
    IFS='|' read -r tech dev seed <<<"$job"
    local exp="v6_4_7_s10p4_s${seed}_${tech}"
    local stem="${exp}_${dev}"
    local log="$LOGDIR/${stem}.log"
    echo "[gpu$gpu] START $stem $(date +%H:%M:%S)"
    if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
        PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n pycircuitsim python -u -m bsimar.cli.train \
          --model direct --size medium \
          --device-type "$dev" --tech-scope "$tech" --cuda --overwrite \
          --data "$DATA/${tech}_v2_${dev}.npz" \
          --apply-filter off --swa-mode ema \
          --sobolev --lam-sobolev "$LAM" $EXTRA \
          --seed "$seed" --exp-name "$exp" \
          >"$log" 2>&1; then
      echo "[gpu$gpu] DONE  $stem $(date +%H:%M:%S)"
    else
      echo "[gpu$gpu] FAIL  $stem (see $log)" >&2
      failed=1
    fi
  done
  return $failed
}

q0=(); q1=(); q2=()
for i in "${!jobs[@]}"; do
  case $((i % 3)) in
    0) q0+=("${jobs[$i]}");;
    1) q1+=("${jobs[$i]}");;
    2) q2+=("${jobs[$i]}");;
  esac
done

run_queue "${GPUS[0]}" "${q0[@]}" & p0=$!
run_queue "${GPUS[1]}" "${q1[@]}" & p1=$!
run_queue "${GPUS[2]}" "${q2[@]}" & p2=$!
rc=0
wait $p0 || rc=1
wait $p1 || rc=1
wait $p2 || rc=1
echo "[s10-full] all queues finished, rc=$rc"
ls "$ROOT/external_compact_models/bsimar/checkpoints/" | grep s10p4 | wc -l
exit $rc
