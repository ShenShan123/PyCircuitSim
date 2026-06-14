#!/usr/bin/env bash
# V6.4.7 S10 (P4) — screen v2: FROM-SCRATCH Sobolev retrains at seed 17.
#
# Why from-scratch (not fine-tune): warm-starting a value-converged net and
# selecting on plain val-MAE always REVERTS the slope correction (screen v1
# confirmed: λ=0.1 degraded val-MAE 4x and early-stopped at epoch 2). A
# from-scratch train integrates the slope objective across the whole
# trajectory, so the best-val epoch carries the Sobolev shaping.
#
# Clean A/B: seed 17 == control-v2 s17 → identical weight init + data split +
# normalizer fit; the ONLY difference is the Sobolev term. control-v2 s17 is
# the direct baseline (opamp 10.46% gain 180>NG163, RO 8.66%, pmos gds_fwd
# 55.8%). Same medium recipe (200ep/patience40/lr1e-3, EMA, v2, filter off).
#
# Fan-out: 6 jobs (3 configs x {nmos,pmos}) round-robin across GPUs 0/1/2.
# Usage: bash scripts/v6_4_7_s10_screen2.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s10_screen2_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

TECH=tsmc7
SEED=17
GPUS=(0 1 2)

# name  lam  extra-flags
CONFIGS=(
  "a 0.02 --sobolev-strong-boost 4.0"
  "b 0.1  --sobolev-corridor-only"
  "c 0.3  --sobolev-corridor-only"
)
DEVS=(nmos pmos)

jobs=()
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; name="$1"; lam="$2"; shift 2; extra="$*"
  for dev in "${DEVS[@]}"; do
    jobs+=("$name|$lam|$extra|$dev")
  done
done
echo "[s10-screen2] ${#jobs[@]} from-scratch Sobolev retrains across GPUs ${GPUS[*]}"

run_queue() {
  local gpu="$1"; shift
  local failed=0
  for job in "$@"; do
    IFS='|' read -r name lam extra dev <<<"$job"
    local exp="v6_4_7_s10sob_${name}_${TECH}"
    local stem="${exp}_${dev}"
    local log="$LOGDIR/${stem}.log"
    echo "[gpu$gpu] START $stem (λ=$lam $extra) $(date +%H:%M:%S)"
    if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
        PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n pycircuitsim python -u -m bsimar.cli.train \
          --model direct --size medium \
          --device-type "$dev" --tech-scope "$TECH" --cuda --overwrite \
          --data "$DATA/${TECH}_v2_${dev}.npz" \
          --apply-filter off --swa-mode ema \
          --sobolev --lam-sobolev "$lam" $extra \
          --seed "$SEED" --exp-name "$exp" \
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
echo "[s10-screen2] all queues finished, rc=$rc"
ls -la "$ROOT/external_compact_models/bsimar/checkpoints/" | grep s10sob || true
exit $rc
