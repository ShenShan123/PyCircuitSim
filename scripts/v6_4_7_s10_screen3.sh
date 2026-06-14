#!/usr/bin/env bash
# V6.4.7 S10 (P4) — screen v3: gentler-λ probe at seed 17. screen v2 showed the
# tsmc7 opamp collapses on ALL of λ in {0.02,0.1,0.3} (binary threshold) while
# deriv fidelity improves monotonically. Question: does a gentler λ stay BELOW
# the opamp-collapse threshold while still improving deriv? Configs d/e.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s10_screen3_logs"
mkdir -p "$LOGDIR"; cd "$ROOT"
TECH=tsmc7; SEED=17; GPUS=(0 1 2)
CONFIGS=("d 0.005 --sobolev-strong-boost 4.0" "e 0.01 --sobolev-strong-boost 4.0")
DEVS=(nmos pmos)
jobs=()
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; name="$1"; lam="$2"; shift 2; extra="$*"
  for dev in "${DEVS[@]}"; do jobs+=("$name|$lam|$extra|$dev"); done
done
echo "[s10-screen3] ${#jobs[@]} jobs across GPUs ${GPUS[*]}"
run_queue() {
  local gpu="$1"; shift; local failed=0
  for job in "$@"; do
    IFS='|' read -r name lam extra dev <<<"$job"
    local exp="v6_4_7_s10sob_${name}_${TECH}"; local stem="${exp}_${dev}"
    local log="$LOGDIR/${stem}.log"
    echo "[gpu$gpu] START $stem (λ=$lam $extra) $(date +%H:%M:%S)"
    if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
        PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n pycircuitsim python -u -m bsimar.cli.train \
          --model direct --size medium --device-type "$dev" --tech-scope "$TECH" \
          --cuda --overwrite --data "$DATA/${TECH}_v2_${dev}.npz" \
          --apply-filter off --swa-mode ema \
          --sobolev --lam-sobolev "$lam" $extra \
          --seed "$SEED" --exp-name "$exp" >"$log" 2>&1; then
      echo "[gpu$gpu] DONE  $stem $(date +%H:%M:%S)"
    else echo "[gpu$gpu] FAIL  $stem" >&2; failed=1; fi
  done
  return $failed
}
q0=(); q1=(); q2=()
for i in "${!jobs[@]}"; do case $((i % 3)) in 0) q0+=("${jobs[$i]}");; 1) q1+=("${jobs[$i]}");; 2) q2+=("${jobs[$i]}");; esac; done
run_queue "${GPUS[0]}" "${q0[@]}" & p0=$!
run_queue "${GPUS[1]}" "${q1[@]}" & p1=$!
run_queue "${GPUS[2]}" "${q2[@]}" & p2=$!
rc=0; wait $p0||rc=1; wait $p1||rc=1; wait $p2||rc=1
echo "[s10-screen3] done rc=$rc"; exit $rc
