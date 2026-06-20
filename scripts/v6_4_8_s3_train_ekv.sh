#!/usr/bin/env bash
# V6.4.8 S3 — EKV analytic-backbone retrain. EXACT control-v2 recipe
# (medium, regen-v2 data, --apply-filter off, EMA) + the single delta
# --ekv-core, so the A/B isolates the EKV core. Checkpoints land as
# v6_4_8_s3ekv_s<seed>_<tech>_<dev>_best.pt (inert until promoted into the
# tsmc{X}_dn_medium resolver slot).
#
# Usage: bash scripts/v6_4_8_s3_train_ekv.sh [tech]   (default tsmc5)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_8/s3_train_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

TECH="${1:-tsmc5}"
SEEDS=(17 42 31)
DEVS=(nmos pmos)
GPUS=(0 1 2)

jobs=()
for seed in "${SEEDS[@]}"; do
    for dev in "${DEVS[@]}"; do
        jobs+=("$dev $seed")
    done
done
echo "[s3ekv:$TECH] ${#jobs[@]} jobs across GPUs ${GPUS[*]}"

run_queue() {
    local gpu="$1"; shift
    local failed=0
    for job in "$@"; do
        read -r dev seed <<<"$job"
        local name="v6_4_8_s3ekv_s${seed}_${TECH}_${dev}"
        local log="$LOGDIR/${name}.log"
        echo "[gpu$gpu] START $name $(date +%H:%M:%S)"
        if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
            PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
            conda run -n pycircuitsim python -u -m bsimar.cli.train \
                --model direct --size medium \
                --device-type "$dev" --tech-scope "$TECH" --cuda --overwrite \
                --data "$DATA/${TECH}_v2_${dev}.npz" \
                --apply-filter off --swa-mode ema --ekv-core \
                --seed "$seed" \
                --exp-name "v6_4_8_s3ekv_s${seed}_${TECH}" \
                >"$log" 2>&1; then
            echo "[gpu$gpu] DONE  $name $(date +%H:%M:%S)"
        else
            echo "[gpu$gpu] FAIL  $name (see $log)" >&2
            failed=1
        fi
    done
    return $failed
}

# Round-robin into one serial queue per GPU.
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
echo "[s3ekv:$TECH] all queues finished, rc=$rc $(date +%H:%M:%S)"
exit $rc
