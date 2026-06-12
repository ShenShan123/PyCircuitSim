#!/usr/bin/env bash
# V6.4.7 S9b — control-v2 retrain: stock medium recipe on the regen-v2 data,
# loader filter OFF, EMA weight averaging (S9), 4 seeds x 8 cells = 32 jobs.
#
# Fan-out per the 2026-06-10 user ruling: seeds run one-per-GPU concurrently
# on GPUs 1/2/3 (GPU 0 is occupied by another user). Jobs are partitioned
# round-robin into three serial queues, one queue per GPU.
#
# Checkpoints land as v6_4_7_ctlv2_s<seed>_<tech>_<dev>_best.pt — deliberately
# NOT matching the parser resolver pattern (inert until promoted).
#
# Usage: bash scripts/v6_4_7_s9b_train_controlv2.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s9b_train_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

SEEDS=(42 17 7 31)
TECHS=(tsmc5 tsmc7 tsmc12 tsmc16)
DEVS=(nmos pmos)
GPUS=(1 2 3)

# Build the flat job list: tech x dev x seed.
jobs=()
for tech in "${TECHS[@]}"; do
    for dev in "${DEVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            jobs+=("$tech $dev $seed")
        done
    done
done
echo "[ctlv2] ${#jobs[@]} jobs across GPUs ${GPUS[*]}"

run_queue() {
    local gpu="$1"; shift
    local failed=0
    for job in "$@"; do
        read -r tech dev seed <<<"$job"
        local name="v6_4_7_ctlv2_s${seed}_${tech}_${dev}"
        local log="$LOGDIR/${name}.log"
        echo "[gpu$gpu] START $name $(date +%H:%M:%S)"
        if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
            conda run -n pycircuitsim python -u -m bsimar.cli.train \
                --model direct --size medium \
                --device-type "$dev" --tech-scope "$tech" --cuda --overwrite \
                --data "$DATA/${tech}_v2_${dev}.npz" \
                --apply-filter off --swa-mode ema \
                --seed "$seed" \
                --exp-name "v6_4_7_ctlv2_s${seed}_${tech}" \
                >"$log" 2>&1; then
            echo "[gpu$gpu] DONE  $name $(date +%H:%M:%S)"
        else
            echo "[gpu$gpu] FAIL  $name (see $log)" >&2
            failed=1
        fi
    done
    return $failed
}

# Round-robin partition into one serial queue per GPU.
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

echo "[ctlv2] all queues finished, rc=$rc"
ls -la "$ROOT/external_compact_models/bsimar/checkpoints/" | grep ctlv2 || true
exit $rc
