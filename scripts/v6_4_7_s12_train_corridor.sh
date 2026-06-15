#!/usr/bin/env bash
# V6.4.7 S12 (P5) — trajectory-corridor arm: control-v2 stock recipe on the
# v2cor data (v2 + traj_corridor overlay), filter OFF, EMA, with the corridor
# class upweighted via --class-weights traj_corridor=$W. 4 seeds x 8 cells.
#
# A/B control = v6_4_7_ctlv2_* (same seeds, same recipe, v2 data WITHOUT the
# corridor) so each delta isolates the corridor data + weight.
#
# Checkpoints land as v6_4_7_s12cor_w<W>_s<seed>_<tech>_<dev>_best.pt —
# deliberately NOT matching the parser resolver pattern (inert until promoted).
#
# Usage: bash scripts/v6_4_7_s12_train_corridor.sh <W> "<gpu0 gpu1 gpu2>"
#   e.g. bash scripts/v6_4_7_s12_train_corridor.sh 3 "0 2"
set -u
W="${1:?usage: $0 <class_weight_W> \"<gpu list>\"}"
read -r -a GPUS <<<"${2:-0 1 2}"
NG="${#GPUS[@]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s12_train_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"

SEEDS=(42 17 7 31)
# RO-relevant techs first: the S12 kill gate is RO < 7 % (tsmc7 8.28 %, tsmc5),
# so tsmc7/tsmc5 checkpoints should finish before tsmc12/16 to surface the
# kill/keep signal soonest under the heavy shared-machine load.
TECHS=(tsmc7 tsmc5 tsmc12 tsmc16)
DEVS=(nmos pmos)

jobs=()
for tech in "${TECHS[@]}"; do
    for dev in "${DEVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            jobs+=("$tech $dev $seed")
        done
    done
done
echo "[s12cor] W=$W  ${#jobs[@]} jobs across GPUs ${GPUS[*]}"

run_queue() {
    local gpu="$1"; shift
    local failed=0
    for job in "$@"; do
        read -r tech dev seed <<<"$job"
        local name="v6_4_7_s12cor_w${W}_s${seed}_${tech}_${dev}"
        local log="$LOGDIR/${name}.log"
        echo "[gpu$gpu] START $name $(date +%H:%M:%S)"
        if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
            PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
            conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
                --model direct --size medium \
                --device-type "$dev" --tech-scope "$tech" --cuda --overwrite \
                --data "$DATA/${tech}_v2cor_${dev}.npz" \
                --apply-filter off --swa-mode ema \
                --class-weights "traj_corridor=${W}" \
                --seed "$seed" \
                --exp-name "v6_4_7_s12cor_w${W}_s${seed}_${tech}" \
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
declare -a Q
for ((i=0; i<NG; i++)); do Q[$i]=""; done
for i in "${!jobs[@]}"; do
    qi=$((i % NG))
    Q[$qi]="${Q[$qi]}|${jobs[$i]}"
done

pids=()
for ((i=0; i<NG; i++)); do
    IFS='|' read -r -a qjobs <<<"${Q[$i]#|}"
    run_queue "${GPUS[$i]}" "${qjobs[@]}" & pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
echo "[s12cor] all queues finished, rc=$rc"
ls -la "$ROOT/external_compact_models/bsimar/checkpoints/" | grep "s12cor_w${W}" | wc -l
exit $rc
