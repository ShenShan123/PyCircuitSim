#!/usr/bin/env bash
# V6.4.7 S11 (P3) — subthreshold-id arm: control-v2 stock recipe on the v2
# (regen-v2, unfiltered) data + the SubthresholdIdLoss value+ceiling term.
# Targets the ship-required SRAM force_ic gate. A/B control = v6_4_7_ctlv2_*
# (same seeds, same recipe, v2 data, NO subthreshold term) so each delta
# isolates the loss recipe.
#
# Checkpoints land as v6_4_7_s11sub_w<LAM>_s<seed>_<tech>_<dev>_best.pt —
# deliberately NOT matching the parser resolver pattern (inert until promoted).
#
# Usage:
#   LAM=0.05 CW=1.0 SEEDS="42 17 7 31" TECHS="tsmc7 tsmc5 tsmc12 tsmc16" \
#     bash scripts/v6_4_7_s11_train.sh "0 1 2"
#   (LAM tag in the checkpoint name uses LAM with the dot stripped, e.g. 0.05->005)
set -u
read -r -a GPUS <<<"${1:-0 1 2}"
NG="${#GPUS[@]}"
LAM="${LAM:-0.05}"
CW="${CW:-1.0}"                       # subthresh-ceiling-w
KK="${KK:-1.0}"                       # subthresh-ceiling-k
read -r -a SEEDS <<<"${SEEDS:-42 17 7 31}"
read -r -a TECHS <<<"${TECHS:-tsmc7 tsmc5 tsmc12 tsmc16}"
read -r -a DEVS <<<"${DEVS:-nmos pmos}"
TAG="${TAG:-w$(echo "$LAM" | tr -d '.')}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s11_train_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"

jobs=()
for tech in "${TECHS[@]}"; do
    for dev in "${DEVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            jobs+=("$tech $dev $seed")
        done
    done
done
echo "[s11sub] LAM=$LAM CW=$CW KK=$KK TAG=$TAG  ${#jobs[@]} jobs across GPUs ${GPUS[*]}"

run_queue() {
    local gpu="$1"; shift
    local failed=0
    for job in "$@"; do
        read -r tech dev seed <<<"$job"
        local name="v6_4_7_s11sub_${TAG}_s${seed}_${tech}_${dev}"
        local log="$LOGDIR/${name}.log"
        echo "[gpu$gpu] START $name $(date +%H:%M:%S)"
        if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
            PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
            conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
                --model direct --size medium \
                --device-type "$dev" --tech-scope "$tech" --cuda --overwrite \
                --data "$DATA/${tech}_v2_${dev}.npz" \
                --apply-filter off --swa-mode ema \
                --subthresh --lam-subthresh "$LAM" \
                --subthresh-ceiling-w "$CW" --subthresh-ceiling-k "$KK" \
                --seed "$seed" \
                --exp-name "v6_4_7_s11sub_${TAG}_s${seed}_${tech}" \
                >"$log" 2>&1; then
            echo "[gpu$gpu] DONE  $name $(date +%H:%M:%S)"
        else
            echo "[gpu$gpu] FAIL  $name (see $log)" >&2
            failed=1
        fi
    done
    return $failed
}

declare -a Q
for ((i=0; i<NG; i++)); do Q[$i]=""; done
for i in "${!jobs[@]}"; do
    qi=$((i % NG))
    Q[$qi]="${Q[$qi]}|${jobs[$i]}"
done

pids=()
for ((i=0; i<NG; i++)); do
    IFS='|' read -r -a qjobs <<<"${Q[$i]#|}"
    [ "${#qjobs[@]}" -eq 0 ] && continue
    run_queue "${GPUS[$i]}" "${qjobs[@]}" & pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
echo "[s11sub] all queues finished, rc=$rc"
ls "$ROOT/external_compact_models/bsimar/checkpoints/" | grep -c "s11sub_${TAG}_"
exit $rc
