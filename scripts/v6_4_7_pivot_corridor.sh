#!/usr/bin/env bash
# V6.4.7 pivot — gentle-dose corridor W-sweep on a tech (the deferred S12
# W-sweep). Trains the v2cor (corridor) data at a low --class-weights
# traj_corridor=$W to find a dose that fixes RO WITHOUT collapsing the
# value-owned opamp. A/B target = the per-tech 14/16 mix.
#
# Checkpoints: v6_4_7_pivcor_w<W>_s<seed>_<tech>_<dev>_best.pt (inert).
#
# Usage: W=1 SEEDS="7 31" TECHS="tsmc7" DEVS="nmos pmos" \
#          bash scripts/v6_4_7_pivot_corridor.sh "0 1 2"
set -u
read -r -a GPUS <<<"${1:-0 1 2}"
NG="${#GPUS[@]}"
W="${W:?set W}"
read -r -a SEEDS <<<"${SEEDS:-7 31}"
read -r -a TECHS <<<"${TECHS:-tsmc7}"
read -r -a DEVS <<<"${DEVS:-nmos pmos}"
TAG="w$(echo "$W" | tr -d '.')"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s11_train_logs"
mkdir -p "$LOGDIR"; cd "$ROOT"
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"

jobs=()
for tech in "${TECHS[@]}"; do for dev in "${DEVS[@]}"; do for seed in "${SEEDS[@]}"; do
    jobs+=("$tech $dev $seed")
done; done; done
echo "[pivcor] W=$W TAG=$TAG  ${#jobs[@]} jobs across GPUs ${GPUS[*]}"

run_queue() {
    local gpu="$1"; shift
    for job in "$@"; do
        read -r tech dev seed <<<"$job"
        local name="v6_4_7_pivcor_${TAG}_s${seed}_${tech}_${dev}"
        echo "[gpu$gpu] START $name $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
            PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
            conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
                --model direct --size medium --device-type "$dev" --tech-scope "$tech" \
                --cuda --overwrite --data "$DATA/${tech}_v2cor_${dev}.npz" \
                --apply-filter off --swa-mode ema \
                --class-weights "traj_corridor=${W}" --seed "$seed" \
                --exp-name "v6_4_7_pivcor_${TAG}_s${seed}_${tech}" \
                >"$LOGDIR/${name}.log" 2>&1 \
            && echo "[gpu$gpu] DONE $name $(date +%H:%M:%S)" \
            || echo "[gpu$gpu] FAIL $name" >&2
    done
}

declare -a Q; for ((i=0;i<NG;i++)); do Q[$i]=""; done
for i in "${!jobs[@]}"; do Q[$((i%NG))]="${Q[$((i%NG))]}|${jobs[$i]}"; done
pids=()
for ((i=0;i<NG;i++)); do
    IFS='|' read -r -a qj <<<"${Q[$i]#|}"; [ "${#qj[@]}" -eq 0 ] && continue
    run_queue "${GPUS[$i]}" "${qj[@]}" & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "[pivcor] done"
