#!/usr/bin/env bash
# V6.4.2 Phase-7 scoped bake-off training driver.
#
# Trains the scoped laggard techs (TSMC5 + TSMC7) under 2 recipes:
#   stock  — plain V6.3.1-recipe medium cell (the seed-lottery control)
#   mono   — Phase 7a --monotonic (residual monotone-in-Vg on the id head)
# x 4 seeds x {nmos,pmos}  =  2 techs x 2 recipes x 4 seeds x 2 dev = 32 cells.
#
# 7b (--spectral-gds) is NOT in the grid: the CLI rejects it as not
# coherently implementable for a shared-trunk MLP (see the plan's 7b
# tension). The bake-off is therefore stock-retrain vs +7a.
#
# Checkpoint naming via --exp-name:
#   stock -> v6_4_2_p7_<tech>_stock_s<S>_<dev>_best.pt
#   mono  -> v6_4_2_p7_<tech>_mono_s<S>_<dev>_best.pt
#
# Persistent flock-guarded worklist; each cell retried once. Logs under
# logs/v6_4_2_phase7/.

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOGDIR="${ROOT}/logs/v6_4_2_phase7"
mkdir -p "${LOGDIR}"
WORKLIST="${LOGDIR}/_worklist.txt"
LOCK="${LOGDIR}/_worklist.lock"
DRIVERLOG="${LOGDIR}/_driver.log"
CKPT="${ROOT}/external_compact_models/bsimar/checkpoints"

TECHS=(tsmc5 tsmc7)
DEVS=(nmos pmos)
SEEDS=(42 123 7 17)
RECIPES=(stock mono)

: > "${WORKLIST}"
for t in "${TECHS[@]}"; do
  for r in "${RECIPES[@]}"; do
    for d in "${DEVS[@]}"; do
      for s in "${SEEDS[@]}"; do
        out="${CKPT}/v6_4_2_p7_${t}_${r}_s${s}_${d}_best.pt"
        [[ -f "${out}" ]] && continue
        echo "${t} ${r} ${d} ${s}" >> "${WORKLIST}"
      done
    done
  done
done
N_CELLS=$(wc -l < "${WORKLIST}")
echo "$(date '+%F %T') driver start: ${N_CELLS} cells queued" | tee -a "${DRIVERLOG}"

train_cell() {
  local gpu="$1" tech="$2" recipe="$3" dev="$4" seed="$5" attempt="$6"
  local exp="v6_4_2_p7_${tech}_${recipe}_s${seed}"
  local log="${LOGDIR}/${tech}_${recipe}_${dev}_s${seed}.log"
  local mono_flag=""
  [[ "${recipe}" == "mono" ]] && mono_flag="--monotonic"
  echo "$(date '+%F %T') [GPU${gpu}] START ${tech} ${recipe} ${dev} s${seed} (try ${attempt})" \
    | tee -a "${DRIVERLOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n pycircuitsim \
    python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type "${dev}" --tech-scope "${tech}" \
    ${mono_flag} \
    --cuda --seed "${seed}" --exp-name "${exp}" --overwrite \
    > "${log}" 2>&1
  local rc=$?
  echo "$(date '+%F %T') [GPU${gpu}] DONE  ${tech} ${recipe} ${dev} s${seed} rc=${rc}" \
    | tee -a "${DRIVERLOG}"
  return ${rc}
}

worker() {
  local gpu="$1" wid="$2"
  sleep "$((wid * 12))"
  while true; do
    local cell=""
    cell="$(
      flock "${LOCK}" bash -c '
        wl="'"${WORKLIST}"'"
        line="$(head -n1 "$wl")"
        [[ -z "$line" ]] && exit 0
        tail -n +2 "$wl" > "$wl.tmp" && mv "$wl.tmp" "$wl"
        echo "$line"
      '
    )"
    [[ -z "${cell}" ]] && break
    read -r t r d s <<< "${cell}"
    local out="${CKPT}/v6_4_2_p7_${t}_${r}_s${s}_${d}_best.pt"
    train_cell "${gpu}" "${t}" "${r}" "${d}" "${s}" 1 || true
    if [[ ! -f "${out}" ]]; then
      echo "$(date '+%F %T') [GPU${gpu}] RETRY ${t} ${r} ${d} s${s}" \
        | tee -a "${DRIVERLOG}"
      sleep 20
      train_cell "${gpu}" "${t}" "${r}" "${d}" "${s}" 2 || true
      [[ ! -f "${out}" ]] && \
        echo "$(date '+%F %T') [GPU${gpu}] FAIL ${t} ${r} ${d} s${s}" \
          | tee -a "${DRIVERLOG}"
    fi
  done
  echo "$(date '+%F %T') worker ${wid} (GPU${gpu}) exit" | tee -a "${DRIVERLOG}"
}

# 4 workers on GPU1 (idle), 2 on GPU3 (idle), 2 on GPU0.
# GPU2 avoided — busy (93% util) with an unrelated job at relaunch time.
worker 1 1 &
worker 1 2 &
worker 1 3 &
worker 1 4 &
worker 3 5 &
worker 3 6 &
worker 0 7 &
worker 0 8 &
wait
echo "$(date '+%F %T') driver: all workers finished" | tee -a "${DRIVERLOG}"
