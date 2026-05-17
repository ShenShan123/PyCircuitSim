#!/usr/bin/env bash
# V6.4.1 Phase 4 best-of-N retrain driver.
#
# Trains 8 techs/devices x 8 seeds = 64 DirectNet medium cells on the
# Phase-4-regenerated datasets (overlays + sinh + extra-NFIN). The
# --keep-offstate flag (plan §4e) is set so the new off-state overlay
# rows survive ingestion.
#
# Checkpoint naming: --exp-name v6_4_1_p4_<tech>_s<S> ->
#   external_compact_models/bsimar/checkpoints/v6_4_1_p4_<tech>_s<S>_<dev>_best.pt
#
# GPU mapping (CUDA_DEVICE_ORDER=PCI_BUS_ID): GPU1 = 97 GB free (4
# workers), GPU2 = 40 GB (2 workers), GPU0 = 40 GB (2 workers, may be
# shared). Persistent flock-guarded worklist; each cell retried once.
#
# Logs: logs/v6_4_1_phase4/<tech>_<dev>_s<S>.log + _driver.log

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

LOGDIR="${ROOT}/logs/v6_4_1_phase4"
mkdir -p "${LOGDIR}"
WORKLIST="${LOGDIR}/_worklist.txt"
LOCK="${LOGDIR}/_worklist.lock"
DRIVERLOG="${LOGDIR}/_driver.log"
CKPT="${ROOT}/external_compact_models/bsimar/checkpoints"

TECHS=(tsmc5 tsmc7 tsmc12 tsmc16)
DEVS=(nmos pmos)
SEEDS=(42 123 7 17 99 256 2024 31337)

# Build worklist (idempotent: skip cells whose ckpt exists).
: > "${WORKLIST}"
for t in "${TECHS[@]}"; do
  for d in "${DEVS[@]}"; do
    for s in "${SEEDS[@]}"; do
      out="${CKPT}/v6_4_1_p4_${t}_s${s}_${d}_best.pt"
      [[ -f "${out}" ]] && continue
      echo "${t} ${d} ${s}" >> "${WORKLIST}"
    done
  done
done
N_CELLS=$(wc -l < "${WORKLIST}")
echo "$(date '+%F %T') driver start: ${N_CELLS} cells queued" | tee -a "${DRIVERLOG}"

train_cell() {
  local gpu="$1" tech="$2" dev="$3" seed="$4" attempt="$5"
  local exp="v6_4_1_p4_${tech}_s${seed}"
  local log="${LOGDIR}/${tech}_${dev}_s${seed}.log"
  echo "$(date '+%F %T') [GPU${gpu}] START ${tech} ${dev} s${seed} (try ${attempt})" \
    | tee -a "${DRIVERLOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n pycircuitsim \
    python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type "${dev}" --tech-scope "${tech}" \
    --keep-offstate \
    --cuda --seed "${seed}" --exp-name "${exp}" --overwrite \
    > "${log}" 2>&1
  local rc=$?
  echo "$(date '+%F %T') [GPU${gpu}] DONE  ${tech} ${dev} s${seed} rc=${rc}" \
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
    read -r t d s <<< "${cell}"
    local out="${CKPT}/v6_4_1_p4_${t}_s${s}_${d}_best.pt"
    train_cell "${gpu}" "${t}" "${d}" "${s}" 1 || true
    if [[ ! -f "${out}" ]]; then
      echo "$(date '+%F %T') [GPU${gpu}] RETRY ${t} ${d} s${s}" \
        | tee -a "${DRIVERLOG}"
      sleep 20
      train_cell "${gpu}" "${t}" "${d}" "${s}" 2 || true
      if [[ ! -f "${out}" ]]; then
        echo "$(date '+%F %T') [GPU${gpu}] FAIL  ${t} ${d} s${s} (2 tries)" \
          | tee -a "${DRIVERLOG}"
      fi
    fi
  done
  echo "$(date '+%F %T') worker ${wid} (GPU${gpu}) exit" | tee -a "${DRIVERLOG}"
}

# 4 workers on GPU1, 2 on GPU2, 2 on GPU0.
worker 1 1 &
worker 1 2 &
worker 1 3 &
worker 1 4 &
worker 2 5 &
worker 2 6 &
worker 0 7 &
worker 0 8 &
wait
echo "$(date '+%F %T') driver: all workers finished" | tee -a "${DRIVERLOG}"
