#!/usr/bin/env bash
# V6.4 Phase 1 best-of-N retrain driver.
#
# Trains the 59 fresh stock-recipe DirectNet medium cells (4 techs x
# 2 devices x 8 seeds, minus 5 pre-existing v6_4_repro_* checkpoints).
# STOCK RECIPE ONLY -- the exact `scripts/train_v6_3_1_parallel.sh`
# invocation plus --seed and --exp-name. No source edits.
#
# Checkpoint naming: --exp-name v6_4_bof_<tech>_s<S>  ->
#   external_compact_models/bsimar/checkpoints/v6_4_bof_<tech>_s<S>_<dev>_best.pt
#
# GPU mapping: CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA_VISIBLE_DEVICES
# match `nvidia-smi` indices. GPU1 = RTX PRO 6000 (97 GB, free) gets 4
# workers; GPU2 = A100-40GB (lightly loaded) gets 2. GPU0 is busy with a
# 39 GB foreign job -> NOT used.
#
# Worker pool: persistent workers each pinned to a GPU pull cells from a
# shared flock-guarded worklist. Each cell is retried once on failure.
#
# Logs: logs/v6_4_bestof/<tech>_<dev>_s<S>.log + _driver.log

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOGDIR="${ROOT}/logs/v6_4_bestof"
mkdir -p "${LOGDIR}"
WORKLIST="${LOGDIR}/_worklist.txt"
LOCK="${LOGDIR}/_worklist.lock"
DRIVERLOG="${LOGDIR}/_driver.log"
CKPT="${ROOT}/external_compact_models/bsimar/checkpoints"

# --- Build the worklist (idempotent: skip cells whose ckpt exists) ----
: > "${WORKLIST}"
TECHS=(tsmc5 tsmc7 tsmc12 tsmc16)
DEVS=(nmos pmos)
SEEDS=(42 123 7 17 99 256 2024 31337)
declare -A HAVE=(
  ["tsmc5 nmos 42"]=1 ["tsmc5 pmos 42"]=1
  ["tsmc7 nmos 42"]=1 ["tsmc7 pmos 42"]=1
  ["tsmc5 nmos 123"]=1
)
for t in "${TECHS[@]}"; do
  for d in "${DEVS[@]}"; do
    for s in "${SEEDS[@]}"; do
      key="${t} ${d} ${s}"
      [[ -n "${HAVE[$key]:-}" ]] && continue
      out="${CKPT}/v6_4_bof_${t}_s${s}_${d}_best.pt"
      [[ -f "${out}" ]] && continue
      echo "${t} ${d} ${s}" >> "${WORKLIST}"
    done
  done
done
N_CELLS=$(wc -l < "${WORKLIST}")
echo "$(date '+%F %T') driver start: ${N_CELLS} cells queued" | tee -a "${DRIVERLOG}"

train_cell() {
  local gpu="$1" tech="$2" dev="$3" seed="$4" attempt="$5"
  local exp="v6_4_bof_${tech}_s${seed}"
  local log="${LOGDIR}/${tech}_${dev}_s${seed}.log"
  echo "$(date '+%F %T') [GPU${gpu}] START ${tech} ${dev} s${seed} (try ${attempt})" \
    | tee -a "${DRIVERLOG}"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n pycircuitsim \
    python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type "${dev}" --tech-scope "${tech}" \
    --cuda --seed "${seed}" --exp-name "${exp}" --overwrite \
    > "${log}" 2>&1
  local rc=$?
  echo "$(date '+%F %T') [GPU${gpu}] DONE  ${tech} ${dev} s${seed} rc=${rc}" \
    | tee -a "${DRIVERLOG}"
  return ${rc}
}

worker() {
  local gpu="$1" wid="$2"
  # Stagger cold start so CUDA-context allocs don't collide.
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
    local out="${CKPT}/v6_4_bof_${t}_s${s}_${d}_best.pt"
    if train_cell "${gpu}" "${t}" "${d}" "${s}" 1; then :; fi
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

# 4 workers on GPU1 (RTX PRO 6000, 97 GB), 2 on GPU2 (A100, lightly used).
worker 1 1 &
worker 1 2 &
worker 1 3 &
worker 1 4 &
worker 2 5 &
worker 2 6 &
wait
echo "$(date '+%F %T') driver: all workers finished" | tee -a "${DRIVERLOG}"
