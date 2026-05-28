#!/usr/bin/env bash
# V6.4 best-of-N: supplementary GPU0 workers.
# GPU0 (A100-40GB) freed up mid-run; add 3 workers that pull from the
# SAME flock-guarded worklist as run_v6_4_bestof.sh. Safe to run
# alongside the main driver — the worklist consume protocol is atomic.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOGDIR="${ROOT}/logs/v6_4_bestof"
WORKLIST="${LOGDIR}/_worklist.txt"
LOCK="${LOGDIR}/_worklist.lock"
DRIVERLOG="${LOGDIR}/_driver.log"
CKPT="${ROOT}/external_compact_models/bsimar/checkpoints"

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
    train_cell "${gpu}" "${t}" "${d}" "${s}" 1 || true
    if [[ ! -f "${out}" ]]; then
      echo "$(date '+%F %T') [GPU${gpu}] RETRY ${t} ${d} s${s}" \
        | tee -a "${DRIVERLOG}"
      sleep 20
      train_cell "${gpu}" "${t}" "${d}" "${s}" 2 || true
      [[ -f "${out}" ]] || echo \
        "$(date '+%F %T') [GPU${gpu}] FAIL  ${t} ${d} s${s} (2 tries)" \
        | tee -a "${DRIVERLOG}"
    fi
  done
  echo "$(date '+%F %T') extra-worker ${wid} (GPU${gpu}) exit" \
    | tee -a "${DRIVERLOG}"
}

# 3 supplementary workers on GPU0.
worker 0 7 &
worker 0 8 &
worker 0 9 &
wait
echo "$(date '+%F %T') extra-gpu0: workers finished" | tee -a "${DRIVERLOG}"
