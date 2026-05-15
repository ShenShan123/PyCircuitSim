#!/usr/bin/env bash
# V6.3.1 retrain — 8 medium DirectNet cells across 4 TSMC techs.
# Parallelizes across GPU 0 (A100), GPU 1 (RTX PRO 6000), GPU 2 (A100, shared).
# Pairs cells so the heaviest dataset gets the largest GPU.
#
# Run-time estimate: ~30-40 min per cell sequential. With 3-way parallel:
# ~3 batches × ~40 min ≈ 2 hours total (vs ~4-5 hours sequential).

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p training_logs/v6_3_1

train_cell() {
  local gpu="$1" tech="$2" dev="$3"
  local tag="${tech}_dn_medium_${dev}"
  local log="training_logs/v6_3_1/${tag}.log"
  echo "==== START ${tag} on GPU${gpu} ===="
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type "${dev}" --tech-scope "${tech}" \
    --cuda --overwrite \
    > "${log}" 2>&1
  echo "==== DONE  ${tag} on GPU${gpu} ===="
}
export -f train_cell

# Batch 1: TSMC5 NMOS+PMOS + TSMC7 NMOS (3-way parallel)
train_cell 0 tsmc5  nmos &
train_cell 1 tsmc5  pmos &
train_cell 2 tsmc7  nmos &
wait

# Batch 2: TSMC7 PMOS + TSMC12 NMOS+PMOS (3-way parallel)
train_cell 0 tsmc7  pmos &
train_cell 1 tsmc12 nmos &
train_cell 2 tsmc12 pmos &
wait

# Batch 3: TSMC16 NMOS+PMOS (2-way parallel, fastest two GPUs)
train_cell 0 tsmc16 nmos &
train_cell 1 tsmc16 pmos &
wait

echo "V6.3.1: all 8 medium cells complete."
