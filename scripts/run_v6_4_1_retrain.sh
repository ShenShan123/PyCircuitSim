#!/usr/bin/env bash
# V6.4.1 single-seed retrain: 8 medium cells (tsmc{5,7,12,16} x {nmos,pmos}),
# seed 42, across 3 GPUs. No --exp-name -> default save_prefix
# tsmc{X}_dn_medium_<dev> (the parser preempt-cascade canonical slot).
#   GPU1 (idle Blackwell) -> tsmc5, tsmc7   (4 cells)
#   GPU0 (A100)           -> tsmc12         (2 cells)
#   GPU2 (A100)           -> tsmc16         (2 cells)
set -u
ROOT="/home/shenshan/NN_SPICE-refactor-nn"
LOG="$ROOT/logs/v6_4_1_retrain"
PY="/home/shenshan/.conda/envs/pycircuitsim/bin/python"
mkdir -p "$LOG"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/external_compact_models"
export PYTHONUNBUFFERED=1

run() {  # gpu tech dev
  CUDA_VISIBLE_DEVICES="$1" "$PY" -u -m bsimar.cli.train \
    --model direct --size medium --device-type "$3" --tech-scope "$2" \
    --seed 42 --cuda --overwrite \
    > "$LOG/${2}_${3}.log" 2>&1 &
}

run 1 tsmc5  nmos
run 1 tsmc5  pmos
run 1 tsmc7  nmos
run 1 tsmc7  pmos
run 0 tsmc12 nmos
run 0 tsmc12 pmos
run 2 tsmc16 nmos
run 2 tsmc16 pmos

wait
echo "ALL 8 V6.4.1 RETRAIN CELLS DONE"
