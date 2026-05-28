#!/usr/bin/env bash
# V6.4 Phase 1 full retrain: TSMC5/7/12/16 × nmos/pmos (8 medium cells).
# Usage: run_v6_4_full.sh <sobolev_weight>
#   sobolev_weight 0.0  → variant A recipe (1c/1d/1e only)
#   sobolev_weight >0   → variant B recipe (+ Sobolev)
# All checkpoints saved under exp-name v6_4_<tech> so the V6.3.1 canonical
# slots are never clobbered. 4 cells per GPU (0 and 1; never 2).
set -u
SOB="${1:?usage: run_v6_4_full.sh <sobolev_weight>}"
ROOT="/home/shenshan/NN_SPICE-refactor-nn"
LOG="$ROOT/logs/v6_4_full"
PY="/home/shenshan/.conda/envs/pycircuitsim/bin/python"
mkdir -p "$LOG"
cd "$ROOT" || exit 1
export PYTHONPATH=external_compact_models
export PYTHONUNBUFFERED=1

run() {  # gpu tech dev
  CUDA_VISIBLE_DEVICES="$1" "$PY" -u -m bsimar.cli.train \
    --model direct --size medium --device-type "$3" --tech-scope "$2" \
    --cuda --exp-name "v6_4_$2" --overwrite --sobolev-weight "$SOB" \
    > "$LOG/v6_4_$2_$3.log" 2>&1 &
}

# GPU 0 — TSMC5, TSMC7
run 0 tsmc5  nmos
run 0 tsmc5  pmos
run 0 tsmc7  nmos
run 0 tsmc7  pmos
# GPU 1 — TSMC12, TSMC16
run 1 tsmc12 nmos
run 1 tsmc12 pmos
run 1 tsmc16 nmos
run 1 tsmc16 pmos

wait
echo "ALL 8 V6.4 FULL CELLS DONE (sobolev=$SOB)"
