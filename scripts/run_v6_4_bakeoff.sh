#!/usr/bin/env bash
# V6.4 Phase 1 subset bake-off: TSMC5+TSMC7 × nmos/pmos, variant A (no
# Sobolev) and variant B (Sobolev). 8 jobs, 4 per GPU (0 and 1; never 2).
set -u
ROOT="/home/shenshan/NN_SPICE-refactor-nn"
LOG="$ROOT/logs/v6_4_bakeoff"
PY="/home/shenshan/.conda/envs/pycircuitsim/bin/python"
cd "$ROOT" || exit 1
export PYTHONPATH=external_compact_models
export PYTHONUNBUFFERED=1

run() {  # gpu exp sob tech dev
  CUDA_VISIBLE_DEVICES="$1" "$PY" -u -m bsimar.cli.train \
    --model direct --size medium --device-type "$5" --tech-scope "$4" \
    --cuda --exp-name "$2" --overwrite --sobolev-weight "$3" \
    > "$LOG/$2_$5.log" 2>&1 &
}

# Variant A — no Sobolev (1c/1d/1e only).  GPU 0.
run 0 v6_4_a_tsmc5 0.0  tsmc5 nmos
run 0 v6_4_a_tsmc5 0.0  tsmc5 pmos
run 0 v6_4_a_tsmc7 0.0  tsmc7 nmos
run 0 v6_4_a_tsmc7 0.0  tsmc7 pmos

# Variant B — Sobolev 0.05.  GPU 1.
run 1 v6_4_b_tsmc5 0.05 tsmc5 nmos
run 1 v6_4_b_tsmc5 0.05 tsmc5 pmos
run 1 v6_4_b_tsmc7 0.05 tsmc7 nmos
run 1 v6_4_b_tsmc7 0.05 tsmc7 pmos

wait
echo "ALL 8 BAKE-OFF CELLS DONE"
