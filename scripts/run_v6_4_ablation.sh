#!/usr/bin/env bash
# V6.4 Phase 1 ablation: isolate which sub-step regressed the inverter VTC.
# 6 cells, TSMC5+TSMC7 nmos+pmos split across 3 recipes. GPU 0 and 1 only.
#
# Recipes (all sobolev=0):
#   c1e  — 1e gds-floor only; 1c column_weights OFF, 1d uniform LDS  (≈V6.3.1+1e)
#   c1cd — 1c+1e (column pin ON, uniform LDS)
#   c1de — 1d+1e (quantile LDS, no column pin)
# Each gets its own --exp-name; nothing clobbers V6.3.1 canonical slots.
set -u
ROOT="/home/shenshan/NN_SPICE-refactor-nn"
LOG="$ROOT/logs/v6_4_ablation"
PY="/home/shenshan/.conda/envs/pycircuitsim/bin/python"
mkdir -p "$LOG"
cd "$ROOT" || exit 1
export PYTHONPATH=external_compact_models
export PYTHONUNBUFFERED=1

# run <gpu> <exp> <tech> <dev> <recipe-env>
run() {
  CUDA_VISIBLE_DEVICES="$1" env $5 "$PY" -u -m bsimar.cli.train \
    --model direct --size medium --device-type "$4" --tech-scope "$3" \
    --cuda --exp-name "$2" --overwrite --sobolev-weight 0.0 \
    > "$LOG/$2_$4.log" 2>&1 &
}

# GPU 0: recipe c1e (1e only) — TSMC5+TSMC7
run 0 v6_4_c1e_tsmc5 tsmc5 nmos "V64_NO_COLW=1 V64_LDS=uniform"
run 0 v6_4_c1e_tsmc5 tsmc5 pmos "V64_NO_COLW=1 V64_LDS=uniform"
run 0 v6_4_c1e_tsmc7 tsmc7 nmos "V64_NO_COLW=1 V64_LDS=uniform"
run 0 v6_4_c1e_tsmc7 tsmc7 pmos "V64_NO_COLW=1 V64_LDS=uniform"
# GPU 1: recipe c1de (1d+1e, quantile LDS, no column pin) — TSMC5+TSMC7
run 1 v6_4_c1de_tsmc5 tsmc5 nmos "V64_NO_COLW=1 V64_LDS=quantile"
run 1 v6_4_c1de_tsmc5 tsmc5 pmos "V64_NO_COLW=1 V64_LDS=quantile"
run 1 v6_4_c1de_tsmc7 tsmc7 nmos "V64_NO_COLW=1 V64_LDS=quantile"
run 1 v6_4_c1de_tsmc7 tsmc7 pmos "V64_NO_COLW=1 V64_LDS=quantile"

wait
echo "ALL 8 ABLATION CELLS DONE"
