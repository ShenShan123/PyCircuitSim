#!/usr/bin/env bash
# V6.4 control: V6.3.1-exact recipe (uniform LDS, no column pin,
# val-loss early-stop) on TSMC5/7 nmos+pmos. Confirms that reverting the
# Phase-1a slice-sum early-stop restores the V6.3.1 inverter VTC.
# 1e (gds asinh floor) stays in but is a verified no-op for TSMC datasets.
set -u
ROOT="/home/shenshan/NN_SPICE-refactor-nn"
LOG="$ROOT/logs/v6_4_control"
PY="/home/shenshan/.conda/envs/pycircuitsim/bin/python"
mkdir -p "$LOG"
cd "$ROOT" || exit 1
export PYTHONPATH=external_compact_models
export PYTHONUNBUFFERED=1

run() {  # gpu tech dev
  CUDA_VISIBLE_DEVICES="$1" \
    env V64_VAL_ES=1 V64_NO_COLW=1 V64_LDS=uniform \
    "$PY" -u -m bsimar.cli.train \
    --model direct --size medium --device-type "$3" --tech-scope "$2" \
    --cuda --exp-name "v6_4_ctl_$2" --overwrite --sobolev-weight 0.0 \
    > "$LOG/v6_4_ctl_$2_$3.log" 2>&1 &
}

run 0 tsmc5 nmos
run 0 tsmc5 pmos
run 1 tsmc7 nmos
run 1 tsmc7 pmos

wait
echo "ALL 4 CONTROL CELLS DONE"
