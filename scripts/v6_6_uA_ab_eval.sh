#!/usr/bin/env bash
# V6.5.1 µA-band loss-lever A/B evaluation.
# Runs switchcap (target metric: charge_err) + DC device sweep (no-regress
# check: Id-Vgs NRMSE) against a given checkpoint pair, CPU-pinned per the gate
# methodology. Usage: v6_6_uA_ab_eval.sh <TECHUC> <ckpt_stem_prefix>
#   e.g. v6_6_uA_ab_eval.sh TSMC5 tsmc5_dn_medium_uA
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
tuc="$1"; stem="$2"
out="$ROOT/results/v6_6_uA_ab/eval/${stem}"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="${stem}_nmos"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="${stem}_pmos"
echo "### $stem ($tuc) ###"
echo "--- switchcap (target: charge_err) ---"
conda run -n pycircuitsim python -u "$ROOT/tests/verify_complex_switchcap.py" --tech "$tuc" \
  > "$out/switchcap.log" 2>&1
grep -E "charge err|charge level|PASS|FAIL|RESULT" "$out/switchcap.log" | tail -6
echo "--- DC device Id-Vgs (no-regress: NRMSE) ---"
conda run -n pycircuitsim python -u "$ROOT/tests/verify_nn_multi_tech_dc.py" --tech "$tuc" \
  > "$out/dc.log" 2>&1
grep -E "RESULT|nmos|pmos" "$out/dc.log" | grep -iE "base|RESULT|PASS|FAIL" | tail -8
echo ""
