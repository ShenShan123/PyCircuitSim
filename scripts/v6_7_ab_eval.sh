#!/usr/bin/env bash
# V6.7 A/B eval: run the switchcap + device AC + opamp AC + DC-sanity gates for a
# given DirectNet checkpoint tag, CPU-pinned against the repo NGSPICE.
#
# Usage:
#   scripts/v6_7_ab_eval.sh <tag> <TECH>      # e.g. tsmc5_dn_csob TSMC5
# The tag must be the checkpoint stem WITHOUT the _{nmos,pmos}_best.pt suffix and
# WITHOUT polarity (e.g. tsmc5_dn_csob). Resolves <tag>_nmos / <tag>_pmos and
# pins them via the parser env override (the `tsmc{X}_dn_` prefix routes the
# local-vocab remap). Baseline tag = tsmc{X}_dn_medium.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TAG="${1:?need checkpoint tag, e.g. tsmc5_dn_csob}"
TECH="${2:?need TECH, e.g. TSMC5}"
NG="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="${TAG}_nmos"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="${TAG}_pmos"

echo "############ A/B EVAL  tag=$TAG  tech=$TECH ############"
echo "==== switchcap ===="
conda run --no-capture-output -n pycircuitsim python tests/verify_complex_switchcap.py --tech "$TECH" 2>&1 \
  | grep -viE "warning|mexp|osdi\[" | grep -iE "charge|droop|PASS|FAIL|level|SUMMARY|Tech "
echo "==== device CS-amp AC ===="
conda run --no-capture-output -n pycircuitsim python tests/verify_nn_ac.py --tech "$TECH" 2>&1 \
  | grep -viE "warning|mexp|osdi\[" | grep -iE "gain0|f3db|nrmse|PASS|FAIL|Tech|dev "
echo "==== opamp open-loop AC ===="
conda run --no-capture-output -n pycircuitsim python tests/verify_complex_opamp_ac.py --tech "$TECH" 2>&1 \
  | grep -viE "warning|mexp|osdi\[" | grep -iE "gain|gbw|pm |PASS|FAIL|Tech"
echo "==== ring-osc + opamp + sram (complex DC/tran) ===="
for t in ring_osc opamp sram_snm; do
  conda run --no-capture-output -n pycircuitsim python tests/verify_complex_${t}.py --tech "$TECH" 2>&1 \
    | grep -viE "warning|mexp|osdi\[" | grep -iE "period|gain_err|snm|PASS|FAIL|trip" | head -6
done
echo "############ END $TAG ############"
