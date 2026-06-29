#!/usr/bin/env bash
# V6.5.8 T3 gate-shop + preservation suite for a tsmc7 opamp candidate.
#
# The candidate swaps ONLY tsmc7's NMOS/PMOS checkpoints, so single-tech tsmc7
# gates can be run via the global env override (every device is tsmc7). The
# multi-tech canary / full matrix must be run AFTER install (symlink repoint),
# NOT via the env override (which would force tsmc7 onto every tech).
#
# Usage:
#   scripts/v6_5_8_gate_t3.sh opamp  <stem>            # opamp gain only (gate-shop)
#   scripts/v6_5_8_gate_t3.sh preserve <stem>          # ring+switchcap+sram+device DC/tran/AC
#   scripts/v6_5_8_gate_t3.sh all <stem>               # opamp + preserve
# where <stem> is e.g. tsmc7_dn_t3a_e24 (resolves *_nmos / *_pmos).
set -u
cd "$(dirname "$0")/.."
export NGSPICE_BIN="${NGSPICE_BIN:-$PWD/tools/ngspice-45.2/bin/ngspice}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${1:?mode: opamp|preserve|all}"
STEM="${2:?candidate stem, e.g. tsmc7_dn_t3a_e24}"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="${STEM}_nmos"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="${STEM}_pmos"
PY="conda run -n pycircuitsim python"

run_opamp() {
  echo "### OPAMP  $STEM"
  $PY tests/verify_complex_opamp.py --tech TSMC7 2>&1 | grep -E "TSMC7 *\||gain gate"
}
run_preserve() {
  echo "### RING  $STEM";       $PY tests/verify_complex_ring_osc.py  --tech TSMC7 2>&1 | grep -iE "TSMC7 *\||PASS|FAIL|pass-rate|/" | tail -4
  echo "### SWITCHCAP  $STEM";  $PY tests/verify_complex_switchcap.py --tech TSMC7 2>&1 | grep -iE "TSMC7 *\||PASS|FAIL|/" | tail -4
  echo "### SRAM  $STEM";       $PY tests/verify_complex_sram_snm.py  --tech TSMC7 2>&1 | grep -iE "TSMC7 *\||PASS|FAIL|/" | tail -4
  echo "### DEVICE DC/TRAN  $STEM"; $PY tests/verify_nn_dc_tran.py --tech TSMC7 2>&1 | grep -iE "inverter|DC |tran |PASS|FAIL|/" | tail -6
  echo "### DEVICE AC  $STEM";  $PY tests/verify_nn_ac.py --tech TSMC7 2>&1 | grep -iE "PASS|FAIL|gain0|f3db|/" | tail -6
}
case "$MODE" in
  opamp)    run_opamp ;;
  preserve) run_preserve ;;
  all)      run_opamp; run_preserve ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
