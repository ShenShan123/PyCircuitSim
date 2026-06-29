#!/usr/bin/env bash
# V6.5.8 T3 — install / revert the production tsmc7 opamp checkpoint + full matrix.
#
# The production resolver slot is the symlink pair
#   tsmc7_dn_medium_{nmos,pmos}_{best.pt,norm.npz}  ->  tsmc7_dn_large_*
# (medium-first preempt, parser.py:141-145). Installing the T3 candidate repoints
# those symlinks to it; reverting points them back to tsmc7_dn_large. After
# install the FULL matrix is run with NO env override (the multi-tech canary +
# other techs must resolve their OWN checkpoints).
#
# Usage:
#   scripts/v6_5_8_install_t3.sh install <stem>   # repoint medium -> stem
#   scripts/v6_5_8_install_t3.sh revert           # repoint medium -> tsmc7_dn_large
#   scripts/v6_5_8_install_t3.sh matrix           # full 16-gate + device + canary
set -u
cd "$(dirname "$0")/.."
CKDIR=external_compact_models/bsimar/checkpoints
export NGSPICE_BIN="${NGSPICE_BIN:-$PWD/tools/ngspice-45.2/bin/ngspice}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="conda run -n pycircuitsim python"
MODE="${1:?install|revert|matrix}"

repoint() {  # $1 = source stem (e.g. tsmc7_dn_t3a_e24 or tsmc7_dn_large)
  local src="$1" dev
  for dev in nmos pmos; do
    ln -sf "${src}_${dev}_best.pt"  "$CKDIR/tsmc7_dn_medium_${dev}_best.pt"
    ln -sf "${src}_${dev}_norm.npz" "$CKDIR/tsmc7_dn_medium_${dev}_norm.npz"
  done
  echo "repointed tsmc7_dn_medium_* -> ${src}_*"
  ls -l "$CKDIR"/tsmc7_dn_medium_*_best.pt
}

case "$MODE" in
  install) repoint "${2:?stem}" ;;
  revert)  repoint "tsmc7_dn_large" ;;
  matrix)
    echo "######## OPAMP (all techs) ########"
    $PY tests/verify_complex_opamp.py 2>&1 | tail -10
    echo "######## RING OSC (all techs) ########"
    $PY tests/verify_complex_ring_osc.py 2>&1 | tail -8
    echo "######## SWITCHCAP (all techs) ########"
    $PY tests/verify_complex_switchcap.py 2>&1 | tail -8
    echo "######## SRAM SNM (all techs) ########"
    $PY tests/verify_complex_sram_snm.py 2>&1 | tail -8
    echo "######## DEVICE DC/TRAN (TSMC techs) ########"
    $PY tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 2>&1 | tail -14
    echo "######## DEVICE AC (TSMC7) ########"
    $PY tests/verify_nn_ac.py --tech TSMC7 2>&1 | tail -8
    echo "######## LIFTED-SOURCE CANARY (multi-tech, no override) ########"
    $PY tests/verify_nn_lifted_source_dc.py 2>&1 | tail -10
    ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
