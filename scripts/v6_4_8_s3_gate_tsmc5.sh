#!/usr/bin/env bash
# V6.4.8 S3 — install a DirectNet checkpoint prefix into the tsmc5_dn_medium
# resolver slot and run the tsmc5 complex-circuit board + protected gates,
# CPU-pinned per the V6.4.8 gate-CPU methodology.
#
# Usage: bash scripts/v6_4_8_s3_gate_tsmc5.sh <exp_prefix>
#   e.g. bash scripts/v6_4_8_s3_gate_tsmc5.sh v6_4_8_s3ekv_s17_tsmc5
#        bash scripts/v6_4_8_s3_gate_tsmc5.sh v6_4_7_ctlv2_s17_tsmc5   (baseline)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
PREFIX="${1:?need checkpoint exp-prefix, e.g. v6_4_8_s3ekv_s17_tsmc5}"
cd "$ROOT"

export NGSPICE_BIN="$ROOT/tools/ngspice-45.2/bin/ngspice"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Install into the resolver slot.
for dev in nmos pmos; do
    for kind in best.pt norm.npz; do
        ln -sf "${PREFIX}_${dev}_${kind}" "$CKPT/tsmc5_dn_medium_${dev}_${kind}"
    done
done
echo "### Gating tsmc5 with slot -> $PREFIX"

run() { echo; echo "## $1"; shift; conda run -n pycircuitsim python "$@" 2>&1 \
    | grep -iE "TSMC5 .*\||gain error|ChgErr|passed|PASS|FAIL|MaxErr|NRMSE|trip shift|^Overall|/[0-9]+ (passed|PASS)" | head -25; }

run "switchcap (PRIMARY)"  tests/verify_complex_switchcap.py --tech TSMC5
run "opamp"                tests/verify_complex_opamp.py --tech TSMC5
run "ring_osc"             tests/verify_complex_ring_osc.py --tech TSMC5
run "sram_snm"             tests/verify_complex_sram_snm.py --tech TSMC5
run "inverter VTC/tran"    tests/verify_nn_dc_tran.py --tech TSMC5 --inverter-only
run "DC-55"                tests/verify_nn_multi_tech_dc.py --tech TSMC5
echo; echo "### done $PREFIX"
