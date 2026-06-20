#!/usr/bin/env bash
# Re-baseline TSMC7 after the broad retrain (plan Phase A step 3): the four
# untouched single-point verify_complex_*.py ship gates + the lifted-source
# canary + the inverter gate, all CPU-pinned. The opamp single-point must still
# PASS or be documented as regressed (the broad-coverage trade-off).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"
export PYTHONPATH="$ROOT:$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

LOGDIR="training_logs/v6_4_8_rebaseline"
mkdir -p "$LOGDIR"

echo "############ TSMC7 single-point ship gates (untouched verify_complex_*.py) ############"
for c in opamp ring_osc switchcap sram_snm; do
  echo "==== verify_complex_${c}.py --tech TSMC7 ===="
  conda run -n pycircuitsim python "tests/verify_complex_${c}.py" --tech TSMC7 \
    2>&1 | tee "$LOGDIR/singlepoint_${c}_tsmc7.log" | grep -E "TSMC7|PASS|FAIL|gain|period|charge|SNM|force_ic|within" || true
done

echo "############ lifted-source canary (TSMC7 rows) ############"
conda run -n pycircuitsim python tests/verify_nn_lifted_source_dc.py --label v648broad \
  2>&1 | tee "$LOGDIR/lifted_source.log" | grep -E "TSMC7|PASS|FAIL" || true

echo "############ done ############"
