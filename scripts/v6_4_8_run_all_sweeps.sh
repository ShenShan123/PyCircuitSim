#!/usr/bin/env bash
# Run all four DirectNet complex-circuit parametric sweeps in parallel for a
# given tech set, under the V6.4.8 gate methodology (CPU-pinned, repo NGSPICE).
# Each circuit logs to training_logs/v6_4_8_sweeps/<circuit>_<techtag>.log and
# records its 3-state exit code to .status.
#
# Usage: scripts/v6_4_8_run_all_sweeps.sh <TECHS> [dimension]
#   e.g. scripts/v6_4_8_run_all_sweeps.sh TSMC16 all
#        scripts/v6_4_8_run_all_sweeps.sh TSMC7  all
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TECHS="${1:?usage: $0 <TECHS> [dimension]}"
DIM="${2:-all}"
TAG="$(echo "$TECHS" | tr ',' '-')"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"
export PYTHONPATH="$ROOT:$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

LOGDIR="training_logs/v6_4_8_sweeps"
mkdir -p "$LOGDIR"

run_one () {
  local circuit="$1"
  local log="$LOGDIR/${circuit}_${TAG}.log"
  conda run -n pycircuitsim python "tests/verify_complex_${circuit}_sweep.py" \
    --tech "$TECHS" --dimension "$DIM" > "$log" 2>&1
  echo "$?" > "$LOGDIR/${circuit}_${TAG}.status"
}

for c in opamp ringosc switchcap sram; do
  run_one "$c" &
done
wait
echo "=== [$(date +%H:%M:%S)] all sweeps for ${TECHS} (${DIM}) done ==="
for c in opamp ringosc switchcap sram; do
  echo "  ${c}: exit=$(cat "$LOGDIR/${c}_${TAG}.status" 2>/dev/null)"
done
