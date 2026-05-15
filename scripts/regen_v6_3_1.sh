#!/usr/bin/env bash
# V6.3.1 regen — reduce inv_trip overlay from 9.83% → ~3.3% by dropping
# the ±0.25·VDD Vbs sweep in _inv_trip_points. See docs/plans/2026-05-14-v6.3-spike-removal.md "Phase C".

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

for tech in tsmc5 tsmc7 tsmc12 tsmc16; do
  echo "==== regen ${tech} ===="
  conda run --no-capture-output -n pycircuitsim python \
    external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech "${tech}" --enable-inv-trip --n-workers 8
done

echo "V6.3.1 regen: all 8 datasets done."
