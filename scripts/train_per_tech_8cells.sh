#!/usr/bin/env bash
# Train the 8-cell per-tech sweep (V6 dedicated TSMC5/7):
#   {small, medium} × {nmos, pmos} × {tsmc5, tsmc7}
#
# Datasets must already exist at:
#   external_compact_models/bsimar/data/datasets/{tsmc5,tsmc7}_{nmos,pmos}.npz
# (see external_compact_models/PyCMG/scripts/generate_nn_data.py --tech ...)
#
# Logs land under training_logs/per_tech/.
# Run sequentially to avoid GPU contention; each cell early-stops well
# before its --patience budget.

set -euo pipefail
cd "$(dirname "$0")/.."

# Pin to a free GPU (override by exporting CUDA_VISIBLE_DEVICES before run).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# Ensure the `bsimar` package is importable (it lives under
# external_compact_models/, not on a normal sys.path).
export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p training_logs/per_tech

for tech in tsmc5 tsmc7; do
  for size in small medium; do
    for dev in nmos pmos; do
      tag="${tech}_dn_${size}_${dev}"
      log="training_logs/per_tech/${tag}.log"
      echo "==== ${tag} ===="
      conda run -n pycircuitsim python -u -m bsimar.cli.train \
        --model direct --size "${size}" \
        --device-type "${dev}" --tech-scope "${tech}" \
        --cuda --overwrite \
        2>&1 | tee "${log}"
    done
  done
done

echo "All 8 cells complete."
