#!/usr/bin/env bash
# Retrain the two TSMC7 medium DirectNet cells on the inv-trip-augmented
# tsmc7_{nmos,pmos}.npz datasets (V6 + inv_trip overlay re-enabled for tsmc7).
# Overwrites the existing tsmc7_dn_medium_{nmos,pmos}_best.pt checkpoints.

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p training_logs/per_tech_v2

for dev in nmos pmos; do
  tag="tsmc7_dn_medium_${dev}_invtrip"
  log="training_logs/per_tech_v2/${tag}.log"
  echo "==== ${tag} ===="
  conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type "${dev}" --tech-scope tsmc7 \
    --cuda --overwrite \
    2>&1 | tee "${log}"
done

echo "TSMC7 medium retrain complete."
