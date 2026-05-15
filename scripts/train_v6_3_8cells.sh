#!/usr/bin/env bash
# V6.3 retrain: medium-only DirectNet across all 4 TSMC techs.
#   {medium} × {nmos, pmos} × {tsmc5, tsmc7, tsmc12, tsmc16} = 8 cells
#
# Datasets expected at:
#   external_compact_models/bsimar/data/datasets/{tsmc5,tsmc7,tsmc12,tsmc16}_{nmos,pmos}.npz
# (regenerated with V6.3 patches: inv_trip recentered on VDD/2,
#  reverse_vds corridor added — see docs/plans/2026-05-14-v6.3-spike-removal.md)
#
# Logs land under training_logs/v6_3/.

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p training_logs/v6_3

for tech in tsmc5 tsmc7 tsmc12 tsmc16; do
  for dev in nmos pmos; do
    tag="${tech}_dn_medium_${dev}"
    log="training_logs/v6_3/${tag}.log"
    echo "==== ${tag} ===="
    conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
      --model direct --size medium \
      --device-type "${dev}" --tech-scope "${tech}" \
      --cuda --overwrite \
      2>&1 | tee "${log}"
  done
done

echo "V6.3: All 8 medium cells complete."
