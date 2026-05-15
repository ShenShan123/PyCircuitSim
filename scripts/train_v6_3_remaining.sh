#!/usr/bin/env bash
# V6.3 retrain — TSMC12/16 medium DirectNet only.
# Reruns the 4 cells that disk-failed on 2026-05-14 after TSMC5/7 already
# finished (see docs/plans/2026-05-14-v6.3-spike-removal.md).
#
# Datasets are accessed via symlinks → /tmp/NN_SPICE/.../datasets/*.npz
# (the original /home/shenshan/NN_SPICE/ tree was moved to /tmp to free /home).
# Cache .npy ends up alongside the .npz under /tmp (sda2, 486G free).

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p training_logs/v6_3

for tech in tsmc12 tsmc16; do
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

echo "V6.3 TSMC12/16: 4 medium cells complete."
