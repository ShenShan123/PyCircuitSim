#!/usr/bin/env bash
# V6.4.8 S1 RE-RUN — reproduce the `--size large` (384x6) tsmc7 board to test
# whether the KILL ("capacity collapses the tsmc7 opamp") is robust to a fresh
# training draw. Same control-v2 recipe as the original S1 (apply-filter off,
# EMA 0.999, v2 data, 800 epochs), same 4 seeds, but a PARALLEL namespace
# (tsmc7_dn_lgB_s<seed>_<dev>) so the original S1 ckpts stay intact for A/B.
#
# Seed-by-seed (nmos then pmos) so a complete opamp board is available early.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd)/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

LOGDIR="training_logs/v6_4_8/rerun"
mkdir -p "$LOGDIR"

for seed in 42 17 7 31; do
  for dev in nmos pmos; do
    tag="tsmc7_dn_lgB_s${seed}_${dev}"
    log="${LOGDIR}/${tag}.log"
    echo "==== $(date '+%H:%M:%S') START ${tag} ===="
    conda run -n pycircuitsim python -u -m bsimar.cli.train \
      --model direct --size large \
      --device-type "${dev}" --tech-scope tsmc7 \
      --data "external_compact_models/bsimar/data/datasets/tsmc7_v2_${dev}.npz" \
      --apply-filter off --swa-mode ema --ema-decay 0.999 \
      --seed "${seed}" --exp-name "tsmc7_dn_lgB_s${seed}" \
      --cuda --overwrite \
      > "${log}" 2>&1
    echo "==== $(date '+%H:%M:%S') DONE  ${tag} ===="
  done
done
echo "ALL tsmc7 lgB re-runs complete."
