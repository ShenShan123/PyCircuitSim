#!/usr/bin/env bash
# V6.4.7 S9b — regenerate all 8 per-tech datasets as v2 (plan rev 3, ruling 5).
#
# v2 = current recipe (grid sampler + inv_trip overlay) PLUS the S9b generator
# fixes: tightened DC internal-solve tolerance + the subvt_off subthreshold/OFF
# densification class, so the 1e-12..1e-6 A id decades are populated
# (acceptance gate: >=1k rows per decade per cell, checked by
# scripts/v6_4_7_s9b_decade_gate.py after this script finishes).
#
# Outputs: external_compact_models/bsimar/data/datasets/tsmc{5,7,12,16}_v2_{nmos,pmos}.npz
# Existing v1 datasets are NOT touched.
#
# Usage: bash scripts/v6_4_7_s9b_regen_v2.sh [n_workers_per_tech]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/external_compact_models/PyCMG/scripts/generate_nn_data.py"
LOGDIR="$ROOT/results/v6_4_7/s9b_regen_logs"
WORKERS="${1:-32}"
mkdir -p "$LOGDIR"

pids=()
techs=(tsmc5 tsmc7 tsmc12 tsmc16)
for tech in "${techs[@]}"; do
    log="$LOGDIR/regen_${tech}.log"
    echo "[regen] $tech (both devices, $WORKERS workers) -> $log"
    conda run -n pycircuitsim python -u "$GEN" \
        --device both --tech "$tech" \
        --enable-inv-trip --enable-subvt-off \
        --version v2 --n-workers "$WORKERS" \
        >"$log" 2>&1 &
    pids+=($!)
done

rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "[regen] FAILED: ${techs[$i]} (see $LOGDIR/regen_${techs[$i]}.log)" >&2
        rc=1
    else
        echo "[regen] done: ${techs[$i]}"
    fi
done
exit $rc
