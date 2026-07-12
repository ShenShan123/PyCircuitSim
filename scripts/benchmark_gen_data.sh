#!/usr/bin/env bash
# Benchmark Phase B — regenerate per-tech NN datasets for the S/M/L capacity study.
#
# Uses the canonical "regen-v2" data recipe (grid sampler + inverter-trip overlay
# + subthreshold/OFF densification) but writes the UNVERSIONED filename
# `{tech}_{dev}.npz`, which is exactly what `bsimar.cli.train` resolves by default
# (datasets/<tech-scope>_<device-type>.npz). Covers ALL Vth variants (--variants
# all, the default) and the full L/NFIN/T geometry grid per tech.
#
# Runs all 8 (tech x device) jobs concurrently — the box has 192 cores.
#
# Usage: bash scripts/benchmark_gen_data.sh [workers_per_job]   (default 20)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/external_compact_models/PyCMG/scripts/generate_nn_data.py"
OUTDIR="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/benchmark_sml/gen_logs"
WORKERS="${1:-20}"
mkdir -p "$OUTDIR" "$LOGDIR"

techs=(tsmc5 tsmc6 tsmc7 tsmc12 tsmc16)
devs=(nmos pmos)

pids=(); jobs=()
for tech in "${techs[@]}"; do
  for dev in "${devs[@]}"; do
    log="$LOGDIR/gen_${tech}_${dev}.log"
    echo "[gen] $tech $dev ($WORKERS workers) -> $log"
    conda run -n pycircuitsim python -u "$GEN" \
        --device "$dev" --tech "$tech" \
        --enable-inv-trip --enable-subvt-off \
        --n-workers "$WORKERS" \
        >"$log" 2>&1 &
    pids+=($!); jobs+=("${tech}_${dev}")
  done
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[gen] DONE: ${jobs[$i]}"
  else
    echo "[gen] FAILED: ${jobs[$i]} (see $LOGDIR/gen_${jobs[$i]}.log)" >&2
    rc=1
  fi
done
if [ "$rc" -eq 0 ]; then echo "[gen] ALL DATASETS COMPLETE"; else echo "[gen] SOME JOBS FAILED"; fi
exit $rc
