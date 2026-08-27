#!/usr/bin/env bash
# Benchmark Phase B — regenerate per-tech NN datasets for the S/M/L capacity study.
#
# Uses the canonical "regen-v2" data recipe (grid sampler + inverter-trip overlay
# + subthreshold/OFF densification) but writes the UNVERSIONED filename
# `{tech}_{dev}.npz`, exactly what `neural_network.cli.train` resolves by default
# (datasets/<tech-scope>_<device-type>.npz). Covers ALL Vth variants (--variants
# all, the default) and the full L/NFIN/T geometry grid per tech.
#
# Runs all 10 (tech x device) jobs concurrently.
#
# Usage: bash scripts/benchmark_gen_data.sh [workers_per_job]   (default 20)
#
# V7.5.17 pins the required 1.35 intra-bin L spacing in the canonical command.
# GEN_EXTRA remains available for explicit diagnostic-only additions.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/external_compact_models/bsim_cmg/scripts/generate_nn_data.py"
OUTDIR="$ROOT/external_compact_models/neural_network/data/datasets"
LOGDIR="${BENCHMARK_GEN_LOG_DIR:-$ROOT/results/benchmark_sml/gen_logs}"
WORKERS="${1:-20}"
OUTPUT_CONTRACT="${OUTPUT_CONTRACT:-reduced}"
case "$OUTPUT_CONTRACT" in
  reduced) DATA_TAG="" ;;
  full-terminal) DATA_TAG="_dnf" ;;
  *) echo "[gen] UNKNOWN OUTPUT_CONTRACT=$OUTPUT_CONTRACT" >&2; exit 2 ;;
esac
mkdir -p "$OUTDIR" "$LOGDIR"

techs=(tsmc5 tsmc6 tsmc7 tsmc12 tsmc16)
devs=(nmos pmos)

pids=(); jobs=()
for tech in "${techs[@]}"; do
  for dev in "${devs[@]}"; do
    log="$LOGDIR/gen_${tech}${DATA_TAG}_${dev}.log"
    # audit C6o — the label sidecar is fingerprinted against the geometry block
    # of the dataset it was built from, and the loader now REFUSES a sidecar
    # whose fingerprint does not match rather than silently re-using it. Rule 1
    # invites regenerating datasets, so retire the stale sidecar here; the next
    # training run rebuilds it. Without this a regen makes every later train
    # die inside load_and_split_bsimar until someone deletes the .npy by hand.
    rm -f "$OUTDIR/${tech}${DATA_TAG}_${dev}_tech_variant_labels.npy" \
          "$OUTDIR/${tech}${DATA_TAG}_${dev}_tech_variant_labels.meta.npz"
    echo "[gen] $tech $dev ($WORKERS workers) -> $log"
    conda run -n pycircuitsim python -u "$GEN" \
        --device "$dev" --tech "$tech" \
        --enable-inv-trip --enable-subvt-off \
        --output-contract "$OUTPUT_CONTRACT" \
        --max-l-ratio 1.35 \
        --n-workers "$WORKERS" ${GEN_EXTRA:-} \
        >"$log" 2>&1 &
    pids+=($!); jobs+=("${tech}${DATA_TAG}_${dev}")
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
