#!/usr/bin/env bash
# V7.2.0 §8.4 T3 — 16-gate complex campaign for the perturbing flag
# bundle, CPU axis: {PYCIRCUITSIM_TRAN_BATCH_COMMIT=1,
# PYCIRCUITSIM_BATCHED_STAMP=1, PYCIRCUITSIM_MNA_ORDERING=<spec>}.
#
# Runs the production DirectNet config (resolver-default stems, i.e.
# a3_omp_one.sh recipe=clean size=large MODEL=direct) over
# 4 suites x 4 techs x OMP∈{1,2,4}, flag bundle exported so every gate
# subprocess inherits it. Binding per plan §8.4: sram_snm + switchcap
# (deterministic, ≤0.3 pp) and ZERO PASS/FAIL flips across OMP;
# ring_osc / opamp are REPORTED ONLY (their run-to-run noise exceeds the
# gate). Evidence lands in results/v720_gpu_regate/t3_cpu_bundle/.
#
# Usage: bash scripts/v720_t3_flag_bundle.sh [ordering]
#   ordering: PYCIRCUITSIM_MNA_ORDERING value (default NATURAL — the
#   real-matrix winner; the plan's original MMD_AT_PLUS_A pick was
#   refuted on real assembled matrices, see bench_ordering_real.py)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORDERING="${1:-NATURAL}"

OUT="$ROOT/results/v720_gpu_regate/t3_cpu_bundle"
mkdir -p "$OUT/logs"

export PYCIRCUITSIM_TRAN_BATCH_COMMIT=1
export PYCIRCUITSIM_BATCHED_STAMP=1
export PYCIRCUITSIM_MNA_ORDERING="$ORDERING"
export GATE_SCRATCH="$OUT/scratch"
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0

echo "T3 CPU flag bundle: TRAN_BATCH_COMMIT=1 BATCHED_STAMP=1 MNA_ORDERING=$ORDERING"
echo "start: $(date '+%F %T')"

# Binding (deterministic) suites first, report-only after.
SUITES="verify_complex_sram_snm verify_complex_switchcap verify_complex_ring_osc verify_complex_opamp"
TECHS="TSMC5 TSMC7 TSMC12 TSMC16"

jobs_file="$OUT/jobs.txt"
: > "$jobs_file"
for suite in $SUITES; do
  for tech in $TECHS; do
    for omp in 1 2 4; do
      echo "$suite $tech $omp" >> "$jobs_file"
    done
  done
done

run_one() {
  suite="$1"; tech="$2"; omp="$3"
  log="$OUT/logs/${suite}_${tech}.log"
  bash "$ROOT/scripts/a3_omp_one.sh" clean large "$tech" "$suite" "$omp" "$log"
}
export -f run_one
export OUT ROOT

xargs -P 3 -L1 bash -c 'run_one "$@"' _ < "$jobs_file"

# Rule-2 canary with the bundle on (single run, OMP=1).
export CUDA_VISIBLE_DEVICES="" NGSPICE_BIN="$ROOT/tools/ngspice-45.2/bin/ngspice"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYCIRCUITSIM_TORCH_THREADS=1 \
  /data1/shenshan/.conda/envs/pycircuitsim/bin/python -u \
  "$ROOT/tests/verify_nn_lifted_source_dc.py" \
  > "$OUT/logs/rule2_canary.log" 2>&1
echo "rule2 canary rc=$?"

echo "done: $(date '+%F %T')"
echo "== summary (PASS/FAIL per cell per OMP) =="
for suite in $SUITES; do
  for tech in $TECHS; do
    for omp in 1 2 4; do
      f="$OUT/logs/${suite}_${tech}.log.omp${omp}"
      [ -f "$f" ] && echo "$(basename "$f"): $(cat "$f")"
    done
  done
done
