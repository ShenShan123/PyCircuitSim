#!/usr/bin/env bash
# V7.2.0 §8.4 T3 — 16-gate complex campaign for the perturbing flag
# bundle, CPU or GPU axis: {PYCIRCUITSIM_TRAN_BATCH_COMMIT=1,
# PYCIRCUITSIM_BATCHED_STAMP=1, PYCIRCUITSIM_MNA_ORDERING=<spec>}.
#
# Runs the production DirectNet config (resolver-default stems, i.e.
# a3_omp_one.sh recipe=clean size=large MODEL=direct) over
# 4 suites x 4 techs x OMP∈{1,2,4}, flag bundle exported so every gate
# subprocess inherits it. Binding per plan §8.4: sram_snm + switchcap
# (deterministic, ≤0.3 pp) and ZERO PASS/FAIL flips across OMP;
# ring_osc / opamp are REPORTED ONLY (their run-to-run noise exceeds the
# gate). Evidence lands in results/v720_gpu_regate/t3_${T3_AXIS}_bundle/.
#
# Usage:
#   bash scripts/v720_t3_flag_bundle.sh [ordering]
#   T3_AXIS=gpu GPU=1 bash scripts/v720_t3_flag_bundle.sh [ordering]
#   ordering: PYCIRCUITSIM_MNA_ORDERING value (default NATURAL — the
#   real-matrix winner; the plan's original MMD_AT_PLUS_A pick was
#   refuted on real assembled matrices, see bench_ordering_real.py)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORDERING="${1:-NATURAL}"
AXIS="${T3_AXIS:-cpu}"
GPU="${GPU:-0}"
PY="${NN_PY:-/data2/home/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
export NN_PY="$PY"

case "$AXIS" in
  cpu)
    export PYCIRCUITSIM_NN_DEVICE=cpu
    export GATE_CUDA_VISIBLE_DEVICES=""
    ;;
  gpu)
    export PYCIRCUITSIM_NN_DEVICE=cuda
    export GATE_CUDA_VISIBLE_DEVICES="$GPU"
    ;;
  *) echo "unknown T3_AXIS=$AXIS (expected cpu or gpu)" >&2; exit 2 ;;
esac

OUT="$ROOT/results/v720_gpu_regate/t3_${AXIS}_bundle"
mkdir -p "$OUT/logs"

export PYCIRCUITSIM_TRAN_BATCH_COMMIT=1
export PYCIRCUITSIM_BATCHED_STAMP=1
export PYCIRCUITSIM_MNA_ORDERING="$ORDERING"
export GATE_SCRATCH="$OUT/scratch"
export OMP_WAIT_POLICY=passive KMP_BLOCKTIME=0

echo "T3 $AXIS flag bundle: TRAN_BATCH_COMMIT=1 BATCHED_STAMP=1 MNA_ORDERING=$ORDERING GPU=$GPU"
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD)"
echo "python: $PY"
echo "start: $(date '+%F %T')"

# Binding (deterministic) suites first, report-only after.
SUITES="verify_complex_sram_snm verify_complex_switchcap verify_complex_ring_osc verify_complex_opamp"
TECHS="TSMC5 TSMC7 TSMC12 TSMC16"

jobs_file="$OUT/jobs.txt"
: > "$jobs_file"
for suite in $SUITES; do
  for tech in $TECHS; do
    for omp in 1 2 4; do
      base="$OUT/logs/${suite}_${tech}.log.omp${omp}"
      rm -f "$base" "${base}.full" "${base}.rc"
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
pool_rc=$?

# Rule-2 canary with the bundle on (single run, OMP=1).
export CUDA_VISIBLE_DEVICES="$GATE_CUDA_VISIBLE_DEVICES" NGSPICE_BIN="$ROOT/tools/ngspice-45.2/bin/ngspice"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYCIRCUITSIM_TORCH_THREADS=1 \
  "$PY" -u \
  "$ROOT/tests/verify_nn_lifted_source_dc.py" \
  > "$OUT/logs/rule2_canary.log" 2>&1
rule2_rc=$?
echo "rule2 canary rc=$rule2_rc"

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

# Fail closed on incomplete or invalid evidence. Ring/opamp are report-only, so
# their stable nonzero verdicts are allowed; infrastructure errors, OMP flips,
# binding-suite failures, and a failed Rule-2 canary are not.
verdict_rc=0
fail_verdict() {
  echo "T3 VALIDATION ERROR: $*" >&2
  verdict_rc=1
}

[ "$pool_rc" -eq 0 ] || fail_verdict "worker pool rc=$pool_rc"
[ "$rule2_rc" -eq 0 ] || fail_verdict "Rule-2 canary rc=$rule2_rc"

for suite in $SUITES; do
  for tech in $TECHS; do
    first_rc=""
    for omp in 1 2 4; do
      base="$OUT/logs/${suite}_${tech}.log.omp${omp}"
      for suffix in "" .full .rc; do
        [ -f "${base}${suffix}" ] || fail_verdict "missing ${base}${suffix}"
      done
      [ -f "${base}.rc" ] || continue
      run_rc="$(<"${base}.rc")"
      case "$run_rc" in
        0|1) ;;
        *) fail_verdict "invalid rc '$run_rc' in ${base}.rc"; continue ;;
      esac
      if [ -z "$first_rc" ]; then
        first_rc="$run_rc"
      elif [ "$run_rc" != "$first_rc" ]; then
        fail_verdict "OMP verdict flip for $suite/$tech ($first_rc -> $run_rc)"
      fi
      case "$suite" in
        verify_complex_sram_snm|verify_complex_switchcap)
          [ "$run_rc" -eq 0 ] || fail_verdict "binding failure for $suite/$tech OMP=$omp"
          ;;
      esac
      if [ -f "${base}.full" ] && grep -qiE \
          'traceback|cuda error|out of memory|segmentation fault|killed worker|infrastructure failure|^MISSING ' \
          "${base}.full"; then
        fail_verdict "infrastructure error in ${base}.full"
      fi
    done
  done
done

if [ "$verdict_rc" -eq 0 ]; then
  echo "T3 binding verdict: PASS"
else
  echo "T3 binding verdict: FAIL" >&2
fi
exit "$verdict_rc"
