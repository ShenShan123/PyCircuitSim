#!/usr/bin/env bash
# Benchmark Phase D — run the NN test suites once per (size, tech).
#
# For each (size, tech) the parser env-override pins the capacity-specific
# checkpoint (PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS} is read FIRST, before the
# medium-first preempt, and routes through the same resolver the complex tests
# use); the suite is invoked with --tech TSMC<XX>.
#
# COLLISION SAFETY: the harnesses key their scratch dirs by (circuit, tech) but
# NOT by size (e.g. tests/verify_complex_results/opamp/TSMC16). So we SERIALIZE
# across sizes and PARALLELIZE within a size — within one size all (tech x suite)
# output paths are disjoint. Authoritative metrics are parsed from the per-size
# stdout logs we capture here (results/benchmark_sml/<size>/<tech>/<suite>.log).
#
# CPU-pinned per the gate methodology: CUDA_VISIBLE_DEVICES="" OMP=MKL=1,
# repo tools/ngspice-45.2. Resumable: skips a job whose log has ===BENCH_DONE.
#
# Usage: PAR=12 bash scripts/benchmark_run_tests.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/neural_network/checkpoints"
PAR="${PAR:-12}"

# V6.8 — MODEL env selects the NN under test (see gate_matrix_iso.sh):
# direct (default; DN pins, results/benchmark_sml) or transformer (BSIMAR:
# `_tf_` stems, TF pins, PYCIRCUITSIM_NN_FORCE_LEVEL=74 harness retarget,
# results/bsimar_bench). DO NOT run both models concurrently — the harness
# scratch dirs are keyed by (circuit, tech) only.
MODEL="${MODEL:-direct}"
case "$MODEL" in
  direct)      TAG="dn"; OUT_DEFAULT="$ROOT/results/benchmark_sml" ;;
  transformer) TAG="tf"; OUT_DEFAULT="$ROOT/results/bsimar_bench" ;;
  *) echo "[test] UNKNOWN MODEL=$MODEL (direct|transformer)"; exit 1 ;;
esac
export MODEL
OUT="${BENCH_OUT:-$OUT_DEFAULT}"

suites=(verify_nn_multi_tech_dc verify_nn_multi_tech_tran \
        verify_complex_ring_osc verify_complex_opamp \
        verify_complex_sram_snm verify_complex_switchcap \
        verify_nn_ac verify_complex_opamp_ac)
techs_uc=(TSMC5 TSMC7 TSMC12 TSMC16)
techs_lc=(tsmc5 tsmc7 tsmc12 tsmc16)
# SIZES env lets us pipeline (run small+medium while large is still training).
read -r -a sizes <<< "${SIZES:-small medium large xl}"

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  size="$2"; tuc="$3"; tlc="$4"; suite="$5"
  log="$OUT/$size/$tlc/${suite}.log"
  mkdir -p "$(dirname "$log")"
  # audit B3 — worker exit code means "did this cell reach a VERDICT?", not
  # "did it pass": a PASS and a FAIL are both scientific results (exit 0), while
  # a cell that could not be judged at all exits 3 so the dispatcher can fail
  # loudly. Never 255 — xargs aborts the whole run on 255.
  # audit B5m — the NO-CKPT marker contains the resume sentinel, so it used to
  # skip its own cell forever, even after the checkpoint had been trained. Retry
  # the existence test on a pill log and clear it if the checkpoints now exist;
  # only a still-missing checkpoint keeps the no-verdict exit 3.
  if grep -q "===BENCH_DONE no-ckpt===" "$log" 2>/dev/null; then
    if [ -f "$CKPT/${tlc}_${TAG}_${size}_nmos_best.pt" ] && [ -f "$CKPT/${tlc}_${TAG}_${size}_pmos_best.pt" ]; then
      echo "[test] RETRY $size/$tlc/$suite (checkpoints appeared since the NO-CKPT record)"; rm -f "$log"
    else
      echo "[test] SKIP $size/$tlc/$suite (recorded NO-CKPT — still no verdict)"; exit 3
    fi
  fi
  if grep -q "===BENCH_DONE" "$log" 2>/dev/null; then echo "[test] SKIP $size/$tlc/$suite"; exit 0; fi
  if [ ! -f "$CKPT/${tlc}_${TAG}_${size}_nmos_best.pt" ] || [ ! -f "$CKPT/${tlc}_${TAG}_${size}_pmos_best.pt" ]; then
    echo "[test] NO-CKPT $size/$tlc -> skip $suite"; echo "===BENCH_DONE no-ckpt===" > "$log"; exit 3
  fi
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
  if [ "$TAG" = "tf" ]; then
    export PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS="${tlc}_tf_${size}_nmos"
    export PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS="${tlc}_tf_${size}_pmos"
    export PYCIRCUITSIM_NN_FORCE_LEVEL=74
  else
    export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="${tlc}_dn_${size}_nmos"
    export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="${tlc}_dn_${size}_pmos"
  fi
  # Campaign suite IDs are persisted in historical evidence. Keep those keys
  # stable while resolving renamed simple-circuit modules on disk.
  script_suite="${suite/verify_complex_/verify_circuit_}"
  test_file="$ROOT/tests/simple_circuits/${script_suite}.py"
  if [ "$suite" = "verify_nn_multi_tech_dc" ]; then
    test_file="$ROOT/tests/single_devices/${suite}.py"
  fi
  [ -f "$test_file" ] || {
    echo "[test] UNKNOWN suite path: $suite" > "$log"
    exit 3
  }
  echo "[test] RUN $size/$tlc/$suite"
  conda run -n pycircuitsim python -u "$test_file" --tech "$tuc" > "$log" 2>&1
  rc=$?
  echo "===BENCH_DONE rc=$rc===" >> "$log"
  echo "[test] END $size/$tlc/$suite rc=$rc"
  # audit B3 — rc 124 (timeout) and rc >= 126 (cannot-exec / killed by signal)
  # mean the suite never printed a verdict; a plain nonzero rc IS the FAIL verdict.
  if [ "$rc" -eq 124 ] || [ "$rc" -ge 126 ]; then exit 3; fi
  # ...as does a log holding nothing but the marker (the empty-log failure mode).
  [ "$(grep -cv '^===BENCH_DONE' "$log")" -gt 0 ] || exit 3
  exit 0
fi

# ---- dispatcher: serialize sizes, parallelize (tech x suite) within a size ----
# audit B3 — xargs' status must be captured BEFORE any echo, or the echo's 0
# overwrites it and every no-verdict cell is swallowed. Propagation convention
# copied from scripts/benchmark_gen_data.sh (rc accumulator + `exit $rc`).
rc=0
for size in "${sizes[@]}"; do
  echo "[test] ===== SIZE $size ====="
  specs=()
  for i in 0 1 2 3; do for suite in "${suites[@]}"; do
    specs+=("$size ${techs_uc[$i]} ${techs_lc[$i]} $suite")
  done; done
  printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
  xrc=$?
  [ "$xrc" -eq 0 ] || { echo "[test] SIZE $size: cells produced no verdict (xargs rc=$xrc)" >&2; rc=1; }
  echo "[test] ===== SIZE $size COMPLETE ====="
done
if [ "$rc" -ne 0 ]; then
  echo "[test] INFRASTRUCTURE FAILURE: some cells produced no verdict (see NO-CKPT / rc>=124 above)" >&2
  exit 1
fi
echo "[test] ALL TESTS COMPLETE"
