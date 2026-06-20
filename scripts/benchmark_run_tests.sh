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
OUT="$ROOT/results/benchmark_sml"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
PAR="${PAR:-12}"

suites=(verify_nn_multi_tech_dc verify_nn_multi_tech_tran \
        verify_complex_ring_osc verify_complex_opamp \
        verify_complex_sram_snm verify_complex_switchcap)
techs_uc=(TSMC5 TSMC7 TSMC12 TSMC16)
techs_lc=(tsmc5 tsmc7 tsmc12 tsmc16)
sizes=(small medium large)

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  size="$2"; tuc="$3"; tlc="$4"; suite="$5"
  log="$OUT/$size/$tlc/${suite}.log"
  mkdir -p "$(dirname "$log")"
  if grep -q "===BENCH_DONE" "$log" 2>/dev/null; then echo "[test] SKIP $size/$tlc/$suite"; exit 0; fi
  if [ ! -f "$CKPT/${tlc}_dn_${size}_nmos_best.pt" ] || [ ! -f "$CKPT/${tlc}_dn_${size}_pmos_best.pt" ]; then
    echo "[test] NO-CKPT $size/$tlc -> skip $suite"; echo "===BENCH_DONE no-ckpt===" > "$log"; exit 0
  fi
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="${tlc}_dn_${size}_nmos"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="${tlc}_dn_${size}_pmos"
  echo "[test] RUN $size/$tlc/$suite"
  conda run -n pycircuitsim python -u "$ROOT/tests/${suite}.py" --tech "$tuc" > "$log" 2>&1
  rc=$?
  echo "===BENCH_DONE rc=$rc===" >> "$log"
  echo "[test] END $size/$tlc/$suite rc=$rc"
  exit 0
fi

# ---- dispatcher: serialize sizes, parallelize (tech x suite) within a size ----
for size in "${sizes[@]}"; do
  echo "[test] ===== SIZE $size ====="
  specs=()
  for i in 0 1 2 3; do for suite in "${suites[@]}"; do
    specs+=("$size ${techs_uc[$i]} ${techs_lc[$i]} $suite")
  done; done
  printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
  echo "[test] ===== SIZE $size COMPLETE ====="
done
echo "[test] ALL TESTS COMPLETE"
