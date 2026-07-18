#!/usr/bin/env bash
# V6.6.1 recipe campaign — evaluate recipe-variant checkpoints on the full NN
# gate matrix (complex circuits + device sweeps + AC), env-pinning each recipe's
# tsmc{X}_dn_{recipe}_{size}_{dev} checkpoints.
#
# For recipe=clean the pin targets the production tsmc{X}_dn_{size}_{dev}
# checkpoints (the control). All other recipes pin tsmc{X}_dn_{recipe}_{size}_{dev}.
# PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS} is read FIRST by the resolver and
# routes through the same per-tech cascade the complex gates use.
#
# COLLISION SAFETY: the verify_complex_* harnesses key scratch dirs by
# (circuit, tech) but NOT by recipe/size — so we SERIALIZE across (recipe,size)
# and PARALLELIZE (tech x suite) within one (recipe,size). Authoritative metrics
# are parsed from the per-cell stdout logs captured here.
#
# CPU-pinned gate methodology: CUDA_VISIBLE_DEVICES="" OMP=MKL=1, repo ngspice.
# Resumable: skips a cell whose log has ===BENCH_DONE.
#
# Usage: RECIPES="csob sob ekv clean" SIZES="large xl" PAR=12 bash scripts/recipe_eval.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
PAR="${PAR:-12}"
# V6.6.2: invoke the env python DIRECTLY (the `conda run` wrapper intermittently
# receives SIGSTKFLT under this harness → empty logs). Override with NN_PY.
NN_PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$NN_PY" ] || NN_PY="python"

# V6.8 — MODEL env selects the NN under test (see gate_matrix_iso.sh):
# direct (default; DN pins, results/recipe_bench) or transformer (BSIMAR:
# `_tf_` stems, TF pins, PYCIRCUITSIM_NN_FORCE_LEVEL=74, results/bsimar_bench).
# Do NOT run both models concurrently (scratch dirs keyed by circuit,tech).
MODEL="${MODEL:-direct}"
case "$MODEL" in
  direct)      TAG="dn"; OUT_DEFAULT="$ROOT/results/recipe_bench" ;;
  transformer) TAG="tf"; OUT_DEFAULT="$ROOT/results/bsimar_bench" ;;
  tabpfn)      TAG="pfn"; OUT_DEFAULT="$ROOT/results/pfn_bench" ;;
  *) echo "[test] UNKNOWN MODEL=$MODEL (direct|transformer|tabpfn)"; exit 1 ;;
esac
export MODEL
OUT="${EVAL_OUT:-$OUT_DEFAULT}"

suites=(verify_nn_multi_tech_dc verify_nn_multi_tech_tran \
        verify_complex_ring_osc verify_complex_opamp \
        verify_complex_sram_snm verify_complex_switchcap \
        verify_nn_ac verify_complex_opamp_ac)
# TECHS env (uppercase, space-separated) lets a campaign target a subset / a new
# tech (e.g. TECHS=TSMC6 for the V6.9.x onboarding gate). Lowercase derived.
read -r -a techs_uc <<< "${TECHS:-TSMC5 TSMC7 TSMC12 TSMC16}"
techs_lc=(); for _t in "${techs_uc[@]}"; do techs_lc+=("$(echo "$_t" | tr '[:upper:]' '[:lower:]')"); done

# Map (recipe,tech,size) -> checkpoint stem. clean -> production names.
stem () {  # recipe tech size dev
  if [ "$1" = "clean" ]; then echo "$2_${TAG}_$3_$4"; else echo "$2_${TAG}_$1_$3_$4"; fi
}

# ---- single-job worker ----
if [ "${1:-}" = "_one" ]; then
  recipe="$2"; size="$3"; tuc="$4"; tlc="$5"; suite="$6"
  log="$OUT/$recipe/$size/$tlc/${suite}.log"
  mkdir -p "$(dirname "$log")"
  if grep -q "===BENCH_DONE" "$log" 2>/dev/null; then echo "[test] SKIP $recipe/$size/$tlc/$suite"; exit 0; fi
  sn="$(stem "$recipe" "$tlc" "$size" nmos)"; sp="$(stem "$recipe" "$tlc" "$size" pmos)"
  if [ ! -f "$CKPT/${sn}_best.pt" ] || [ ! -f "$CKPT/${sp}_best.pt" ]; then
    echo "[test] NO-CKPT $recipe/$size/$tlc -> skip $suite"; echo "===BENCH_DONE no-ckpt===" > "$log"; exit 0
  fi
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
  if [ "$TAG" = "tf" ]; then
    export PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS="$sn"
    export PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS="$sp"
    export PYCIRCUITSIM_NN_FORCE_LEVEL=74
  elif [ "$TAG" = "pfn" ]; then
    export PYCIRCUITSIM_NN_CHECKPOINT_PFN_NMOS="$sn"
    export PYCIRCUITSIM_NN_CHECKPOINT_PFN_PMOS="$sp"
    export PYCIRCUITSIM_NN_FORCE_LEVEL=75
  else
    export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn"
    export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
  fi
  echo "[test] RUN $recipe/$size/$tlc/$suite (pin $sn / $sp)"
  "$NN_PY" -u "$ROOT/tests/${suite}.py" --tech "$tuc" > "$log" 2>&1
  rc=$?
  echo "===BENCH_DONE rc=$rc===" >> "$log"
  echo "[test] END $recipe/$size/$tlc/$suite rc=$rc"
  exit 0
fi

# ---- dispatcher ----
read -r -a recipes <<< "${RECIPES:-csob sob ekv clean}"
read -r -a sizes   <<< "${SIZES:-large xl}"
for recipe in "${recipes[@]}"; do for size in "${sizes[@]}"; do
  echo "[test] ===== $recipe / $size ====="
  specs=()
  for i in "${!techs_uc[@]}"; do for suite in "${suites[@]}"; do
    specs+=("$recipe $size ${techs_uc[$i]} ${techs_lc[$i]} $suite")
  done; done
  printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
  echo "[test] ===== $recipe / $size COMPLETE ====="
done; done
echo "[test] ALL RECIPE TESTS COMPLETE"
