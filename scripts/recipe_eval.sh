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
OUT="$ROOT/results/recipe_bench"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
PAR="${PAR:-12}"

suites=(verify_nn_multi_tech_dc verify_nn_multi_tech_tran \
        verify_complex_ring_osc verify_complex_opamp \
        verify_complex_sram_snm verify_complex_switchcap \
        verify_nn_ac verify_complex_opamp_ac)
techs_uc=(TSMC5 TSMC7 TSMC12 TSMC16)
techs_lc=(tsmc5 tsmc7 tsmc12 tsmc16)

# Map (recipe,tech,size) -> checkpoint stem. clean -> production names.
stem () {  # recipe tech size dev
  if [ "$1" = "clean" ]; then echo "$2_dn_$3_$4"; else echo "$2_dn_$1_$3_$4"; fi
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
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
  echo "[test] RUN $recipe/$size/$tlc/$suite (pin $sn / $sp)"
  conda run -n pycircuitsim python -u "$ROOT/tests/${suite}.py" --tech "$tuc" > "$log" 2>&1
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
  for i in 0 1 2 3; do for suite in "${suites[@]}"; do
    specs+=("$recipe $size ${techs_uc[$i]} ${techs_lc[$i]} $suite")
  done; done
  printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
  echo "[test] ===== $recipe / $size COMPLETE ====="
done; done
echo "[test] ALL RECIPE TESTS COMPLETE"
