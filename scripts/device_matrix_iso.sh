#!/usr/bin/env bash
# V6.6.2 re-test — fully-ISOLATED parallel device-level suites (DC sweep, tran
# sweep, device AC) for finalist recipes. Mirrors gate_matrix_iso.sh: each cell
# pins the recipe checkpoints and gets its OWN PYCIRCUITSIM_NN_RESULTS dir, so
# all (recipe × tech × suite) cells run concurrently with zero collision.
# CPU-pinned. Logs -> results/recipe_bench/device_iso/<recipe>/<tech>_<suite>.log
#
# Usage: RECIPES="clean crit15 csob" PAR=24 bash scripts/device_matrix_iso.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
OUT="$ROOT/results/recipe_bench/device_iso"
PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$PY" ] || PY="python"
PAR="${PAR:-24}"
SCRATCH="${DEV_SCRATCH:-/tmp/pycircuitsim_device_iso}"

read -r -a suites <<< "${SUITES:-verify_nn_multi_tech_dc verify_nn_multi_tech_tran verify_nn_ac}"

stem () { [ "$1" = clean ] && echo "$2_dn_large_$3" || echo "$2_dn_$1_large_$3"; }

if [ "${1:-}" = "_one" ]; then
  recipe="$2"; tuc="$3"; suite="$4"
  tlc="$(echo "$tuc" | tr 'A-Z' 'a-z')"
  sn="$(stem "$recipe" "$tlc" nmos)"; sp="$(stem "$recipe" "$tlc" pmos)"
  log="$OUT/$recipe/${tlc}_${suite}.log"
  mkdir -p "$(dirname "$log")"
  # audit B3 — exit 3 = "this cell reached NO verdict" (a FAIL is a verdict and
  # still exits 0). Never 255: xargs aborts the whole run on 255.
  if [ ! -f "$CKPT/${sn}_best.pt" ] || [ ! -f "$CKPT/${sp}_best.pt" ]; then
    echo "NO-CKPT $recipe $tlc" | tee "$log"; exit 3
  fi
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
  export PYCIRCUITSIM_NN_RESULTS="$SCRATCH/$recipe/${tlc}_${suite}"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
  mkdir -p "$PYCIRCUITSIM_NN_RESULTS"
  "$PY" -u "$ROOT/tests/${suite}.py" --tech "$tuc" > "$log" 2>&1
  rc=$?
  echo "[dev] $recipe/$tlc/$suite rc=$rc"
  # rc 124 (timeout) / >=126 (cannot-exec, killed by signal) = no verdict.
  if [ "$rc" -eq 124 ] || [ "$rc" -ge 126 ]; then exit 3; fi
  exit 0
fi

read -r -a recipes <<< "${RECIPES:-clean}"
read -r -a techs <<< "${TECHS:-TSMC5 TSMC7 TSMC12 TSMC16}"
specs=()
for r in "${recipes[@]}"; do for t in "${techs[@]}"; do for s in "${suites[@]}"; do
  specs+=("$r $t $s")
done; done; done
# audit B3 — capture xargs BEFORE the trailing echo, which would otherwise
# overwrite $? with 0 and report ALL DONE over cells that never ran.
printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
xrc=$?
if [ "$xrc" -ne 0 ]; then
  echo "[dev] INFRASTRUCTURE FAILURE: cells produced no verdict (xargs rc=$xrc)" >&2
  exit 1
fi
echo "[dev] ALL DONE"
