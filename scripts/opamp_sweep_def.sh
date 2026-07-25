#!/usr/bin/env bash
# V6.6.2 DEFINITIVE opamp verdict — the opamp OP is multistable and CPU-float-
# basin-sensitive (memory v648/v659), so the authoritative gate is OMP∈{1,2,4},
# UNCONTENDED (no concurrent training), isolated dirs. A cell PASSES only if ALL
# THREE OMP runs pass (§9 discipline #3 — bank only deterministic margin).
#
# Also multiruns ring_osc (the tsmc7-ring 0.18%-margin fragile hold) the same way.
# Fans out (recipe × tech × circuit × OMP) in parallel with per-process OMP pins
# (each process keeps its own thread count → multistability probe stays valid)
# and isolated PYCIRCUITSIM_COMPLEX_RESULTS.
#
# Usage: RECIPES="crit15 crit20 crit30 crit15m crit15h crit10" CIRCS="opamp ring_osc" \
#        PAR=24 bash scripts/opamp_sweep_def.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"; [ -x "$PY" ] || PY="python"
OUT="${OPDEF_OUT:-$ROOT/results/recipe_bench/opamp_def}"
SCR="${OPDEF_SCRATCH:-/tmp/claude-1001/-data2-shenshan-PyCircuitSim/6bfa5fc9-5de2-4965-944a-7f36ddc3c72c/scratchpad/opdef}"
PAR="${PAR:-24}"
SIZE="${SIZE:-large}"

stem () { [ "$1" = clean ] && echo "$2_dn_${SIZE}_$3" || echo "$2_dn_$1_${SIZE}_$3"; }

if [ "${1:-}" = "_one" ]; then
  recipe="$2"; tuc="$3"; circ="$4"; omp="$5"
  tlc="$(echo "$tuc" | tr 'A-Z' 'a-z')"
  sn="$(stem "$recipe" "$tlc" nmos)"; sp="$(stem "$recipe" "$tlc" pmos)"
  d="$OUT/$recipe"; mkdir -p "$d"
  # audit B3 — exit 3 = "this cell reached NO verdict"; an rc=1 FAIL run IS a
  # verdict and exits 0. Never 255: xargs aborts the whole run on 255.
  [ -f "$CKPT/${sn}_best.pt" ] || { echo "$tuc $circ OMP$omp NO-CKPT" >> "$d/${circ}.txt"; exit 3; }
  # PYCIRCUITSIM_TORCH_THREADS keeps the multistability probe alive now that
  # tests/common/complex.py pins torch threads to 1 by default (v664 P0)
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$omp" MKL_NUM_THREADS="$omp" \
         PYCIRCUITSIM_TORCH_THREADS="$omp" NGSPICE_BIN="$NG"
  export PYCIRCUITSIM_COMPLEX_RESULTS="$SCR/${recipe}_${tlc}_${circ}_omp${omp}"
  mkdir -p "$PYCIRCUITSIM_COMPLEX_RESULTS"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
  out="$("$PY" -u "$ROOT/tests/verify_complex_${circ}.py" --tech "$tuc" 2>&1)"; rc=$?
  hl="$(printf '%s\n' "$out" | grep -iE "gain error|period error" | grep -viE "resolver|MEXP" | head -1 | sed 's/^ *//')"
  echo "$tuc $circ OMP$omp rc=$rc | $hl" >> "$d/${circ}.txt"
  # rc 124 (timeout) / >=126 (cannot-exec, killed by signal) = no verdict; the
  # tally below would otherwise read the row as a plain FAIL.
  if [ "$rc" -eq 124 ] || [ "$rc" -ge 126 ]; then exit 3; fi
  exit 0
fi

read -r -a recipes <<< "${RECIPES:-crit15 crit20 crit30 crit15m crit15h crit10}"
read -r -a circs   <<< "${CIRCS:-opamp ring_osc}"
read -r -a techs   <<< "${TECHS:-TSMC5 TSMC7 TSMC12 TSMC16}"
specs=()
for r in "${recipes[@]}"; do rm -f "$OUT/$r"/*.txt 2>/dev/null; mkdir -p "$OUT/$r"
  for t in "${techs[@]}"; do for c in "${circs[@]}"; do for o in ${OMPS:-1 2 4}; do
    specs+=("$r $t $c $o"); done; done; done; done
# audit B3 — capture xargs BEFORE the trailing echo/tally, which would otherwise
# overwrite $? with 0. Verdict is decided after the tally has been printed.
printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
xrc=$?
echo "=== DEFINITIVE OPAMP/RING SWEEP DONE ==="
# tally: a (recipe,tech,circ) cell PASSES iff all 3 OMP runs rc=0
for r in "${recipes[@]}"; do
  echo "----- $r -----"
  for c in "${circs[@]}"; do
    [ -f "$OUT/$r/${c}.txt" ] || continue
    for t in "${techs[@]}"; do
      pass=$(grep -c "^$t $c .*rc=0" "$OUT/$r/${c}.txt" 2>/dev/null)
      tot=$(grep -c "^$t $c " "$OUT/$r/${c}.txt" 2>/dev/null)
      v="FAIL"; [ "$pass" = "$tot" ] && [ "$tot" -gt 0 ] && v="PASS"
      det=$(grep "^$t $c " "$OUT/$r/${c}.txt" 2>/dev/null | grep -oE "(gain|period) error = [0-9.]+%" | tr '\n' ' ')
      printf "  %-7s %-9s %s (%s/%s) | %s\n" "$t" "$c" "$v" "$pass" "$tot" "$det"
    done
  done
done
# audit B3/B5f — a cell that recorded no row at all silently shrinks the (pass/tot)
# denominators printed above, so a half-killed sweep reads as a smaller-but-clean
# tally. Compare rows on disk against the number of cells dispatched.
recorded=0
for r in "${recipes[@]}"; do for c in "${circs[@]}"; do
  [ -f "$OUT/$r/${c}.txt" ] || continue
  recorded=$((recorded + $(grep -c . "$OUT/$r/${c}.txt")))
done; done
if [ "$xrc" -ne 0 ] || [ "$recorded" -lt "${#specs[@]}" ]; then
  echo "[opdef] INFRASTRUCTURE FAILURE: $recorded/${#specs[@]} cells recorded a row" \
       "(xargs rc=$xrc) — the tallies above are PARTIAL" >&2
  exit 1
fi
