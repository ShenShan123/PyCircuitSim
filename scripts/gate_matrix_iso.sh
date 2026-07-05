#!/usr/bin/env bash
# V6.6.2 verification — fully-ISOLATED parallel gate of any recipe over the full
# 16-cell complex matrix (4 techs x 4 circuits). Each cell gets its OWN
# PYCIRCUITSIM_COMPLEX_RESULTS dir (complex.py:56 env override) so ALL cells —
# even different recipes on the same (circuit,tech) — run concurrently with zero
# scratch collision. CPU-pinned (CUDA_VISIBLE_DEVICES="" OMP=MKL=1), repo ngspice.
#
# Per-cell verdict = the gate's EXIT CODE (0 iff that tech PASSED) + the printed
# headline. Results -> results/recipe_bench/gate_iso/<recipe>/<tech>_<circuit>.log
# and a one-line summary appended to .../<recipe>/SUMMARY.txt.
#
# Usage:
#   RECIPES="clean invtripft corroft" OMP=1 PAR=32 bash scripts/gate_matrix_iso.sh
#   (single-tech multirun): as worker, see recipe_multirun_gate.sh for OMP sweep.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
OUT="${GATE_OUT:-$ROOT/results/recipe_bench/gate_iso}"
PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$PY" ] || PY="python"
PAR="${PAR:-32}"
OMP="${OMP:-1}"
SIZE="${SIZE:-large}"
SCRATCH="${GATE_SCRATCH:-/tmp/claude-1001/-data2-shenshan-PyCircuitSim/6bfa5fc9-5de2-4965-944a-7f36ddc3c72c/scratchpad/gate_iso}"

# V6.8 — MODEL env selects the NN under test: direct (default, DN pins,
# LEVEL=73 decks as-is) or transformer (BSIMAR: `_tf_` stems, TF env pins,
# and PYCIRCUITSIM_NN_FORCE_LEVEL=74 so the LEVEL=73 harness decks retarget
# to the Transformer at parse time — see pycircuitsim/parser.py).
MODEL="${MODEL:-direct}"
case "$MODEL" in
  direct)      TAG="dn" ;;
  transformer) TAG="tf" ;;
  *) echo "[gate] UNKNOWN MODEL=$MODEL (direct|transformer)"; exit 1 ;;
esac
export MODEL

read -r -a circuits <<< "${CIRCS:-ring_osc opamp sram_snm switchcap}"

stem () {  # recipe tech dev
  if [ "$1" = "clean" ]; then echo "$2_${TAG}_${SIZE}_$3"; else echo "$2_${TAG}_$1_${SIZE}_$3"; fi
}

if [ "${1:-}" = "_one" ]; then
  recipe="$2"; tuc="$3"; circ="$4"
  tlc="$(echo "$tuc" | tr 'A-Z' 'a-z')"
  sn="$(stem "$recipe" "$tlc" nmos)"; sp="$(stem "$recipe" "$tlc" pmos)"
  log="$OUT/$recipe/${tlc}_${circ}.log"
  mkdir -p "$(dirname "$log")"
  cell="$OUT/$recipe/.cell_${tlc}_${circ}"
  if [ ! -f "$CKPT/${sn}_best.pt" ] || [ ! -f "$CKPT/${sp}_best.pt" ]; then
    echo "NO-CKPT $recipe $tlc ($sn/$sp)" | tee "$log"
    echo "$tuc $circ | NO-CKPT" > "$cell"; exit 0
  fi
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" NGSPICE_BIN="$NG"
  export PYCIRCUITSIM_COMPLEX_RESULTS="$SCRATCH/$recipe/${tlc}_${circ}"
  if [ "$TAG" = "tf" ]; then
    export PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS="$sn"
    export PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS="$sp"
    export PYCIRCUITSIM_NN_FORCE_LEVEL=74
  else
    export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn"
    export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
  fi
  mkdir -p "$PYCIRCUITSIM_COMPLEX_RESULTS"
  "$PY" -u "$ROOT/tests/verify_complex_${circ}.py" --tech "$tuc" > "$log" 2>&1
  rc=$?
  hl="$(grep -iE "period error|gain error|charge_err|charge error|hold droop|SNM=|butterfly|NRMSE|-> *(PASS|FAIL)" "$log" \
        | grep -viE "resolver|MEXP" | tr '\n' ' ' | sed 's/  */ /g')"
  verdict=$([ "$rc" -eq 0 ] && echo PASS || echo FAIL)
  # one file per cell (not an append to SUMMARY.txt): a subset re-run must
  # update only its own cells — the old append-after-rm wiped every other
  # circuit's rc line from SUMMARY.txt
  echo "$tuc $circ | rc=$rc $verdict | $hl" > "$cell"
  echo "[gate] $recipe/$tlc/$circ rc=$rc $verdict"
  exit 0
fi

read -r -a recipes <<< "${RECIPES:-clean}"
read -r -a techs <<< "${TECHS:-TSMC5 TSMC7 TSMC12 TSMC16}"
specs=()
for recipe in "${recipes[@]}"; do
  mkdir -p "$OUT/$recipe"
  for tuc in "${techs[@]}"; do for circ in "${circuits[@]}"; do
    specs+=("$recipe $tuc $circ")
  done; done
done
printf '%s\n' "${specs[@]}" | xargs -P "$PAR" -L1 "$SELF" _one
echo "[gate] ALL DONE — summaries:"
for recipe in "${recipes[@]}"; do
  dir="$OUT/$recipe"
  # rebuild SUMMARY.txt from the per-cell files, keeping any legacy line
  # whose cell was never re-run under the per-cell scheme
  tmp="$dir/.SUMMARY.tmp"; : > "$tmp"
  if [ -f "$dir/SUMMARY.txt" ]; then
    while IFS= read -r line; do
      key="$(echo "$line" | awk '{printf "%s_%s", tolower($1), $2}')"
      [ -f "$dir/.cell_$key" ] || printf '%s\n' "$line" >> "$tmp"
    done < "$dir/SUMMARY.txt"
  fi
  cat "$dir"/.cell_* >> "$tmp" 2>/dev/null
  sort "$tmp" > "$dir/SUMMARY.txt"; rm -f "$tmp"
  # tally the 4 circuit gates only (SUMMARY may also carry opamp_ac lines)
  np="$(grep -cE ' (ring_osc|opamp|sram_snm|switchcap) \| rc=0 PASS' "$dir/SUMMARY.txt" 2>/dev/null || echo 0)"
  echo "  === $recipe : ${np}/16 ==="
  cat "$dir/SUMMARY.txt" 2>/dev/null
done
