#!/usr/bin/env bash
# V6.6.2 — gate a recipe's 16 complex gates via the env python DIRECTLY
# (the `conda run` wrapper intermittently gets SIGSTKFLT under this harness).
# Writes one headline line per gate to results/recipe_bench/GATE_<recipe>.txt.
# Usage: bash scripts/corridor_gate_direct.sh <recipe> [TECH ...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
recipe="$1"; shift
techs=("$@"); [ ${#techs[@]} -eq 0 ] && techs=(TSMC5 TSMC7 TSMC12 TSMC16)
OUT="$ROOT/results/recipe_bench/GATE_${recipe}.txt"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NGSPICE_BIN="$NG"
stem () { [ "$recipe" = clean ] && echo "$1_dn_large_$2" || echo "$1_dn_${recipe}_large_$2"; }
# audit B3 — sequential driver: accumulate a no-verdict flag and `exit $rc` at
# the end (same convention as scripts/benchmark_gen_data.sh). The trailing echo
# used to be the last command, so this script always returned 0.
rc=0
for tuc in "${techs[@]}"; do
  tlc="$(echo "$tuc" | tr 'A-Z' 'a-z')"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$(stem "$tlc" nmos)"
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$(stem "$tlc" pmos)"
  for suite in verify_complex_ring_osc verify_complex_opamp verify_complex_sram_snm verify_complex_switchcap; do
    # Capture the raw run first (opamp_sweep_def.sh idiom): piping the suite
    # straight into grep threw its exit status away, and the filtered `line` is
    # byte-identical either way.
    raw="$("$PY" -u "$ROOT/tests/${suite}.py" --tech "$tuc" 2>&1)"; prc=$?
    line="$(printf '%s\n' "$raw" \
        | grep -viE "MEXP|resolver" \
        | grep -iE "period error|gain error|charge_err|charge error|GATE: PASS|GATE: FAIL|-> *(PASS|FAIL)" \
        | tr '\n' ' ')"
    echo "$tuc ${suite#verify_complex_} | $line" >> "$OUT"
    # An empty headline means no gate line was ever printed (crash / killed run)
    # — a blank row in GATE_<recipe>.txt reads as "ran, said nothing", not as an
    # error. rc >= 126 is cannot-exec / killed by signal. A FAIL row is a result.
    if [ -z "${line// /}" ] || [ "$prc" -ge 126 ]; then
      echo "[corridor] NO VERDICT: $tuc ${suite#verify_complex_} (rc=$prc)" >&2
      rc=1
    fi
  done
done
echo "=== GATE_${recipe}_DONE ${techs[*]} ===" >> "$OUT"
if [ "$rc" -ne 0 ]; then
  echo "[corridor] INFRASTRUCTURE FAILURE: some cells produced no verdict — $OUT is PARTIAL" >&2
fi
exit $rc
