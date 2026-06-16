#!/usr/bin/env bash
# S14a — authoritative-gate opamp sweep over existing seed checkpoints.
# For each candidate stem, install it at {tech}_dn_medium in an isolated
# BSIMAR_CHECKPOINT_DIR and run verify_complex_opamp.py (the authoritative
# gate, NOT the scorer). Emits one line per candidate: stem  gain_err  status.
#
# Usage: bash scripts/v6_4_7_s14a_opamp_sweep.sh <TECH> <stem1> [stem2 ...]
set -u
TECH="${1:?usage: $0 <TECH> <stem...>}"; shift
tech_lc=$(echo "$TECH" | tr 'A-Z' 'a-z')
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CK="$ROOT/external_compact_models/bsimar/checkpoints"
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG"
cd "$ROOT"

one() {
  local stem="$1"
  local vd; vd=$(mktemp -d /tmp/s14a_XXXX)
  local ok=1
  for dev in nmos pmos; do
    cp "$CK/${stem}_${dev}_best.pt"  "$vd/${tech_lc}_dn_medium_${dev}_best.pt"  2>/dev/null || ok=0
    cp "$CK/${stem}_${dev}_norm.npz" "$vd/${tech_lc}_dn_medium_${dev}_norm.npz" 2>/dev/null || ok=0
  done
  if [ "$ok" = 0 ]; then echo "$stem  MISSING"; rm -rf "$vd"; return; fi
  local out
  out=$(BSIMAR_CHECKPOINT_DIR="$vd" timeout 400 conda run --no-capture-output -n pycircuitsim \
        python tests/verify_complex_opamp.py --tech "$TECH" 2>/dev/null \
        | grep -E "gain error|DirectNet gain=")
  local gain gerr st
  gain=$(echo "$out" | grep -oE "DirectNet gain=[0-9.]+" | head -1 | grep -oE "[0-9.]+")
  gerr=$(echo "$out" | grep -oE "gain error = [0-9.]+%" | head -1 | grep -oE "[0-9.]+")
  st=$(echo "$out" | grep -qE "FAIL" && echo FAIL || echo PASS)
  printf "%-40s gain=%-8s gain_err=%-8s %s\n" "$stem" "${gain:-NA}" "${gerr:-NA}" "$st"
  rm -rf "$vd"
}
export -f one
export TECH tech_lc ROOT CK NGSPICE_BIN PYTHONPATH

printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'one "$@"' _ {}
