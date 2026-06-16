#!/usr/bin/env bash
# S14c — force_ic seed sweep over existing checkpoints (fast probe, no butterfly).
# Usage: bash scripts/v6_4_7_s14c_forceic_sweep.sh <TECH> <stem1> [stem2 ...]
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
  local vd; vd=$(mktemp -d /tmp/s14c_XXXX)
  local ok=1
  for dev in nmos pmos; do
    cp "$CK/${stem}_${dev}_best.pt"  "$vd/${tech_lc}_dn_medium_${dev}_best.pt"  2>/dev/null || ok=0
    cp "$CK/${stem}_${dev}_norm.npz" "$vd/${tech_lc}_dn_medium_${dev}_norm.npz" 2>/dev/null || ok=0
  done
  if [ "$ok" = 0 ]; then echo "$stem  MISSING"; rm -rf "$vd"; return; fi
  BSIMAR_CHECKPOINT_DIR="$vd" timeout 300 conda run --no-capture-output -n pycircuitsim \
    python scripts/v6_4_7_s14c_forceic_sweep.py "$TECH" "$stem" 2>/dev/null \
    | grep -E "RESULT|force_ic state" | tr '\n' ' '; echo
  rm -rf "$vd"
}
export -f one
export TECH tech_lc ROOT CK NGSPICE_BIN PYTHONPATH

printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'one "$@"' _ {}
