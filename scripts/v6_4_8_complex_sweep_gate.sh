#!/usr/bin/env bash
# Run a DirectNet complex-circuit parametric sweep under the V6.4.8 gate
# methodology: CPU-only (CUDA_VISIBLE_DEVICES=""), single-thread (OMP=MKL=1),
# repo-local NGSPICE (tools/ngspice-45.2). The high-gain opamp / SRAM trip are
# multistable — float-reduction order changes the NR basin, so pinning is
# mandatory (V6.4.8 MEMORY: run gates on CPU, not CUDA).
#
# Usage: scripts/v6_4_8_complex_sweep_gate.sh <opamp|ringosc|switchcap|sram> [--tech ...] [--dimension ...]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
CIRCUIT="${1:?usage: $0 <opamp|ringosc|switchcap|sram> [driver args...]}"; shift || true

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"
export PYTHONPATH="$ROOT:$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

conda run -n pycircuitsim python "tests/verify_complex_${CIRCUIT}_sweep.py" "$@"
