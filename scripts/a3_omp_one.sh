#!/usr/bin/env bash
# V6.13.0 A3 re-gate — run ONE (recipe, size, tech, suite, OMP) multistability
# probe and APPEND its verdict to the shared per-cell log.
#
# recipe_multirun_gate.sh always runs OMP∈{1,2,4} back-to-back, so an
# interrupted sweep can only be resumed by paying again for the runs it already
# banked — 8-12 h per BSIM-AR opamp cell. This runs exactly one OMP value and
# emits a line byte-compatible with that driver ("  OMP=n | <headline>"), so the
# same collector reads logs assembled from either. Each invocation also gets its
# own PYCIRCUITSIM_COMPLEX_RESULTS, which recipe_multirun_gate.sh does not, so
# these are safe to fan out in parallel.
#
# Usage:
#   MODEL=transformer bash scripts/a3_omp_one.sh <recipe> <size> <TECH_UC> <suite> <omp> <logfile>
# e.g.
#   MODEL=transformer bash scripts/a3_omp_one.sh clean large TSMC5 verify_complex_opamp 4 out.log
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$PY" ] || PY="python"

recipe="$1"; size="$2"; tuc="$3"; suite="$4"; omp="$5"; log="$6"
tlc="$(echo "$tuc" | tr 'A-Z' 'a-z')"

MODEL="${MODEL:-direct}"
case "$MODEL" in
  direct)      TAG="dn" ;;
  transformer) TAG="tf" ;;
  tabpfn)      TAG="pfn" ;;
  *) echo "UNKNOWN MODEL=$MODEL (direct|transformer|tabpfn)"; exit 1 ;;
esac
if [ "$recipe" = "clean" ]; then sn="${tlc}_${TAG}_${size}_nmos"; sp="${tlc}_${TAG}_${size}_pmos"
else                             sn="${tlc}_${TAG}_${recipe}_${size}_nmos"; sp="${tlc}_${TAG}_${recipe}_${size}_pmos"; fi
[ -f "$CKPT/${sn}_best.pt" ] || { echo "MISSING $CKPT/${sn}_best.pt"; exit 1; }
[ -f "$CKPT/${sp}_best.pt" ] || { echo "MISSING $CKPT/${sp}_best.pt"; exit 1; }

# Accuracy campaigns default to CPU.  The V7.2 GPU-axis fidelity campaign
# opts in explicitly through GATE_CUDA_VISIBLE_DEVICES; keeping this selection
# separate from the simulator's PYCIRCUITSIM_NN_DEVICE flag makes accidental
# CUDA scoring impossible for every existing caller.
export CUDA_VISIBLE_DEVICES="${GATE_CUDA_VISIBLE_DEVICES:-}" NGSPICE_BIN="$NG"
if [ "$TAG" = "tf" ]; then
  export PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS="$sp"
  export PYCIRCUITSIM_NN_FORCE_LEVEL=74
elif [ "$TAG" = "pfn" ]; then
  export PYCIRCUITSIM_NN_CHECKPOINT_PFN_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_PFN_PMOS="$sp"
  export PYCIRCUITSIM_NN_FORCE_LEVEL=75
else
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
fi

SCRATCH="${GATE_SCRATCH:-/tmp/a3_omp}"
export PYCIRCUITSIM_COMPLEX_RESULTS="$SCRATCH/${TAG}_${recipe}_${size}_${tlc}_${suite}_omp${omp}"
mkdir -p "$PYCIRCUITSIM_COMPLEX_RESULTS" "$(dirname "$log")"

# PYCIRCUITSIM_TORCH_THREADS=$omp: since the V6.6.6 harness thread-pin,
# OMP_NUM_THREADS alone no longer perturbs torch GEMM threading — the override
# is what makes this probe exercise the multistability axis.
out="$(OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp PYCIRCUITSIM_TORCH_THREADS=$omp \
      "$PY" -u "$ROOT/tests/${suite}.py" --tech "$tuc" 2>&1)"
rc=$?
hl="$(printf '%s\n' "$out" | grep -iE "period error|gain_err|gain error|charge_err|charge error|SNM|butterfly|-> *(PASS|FAIL)|Status" | tr '\n' ' ')"
# One sidecar per OMP value, never an append to the shared log: sibling OMP
# values of the same cell run concurrently, and concurrent appends interleave.
# The orchestrator folds the sidecars back into <log> in OMP order at the end.
echo "  OMP=$omp | $hl" > "${log}.omp${omp}"
printf '%s\n' "$out" > "${log}.omp${omp}.full"
echo "[omp] $TAG/$recipe/$size/$tlc/$suite OMP=$omp rc=$rc"
