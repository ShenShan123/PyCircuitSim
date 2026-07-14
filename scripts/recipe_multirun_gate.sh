#!/usr/bin/env bash
# V6.6.2 (plan docs/plans/2026-07-01) §9 discipline #3 — multi-run ONE
# (recipe, size, tech, suite) gate at OMP∈{1,2,4}, pinned to that recipe's
# per-tech checkpoints, so we bank only DETERMINISTIC margin, not a single
# edge-of-threshold coin-flip. VTC/opamp trips have ~±1% run-to-run scatter and
# the opamp is multistable (memory v648/v659); a lone pass near the gate is
# meaningless. CPU-pinned, repo ngspice (methodology matches recipe_eval.sh).
#
# Usage: bash scripts/recipe_multirun_gate.sh <recipe> <size> <TECH_UC> <suite>
#   e.g. bash scripts/recipe_multirun_gate.sh invtripft large TSMC5 verify_complex_ring_osc
#   recipe=clean pins the production tsmc{X}_dn_{size}_{dev} control.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NG="$ROOT/tools/ngspice-45.2/bin/ngspice"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
recipe="$1"; size="$2"; tuc="$3"; suite="$4"
tlc="$(echo "$tuc" | tr 'A-Z' 'a-z')"
# V6.8 — MODEL env selects the NN under test (direct default | transformer);
# see gate_matrix_iso.sh. Transformer: `_tf_` stems + TF pins + FORCE_LEVEL.
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
export CUDA_VISIBLE_DEVICES="" NGSPICE_BIN="$NG"
if [ "$TAG" = "tf" ]; then
  export PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS="$sp"
  export PYCIRCUITSIM_NN_FORCE_LEVEL=74
elif [ "$TAG" = "pfn" ]; then
  export PYCIRCUITSIM_NN_CHECKPOINT_PFN_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_PFN_PMOS="$sp"
  export PYCIRCUITSIM_NN_FORCE_LEVEL=75
else
  export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$sn" PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$sp"
fi
echo "### multirun $recipe/$size/$tlc/$suite  (pin $sn / $sp, model $MODEL)"
for omp in 1 2 4; do
  # PYCIRCUITSIM_TORCH_THREADS=$omp: since the V6.6.6 harness thread-pin,
  # OMP_NUM_THREADS alone no longer perturbs torch GEMM threading — the
  # override is what makes this probe exercise the multistability axis.
  out="$(OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp PYCIRCUITSIM_TORCH_THREADS=$omp \
        conda run -n pycircuitsim python -u "$ROOT/tests/${suite}.py" --tech "$tuc" 2>&1)"
  # headline lines the gates print (period error / gain / charge / SNM / status)
  hl="$(printf '%s\n' "$out" | grep -iE "period error|gain_err|gain error|charge_err|charge error|SNM|butterfly|-> *(PASS|FAIL)|Status" | tr '\n' ' ')"
  echo "  OMP=$omp | $hl"
done
