#!/usr/bin/env bash
# V6.5.5 T1 — gate a KCL-fine-tuned tsmc7 checkpoint pair.
#
# Swapping ONLY tsmc7's N/P checkpoints affects ONLY the 4 tsmc7 gates (the other
# 12 use different per-tech checkpoints), so the "full 16-gate matrix" check
# reduces to tsmc7's own gates + device DC/tran/AC + the lifted-source canary,
# PLUS the two routing confirmations (KCL residual must drop; 1c gain must hold).
#
# Usage:  bash scripts/v6_5_5_gate_kcl.sh tsmc7_dn_kcl_l05
set -u
STEM="${1:?usage: v6_5_5_gate_kcl.sh <exp_stem e.g. tsmc7_dn_kcl_l05>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="$STEM"
export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="$STEM"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NGSPICE_BIN="$ROOT/tools/ngspice-45.2/bin/ngspice"
PY="conda run --no-capture-output -n pycircuitsim python"
LOG="$ROOT/results/v6_5_5/kcl_logs/gate_${STEM}.log"
mkdir -p "$(dirname "$LOG")"
echo "==== GATE $STEM ====" | tee "$LOG"

run() {  # label  cmd...
  local label="$1"; shift
  echo -e "\n----- $label -----" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  # surface the key result lines
  grep -iE "VERDICT|RESULT|PASS|FAIL|gain|frac|vo1i|NRMSE|0/|1/|2/|3/|4/" "$LOG" | tail -8
  echo "[$label] exit=$rc"
}

# 1. routing confirmations (tsmc7 only — do NOT pass tsmc12 here, the global
#    override would force it onto the tsmc7 checkpoint).
run "KCL-residual (vo1i frac must drop from 0.128)" \
    $PY tests/diag_opamp_kcl_residual.py --tech TSMC7
run "1c basin-seed (gain must HOLD >0 when seeded from L72 OP)" \
    $PY tests/diag_opamp_basin_seed.py --tech TSMC7

# 2. the win + the no-regress tsmc7 complex gates
run "OPAMP gate (TARGET: FAIL->PASS)"   $PY tests/verify_complex_opamp.py    --tech TSMC7
run "RING gate (must stay PASS)"        $PY tests/verify_complex_ring_osc.py --tech TSMC7
run "SWITCHCAP gate (must stay PASS)"   $PY tests/verify_complex_switchcap.py --tech TSMC7
run "SRAM gate (must stay PASS)"        $PY tests/verify_complex_sram_snm.py --tech TSMC7

# 3. tsmc7 device-level DC/tran (inverter VTC + transient + per-device) and AC
run "DEVICE DC/TRAN (TSMC7)"            $PY tests/verify_nn_dc_tran.py --tech TSMC7
run "DEVICE AC (TSMC7)"                 $PY tests/verify_nn_ac.py --tech TSMC7 --device nmos,pmos

# 4. lifted-source canary (Rule 2). Global override forces ALL techs onto the
#    tsmc7 checkpoint, so ONLY the tsmc7 row is meaningful here — read it.
run "LIFTED-SOURCE canary (read TSMC7 row only)" \
    $PY tests/verify_nn_lifted_source_dc.py

echo -e "\n==== GATE $STEM COMPLETE (full log: $LOG) ===="
