#!/usr/bin/env bash
# V6.4.7 S11 (P3) — score the subthreshold arm. For a candidate PREFIX:
#   1. force_ic + subvt probe (CPU) per seed  -> s11_force_ic_<prefix>_s<seed>.json
#   2. the 16-cell + inverter scorer (GPU, workers=1) via the s9b gate driver
#      -> S11_gate_<prefix>.md  (ring_osc/opamp/switchcap/inverter vs baseline)
# CPU force_ic gates run concurrently with the single-GPU scorer.
#
# Usage: PREFIX=v6_4_7_s11sub_w005 SEEDS="42 17 7 31" TECHS="tsmc7 tsmc5 tsmc12 tsmc16" \
#          bash scripts/v6_4_7_s11_score.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PREFIX="${PREFIX:?set PREFIX}"
read -r -a SEEDS <<<"${SEEDS:-42 17 7 31}"
TECHS="${TECHS:-tsmc7 tsmc5 tsmc12 tsmc16}"
TECHS_CSV="$(echo "$TECHS" | tr ' ' ',')"
OUTDIR="$ROOT/results/v6_4_7"
export NGSPICE_BIN="${NGSPICE_BIN:-$ROOT/tools/ngspice-45.2/bin/ngspice}"
export PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}"
PY=/data1/shenshan/.conda/envs/pycircuitsim/bin/python

# 1. force_ic + probe (CPU), one job per seed, ≤3 concurrent.
echo "[s11score] force_ic gates for ${PREFIX} seeds ${SEEDS[*]}"
fic_pids=()
for seed in "${SEEDS[@]}"; do
    ( CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        "$PY" scripts/v6_4_7_s11_sram_gate.py \
          --prefix "$PREFIX" --seed "$seed" \
          --techs "$(echo "$TECHS_CSV" | tr 'a-z' 'A-Z')" \
          --out "$OUTDIR/s11_force_ic_${PREFIX}_s${seed}.json" \
          > "$OUTDIR/s11_train_logs/force_ic_${PREFIX}_s${seed}.log" 2>&1 ) &
    fic_pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
done

# 2. 16-cell + inverter scorer (GPU, workers=1) via the s9b gate driver.
echo "[s11score] scorer (16-cell + inverter) for ${PREFIX}"
"$PY" scripts/v6_4_7_s9b_gate_controlv2.py \
    --prefix "$PREFIX" --seeds "$(echo "${SEEDS[*]}" | tr ' ' ',')" \
    --techs "$TECHS_CSV" \
    --out "$OUTDIR/S11_gate_${PREFIX}.md" || true

for pid in "${fic_pids[@]}"; do wait "$pid" || true; done
echo "[s11score] DONE. force_ic jsons + S11_gate_${PREFIX}.md written."
echo "=== force_ic summary ==="
for seed in "${SEEDS[@]}"; do
    f="$OUTDIR/s11_force_ic_${PREFIX}_s${seed}.json"
    [ -f "$f" ] && "$PY" -c "import json,sys; d=json.load(open('$f')); print('  s${seed}:', d['force_ic_n_pass'],'/',d['force_ic_n_total'],'railed', {t:[int(v.get('state1',0)),int(v.get('state0',0))] for t,v in d['force_ic'].items()})"
done
