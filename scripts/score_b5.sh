#!/bin/bash
# Score all 3 B5 lambda candidates and save JSON results.
# Run with CPU only (deterministic).
set -e

LOGDIR=/data2/home/shenshan/NN_SPICE-trackb/results/v6_4_5_track_b/B5_logs
mkdir -p "$LOGDIR"

cd /data2/home/shenshan/NN_SPICE-trackb

for LAM_FLOAT in 0.001 0.01 0.1; do
    # Build the lam tag matching the save_prefix in train_b5
    # Python generates: f"lam{lambda_j:.3f}".replace(".", "p")
    # 0.001 -> lam0p001, 0.01 -> lam0p010, 0.1 -> lam0p100
    LAM_TAG=$(python3 -c "l=${LAM_FLOAT}; print(f'lam{l:.3f}'.replace('.','p'))")
    LAM="${LAM_FLOAT}"
    NMOS_STEM="b5_jd_${LAM_TAG}_tsmc7_nmos"
    PMOS_STEM="b5_jd_${LAM_TAG}_tsmc7_pmos"

    # Check candidates exist
    CKPT_DIR=external_compact_models/bsimar/checkpoints
    if [ ! -f "${CKPT_DIR}/${NMOS_STEM}_best.pt" ] || [ ! -f "${CKPT_DIR}/${PMOS_STEM}_best.pt" ]; then
        echo "[SKIP] Missing checkpoint for lam=${LAM}: ${NMOS_STEM} or ${PMOS_STEM}"
        continue
    fi

    OUTFILE="$LOGDIR/score_${LAM_TAG}.json"
    echo "[$(date)] Scoring lam=${LAM} (tag=${LAM_TAG})..."
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" conda run -n pycircuitsim python \
        scripts/eval_v6_4_5_candidate.py \
        --tech TSMC7 \
        --nmos "$NMOS_STEM" \
        --pmos "$PMOS_STEM" \
        --json \
        2>&1 | tee "${LOGDIR}/score_${LAM_TAG}.log" | grep "^RESULT " | sed 's/^RESULT //' > "$OUTFILE"
    echo "[$(date)] Done lam=${LAM} → $OUTFILE"
done

echo "All scoring complete."
