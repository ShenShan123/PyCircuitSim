#!/bin/bash
# B5 OSDI Jacobian distillation training: 6 jobs (3 lambdas x 2 device types)
# Runs on GPU 1 sequentially. Logs go to results/v6_4_5_track_b/B5_logs/
set -e

LOGDIR=/data2/home/shenshan/NN_SPICE-trackb/results/v6_4_5_track_b/B5_logs
mkdir -p "$LOGDIR"

cd /data2/home/shenshan/NN_SPICE-trackb

for LAM in 0.001 0.01 0.1; do
    for DT in nmos pmos; do
        LOGFILE="$LOGDIR/lam${LAM}_${DT}.log"
        echo "[$(date)] Starting lambda=${LAM} ${DT}" | tee -a "$LOGFILE"
        CUDA_VISIBLE_DEVICES=1 conda run -n pycircuitsim python \
            experiments/v6_4_5_track_b/B5_osdi_jacobian_distill.py \
            --lambda-j "$LAM" \
            --device-type "$DT" \
            --tech-scope tsmc7 \
            --epochs 200 \
            --patience 40 \
            --lr 5e-4 \
            --batch-size 2048 \
            --overwrite \
            2>&1 | tee -a "$LOGFILE"
        echo "[$(date)] Done lambda=${LAM} ${DT}" | tee -a "$LOGFILE"
    done
done

echo "All 6 B5 training jobs complete."
