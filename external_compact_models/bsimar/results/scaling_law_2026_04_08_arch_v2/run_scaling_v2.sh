#!/usr/bin/env bash
# Scaling-law re-run for BSIM-AR Transformer on universal_nmos
# with the NEW architecture (post bsimar-arch-experiments squash @99f8086):
#
#   - parallel_caps=True (P4)         hard-wired in trainer.py
#   - grouped_inputs=True (A2)        hard-wired in trainer.py
#   - norm-mode asinh (T2)            CLI flag
#   - T1 phys-best checkpoint selection in trainer.py
#
# Mirrors results/scaling_law_2026_04_08/run_scaling.sh:
#   Fixed: data, loss (mae+lds), epochs(50), batch(1024), lr(8e-4),
#          patience(50), seed(42).
#   Variable: d_model, nhead, num_layers, dim_feedforward.
#
# Param counts (verified vs old run for old arch; under A2 the
# tokenizer adds a tiny number of params from the 3 group MLPs but
# subtracts the 19-token positional / context cost. Net is approximately
# +/- 1% from the old counts):
#   small  : d=128, h=4, L=3, ff=256  ~  404K
#   medium : d=256, h=8, L=6, ff=1024 ~ 4.75M
#   large  : d=384, h=8, L=8, ff=1536 ~ 14.2M
#
# Each tier runs on its own GPU in parallel:
#   small  -> GPU 1 (A100 40GB)
#   medium -> GPU 2 (Blackwell 96GB)
#   large  -> GPU 3 (A100 40GB)

set -eux -o pipefail

export PYTHONPATH=/home/shenshan/NN_SPICE/external_compact_models:${PYTHONPATH:-}
cd /home/shenshan/NN_SPICE

RESDIR="/home/shenshan/NN_SPICE/external_compact_models/bsimar/results/scaling_law_2026_04_08_arch_v2"
CKPT="/home/shenshan/NN_SPICE/external_compact_models/bsimar/checkpoints"

# Clean any prior v2 checkpoints so the trainer's no-overwrite guard
# does not abort.
rm -f "$CKPT"/scaling_v2_small_nmos_*  \
      "$CKPT"/scaling_v2_medium_nmos_* \
      "$CKPT"/scaling_v2_large_nmos_*  || true

COMMON_FLAGS="--model transformer --device-type nmos \
  --universal --loss mae --lds --norm-mode asinh \
  --epochs 50 --batch-size 1024 --lr 8e-4 --patience 50 --seed 42 --cuda"

# Small
CUDA_VISIBLE_DEVICES=1 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train $COMMON_FLAGS \
  --d-model 128 --nhead 4 --num-layers 3 --dim-feedforward 256 \
  --exp-name scaling_v2_small \
  > "$RESDIR/small.log" 2>&1 &
PID_SMALL=$!

# Medium
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train $COMMON_FLAGS \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --exp-name scaling_v2_medium \
  > "$RESDIR/medium.log" 2>&1 &
PID_MEDIUM=$!

# Large
CUDA_VISIBLE_DEVICES=3 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train $COMMON_FLAGS \
  --d-model 384 --nhead 8 --num-layers 8 --dim-feedforward 1536 \
  --exp-name scaling_v2_large \
  > "$RESDIR/large.log" 2>&1 &
PID_LARGE=$!

echo "small  PID=$PID_SMALL  GPU=1"
echo "medium PID=$PID_MEDIUM GPU=2"
echo "large  PID=$PID_LARGE  GPU=3"

# Wait for all three. Use `wait` so the script returns the OR of all
# child exit codes.
RC=0
wait $PID_SMALL  || { echo "small  FAILED"; RC=1; }
wait $PID_MEDIUM || { echo "medium FAILED"; RC=1; }
wait $PID_LARGE  || { echo "large  FAILED"; RC=1; }

echo "=== ALL SCALING_V2 RUNS FINISHED (rc=$RC) ==="
exit $RC
