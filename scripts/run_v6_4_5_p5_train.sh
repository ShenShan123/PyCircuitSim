#!/bin/bash
# V6.4.5 Phase-5 TSMC7 retrain sweep.
# Plan §Phase 5: 16-seed stock + 8-seed mono, TSMC7 NMOS+PMOS.
# The 4 seeds {42,7,17,123} already exist on disk as the V6.4.2 Phase-7a
# artifacts (v6_4_2_p7_tsmc7_{stock,mono}_s*) and are reused in scoring; this
# script trains the REMAINING seeds fresh under the v6_4_5_p5 naming.
#   stock new seeds: 1 2 3 5 11 13 31 47 73 91 137 211   (12 -> 24 trainings)
#   mono  new seeds: 1 2 3 5                              (4  ->  8 trainings)
# Total 32 fresh trainings, scheduled round-robin across GPUs 1/2/3, bounded
# concurrency. Page cache for tsmc7_{nmos,pmos}.npz should be pre-warmed.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
LOGDIR="$ROOT/logs/v6_4_5_p5"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
mkdir -p "$LOGDIR"

GPUS=(1 2 3)
MAXCONC=6                       # ~2 per GPU; DirectNet-medium is ~1GB GPU each
STOCK_SEEDS=(1 2 3 5 11 13 31 47 73 91 137 211)
MONO_SEEDS=(1 2 3 5)

JOBS=()
for s in "${STOCK_SEEDS[@]}"; do for d in nmos pmos; do JOBS+=("stock $s $d"); done; done
for s in "${MONO_SEEDS[@]}";  do for d in nmos pmos; do JOBS+=("mono $s $d");  done; done

i=0
for job in "${JOBS[@]}"; do
  set -- $job; recipe=$1; seed=$2; dev=$3
  name="v6_4_5_p5_tsmc7_${recipe}_s${seed}"
  if [ -f "$CKPT/${name}_${dev}_best.pt" ]; then
    echo "skip (exists): ${name}_${dev}"; i=$((i+1)); continue
  fi
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  mono_flag=""; [ "$recipe" = mono ] && mono_flag="--monotonic"
  echo "launch gpu$gpu  ${name}_${dev}"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 conda run --no-capture-output -n pycircuitsim \
    python -u -m bsimar.cli.train --model direct --size medium --device-type "$dev" \
    --tech-scope tsmc7 --cuda --seed "$seed" --overwrite $mono_flag --exp-name "$name" \
    > "$LOGDIR/train_${recipe}_s${seed}_${dev}.log" 2>&1 &
  i=$((i+1))
  while [ "$(jobs -r | wc -l)" -ge "$MAXCONC" ]; do sleep 10; done
done
wait
echo "=== all Phase-5 trainings done ($((i)) jobs dispatched) ==="
ls "$CKPT"/v6_4_5_p5_tsmc7_*_best.pt 2>/dev/null | wc -l | xargs echo "v6_4_5_p5 checkpoints on disk:"
