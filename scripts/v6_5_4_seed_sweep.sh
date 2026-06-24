#!/usr/bin/env bash
# V6.5.4 Stage B — per-tech seed sweep at `large` for the value-surface-fragile
# techs (tsmc5 ring, tsmc7/tsmc16 opamp). Same clean recipe as the base matrix;
# only the seed varies. Saves tsmc{X}_dn_lgs{seed}_{dev} (override-safe, correct
# local scope). tsmc12 is already 4/4 at large (seed 42) so it is excluded.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
LOGDIR="$ROOT/results/v6_5_4_retrain/seed_logs"
mkdir -p "$LOGDIR"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
NGPU=3
NSTREAMS="${NSTREAMS:-9}"

read -r -a techs <<< "${TECHS:-tsmc5 tsmc7 tsmc16}"
read -r -a seeds <<< "${SEEDS:-7 17 31}"
read -r -a sizes <<< "${SIZES:-large}"
read -r -a devs  <<< "${DEVS:-nmos pmos}"

if [ "${1:-}" = "_one" ]; then
  tech="$2"; size="$3"; dev="$4"; seed="$5"; gpu="$6"
  tag="$([ "$size" = large ] && echo lg || echo "$size")s${seed}"
  name="${tech}_dn_${tag}_${dev}"
  log="$LOGDIR/${name}.log"
  echo "[seed] START $name (seed=$seed) on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size "$size" --device-type "$dev" --tech-scope "$tech" \
    --apply-filter off --swa-mode ema --seed "$seed" \
    --exp-name "${tech}_dn_${tag}" --cuda --overwrite > "$log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$CKPT/${name}_best.pt" ] && echo "[seed] DONE $name" || echo "[seed] FAIL $name (rc=$rc)"
  exit 0
fi

jobs=()
for tech in "${techs[@]}"; do for size in "${sizes[@]}"; do for seed in "${seeds[@]}"; do for dev in "${devs[@]}"; do
  jobs+=("$tech $size $dev $seed")
done; done; done; done
echo "[seed] launching ${#jobs[@]} jobs, $NSTREAMS concurrent across $NGPU GPUs"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
i=0; running=0
for spec in "${jobs[@]}"; do
  read -r t s d sd <<< "$spec"
  gpu=$((i % NGPU))
  bash "$SELF" _one "$t" "$s" "$d" "$sd" "$gpu" &
  i=$((i+1)); running=$((running+1))
  if [ "$running" -ge "$NSTREAMS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "[seed] ALL SEED-SWEEP TRAINING COMPLETE"
