#!/usr/bin/env bash
# Benchmark Phase C — train the 24 DirectNet capacity checkpoints.
#
# small/medium/large x {tsmc5,tsmc7,tsmc12,tsmc16} x {nmos,pmos}, all on ONE
# identical clean stock recipe (the project's "control-v2" recipe) so capacity
# is the only variable:
#     --apply-filter off --swa-mode ema --seed 42  (no loss preset / ekv / sobolev)
# Save names auto-resolve to tsmc{X}_dn_{size}_{dev}_best.pt  (the parser slots).
#
# The 4090s sit near-idle on these tiny nets (GPU-light, data-loader bound) and
# the box has 192 cores, so we run NSTREAMS jobs concurrently, pinned round-robin
# across the 3 GPUs (job index % NGPU). Resumable: skips a job whose _best.pt
# exists unless $1 == --force.
#
# Usage: NSTREAMS=9 bash scripts/benchmark_train_sml.sh [--force]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
DS="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/benchmark_sml/train_logs"
mkdir -p "$LOGDIR"
NGPU=3
NSTREAMS="${NSTREAMS:-9}"

# The `bsimar` package lives under external_compact_models/ (not on sys.path);
# it bootstraps pycmg internally. Mirror scripts/train_per_tech_8cells.sh.
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

# ---- single-job worker: SELF _one <tech> <size> <dev> <gpu> <force> ----
if [ "${1:-}" = "_one" ]; then
  tech="$2"; size="$3"; dev="$4"; gpu="$5"; force="${6:-}"
  name="${tech}_dn_${size}_${dev}"
  ckpt="$CKPT/${name}_best.pt"
  log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ "$force" != "--force" ]; then echo "[train] SKIP existing $name"; exit 0; fi
  echo "[train] START $name on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size "$size" --device-type "$dev" --tech-scope "$tech" \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite \
    > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then echo "[train] DONE $name"; else echo "[train] FAIL $name (rc=$rc, see $log)"; fi
  exit 0
fi

# ---- dispatcher ----
FORCE="${1:-}"
techs=(tsmc5 tsmc7 tsmc12 tsmc16)
sizes=(small medium large)
devs=(nmos pmos)

missing=0
for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  [ -f "$DS/${tech}_${dev}.npz" ] || { echo "[train] MISSING dataset $DS/${tech}_${dev}.npz"; missing=1; }
done; done
[ "$missing" -eq 0 ] || { echo "[train] ABORT: datasets incomplete"; exit 1; }

# size-major ordering (small first for fast feedback); gpu = index % NGPU
lines=(); i=0
for size in "${sizes[@]}"; do for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  lines+=("$tech $size $dev $((i % NGPU)) $FORCE")
  i=$((i+1))
done; done; done

echo "[train] launching ${#lines[@]} jobs, $NSTREAMS concurrent across $NGPU GPUs"
printf '%s\n' "${lines[@]}" | xargs -P "$NSTREAMS" -L1 "$SELF" _one
echo "[train] ALL TRAINING COMPLETE"
