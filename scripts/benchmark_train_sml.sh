#!/usr/bin/env bash
# Benchmark Phase C — train the 24 DirectNet capacity checkpoints.
#
# small/medium/large x {tsmc5,tsmc7,tsmc12,tsmc16} x {nmos,pmos}, all on ONE
# identical clean stock recipe (the project's "control-v2" recipe) so capacity
# is the only variable:
#     --apply-filter off --swa-mode ema --seed 42  (no loss preset / ekv / sobolev)
# Save names auto-resolve to tsmc{X}_dn_{size}_{dev}_best.pt  (the parser slots).
#
# 24 jobs distributed round-robin across the 3 GPUs as 3 sequential streams.
# Resumable: skips a job whose _best.pt already exists unless $1 == --force.
#
# Usage: bash scripts/benchmark_train_sml.sh [--force]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
DS="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/benchmark_sml/train_logs"
mkdir -p "$LOGDIR"
FORCE="${1:-}"
NGPU=3

# The `bsimar` package lives under external_compact_models/ (not on sys.path);
# it bootstraps pycmg internally. Mirror scripts/train_per_tech_8cells.sh.
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

techs=(tsmc5 tsmc7 tsmc12 tsmc16)
sizes=(small medium large)
devs=(nmos pmos)

# --- guard: all 8 datasets must be present ---
missing=0
for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  [ -f "$DS/${tech}_${dev}.npz" ] || { echo "[train] MISSING dataset $DS/${tech}_${dev}.npz"; missing=1; }
done; done
[ "$missing" -eq 0 ] || { echo "[train] ABORT: datasets incomplete"; exit 1; }

# --- ordered job list: size-major (small first for fast feedback) ---
jobs=()
for size in "${sizes[@]}"; do for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  jobs+=("$tech:$size:$dev")
done; done; done

run_job () {
  local spec="$1" gpu="$2"
  IFS=: read -r tech size dev <<< "$spec"
  local name="${tech}_dn_${size}_${dev}"
  local ckpt="$CKPT/${name}_best.pt"
  local log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ "$FORCE" != "--force" ]; then
    echo "[train] SKIP existing $name"; return 0
  fi
  echo "[train] START $name on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size "$size" --device-type "$dev" --tech-scope "$tech" \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite \
    > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ]; then echo "[train] DONE $name"; else echo "[train] FAIL $name (rc=$rc, see $log)"; fi
  return 0
}

stream () { local gpu="$1"; shift; for spec in "$@"; do run_job "$spec" "$gpu"; done; echo "[train] GPU$gpu stream complete"; }

declare -a g0 g1 g2; i=0
for spec in "${jobs[@]}"; do
  case $((i % NGPU)) in 0) g0+=("$spec");; 1) g1+=("$spec");; 2) g2+=("$spec");; esac
  i=$((i+1))
done

stream 0 "${g0[@]}" &
stream 1 "${g1[@]}" &
stream 2 "${g2[@]}" &
wait
echo "[train] ALL TRAINING COMPLETE"
