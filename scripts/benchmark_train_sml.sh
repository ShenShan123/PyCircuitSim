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
CKPT="$ROOT/external_compact_models/neural_network/checkpoints"
DS="$ROOT/external_compact_models/neural_network/data/datasets"
LOGDIR="$ROOT/results/benchmark_sml/train_logs"
mkdir -p "$LOGDIR"
# GPUS env (space-separated physical GPU ids) lets a run dodge a busy/shared GPU
# without editing this file; default = all three. NGPU derives from the list.
read -r -a GPU_IDS <<< "${GPUS:-0 1 2}"
NGPU=${#GPU_IDS[@]}
NSTREAMS="${NSTREAMS:-9}"

# The `neural_network` package lives under external_compact_models/;
# it bootstraps pycmg internally.
#
# audit B5a — this script also absorbed the deleted scripts/train_per_tech_8cells.sh
# (the 8-cell {tsmc5,tsmc7} x {small,medium} x {nmos,pmos} sweep). That one ran the
# CLI with `--overwrite` and NO `--exp-name`, so it wrote straight into the
# production tsmc{X}_dn_{size}_{dev} resolver slots, on the CLI defaults rather
# than the clean control-v2 recipe. Use instead:
#     TECHS='tsmc5 tsmc7' SIZES='small medium' bash scripts/benchmark_train_sml.sh
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

# ---- single-job worker: SELF _one <tech> <size> <dev> <gpu> <force> ----
if [ "${1:-}" = "_one" ]; then
  tech="$2"; size="$3"; dev="$4"; gpu="$5"; force="${6:-}"
  name="${tech}_dn_${size}_${dev}"
  ckpt="$CKPT/${name}_best.pt"
  norm="$CKPT/${name}_norm.npz"
  complete="$ckpt.complete"
  log="$LOGDIR/${name}.log"
  if [ -f "$ckpt" ] && [ -f "$norm" ] && [ -f "$complete" ] && [ "$force" != "--force" ]; then
    echo "[train] SKIP existing $name"; exit 0
  fi
  # Completion belongs to one exact successful overwrite.  Leaving the prior
  # marker in place while --overwrite writes best-so-far weights lets a killed
  # retrain masquerade as complete to downstream scored campaigns.
  rm -f "$complete"
  echo "[train] START $name on GPU$gpu"
  # EXTRA_TRAIN_ARGS lets a campaign inject a recipe addendum (e.g. the V6.5.1
  # µA-band loss lever: '--subthresh --subthresh-s2 1e-7 --subthresh-upper 3e-5')
  # without editing this script; empty by default = the clean control-v2 recipe.
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n pycircuitsim python -u -m neural_network.cli.train \
    --model direct --size "$size" --device-type "$dev" --tech-scope "$tech" \
    --apply-filter off --swa-mode ema --seed 42 --cuda --overwrite ${EXTRA_TRAIN_ARGS:-} \
    > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$ckpt" ] && [ -f "$norm" ]; then
    touch "$complete"
    echo "[train] DONE $name"
  else
    rm -f "$complete"
    # audit B3 — a failed train produces NO artifact, so there is nothing for a
    # downstream gate to judge; exit 3 (never 255: xargs aborts the run on 255)
    # so the dispatcher fails loudly instead of printing FAIL and returning 0.
    echo "[train] FAIL $name (rc=$rc, see $log)"
    exit 3
  fi
  exit 0
fi

# ---- dispatcher ----
FORCE="${1:-}"
# TECHS / SIZES / DEVS env-overridable so a campaign can train a subset
# (e.g. SIZES=xl, or TECHS='tsmc5 tsmc12' for the µA-lever A/B).
read -r -a techs <<< "${TECHS:-tsmc5 tsmc7 tsmc12 tsmc16}"
read -r -a sizes <<< "${SIZES:-small medium large xl}"
read -r -a devs  <<< "${DEVS:-nmos pmos}"

missing=0
for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  [ -f "$DS/${tech}_${dev}.npz" ] || { echo "[train] MISSING dataset $DS/${tech}_${dev}.npz"; missing=1; }
done; done
[ "$missing" -eq 0 ] || { echo "[train] ABORT: datasets incomplete"; exit 1; }

# size-major ordering (small first for fast feedback); gpu = index % NGPU.
# The last field is ALWAYS non-empty (`noforce` placeholder when --force is
# absent): an empty trailing field leaves a trailing blank, and `xargs -L1`
# treats a trailing blank as a line continuation — silently joining every job
# into ONE command so only the first runs. (Bit us on the no-force XL relaunch.)
lines=(); i=0
for size in "${sizes[@]}"; do for tech in "${techs[@]}"; do for dev in "${devs[@]}"; do
  lines+=("$tech $size $dev ${GPU_IDS[$((i % NGPU))]} ${FORCE:-noforce}")
  i=$((i+1))
done; done; done

echo "[train] launching ${#lines[@]} jobs, $NSTREAMS concurrent across $NGPU GPUs"
# audit B3 — capture xargs BEFORE the trailing echo, which would otherwise
# overwrite $? with 0 and report a clean sweep over a wave of failed trainings.
printf '%s\n' "${lines[@]}" | xargs -P "$NSTREAMS" -L1 "$SELF" _one
xrc=$?
if [ "$xrc" -ne 0 ]; then
  echo "[train] INFRASTRUCTURE FAILURE: some jobs produced no checkpoint (xargs rc=$xrc, see $LOGDIR)" >&2
  exit 1
fi
echo "[train] ALL TRAINING COMPLETE"
