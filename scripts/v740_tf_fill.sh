#!/usr/bin/env bash
# V7.4.0 — BSIM-AR gap-fill, pinned to the roomy GPU.
#
# The first BSIM-AR wave was dispatched round-robin over GPUS='1 1 0'. GPU 0 is
# shared with a foreign 30.5 GB process, leaving ~9.5 GB — enough for the
# transformer `small`/`medium` tiers but not for several `large`/`xl` jobs at
# once, so those OOM. They OOM at epoch 0 (memory is claimed up front), which
# is why the first wave is left alone to finish rather than restarted: a failed
# job costs seconds, and killing the wave would throw away in-flight jobs that
# were already ~75 epochs deep.
#
# This driver waits for that wave, then retrains whatever is missing with
# GPUS='1' so nothing lands on the cramped device, loops until it stops making
# progress, and finally gates + collects the cells its own run unblocked (the
# main orchestrator may have already exited by then, since a wave that ends in
# INFRASTRUCTURE FAILURE still counts as terminated).
#
# Idempotent and resumable: training skips any stem carrying .complete, and
# v710_regate.sh skips any gate job whose log carries ===V710_DONE.
#
# Usage: nohup bash scripts/v740_tf_fill.sh &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
OUT="$ROOT/results/v740_regate"
SC="${V740_SCRATCH:-/tmp/v740_campaign}"
PAR="${PAR:-32}"
PY="${NN_PY:-/data2/home/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
mkdir -p "$SC"

say () { echo "[tf-fill] $(date '+%F %T') $*"; }

n_complete () { ls "$CKPT"/*_tf_*_best.pt.complete 2>/dev/null | wc -l; }

# ---- 1. wait for the first wave to terminate --------------------------------
say "waiting for the first BSIM-AR wave to terminate"
# Both families run the same dispatcher script, so a pgrep cannot tell them
# apart; the master log's terminal line is the authoritative signal.
while ! grep -qE "ALL TRAINING COMPLETE|INFRASTRUCTURE FAILURE" \
        "$ROOT/results/tf_train_master.log" 2>/dev/null; do sleep 60; done
say "first wave terminated with $(n_complete)/40 complete checkpoints"

# ---- 2. refill rounds, pinned to GPU 1 --------------------------------------
# expandable_segments trims fragmentation, which is what turns a marginal
# allocation into an OOM on a device this contended.
for r in $(seq 1 "$MAX_ROUNDS"); do
  before="$(n_complete)"
  [ "$before" -ge 40 ] && { say "all 40 BSIM-AR checkpoints complete"; break; }
  say "refill round $r: $before/40 complete, retraining the gaps on GPU 1"
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  MODEL=transformer RECIPES=clean \
  TECHS='tsmc5 tsmc6 tsmc7 tsmc12 tsmc16' SIZES='small medium large xl' \
  DEVS='nmos pmos' GPUS='1' NSTREAMS="${NSTREAMS:-6}" TRAIN_OMP=4 \
    bash "$ROOT/scripts/recipe_train.sh" >> "$ROOT/results/tf_fill_train.log" 2>&1 \
    || say "round $r: dispatcher reported failures (continuing)"
  after="$(n_complete)"
  say "refill round $r done: $before -> $after complete"
  # A round that trains nothing new is not going to succeed by repeating.
  [ "$after" -gt "$before" ] || { say "round $r made no progress — stopping refill"; break; }
done

# A stem with _best.pt but no .complete is a killed/failed run; the gate
# emitter's --require-complete already excludes it, but say so loudly.
for f in "$CKPT"/*_tf_*_best.pt; do
  [ -e "$f" ] || continue
  [ -f "$f.complete" ] || say "WARNING incomplete checkpoint excluded from gating: $(basename "$f")"
done

# ---- 3. gate + collect whatever this run unblocked --------------------------
jobs="$SC/jobs_tf_fill.txt"
"$PY" "$ROOT/scripts/v730_coverage.py" --set clean --tag tf \
      --require-complete --emit-jobs "$jobs" | tail -3
n="$(grep -cve '^\s*$' "$jobs" || true)"
if [ "$n" -gt 0 ]; then
  say "dispatching $n BSIM-AR gate jobs at PAR=$PAR"
  NN_PY="$PY" V710_OUT="$OUT" V710_SCRATCH="$SC/scratch" PAR="$PAR" JOBS="$jobs" \
    bash "$ROOT/scripts/v710_regate.sh" || say "gate pool returned nonzero"
else
  say "no BSIM-AR gate cells outstanding"
fi

"$PY" "$ROOT/scripts/v710_regate_collect.py" --root "$OUT" || say "collector FAILED"
say "tf-fill complete — $(n_complete)/40 checkpoints, evidence collected"
