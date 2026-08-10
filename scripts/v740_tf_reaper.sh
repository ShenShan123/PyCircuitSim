#!/usr/bin/env bash
# V7.4.0 — reaper for BSIM-AR stems that died mid-training.
#
# Closes a hole the rescue/fill drivers cannot: `recipe_train.sh _one` SKIPS
# (exit 0) any stem whose `_best.pt` already exists unless it is given
# `--force`. The trainer writes `_best.pt` at every val improvement, so a job
# killed at epoch 250 leaves one behind — and every later recovery attempt then
# reports success without training anything. The stem never gains `.complete`,
# `--require-complete` drops it from gating, and the tier ends up with a silent
# hole. The earlier OOMs all struck at epoch 0, before any `_best.pt` existed,
# which is exactly why this was not visible until the xl tier.
#
# A dead-incomplete stem is retrained with `--force`, i.e. FROM SCRATCH. That is
# deliberate: a clean-recipe checkpoint must be one uninterrupted run, so
# warm-starting off the partial file would silently turn the control into a
# curriculum. Losing epochs is the cheaper error.
#
# Confirmation is two-cycle — a stem must look dead twice in a row before it is
# reaped — so a process caught between exec and its first log write is never
# killed off by mistake.
#
# Usage: nohup bash scripts/v740_tf_reaper.sh &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CK="$ROOT/external_compact_models/bsimar/checkpoints"
GPU="${REAPER_GPU:-1}"
PARALLEL="${REAPER_PARALLEL:-2}"
INTERVAL="${REAPER_INTERVAL:-300}"
TECHS="${TECHS:-tsmc5 tsmc6 tsmc7 tsmc12 tsmc16}"
SIZES="${SIZES:-small medium large xl}"

say () { echo "[tf-reaper] $(date '+%F %T') $*"; }

is_running () {  # tech size dev
  ps -eo args 2>/dev/null | grep -q -- \
    "--size $2 --device-type $3 --tech-scope $1"
}

declare -A suspect=()

say "watching for dead-incomplete BSIM-AR stems (retrain GPU $GPU, ${INTERVAL}s cycle)"
while true; do
  dead=()
  for t in $TECHS; do for s in $SIZES; do for d in nmos pmos; do
    stem="${t}_tf_${s}_${d}"
    [ -f "$CK/${stem}_best.pt.complete" ] && { unset "suspect[$stem]"; continue; }
    if is_running "$t" "$s" "$d"; then unset "suspect[$stem]"; continue; fi
    # No marker and no process. Only reap what has a stale partial file — a
    # stem with nothing on disk was never started and belongs to the
    # dispatcher, not here.
    [ -f "$CK/${stem}_best.pt" ] || { unset "suspect[$stem]"; continue; }
    if [ -n "${suspect[$stem]:-}" ]; then
      dead+=("$t $s $d"); unset "suspect[$stem]"
    else
      suspect[$stem]=1
      say "SUSPECT $stem (partial checkpoint, no process) — confirming next cycle"
    fi
  done; done; done

  if [ "${#dead[@]}" -gt 0 ]; then
    say "REAPING ${#dead[@]} dead-incomplete stem(s), retraining from scratch with --force"
    for x in "${dead[@]}"; do say "  reap: $(echo "$x" | tr ' ' '_')"; done
    printf '%s\n' "${dead[@]}" \
      | while read -r t s d; do echo "clean $t $s $d $GPU --force"; done \
      | PYTORCH_ALLOC_CONF=expandable_segments:True MODEL=transformer TRAIN_OMP=4 \
        xargs -P "$PARALLEL" -L1 bash "$ROOT/scripts/recipe_train.sh" _one
    say "reap batch finished"
  fi

  n="$(ls "$CK"/*_tf_*_best.pt.complete 2>/dev/null | wc -l)"
  [ "$n" -ge 40 ] && { say "all 40 BSIM-AR checkpoints complete — reaper exiting"; break; }
  sleep "$INTERVAL"
done
