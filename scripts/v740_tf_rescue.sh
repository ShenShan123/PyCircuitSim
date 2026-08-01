#!/usr/bin/env bash
# V7.4.0 — rescue BSIM-AR stems that OOM'd on the cramped GPU, in parallel with
# the still-running first wave.
#
# A stem that FAILED is abandoned by the running dispatcher — it is not retried
# — so claiming it here cannot race that dispatcher. Retraining it now, pinned
# to GPU 1, converts what would otherwise be a serial post-wave backlog into
# work that overlaps the wave.
#
# Polls the master log so stems that fail later are picked up too, and exits
# once the wave has terminated and nothing is left to rescue. `recipe_train.sh
# _one` skips any stem that already carries .complete, so a stem rescued here
# and later revisited costs nothing.
#
# Usage: nohup bash scripts/v740_tf_rescue.sh &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
MASTER="$ROOT/results/tf_train_master.log"
GPU="${RESCUE_GPU:-1}"
PARALLEL="${RESCUE_PARALLEL:-3}"

say () { echo "[tf-rescue] $(date '+%F %T') $*"; }

claimed="/tmp/v740_campaign/rescue_claimed.txt"
mkdir -p "$(dirname "$claimed")"; : > "$claimed"

while true; do
  # Stems the wave gave up on, minus ones already complete or already claimed.
  mapfile -t stems < <(grep -o "FAIL [a-z0-9_]*" "$MASTER" 2>/dev/null \
                       | awk '{print $2}' | sort -u)
  todo=()
  for s in "${stems[@]:-}"; do
    [ -n "$s" ] || continue
    [ -f "$CKPT/${s}_best.pt.complete" ] && continue
    grep -qx "$s" "$claimed" && continue
    todo+=("$s")
  done

  if [ "${#todo[@]}" -gt 0 ]; then
    say "rescuing ${#todo[@]} stem(s) on GPU $GPU: ${todo[*]}"
    for s in "${todo[@]}"; do echo "$s" >> "$claimed"; done
    # stem = {tech}_tf_{size}_{dev}; recipe_train.sh _one wants them split out.
    printf '%s\n' "${todo[@]}" \
      | sed -E 's/^([a-z0-9]+)_tf_([a-z]+)_([a-z]+)$/clean \1 \2 \3/' \
      | while read -r recipe tech size dev; do
          echo "$recipe $tech $size $dev $GPU"
        done \
      | PYTORCH_ALLOC_CONF=expandable_segments:True MODEL=transformer TRAIN_OMP=4 \
        xargs -P "$PARALLEL" -L1 bash "$ROOT/scripts/recipe_train.sh" _one
    say "rescue batch finished"
  fi

  if grep -qE "ALL TRAINING COMPLETE|INFRASTRUCTURE FAILURE" "$MASTER" 2>/dev/null \
     && [ "${#todo[@]}" -eq 0 ]; then
    say "wave terminated and nothing left to rescue"
    break
  fi
  sleep 120
done

n="$(ls "$CKPT"/*_tf_*_best.pt.complete 2>/dev/null | wc -l)"
say "rescue driver done — $n/40 BSIM-AR checkpoints complete"
