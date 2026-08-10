#!/usr/bin/env bash
# V7.4.0 clean-rebuild campaign driver.
#
# Waits for the two training waves to finish, then gates every clean cell that
# has a COMPLETED checkpoint and collects the evidence. Safe to launch while
# training is still running, and safe to re-launch: every stage is resumable
# (v710_regate.sh skips a job whose log carries ===V710_DONE, and the coverage
# emitter only ever emits cells that are missing AND have a checkpoint).
#
# Why one driver rather than eight hand-run pools: the campaign spans days and
# each tier unblocks at a different time. Polling here keeps the box saturated
# without a human deciding when each wave is ready.
#
# Gates are CPU-pinned by v710_regate.sh per the methodology — training may be
# on GPU, evaluation never is.
#
# Usage: PAR=32 bash scripts/v740_campaign.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# V7.4.2: overridable so a later pass writes its own evidence dir
# instead of backfilling the published V7.4.0 one. Defaults keep
# the V7.4.0 invocation byte-identical.
OUT="${V740_OUT:-$ROOT/results/v740_regate}"
SC="${V740_SCRATCH:-/tmp/v740_campaign}"
DN_LOG="${V740_DN_LOG:-$ROOT/results/dn_train_master.log}"
TF_LOG="${V740_TF_LOG:-$ROOT/results/tf_train_master.log}"
PAR="${PAR:-32}"
PY="${NN_PY:-/data2/home/shenshan/.conda/envs/pycircuitsim/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"
mkdir -p "$OUT" "$SC"

say () { echo "[v740] $(date '+%F %T') $*"; }

# ---- 1+2. drain loop: gate each tier the moment its training finishes --------
# Waiting for BOTH waves before gating anything would idle the CPU for the
# many hours BSIM-AR trains after DirectNet is done — the gates are CPU-only
# and the training is GPU-bound, so they belong in flight together. Each round
# gates whatever has newly acquired a COMPLETED checkpoint and then re-checks.
#
# --require-complete: a bare _best.pt can be a killed run (the trainer writes
# one at every val improvement); the marker is the discipline. `dn` first each
# round — its cells are ~40x cheaper than BSIM-AR's, so the DirectNet report
# unblocks while BSIM-AR is still grinding.
trained () {  # master-log -> 0 once the wave has terminated, either way
  grep -qE "ALL (RECIPE )?TRAINING COMPLETE|INFRASTRUCTURE FAILURE" "$1" 2>/dev/null
}

round=0
while true; do
  round=$((round + 1))
  dispatched=0
  for tag in dn tf; do
    jobs="$SC/jobs_${tag}.txt"
    "$PY" "$ROOT/scripts/v730_coverage.py" --set clean --tag "$tag" \
          --require-complete --emit-jobs "$jobs" ${V740_COVERAGE_ARGS:-} \
          >/dev/null 2>&1 || continue
    n="$(grep -cve '^\s*$' "$jobs" || true)"
    [ "$n" -gt 0 ] || continue
    dispatched=$((dispatched + n))
    say "round $round / $tag: dispatching $n gate jobs at PAR=$PAR"
    NN_PY="$PY" V710_OUT="$OUT" V710_SCRATCH="$SC/scratch" PAR="$PAR" JOBS="$jobs" \
      bash "$ROOT/scripts/v710_regate.sh" \
      || say "round $round / $tag: pool returned nonzero (no-verdict cells — see logs)"
    say "round $round / $tag: pool complete"
  done

  if trained "$DN_LOG" \
     && trained "$TF_LOG" \
     && [ "$dispatched" -eq 0 ]; then
    say "both training waves finished and no gateable cells remain"
    break
  fi
  # Nothing new to do yet — the next tier is still training. Idle briefly
  # rather than spinning the coverage scan.
  [ "$dispatched" -gt 0 ] || sleep 180
done

# ---- 3. collect ------------------------------------------------------------
"$PY" "$ROOT/scripts/v710_regate_collect.py" --root "$OUT" || say "collector FAILED"
say "campaign complete — data.json written to $OUT"
say "next: review coverage, then python scripts/v730_docs_build.py"
