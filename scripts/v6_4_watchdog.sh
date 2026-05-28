#!/usr/bin/env bash
# V6.4 best-of-N watchdog. Periodically scans training logs; any cell
# that failed (OOM / non-zero exit) AND has no checkpoint on disk is
# re-appended to the worklist so a free worker retries it. Exits when
# the worklist is empty and no train procs remain.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOGDIR="${ROOT}/logs/v6_4_bestof"
WORKLIST="${LOGDIR}/_worklist.txt"
LOCK="${LOGDIR}/_worklist.lock"
CKPT="${ROOT}/external_compact_models/bsimar/checkpoints"
WLOG="${LOGDIR}/_watchdog.log"
declare -A RETRIED

while true; do
  for f in "${LOGDIR}"/*_s*.log; do
    [[ -e "$f" ]] || continue
    b="$(basename "$f" .log)"            # <tech>_<dev>_s<seed>
    tech="${b%%_*}"; rest="${b#*_}"
    dev="${rest%%_*}"; seed="${rest##*_s}"
    out="${CKPT}/v6_4_bof_${tech}_s${seed}_${dev}_best.pt"
    [[ -f "${out}" ]] && continue        # succeeded
    grep -qE "CUDA error|out of memory|Traceback|RuntimeError" "$f" 2>/dev/null \
      || continue
    # still being written to? skip if modified in last 90s.
    [[ -n "$(find "$f" -mmin -1.5)" ]] && continue
    key="${tech}_${dev}_${seed}"
    [[ -n "${RETRIED[$key]:-}" ]] && continue
    RETRIED[$key]=1
    flock "${LOCK}" bash -c "echo '${tech} ${dev} ${seed}' >> '${WORKLIST}'"
    echo "$(date '+%F %T') requeued ${tech} ${dev} s${seed}" >> "${WLOG}"
  done
  remaining=$(wc -l < "${WORKLIST}" 2>/dev/null || echo 0)
  nproc_train=$(pgrep -fc "bsimar.cli.train" || echo 0)
  if [[ "${remaining}" -eq 0 && "${nproc_train}" -eq 0 ]]; then
    echo "$(date '+%F %T') watchdog: all done, exiting" >> "${WLOG}"
    break
  fi
  sleep 120
done
