#!/bin/bash
# Score the 16 NEW Phase-5 diagonal candidates (same-seed N&P) under the
# multi-circuit vector. Waits for the training batch driver to exit, then runs
# the scorer 8-way. Appends plain RESULT JSON (recipe/seed are encoded in the
# nmos/pmos stem names).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/v6_4_5/phase3_logs/search_TSMC7_p5.jsonl"
DRIVER_PID="${1:-}"
: > "$OUT"

if [ -n "$DRIVER_PID" ]; then
  while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 20; done
fi
echo "batch ckpts: $(ls "$ROOT"/external_compact_models/bsimar/checkpoints/v6_4_5_p5_tsmc7_*_best.pt 2>/dev/null | wc -l)/32"

score() {
  local recipe=$1 seed=$2
  local r
  r=$(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n pycircuitsim \
        python "$ROOT/scripts/eval_v6_4_5_candidate.py" --tech TSMC7 \
        --nmos "v6_4_5_p5_tsmc7_${recipe}_s${seed}_nmos" \
        --pmos "v6_4_5_p5_tsmc7_${recipe}_s${seed}_pmos" --json 2>/dev/null \
      | grep '^RESULT ' | sed 's/^RESULT //')
  if [ -n "$r" ]; then echo "$r" >> "$OUT"; echo "done ${recipe}_s${seed}"; \
  else echo "FAIL ${recipe}_s${seed}"; fi
}
export -f score; export ROOT OUT

( for s in 1 2 3 5 11 13 31 47 73 91 137 211; do echo stock $s; done; \
  for s in 1 2 3 5; do echo mono $s; done ) \
  | xargs -P 8 -n 2 bash -c 'score "$@"' _

echo "=== scored $(wc -l < "$OUT") / 16 candidates ==="
