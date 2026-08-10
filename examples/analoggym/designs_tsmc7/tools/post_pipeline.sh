#!/bin/sh
# Second pass after pipeline.sh: re-tune anything still failing (including
# designs the first pass could not run), refresh RESULTS.md, and produce the
# tree's official run_all summary.
set -e
PY=${PY:-/data2/home/shenshan/.conda/envs/analoggym-env/bin/python}
cd "$(dirname "$0")/.."
echo "== retune pass 2 ($(basename "$PWD"))"
AG_EVALS_SCALE=${AG_EVALS_SCALE:-2} $PY tools/retune.py
echo "== report"
$PY tools/report.py
echo "== run_all (summary.csv)"
$PY run_all.py
echo "== post done $(basename "$PWD")"
