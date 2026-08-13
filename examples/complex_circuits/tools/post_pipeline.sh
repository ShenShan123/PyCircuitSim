#!/bin/sh
# Second pass after pipeline.sh: re-tune anything still failing (including
# designs the first pass could not run), refresh RESULTS.md, and produce the
# tree's official run_all summary.
#
# Run from the per-tech tree, or name it with AG_TECH — see tools/pipeline.sh.
set -e
PY=${PY:-/data2/home/shenshan/.conda/envs/analoggym-env/bin/python}
TOOLS=$(cd "$(dirname "$0")" && pwd)
echo "== retune pass 2 ($(basename "$PWD"))"
AG_EVALS_SCALE=${AG_EVALS_SCALE:-2} $PY "$TOOLS/retune.py"
echo "== report"
$PY "$TOOLS/report.py"
echo "== run_all (summary.csv)"
$PY "$TOOLS/../run_all.py"
echo "== post done $(basename "$PWD")"
