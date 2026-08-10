#!/bin/sh
# Full per-tech pipeline: port TSMC16 sizes -> measure -> re-tune failures ->
# regenerate RESULTS.md.  Run from a per-tech tree root.
set -e
PY=${PY:-/data2/home/shenshan/.conda/envs/analoggym-env/bin/python}
cd "$(dirname "$0")/.."
echo "== port_tech ($(basename "$PWD"))"
$PY tools/port_tech.py
echo "== derive three-output reference"
$PY ../derive_three_vref.py .
echo "== finalize (measure ported designs)"
$PY tools/finalize.py
echo "== retune (designs with failing gates)"
$PY tools/retune.py
echo "== report"
$PY tools/report.py
echo "== done $(basename "$PWD")"
