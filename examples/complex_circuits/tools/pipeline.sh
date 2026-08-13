#!/bin/sh
# Full per-tech pipeline: port TSMC16 sizes -> measure -> re-tune failures ->
# regenerate RESULTS.md.
#
# Run from the per-tech tree you want built:
#     cd designs_tsmc7 && sh ../tools/pipeline.sh
# or name the tree explicitly from anywhere:
#     AG_TECH=tsmc7 sh tools/pipeline.sh
#
# Since V7.5.8 one shared tools/ serves every tech, so this script must NOT cd
# to its own parent the way it used to — that parent is now the tree-of-trees.
# The working directory (or AG_TECH / AG_TREE) is what selects the tech; see
# tools/pycmg_lib.py::_resolve_tree.
set -e
PY=${PY:-/data2/home/shenshan/.conda/envs/analoggym-env/bin/python}
TOOLS=$(cd "$(dirname "$0")" && pwd)
echo "== port_tech ($(basename "$PWD"))"
$PY "$TOOLS/port_tech.py"
echo "== finalize (measure ported designs)"
$PY "$TOOLS/finalize.py"
echo "== retune (designs with failing gates)"
$PY "$TOOLS/retune.py"
echo "== report"
$PY "$TOOLS/report.py"
echo "== done $(basename "$PWD")"
