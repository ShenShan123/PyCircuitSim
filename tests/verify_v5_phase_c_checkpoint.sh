#!/usr/bin/env bash
#
# V5 Phase C verification driver.
#
# For a given v5 NMOS prefix and PMOS prefix, runs verify_nn_dc_tran on
# the post-Phase-A simulator with PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE
# set to a wrapper-prefix that the resolver maps to one polarity per
# device.  The override prefix is shared between LEVEL=73 and LEVEL=74,
# so this script forces both DirectNet *and* BSIMAR to look at the same
# v5_*_<polarity> name.  In practice we run *paired* DN+TF runs with
# matching exp-names so that's what we want.
#
# Usage:
#   ./tests/verify_v5_phase_c_checkpoint.sh <nmos_prefix> <pmos_prefix> <out_csv>
#
# Example:
#   ./tests/verify_v5_phase_c_checkpoint.sh \
#       v5_dn_s_nmos_mae v5_dn_s_pmos_mae \
#       /tmp/v5_phase_c_v5_dn_s_mae_summary.csv
#
# Strategy: the env-var override is one prefix.  We need *different*
# prefixes for NMOS and PMOS (they're separate trainings).  Solution:
# the resolver looks at the device_key to decide whether to use the
# override base.  We pass a polarity-agnostic base, but the script
# must be invoked once per device polarity.  Since verify_nn_dc_tran
# calls both NMOS and PMOS in a single Python invocation, we can't
# swap env vars mid-run.  Workaround: split-prefix env var that
# encodes both:
#
#   PYCIRCUITSIM_NN_CHECKPOINT_NMOS=v5_dn_s_nmos_mae
#   PYCIRCUITSIM_NN_CHECKPOINT_PMOS=v5_dn_s_pmos_mae
#
# The parser reads whichever matches device_key.  See parser.py.

set -euo pipefail

NMOS_PREFIX="${1:-}"
PMOS_PREFIX="${2:-}"
OUT_CSV="${3:-}"

if [[ -z "$NMOS_PREFIX" || -z "$PMOS_PREFIX" || -z "$OUT_CSV" ]]; then
    echo "usage: $0 <nmos_prefix> <pmos_prefix> <out_csv>" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# We can't swap one env var per device — both polarities are evaluated in a
# single python invocation.  Use a unified prefix that ends in either
# "_nmos" or "_pmos" — the resolver in parser.py will match the suffix and
# fall through for the other polarity.  So we set a *pair* of prefixes via
# a join env var the parser knows about.
export PYCIRCUITSIM_NN_CHECKPOINT_NMOS="$NMOS_PREFIX"
export PYCIRCUITSIM_NN_CHECKPOINT_PMOS="$PMOS_PREFIX"

echo "[verify_v5_phase_c] NMOS=$NMOS_PREFIX, PMOS=$PMOS_PREFIX"
echo "[verify_v5_phase_c] Saving to $OUT_CSV"

conda run -n pycircuitsim python tests/verify_nn_dc_tran.py \
    --tech TSMC5,TSMC7,TSMC12,TSMC16

cp "$PROJECT_ROOT/tests/verify_nn_dc_tran_results/summary.csv" "$OUT_CSV"
echo "[verify_v5_phase_c] saved: $OUT_CSV"
