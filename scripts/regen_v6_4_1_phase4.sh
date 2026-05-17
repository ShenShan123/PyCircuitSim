#!/usr/bin/env bash
# V6.4.1 Phase 4 data regeneration driver.
#
# Regenerates per-tech DirectNet training data for TSMC5/7/12/16 with
# the Phase-4 overlays enabled (plan §4a-4e):
#   --overlays inv_trip,diff_pair_sat,ring_osc_trip,bistable_static,
#              switched_cap_offstate,vbs_lhs
#   --sinh-sampling           (§4b curvature-aware bulk grid)
#   --extra-nfin 6,12         (§4d interpolation NFINs; trimmed from the
#                              spec {4,6,8,12} to stay inside the ~1.5x
#                              total-volume budget — {6,8,12} measured
#                              1.62x on TSMC5, {6,12} lands ~1.42x. 6 and
#                              12 are the interpolation midpoints of the
#                              densest PDK NFIN bins {5,10} and {10,15}.)
#
# Each tech runs both devices. Output overwrites the canonical
# tsmcX_{nmos,pmos}.npz consumed by --tech-scope training. The v6.4.1
# datasets are backed up at /tmp/v6_4_1_phase4_backup/datasets/.
#
# Logs: logs/v6_4_1_phase4/regen_<tech>.log

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PYCMG="${ROOT}/external_compact_models/PyCMG"
LOGDIR="${ROOT}/logs/v6_4_1_phase4"
mkdir -p "${LOGDIR}"

OVERLAYS="inv_trip,diff_pair_sat,ring_osc_trip,bistable_static,switched_cap_offstate,vbs_lhs"
TECHS=("${@:-tsmc7 tsmc12 tsmc16}")

for tech in ${TECHS[@]}; do
  log="${LOGDIR}/regen_${tech}.log"
  echo "$(date '+%F %T') regen ${tech} -> ${log}"
  conda run --no-capture-output -n pycircuitsim \
    python "${PYCMG}/scripts/generate_nn_data.py" \
    --device both --tech "${tech}" \
    --overlays "${OVERLAYS}" --sinh-sampling --extra-nfin 6,12 \
    --n-workers 8 > "${log}" 2>&1
  echo "$(date '+%F %T') regen ${tech} done rc=$?"
done
echo "$(date '+%F %T') all regen done"
