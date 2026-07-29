#!/usr/bin/env bash
# V7.1.0 — gate the PFN xl checkpoints once training finishes.
#
# PFN had no xl tier until V7.1.0; this closes the 4-scale matrix for the third
# family. Waits for all 8 `.complete` markers (4 techs x nmos/pmos), then runs
# the 4-cell complex matrix + the 4 device suites at OMP=1 plus OMP{2,4} on
# opamp and ring_osc, through the same V7.1.0 driver every other number uses.
#
# Usage:  PAR=10 bash scripts/pfnxl_gate_campaign.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
JOBS="${JOBS:-/tmp/pfnxl_gate_jobs.txt}"
PAR="${PAR:-10}"

echo "[pfnxl-gate] waiting for the 8 xl checkpoints … $(date '+%F %T')"
while :; do
  missing=0
  for t in tsmc5 tsmc7 tsmc12 tsmc16; do for d in nmos pmos; do
    [ -f "$CKPT/${t}_pfn_xl_${d}_best.pt.complete" ] || missing=1
  done; done
  [ "$missing" -eq 0 ] && break
  sleep 600
done

: > "$JOBS"
for t in TSMC5 TSMC7 TSMC12 TSMC16; do
  for s in verify_nn_ac verify_complex_opamp_ac verify_nn_multi_tech_dc \
           verify_nn_multi_tech_tran verify_complex_sram_snm verify_complex_switchcap; do
    echo "pfn xl $t $s 1" >> "$JOBS"
  done
  for s in verify_complex_opamp verify_complex_ring_osc; do for o in 1 2 4; do
    echo "pfn xl $t $s $o" >> "$JOBS"; done; done
done
echo "[pfnxl-gate] $(grep -c . "$JOBS") jobs $(date '+%F %T')"
PAR="$PAR" JOBS="$JOBS" bash "$ROOT/scripts/v710_regate.sh"
python "$ROOT/scripts/v710_regate_collect.py" || true
python "$ROOT/scripts/v730_docs_build.py" || true
echo "[pfnxl-gate] DONE $(date '+%F %T') — check the PFN clean report's xl row"
