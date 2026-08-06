#!/usr/bin/env bash
# V7.1.0 — gate the restored TSMC6 checkpoints, family by family as they finish.
#
# Companion to scripts/tsmc6_restore_campaign.sh (which trains them). Reuses the
# V7.1.0 re-gate driver, so isolation, CPU pinning, resume and the verdict
# convention are identical to every other number in docs/accuracy/.
#
# Per (family, size): the 4-cell complex matrix + the 4 device suites at OMP=1,
# plus OMP∈{2,4} on opamp and ring_osc for the strict verdict.
#
# TSMC6 is scored in its OWN /4 column and never folded into the /16 — it is
# TSMC7 relabelled and a duplicate in the headline denominator inflates every
# total (docs/accuracy/methodology.md §2, §7).
#
# Usage:  PAR=12 bash scripts/tsmc6_gate_campaign.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
JOBDIR="${JOBDIR:-/tmp/tsmc6_gate_jobs}"
PAR="${PAR:-12}"
SIZES="${SIZES:-small medium large xl}"
mkdir -p "$JOBDIR"

DEVICE_SUITES="verify_nn_ac verify_complex_opamp_ac verify_nn_multi_tech_dc verify_nn_multi_tech_tran"
DETERMINISTIC="verify_complex_sram_snm verify_complex_switchcap"
MULTISTABLE="verify_complex_opamp verify_complex_ring_osc"

for tag in dn tf pfn; do
  echo "[tsmc6-gate] ===== $tag: waiting for checkpoints ($(date '+%F %T')) ====="
  while :; do
    missing=0
    for size in $SIZES; do for dev in nmos pmos; do
      [ -f "$CKPT/tsmc6_${tag}_${size}_${dev}_best.pt.complete" ] || missing=1
    done; done
    [ "$missing" -eq 0 ] && break
    sleep 300
  done
  echo "[tsmc6-gate] ===== $tag: all 8 checkpoints complete, gating ====="

  jobs="$JOBDIR/jobs_${tag}.txt"; : > "$jobs"
  for size in $SIZES; do
    for s in $DEVICE_SUITES $DETERMINISTIC; do echo "$tag $size TSMC6 $s 1" >> "$jobs"; done
    for s in $MULTISTABLE; do for omp in 1 2 4; do
      echo "$tag $size TSMC6 $s $omp" >> "$jobs"; done; done
  done
  echo "[tsmc6-gate] $tag: $(grep -c . "$jobs") jobs"
  PAR="$PAR" JOBS="$jobs" bash "$ROOT/scripts/v710_regate.sh"
  echo "[tsmc6-gate] ===== $tag done rc=$? ($(date '+%F %T')) ====="
  python "$ROOT/scripts/v710_regate_collect.py" || true
done

echo "[tsmc6-gate] ALL FAMILIES GATED $(date '+%F %T')"
echo "[tsmc6-gate] next: python scripts/v730_docs_build.py"
