#!/usr/bin/env bash
# V6.4.8 complex-circuit-sweeps Phase A: broad-coverage TSMC7 DirectNet retrain.
#
# WHY: the four complex-circuit *parametric* sweeps walk geometry/VT/VDD/stimulus
# space that the *specialized* shipping checkpoint (v6_4_7_pivcor_w2_s7_tsmc7,
# opamp-corridor-trained) treats as out-of-distribution. Retrain a broad
# generalist on the broad v2 dataset (L 8-120nm, NFIN 2-12, all variants) so
# sweep configs are in-distribution. Size = medium (V6.4.8 S1 proved `large`
# COLLAPSES the opamp). Overwrites the shipping `tsmc7_dn_medium`.
#
# The previous specialized checkpoint stays recoverable: the pivcor real files
# v6_4_7_pivcor_w2_s7_tsmc7_* are NOT touched (we only drop the symlinks that
# pointed at them); re-link to recover.
#
# Multi-GPU: NMOS on GPU 0, PMOS on GPU 1, in parallel.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

export PYTHONPATH="${ROOT}/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
CKDIR="external_compact_models/bsimar/checkpoints"
DATADIR="external_compact_models/bsimar/data/datasets"
EXP="v6_4_8_broad_tsmc7"
LOGDIR="training_logs/v6_4_8_broad"
mkdir -p "$LOGDIR"

echo "=== [$(date +%H:%M:%S)] Dropping tsmc7_dn_medium symlinks (pivcor backup preserved) ==="
for f in tsmc7_dn_medium_nmos_best.pt tsmc7_dn_medium_nmos_norm.npz \
         tsmc7_dn_medium_pmos_best.pt tsmc7_dn_medium_pmos_norm.npz; do
  if [ -L "$CKDIR/$f" ]; then
    echo "  rm symlink $f (was -> $(readlink "$CKDIR/$f"))"
    rm -f "$CKDIR/$f"
  fi
done

train_one () {
  local dev="$1" gpu="$2"
  local log="$LOGDIR/${EXP}_${dev}.log"
  echo "=== [$(date +%H:%M:%S)] train ${dev} on GPU ${gpu} -> ${EXP}_${dev} ==="
  CUDA_VISIBLE_DEVICES="$gpu" conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type "$dev" --tech-scope tsmc7 \
    --data "${DATADIR}/tsmc7_v2_${dev}.npz" \
    --exp-name "$EXP" \
    --cuda --overwrite > "$log" 2>&1
  echo "$?" > "$LOGDIR/${EXP}_${dev}.status"
  echo "=== [$(date +%H:%M:%S)] ${dev} finished, exit=$(cat "$LOGDIR/${EXP}_${dev}.status") ==="
}

train_one nmos 0 &
PID_N=$!
train_one pmos 1 &
PID_P=$!
wait $PID_N
wait $PID_P

SN=$(cat "$LOGDIR/${EXP}_nmos.status" 2>/dev/null || echo 1)
SP=$(cat "$LOGDIR/${EXP}_pmos.status" 2>/dev/null || echo 1)
echo "=== [$(date +%H:%M:%S)] training done: nmos=$SN pmos=$SP ==="

if [ "$SN" = "0" ] && [ "$SP" = "0" ]; then
  echo "=== Linking tsmc7_dn_medium -> ${EXP} ==="
  for dev in nmos pmos; do
    for suf in best.pt norm.npz; do
      tgt="${EXP}_${dev}_${suf}"
      lnk="tsmc7_dn_medium_${dev}_${suf}"
      if [ -f "$CKDIR/$tgt" ]; then
        ln -sf "$tgt" "$CKDIR/$lnk"
        echo "  $lnk -> $tgt"
      else
        echo "  MISSING expected artifact $CKDIR/$tgt"
      fi
    done
  done
  echo "=== [$(date +%H:%M:%S)] RETRAIN COMPLETE ==="
else
  echo "=== RETRAIN FAILED (nmos=$SN pmos=$SP) — restoring pivcor symlinks ==="
  for dev in nmos pmos; do
    for suf in best.pt norm.npz; do
      ln -sf "v6_4_7_pivcor_w2_s7_tsmc7_${dev}_${suf}" "$CKDIR/tsmc7_dn_medium_${dev}_${suf}"
    done
  done
  exit 1
fi