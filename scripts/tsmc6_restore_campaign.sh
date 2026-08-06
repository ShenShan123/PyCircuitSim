#!/usr/bin/env bash
# V7.1.0 — TSMC6 restoration campaign: clean capacity sweep, all three NN
# families, all four scales.
#
# TSMC6 was retired in V6.13.0 because it is TSMC7 relabelled under BSIM-CMG
# (audit D1) — that finding is unchanged and is documented at the registry
# entry in `bsimar/config.py` and in `docs/accuracy/methodology.md` §7. It is being
# carried again by explicit decision, as the project's only controlled repeat
# experiment: same data, same recipe, different training run.
#
# Waits for the regenerated datasets, then runs three training waves in
# sequence (DirectNet → BSIM-AR → PFN) so the GPUs are not oversubscribed:
#
#   DirectNet  4 sizes x 2 devices   (cheap)
#   BSIM-AR    4 sizes x 2 devices   (xl is the expensive one)
#   PFN        4 sizes x 2 devices   (--amp, per the V6.10 large-tier recipe)
#
# 24 checkpoints total, one identical clean recipe
# (`--apply-filter off --swa-mode ema --seed 42`) — the uniformity contract in
# docs/accuracy/methodology.md §5.
#
# Usage:  GPUS="0 1 2" bash scripts/tsmc6_restore_campaign.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="$ROOT/external_compact_models/bsimar/data/datasets"
GPUS="${GPUS:-0 1 2}"
SIZES="${SIZES:-small medium large xl}"

echo "[tsmc6] campaign start $(date '+%F %T')"

# ---- 1. wait for the regenerated datasets ----
while [ ! -f "$DS/tsmc6_nmos.npz" ] || [ ! -f "$DS/tsmc6_pmos.npz" ]; do
  echo "[tsmc6] waiting for datasets … $(date '+%T')"
  sleep 300
done
# A dataset still being written is worse than a missing one: wait for the size
# to stop changing before any trainer opens it.
for f in "$DS/tsmc6_nmos.npz" "$DS/tsmc6_pmos.npz"; do
  prev=0
  while :; do
    cur=$(stat -c %s "$f")
    [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ] && break
    prev=$cur; sleep 60
  done
  echo "[tsmc6] dataset ready: $f ($(numfmt --to=iec "$prev" 2>/dev/null || echo "$prev" bytes))"
done

# ---- 1b. verify the repeat is actually controlled ----
# TSMC6 exists to be a bit-identical repeat of TSMC7 (docs/accuracy/methodology.md
# §7). If the regenerated data is NOT array_equal to tsmc7_*, the experiment is
# confounded and the training is a waste of GPU-days — so refuse, don't warn.
# This check has already earned itself once: the first regeneration followed
# CLAUDE.md's recipe, which omitted --enable-subvt-off, and came out 4.7 %
# smaller (sample_class 11 missing) while looking perfectly healthy in the log.
echo "[tsmc6] verifying the dataset is bit-identical to tsmc7 …"
"${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}" - "$DS" <<'PYEOF' || exit 1
import numpy as np, pathlib, sys
D = pathlib.Path(sys.argv[1])
bad = 0
for dev in ("nmos", "pmos"):
    a = np.load(D / f"tsmc6_{dev}.npz", allow_pickle=True)
    b = np.load(D / f"tsmc7_{dev}.npz", allow_pickle=True)
    for k in ("inputs", "geometry", "outputs", "sample_class"):
        same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
        print(f"  {dev} {k}: {a[k].shape} vs {b[k].shape} array_equal={same}")
        bad += not same
if bad:
    print("[tsmc6] ABORT: regenerated data differs from tsmc7 — the repeat would "
          "be confounded. Check the generator flags (--enable-inv-trip AND "
          "--enable-subvt-off) before training.")
    sys.exit(1)
print("[tsmc6] datasets are bit-identical to tsmc7 — the repeat is controlled.")
PYEOF

# ---- 2. training waves ----
wave () {
  local model="$1" streams="$2"; shift 2
  echo "[tsmc6] ===== WAVE $model ($(date '+%F %T')) ====="
  MODEL="$model" RECIPES=clean TECHS=tsmc6 SIZES="$SIZES" DEVS="nmos pmos" \
    GPUS="$GPUS" NSTREAMS="$streams" EXTRA_ARGS="${EXTRA:-}" \
    bash "$ROOT/scripts/recipe_train.sh"
  echo "[tsmc6] ===== WAVE $model rc=$? ====="
}

EXTRA="" wave direct 6
EXTRA="" wave transformer 3
EXTRA="--amp" wave tabpfn 3

echo "[tsmc6] ALL WAVES DONE $(date '+%F %T')"
ls -1 "$ROOT/external_compact_models/bsimar/checkpoints/" | grep -c "^tsmc6_.*_best\.pt\.complete$" \
  | xargs -I{} echo "[tsmc6] completed checkpoints: {} / 24"
