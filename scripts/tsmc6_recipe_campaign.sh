#!/usr/bin/env bash
# V7.3.0 — train the surviving recipes for TSMC6, so the recipe reports score
# /20 like the clean reports rather than /16.
#
# TSMC6 has only ever carried clean checkpoints. V7.3.0 folds it into the
# headline denominator, which left the recipe tables scoring /16 against the
# clean tables' /20 — a mixed denominator inside one document set. This closes
# it: every recipe kept by the V7.3.0 filter gets its TSMC6 pair.
#
# Precondition, and it is not optional: the ring-only corridor dataset
# tsmc6_corro_{nmos,pmos}.npz must exist and be array_equal to tsmc7's. TSMC6
# is TSMC7 relabelled (docs/accuracy/methodology.md §7) and its entire value is
# as a *controlled* repeat — if the corridor differs, the repeat is confounded
# and the GPU-days are wasted. Refuse rather than warn.
#
# Warm starts resolve correctly without special-casing: recipe_train.sh points
# a curriculum fine-tune at {tech}_{tag}_{size}_{dev}, which for TSMC6 IS the
# clean checkpoint (the other techs need the v660clean archive at `large`
# because their production slot carries crit30f). Same rule, same result.
#
# Waves run in sequence so the three 4090s are not oversubscribed; xl runs at
# lower concurrency because those checkpoints are ~15 M parameters.
#
#   Usage:  GPUS="0 1 2" bash scripts/tsmc6_recipe_campaign.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="$ROOT/external_compact_models/bsimar/data/datasets"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
GPUS="${GPUS:-0 1 2}"
PY="${NN_PY:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"

echo "[tsmc6-recipe] start $(date '+%F %T')"

# ---- 0. the repeat must be controlled ----
"$PY" - "$DS" <<'PYEOF' || { echo "[tsmc6-recipe] ABORT"; exit 1; }
import numpy as np, pathlib, sys
D = pathlib.Path(sys.argv[1]); bad = 0
for dev in ("nmos", "pmos"):
    try:
        a = np.load(D / f"tsmc6_corro_{dev}.npz", allow_pickle=True)
        b = np.load(D / f"tsmc7_corro_{dev}.npz", allow_pickle=True)
    except OSError as e:
        print(f"[tsmc6-recipe] missing corridor dataset: {e}"); sys.exit(1)
    for k in ("inputs", "geometry", "outputs", "sample_class"):
        same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
        print(f"  corro {dev} {k}: {a[k].shape} array_equal={same}")
        bad += not same
if bad:
    print("[tsmc6-recipe] corridor differs from tsmc7 — the repeat would be "
          "confounded. Re-harvest with --circuits ring_osc --frag-tag O.")
    sys.exit(1)
PYEOF

# ---- 1. wait out any training wave already on the GPUs ----
while pgrep -f 'recipe_train.sh _one' >/dev/null; do
  echo "[tsmc6-recipe] GPUs busy with another wave … $(date '+%T')"
  sleep 300
done

# wave <MODEL> <SIZE> <NSTREAMS> <recipes...>
wave () {
  local model="$1" size="$2" streams="$3"; shift 3
  echo "[tsmc6-recipe] === $model $size: $* ==="
  MODEL="$model" RECIPES="$*" SIZES="$size" TECHS=tsmc6 DEVS="nmos pmos" \
    GPUS="$GPUS" NSTREAMS="$streams" \
    ${EXTRA_ARGS:+EXTRA_ARGS="$EXTRA_ARGS"} \
    bash "$ROOT/scripts/recipe_train.sh" || echo "[tsmc6-recipe] wave returned $?"
}

# DirectNet first — cheapest, and it is the production family.
# csob is from-scratch (no --epochs override); the rest are 120-epoch warm starts.
wave direct large 4 crit30f csob
wave direct xl    4 corroft crit15m

# BSIM-AR: all warm-start fine-tunes. xl is ~14.8 M params, hence 2 streams.
wave transformer medium 4 corroft corro15
wave transformer large  4 corroft crit15m crit30
wave transformer xl     2 corroft corro15 crit15m crit30

n=$(ls -1 "$CKPT" | grep -cE '^tsmc6_(dn|tf)_[a-z0-9]+_(large|xl|medium)_(nmos|pmos)_best\.pt\.complete$')
echo "[tsmc6-recipe] DONE $(date '+%F %T') — $n/26 recipe checkpoints complete"
