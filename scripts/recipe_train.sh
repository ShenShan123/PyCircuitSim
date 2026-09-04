#!/usr/bin/env bash
# Parallel full-terminal checkpoint training. DirectNet writes dnf bundles;
# BSIM-AR writes tff bundles. Canonical data are always {tech}_dnf_{dev}.npz.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: [ENV=VALUE ...] bash scripts/recipe_train.sh [--force]

Environment:
  MODEL        direct | transformer (default: direct)
  RECIPES      clean | s7 | s17 | s123 | sub | ar3 | ar3roll | corridor
  TECHS        space-separated lowercase technology names
  SIZES        space-separated small | medium | large | xl
  DEVS         space-separated nmos | pmos
  GPUS         space-separated physical GPU IDs
  NSTREAMS     maximum concurrent training jobs
  TRAIN_OMP    CPU threads per job
  EXTRA_ARGS   extra full-terminal training arguments
  BSIMAR_DATA_DIR, BSIMAR_CHECKPOINT_DIR, RECIPE_TRAIN_LOG_DIR
EOF
  exit 0
fi

CKPT="${BSIMAR_CHECKPOINT_DIR:-$ROOT/external_compact_models/neural_network/checkpoints}"
DATA="${BSIMAR_DATA_DIR:-$ROOT/external_compact_models/neural_network/data/datasets}"
LOGDIR="${RECIPE_TRAIN_LOG_DIR:-$ROOT/results/recipe_bench/train_logs}"
MODEL="${MODEL:-direct}"
case "$MODEL" in
  direct) TAG="dnf" ;;
  transformer) TAG="tff" ;;
  *) echo "[train] UNKNOWN MODEL=$MODEL (direct|transformer)" >&2; exit 2 ;;
esac

mkdir -p "$CKPT" "$LOGDIR"
read -r -a GPU_IDS <<< "${GPUS:-0 1 2}"
NGPU=${#GPU_IDS[@]}
NSTREAMS="${NSTREAMS:-9}"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"

bundle_ready () {
  local stem="$1"
  [ -f "$CKPT/${stem}_best.pt" ] || return 1
  [ -f "$CKPT/${stem}_norm.npz" ] || return 1
  [ -f "$CKPT/${stem}_best.pt.complete" ] || return 1
  if [ "$TAG" = "tff" ]; then
    [ -f "$CKPT/${stem}_config.npz" ] || return 1
  fi
}

if [ "${1:-}" = "_one" ]; then
  recipe="$2"; tech="$3"; size="$4"; dev="$5"; gpu="$6"; force="${7:-}"
  if [ "$recipe" = "clean" ]; then
    name="${tech}_${TAG}_${size}_${dev}"
  else
    name="${tech}_${TAG}_${recipe}_${size}_${dev}"
  fi
  checkpoint="$CKPT/${name}_best.pt"
  log="$LOGDIR/${name}.log"
  if bundle_ready "$name" && [ "$force" != "--force" ]; then
    echo "[train] SKIP existing $name"
    exit 0
  fi
  if [ -f "$checkpoint" ] && [ "$force" != "--force" ]; then
    echo "[train] RETRAIN incomplete $name"
  fi

  dataset="$DATA/${tech}_dnf_${dev}.npz"
  extra=()
  case "$recipe" in
    clean) ;;
    s7) extra+=(--seed 7) ;;
    s17) extra+=(--seed 17) ;;
    s123) extra+=(--seed 123) ;;
    sub)
      [ "$MODEL" = "transformer" ] || {
        echo "[train] recipe sub requires MODEL=transformer" >&2; exit 2;
      }
      extra+=(--subthresh)
      ;;
    ar3)
      [ "$MODEL" = "transformer" ] || {
        echo "[train] recipe ar3 requires MODEL=transformer" >&2; exit 2;
      }
      extra+=(--full-terminal-ar-targets 3)
      ;;
    ar3roll)
      [ "$MODEL" = "transformer" ] || {
        echo "[train] recipe ar3roll requires MODEL=transformer" >&2; exit 2;
      }
      extra+=(--full-terminal-ar-targets 3 --autoregressive-training)
      ;;
    corridor)
      dataset="$DATA/${tech}_dnf_corridor_${dev}.npz"
      extra+=(--training-overlay-classes traj_corridor
              --class-weights traj_corridor=3.0)
      ;;
    *) echo "[train] UNKNOWN recipe $recipe" >&2; exit 2 ;;
  esac
  [ -f "$dataset" ] && [ -f "$dataset.complete" ] || {
    echo "[train] MISSING canonical dataset bundle $dataset" >&2; exit 1;
  }
  if [ "$recipe" != "clean" ]; then
    extra+=(--exp-name "${tech}_${TAG}_${recipe}_${size}")
  fi
  if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a operator_extra <<< "$EXTRA_ARGS"
    extra+=("${operator_extra[@]}")
  fi

  rm -f "$checkpoint.complete"
  echo "[train] START $name on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="${TRAIN_OMP:-4}" \
  MKL_NUM_THREADS="${TRAIN_OMP:-4}" \
    conda run --no-capture-output -n pycircuitsim \
    python -u -m neural_network.cli.train \
    --model "$MODEL" --size "$size" --device-type "$dev" \
    --tech-scope "$tech" --data "$dataset" \
    --swa-mode ema --seed 42 --cuda --overwrite \
    "${extra[@]}" >"$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ] && bundle_ready "$name"; then
    echo "[train] DONE $name"
    exit 0
  fi
  rm -f "$checkpoint.complete"
  echo "[train] FAIL $name (rc=$rc, see $log)" >&2
  exit 3
fi

if [ "$#" -gt 1 ]; then
  echo "[train] UNKNOWN arguments: $*" >&2
  echo "Usage: [ENV=VALUE ...] bash scripts/recipe_train.sh [--force]" >&2
  exit 2
fi
FORCE="${1:-}"
if [ -n "$FORCE" ] && [ "$FORCE" != "--force" ]; then
  echo "[train] UNKNOWN argument: $FORCE" >&2
  echo "Usage: [ENV=VALUE ...] bash scripts/recipe_train.sh [--force]" >&2
  exit 2
fi

read -r -a recipes <<< "${RECIPES:-clean}"
read -r -a techs <<< "${TECHS:-tsmc5 tsmc6 tsmc7 tsmc12 tsmc16}"
read -r -a sizes <<< "${SIZES:-small medium large xl}"
read -r -a devs <<< "${DEVS:-nmos pmos}"

missing=0
for tech in "${techs[@]}"; do
  for dev in "${devs[@]}"; do
    dataset="$DATA/${tech}_dnf_${dev}.npz"
    [ -f "$dataset" ] && [ -f "$dataset.complete" ] || {
      echo "[train] MISSING canonical dataset bundle $dataset" >&2
      missing=1
    }
  done
done
[ "$missing" -eq 0 ] || exit 1

lines=(); index=0
for size in "${sizes[@]}"; do
  for recipe in "${recipes[@]}"; do
    for tech in "${techs[@]}"; do
      for dev in "${devs[@]}"; do
        if [ "$recipe" = "clean" ]; then
          name="${tech}_${TAG}_${size}_${dev}"
        else
          name="${tech}_${TAG}_${recipe}_${size}_${dev}"
        fi
        if bundle_ready "$name" && [ "$FORCE" != "--force" ]; then
          continue
        fi
        lines+=("$recipe $tech $size $dev ${GPU_IDS[$((index % NGPU))]} ${FORCE:-noforce}")
        index=$((index + 1))
      done
    done
  done
done

if [ "${#lines[@]}" -eq 0 ]; then
  echo "[train] nothing to do — all requested bundles are complete"
  exit 0
fi
echo "[train] launching ${#lines[@]} jobs, $NSTREAMS concurrent"
printf '%s\n' "${lines[@]}" | xargs -P "$NSTREAMS" -L1 "$SELF" _one
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[train] INFRASTRUCTURE FAILURE: at least one bundle is incomplete" >&2
  exit 1
fi
echo "[train] ALL TRAINING COMPLETE"
