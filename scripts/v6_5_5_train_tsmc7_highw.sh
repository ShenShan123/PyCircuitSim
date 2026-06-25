#!/usr/bin/env bash
# V6.5.5 — tsmc7 opamp corridor, stronger weight (the 1c failure mode = the
# high-gain OP is unstable; a heavier corridor weight forces the NN to match
# the ground-truth currents at that OP, which is what could stabilize it).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOG="$ROOT/results/v6_5_5/train_logs"; mkdir -p "$LOG"; cd "$ROOT"
read -r -a GPUS <<<"${1:-0 1 2}"; NG=${#GPUS[@]}
W="${2:-8}"
SEEDS=(7 17 42 31)
jobs=()
for seed in "${SEEDS[@]}"; do for dev in nmos pmos; do jobs+=("$dev $seed"); done; done
run_one() {
  local gpu="$1" dev="$2" seed="$3"
  local name="tsmc7_dn_coropampL_w${W}_s${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG" \
    conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train \
      --model direct --size large --device-type "$dev" --tech-scope tsmc7 --cuda --overwrite \
      --data "$DATA/tsmc7_cor_${dev}.npz" --apply-filter off --swa-mode ema \
      --class-weights "traj_corridor=${W}" --seed "$seed" --exp-name "$name" \
      >"$LOG/${name}_${dev}.log" 2>&1 \
    && echo "[gpu$gpu] DONE $name $dev" || echo "[gpu$gpu] FAIL $name $dev"
}
nq=$((NG*2)); declare -a Q
for ((i=0;i<${#jobs[@]};i++)); do Q[$((i%nq))]+="${jobs[$i]}|"; done
pids=()
for ((qi=0;qi<nq;qi++)); do
  gpu="${GPUS[$((qi%NG))]}"; IFS='|' read -r -a qj <<<"${Q[$qi]}"
  ( for j in "${qj[@]}"; do [ -z "$j" ] && continue; read -r dev seed <<<"$j"; run_one "$gpu" "$dev" "$seed"; done ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[tsmc7-highw W=$W] done"
