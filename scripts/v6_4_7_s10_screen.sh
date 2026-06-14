#!/usr/bin/env bash
# V6.4.7 S10 (P4) — fine-tune λ-screen: warm-start tsmc7 nmos+pmos from the
# HEALTHY control-v2 seed (s17: RO 8.66% best, opamp gain 180 vs NG 163 — gain
# too HIGH because the autograd gds is under-predicted; the Sobolev term raises
# gds toward OSDI → lowers gain toward NG, the right direction) and add the
# Sobolev id-derivative term at a few (λ, strong_boost) settings.
#
# Screen kill gate (plan S10): best config must cut TSMC7 opamp gain err below
# ~15% with the inverter held. Scored separately by v6_4_7_s10_score.sh.
#
# Fan-out: 6 jobs (3 configs x {nmos,pmos}) round-robin across GPUs 0/1/2,
# one serial queue per GPU (user ruling: parallel GPUs). DirectNet medium uses
# <2 GB so co-location with the GPU-1 tenant is fine.
#
# Usage: bash scripts/v6_4_7_s10_screen.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/external_compact_models/bsimar/data/datasets"
LOGDIR="$ROOT/results/v6_4_7/s10_screen_logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

TECH=tsmc7
SRC_SEED=17                      # healthy warm-start seed
FT_SEED=17                       # dropout RNG only (data split is seed=42 fixed)
EPOCHS=40
PATIENCE=15
LR=1e-4
GPUS=(0 1 2)

# config: name  lam  strong_boost
CONFIGS=(
  "a 0.1 1.0"
  "b 0.1 4.0"
  "c 0.3 4.0"
)
DEVS=(nmos pmos)

jobs=()
for cfg in "${CONFIGS[@]}"; do
  read -r name lam boost <<<"$cfg"
  for dev in "${DEVS[@]}"; do
    jobs+=("$name $lam $boost $dev")
  done
done
echo "[s10-screen] ${#jobs[@]} fine-tune jobs across GPUs ${GPUS[*]}"

run_queue() {
  local gpu="$1"; shift
  local failed=0
  for job in "$@"; do
    read -r name lam boost dev <<<"$job"
    local src="v6_4_7_ctlv2_s${SRC_SEED}_${TECH}_${dev}"
    local exp="v6_4_7_s10ft_${name}_${TECH}"
    local stem="${exp}_${dev}"
    local log="$LOGDIR/${stem}.log"
    echo "[gpu$gpu] START $stem (λ=$lam boost=$boost) $(date +%H:%M:%S)"
    if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
        PYTHONPATH="$ROOT/external_compact_models:$ROOT/external_compact_models/PyCMG${PYTHONPATH:+:$PYTHONPATH}" \
        conda run -n pycircuitsim python -u -m bsimar.cli.train \
          --model direct --size medium \
          --device-type "$dev" --tech-scope "$TECH" --cuda --overwrite \
          --data "$DATA/${TECH}_v2_${dev}.npz" \
          --apply-filter off --swa-mode ema \
          --sobolev --lam-sobolev "$lam" --sobolev-strong-boost "$boost" \
          --init-from "$src" \
          --epochs "$EPOCHS" --patience "$PATIENCE" --lr "$LR" \
          --seed "$FT_SEED" --exp-name "$exp" \
          >"$log" 2>&1; then
      echo "[gpu$gpu] DONE  $stem $(date +%H:%M:%S)"
    else
      echo "[gpu$gpu] FAIL  $stem (see $log)" >&2
      failed=1
    fi
  done
  return $failed
}

q0=(); q1=(); q2=()
for i in "${!jobs[@]}"; do
  case $((i % 3)) in
    0) q0+=("${jobs[$i]}");;
    1) q1+=("${jobs[$i]}");;
    2) q2+=("${jobs[$i]}");;
  esac
done

run_queue "${GPUS[0]}" "${q0[@]}" & p0=$!
run_queue "${GPUS[1]}" "${q1[@]}" & p1=$!
run_queue "${GPUS[2]}" "${q2[@]}" & p2=$!
rc=0
wait $p0 || rc=1
wait $p1 || rc=1
wait $p2 || rc=1
echo "[s10-screen] all queues finished, rc=$rc"
ls -la "$ROOT/external_compact_models/bsimar/checkpoints/" | grep s10ft || true
exit $rc
