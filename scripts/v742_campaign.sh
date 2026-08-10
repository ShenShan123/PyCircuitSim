#!/usr/bin/env bash
# V7.4.2 clean-rebuild campaign — the L-densified data rebuild.
#
# Same driver as V7.4.0 (scripts/v740_campaign.sh), pointed at its own
# evidence directory so the published V7.4.0 pass stays intact and the
# docs builder's per-report pass pin can tell the two apart.
#
# What changed upstream of this: the datasets were regenerated with
# `--max-l-ratio 1.35`, which samples inside each PDK length bin instead of
# at its lower corner only. See docs/plans/2026-08-10-v742-bsimar-capacity.md.
#
# Resumable and safe to launch while training is still running.
#
# Usage: PAR=32 bash scripts/v742_campaign.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V740_OUT="${V742_OUT:-$ROOT/results/v742_regate}" \
V740_SCRATCH="${V742_SCRATCH:-/tmp/v742_campaign}" \
V740_COVERAGE_ARGS="--passes v742" \
V740_DN_LOG="$ROOT/results/dn_train_master_v742.log" \
V740_TF_LOG="$ROOT/results/tf_train_master_v742.log" \
  exec bash "$ROOT/scripts/v740_campaign.sh" "$@"
