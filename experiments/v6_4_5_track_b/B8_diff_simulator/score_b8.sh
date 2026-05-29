#!/usr/bin/env bash
# Score the B8 TTFT candidate with the PRODUCTION scorer (the real test).
set -euo pipefail
cd "$(dirname "$0")/../../.."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  conda run -n pycircuitsim python scripts/eval_v6_4_5_candidate.py \
    --tech TSMC7 --nmos b8_ttft_tsmc7_nmos --pmos b8_ttft_tsmc7_pmos --json
