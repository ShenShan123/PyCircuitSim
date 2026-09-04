#!/usr/bin/env python3
"""NN inverter gate: DirectNet-Full / BSIM-AR-Full vs BSIM-CMG.

CMOS inverter VTC and pulse response across the TSMC techs. This is the first
tier where devices interact, so it is where a model error that cancels in a
single-device sweep shows up — the VTC trip point in particular is a high-gain
fixed point and will move under a bias error the Id-Vgs curve absorbs.

The device half of the old flat `verify_nn_dc_tran.py` is now
`tests/single_devices/verify_nn_dc.py`. Both drive the same suite bodies in
`tests.common.nn_gate` — the split moved the entry points, not the scoring.

Usage:
    conda run -n pycircuitsim python tests/simple_circuits/verify_nn_inverter.py
    conda run -n pycircuitsim python tests/simple_circuits/verify_nn_inverter.py --tech TSMC5
    conda run -n pycircuitsim python tests/simple_circuits/verify_nn_inverter.py --vtc-only

Pin the threads — the VTC trip has ~±1 % run-to-run scatter under
multi-threaded BLAS, so this is not optional for a reproducible verdict:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn_gate import (  # noqa: E402
    parse_techs, run_gate, tech_arg_parser,
)


def main() -> int:
    ap = tech_arg_parser(
        "NN inverter verification: full-terminal families vs BSIM-CMG")
    ap.add_argument("--vtc-only", action="store_true",
                    help="Run the inverter VTC (DC) tests only")
    ap.add_argument("--tran-only", action="store_true",
                    help="Run the inverter transient tests only")
    args = ap.parse_args()

    explicit = args.vtc_only or args.tran_only
    return run_gate(
        parse_techs(args),
        title="NN Inverter Verification (VTC + transient)",
        inverter_vtc=args.vtc_only or not explicit,
        inverter_tran=args.tran_only or not explicit,
    )


if __name__ == "__main__":
    sys.exit(main())
