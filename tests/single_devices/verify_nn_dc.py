#!/usr/bin/env python3
"""NN device gate: DirectNet-Full / BSIM-AR-Full vs BSIM-CMG.

Single NMOS and PMOS across the TSMC techs — Id-Vgs DC sweeps and the NMOS
transient. This is the model surface itself: there is no circuit here to
absorb an error, so a failure is the compact model.

The inverter half of the old flat `verify_nn_dc_tran.py` is now
`tests/simple_circuits/verify_nn_inverter.py`. Both drive the same suite
bodies in `tests.common.nn_gate` — the split moved the entry points, not the
scoring.

Usage:
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py --tech TSMC5
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py --dc-only
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py --idvds-diagnostic

Pin the threads for a reproducible verdict:
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
        "NN device verification: full-terminal families vs BSIM-CMG")
    ap.add_argument("--dc-only", action="store_true",
                    help="Run NMOS DC tests only")
    ap.add_argument("--tran-only", action="store_true",
                    help="Run NMOS transient tests only")
    ap.add_argument("--pmos-only", action="store_true",
                    help="Run PMOS DC tests only")
    ap.add_argument("--sign-diagnostic", action="store_true",
                    help="Run sign pre-screen diagnostic (Vgs=0 bias points)")
    ap.add_argument("--idvds-diagnostic", action="store_true",
                    help="Run Id-Vds curve diagnostic at Vgs=0")
    args = ap.parse_args()

    explicit = (args.dc_only or args.tran_only or args.pmos_only
                or args.sign_diagnostic or args.idvds_diagnostic)
    return run_gate(
        parse_techs(args),
        title="NN Device Verification (single NMOS / PMOS)",
        nmos_dc=args.dc_only or not explicit,
        pmos_dc=args.pmos_only or not explicit,
        nmos_tran=args.tran_only or not explicit,
        sign_diag=args.sign_diagnostic,
        idvds_diag=args.idvds_diagnostic,
    )


if __name__ == "__main__":
    sys.exit(main())
