#!/usr/bin/env python3
"""Generate full-terminal re-gate job lists for scripts/v710_regate.sh.

One line per job: ``tag variant TECH suite omp``.

Pools (write one file each so they can be dispatched with different PAR):

* ``clean`` — DirectNet-Full and BSIM-AR-Full S/M/L/XL tiers.
* ``simple_v2`` — nominal held-out topology screen for both families.

Usage: python scripts/v710_regate_jobs.py <outdir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common.simple_circuit_catalog import (  # noqa: E402
    SIMPLE_V1,
    SIMPLE_V2,
    cases,
)

TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]

DEVICE_SUITES = [
    "verify_device_integrity",        # full DC surface/derivative diagnostics
    "verify_terminal_integrity",      # terminal currents + 4x4 capacitance
    "verify_nn_subckt",               # flat/nested NN model integration
    "verify_nn_ac",                 # device CS-amp small-signal
    "verify_circuit_opamp_ac",      # two-stage Miller open-loop AC
    "verify_nn_multi_tech_dc",      # parametric Id-Vgs
    "verify_nn_multi_tech_tran",    # parametric inverter transient
]
_SIMPLE_V1_CASES = cases(score_version=SIMPLE_V1)
_SIMPLE_V2_CASES = cases(score_version=SIMPLE_V2)
MULTISTABLE = [
    case.campaign_suite
    for case in _SIMPLE_V1_CASES
    if len(case.omp_threads) > 1
]
DETERMINISTIC = [
    case.campaign_suite
    for case in _SIMPLE_V1_CASES
    if len(case.omp_threads) == 1
]
SIMPLE_V2_SUITES = [case.campaign_suite for case in _SIMPLE_V2_CASES]

CLEAN_VARIANTS = ["small", "medium", "large", "xl"]


def full(
    tag: str,
    variants: list[str],
    techs: list[str] = TECHS,
) -> list[str]:
    jobs = []
    for v in variants:
        for t in techs:
            for s in DEVICE_SUITES:
                jobs.append(f"{tag} {v} {t} {s} 1")
            for s in DETERMINISTIC:
                jobs.append(f"{tag} {v} {t} {s} 1")
            for s in MULTISTABLE:
                for omp in (1, 2, 4):
                    jobs.append(f"{tag} {v} {t} {s} {omp}")
    return jobs


def simple_v2(tag: str, variants: list[str]) -> list[str]:
    """Nominal diagnostic cells stay OMP=1 and outside qualification totals."""
    return [
        f"{tag} {variant} {tech} {suite} 1"
        for variant in variants
        for tech in TECHS
        for suite in SIMPLE_V2_SUITES
    ]


def build_pools() -> dict[str, list[str]]:
    """Return every campaign pool from one testable source of truth."""
    return {
        "clean": [
            *full("dnf", CLEAN_VARIANTS),
            *full("tff", CLEAN_VARIANTS),
        ],
        "simple_v2": [
            *simple_v2("dnf", CLEAN_VARIANTS),
            *simple_v2("tff", CLEAN_VARIANTS),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outdir", type=Path, help="directory for generated job lists",
    )
    args = parser.parse_args(argv)
    out: Path = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    pools = build_pools()
    for name, jobs in pools.items():
        p = out / f"jobs_{name}.txt"
        p.write_text("\n".join(jobs) + "\n")
        print(f"{p}  {len(jobs)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
