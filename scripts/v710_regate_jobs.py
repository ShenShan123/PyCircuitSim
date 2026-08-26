#!/usr/bin/env python3
"""Generate V7.1.0 re-gate job lists for scripts/v710_regate.sh.

One line per job: ``tag variant TECH suite omp``.

Pools (write one file each so they can be dispatched with different PAR):

* ``dn``   — DirectNet, all 10 on-disk variants, full re-gate
             (4 device suites + 4 complex circuits + OMP{1,2,4} on opamp/ring).
* ``tf_dev``    — BSIM-AR device suites only, 5 priority variants (~40x per eval).
* ``tf_strict`` — BSIM-AR strict-OMP sweep for the `large` corridor recipes,
                  the one gap BSIM-AR-L74-clean.md flags explicitly.
* ``clean`` — DirectNet and BSIM-AR clean S/M/L/XL tiers across all five
              reported technologies, including the TSMC6 repeat.

Usage: python scripts/v710_regate_jobs.py <outdir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

TECHS = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]
CLEAN_TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]

DEVICE_SUITES = [
    "verify_nn_ac",                 # device CS-amp small-signal
    "verify_complex_opamp_ac",      # two-stage Miller open-loop AC
    "verify_nn_multi_tech_dc",      # parametric Id-Vgs
    "verify_nn_multi_tech_tran",    # parametric inverter transient
]
MULTISTABLE = ["verify_complex_opamp", "verify_complex_ring_osc"]
DETERMINISTIC = ["verify_complex_sram_snm", "verify_complex_switchcap"]

DN_VARIANTS = [
    "small", "medium", "large", "xl",          # `large` = production crit30f weights
    "v660clean_large", "crit30f_large", "csob_large",
    "corroft_xl", "crit10_xl", "crit15m_xl",
]
TF_DEV_VARIANTS = ["small", "medium", "large", "xl", "corroft_medium"]
TF_STRICT_VARIANTS = ["corroft_large", "crit15m_large", "crit30_large", "corro15_medium"]
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


def device_only(tag: str, variants: list[str]) -> list[str]:
    return [f"{tag} {v} {t} {s} 1"
            for v in variants for t in TECHS for s in DEVICE_SUITES]


def strict_only(tag: str, variants: list[str]) -> list[str]:
    return [f"{tag} {v} {t} {s} {omp}"
            for v in variants for t in TECHS for s in MULTISTABLE for omp in (1, 2, 4)]


def build_pools() -> dict[str, list[str]]:
    """Return every campaign pool from one testable source of truth."""
    return {
        "dn": full("dn", DN_VARIANTS),
        "tf_dev": device_only("tf", TF_DEV_VARIANTS),
        "tf_strict": strict_only("tf", TF_STRICT_VARIANTS),
        "clean": [
            *full("dn", CLEAN_VARIANTS, CLEAN_TECHS),
            *full("tf", CLEAN_VARIANTS, CLEAN_TECHS),
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
