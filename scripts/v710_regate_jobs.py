#!/usr/bin/env python3
"""Generate full-terminal re-gate job lists for scripts/v710_regate.sh.

One line per job: ``tag variant TECH suite omp``.

Pools (write one file each so they can be dispatched with different PAR):

* ``clean`` — DirectNet-Full and BSIM-AR-Full S/M/L/XL tiers.
* ``simple_v2`` — nominal held-out topology screen for both families.
* ``canary`` — the source-relative-frame canary per checkpoint group. It is
  its own pool so the qualification denominator of ``clean`` is unchanged.

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
from external_compact_models.cli_options import TRAINING_RECIPES  # noqa: E402

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
# AGENTS.md names this gate as the canary for the source-relative inference
# contract. It answers a runtime-frame question per checkpoint group rather
# than a per-device accuracy question, so it runs outside the clean pool.
CANARY_SUITES = ["verify_nn_lifted_source_dc"]

CLEAN_VARIANTS = ["small", "medium", "large", "xl"]


def evaluation_cases() -> dict[str, dict[str, str]]:
    """Expose catalog circuit IDs and campaign suite names with their owners."""
    inventory = {}
    for case in cases():
        inventory[case.case_id] = {
            "group": "simple_circuits", "suite": case.campaign_suite,
            "pool": "clean" if case.score_version == SIMPLE_V1 else "simple_v2",
            "role": case.role, "description": case.label,
        }
    for suite in [*DEVICE_SUITES, *CANARY_SUITES]:
        name = suite.removeprefix("verify_")
        group = ("single_devices" if (ROOT / "tests/single_devices" / f"{suite}.py").is_file()
                 else "simple_circuits")
        inventory[name] = {"group": group, "suite": suite,
                           "pool": "canary" if suite in CANARY_SUITES else "clean",
                           "role": "suite", "description": suite}
    return inventory


def select_cases(
    pools: list[str] | None, groups: list[str] | None, names: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve explicit selections without silently dropping a requested case."""
    inventory = evaluation_cases()
    if names:
        unknown = set(names) - inventory.keys()
        if unknown:
            raise ValueError(f"unknown evaluation cases: {sorted(unknown)}")
        if groups and any(inventory[name]["group"] not in groups for name in names):
            raise ValueError("case selection conflicts with --evaluation-group")
        selected = list(names)
    else:
        selected = [name for name, item in inventory.items()
                    if (not groups or item["group"] in groups)
                    and (not pools or item["pool"] in pools)]
    if pools and any(inventory[name]["pool"] not in pools for name in selected):
        raise ValueError("case selection conflicts with --pools")
    chosen_pools = [pool for pool in ("clean", "simple_v2", "canary")
                    if any(inventory[name]["pool"] == pool for name in selected)]
    if not selected or (pools and set(chosen_pools) != set(pools)):
        raise ValueError("selection produces an empty evaluation pool")
    return chosen_pools, selected


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


def canary(tag: str, variants: list[str]) -> list[str]:
    """Runtime-contract canaries: one OMP=1 cell per checkpoint group."""
    return [
        f"{tag} {variant} {tech} {suite} 1"
        for variant in variants
        for tech in TECHS
        for suite in CANARY_SUITES
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
        "canary": [
            *canary("dnf", CLEAN_VARIANTS),
            *canary("tff", CLEAN_VARIANTS),
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outdir", type=Path, help="directory for generated job lists",
    )
    parser.add_argument("--tech", nargs="+", choices=TECHS, default=TECHS)
    parser.add_argument("--tag", nargs="+", choices=("dnf", "tff"), default=["dnf", "tff"])
    parser.add_argument("--size", nargs="+", choices=CLEAN_VARIANTS, default=CLEAN_VARIANTS)
    parser.add_argument("--omp", nargs="+", choices=("1", "2", "4"), default=["1", "2", "4"])
    parser.add_argument("--pools", nargs="+", choices=("clean", "simple_v2", "canary"))
    parser.add_argument("--case", nargs="+", choices=tuple(evaluation_cases()))
    parser.add_argument("--recipe", nargs="+", choices=tuple(TRAINING_RECIPES), default=["clean"])
    args = parser.parse_args(argv)
    for name in ("tech", "tag", "size", "omp", "pools", "case", "recipe"):
        values = getattr(args, name)
        if values and len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates")
    try:
        selected_pools, selected_cases = select_cases(args.pools, None, args.case)
    except ValueError as exc:
        parser.error(str(exc))
    inventory = evaluation_cases()
    suites = {inventory[name]["suite"] for name in selected_cases}
    out: Path = args.outdir
    pools = build_pools()
    pools = {name: [line for line in jobs
                    if line.split()[0] in args.tag and line.split()[1] in args.size
                    and line.split()[2] in args.tech and line.split()[4] in args.omp
                    and line.split()[3] in suites]
             for name, jobs in pools.items() if name in selected_pools}
    pools = {name: [" ".join([parts[0], parts[1] if recipe == "clean" else f"{recipe}_{parts[1]}",
                              *parts[2:]])
                    for recipe in args.recipe for line in jobs for parts in [line.split()]]
             for name, jobs in pools.items()}
    if any(not jobs for jobs in pools.values()):
        parser.error("selection produces an empty pool; include --omp 1")
    # A different selection must not rewrite a running/resumable campaign.
    for name, jobs in pools.items():
        path = out / f"jobs_{name}.txt"
        if path.exists() and path.read_text() != "\n".join(jobs) + "\n":
            parser.error(f"job selection changed: {path}; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name, jobs in pools.items():
        p = out / f"jobs_{name}.txt"
        p.write_text("\n".join(jobs) + "\n")
        print(f"{p}  {len(jobs)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
