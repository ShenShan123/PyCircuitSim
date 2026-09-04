#!/usr/bin/env python3
"""Compare NN four-terminal currents and capacitances with LEVEL=72 OSDI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYCIRCUITSIM_TORCH_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH,
    BENCH_TECHS,
    RESULTS_BASE,
)
from tests.common.gate_result import result_exit_code  # noqa: E402
from tests.common.simple_circuit_harness import CORNERS, RunSpec  # noqa: E402
from tests.common.terminal_integrity import (  # noqa: E402
    run_terminal_integrity, terminal_corner_applies,
)


def _values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech", default=",".join(BENCH_TECHS))
    parser.add_argument("--device", default="nmos,pmos")
    parser.add_argument("--corner", default="nominal",
                        help="comma-separated corners, or all")
    parser.add_argument("--list", action="store_true",
                        help="list the declared corner matrix without running")
    args = parser.parse_args(argv)

    techs = [value.upper() for value in _values(args.tech)]
    devices = [value.lower() for value in _values(args.device)]
    corners = (list(CORNERS) if args.corner == "all"
               else _values(args.corner))
    unknown_techs = [value for value in techs if value not in BENCH]
    unknown_devices = [value for value in devices
                       if value not in {"nmos", "pmos"}]
    unknown_corners = [value for value in corners if value not in CORNERS]
    if not techs or unknown_techs:
        parser.error(
            f"unknown technologies {unknown_techs or techs}; "
            f"available: {list(BENCH)}"
        )
    if not devices or unknown_devices:
        parser.error(
            f"unknown devices {unknown_devices or devices}; "
            "available: ['nmos', 'pmos']"
        )
    if not corners or unknown_corners:
        parser.error(
            f"unknown corners {unknown_corners or corners}; "
            f"available: {list(CORNERS)}"
        )
    if (len(set(techs)) != len(techs) or len(set(devices)) != len(devices)
            or len(set(corners)) != len(corners)):
        parser.error(
            "technology, device, and corner selections must not contain "
            "duplicates"
        )
    if args.list:
        print("terminal-integrity corners: " + ",".join(CORNERS))
        return 0
    try:
        run_spec = RunSpec.from_environment()
        run_spec.validate_checkpoint_pins(Path(os.environ.get(
            "BSIMAR_CHECKPOINT_DIR",
            PROJECT_ROOT / "external_compact_models" / "neural_network"
            / "checkpoints",
        )))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    root = (
        RESULTS_BASE / "terminal-integrity"
        / f"level-{run_spec.model_level}"
    )
    results = []
    for tech in techs:
        for corner_name in corners:
            corner = CORNERS[corner_name]
            skipped = [device for device in devices
                       if not terminal_corner_applies(
                           BENCH[tech], device, corner)]
            if skipped:
                print(f"{tech}/{corner_name} NOT-APPLICABLE: "
                      + ", ".join(skipped))
            tech_results = run_terminal_integrity(
                BENCH[tech], devices, root / tech / corner_name, run_spec,
                corner,
            )
            results.extend(tech_results)
            for result in tech_results:
                print(result.marker())
                print(
                    f"{result.tech}/{result.corner}/{result.analysis}: "
                    f"{result.status.upper()} "
                    f"{result.error or result.metrics.get('nrmse_pct', '')}"
                )

    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_results.json").write_text(json.dumps(
        [result.payload() for result in results],
        indent=2,
        sort_keys=True,
    ) + "\n")
    return result_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
