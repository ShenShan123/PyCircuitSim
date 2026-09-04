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
from tests.common.simple_circuit_harness import RunSpec  # noqa: E402
from tests.common.terminal_integrity import run_terminal_integrity  # noqa: E402


def _values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tech", default=",".join(BENCH_TECHS))
    parser.add_argument("--device", default="nmos,pmos")
    args = parser.parse_args(argv)

    techs = [value.upper() for value in _values(args.tech)]
    devices = [value.lower() for value in _values(args.device)]
    unknown_techs = [value for value in techs if value not in BENCH]
    unknown_devices = [value for value in devices
                       if value not in {"nmos", "pmos"}]
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
    if len(set(techs)) != len(techs) or len(set(devices)) != len(devices):
        parser.error("technology and device selections must not contain duplicates")
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
        tech_results = run_terminal_integrity(
            BENCH[tech], devices, root / tech, run_spec,
        )
        results.extend(tech_results)
        for result in tech_results:
            print(result.marker())
            print(
                f"{result.tech}/{result.analysis}: {result.status.upper()} "
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
