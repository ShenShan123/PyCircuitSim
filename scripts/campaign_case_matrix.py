#!/usr/bin/env python3
"""Render a technology x test case x model-type accuracy matrix from raw logs.

Structured as L0-L4 level evaluations (the ``circuit_templates/`` tiers), then
the flat/nested hierarchy fixture, which sits outside L0-L4. Each level opens
with a summary (technology x model: median NRMSE over the level's converged
rows and the converged share), followed by one table per technology whose
rows are the level's test cases and whose columns are the eight model types
(DirectNet-Full and BSIM-AR-Full at small/medium/large/xl). A case cell is the
median NRMSE (%) over the case's converged rows, suffixed with ``kE`` for
ERROR rows (no physical fixed point; they keep their denominator slot) and
``kF`` for failed qualification rows.

Reads only ``===PYCIRCUITSIM_GATE_RESULT`` markers from OMP=1 verdict logs,
``<pool>/<tag>/<size>/<tech>/<suite>.omp1.log``. Fails closed on a missing or
unfinished log unless ``--partial``.

Usage:
  python scripts/campaign_case_matrix.py --pool <arm>/clean --pool <arm>/simple_v2 \
      --pool <arm>/canary --title "V7.7.6" --out docs/accuracy/<name>.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common.simple_circuit_catalog import cases  # noqa: E402

MARK = "===PYCIRCUITSIM_GATE_RESULT "
TECHS = ("TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16")
MODELS = tuple((tag, size) for tag in ("dnf", "tff")
               for size in ("small", "medium", "large", "xl"))
HEADER = {"dnf": "DN", "tff": "AR"}
SIZE = {"small": "S", "medium": "M", "large": "L", "xl": "XL"}
# Non-catalog suites: the template tier each exercises (V7.7.5 round-3 report,
# docs/accuracy/v775-round3.md at bde2c11).
SUITE_TIER = {
    "verify_nn_multi_tech_dc": "L0", "verify_nn_lifted_source_dc": "L0",
    "verify_device_integrity": "L0", "verify_terminal_integrity": "L0",
    "verify_nn_ac": "L1", "verify_nn_multi_tech_tran": "L2",
    "verify_circuit_opamp_ac": "L3", "verify_nn_subckt": "hier",
}
LEVELS = (
    ("L0", "devices: one device biased from ideal sources"),
    ("L1", "primitives: a device with a passive load"),
    ("L2", "stages: coupled devices, ideal gate rails"),
    ("L3", "blocks: the model sets its own operating point"),
    ("L4", "systems: closed negative-feedback loops"),
    ("hier", "hierarchy: flat vs nested netlist, outside L0-L4"),
)
Cell = Tuple[str, str, str, str]  # tag, size, tech, suite


def suite_tiers() -> Dict[str, str]:
    tiers = {case.campaign_suite: case.tier.split("_")[0] for case in cases()}
    tiers.update(SUITE_TIER)
    return tiers


# Suites named for their circuit rather than their script.
SUITE_LABEL = {
    "verify_nn_multi_tech_dc": "device_dc",
    "verify_nn_lifted_source_dc": "lifted_source_canary",
    "verify_nn_ac": "common_source_ac",
    "verify_nn_multi_tech_tran": "inverter_vtc_tran",
    "verify_circuit_opamp_ac": "opamp_miller_ac",
    "verify_nn_subckt": "buffer_flat_vs_nested",
}


def label(suite: str) -> str:
    if suite in SUITE_LABEL:
        return SUITE_LABEL[suite]
    return (suite.replace("verify_circuit_topologies__", "")
            .replace("verify_circuit_", "").replace("verify_", ""))


def parse(log: Path) -> Optional[List[dict]]:
    """GateResult rows of a finished log, or None while it has no verdict."""
    text = log.read_text(errors="replace")
    if "===V710_DONE rc=" not in text:
        return None
    return [json.loads(line[len(MARK):-3]) for line in text.splitlines()
            if line.startswith(MARK) and line.endswith("===")]


def collect(pools: List[Path], partial: bool) -> Dict[Cell, List[dict]]:
    cells: Dict[Cell, List[dict]] = {}
    problems = []
    for pool in pools:
        manifest = json.loads((pool / "campaign_manifest.json").read_text())
        jobs = manifest["job_count"]
        seen = 0
        for log in sorted(pool.glob("*/*/*/*.omp*.log")):
            tag, size, tech = log.parts[-4], log.parts[-3], log.parts[-2]
            suite, omp = log.name[:-4].rsplit(".omp", 1)
            seen += 1
            if omp != "1":
                continue
            rows = parse(log)
            if rows is None:
                problems.append(f"no verdict: {log}")
                continue
            cells[(tag, size, tech.upper(), suite)] = rows
        if seen != jobs:
            problems.append(f"{pool}: {seen} logs for {jobs} jobs")
    if problems and not partial:
        sys.exit("incomplete campaign:\n" + "\n".join(problems[:20]))
    return cells


def nrmse(row: dict) -> Optional[float]:
    metrics = row.get("metrics") or {}
    for key in ("nrmse_pct", "worst_nrmse_pct"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def cell_text(rows: Optional[List[dict]]) -> str:
    if rows is None:
        return "…"
    errors = sum(row.get("status") == "error" for row in rows)
    fails = sum(row.get("status") == "fail" for row in rows)
    values = [v for row in rows if row.get("status") != "error"
              for v in [nrmse(row)] if v is not None]
    if values:
        median = statistics.median(values)
        text = f"{median:.2f}" if median < 10 else f"{median:.0f}"
    else:
        text = "E" if errors == len(rows) and rows else "—"
    marks = [f"{errors}E" if errors and values else "",
             f"{fails}F" if fails else ""]
    return " ".join(part for part in (text, *marks) if part)


def level_text(rows: List[dict]) -> str:
    """Median NRMSE over a level's converged rows, then the converged share."""
    if not rows:
        return "…"
    converged = [row for row in rows if row.get("status") != "error"]
    values = [v for row in converged for v in [nrmse(row)] if v is not None]
    share = f"{100.0 * len(converged) / len(rows):.0f}%"
    if not values:
        return f"— · {share}"
    median = statistics.median(values)
    return (f"{median:.2f}" if median < 10 else f"{median:.0f}") + f" · {share}"


def render(cells: Dict[Cell, List[dict]], title: str) -> str:
    tiers = suite_tiers()
    models = " | ".join(f"{HEADER[t]}-{SIZE[s]}" for t, s in MODELS)
    rule = "---:|" * len(MODELS)
    out = [f"# {title} — L0–L4 accuracy by technology, test case and model", "",
           "Level summary cell: median NRMSE (%) over the level's converged rows "
           "· converged share of its rows. Case cell: median NRMSE (%) over the "
           "case's converged rows; `kE` = ERROR rows (no physical fixed point, "
           "kept in the denominator); `kF` = failed qualification rows; `E` = "
           "every row ERROR; `—` = no trace metric. DN = DirectNet-Full (L75), "
           "AR = BSIM-AR-Full (L76); S/M/L/XL = model size. NRMSE is "
           "range-normalized: AC cases with a nearly flat reference "
           "(common-mode, supply) read high even when the gain error is small. "
           "Levels follow `circuit_templates/`. Generated by "
           "`scripts/campaign_case_matrix.py`.", ""]
    for level, description in LEVELS:
        suites = sorted((s for s in {cell[3] for cell in cells}
                         if tiers.get(s) == level), key=label)
        if not suites:
            continue
        out += [f"## {level} — {description}", "",
                f"| technology | {models} |", "|---|" + rule]
        for tech in (*TECHS, "all"):
            row = []
            for tag, size in MODELS:
                pooled = [r for (t, z, k, s), rows in cells.items()
                          if (t, z) == (tag, size) and s in suites
                          and (tech == "all" or k == tech) for r in rows]
                row.append(level_text(pooled))
            name = "**all**" if tech == "all" else tech
            out.append(f"| {name} | " + " | ".join(row) + " |")
        out.append("")
        for tech in TECHS:
            out += [f"### {level} · {tech}", "",
                    f"| test case | {models} |", "|---|" + rule]
            for suite in suites:
                row = [cell_text(cells.get((tag, size, tech, suite)))
                       for tag, size in MODELS]
                out.append(f"| {label(suite)} | " + " | ".join(row) + " |")
            out.append("")
    unplaced = sorted({cell[3] for cell in cells}
                      - {s for s in tiers if tiers[s] in dict(LEVELS)})
    if unplaced:
        sys.exit(f"suites without a level: {unplaced}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", action="append", required=True, type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    cells = collect(args.pool, args.partial)
    args.out.write_text(render(cells, args.title) + "\n")
    print(f"{len(cells)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
