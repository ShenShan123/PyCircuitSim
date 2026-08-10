#!/usr/bin/env python3
"""Validate generated TSMC dimensions and local PyCMG model bindings."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT: Path = Path(__file__).resolve().parent
LIMITS: Dict[str, Tuple[int, int, int]] = {
    "tsmc5": (6, 135, 12),
    "tsmc6": (8, 240, 20),
    "tsmc7": (8, 240, 20),
    "tsmc12": (16, 240, 20),
    "tsmc16": (16, 240, 20),
}
CATEGORIES: Tuple[str, ...] = (
    "amplifier",
    "ldo",
    "sensing_front_end",
    "voltage_reference",
    "charge_pump",
)
MODEL_DEF_RE = re.compile(r"^\s*\.model\s+(\S+)\s+bsimcmg\b", re.I)
MODEL_ALIAS_RE = re.compile(r"^[np][a-z0-9]*_l(\d+)_f(\d+)$", re.I)
MOS_RE = re.compile(
    r"^\s*N\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)(.*)$", re.I
)
MULT_RE = re.compile(r"\bm\s*=\s*(\d+)\b", re.I)


@dataclass
class Stats:
    designs: int = 0
    devices: int = 0
    vectors: int = 0
    models: int = 0


def _geometry_vectors(design: Dict[str, Any]) -> Iterable[Tuple[str, float, int, int]]:
    for name, value in (design.get("geoms") or {}).items():
        if isinstance(value, list) and len(value) == 3:
            yield str(name), float(value[0]), int(value[1]), int(value[2])
    for name, value in (design.get("roles") or {}).items():
        if isinstance(value, dict):
            yield (str(name), float(value["l_nm"]), int(value["nfin"]),
                   int(value["m"]))


def _model_definitions(paths: Iterable[Path]) -> Dict[str, Path]:
    definitions: Dict[str, Path] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            match = MODEL_DEF_RE.match(line)
            if match:
                definitions[match.group(1).lower()] = path
    return definitions


def verify_tree(tree: Path) -> Tuple[Stats, List[str]]:
    tech = tree.name.removeprefix("designs_").lower()
    if tech not in LIMITS:
        return Stats(), [f"{tree.name}: no geometry limits registered"]
    l_min, l_max, nfin_max = LIMITS[tech]
    stats = Stats()
    problems: List[str] = []
    for category in CATEGORIES:
        category_dir = tree / category
        if not category_dir.is_dir():
            continue
        for design_dir in sorted(path for path in category_dir.iterdir()
                                 if path.is_dir()):
            netlist = design_dir / "netlist.spice"
            vector_path = design_dir / "design.json"
            if not netlist.exists() or not vector_path.exists():
                continue
            label = f"{tree.name}/{category}/{design_dir.name}"
            stats.designs += 1
            design: Dict[str, Any] = json.loads(vector_path.read_text())
            for name, length_nm, nfin, mult in _geometry_vectors(design):
                stats.vectors += 1
                if not l_min <= length_nm <= l_max:
                    problems.append(
                        f"{label}/{name}: L={length_nm:g}nm outside "
                        f"[{l_min},{l_max}]nm"
                    )
                if not 1 <= nfin <= nfin_max:
                    problems.append(
                        f"{label}/{name}: NFIN={nfin} outside [1,{nfin_max}]"
                    )
                if mult < 1:
                    problems.append(f"{label}/{name}: m={mult} is not positive")

            definitions = _model_definitions(design_dir.glob("*_models.spice"))
            stats.models += len(definitions)
            for model in definitions:
                alias = MODEL_ALIAS_RE.match(model)
                if not alias:
                    problems.append(f"{label}: malformed model alias {model}")
                    continue
                length_nm, nfin = int(alias.group(1)), int(alias.group(2))
                if not l_min <= length_nm <= l_max or not 1 <= nfin <= nfin_max:
                    problems.append(
                        f"{label}: model {model} is outside the {tech} envelope"
                    )

            for line in netlist.read_text().splitlines():
                match = MOS_RE.match(line.split("$")[0])
                if not match:
                    continue
                stats.devices += 1
                model = match.group(1).lower()
                if model not in definitions:
                    problems.append(f"{label}: undefined model {model}")
                mult = MULT_RE.search(match.group(2) or "")
                if not mult or int(mult.group(1)) < 1:
                    problems.append(f"{label}: missing/nonpositive MOS multiplicity")
    return stats, problems


def main() -> None:
    trees = [Path(arg).resolve() for arg in sys.argv[1:]] or sorted(
        ROOT.glob("designs_tsmc*")
    )
    all_problems: List[str] = []
    total = Stats()
    for tree in trees:
        stats, problems = verify_tree(tree)
        all_problems.extend(problems)
        total.designs += stats.designs
        total.devices += stats.devices
        total.vectors += stats.vectors
        total.models += stats.models
        print(
            f"{tree.name}: {stats.designs} designs, {stats.devices} MOS, "
            f"{stats.vectors} sizing vectors, {stats.models} model aliases, "
            f"{len(problems)} problems"
        )
    for problem in all_problems:
        print(f"FAIL {problem}")
    print(
        f"\nchecked {total.designs} designs / {total.devices} MOS / "
        f"{total.vectors} vectors / {total.models} model aliases"
    )
    raise SystemExit(1 if all_problems else 0)


if __name__ == "__main__":
    main()
