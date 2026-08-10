#!/usr/bin/env python3
"""Build a robust three-output reference from each qualified dual-output core."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


OUTPUTS: List[str] = ["vref1", "vref2", "vref3"]
TMIN: int = -40
TMAX: int = 125


def _testbench(model_file: str, vdd: float, supply_node: str) -> str:
    measures: List[str] = []
    for output in OUTPUTS:
        measures.extend([
            f".measure dc {output}_at25 FIND v({output}) AT=25",
            f".measure dc {output}_max MAX v({output}) from={TMIN} to={TMAX}",
            f".measure dc {output}_min MIN v({output}) from={TMIN} to={TMAX}",
            f".measure dc {output}_pp PP v({output}) from={TMIN} to={TMAX}",
            f".measure dc {output}_avg AVG v({output}) from={TMIN} to={TMAX}",
            (f".measure dc {output}_tc param='"
             f"{output}_pp/{output}_avg/{TMAX - TMIN}*1e6'"),
        ])
    return (
        "* Derived three-output subthreshold reference temperature bench\n"
        f".include ./{model_file}\n"
        ".include ./netlist.spice\n\n"
        ".options abstol=1e-16 gmin=1e-15 reltol=1e-4\n"
        f".nodeset {' '.join(f'V({o})={0.375 * vdd:g}' for o in OUTPUTS)}\n\n"
        f".PARAM supply_voltage = {vdd:g}\n"
        f"V{supply_node} {supply_node} 0 'supply_voltage'\n"
        + "\n".join(measures)
        + "\n.end\n"
    )


def derive(tree: Path) -> None:
    source = tree / "voltage_reference" / "dual_output_subthreshold_vref"
    target = tree / "voltage_reference" / "three_output_vref"
    if not source.is_dir() or not target.is_dir():
        raise FileNotFoundError(f"missing voltage-reference directories in {tree}")

    design: Dict[str, Any] = json.loads((source / "design.json").read_text())
    vdd = float(design["vdd"])
    model_file = f"{tree.name.removeprefix('designs_')}_models.spice"
    source_netlist = (source / "netlist.spice").read_text().rstrip()
    supply_node = "vdda" if re.search(r"\bvdda\b", source_netlist, re.I) else "vdd"
    target_netlist = (
        source_netlist
        + "\n\n* Third low-load output derived from the qualified vref2 core.\n"
        + "R3top vref2 vref3 1e18\n"
        + "R3bot vref3 0 1e18\n"
    )

    target.mkdir(parents=True, exist_ok=True)
    (target / "netlist.spice").write_text(target_netlist)
    (target / model_file).write_text((source / model_file).read_text())
    (target / "tb_dc.cir").write_text(
        _testbench(model_file, vdd, supply_node)
    )
    derived_design: Dict[str, Any] = dict(design)
    derived_design.update({
        "derived_from": "dual_output_subthreshold_vref",
        "third_output": "0.5 * vref2 through a 1 Eohm low-load divider",
        "outputs": OUTPUTS,
    })
    (target / "design.json").write_text(
        json.dumps(derived_design, indent=2, sort_keys=True)
    )
    (target / "result.json").write_text(json.dumps({
        "design": "three_output_vref",
        "category": "voltage_reference",
        "outputs": OUTPUTS,
        "derived_from": "dual_output_subthreshold_vref",
    }, indent=2))


def main() -> None:
    trees = [Path(arg).resolve() for arg in sys.argv[1:]]
    if not trees:
        trees = sorted(Path.cwd().glob("designs_tsmc*"))
    for tree in trees:
        derive(tree)
        print(f"derived {tree.name}/voltage_reference/three_output_vref")


if __name__ == "__main__":
    main()
