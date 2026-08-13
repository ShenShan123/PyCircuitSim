"""Check that generated netlists faithfully preserve their source topology.

The re-design changes geometry, bias and compensation.  It must not change the
circuit: same transistors, same nodes, same polarity, and -- the one that is
easy to break silently -- the same mirror ratios, since ``m`` on each instance
is ``role.m * instance.mult`` and a dropped factor turns a 1:8 mirror into 1:1
without any simulation error.

    python3 tools/verify_port.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from skyparse import parse_netlist          # noqa: E402
from geom_port import parse_generic, parse_params  # noqa: E402

SKY = ROOT.parent / "designs"
_DEV = re.compile(r"^N(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)$", re.I)
_PASSIVE = re.compile(r"^([RC]\S+)\s+(\S+)\s+(\S+)\s+(\S+)", re.I)


def emitted_devices(path: Path) -> Dict[str, Tuple[Tuple[str, ...], int]]:
    """{name: ((d,g,s,b), m)} from a generated TSMC16 netlist."""
    out: Dict[str, Tuple[Tuple[str, ...], int]] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("$")[0].strip()
        m = _DEV.match(line)
        if not m:
            continue
        mult = re.search(r"\bm\s*=\s*(\d+)", m.group(7) or "", re.I)
        out[m.group(1).lower()] = (
            tuple(x.lower() for x in m.groups()[1:5]),
            int(mult.group(1)) if mult else 1,
        )
    return out


def emitted_subcircuits(path: Path) -> Dict[str, Dict[str, Tuple[Tuple[str, ...], int]]]:
    """Return emitted devices by subcircuit, preserving repeated local names."""
    out: Dict[str, Dict[str, Tuple[Tuple[str, ...], int]]] = {}
    current = ""
    for raw in path.read_text().splitlines():
        line = raw.split("$")[0].strip()
        if line.lower().startswith(".subckt"):
            current = line.split()[1].lower()
            out[current] = {}
            continue
        if line.lower().startswith(".ends"):
            current = ""
            continue
        match = _DEV.match(line)
        if not current or not match:
            continue
        mult = re.search(r"\bm\s*=\s*(\d+)", match.group(7) or "", re.I)
        out[current][match.group(1).lower()] = (
            tuple(node.lower() for node in match.groups()[1:5]),
            int(mult.group(1)) if mult else 1,
        )
    return out


def emitted_passives(path: Path) -> Dict[str, Tuple[str, str]]:
    """Return emitted R/C connectivity; values may legitimately be re-sized."""
    out: Dict[str, Tuple[str, str]] = {}
    for raw in path.read_text().splitlines():
        match = _PASSIVE.match(raw.split("$")[0].strip())
        if match:
            out[match.group(1).lower()] = (_norm(match.group(2)),
                                           _norm(match.group(3)))
    return out


def check_amplifier(name: str) -> List[str]:
    problems: List[str] = []
    topo = parse_netlist(SKY / "amplifier" / name / "netlist.spice")
    d = ROOT / "amplifier" / name
    got = emitted_devices(d / "netlist.spice")
    design = json.loads((d / "design.json").read_text())

    if len(got) != len(topo.mos):
        problems.append(f"device count {len(got)} != {len(topo.mos)}")
    for mos in topo.mos:
        key = mos.name.lstrip("xX").lower()
        if key not in got:
            problems.append(f"missing device {mos.name}")
            continue
        nodes, mult = got[key]
        want_nodes = (mos.d.lower(), mos.g.lower(), mos.s.lower(), mos.b.lower())
        if nodes != want_nodes:
            problems.append(f"{mos.name}: nodes {nodes} != {want_nodes}")
        want_m = max(1, int(round(design["roles"][mos.role]["m"] * mos.mult)))
        if mult != want_m:
            problems.append(f"{mos.name}: m={mult} != role.m*{mos.mult}={want_m}")
        model_kind = None
        for line in (d / "netlist.spice").read_text().splitlines():
            mm = _DEV.match(line.split("$")[0].strip())
            if mm and mm.group(1).lower() == key:
                model_kind = mm.group(6)[0].lower()
        if model_kind != mos.kind:
            problems.append(f"{mos.name}: channel type {model_kind} != {mos.kind}")
    return problems


# The reference emitters rename the design's ground net to node 0: ngspice
# rejects `Vgnd gnd 0 0` as a shorted VSRC, so the net cannot simply be tied.
_GROUND_ALIASES = {"gnd", "gnda", "vss", "0"}


def _norm(node: str) -> str:
    return "0" if node.lower() in _GROUND_ALIASES else node.lower()


def check_generic(category: str, name: str) -> List[str]:
    """Node-level check for the categories ported from explicit geometry."""
    problems: List[str] = []
    subs, _ = parse_generic(SKY / category / name / "netlist.spice")
    src = [m for s in subs for m in s.mos]
    d = ROOT / category / name
    if not (d / "netlist.spice").exists():
        return [f"{name}: not built"]
    got = emitted_devices(d / "netlist.spice")
    if len(got) != len(src):
        problems.append(f"device count {len(got)} != {len(src)}")
    for m in src:
        key = m.name.lstrip("xX").lower()
        if key not in got:
            problems.append(f"missing device {m.name}")
            continue
        nodes, _mult = got[key]
        want = tuple(_norm(n) for n in m.nodes)
        if tuple(_norm(n) for n in nodes) != want:
            problems.append(f"{m.name}: nodes {nodes} != {want}")
        model_kind = None
        for line in (d / "netlist.spice").read_text().splitlines():
            match = _DEV.match(line.split("$")[0].strip())
            if match and match.group(1).lower() == key:
                model_kind = match.group(6)[0].lower()
                break
        if model_kind != m.kind:
            problems.append(f"{m.name}: channel type {model_kind} != {m.kind}")
    wanted_passives: Dict[str, Tuple[str, str]] = {}
    for sub in subs:
        for passive in sub.passives:
            wanted_passives[passive.name.lower()] = tuple(
                _norm(node) for node in passive.nodes[:2]
            )
        for inst_name, nodes, line in sub.insts:
            low = line.lower()
            prefix = "c" if "cap_mim" in low else (
                "r" if "res_high_po" in low or "res_generic" in low else ""
            )
            if prefix:
                converted = prefix + inst_name.lstrip("xX")
                wanted_passives[converted.lower()] = tuple(
                    _norm(node) for node in nodes[:2]
                )
    got_passives = emitted_passives(d / "netlist.spice")
    allowed_extra = {"ccomp", "rcomp"} if category == "ldo" and name == "ldo_1" else set()
    for passive, nodes in wanted_passives.items():
        if passive not in got_passives:
            problems.append(f"missing passive {passive}")
        elif got_passives[passive] != nodes:
            problems.append(f"{passive}: nodes {got_passives[passive]} != {nodes}")
    extras = set(got_passives) - set(wanted_passives) - allowed_extra
    if extras:
        problems.append(f"unexpected passives {sorted(extras)}")
    if allowed_extra and (got_passives.get("ccomp") != ("net3", "ncomp")
                          or got_passives.get("rcomp") != ("ncomp", "vout")):
        problems.append("ldo_1 compensation network is miswired")
    return problems


def check_derived_reference(name: str) -> List[str]:
    """The derived three-output reference must preserve its qualified core."""
    d = ROOT / "voltage_reference" / name
    design = json.loads((d / "design.json").read_text())
    source_name = design.get("derived_from")
    if not source_name:
        return check_generic("voltage_reference", name)
    source = ROOT / "voltage_reference" / source_name / "netlist.spice"
    got = emitted_devices(d / "netlist.spice")
    want = emitted_devices(source)
    problems = [] if got == want else ["derived MOS core differs from qualified source"]
    text = (d / "netlist.spice").read_text().lower()
    if "r3top vref2 vref3 1e18" not in text or "r3bot vref3 0 1e18" not in text:
        problems.append("missing high-impedance vref3 divider")
    return problems


def check_charge_pump(name: str) -> List[str]:
    """Check only the pump blocks reachable from the charge-pump testbench."""
    source_dir = SKY / "charge_pump" / name
    params = parse_params(source_dir / "param.spice")
    source_subs, _ = parse_generic(source_dir / "netlist.spice", params)
    keep = {"pll_chargepump", "pll_quench_v33"}
    source = {sub.name.lower(): sub for sub in source_subs
              if sub.name.lower() in keep}
    got = emitted_subcircuits(ROOT / "charge_pump" / name / "netlist.spice")
    problems: List[str] = []
    if set(source) != keep:
        problems.append(f"source blocks {sorted(source)} != {sorted(keep)}")
    if set(got) != keep:
        problems.append(f"emitted blocks {sorted(got)} != {sorted(keep)}")
    for block in sorted(keep & set(source) & set(got)):
        wanted = source[block]
        devices = got[block]
        if len(devices) != len(wanted.mos):
            problems.append(f"{block}: device count {len(devices)} != {len(wanted.mos)}")
        for mos in wanted.mos:
            key = mos.name.lstrip("xX").lower()
            if key not in devices:
                problems.append(f"{block}: missing device {mos.name}")
                continue
            nodes, _ = devices[key]
            want_nodes = tuple(_norm(node) for node in mos.nodes)
            if tuple(_norm(node) for node in nodes) != want_nodes:
                problems.append(f"{block}/{mos.name}: nodes {nodes} != {want_nodes}")
            text = (ROOT / "charge_pump" / name / "netlist.spice").read_text()
            model_kind = None
            in_block = False
            for line in text.splitlines():
                if line.lower().startswith(f".subckt {block}"):
                    in_block = True
                    continue
                if in_block and line.lower().startswith(".ends"):
                    break
                match = _DEV.match(line.split("$")[0].strip()) if in_block else None
                if match and match.group(1).lower() == key:
                    model_kind = match.group(6)[0].lower()
                    break
            if model_kind != mos.kind:
                problems.append(f"{block}/{mos.name}: channel type "
                                f"{model_kind} != {mos.kind}")
    emitted_text = (ROOT / "charge_pump" / name / "netlist.spice").read_text().lower()
    if "xi40 _net0 cp_out quench pb2 vss33 pll_quench_v33" not in emitted_text:
        problems.append("missing PLL_CHARGEPUMP -> PLL_QUENCH_v33 hierarchy")
    return problems


def main() -> None:
    total = failed = 0
    for d in sorted((ROOT / "amplifier").glob("*/")):
        if not (d / "netlist.spice").exists():
            continue
        total += 1
        probs = check_amplifier(d.name)
        if probs:
            failed += 1
            print(f"FAIL amplifier/{d.name}")
            for p in probs[:5]:
                print(f"       {p}")
    for cat in ("sensing_front_end", "voltage_reference", "ldo", "charge_pump"):
        for d in sorted((ROOT / cat).glob("*/")):
            if not (d / "netlist.spice").exists():
                continue
            total += 1
            if cat == "voltage_reference":
                probs = check_derived_reference(d.name)
            elif cat == "charge_pump":
                probs = check_charge_pump(d.name)
            else:
                probs = check_generic(cat, d.name)
            if probs:
                failed += 1
                print(f"FAIL {cat}/{d.name}")
                for p in probs[:5]:
                    print(f"       {p}")
    print(f"\n{total - failed}/{total} designs match their source/qualified core "
          f"(devices, passives, nodes, channel type, mirror ratios)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
