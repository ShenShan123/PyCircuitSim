"""Port and re-size the AnalogGym low-dropout regulators for TSMC FinFET.

The five LDOs do not share a parameter convention -- Basic_LDO tags its devices
with AnalogGym roles, the others name a variable per device -- and they do not
share a port list either.  What they do share is a structure: an error amplifier,
a pass device, and a feedback path.  Devices are grouped by whatever names their
geometry (role tag, per-device variable, or the literal W/L), so matched devices
stay matched under either convention.

TSMC16 here ships no passive models, so the mim capacitors and poly resistors
become ideal R and C.  Their values are design variables, which they effectively
were already: the feedback divider sets the output and the compensation network
sets the loop.

Supply moves 2.0 V down to the tech's core rail and the reference with it; the
5-55 mA load range is AnalogGym's and is kept, so the pass device is sized to
carry it at a quarter-rail of dropout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from geom_port import GMos, GSubckt, parse_generic, parse_value
import pycmg_lib
from pycmg_lib import MODELS_FILE, NFIN_MAX, TECH, ModelLibrary, scale_l

VDD = pycmg_lib.VDD
VREF = 0.375 * VDD    # error-amp reference; output = VREF * divider ratio
DIV_RATIO = 2.0       # r_top / r_bot + 1  -> Vout = 0.75*VDD, VDD/4 of dropout
VOUT_NOM = VREF * DIV_RATIO
ILOAD_MIN = 5e-3
ILOAD_MAX = 55e-3

# ldo_1 is a three-stage loop.  Its source topology has no explicit
# compensation, which leaves the high-load loop with single-digit phase
# margin on several nodes.  These per-node series-RC values were swept with
# the real PyCMG modelcards and keep both load extremes stable.
_LDO1_COMP: Dict[str, Tuple[float, float]] = {
    "TSMC5": (24.0e-12, 5e3),
    # TSMC6/7 re-swept 2026-08: (12p, 25k) left the high-load margin at
    # 46.0 deg -- technically past the 45 deg gate but with no headroom.
    # (24p, 15k) balances the extremes at ~72 deg on both load corners.
    "TSMC6": (24.0e-12, 15e3),
    "TSMC7": (24.0e-12, 15e3),
    "TSMC12": (12.0e-12, 25e3),
    "TSMC16": (9.0e-12, 25e3),
}

# Port-name heuristics.  Every LDO subckt names its supply, ground and regulated
# output recognisably; the rest are the amplifier inputs and bias pins.
_GND = re.compile(r"^(gnd|vss|gnda)$", re.I)
_VDD = re.compile(r"^(vdd|vdda)$", re.I)
_OUT = re.compile(r"^(vout|vreg)$", re.I)
_FB = re.compile(r"^vfb$", re.I)
_INN = re.compile(r"^(vinn|vref)$", re.I)
_INP = re.compile(r"^vinp$", re.I)
_IB = re.compile(r"^(ib|vb|vb1)$", re.I)


@dataclass
class LdoDesign:
    vdd: float = VDD
    vref: float = VREF
    vt: str = "svt"
    # group key -> (L[nm], NFIN, m)
    geoms: Dict[str, Tuple[float, int, int]] = field(default_factory=dict)
    # passive name -> value
    passives: Dict[str, float] = field(default_factory=dict)
    # device name (leading X stripped, upper case) -> Vt flavor, for the few
    # devices whose headroom needs a different threshold than the design's
    # default flavor (e.g. Basic_LDO's pass device and its gate-drive
    # follower, which set the dropout floor).
    vt_overrides: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "vdd": self.vdd, "vref": self.vref, "vt": self.vt,
            "vout_nominal": VOUT_NOM,
            "iload_min": ILOAD_MIN, "iload_max": ILOAD_MAX,
            "geoms": {k: list(v) for k, v in self.geoms.items()},
            "passives": self.passives,
            "vt_overrides": self.vt_overrides,
        }, indent=2, sort_keys=True)


def is_pass_device(m: GMos, out_nodes: List[str]) -> bool:
    """The pass device is the PMOS whose drain is the regulated output."""
    return m.kind == "p" and m.nodes[0].lower() in out_nodes


def initial_design(subs: List[GSubckt], out_nodes: List[str]) -> LdoDesign:
    """Seed geometry: a wide short pass device, long-channel everything else."""
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for sub in subs:
        for m in sub.mos:
            if m.group in geoms:
                continue
            if is_pass_device(m, out_nodes):
                # 55 mA at a quarter-rail dropout needs a few thousand fins.
                geoms[m.group] = (scale_l(20), NFIN_MAX, 1500)
            else:
                geoms[m.group] = (scale_l(120), 4, 8)
    return LdoDesign(geoms=geoms, passives={})


# sky130 mim is ~2 fF/um^2; a W=10 L=10 instance is ~200 fF per MF.
_MIM_FF_PER_UM2 = 2e-15


def _passive_defaults(subs: List[GSubckt]) -> Dict[str, float]:
    """Starting values for what used to be mim caps and poly resistors."""
    out: Dict[str, float] = {}
    for sub in subs:
        for p in sub.passives:
            v = parse_value(p.value)
            out[p.name] = v if v is not None else 100e3
        for name, nodes, line in sub.insts:
            low = line.lower()
            if "cap_mim" in low:
                out[name] = 2e-12
            elif "res_high_po" in low or "res_generic" in low:
                out[name] = 100e3
    return out


def emit(out_dir: Path, subs: List[GSubckt], top_lines: List[str],
         design: LdoDesign, sub_name: str, ports: List[str],
         out_nodes: List[str]) -> None:
    """Write the TSMC16 netlist, model library and design vector."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = ModelLibrary()
    lines = [
        f"* {sub_name} -- AnalogGym LDO on {TECH} BSIM-CMG (LEVEL=72)",
        f"* Topology is the shipped one.  Geometry is re-designed for a {VDD:g} V",
        "* rail, and the mim caps / poly resistors become ideal C and R because",
        f"* this {TECH} view carries transistor models only.",
        f".subckt {sub_name} {' '.join(ports)}",
    ]
    if sub_name.lower() == "ldo_1":
        ccomp, rcomp = _LDO1_COMP[TECH.upper()]
        design.passives["ccomp"] = ccomp
        design.passives["rcomp"] = rcomp
    for sub in subs:
        for m in sub.mos:
            l_nm, nfin, mult = design.geoms[m.group]
            flavor = design.vt_overrides.get(
                m.name.lstrip("xX").upper(), design.vt)
            model = lib.model_name(m.kind, flavor, l_nm * 1e-9, nfin)
            tag = " $ pass device" if is_pass_device(m, out_nodes) else ""
            lines.append(f"N{m.name.lstrip('xX')} {' '.join(m.nodes)} "
                         f"{model} m={mult}{tag}")
        for p in sub.passives:
            val = design.passives.get(p.name)
            lines.append(f"{p.name} {' '.join(p.nodes)} "
                         f"{val:.6g}" if val is not None
                         else f"{p.name} {' '.join(p.nodes)} {p.value}")
        for name, nodes, line in sub.insts:
            low = line.lower()
            val = design.passives.get(name)
            if "cap_mim" in low and val is not None:
                lines.append(f"C{name.lstrip('xX')} {nodes[0]} {nodes[1]} "
                             f"{val:.6g}   $ was {name} (mim)")
            elif ("res_high_po" in low or "res_generic" in low) and val is not None:
                lines.append(f"R{name.lstrip('xX')} {nodes[0]} {nodes[1]} "
                             f"{val:.6g}   $ was {name} (poly)")
            else:
                lines.append(f"* dropped (no {TECH} equivalent): {line}")
    if sub_name.lower() == "ldo_1":
        lines.extend([
            f"Ccomp net3 ncomp {design.passives['ccomp']:.6g}",
            f"Rcomp ncomp vout {design.passives['rcomp']:.6g}",
        ])
    lines.append(f".ends {sub_name}")

    lib.write(out_dir / MODELS_FILE)
    (out_dir / "netlist.spice").write_text("\n".join(lines) + "\n")
    (out_dir / "design.json").write_text(design.to_json())
