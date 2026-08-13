"""Port and re-size the AnalogGym sensing front ends (PTAT sensors) for TSMC.

Every one of these is a short stack of MOSFETs biased in weak inversion; the
output voltage is a sum of gate-source differences between devices run at
different current densities, so the *ratio* of W/L between the stacked devices
is the whole design and the absolute sizes barely matter.

That ratio does not survive the process change intact.  Several of the sources
use a 20 um channel as a high-impedance element, and the FinFET L bins stop far
short of that --
clamping L there while holding W/L would ask for a 2.6 nm wide device, a
thirtieth of a single fin.  So these are re-designed rather than transcribed:
the topology (which device is diode-connected, what stacks on what) is kept
exactly, and the per-group geometry is re-chosen so the stack lands back in
weak inversion and produces a monotonic PTAT slope on the tech's core rail.

Devices that share a geometry in the source are grouped and stay matched.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from geom_port import GMos, GSubckt, parse_generic
from meas import SimError, run_deck
import pycmg_lib
from pycmg_lib import (L_MAX_NM, MODELS_FILE, TECH, ModelLibrary, W_PER_FIN,
                       snap_l, snap_nfin)

VDD_DEFAULT = pycmg_lib.VDD


def group_key(m: GMos) -> str:
    """Devices with the same source geometry are one matched group."""
    return f"{m.kind}_w{m.w:.4g}_l{m.l:.4g}_x{m.mult:g}"


def groups_of(mos: List[GMos]) -> List[str]:
    seen: List[str] = []
    for m in mos:
        k = group_key(m)
        if k not in seen:
            seen.append(k)
    return seen


@dataclass
class SfeDesign:
    """Design vector: one (L, NFIN, m) per matched group, plus the supply."""
    vdd: float = VDD_DEFAULT
    geoms: Dict[str, Tuple[float, int, int]] = field(default_factory=dict)
    vt: str = "svt"
    # Optional per-group Vt override; empty means the global vt everywhere.
    # A PTAT output is a sum of Vgs differences, so a Vt *difference* between
    # groups is a designable level/slope term of its own.
    vts: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"vdd": self.vdd, "vt": self.vt, "vts": self.vts,
                           "geoms": {k: list(v) for k, v in self.geoms.items()}},
                          indent=2, sort_keys=True)


def initial_design(mos: List[GMos]) -> SfeDesign:
    """Seed each group from the source W/L, ranked so ratios are preserved.

    The absolute widths cannot carry over, but the *ordering* and rough spread of
    W/L can: groups are placed on a geometric ladder of fin counts spanning the
    same number of decades the source used, capped at what one model bin allows.
    """
    keys = groups_of(mos)
    by_key: Dict[str, GMos] = {}
    for m in mos:
        by_key.setdefault(group_key(m), m)

    ratios = [by_key[k].w / by_key[k].l * by_key[k].mult for k in keys]
    lo, hi = min(ratios), max(ratios)
    span = math.log10(hi / lo) if hi > lo else 0.0
    # Compress the source's W/L spread into at most 2 decades of m.
    squeeze = min(1.0, 2.0 / span) if span > 0 else 1.0

    geoms: Dict[str, Tuple[float, int, int]] = {}
    for k, r in zip(keys, ratios):
        decades = math.log10(r / lo) * squeeze if lo > 0 else 0.0
        m_mult = max(1, int(round(2 * (10 ** decades))))
        geoms[k] = (float(L_MAX_NM), 4, min(4000, m_mult))
    return SfeDesign(vdd=VDD_DEFAULT, geoms=geoms)


def emit(out_dir: Path, subs: List[GSubckt], design: SfeDesign,
         ports: List[str], sub_name: str,
         key_fn: Callable[[GMos], str] = group_key) -> None:
    """Write netlist + model library + DC temperature bench.

    *key_fn* maps a device to its geometry key.  The default groups devices
    that share a source geometry; the polish pass keys per device instead,
    because AnalogGym's sensor templates ship every device at the same
    placeholder W/L and size each one independently.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = ModelLibrary()

    lines = [f"* {sub_name} -- AnalogGym sensing front end on {TECH} BSIM-CMG",
             f"* Topology is the shipped one; geometry is re-designed for "
             f"{design.vdd:g} V.",
             f".subckt {sub_name} {' '.join(ports)}"]
    for sub in subs:
        for mos in sub.mos:
            key = key_fn(mos)
            l_nm, nfin, mult = design.geoms[key]
            model = lib.model_name(mos.kind, design.vts.get(key, design.vt),
                                   l_nm * 1e-9, nfin)
            lines.append(f"N{mos.name.lstrip('xX')} {' '.join(mos.nodes)} "
                         f"{model} m={mult}")
        for p in sub.passives:
            lines.append(f"{p.name} {' '.join(p.nodes)} {p.value}")
    lines.append(f".ends {sub_name}")

    lib.write(out_dir / MODELS_FILE)
    (out_dir / "netlist.spice").write_text("\n".join(lines) + "\n")
    (out_dir / "design.json").write_text(design.to_json())

    gnd, vdd, vout = ports_roles(ports)
    order = " ".join({gnd: "0", vdd: "vdd", vout: "vout"}[p] for p in ports)
    (out_dir / "tb_dc.cir").write_text(f"""\
* {sub_name} -- AnalogGym sensing front end DC/temperature bench on {TECH}
* Measurements are the shipped TB_2T_sensor_core set; supply is the {TECH} rail.
.include ./{MODELS_FILE}
.include ./netlist.spice

.PARAM supply_voltage = {design.vdd:g}
Vvdd vdd 0 'supply_voltage'
x0 {order} {sub_name}
c0 vout 0 0.2p

.measure dc vout0   FIND V(vout) AT=0
.measure dc vout25  FIND V(vout) AT=25
.measure dc vout50  FIND V(vout) AT=50
.measure dc vout75  FIND V(vout) AT=75
.measure dc vout100 FIND V(vout) AT=100
.measure dc lsb_25_75C param='(vout75-vout25)/50'
.measure dc maxval MAX V(vout) from=-20 to=120
.measure dc minval MIN V(vout) from=-20 to=120
.measure dc ppval  PP  V(vout) from=-20 to=120
.end
""")


def ports_roles(ports: List[str]) -> Tuple[str, str, str]:
    """Identify (gnd, vdd, out) among a 3-port sensor interface."""
    gnd = next((p for p in ports if "gnd" in p.lower() or p.lower() == "vss"), None)
    vdd = next((p for p in ports if "vdd" in p.lower()), None)
    rest = [p for p in ports if p not in (gnd, vdd)]
    if gnd is None or vdd is None or len(rest) != 1:
        raise ValueError(f"cannot classify sensor ports: {ports}")
    return gnd, vdd, rest[0]


DC_CONTROL = "dc temp -20 120 0.5"
DC_CONTROL_FAST = "dc temp -20 120 5"

# ---------------------------------------------------------------------------
# Full-sweep verdict data
# ---------------------------------------------------------------------------
# The five .meas samples (0/25/50/75/100 C) cannot see an operating-branch
# staircase whose jump sits between samples, so the verdict is computed over
# EVERY solved sweep point: the runner appends a wrdata dump of v(vout) to
# the DC control and derives the smoothness metrics below.
#
# Thresholds (staircase killers, not new performance asks):
# * SLOPE_FLOOR_LOCAL: minimum local slope anywhere in 25..75 C.  A genuine
#   weak-inversion PTAT keeps the same order of slope across the window; a
#   branch-jump staircase sits orders of magnitude below it between jumps.
#   One third of the 0.3 mV/C global sensitivity floor tolerates curvature
#   while killing flat treads.
# * STEP_FRACTION_MAX: no single sweep step may carry more than half of the
#   total 25..75 C rise.  A continuous sensor at 0.5 C resolution puts ~1 %
#   of the rise in each step, so 50x concentration is a discontinuity, with
#   generous headroom for curvature.
# * Pseudo-transient fallbacks: a DC point ngspice could only solve through
#   the transient-op fallback is a suspect operating point.  The first point
#   of a cold sweep legitimately needs op-finding aid (the accepted
#   continuation ladder), so small counts pass silently; a design that needs
#   it across the sweep (front_end_25_6T pattern) is flagged in errors.
SLOPE_FLOOR_LOCAL = 1e-4        # V/C, over 25..75 C
STEP_FRACTION_MAX = 0.5         # single-step share of the 25..75 C rise
FALLBACK_FLAG_MIN = 3           # flag when fallbacks exceed both of these
FALLBACK_FLAG_FRACTION = 0.02
_FALLBACK_RE = re.compile(r"Transient op started")


def read_dc_sweep(path: Path) -> Tuple[List[float], List[float]]:
    """Parse a ``wrdata temp v(vout)`` dump into (temps, vouts)."""
    temps: List[float] = []
    vouts: List[float] = []
    if not path.exists():
        return temps, vouts
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t, v = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if math.isfinite(t) and math.isfinite(v):
            temps.append(t)
            vouts.append(v)
    return temps, vouts


def sweep_metrics(temps: List[float],
                  vouts: List[float]) -> Dict[str, Optional[float]]:
    """Smoothness metrics over the full solved sweep.

    ``mono_violations`` counts non-increasing adjacent pairs in 0..100 C;
    ``min_slope_25_75c`` / ``max_step_frac_25_75c`` implement the staircase
    gates documented above.  All ``None`` when the sweep is unusable.
    """
    out: Dict[str, Optional[float]] = {
        "sweep_points": None, "mono_violations": None,
        "min_slope_25_75c": None, "max_step_frac_25_75c": None,
    }
    if len(temps) < 3:
        return out
    out["sweep_points"] = float(len(temps))
    window = [(t, v) for t, v in zip(temps, vouts) if -1e-9 <= t <= 100.0 + 1e-9]
    if len(window) >= 2:
        out["mono_violations"] = float(sum(
            1 for (_, v0), (_, v1) in zip(window, window[1:]) if v1 <= v0))
    seg = [(t, v) for t, v in zip(temps, vouts)
           if 25.0 - 1e-9 <= t <= 75.0 + 1e-9]
    if len(seg) >= 2:
        slopes = [(v1 - v0) / (t1 - t0)
                  for (t0, v0), (t1, v1) in zip(seg, seg[1:]) if t1 > t0]
        if slopes:
            out["min_slope_25_75c"] = min(slopes)
        rise = seg[-1][1] - seg[0][1]
        max_step = max(v1 - v0 for (_, v0), (_, v1) in zip(seg, seg[1:]))
        out["max_step_frac_25_75c"] = (max_step / rise) if rise > 0 else None
    return out


def run_sfe_dc(deck: Path, control: str, work: Path, tag: str,
               timeout: float = 300.0
               ) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Run a sensor DC bench with the full-sweep dump and fallback scan.

    Returns (metrics, warnings): the ``.meas`` results merged with the
    ``sweep_metrics`` keys and ``dc_fallback_points``; *warnings* carries the
    pseudo-transient flag destined for the result's errors list.
    """
    sweep = (work / f"{tag}.sweep").resolve()
    sweep.unlink(missing_ok=True)
    ctl = f"{control}\nwrdata {sweep} v(vout)"
    metrics = run_deck(deck, ctl, work, tag, timeout=timeout)
    temps, vouts = read_dc_sweep(sweep)
    metrics.update(sweep_metrics(temps, vouts))
    warnings: List[str] = []
    log = work / f"{tag}.log"
    fallbacks = len(_FALLBACK_RE.findall(log.read_text())) if log.exists() else 0
    metrics["dc_fallback_points"] = float(fallbacks)
    npts = len(temps)
    if fallbacks >= FALLBACK_FLAG_MIN and npts \
            and fallbacks >= FALLBACK_FLAG_FRACTION * npts:
        warnings.append(
            f"{deck.name}: {fallbacks}/{npts} DC sweep points needed the "
            f"pseudo-transient fallback -- the DC operating branch is not "
            f"solved by continuation")
    return metrics, warnings


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SfeTargets:
    """What a usable PTAT front end looks like on this tech's rail."""
    lsb_min: float = 3e-4       # V/C -- below this the sensor is not resolvable
    lsb_max: float = 6e-3       # V/C -- above this it rails across the range
    vout_lo: float = 0.05       # V at 25 C, at least
    vout_hi: float = 0.875 * VDD_DEFAULT  # V at 25 C, at most
    pp_max: float = 0.75 * VDD_DEFAULT    # V swing over -20..120 C, at most


SFE_TARGETS = SfeTargets()


def sfe_score(m: Dict[str, Optional[float]], t: SfeTargets = SFE_TARGETS) -> float:
    """Penalty; 0 when the sensor is monotonic, resolvable and in range."""
    need = ("vout0", "vout25", "vout50", "vout75", "vout100",
            "lsb_25_75c", "ppval")
    if any(m.get(k) is None for k in need):
        return 50.0
    v = [m[f"vout{x}"] for x in (0, 25, 50, 75, 100)]
    pen = 0.0
    # Monotonic rise is the defining property of a PTAT front end.
    for a, b in zip(v, v[1:]):
        if b <= a:
            pen += 2.0 + 20.0 * (a - b)
    lsb = m["lsb_25_75c"]
    if lsb < t.lsb_min:
        pen += (t.lsb_min - lsb) / t.lsb_min
    if lsb > t.lsb_max:
        pen += (lsb - t.lsb_max) / t.lsb_max
    v25 = m["vout25"]
    if v25 < t.vout_lo:
        pen += (t.vout_lo - v25) / 0.05
    if v25 > t.vout_hi:
        pen += (v25 - t.vout_hi) / 0.05
    if m["ppval"] > t.pp_max:
        pen += (m["ppval"] - t.pp_max) / 0.1
    # Full-sweep smoothness terms (see the threshold notes above): a metrics
    # set without them came from a runner that did not dump the sweep, and a
    # verdict cannot be trusted on samples alone.
    mono = m.get("mono_violations")
    slope = m.get("min_slope_25_75c")
    frac = m.get("max_step_frac_25_75c")
    if mono is None or slope is None or frac is None:
        pen += 25.0
    else:
        pen += 1.0 * mono
        if slope < SLOPE_FLOOR_LOCAL:
            pen += (SLOPE_FLOOR_LOCAL - slope) / SLOPE_FLOOR_LOCAL
        if frac > STEP_FRACTION_MAX:
            pen += 4.0 * (frac - STEP_FRACTION_MAX)
    return pen


def sfe_report(m: Dict[str, Optional[float]],
               t: SfeTargets = SFE_TARGETS) -> Dict[str, bool]:
    need = ("vout0", "vout25", "vout50", "vout75", "vout100", "lsb_25_75c")
    if any(m.get(k) is None for k in need):
        return {"monotonic": False, "sensitivity": False,
                "smooth": False, "in_range": False}
    mono = m.get("mono_violations")
    slope = m.get("min_slope_25_75c")
    frac = m.get("max_step_frac_25_75c")
    return {
        # Every solved sweep point in 0..100 C, not just the five samples.
        "monotonic": mono is not None and mono == 0,
        "sensitivity": t.lsb_min <= m["lsb_25_75c"] <= t.lsb_max,
        # Staircase gate: thresholds documented above SLOPE_FLOOR_LOCAL.
        "smooth": (slope is not None and frac is not None
                   and slope >= SLOPE_FLOOR_LOCAL
                   and frac <= STEP_FRACTION_MAX),
        "in_range": t.vout_lo <= m["vout25"] <= t.vout_hi
                    and (m.get("ppval") or 0) <= t.pp_max,
    }
