"""Port and size the AnalogGym low-dropout regulators for this tree's tech.

Wiring is derived from each subckt's port names rather than hard-coded per
design: the five regulators disagree on port order and on whether the feedback
divider is inside the block, but they all name their supply, ground, regulated
output, feedback tap, amplifier inputs and bias pins recognisably.

Benches follow AnalogGym's TB_LDO_ACDC / TB_LDO_Tran measurement set -- line
regulation at both load extremes, load regulation, power, offset, loop gain /
GBW / phase margin, PSRR, and the load-step undershoot -- one operating point
per deck, for the same convergence reason as the amplifiers.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acstab import run_deck_auto
from geom_port import GSubckt, parse_generic
from ldo import (DIV_RATIO, ILOAD_MAX, ILOAD_MIN, LdoDesign, VDD, VOUT_NOM,
                 _passive_defaults, emit, initial_design, is_pass_device)
from meas import SimError
from pycmg_lib import L_CHOICES_FULL, MODELS_FILE, NFIN_CHOICES, TECH

ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT.parent / "designs" / "ldo"

_GND = re.compile(r"^(gnd|vss|gnda)$", re.I)
_VDD = re.compile(r"^(vdd|vdda)$", re.I)
_OUT = re.compile(r"^(vout|vreg)$", re.I)
_FB = re.compile(r"^vfb$", re.I)
_INN = re.compile(r"^(vinn|vref)$", re.I)
_INP = re.compile(r"^vinp$", re.I)
_IBIAS = re.compile(r"^ib$", re.I)
_VBIAS = re.compile(r"^vb\d*$", re.I)

BAD = 1e3
M_MAX = 100000
NON_PASS_M_MAX = 20000
CORE_GATES = ("regulates", "gain", "pm", "gbw", "power")
# Output capacitor.  AnalogGym's mim instance is ~2 pF, which cannot hold a
# 50 mA load step at all -- its own sky130 run reports 1.1 V to 18 V of
# undershoot on a 1.8 V output.  A regulator for this load range carries a real
# output capacitor, so it gets one.
CL = 100e-9
IBIAS = 5e-6
L_CHOICES = L_CHOICES_FULL
NFIN_CHOICES = list(NFIN_CHOICES)


def has_resistive_divider(subs, out_nodes: List[str]) -> bool:
    """True when a resistor spans the regulated output and the vfb node."""
    for s in subs:
        for p in s.passives:
            if not p.name.lower().startswith("r"):
                continue
            nodes = {n.lower() for n in p.nodes}
            if nodes & set(out_nodes) and "vfb" in nodes:
                return True
    return False


def wiring(ports: List[str], vdd_net: str, out_net: str,
           closed: bool = False, has_divider: bool = True,
           delivered_output_feedback: bool = False) -> Tuple[str, Dict]:
    """Map subckt ports onto bench nets; return (instance node list, info).

    With *closed* the amplifier's non-inverting input is tied straight to the
    feedback node instead of through the bench's 1T inductor.  For ``ldo_2``,
    *delivered_output_feedback* qualifies the delivered output while leaving
    the exposed replica ``vfb`` port on its own node.  PSRR is a closed-loop
    property -- measuring it with the loop broken at AC reports the unregulated
    supply feedthrough, which is not what the number means.
    """
    nodes: List[str] = []
    info = {"has_fb": False, "ibias": [], "vbias": []}
    for p in ports:
        if _GND.match(p):
            nodes.append("0")
        elif _VDD.match(p):
            nodes.append(vdd_net)
        elif _OUT.match(p):
            nodes.append(out_net)
        elif _FB.match(p):
            nodes.append(f"{out_net}_fb")
            info["has_fb"] = True
        elif _INN.match(p):
            nodes.append("vref_in")
        elif _INP.match(p):
            if closed:
                nodes.append(out_net if delivered_output_feedback else
                             _fb_net(out_net, ports, has_divider))
            else:
                nodes.append(f"{out_net}_inp")
        elif _IBIAS.match(p):
            nodes.append(f"{out_net}_ib")
            info["ibias"].append(f"{out_net}_ib")
        elif _VBIAS.match(p):
            net = f"{out_net}_{p.lower()}"
            nodes.append(net)
            info["vbias"].append((net, p.lower()))
        else:
            raise ValueError(f"unclassified LDO port {p!r} in {ports}")
    return " ".join(nodes), info


def _fb_net(out_net: str, ports: List[str], has_divider: bool = True) -> str:
    """Node used by the source regulation loop: exposed vfb, else output."""
    return f"{out_net}_fb" if any(_FB.match(p) for p in ports) else out_net


def _bias_lines(info: Dict, design: LdoDesign, tag: str,
                supply_net: Optional[str] = None) -> str:
    out = []
    for net in info["ibias"]:
        # Bias current is a design variable when the polish loop exposes it;
        # the original fixed 5 uA remains the default.
        source, sink = (supply_net, net) if supply_net else (net, "0")
        out.append(f"I{tag}_b {source} {sink} "
                   f"{design.passives.get('ibias', IBIAS):g}")
    for net, name in info["vbias"]:
        out.append(f"V{tag}_{name} {net} 0 "
                   f"{design.passives.get('bias_' + name, 0.5 * VDD):g}")
    return "\n".join(out)


# Extra .nodeset lines for the load-step transient, keyed by subckt name:
# the input-pair tail and bias-diode nodes the transient op cannot find on
# its own for these designs.
_TRAN_NODESETS = {
    "ldo_2": f"V(x1.net42)={0.38 * VDD:g} V(vo_ib)={0.49 * VDD:g}",
}

# Per-design transient integrator hints.  ldo_2's load-step recovery has a
# stiff corner at the diff-pair tail that trapezoidal integration rings on
# until the timestep collapses; Gear rides through it and the measured
# excursions match a trap run with a 5 ns step cap where both complete.
_TRAN_OPTIONS = {
    "ldo_2": ".option method=gear\n",
    "basic_ldo": ".option method=gear\n",
}

def line_control(tag: str, vdd: float) -> str:
    """Two-segment line sweep from the nominal rail outward, recombined.

    Emits the same ``lnr_avg{tag}`` / ``lnr_pp{tag}`` / ``lnr{tag}`` result
    lines a monolithic ``.meas dc`` sweep would, in the ``name = value``
    format ``run_deck`` parses.  Sweeping outward from the qualified nominal
    operating point keeps the DC continuation on the regulating branch.
    """
    lo, hi = 0.9 * vdd, 1.1 * vdd
    return "\n".join([
        f"dc V1 {vdd:g} {hi:g} 0.005",
        f"dc V1 {vdd:g} {lo:g} -0.005",
        "let seg_up_max = vecmax(dc1.v(vo))",
        "let seg_dn_max = vecmax(dc2.v(vo))",
        "let seg_up_min = vecmin(dc1.v(vo))",
        "let seg_dn_min = vecmin(dc2.v(vo))",
        "let seg_max = (seg_up_max + seg_dn_max + abs(seg_up_max - seg_dn_max)) / 2",
        "let seg_min = (seg_up_min + seg_dn_min - abs(seg_up_min - seg_dn_min)) / 2",
        "let seg_avg = (mean(dc1.v(vo)) + mean(dc2.v(vo))) / 2",
        "let seg_pp = seg_max - seg_min",
        "let seg_lnr = seg_pp / seg_avg / 0.2",
        f"echo lnr_avg{tag} = $&seg_avg",
        f"echo lnr_pp{tag} = $&seg_pp",
        f"echo lnr{tag} = $&seg_lnr",
    ])


HEADER = """\
* {sub} -- {what} on {tech} BSIM-CMG
* Measurement set is AnalogGym's TB_LDO_ACDC / TB_LDO_Tran.
.include ./{models_file}
.include ./netlist.spice

.PARAM supply_voltage = {vdd:g}
.PARAM Vref = {vref:g}
.PARAM VOUT_NOM = {vout:g}
V2 vss 0 0
Vindc vref_in 0 'Vref'
"""


def decks(sub: GSubckt, design: LdoDesign,
          has_divider: bool = True) -> Dict[str, Tuple[str, str]]:
    """Build every bench deck: name -> (text, ngspice control block)."""
    # ldo_2's error amp regulates the exposed replica node vfb (the source
    # bench ties vinp to vfb); the delivered output has its own local loop
    # through PM5/NM2 whose setpoint derives from vfb.  Closing vinp around
    # vo instead double-drives the output with two different setpoints and
    # the operating point walks away in transient.  Qualification still
    # senses v(vo): every gate is measured on the delivered output.
    delivered_output_feedback = False
    n_dc, i_dc = wiring(sub.ports, "vdd", "vo", has_divider=has_divider)
    n_ps, i_ps = wiring(sub.ports, "vddpsrr", "vp", closed=True,
                        has_divider=has_divider,
                        delivered_output_feedback=delivered_output_feedback)
    head = HEADER.format(sub=sub.name, vdd=design.vdd, vref=design.vref,
                         vout=VOUT_NOM, what="{what}",
                         tech=TECH, models_file=MODELS_FILE)

    # Feedback: the block either brings its own divider out on vfb, or the
    # bench closes the loop straight from the output.
    fb_dc = "vo" if delivered_output_feedback else \
        _fb_net("vo", sub.ports, has_divider)
    # ldo_1/ldo_2 expose an NMOS diode bias and require current injected from
    # the positive rail.  Basic_LDO exposes a PMOS diode and sinks its bias to
    # ground.  The source benches make this polarity distinction explicitly.
    ibias_from_supply = sub.name.lower() in {"ldo_1", "ldo_2"}
    ldo2_ns = (f".nodeset V(x1.net42)={0.38 * VDD:g} "
               f"V(vo_ib)={0.49 * VDD:g}\n") if sub.name.lower() == "ldo_2" else ""

    # Closed-loop wiring for everything except the loop-gain measurement.  The
    # 1T inductor is a short at DC but an open in transient, so a load step
    # measured through it runs with no feedback at all and the output simply
    # collapses -- AnalogGym's own transient bench ties the feedback directly
    # for exactly this reason.
    n_cl, i_cl = wiring(sub.ports, "vdd", "vo", closed=True,
                        has_divider=has_divider,
                        delivered_output_feedback=delivered_output_feedback)
    common_dc = (head.format(what="{what}") +
                 f"V1 vdd 0 'supply_voltage'\n"
                 f".nodeset V(vo)={VOUT_NOM:g}\n"
                 f"x1 {n_cl} {sub.name}\n"
                 f"{ldo2_ns}"
                 f"{_bias_lines(i_cl, design, '1', 'vdd' if ibias_from_supply else None)}\n"
                 f"CL vo 0 {CL:g}\n")
    # Open-loop (AC) variant keeps the inductor so the gain bench sees the loop
    # broken above DC.
    common_ol = (head.format(what="{what}") +
                 f"V1 vdd 0 'supply_voltage'\n"
                 f".nodeset V(vo)={VOUT_NOM:g}\n"
                 f"x1 {n_dc} {sub.name}\n"
                 f"{ldo2_ns}"
                 f"{_bias_lines(i_dc, design, '1', 'vdd' if ibias_from_supply else None)}\n"
                 f"CL vo 0 {CL:g}\n")

    out: Dict[str, Tuple[str, str]] = {}

    # --- line regulation, both load extremes -----------------------------
    # The sweep runs as two continuations from the nominal supply outward
    # (same approach as the amplifier temperature sweeps): a monolithic
    # lo->hi sweep starts Newton cold at the worst-headroom corner and the
    # high-gain loops jump onto a non-regulating branch mid-sweep, reporting
    # multi-volt excursions that no reachable operating point has.  The
    # measures are recombined in the control script, so metric keys and
    # meanings are unchanged.
    for tag, iload in (("max", ILOAD_MAX), ("min", ILOAD_MIN)):
        out[f"tb_line_{tag}.cir"] = (
            common_dc.format(what=f"line regulation, {tag} load")
            + f"Iload vo 0 {iload:g}\n"
              ".end\n",
            line_control(tag, design.vdd),
        )

    # --- load regulation, power, offset ----------------------------------
    out["tb_load.cir"] = (
        common_dc.format(what="load regulation / power / offset")
        + f"Iload vo 0 {ILOAD_MIN:g}\n"
        + f".meas dc lr_avg AVG V(vo) from={ILOAD_MIN:g} to={ILOAD_MAX:g}\n"
          f".meas dc lr_pp  PP  V(vo) from={ILOAD_MIN:g} to={ILOAD_MAX:g}\n"
          f".meas dc lr param='lr_pp/lr_avg/{ILOAD_MAX - ILOAD_MIN:g}'\n"
          f".meas dc ivdd_max FIND I(V1) AT={ILOAD_MAX:g}\n"
          f".meas dc ivdd_min FIND I(V1) AT={ILOAD_MIN:g}\n"
          f".meas dc power_max param='-1*ivdd_max*{design.vdd:g}'\n"
          f".meas dc power_min param='-1*ivdd_min*{design.vdd:g}'\n"
          f".meas dc vout_max FIND V(vo) AT={ILOAD_MAX:g}\n"
          f".meas dc vout_min FIND V(vo) AT={ILOAD_MIN:g}\n"
          f".meas dc vos_max param='vout_max-{VOUT_NOM:g}'\n"
          f".meas dc vos_min param='vout_min-{VOUT_NOM:g}'\n"
          ".end\n",
        f"dc Iload {ILOAD_MIN:g} {ILOAD_MAX:g} {(ILOAD_MAX-ILOAD_MIN)/100:g}",
    )

    # --- loop gain / GBW / phase margin, both load extremes ---------------
    for tag, iload in (("max", ILOAD_MAX), ("min", ILOAD_MIN)):
        out[f"tb_loop_{tag}.cir"] = (
            common_ol.format(what=f"loop response, {tag} load")
            + f"Iload vo 0 {iload:g}\n"
              f"Vin signal_in 0 dc 'Vref' ac 1\n"
              f"Lfb vo_inp {fb_dc} 1T\n"
              f"Cfb vo_inp signal_in 1T\n"
              f".meas ac dcgain_{tag} find vdb(vo) at=0.1\n"
              f".meas ac gbw_{tag} when vdb(vo)=0\n"
              f".meas ac ph_rad_{tag} find vp(vo) when vdb(vo)=0\n"
              # pm_{tag} is abs() of vp()'s PRINCIPAL VALUE at crossover and
              # is kept only as the raw auditable reading: it cannot tell a
              # lead-recovered loop (Basic_LDO's feedforward zeros) from one
              # whose phase truly fell through -180 and wrapped back.  The
              # gated margin is pm_true_{tag}, computed by the runner
              # (tools/acstab.py) from the unwrapped full-sweep dump.
              f".meas ac pm_{tag} param='abs(ph_rad_{tag})*180/3.1416'\n"
              ".end\n",
            "ac dec 20 0.1 1G",
        )

    # --- PSRR, both load extremes ----------------------------------------
    for tag, iload in (("max", ILOAD_MAX), ("min", ILOAD_MIN)):
        out[f"tb_psrr_{tag}.cir"] = (
            head.format(what=f"PSRR, {tag} load")
            + f"VVDDApsrr vddpsrr 0 'supply_voltage' AC=1\n"
              f".nodeset V(vp)={VOUT_NOM:g}\n"
              f"x2 {n_ps} {sub.name}\n"
              + (f".nodeset V(x2.net42)={0.38 * VDD:g} V(vp_ib)={0.49 * VDD:g}\n"
                 if sub.name.lower() == "ldo_2" else "")
              + f"{_bias_lines(i_ps, design, '2', 'vddpsrr' if ibias_from_supply else None)}\n"
              f"CL2 vp 0 {CL:g}\n"
              f"Iload2 vp 0 {iload:g}\n"
              f".meas ac psrr_{tag} find vdb(vp) at=0.1\n"
              ".end\n",
            "ac dec 20 0.1 1G",
        )

    # --- load-step transient ---------------------------------------------
    # cshunt: 1 fF to ground on every node.  The transient operating point
    # (a different algorithm from the DC op the other decks use) fails on the
    # bias-diode node for the large-pass designs; the shunt gives it a path.
    # Far below any device capacitance here, so no measurement changes.
    # A couple of designs additionally need their internal tail node seeded
    # (the DC decks solve without it; the transient op does not).
    extra_ns = _TRAN_NODESETS.get(sub.name.lower(), "")
    out["tb_tran.cir"] = (
        common_dc.format(what="load-step transient")
        + ".option cshunt=1e-15\n"
        + _TRAN_OPTIONS.get(sub.name.lower(), "")
        + (f".nodeset {extra_ns}\n" if extra_ns else "")
        + f"Iload vo 0 pulse({ILOAD_MIN:g} {ILOAD_MAX:g} 2u 100n 100n 20u 60u)\n"
          f".meas tran v_pre AVG v(vo) from=1u to=1.9u\n"
          f".meas tran v_min MIN v(vo) from=2u to=22u\n"
          f".meas tran v_max MAX v(vo) from=22u to=42u\n"
          f".meas tran undershoot param='v_pre-v_min'\n"
          f".meas tran overshoot param='max(0,v_max-v_pre)'\n"
          ".end\n",
        "tran 20n 44u",
    )
    return out


DECK_ORDER = ["tb_load.cir", "tb_loop_max.cir", "tb_loop_min.cir",
              "tb_line_max.cir", "tb_line_min.cir",
              "tb_psrr_max.cir", "tb_psrr_min.cir", "tb_tran.cir"]
FAST_ORDER = ["tb_load.cir", "tb_loop_max.cir", "tb_loop_min.cir"]


def run_all(where: Path, sub: GSubckt, design: LdoDesign,
            order: List[str], timeout: float,
            has_divider: bool = True) -> Dict[str, float]:
    built = decks(sub, design, has_divider)
    merged: Dict[str, float] = {}
    for name in order:
        text, control = built[name]
        (where / name).write_text(text)
        # run_deck_auto upgrades the tb_loop decks to the wrap-aware
        # stability runner, so search and report share one PM semantics.
        merged.update(run_deck_auto(where / name, control, where,
                                    name.replace(".cir", ""), timeout=timeout))
    return merged


def _finite_metric(m: Dict[str, float], key: str) -> Optional[float]:
    value = m.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _minimum_penalty(value: Optional[float], limit: float,
                     scale: float, missing: float = 3.0) -> float:
    return missing if value is None else max(0.0, (limit - value) / scale)


def _maximum_penalty(value: Optional[float], limit: float,
                     scale: float, missing: float = 3.0) -> float:
    return missing if value is None else max(0.0, (value - limit) / scale)


def _range_penalty(value: Optional[float], low: float, high: float,
                   scale: float, missing: float = 3.0) -> float:
    if value is None:
        return missing
    if value < low:
        return (low - value) / scale
    return max(0.0, (value - high) / scale)


def ldo_core_score(m: Dict[str, float]) -> float:
    """Penalty for regulation, gain, stability, bandwidth, and power gates."""
    pen = 0.0
    for tag in ("max", "min"):
        pen += _minimum_penalty(_finite_metric(m, f"dcgain_{tag}"), 40, 10)
        # True (wrap-aware) margin when the stability runner produced it;
        # the raw abs()-of-principal-value pm_{tag} is a fallback only for
        # metric sets measured without the sweep dump.
        pm_key = f"pm_true_{tag}" if f"pm_true_{tag}" in m else f"pm_{tag}"
        pen += _minimum_penalty(_finite_metric(m, pm_key), 45, 15)
        pen += _minimum_penalty(_finite_metric(m, f"gbw_{tag}"), 1e5, 5e4,
                                missing=2.0)
        vout = _finite_metric(m, f"vout_{tag}")
        offset = None if vout is None else abs(vout - VOUT_NOM)
        pen += _maximum_penalty(offset, 0.05, 0.05)
    pen += _maximum_penalty(_finite_metric(m, "power_max"), 80e-3, 20e-3)
    return pen


def ldo_score(m: Dict[str, float]) -> float:
    """Penalty for every metric used by the official nine-gate LDO audit."""
    pen = ldo_core_score(m)
    for key in ("lnrmax", "lnrmin"):
        pen += _range_penalty(_finite_metric(m, key), 0.0, 0.25, 0.25)
    pen += _range_penalty(_finite_metric(m, "lr"), 0.0, 1.0, 1.0)
    for key in ("psrr_max", "psrr_min"):
        pen += _maximum_penalty(_finite_metric(m, key), -20.0, 20.0)
    pen += _range_penalty(_finite_metric(m, "undershoot"), 0.0,
                          0.25 * VDD, 0.25 * VDD)
    pen += _range_penalty(_finite_metric(m, "overshoot"), 0.0,
                          0.15 * VDD, 0.15 * VDD)
    return pen


def ldo_report(m: Dict[str, float]) -> Dict[str, bool]:
    def ok(key: str, predicate: Callable[[float], bool]) -> bool:
        value = _finite_metric(m, key)
        return value is not None and predicate(value)

    return {
        "regulates": ok("vout_max", lambda v: abs(v - VOUT_NOM) <= 0.05)
                     and ok("vout_min", lambda v: abs(v - VOUT_NOM) <= 0.05),
        "gain": ok("dcgain_max", lambda v: v >= 40) and ok("dcgain_min", lambda v: v >= 40),
        # Gated on the TRUE margin (unwrapped sweep, tools/acstab.py); the
        # wrapped abs()-of-principal-value pm_max/pm_min stay in the metrics
        # as the raw auditable readings but cannot pass this gate.
        "pm": ok("pm_true_max", lambda v: v >= 45)
              and ok("pm_true_min", lambda v: v >= 45),
        "gbw": ok("gbw_max", lambda v: v >= 1e5) and ok("gbw_min", lambda v: v >= 1e5),
        "power": ok("power_max", lambda v: v <= 80e-3),
        "line_regulation": ok("lnrmax", lambda v: 0 <= v <= 0.25)
                           and ok("lnrmin", lambda v: 0 <= v <= 0.25),
        "load_regulation": ok("lr", lambda v: 0 <= v <= 1.0),
        "psrr": ok("psrr_max", lambda v: v <= -20)
                and ok("psrr_min", lambda v: v <= -20),
        "load_step": ok("undershoot", lambda v: 0 <= v <= 0.25 * VDD)
                     and ok("overshoot", lambda v: 0 <= v <= 0.15 * VDD),
    }


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def _vec(d: LdoDesign, keys: List[str], pkeys: List[str]) -> List[float]:
    v: List[float] = []
    for k in keys:
        l_nm, nfin, m = d.geoms[k]
        v += [l_nm, float(nfin), float(m)]
    v += [d.passives[p] for p in pkeys]
    return v


def _unvec(v: List[float], d: LdoDesign, keys: List[str],
           pkeys: List[str], pass_keys: Set[str]) -> LdoDesign:
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for i, k in enumerate(keys):
        l_nm = min(L_CHOICES, key=lambda c: abs(c - v[3 * i]))
        nfin = min(NFIN_CHOICES, key=lambda c: abs(c - v[3 * i + 1]))
        mult_max = M_MAX if k in pass_keys else NON_PASS_M_MAX
        geoms[k] = (float(l_nm), int(nfin),
                    max(1, min(mult_max, int(round(v[3 * i + 2])))))
    passives = dict(d.passives)
    for j, p in enumerate(pkeys):
        val = v[3 * len(keys) + j]
        if p.startswith("bias_"):
            # Folded-cascode Vb2 is intentionally near ground (1.25 % VDD in
            # the source design); the generic 50 mV floor erased that basin.
            passives[p] = max(1e-3, min(VDD - 1e-3, val))
        elif p == "vref":
            passives[p] = max(0.05, min(VDD - 0.05, val))
        else:
            passives[p] = max(1e-15, val)
    return replace(d, geoms=geoms, passives=passives, vref=passives["vref"])


def _seeded(base: LdoDesign, seed: int,
            pass_keys: Set[str]) -> LdoDesign:
    """Perturbed starting point; the pass device keeps its own, larger scale.

    Pattern search finds one basin from one start.  The regulators that stall
    stall on loop stability at one load extreme, which is a basin property --
    a different starting geometry moves it, more iterations from the same start
    do not.
    """
    if seed == 0:
        return base
    if seed in (1, 2):
        pass_scale = 4 if seed == 1 else 8
        geoms = dict(base.geoms)
        for key in pass_keys:
            l_nm, nfin, mult = geoms[key]
            geoms[key] = (l_nm, nfin, min(M_MAX, mult * pass_scale))
        return replace(base, geoms=geoms)
    rng = random.Random(seed)
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for k, (l_nm, nfin, m) in base.geoms.items():
        scale = rng.choice([0.5, 1, 2, 4, 8]) if k in pass_keys \
            else rng.choice([0.25, 0.5, 1, 2, 4])
        mult_max = M_MAX if k in pass_keys else NON_PASS_M_MAX
        geoms[k] = (float(rng.choice(L_CHOICES)),
                    int(rng.choice(NFIN_CHOICES)),
                    max(1, min(mult_max, int(m * scale))))
    passives = dict(base.passives)
    for k in passives:
        if k.lower().startswith(("c", "xc")):
            passives[k] = passives[k] * rng.choice([0.25, 1, 4, 16])
    return replace(base, geoms=geoms, passives=passives)


def size(name: str, max_evals: int = 200, restarts: int = 1) -> dict:
    subs, top = parse_generic(SKY / name / "netlist.spice")
    sub = next(s for s in subs if s.mos)
    has_div = has_resistive_divider(subs, [p.lower() for p in sub.ports
                                           if _OUT.match(p)])
    _n, info = wiring(sub.ports, "vdd", "vo", has_divider=has_div)

    out_nodes = [p.lower() for p in sub.ports if _OUT.match(p)]
    design = initial_design(subs, out_nodes)
    design.passives = _passive_defaults(subs)
    # Blocks that expose a vfb tap divide internally, so the error amp sees
    # VOUT/DIV_RATIO; the rest close the loop straight from the output and the
    # reference must therefore sit at the output voltage itself.
    design.vref = VOUT_NOM / DIV_RATIO if (info["has_fb"] and has_div) \
        else VOUT_NOM
    # Feedback divider: identify the legs by what they touch, not by name --
    # Basic_LDO and ldo_1 use r0/r1 for opposite legs.  The top leg is the one
    # on the regulated output, the bottom leg the one on ground.  The shipped
    # 300k/100k ratio of 4 puts the reference at 150 mV on a 0.8 V rail, so it
    # is re-chosen along with everything else.
    gnd_ports = {p.lower() for p in sub.ports if _GND.match(p)}
    for sub_ in subs:
        for p in sub_.passives:
            if not p.name.lower().startswith("r"):
                continue
            nodes = {n.lower() for n in p.nodes}
            if nodes & set(out_nodes):
                design.passives[p.name] = 100e3 * (DIV_RATIO - 1)   # top leg
            elif nodes & (gnd_ports | {"0"}):
                design.passives[p.name] = 100e3                     # bottom leg
    for net, bname in info["vbias"]:
        design.passives.setdefault(f"bias_{bname}", 0.5 * VDD)
    if sub.name.lower() == "ldo_folded_cascode":
        # Preserve the source topology's strongly asymmetric cascode biases.
        # Seeding both pins at half-supply strands this loop on a different DC
        # branch where line regulation and PSRR have no useful search gradient.
        design.passives["bias_vb1"] = 0.5 * VDD
        design.passives["bias_vb2"] = 0.0125 * VDD
    if info["ibias"]:
        design.passives["ibias"] = IBIAS

    keys = list(design.geoms)
    # Reference voltage is tuned like any other variable: for ldo_2 the vfb pin
    # is an internal replica node, not a divider tap, so no fixed ratio predicts
    # where the reference has to sit for the output to land on VOUT_NOM.
    design.passives["vref"] = design.vref
    pkeys = [k for k in design.passives
             if k.startswith("bias_") or k in ("vref", "ibias")
             or k.lower().startswith(("c", "xc"))]

    work = ROOT / "work" / "ldo" / name
    out = ROOT / "ldo" / name
    work.mkdir(parents=True, exist_ok=True)

    def evaluate(d: LdoDesign, where: Path, order, timeout):
        try:
            emit(where, subs, top, d, sub.name, sub.ports, out_nodes)
            m = run_all(where, sub, d, order, timeout, has_div)
        except Exception:
            return BAD, {}
        return ldo_score(m), m

    pass_keys = {m.group for s in subs for m in s.mos
                 if is_pass_device(m, out_nodes)}
    best, best_score, best_m = None, float("inf"), {}
    for seed in range(restarts):
        start = _seeded(design, seed, pass_keys)
        s0, m0 = evaluate(start, work, FAST_ORDER, 25)
        if s0 < best_score:
            best, best_score, best_m = start, s0, m0
        if best_score <= 0:
            break
    evals = restarts
    step = [2.0, 1.6, 3.0] * len(keys) + [1.5] * len(pkeys)
    t0 = time.time()

    while evals < max_evals and best_score > 0 and max(step) > 1.05:
        improved = False
        vec = _vec(best, keys, pkeys)
        for i in range(len(vec)):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                cv = list(vec)
                cv[i] *= step[i] ** direction
                cand = _unvec(cv, best, keys, pkeys, pass_keys)
                if _vec(cand, keys, pkeys) == _vec(best, keys, pkeys):
                    continue
                s, m = evaluate(cand, work, FAST_ORDER, 25)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    vec = _vec(best, keys, pkeys)
                    break
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]

    elapsed = time.time() - t0
    emit(out, subs, top, best, sub.name, sub.ports, out_nodes)
    final, errors = {}, []
    built = decks(sub, best, has_div)
    for deck in DECK_ORDER:
        text, control = built[deck]
        (out / deck).write_text(text)
        try:
            final.update(run_deck_auto(out / deck, control, out,
                                       deck.replace(".cir", ""), timeout=600))
        except SimError as exc:
            errors.append(f"{deck}: {str(exc).splitlines()[0]}")

    result = {"design": name, "score": ldo_score(final),
              "evals": evals, "evals_seconds": round(elapsed, 1),
              "metrics": final, "pass": ldo_report(final), "errors": errors}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _one(args):
    name, evals = args
    try:
        return size(name, evals, restarts=10)
    except Exception as exc:
        return {"design": name, "error": f"{type(exc).__name__}: {exc}",
                "pass": {}, "metrics": {}}


def main() -> None:
    evals = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    names = sorted(d.name for d in SKY.iterdir() if d.is_dir())
    results = []
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        futs = {pool.submit(_one, (n, evals)): n for n in names}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            p = r.get("pass", {})
            print(f"[{len(results)}/{len(names)}] {r['design']:22s} "
                  f"{'ERR ' + r['error'][:60] if r.get('error') else ''}"
                  f"score={r.get('score', -1):.3f} "
                  f"pass={sum(p.values())}/{len(p) or 1}", flush=True)
    outp = ROOT / "results" / "ldo_sizing.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(sorted(results, key=lambda r: r["design"]),
                               indent=2, default=str))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
