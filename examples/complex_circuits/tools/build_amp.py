"""Emit a TSMC amplifier design directory from a topology + a design vector.

Produces, per design:

    netlist.spice        the AnalogGym topology with TSMC BSIM-CMG devices
    <tech>_models.spice  the baked model cards those devices name
    design.json          the design vector that generated them
    tb_gain.cir          open-loop gain / GBW / phase
    tb_cmrr.cir          common-mode rejection
    tb_psrrp.cir         supply rejection, positive rail
    tb_psrrn.cir         supply rejection, negative rail
    tb_dc.cir            -40..125 C sweep: TC, power, offset
    tb_tran.cir          slew rate

AnalogGym packs the first five into one deck; see the note in _HEADER for why
they are split here.

Connectivity and mirror ratios are carried over verbatim; only the device
geometry, the supply and the passive values are re-designed.  ``m`` on each
instance is ``role.m * instance.mult`` so the ratios inside the topology --
which is what makes a current mirror a mirror -- survive the port.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pycmg_lib import (MODELS_FILE, ModelLibrary, OSDI_PATH, TECH, VDD,
                       snap_l, snap_nfin)
from skyparse import Topology


@dataclass
class RoleGeom:
    """Geometry shared by every transistor carrying one AnalogGym role."""
    vt: str = "svt"
    l_nm: float = 60.0
    nfin: int = 4
    m: int = 4


@dataclass
class AmpDesign:
    """The full design vector for one amplifier."""
    vdd: float = VDD
    vcm: float = 0.25 * VDD
    cload: float = 500e-12
    gbw_ideal: float = 1e6
    roles: Dict[str, RoleGeom] = None          # type: ignore[assignment]
    passives: Dict[str, float] = None          # type: ignore[assignment]

    def to_json(self) -> str:
        return json.dumps(
            {
                "vdd": self.vdd, "vcm": self.vcm, "cload": self.cload,
                "gbw_ideal": self.gbw_ideal,
                "roles": {k: asdict(v) for k, v in self.roles.items()},
                "passives": self.passives,
            },
            indent=2, sort_keys=True,
        )

    @staticmethod
    def from_json(text: str) -> "AmpDesign":
        d = json.loads(text)
        return AmpDesign(
            vdd=d["vdd"], vcm=d["vcm"], cload=d["cload"],
            gbw_ideal=d.get("gbw_ideal", 1e6),
            roles={k: RoleGeom(**v) for k, v in d["roles"].items()},
            passives=d["passives"],
        )


def _eng(x: float) -> str:
    """Format a value for SPICE without losing precision to unit suffixes."""
    return f"{x:.6g}"


def emit_netlist(topo: Topology, design: AmpDesign,
                 lib: ModelLibrary) -> str:
    """Render the topology with TSMC16 devices, registering geometries in *lib*."""
    lines = [
        f"* {topo.subckt} -- AnalogGym topology on {TECH} BSIM-CMG (LEVEL=72)",
        f"* Connectivity and mirror ratios are the shipped ones; geometry is "
        f"re-designed for {VDD:g} V FinFET.",
        f".subckt {topo.subckt} {' '.join(topo.ports)}",
    ]

    for mos in topo.mos:
        g = design.roles[mos.role]
        model = lib.model_name(mos.kind, g.vt, g.l_nm * 1e-9, g.nfin)
        m_total = max(1, int(round(g.m * mos.mult)))
        # 'x' prefixes came from sky130's subcircuit devices; OSDI instances
        # take the N prefix in ngspice.
        name = "N" + mos.name.lstrip("xX")
        lines.append(
            f"{name} {mos.d} {mos.g} {mos.s} {mos.b} {model} m={m_total}"
            f"   $ {mos.role}"
        )

    for p in topo.passives:
        val = design.passives.get(p.var)
        if val is None:
            raise KeyError(
                f"{topo.subckt}: passive {p.name} needs a value for {p.var!r}"
            )
        lines.append(f"{p.name} {p.n1} {p.n2} {_eng(val)}")

    lines.append(f".ends {topo.subckt}")
    return "\n".join(lines) + "\n"


_HEADER = """\
* {subckt} -- {what} on {tech} BSIM-CMG
* Measurements are AnalogGym's TB_Amplifier_{orig}.cir verbatim; supply, common
* mode and load are the {tech} design point.
*
* AnalogGym runs all five amplifier instances from one deck.  They share only
* ideal supply rails, so they are electrically independent and splitting them
* changes no measurement -- but five 90 dB loops in one matrix does not solve:
* gmin, source stepping and the transient op all fail, and which groupings
* happen to converge is luck of the Newton basin, not a rule.  One instance per
* deck is the only robust structure.
.include ./{models_file}
.include ./netlist.spice

.PARAM supply_voltage = {vdd}
.PARAM VCM = {vcm}
.PARAM PARAM_CLOAD = {cload}

V1 vdd 0 'supply_voltage'
V2 vss 0 0
Vindc opin 0 'VCM'
"""

# Each instance is wired in unity feedback, so its output sits at VCM by
# construction; seeding that is what lets the operating point solve at all.
TB_GAIN = _HEADER + """\
Vin signal_in 0 dc 'VCM' ac 1
.nodeset V(opout)={vcm} V(opout_dc)={vcm}

Lfb opout opout_dc 1T
Cin opout_dc signal_in 1T

*    ADM TB
Xop1 vss vdd opout_dc opin opout {subckt}
Cload1 opout 0 'PARAM_CLOAD'

.meas ac dcgain find vdb(opout) at = 0.1
.meas ac gain_bandwidth_product when vdb(opout)=0
.meas ac phase_in_rad find vp(opout) when vdb(opout)=0
.meas ac phase_in_deg param='phase_in_rad*180/3.1416'
.end
"""

TB_CMRR = _HEADER + """\
* cshunt: 1 fF from every node to ground, far below any device capacitance
* here.  The common-mode deck's operating point fails outright on some
* tech/design pairs (TSMC6 Qu_LEC) without it; measured CMRR is unchanged
* on decks that converge either way.  rshunt is added only to the
* characterized tech/design pairs that need it.
.option cshunt=1e-15{cmrr_rshunt}
.nodeset V(cm3)={vcm} V(cm2)={vcm}

*    ACM TB
xop2 vss vdd cm2 cm1 cm3 {subckt}
Cload2 cm3 0 'PARAM_CLOAD'
vcmdc cm0 0 'VCM'
vcmac1 cm1 cm0 0 ac=1
vcmac2 cm2 cm3 0 ac=1

.meas ac cmrrdc find vdb(cm3) at = 0.1
.end
"""

TB_PSRRP = _HEADER + """\
.nodeset V(ppsr1)={vcm}

*    PSRR+ TB
VVDDApsrr vddpsrr 0 'supply_voltage' AC=1
xop3 vss vddpsrr ppsr1 opin ppsr1 {subckt}
Cload3 ppsr1 0 'PARAM_CLOAD'

.measure ac DCPSRp find vdb(ppsr1) at = 0.1
.end
"""

TB_PSRRN = _HEADER + """\
{psrrn_option}
.nodeset V(npsr1)={vcm}

*    PSRR- TB
VGNDApsrr gndpsrr 0 0 AC=1
xop4 gndpsrr vdd npsr1 opin npsr1 {subckt}
Cload4 npsr1 0 'PARAM_CLOAD'

.measure ac DCPSRn find vdb(npsr1) at = 0.1
.end
"""

TB_DC = _HEADER + """\
.nodeset V(vout6)={vcm}

*    DC ALL TB
VVDDdc VDDdc 0 'supply_voltage'
xop5 vss vdddc vout6 opin vout6 {subckt}
Cload5 vout6 0 'PARAM_CLOAD'

.measure dc maxval MAX V(vout6) from=-40 to=125
.measure dc minval MIN V(vout6) from=-40 to=125
.measure dc avgval AVG V(vout6) from=-40 to=125
.measure dc ppavl  PP V(vout6) from=-40 to=125
.measure dc TC param='ppavl/avgval/165'
* Segment measurements let finalize recover the exact full range with two
* continuation sweeps when a single 165 C sweep loses its Newton branch.
.measure dc max_hot  MAX V(vout6) from=25 to=125
.measure dc min_hot  MIN V(vout6) from=25 to=125
.measure dc avg_hot  AVG V(vout6) from=25 to=125
.measure dc max_cold MAX V(vout6) from=-40 to=25
.measure dc min_cold MIN V(vout6) from=-40 to=25
.measure dc avg_cold AVG V(vout6) from=-40 to=25
.meas dc Ivdd25 FIND I(VVDDDC) AT=25
.meas dc Power param='-1*Ivdd25*supply_voltage'
.meas dc vout25 FIND V(vout6) AT=25
* a .meas param expression cannot read a .PARAM, so VCM is inlined
.meas dc vos25 param = 'vout25-{vcm}'
.end
"""

TB_TRAN = _HEADER + """\
.options method=gear maxord=2 cshunt=1e-14 vntol=1e-5{tran_rshunt}
.PARAM GBW_ideal = {gbw_ideal}
.PARAM STEP_TIME = '10/GBW_ideal'
* The transient op must settle at the pulse BASELINE, not the common mode:
* seeding vout3 at VCM makes the pre-edge interval a slew event of its own,
* which slew-marginal designs do not survive.
.nodeset V(vout3)={nodeset}

VVISR visr 0 pulse({val0} {val1} {t_delay} {t_edge} {t_edge} {t_pw} {t_per})
xop6 vss vdd vout3 visr vout3 {subckt}
CLoad6 vout3 0 'PARAM_CLOAD'

* Search-only diagnostics sample the settled ends of each commanded plateau.
* They do not change the official slew measurements or their crossing windows.
.meas tran v_pre AVG v(vout3) from={t_pre_start} to={t_pre_end}
.meas tran v_high AVG v(vout3) from={t_high_start} to={t_high_end}
.meas tran v_low AVG v(vout3) from={t_low_start} to={t_low_end}

* AnalogGym's fall-edge block names t_fall twice and the second reads a t_fall_
* that never exists, so sr_fall lands 1e6x off sr_rise.  The rise block is the
* correct one and is mirrored here.
.meas tran t_rise_edge when v(vout3)={vmid} rise=1 td={t_delay}
.meas tran t_rise_ param='t_rise_edge-{t_delay}'
.meas tran t_rise param='t_rise_*1e6'
.meas tran sr_rise param='{half_step}/t_rise'

.meas tran t_fall_edge when v(vout3)={vmid} fall=1 td={t_fall_start}
.meas tran t_fall_ param='t_fall_edge-{t_delay}-{t_pw}'
.meas tran t_fall param='t_fall_*1e6'
.meas tran sr_fall param='{half_step}/t_fall'
.end
"""


# Alternate nodesets are emergency reference fallbacks, not a second transient
# test family. Every entry must have dual-engine evidence that the primary seed
# fails while the alternate seed passes the full transient metric gate.
ALT_NODESET_DECKS: Set[Tuple[str, str]] = {
    ("TSMC5", "qu2017_azc_pin_3"),
}


def emit_testbenches(topo: Topology, design: AmpDesign) -> Dict[str, str]:
    """Render AnalogGym's amplifier benches at the TSMC16 design point."""
    # AnalogGym steps the input by ~11 % of the supply about the common mode and
    # times the crossing of the midpoint; sr = half-step / interval.
    step = 0.11 * design.vdd
    step_time = 10.0 / design.gbw_ideal
    t_delay = 0.1 * step_time
    t_pw = step_time
    t_fall_start = 1.1001 * step_time
    t_end = 2.2 * step_time
    deck_key = (TECH, topo.subckt.lower())
    cmrr_rshunt = " rshunt=1e10" if deck_key in {
        ("TSMC12", "peng_acbc_pin_3"),
        ("TSMC12", "yan_az_pin_3"),
        ("TSMC16", "leung_nmcf_pin_3"),
        ("TSMC16", "song_dacfc_pin_3"),
        ("TSMC16", "tan_clia_pin_3"),
        ("TSMC5", "leung_dfcfc1_pin_3"),
    } else ""
    psrrn_option = ".option rshunt=1e10" if deck_key in {
        ("TSMC12", "leung_dfcfc2_pin_3"),
        ("TSMC6", "qu2017_azc_pin_3"),
        ("TSMC7", "qu2017_azc_pin_3"),
        ("TSMC5", "leung_dfcfc1_pin_3"),
    } else ""
    tran_rshunt = " rshunt=1e12" if deck_key in {
        ("TSMC6", "tan_clia_pin_3"),
        ("TSMC7", "tan_clia_pin_3"),
        ("TSMC5", "qu_lec_pin_3"),
    } else ""
    common = dict(
        subckt=topo.subckt, tech=TECH, models_file=MODELS_FILE,
        vdd=_eng(design.vdd), vcm=_eng(design.vcm),
        cload=_eng(design.cload),
    )
    tran_common = dict(
        orig="Tran", what="slew-rate bench",
        tran_rshunt=tran_rshunt,
        val1=_eng(design.vcm + step / 2),
        vmid=_eng(design.vcm), half_step=_eng(step / 2),
        gbw_ideal=_eng(design.gbw_ideal),
        t_delay=_eng(t_delay), t_edge=_eng(1e-4 * step_time),
        t_fall_start=_eng(t_fall_start),
        t_pw=_eng(t_pw), t_per=_eng(1e3 * step_time),
        t_pre_start=_eng(0.5 * t_delay),
        t_pre_end=_eng(0.9 * t_delay),
        t_high_start=_eng(t_delay + 0.8 * t_pw),
        t_high_end=_eng(t_fall_start - 0.05 * t_pw),
        t_low_start=_eng(t_fall_start
                         + 0.8 * (t_end - t_fall_start)),
        t_low_end=_eng(t_end),
    )

    def render_transient(nodeset: float) -> str:
        """Render the shared pulse bench with an explicit startup seed."""
        return TB_TRAN.format(
            **common,
            **tran_common,
            val0=_eng(design.vcm - step / 2),
            nodeset=_eng(nodeset),
        )

    benches = {
        "tb_gain.cir": TB_GAIN.format(**common, orig="ACDC",
                                      what="open-loop gain bench"),
        "tb_cmrr.cir": TB_CMRR.format(
            **common, orig="ACDC", what="common-mode bench",
            cmrr_rshunt=cmrr_rshunt,
        ),
        "tb_psrrp.cir": TB_PSRRP.format(**common, orig="ACDC",
                                        what="PSRR+ bench"),
        "tb_psrrn.cir": TB_PSRRN.format(
            **common, orig="ACDC", what="PSRR- bench",
            psrrn_option=psrrn_option,
        ),
        "tb_dc.cir": TB_DC.format(**common, orig="ACDC",
                                  what="DC / temperature bench"),
        # Primary transient seeds the output at the pulse baseline (the true
        # pre-edge settle point).
        "tb_tran.cir": render_transient(design.vcm - step / 2),
    }
    if deck_key in ALT_NODESET_DECKS:
        # This validated fallback keeps the historic common-mode seed for the
        # one design whose NGSPICE transient op rejects the primary seed.
        benches["tb_tran_altns.cir"] = render_transient(design.vcm)
    return benches


# Per-design transient max-step caps, keyed like the deck rshunt exceptions.
# Qu_LEC on the 0.65 V rail rings through its fall edge until the timestep
# collapses unless the step is held at 2 ns; the measured slews match the
# uncapped run everywhere both complete.
TRAN_MAXSTEP = {
    ("TSMC5", "qu_lec_pin_3"): "2n",
}


def tran_control(design: AmpDesign, subckt: str = "") -> str:
    """``.control`` line for the slew bench, scaled to the design's step time."""
    step_time = 10.0 / design.gbw_ideal
    ctl = f"tran {_eng(step_time / 2000)} {_eng(2.2 * step_time)}"
    cap = TRAN_MAXSTEP.get((TECH, subckt.lower()))
    return f"{ctl} 0 {cap}" if cap else ctl


_AC_SWEEP = "ac dec 20 0.1 10G"
# The search runs a coarser decade so each evaluation is cheap; GBW and phase
# come from interpolation between points either way, and the optimisation
# targets sit well inside the reporting ones so the coarser grid cannot flip a
# verdict.  Reporting always uses the fine sweep above.
_AC_SWEEP_FAST = "ac dec 8 0.1 10G"

# (deck, control) for a full reporting run.  The DC bench is AnalogGym's
# -40..125 C sweep at 0.1 C; the sizing loop swaps in a 3-point sweep that still
# brackets 25 C, which is all Power / vos25 / vout25 need.
FULL_DECKS: List[tuple] = [
    ("tb_gain.cir", _AC_SWEEP),
    ("tb_cmrr.cir", _AC_SWEEP),
    ("tb_psrrp.cir", _AC_SWEEP),
    ("tb_psrrn.cir", _AC_SWEEP),
    # Swept hot-to-cold on purpose.  Starting at -40 C the operating point does
    # not solve, and a DC sweep whose first point fails produces no data at all
    # -- every measurement then reports "out of interval", including Power and
    # the offset.  From 125 C the first point solves and continuation carries
    # the rest down.  The measurement window is unchanged.
    ("tb_dc.cir", "dc temp 125 -40 -0.1"),
]

# The sizing loop runs every deck the score reads -- omitting cmrr/psrr would
# leave a constant penalty the search can never retire, so it would burn its
# whole budget instead of stopping when the design is good.  The DC bench drops
# to a 3-point sweep that still brackets 25 C, which is all Power/vos25 need.
FAST_DECKS: List[tuple] = [
    ("tb_gain.cir", _AC_SWEEP_FAST),
    ("tb_cmrr.cir", _AC_SWEEP_FAST),
    ("tb_psrrp.cir", _AC_SWEEP_FAST),
    ("tb_dc.cir", "dc temp 20 30 5"),
]


def write_design(out_dir: Path, topo: Topology, design: AmpDesign) -> Path:
    """Write a complete, runnable design directory for this tree's tech."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = ModelLibrary()
    netlist = emit_netlist(topo, design, lib)
    lib.write(out_dir / MODELS_FILE)
    (out_dir / "netlist.spice").write_text(netlist)
    (out_dir / "design.json").write_text(design.to_json())
    benches = emit_testbenches(topo, design)
    if "tb_tran_altns.cir" not in benches:
        # Regeneration must also remove helpers emitted by older tool versions.
        (out_dir / "tb_tran_altns.cir").unlink(missing_ok=True)
    for name, text in benches.items():
        (out_dir / name).write_text(text)
    return out_dir
