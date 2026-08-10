"""SMCNR_SE_2st_AMP -- the amplifier shipped in AnalogGym's sensing_front_end.

Every other sensing front end is a PTAT stack measured on a DC temperature
sweep; this one is a MOS-only two-stage Miller amplifier (pch input pair, nch
mirror load, 1:2:10 pch bias mirror, nch output device, RC nulling network,
ideal 100 nA bias) with its own AC bench, so it gets its own builder and gates
instead of the ``sfe.py`` PTAT flow (``size_sfe.py`` lists it in SKIP).

The topology and the 1:2:10 mirror ratios are the shipped ones; geometry, bias
current, compensation and common mode are re-designed for this tree's rail.
The design vector is one ``RoleGeom`` per matched group:

    n_mirror   xm1/xm3 (nch mirror load, m) and xm4 (output nch, 10*m -- the
               source keeps xm4 at the same current density as the load diode,
               which is what zeroes the systematic offset, so the group is
               shared here too)
    p_in       xm0/xm2, the matched pch input pair
    p_bias     xm7/xm6/xm5, the 1:2:10 pch bias mirror (m, 2*m, 10*m)

Benches (AnalogGym's TB_AC_SMCNR_SE_2st_AMP arrangement at this tech's design
point): ``tb_ac.cir`` closes DC unity feedback through a 1T inductor and
injects AC through a 1T capacitor into the inverting input; gain / GBW / phase
at 0 dB come out of one sweep.  The deck's ``phase_in_deg`` is ``abs()`` of
the principal-value phase at crossover and is kept as the raw auditable
reading; the gated margin is ``pm_true``, computed by the wrap-aware
stability runner (``tools/acstab.py``) from the unwrapped full-sweep dump --
the same measure the amplifier and LDO loop benches gate on.
``tb_dc.cir`` is the identical operating point on the category's temperature
sweep and reads the supply current for power.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acstab import run_deck_auto
from amp_spec import _shortfall
from build_amp import RoleGeom
from meas import SimError
from pycmg_lib import L_MAX_NM, MODELS_FILE, ModelLibrary, TECH, VDD

ROOT = Path(__file__).resolve().parents[1]

NAME = "SMCNR_SE_2st_AMP"
GROUPS = ("n_mirror", "p_in", "p_bias")

# (deck, ngspice control) for a full reporting run.  The DC control is the
# sensing_front_end category sweep, so run_all/finalize can drive both decks
# from the category's control table.
AC_CONTROL = "ac dec 20 0.1 1G"
DC_CONTROL = "dc temp -20 120 0.5"
DECKS: List[Tuple[str, str]] = [("tb_ac.cir", AC_CONTROL),
                                ("tb_dc.cir", DC_CONTROL)]


@dataclass
class SfeAmpDesign:
    """Full design vector: per-group geometry plus bias and compensation."""
    vdd: float = VDD
    vcm: float = 0.25 * VDD
    cload: float = 1e-12
    ibias: float = 200e-9
    r0: float = 100e3
    c0: float = 2e-12
    groups: Dict[str, RoleGeom] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"vdd": self.vdd, "vcm": self.vcm, "cload": self.cload,
             "ibias": self.ibias, "r0": self.r0, "c0": self.c0,
             "groups": {k: asdict(v) for k, v in self.groups.items()}},
            indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "SfeAmpDesign":
        d = json.loads(text)
        return SfeAmpDesign(
            vdd=d["vdd"], vcm=d["vcm"], cload=d["cload"], ibias=d["ibias"],
            r0=d["r0"], c0=d["c0"],
            groups={k: RoleGeom(**v) for k, v in d["groups"].items()})


def default_design() -> SfeAmpDesign:
    """The qualified design point, expressed tech-parametrically.

    Long-L devices everywhere buy the two gm*ro products the 60 dB gate lives
    on; the lvt input pair keeps ~0.3 V of tail headroom at VCM = 0.25*VDD.
    ibias is 2x the source's 100 nA for robustness across the -20..120 C
    sweep; the RC nulling network carries over (R0 >> 1/gm of the output
    device, so the compensation zero is LHP and adds lead at crossover).
    """
    l_long = float(L_MAX_NM)
    return SfeAmpDesign(
        vdd=VDD, vcm=0.25 * VDD, cload=1e-12, ibias=200e-9,
        r0=100e3, c0=2e-12,
        groups={
            "n_mirror": RoleGeom(vt="svt", l_nm=l_long, nfin=2, m=2),
            "p_in": RoleGeom(vt="lvt", l_nm=l_long, nfin=4, m=4),
            "p_bias": RoleGeom(vt="svt", l_nm=l_long, nfin=2, m=2),
        })


def _eng(x: float) -> str:
    return f"{x:.6g}"


def emit_netlist(design: SfeAmpDesign, lib: ModelLibrary) -> str:
    """Render the shipped topology with this tree's devices."""
    for g in GROUPS:
        if g not in design.groups:
            raise KeyError(f"{NAME}: design is missing group {g!r}")
    nm = design.groups["n_mirror"]
    pi = design.groups["p_in"]
    pb = design.groups["p_bias"]
    n_model = lib.model_name("n", nm.vt, nm.l_nm * 1e-9, nm.nfin)
    in_model = lib.model_name("p", pi.vt, pi.l_nm * 1e-9, pi.nfin)
    b_model = lib.model_name("p", pb.vt, pb.l_nm * 1e-9, pb.nfin)
    lines = [
        f"* {NAME} -- AnalogGym two-stage amplifier on {TECH} BSIM-CMG "
        "(LEVEL=72)",
        "* Connectivity and the 1:2:10 mirror ratios are the shipped ones; "
        f"geometry is re-designed for {design.vdd:g} V FinFET.",
        f".subckt {NAME} vdda gnda vin vip vout",
        f"Nm1 outp outp gnda gnda {n_model} m={nm.m}   $ n_mirror",
        f"Nm3 outn outp gnda gnda {n_model} m={nm.m}   $ n_mirror",
        f"Nm7 ibias ibias vdda vdda {b_model} m={pb.m}   $ p_bias 1x",
        f"Nm6 net53 ibias vdda vdda {b_model} m={2 * pb.m}   $ p_bias 2x",
        f"Nm5 vout ibias vdda vdda {b_model} m={10 * pb.m}   $ p_bias 10x",
        f"Nm2 outn vip net53 vdda {in_model} m={pi.m}   $ p_in",
        f"Nm0 outp vin net53 vdda {in_model} m={pi.m}   $ p_in",
        f"Nm4 vout outn gnda gnda {n_model} m={10 * nm.m}   $ n_mirror 10x",
        f"ibias0 ibias 0 {_eng(design.ibias)}",
        f"r0 net027 vout {_eng(design.r0)}",
        f"c0 outn net027 {_eng(design.c0)}",
        f".ends {NAME}",
    ]
    return "\n".join(lines) + "\n"


_TB_COMMON = """\
.include ./{models_file}
.include ./netlist.spice

.PARAM supply_voltage = {vdd}
.PARAM VCM = {vcm}
.PARAM PARAM_CLOAD = {cload}
Vvdda vdda 0 'supply_voltage'
Vgnda gnda 0 0

* Differential path, broken by a huge L/C so DC bias is preserved.  Both ends
* of the 1T inductor are seeded: with vout alone the operating point falls
* through gmin/source stepping into the transient fallback and never solves.
Vin signal_in 0 dc 'VCM'{acspec}
Lfb vout vfb 1T
Cin vfb signal_in 1T
xi1 vdda gnda vfb vin_p vout {name}
Vip vin_p 0 'VCM'
Cload vout 0 'PARAM_CLOAD'
* Internal seeds (rough rail fractions) let the cold end of the temperature
* sweep solve directly instead of through the transient-op fallback.
.nodeset V(vout)={vcm} V(vfb)={vcm}
.nodeset V(xi1.outp)={vmirror} V(xi1.outn)={vmirror} V(xi1.ibias)={vmirror}
.nodeset V(xi1.net53)={vtail}
"""

TB_AC = """\
* {name} -- AC gain / GBW / phase bench on {tech} BSIM-CMG
* Measurements follow AnalogGym's TB_AC_{name}; supply, common mode and load
* are the {tech} design point.  phase_in_deg is abs() of the principal value
* at crossover (raw auditable reading); the gated margin is pm_true from the
* runner's unwrapped full-sweep dump (tools/acstab.py).
""" + _TB_COMMON + """\

.measure ac dcgain find vdb(vout) at=0.1
.measure ac gain_bandwidth_product when vdb(vout)=0
.measure ac ph_rad find vp(vout) when vdb(vout)=0
.measure ac phase_in_deg param='abs(ph_rad)*180/3.1416'
.end
"""

TB_DC = """\
* {name} -- DC / temperature / power bench on {tech} BSIM-CMG
* Same closed-loop operating point as tb_ac.cir on the sensing_front_end
* category's temperature sweep; power is the 25 C supply draw.
""" + _TB_COMMON + """\

.measure dc ivdd25 FIND I(Vvdda) AT=25
.measure dc power param='-1*ivdd25*supply_voltage'
.measure dc vout25 FIND V(vout) AT=25
.end
"""


def emit(out_dir: Path, design: SfeAmpDesign) -> Path:
    """Write a complete, runnable design directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = ModelLibrary()
    netlist = emit_netlist(design, lib)
    lib.write(out_dir / MODELS_FILE)
    (out_dir / "netlist.spice").write_text(netlist)
    (out_dir / "design.json").write_text(design.to_json())
    common = dict(name=NAME, tech=TECH, models_file=MODELS_FILE,
                  vdd=_eng(design.vdd), vcm=_eng(design.vcm),
                  cload=_eng(design.cload),
                  vmirror=_eng(0.45 * design.vdd),
                  vtail=_eng(0.625 * design.vdd))
    (out_dir / "tb_ac.cir").write_text(TB_AC.format(**common, acspec=" ac 1"))
    (out_dir / "tb_dc.cir").write_text(TB_DC.format(**common, acspec=""))
    return out_dir


# ---------------------------------------------------------------------------
# Targets and scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SfeAmpTargets:
    """What a healthy port of this amplifier looks like on this tree's rail."""
    gain_db: float = 60.0     # closed-bench DC gain, at least
    gbw_hz: float = 1e4       # unity-gain frequency, at least
    pm_deg: float = 45.0      # TRUE (unwrapped) margin at 0 dB, at least
    power_w: float = 100e-6   # total supply power, at most


SFE_AMP_TARGETS = SfeAmpTargets()


def sfe_amp_score(m: Dict[str, Optional[float]],
                  t: SfeAmpTargets = SFE_AMP_TARGETS) -> float:
    """Penalty; 0 when every gate is met (same shortfall shape as amp_spec)."""
    gbw = m.get("gain_bandwidth_product")
    # True margin when the stability runner produced it; raw principal-value
    # phase only as a fallback for metric sets measured without the dump.
    pm = m["pm_true"] if "pm_true" in m else m.get("phase_in_deg")
    return (
        _shortfall(m.get("dcgain"), t.gain_db, higher_is_better=True,
                   scale=10.0)
        + _shortfall(math.log10(gbw) if gbw and gbw > 0 else None,
                     math.log10(t.gbw_hz), higher_is_better=True, scale=0.3)
        + _shortfall(pm, t.pm_deg,
                     higher_is_better=True, scale=15.0)
        + _shortfall(m.get("power"), t.power_w,
                     higher_is_better=False, scale=1e-4)
    )


def sfe_amp_report(m: Dict[str, Optional[float]],
                   t: SfeAmpTargets = SFE_AMP_TARGETS) -> Dict[str, bool]:
    """Per-gate pass/fail against the targets."""
    gbw = m.get("gain_bandwidth_product")
    return {
        "gain": m.get("dcgain") is not None and m["dcgain"] >= t.gain_db,
        "gbw": gbw is not None and gbw >= t.gbw_hz,
        # Gated on the TRUE (unwrapped) margin -- see tools/acstab.py.
        "pm": m.get("pm_true") is not None
              and m["pm_true"] >= t.pm_deg,
        "power": m.get("power") is not None and m["power"] <= t.power_w,
    }


# ---------------------------------------------------------------------------
# Build / verify entry points
# ---------------------------------------------------------------------------
def rebuild() -> Dict:
    """Re-emit from the shipped design.json (or defaults), re-measure, and
    rewrite result.json.  This is the design's retune.py entry point: the
    design point is fixed, so 'retuning' is a rebuild plus a fresh verdict."""
    out = ROOT / "sensing_front_end" / NAME
    dj = out / "design.json"
    design = (SfeAmpDesign.from_json(dj.read_text()) if dj.exists()
              else default_design())
    emit(out, design)

    metrics: Dict[str, Optional[float]] = {}
    errors: List[str] = []
    for deck, control in DECKS:
        try:
            metrics.update(run_deck_auto(out / deck, control, out,
                                         deck.replace(".cir", ""),
                                         timeout=900))
        except SimError as exc:
            errors.append(f"{deck}: {str(exc).splitlines()[0]}")

    result = {"design": NAME, "category": "sensing_front_end",
              "metrics": metrics, "pass": sfe_amp_report(metrics),
              "score": sfe_amp_score(metrics), "errors": errors}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    r = rebuild()
    p = r["pass"]
    print(f"{NAME}: pass={sum(p.values())}/{len(p)} score={r['score']:.4f}")
    for key in ("dcgain", "gain_bandwidth_product", "phase_in_deg", "pm_true",
                "power", "vout25"):
        print(f"  {key:24s} {r['metrics'].get(key)}")
    for e in r["errors"]:
        print(f"  ERROR {e}")


if __name__ == "__main__":
    main()
