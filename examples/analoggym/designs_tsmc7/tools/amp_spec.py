"""Initial design point and scoring for the TSMC FinFET amplifiers.

AnalogGym tags every transistor with a role (``gm1_PMOS``, ``BIASCM_NMOS``,
``LOAD2_NMOS``, ...).  That tagging is the whole reason a topology-agnostic
re-design is possible: the role says what a device is *for*, so a sizing rule
can be written once and applied to all seventeen amplifiers.

The targets below are what "reasonable" means for a 0.8 V FinFET amplifier
driving AnalogGym's 500 pF load; ``score`` turns a measurement dict into a
single penalty the sizing loop minimises.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from build_amp import AmpDesign, RoleGeom
from pycmg_lib import VDD, scale_l

# ---------------------------------------------------------------------------
# Role -> initial geometry
# ---------------------------------------------------------------------------
# (regex on the role name, vt, L[nm], NFIN, m).  First match wins.
#
# Rationale at 0.8 V / 500 pF:
#   * output stage (gm3/gmf2) needs ~10 mS to keep its pole above GBW, which is
#     hundreds of microamps -- large m, short-ish L.
#   * input stage sets GBW = gm1/(2*pi*Cc) at a few tens of microsiemens, so it
#     stays small and long-channel for gain.
#   * bias mirrors and active loads run long (120 nm) for output resistance and
#     matching.
_ROLE_RULES: List[Tuple[str, str, float, int, int]] = [
    (r"^gm3_",     "svt",  scale_l(36), 12, 32),   # output transconductance
    (r"^gmf2_",    "svt",  scale_l(36), 10, 16),   # output feedforward path
    (r"^gmf1?_",   "svt",  scale_l(60),  8,  8),   # inner feedforward
    (r"^gm2_",     "svt",  scale_l(60),  8,  8),   # second stage
    (r"^gm2[0-9]", "svt",  scale_l(60),  6,  4),
    (r"^gm[45689]_", "svt", scale_l(60),  6,  4),  # extra gain stages
    (r"^gmb",      "svt",  scale_l(60),  6,  4),   # feedback / buffer paths
    (r"^gmc_",     "svt",  scale_l(60),  6,  4),
    (r"^gmt_",     "svt",  scale_l(60),  6,  4),
    (r"^gma",      "svt",  scale_l(60),  6,  4),
    (r"^gm1_",     "svt",  scale_l(60),  8,  4),   # input differential pair
    (r"^ma1_",     "svt",  scale_l(60),  6,  4),
    (r"^AZC",      "svt",  scale_l(60),  4,  2),   # auto-zero chopping devices
    (r"^LOAD",     "svt", scale_l(120),  4,  4),   # active loads
    (r"^HSBCM_",   "svt", scale_l(120),  4,  4),   # high-swing cascode bias
    (r"^BIASCM_",  "svt", scale_l(120),  4,  4),   # bias current mirrors
]

_DEFAULT = ("svt", scale_l(60.0), 4, 4)


def initial_roles(roles: List[str]) -> Dict[str, RoleGeom]:
    """Assign every role its starting geometry."""
    out: Dict[str, RoleGeom] = {}
    for role in roles:
        vt, l_nm, nfin, m = _DEFAULT
        for pattern, r_vt, r_l, r_nfin, r_m in _ROLE_RULES:
            if re.match(pattern, role, re.IGNORECASE):
                vt, l_nm, nfin, m = r_vt, r_l, r_nfin, r_m
                break
        out[role] = RoleGeom(vt=vt, l_nm=l_nm, nfin=nfin, m=m)
    return out


# Starting values for the parameterised passives.  Miller capacitors dominate
# GBW, so they are the loop's main compensation knob.
_PASSIVE_DEFAULTS: List[Tuple[str, float]] = [
    (r"^CAPACITOR_0$", 2e-12),
    (r"^CAPACITOR_1$", 1e-12),
    (r"^CAPACITOR_2$", 1e-12),
    (r"^CURRENT_\d+_BIAS$", 5e-6),
    (r"^RESISTOR_\d+$", 20e3),
]


def initial_passives(names: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in names:
        for pattern, val in _PASSIVE_DEFAULTS:
            if re.match(pattern, name, re.IGNORECASE):
                out[name] = val
                break
        else:
            raise KeyError(f"no default for passive {name!r}")
    return out


def initial_design(roles: List[str], passives: List[str]) -> AmpDesign:
    return AmpDesign(
        vdd=VDD, vcm=0.25 * VDD, cload=500e-12, gbw_ideal=1e6,
        roles=initial_roles(roles),
        passives=initial_passives(passives),
    )


# ---------------------------------------------------------------------------
# Targets and scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Targets:
    """What a healthy 0.8 V amplifier on this bench looks like."""
    gain_db: float = 60.0        # open-loop DC gain, at least
    gbw_hz: float = 3e5          # unity-gain frequency, at least
    pm_deg: float = 45.0         # TRUE (unwrapped) margin at 0 dB, at least
    power_w: float = 2.0e-3      # total supply power, at most
    vos_v: float = 20e-3         # |input-referred offset| at 25 C, at most
    cmrr_db: float = -30.0       # common-mode gain at 0.1 Hz, at most
    psrr_db: float = -30.0       # supply gain at 0.1 Hz, at most
    tc_min: float = 0.0          # full-range output TC, at least
    tc_max: float = 0.02         # full-range output TC, at most
    slew_v_per_us: float = 0.01  # each slew direction, at least


TARGETS = Targets()

# The loop optimises against a stricter set than it reports against, so a design
# lands with margin rather than exactly on a threshold.
OPT_TARGETS = Targets(gain_db=75.0, gbw_hz=5e5, pm_deg=55.0, power_w=1.0e-3,
                      vos_v=10e-3, cmrr_db=-35.0, psrr_db=-35.0,
                      tc_min=0.0, tc_max=0.018, slew_v_per_us=0.012)


def _shortfall(value: Optional[float], target: float, *,
               higher_is_better: bool, scale: float) -> float:
    """Normalised penalty: 0 once the target is met, growing linearly past it.

    A missing measurement is the worst outcome -- an amplifier whose gain never
    reaches 0 dB has no GBW to report -- so it takes a fixed large penalty.
    """
    if value is None or not math.isfinite(value):
        return 10.0
    if higher_is_better:
        return min(10.0, max(0.0, (target - value) / scale))
    return min(10.0, max(0.0, (value - target) / scale))


def _range_shortfall(value: Optional[float], lower: float, upper: float,
                     *, scale: float) -> float:
    """Finite-safe penalty for a value that must remain in a closed range."""
    if value is None or not math.isfinite(value):
        return 10.0
    if value < lower:
        return min(10.0, (lower - value) / scale)
    if value > upper:
        return min(10.0, (value - upper) / scale)
    return 0.0


def score(m: Dict[str, Optional[float]], t: Targets = TARGETS) -> float:
    """Total penalty for one measurement set; 0 means every target is met."""
    gbw = m.get("gain_bandwidth_product")
    # True (wrap-aware) margin from the stability runner (tools/acstab.py)
    # when it ran; the raw principal-value phase is a search-only fallback
    # for the fast decks, which skip the sweep dump.
    pm = m["pm_true"] if "pm_true" in m else m.get("phase_in_deg")
    return (
        _shortfall(m.get("dcgain"), t.gain_db, higher_is_better=True, scale=10.0)
        + _shortfall(math.log10(gbw) if gbw and gbw > 0 else None,
                     math.log10(t.gbw_hz), higher_is_better=True, scale=0.3)
        + _shortfall(pm, t.pm_deg,
                     higher_is_better=True, scale=15.0)
        + _shortfall(m.get("power"), t.power_w,
                     higher_is_better=False, scale=1e-3)
        + _shortfall(abs(m["vos25"]) if m.get("vos25") is not None else None,
                     t.vos_v, higher_is_better=False, scale=20e-3)
        + 0.5 * _shortfall(m.get("cmrrdc"), t.cmrr_db,
                           higher_is_better=False, scale=20.0)
        + 0.5 * _shortfall(m.get("dcpsrp"), t.psrr_db,
                           higher_is_better=False, scale=20.0)
        + 0.5 * _shortfall(m.get("dcpsrn"), t.psrr_db,
                           higher_is_better=False, scale=20.0)
        + _range_shortfall(m.get("tc"), t.tc_min, t.tc_max, scale=0.005)
        + 0.5 * _shortfall(m.get("sr_rise"), t.slew_v_per_us,
                           higher_is_better=True, scale=0.01)
        + 0.5 * _shortfall(m.get("sr_fall"), t.slew_v_per_us,
                           higher_is_better=True, scale=0.01)
    )


def report(m: Dict[str, Optional[float]], t: Targets = TARGETS) -> Dict[str, bool]:
    """Per-metric pass/fail against the targets."""
    gbw = m.get("gain_bandwidth_product")
    vos = m.get("vos25")
    return {
        "gain": m.get("dcgain") is not None and m["dcgain"] >= t.gain_db,
        "gbw": gbw is not None and gbw >= t.gbw_hz,
        # Gated on the TRUE margin, unwrapped across the sweep from the
        # low-frequency anchor (tools/acstab.py).  The wrapped principal-value
        # ``phase_in_deg`` cannot pass this gate on its own: a loop whose
        # phase fell through -180 before crossover reads a NEGATIVE pm_true.
        "pm": m.get("pm_true") is not None and m["pm_true"] >= t.pm_deg,
        "power": m.get("power") is not None and m["power"] <= t.power_w,
        "vos": vos is not None and abs(vos) <= t.vos_v,
        "cmrr": m.get("cmrrdc") is not None and m["cmrrdc"] <= t.cmrr_db,
        "psrr": m.get("dcpsrp") is not None and m["dcpsrp"] <= t.psrr_db,
        "psrrn": m.get("dcpsrn") is not None and m["dcpsrn"] <= t.psrr_db,
        "temperature": m.get("tc") is not None
                       and t.tc_min <= m["tc"] <= t.tc_max,
        "slew": m.get("sr_rise") is not None
                and m["sr_rise"] >= t.slew_v_per_us
                and m.get("sr_fall") is not None
                and m["sr_fall"] >= t.slew_v_per_us,
    }
