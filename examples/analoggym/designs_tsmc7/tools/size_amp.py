"""Re-size one AnalogGym amplifier for this tree's tech until it meets target.

The design vector has one geometry per *role*, not per transistor, so the search
space stays small and matched devices stay matched.  It is reduced further to a
handful of physically meaningful knobs -- compensation strength, bias level,
per-stage drive, channel length for gain, and the input common mode -- because
those are the levers that actually move gain / GBW / phase margin, and moving a
role's ``m`` in isolation mostly just breaks a mirror ratio.

Search is a pattern (coordinate) search: deterministic, derivative-free, and it
copes with the discontinuities that come from a measurement simply failing when
the gain never crosses 0 dB.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from build_amp import AmpDesign, RoleGeom, write_design, tran_control, FAST_DECKS
from amp_spec import OPT_TARGETS, TARGETS, Targets, score, report
from meas import run_deck, SimError
from pycmg_lib import L_MAX_NM, L_MIN_NM, VDD, scale_l, snap_l, snap_nfin
from skyparse import Topology

# Role groups the knobs act on.  Membership is by the AnalogGym role tag.
_OUT_ROLES = ("gm3_", "gmf2_")
_IN_ROLES = ("gm1_",)
_BIAS_ROLES = ("BIASCM_", "LOAD", "HSBCM_")


def _group(role: str) -> str:
    r = role.lower()
    if any(r.startswith(p.lower()) for p in _OUT_ROLES):
        return "out"
    if any(r.startswith(p.lower()) for p in _IN_ROLES):
        return "in"
    if any(r.startswith(p.lower()) for p in _BIAS_ROLES):
        return "bias"
    return "mid"


@dataclass
class Knobs:
    """Reduced, continuous design vector.  All k_* are multiplicative."""
    k_cc: float = 1.0        # every compensation capacitor
    k_ibias: float = 1.0     # every bias current source
    k_out: float = 1.0       # m of the output-stage roles
    k_in: float = 1.0        # m of the input pair
    k_mid: float = 1.0       # m of the intermediate stages
    k_bias: float = 1.0      # m of mirrors and active loads
    l_bias: float = scale_l(120)  # channel length [nm] of mirrors and loads
    l_out: float = scale_l(36)    # channel length [nm] of the output stage
    l_in: float = scale_l(60)     # channel length [nm] of the input pair
    vcm: float = 0.25 * VDD       # input common mode [V]

    def as_list(self) -> List[float]:
        return [self.k_cc, self.k_ibias, self.k_out, self.k_in, self.k_mid,
                self.k_bias, self.l_bias, self.l_out, self.l_in, self.vcm]

    @staticmethod
    def from_list(v: List[float]) -> "Knobs":
        return Knobs(*v)


# (name, lower, upper, multiplicative?) -- multiplicative knobs step in log space
_BOUNDS: List[Tuple[str, float, float, bool]] = [
    ("k_cc",    0.2,  60.0, True),
    ("k_ibias", 0.1,  40.0, True),
    ("k_out",   0.2, 100.0, True),
    ("k_in",    0.1,  10.0, True),
    ("k_mid",   0.2,  20.0, True),
    ("k_bias",  0.2,  20.0, True),
    ("l_bias",  max(L_MIN_NM, scale_l(20)), L_MAX_NM, False),
    ("l_out",   L_MIN_NM, L_MAX_NM, False),
    ("l_in",    max(L_MIN_NM, scale_l(20)), L_MAX_NM, False),
    ("vcm",     0.10 * VDD, 0.50 * VDD, False),
]


def _clip(knobs: Knobs) -> Knobs:
    vals = knobs.as_list()
    for i, (_, lo, hi, _mul) in enumerate(_BOUNDS):
        vals[i] = min(hi, max(lo, vals[i]))
    return Knobs.from_list(vals)


def apply_knobs(base: AmpDesign, knobs: Knobs) -> AmpDesign:
    """Turn the reduced knob vector into a full design vector."""
    knobs = _clip(knobs)
    roles: Dict[str, RoleGeom] = {}
    for name, g in base.roles.items():
        grp = _group(name)
        k = {"out": knobs.k_out, "in": knobs.k_in,
             "mid": knobs.k_mid, "bias": knobs.k_bias}[grp]
        l_nm = {"out": knobs.l_out, "in": knobs.l_in,
                "bias": knobs.l_bias, "mid": knobs.l_in}[grp]
        roles[name] = RoleGeom(
            vt=g.vt,
            l_nm=round(snap_l(l_nm * 1e-9) * 1e9),
            nfin=snap_nfin(g.nfin),
            m=max(1, int(round(g.m * k))),
        )

    passives: Dict[str, float] = {}
    for name, val in base.passives.items():
        up = name.upper()
        if up.startswith("CAPACITOR"):
            passives[name] = val * knobs.k_cc
        elif "CURRENT" in up:
            passives[name] = val * knobs.k_ibias
        else:
            passives[name] = val
    return replace(base, roles=roles, passives=passives, vcm=knobs.vcm)


BAD = 1e3


def make_evaluator(topo: Topology, base: AmpDesign, work: Path,
                   targets: Targets = OPT_TARGETS) -> Callable[[Knobs], Tuple[float, Dict]]:
    """Build a knobs -> (penalty, measurements) function backed by ngspice."""
    work.mkdir(parents=True, exist_ok=True)

    def evaluate(knobs: Knobs) -> Tuple[float, Dict]:
        design = apply_knobs(base, knobs)
        try:
            write_design(work, topo, design)
        except Exception:
            return BAD, {}
        merged: Dict = {}
        for deck, control in FAST_DECKS:
            try:
                # A healthy deck is well under a second; a candidate whose
                # operating point does not solve costs ~40 s in ngspice's
                # gmin / source / transient-op fallback chain.  Measured on the
                # real search, failing candidates were 90 % of the wall clock,
                # so the cap is set just above a healthy run.  A design needing
                # heroic convergence aid is not one worth keeping anyway.
                merged.update(run_deck(work / deck, control, work,
                                       deck.replace(".cir", ""), timeout=6))
            except SimError:
                return BAD, merged
        s = score(merged, targets)
        # Tie-break among designs that already meet every target: prefer the
        # one that spends less power.  Small enough never to trade away a target.
        power = merged.get("power")
        if power and power > 0:
            s += 0.02 * power / targets.power_w
        return s, merged

    return evaluate


def pattern_search(evaluate: Callable[[Knobs], Tuple[float, Dict]],
                   start: Knobs, *, max_evals: int = 300,
                   log: Optional[Callable[[str], None]] = None
                   ) -> Tuple[Knobs, float, Dict]:
    """Coordinate pattern search with a shrinking step, minimising the penalty."""
    best = _clip(start)
    best_score, best_meas = evaluate(best)
    evals = 1
    # Multiplicative knobs step by a factor; additive ones by an absolute amount.
    step = [2.5, 2.5, 3.0, 2.0, 2.0, 2.0,
            60.0 * L_MAX_NM / 240, 40.0 * L_MAX_NM / 240,
            40.0 * L_MAX_NM / 240, 0.075 * VDD]

    while evals < max_evals and max(
            [s if not _BOUNDS[i][3] else s - 1.0 for i, s in enumerate(step)]
    ) > 1e-3 and best_score > 0:
        improved = False
        for i, (name, _lo, _hi, mul) in enumerate(_BOUNDS):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                vals = best.as_list()
                if mul:
                    vals[i] *= step[i] ** direction
                else:
                    vals[i] += direction * step[i]
                cand = _clip(Knobs.from_list(vals))
                if cand.as_list() == best.as_list():
                    continue
                s, meas = evaluate(cand)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_meas = cand, s, meas
                    improved = True
                    if log:
                        log(f"    eval {evals:3d}  {name:8s} "
                            f"{'x' if mul else '+'}{step[i]:g}^{direction:+d}"
                            f"  -> {best_score:.4f}")
                    break
                if evals >= max_evals:
                    break
        if not improved:
            step = [1 + (s - 1) / 2 if _BOUNDS[i][3] else s / 2
                    for i, s in enumerate(step)]
            if log:
                log(f"    shrink step (evals={evals}, score={best_score:.4f})")
    return best, best_score, best_meas
