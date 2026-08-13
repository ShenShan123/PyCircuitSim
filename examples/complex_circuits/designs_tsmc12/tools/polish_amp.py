"""Warm-start, per-role refinement for amplifiers the reduced search missed.

``size_amp.py`` searches a 10-knob reduced vector in which whole role groups
share a multiplier.  That is what makes seventeen amplifiers tractable, but it
failed Ramos_PFC_Pin_3 structurally: its ``gmf1`` device is the *output
pull-up* (drain on VOUT), yet the reduced vector files it under the "mid"
group -- so the search drove the pull-down (``gm3``, ``k_out`` = 81) and the
pull-up (``k_mid`` = 0.25) a thousand-fold apart, and parked both compensation
capacitors on the shared ``k_cc`` bound at a frozen 2:1 ratio.

This tool searches the full design vector instead -- (L, NFIN, m) per role,
every passive on its own axis, plus the common mode -- warm-started from the
shipped ``design.json`` and from structural seeds that restore the push-pull
balance.  Only worth the dimensionality for the designs the reduced search
left partial.

The shipped directory is replaced only when the full-bench score improves.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from amp_spec import OPT_TARGETS, TARGETS, initial_design, report, score
from build_amp import (AmpDesign, FULL_DECKS, RoleGeom,
                       _AC_SWEEP, tran_control, write_design)

# The dec-8 search sweep can misread the phase at 0 dB when the phase wraps
# near the crossing -- a candidate tuned on it can collapse on the dec-20
# reporting sweep.  The polish search therefore measures at full resolution;
# it only runs for one design at a time, so the extra cost is acceptable.
SEARCH_DECKS: List[Tuple[str, str]] = [
    ("tb_gain.cir", _AC_SWEEP),
    ("tb_cmrr.cir", _AC_SWEEP),
    ("tb_psrrp.cir", _AC_SWEEP),
    ("tb_psrrn.cir", _AC_SWEEP),
    ("tb_dc.cir", "dc temp 125 -40 -5"),
]
SEARCH_TIMEOUTS: Dict[str, float] = {
    "tb_dc.cir": 45.0,
    "tb_tran.cir": 45.0,
}
from acstab import run_deck_auto
from meas import SimError, run_deck
from pycmg_lib import NFIN_MAX, NFIN_MIN, VDD, scale_l, snap_l, snap_nfin
from skyparse import parse_netlist

ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT.parent / "designs" / "amplifier"
BAD = 1e3
M_MAX = 20000
Measurements = Dict[str, Optional[float]]
Quality = Tuple[int, int, float]
SearchQuality = Tuple[int, float]
_SEARCH_MEAS_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)\s*(?:\S+=.*)?$"
)


# ---------------------------------------------------------------------------
# Full design vector <-> flat list
# ---------------------------------------------------------------------------
def _vec(d: AmpDesign, roles: List[str], pkeys: List[str]) -> List[float]:
    v: List[float] = []
    for r in roles:
        g = d.roles[r]
        v += [g.l_nm, float(g.nfin), float(g.m)]
    v += [d.passives[p] for p in pkeys]
    v.append(d.vcm)
    return v


def _unvec(v: List[float], d: AmpDesign, roles: List[str],
           pkeys: List[str]) -> AmpDesign:
    rg: Dict[str, RoleGeom] = {}
    for i, r in enumerate(roles):
        rg[r] = RoleGeom(
            vt=d.roles[r].vt,
            l_nm=round(snap_l(v[3 * i] * 1e-9) * 1e9),
            nfin=snap_nfin(v[3 * i + 1]),
            m=max(1, min(M_MAX, int(round(v[3 * i + 2])))),
        )
    passives = dict(d.passives)
    for j, p in enumerate(pkeys):
        passives[p] = max(1e-15, v[3 * len(roles) + j])
    vcm = min(0.55 * VDD, max(0.10 * VDD, v[-1]))
    return replace(d, roles=rg, passives=passives, vcm=vcm)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def _tweak(base: AmpDesign,
           roles: Optional[Dict[str, Dict[str, Any]]] = None,
           passives: Optional[Dict[str, float]] = None,
           vcm: Optional[float] = None) -> AmpDesign:
    rg = dict(base.roles)
    for name, upd in (roles or {}).items():
        if name in rg:
            rg[name] = replace(rg[name], **upd)
    pv = dict(base.passives)
    pv.update(passives or {})
    return replace(base, roles=rg, passives=pv,
                   vcm=base.vcm if vcm is None else vcm)


def _drive_seed(base: AmpDesign, *, drive_scale: float, cap_scale: float,
                vcm: float) -> AmpDesign:
    """Raise bias/output drive and reduce compensation in a bounded way."""
    roles: Dict[str, Dict[str, Any]] = {}
    for name, geom in base.roles.items():
        if re.match(r"^(?:gm1|gm2|gm3|gmf2)", name, re.IGNORECASE):
            roles[name] = {
                "m": max(1, min(M_MAX, int(round(geom.m * drive_scale))))
            }
    passives = {
        name: (value * drive_scale if name.upper().startswith("CURRENT_")
               else value * cap_scale)
        for name, value in base.passives.items()
        if (name.upper().startswith("CURRENT_")
            or name.upper().startswith("CAPACITOR_"))
    }
    return _tweak(base, roles=roles, passives=passives, vcm=vcm)


def _seeds(warm: AmpDesign, fresh: AmpDesign, n: int) -> List[AmpDesign]:
    out = [warm,
           _tweak(warm, vcm=0.4 * VDD),
           _tweak(warm, vcm=0.5 * VDD)]
    if "gmf1_PMOS" in warm.roles and "gm3_NMOS" in warm.roles:
        # Restore the push-pull: pull-up within a small factor of the
        # pull-down, mirrors strong enough to drive the output gate, and the
        # two compensation capacitors freed from the 2:1 ratio.
        out.append(_tweak(
            warm,
            roles={"gmf1_PMOS": dict(l_nm=scale_l(36), nfin=12, m=384),
                   "gm2_PMOS": dict(l_nm=scale_l(60), nfin=8, m=16),
                   "gm3_NMOS": dict(m=768),
                   "LOAD2_NMOS": dict(l_nm=scale_l(120), nfin=4, m=8),
                   "BIASCM_PMOS": dict(m=2)},
            passives={"CAPACITOR_0": 4e-12, "CAPACITOR_1": 2e-12},
            vcm=0.25 * VDD))
        out.append(_tweak(
            warm,
            roles={"gmf1_PMOS": dict(l_nm=scale_l(36), nfin=12, m=512),
                   "gm2_PMOS": dict(l_nm=scale_l(60), nfin=8, m=32),
                   "gm3_NMOS": dict(m=1024),
                   "LOAD2_NMOS": dict(l_nm=scale_l(90), nfin=6, m=16),
                   "BIASCM_PMOS": dict(m=2), "BIASCM_NMOS": dict(m=2)},
            passives={"CAPACITOR_0": 2e-12, "CAPACITOR_1": 6e-12},
            vcm=0.25 * VDD))
    out.extend([
        _drive_seed(warm, drive_scale=4.0, cap_scale=0.5,
                    vcm=0.4 * VDD),
        fresh,
        _drive_seed(fresh, drive_scale=8.0, cap_scale=0.25,
                    vcm=0.5 * VDD),
    ])

    rng = random.Random(20260731)
    bases = out[:]
    while len(out) < n:
        b = bases[rng.randrange(len(bases))]
        rg = {}
        for name, g in b.roles.items():
            if rng.random() < 0.4:
                rg[name] = dict(
                    l_nm=round(snap_l(g.l_nm * 1e-9
                                      * rng.choice([0.5, 1, 2])) * 1e9),
                    nfin=snap_nfin(g.nfin * rng.choice([0.5, 1, 2])),
                    m=max(1, min(M_MAX,
                                 int(g.m * rng.choice([0.25, 0.5, 2, 4])))))
        pv = {k: v * rng.choice([0.25, 0.5, 1, 2, 4])
              for k, v in b.passives.items()}
        out.append(_tweak(b, roles=rg, passives=pv,
                          vcm=rng.choice([0.15 * VDD, 0.25 * VDD,
                                          0.35 * VDD])))
    return out[:n]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def _complete(metrics: Measurements) -> bool:
    """Whether every gate in the complete, margin-bearing objective passes."""
    return all(report(metrics, OPT_TARGETS).values())


def _tracking_penalty(metrics: Measurements, design: AmpDesign) -> float:
    """Return search-only endpoint error normalised to the commanded step."""
    step = 0.11 * design.vdd
    val0 = design.vcm - step / 2
    val1 = design.vcm + step / 2
    penalty = 0.0
    for key, target in (("v_pre", val0), ("v_high", val1),
                        ("v_low", val0)):
        value = metrics.get(key)
        if value is None or not math.isfinite(value):
            penalty += 10.0
        else:
            penalty += abs(value - target) / step
    return penalty


def _numeric_log_metrics(log: Path) -> Measurements:
    """Parse numeric measurements preserved before an analysis failure."""
    try:
        lines = log.read_text().splitlines()
    except OSError:
        return {}
    metrics: Measurements = {}
    for line in lines:
        match = _SEARCH_MEAS_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[match.group(1).lower()] = value
    return metrics


def _quality(metrics: Measurements, errors: Sequence[str]) -> Quality:
    """Rank full-bench results without trusting a legacy partial score."""
    verdict = report(metrics, TARGETS)
    return len(errors), sum(not passed for passed in verdict.values()), score(
        metrics, TARGETS
    )


def _json_metrics(value: Any) -> Measurements:
    """Recover the numeric measurement subset from an existing result."""
    if not isinstance(value, dict):
        return {}
    metrics: Measurements = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if item is None:
            metrics[key] = None
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            metrics[key] = float(item)
    return metrics


def _json_errors(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def polish(name: str, max_evals: int = 500,
           n_seeds: int = 10) -> Dict[str, Any]:
    topo = parse_netlist(SKY / name / "netlist.spice")
    out = ROOT / "amplifier" / name
    work = ROOT / "work" / "amp_polish" / name
    work.mkdir(parents=True, exist_ok=True)

    warm = AmpDesign.from_json((out / "design.json").read_text())
    fresh = initial_design(topo.roles, topo.passive_vars)
    roles = sorted(warm.roles)
    pkeys = sorted(warm.passives)

    def evaluate(d: AmpDesign) -> Tuple[SearchQuality, Measurements, bool]:
        try:
            write_design(work, topo, d)
        except Exception:
            return (1, BAD), {}, False
        merged: Measurements = {}
        analysis_errors = 0
        decks = SEARCH_DECKS + [("tb_tran.cir", tran_control(d, topo.subckt))]
        for deck, control in decks:
            tag = deck.replace(".cir", "")
            stable_log = work / f"{tag}.log"
            stable_log.unlink(missing_ok=True)
            try:
                # tb_gain goes through the wrap-aware stability runner, so
                # the search optimises the same pm_true the gate reads.
                merged.update(run_deck_auto(work / deck, control, work,
                                            tag, timeout=SEARCH_TIMEOUTS.get(
                                                deck, 12.0)))
            except SimError:
                # The slew bench ships a second nodeset variant; some designs'
                # transient op only solves from one of the two seeds.
                if deck == "tb_tran.cir":
                    try:
                        merged.update(run_deck(
                            work / "tb_tran_altns.cir", control, work,
                            "tb_tran_altns",
                            timeout=SEARCH_TIMEOUTS.get(deck, 12.0)))
                        continue
                    except SimError:
                        pass
                merged.update(_numeric_log_metrics(stable_log))
                analysis_errors += 1
        objective = score(merged, OPT_TARGETS) + _tracking_penalty(merged, d)
        quality = analysis_errors, objective
        return quality, merged, analysis_errors == 0 and _complete(merged)

    t0 = time.time()
    if max_evals < 1:
        raise ValueError("max_evals must be positive")
    best: AmpDesign = warm
    best_quality: SearchQuality = (len(SEARCH_DECKS) + 2, float("inf"))
    best_m: Measurements = {}
    best_complete = False
    evals = 0
    seed_budget = min(max_evals, max(1, n_seeds))
    for i, seed in enumerate(_seeds(warm, fresh, seed_budget)):
        quality, m, complete = evaluate(seed)
        evals += 1
        print(f"  seed {i}: errors={quality[0]} "
              f"objective={quality[1]:.4f}", flush=True)
        if quality < best_quality:
            best, best_quality, best_m = seed, quality, m
            best_complete = complete
        if best_complete:
            break

    # L x1.6, NFIN x1.6, m x3 per role; passives x2.5; vcm +-0.06 additive.
    step = [1.6, 1.6, 3.0] * len(roles) + [2.5] * len(pkeys) + [0.075 * VDD]
    while (evals < max_evals and not best_complete
           and max(step[:-1]) > 1.05):
        improved = False
        vec = _vec(best, roles, pkeys)
        for i in range(len(vec)):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                cv = list(vec)
                if i == len(vec) - 1:
                    cv[i] += direction * step[i]
                else:
                    cv[i] *= step[i] ** direction
                cand = _unvec(cv, best, roles, pkeys)
                if _vec(cand, roles, pkeys) == _vec(best, roles, pkeys):
                    continue
                quality, m, complete = evaluate(cand)
                evals += 1
                if quality < (best_quality[0], best_quality[1] - 1e-9):
                    best, best_quality, best_m = cand, quality, m
                    best_complete = complete
                    improved = True
                    vec = _vec(best, roles, pkeys)
                    print(f"  eval {evals}: errors={best_quality[0]} "
                          f"objective={best_quality[1]:.4f}", flush=True)
                    break
        if not improved:
            step = [s / 2 if i == len(step) - 1 else 1 + (s - 1) / 2
                    for i, s in enumerate(step)]
            print(f"  shrink (evals={evals}, errors={best_quality[0]}, "
                  f"best={best_quality[1]:.4f})",
                  flush=True)
    elapsed = time.time() - t0

    # Full bench in the work directory, then keep the better design.
    write_design(work, topo, best)
    final: Measurements = {}
    errors: List[str] = []
    for deck, control in FULL_DECKS + [("tb_tran.cir", tran_control(best, topo.subckt))]:
        try:
            final.update(run_deck_auto(work / deck, control, work,
                                       deck.replace(".cir", ""), timeout=600))
        except SimError as exc:
            if deck == "tb_tran.cir":
                try:
                    final.update(run_deck(work / "tb_tran_altns.cir", control,
                                          work, "tb_tran_altns", timeout=600))
                    continue
                except SimError:
                    pass
            errors.append(f"{deck}: {exc}".split("\n")[0])

    new_quality = _quality(final, errors)
    new_score = new_quality[2]
    print(f"  candidate full-bench score {new_score:.4f} "
          f"(search errors={best_quality[0]} "
          f"objective={best_quality[1]:.4f}); metrics: "
          f"g={final.get('dcgain')} pm_true={final.get('pm_true')} "
          f"cmrr={final.get('cmrrdc')} psrr={final.get('dcpsrp')}", flush=True)
    prev = json.loads((out / "result.json").read_text())
    prev_metrics = _json_metrics(prev.get("metrics"))
    prev_errors = _json_errors(prev.get("errors"))
    prev_quality = _quality(prev_metrics, prev_errors)
    improved = new_quality < prev_quality

    if improved:
        write_design(out, topo, best)
        result: Dict[str, Any] = {
                  "design": name, "score": new_score,
                  "evals": evals, "evals_seconds": round(elapsed, 1),
                  "metrics": final, "pass": report(final),
                  "errors": errors,
                  "polished_from": {
                      "analysis_errors": prev_quality[0],
                      "failed_gates": prev_quality[1],
                      "score": prev_quality[2],
                  }}
        (out / "result.json").write_text(
            json.dumps(result, indent=2, default=str))
    else:
        result = dict(prev, polish_rejected={
            "analysis_errors": new_quality[0],
            "failed_gates": new_quality[1],
            "score": new_quality[2],
        })
    result["improved"] = improved
    return result


if __name__ == "__main__":
    evals = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    res = polish(sys.argv[1], evals)
    m, p = res["metrics"], res["pass"]
    print(f"\n{res['design']}  score={res.get('score', -1):.3f}  "
          f"improved={res['improved']}")
    for k in ("dcgain", "gain_bandwidth_product", "phase_in_deg", "pm_true",
              "cmrrdc", "dcpsrp", "dcpsrn", "power", "vos25"):
        print(f"  {k:24s} {m.get(k)}")
    print("  pass:", {k: v for k, v in p.items()})
