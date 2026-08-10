"""Warm-start refinement for the LDOs that did not fully pass.

``size_ldo.py`` searches from scratch; this tool restarts from the shipped
``design.json`` of a design that landed partial, with two things the first
pass did not have:

* **More knobs.**  Compensation resistors (any resistor that is not a leg of
  the output divider) and the bias current were fixed at 100 k / 5 uA in the
  first pass.  Both are first-order levers on phase margin, which is exactly
  what three of the four partials fail on.
* **Structured seeds.**  The warm design plus targeted perturbations of it.
  Basic_LDO's failure mode, for instance, is drive: its pass device sources
  ~12 mA at the gate floor the level shifter allows, against a 55 mA load
  (measured -- see work/iv).  Dedicated x4 and x8 pass-multiplier seeds start
  the search in basins where the output can actually regulate.

The result only replaces the shipped design when its full-bench acceptance key
(analysis errors, failed gates, full penalty, core penalty) improves; otherwise
the directory is left untouched.
"""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom_port import GSubckt, parse_generic
from ldo import (DIV_RATIO, LdoDesign, VOUT_NOM, _passive_defaults,
                 emit, initial_design, is_pass_device)
from acstab import run_deck_auto
from meas import SimError
from ldo import VDD
from pycmg_lib import VT_FLAVORS
from size_ldo import (BAD, CORE_GATES, DECK_ORDER, FAST_ORDER, IBIAS, L_CHOICES,
                      M_MAX, NON_PASS_M_MAX, NFIN_CHOICES, ROOT, SKY, _FB,
                      _GND, _OUT, _finite_metric, _unvec, _vec, decks,
                      has_resistive_divider, ldo_core_score, ldo_report,
                      ldo_score, wiring)

DEFAULT_NAMES = ["Basic_LDO", "ldo_1", "ldo_2", "ldo_folded_cascode",
                 "ldo_simple"]
SLOW_ORDER = [deck for deck in DECK_ORDER if deck not in FAST_ORDER]
CandidateKey = Tuple[int, int, float, float]
Metrics = Dict[str, float]


def _soft(short: float, none_pen: float) -> float:
    """Shortfall penalty that always stays below the missing-value penalty.

    Linear near the bar, then compressed: a loop that measures at -157 dB must
    score worse than one at -5 dB but *better* than one that does not converge
    at all -- otherwise the search prefers the dead basin (observed on TSMC5
    Basic_LDO: svt never converges and scored 25.0; the converging ulvt seed
    scored 27.8 and was discarded).
    """
    short = max(0.0, short)
    knee = 2.0
    if short <= knee:
        return short
    return knee + (none_pen - 0.1 - knee) * (1.0 - knee / short)


def _search_minimum(m: Metrics, key: str, limit: float, scale: float,
                    missing: float = 3.0) -> float:
    value = _finite_metric(m, key)
    return missing if value is None else _soft((limit - value) / scale, missing)


def _search_maximum(m: Metrics, key: str, limit: float, scale: float,
                    missing: float = 3.0) -> float:
    value = _finite_metric(m, key)
    return missing if value is None else _soft((value - limit) / scale, missing)


def _search_range(m: Metrics, key: str, low: float, high: float,
                  scale: float, missing: float = 3.0) -> float:
    value = _finite_metric(m, key)
    if value is None:
        return missing
    short = (low - value) / scale if value < low else (value - high) / scale
    return _soft(short, missing)


def search_core_score(m: Metrics) -> float:
    """Core-gate penalty with margin beyond the official report limits."""
    pen = 0.0
    for tag in ("max", "min"):
        pen += _search_minimum(m, f"dcgain_{tag}", 42.0, 10.0)
        # Same key preference as ldo_core_score: the true (unwrapped) margin
        # when the stability runner produced it, raw pm_{tag} otherwise.
        pm_key = f"pm_true_{tag}" if f"pm_true_{tag}" in m else f"pm_{tag}"
        pen += _search_minimum(m, pm_key, 50.0, 15.0)
        pen += _search_minimum(m, f"gbw_{tag}", 1.2e5, 6e4, missing=2.0)
        vout = _finite_metric(m, f"vout_{tag}")
        if vout is None:
            pen += 3.0
        else:
            pen += _soft((abs(vout - VOUT_NOM) - 0.04) / 0.05, 3.0)
    pen += _search_maximum(m, "power_max", 70e-3, 20e-3)
    return pen


def search_score(m: Metrics) -> float:
    """ldo_score with margin: optimise past the bar, report against the bar.

    The first pass optimised against the report thresholds themselves and
    delivered ldo_folded_cascode at 39.90 dB against a 40 dB bar.  Requiring
    42 dB / 50 deg / 120 kHz / 40 mV in the search leaves room to land.
    A missing measurement is always the worst outcome (see _soft).
    """
    pen = search_core_score(m)
    for key in ("lnrmax", "lnrmin"):
        pen += _search_range(m, key, 0.0, 0.20, 0.25)
    pen += _search_range(m, "lr", 0.0, 0.80, 1.0)
    for key in ("psrr_max", "psrr_min"):
        pen += _search_maximum(m, key, -25.0, 20.0)
    pen += _search_range(m, "undershoot", 0.0, 0.20 * VDD, 0.25 * VDD)
    pen += _search_range(m, "overshoot", 0.0, 0.12 * VDD, 0.15 * VDD)
    return pen


def _failed_gate_count(m: Metrics) -> int:
    return sum(1 for passed in ldo_report(m).values() if not passed)


def _search_key(m: Metrics, errors: List[str]) -> CandidateKey:
    return (len(errors), _failed_gate_count(m), search_score(m),
            search_core_score(m))


def _acceptance_key(m: Metrics, errors: List[str]) -> CandidateKey:
    return (len(errors), _failed_gate_count(m), ldo_score(m),
            ldo_core_score(m))


def load_design(name: str) -> LdoDesign:
    """Shipped design vector; PRESEED_JSON overrides it (hand-seeded starts)."""
    import os
    preseed = os.environ.get("PRESEED_JSON")
    path = Path(preseed) if preseed else ROOT / "ldo" / name / "design.json"
    d = json.loads(path.read_text())
    return LdoDesign(
        vdd=d["vdd"], vref=d["vref"], vt=d["vt"],
        geoms={k: (float(v[0]), int(v[1]), int(v[2]))
               for k, v in d["geoms"].items()},
        passives=dict(d["passives"]),
        vt_overrides=dict(d.get("vt_overrides", {})),
    )


def _divider_legs(subs: List[GSubckt], design: LdoDesign) -> List[str]:
    """Resistors that form the output divider (they touch the vfb net)."""
    legs = []
    for s in subs:
        for p in s.passives:
            if p.name.lower().startswith("r") and p.name in design.passives \
                    and "vfb" in {n.lower() for n in p.nodes}:
                legs.append(p.name)
    return legs


def _comp_resistors(subs: List[GSubckt], design: LdoDesign) -> List[str]:
    """Resistor-valued passives that are not the feedback divider.

    The divider legs set the output level together with vref and are left
    alone; everything else resistor-shaped is compensation and becomes a knob.
    """
    out: List[str] = []
    seen = set()
    for s in subs:
        for p in s.passives:
            if p.name.lower().startswith("r") and p.name in design.passives:
                nodes = {n.lower() for n in p.nodes}
                if "vfb" not in nodes:
                    out.append(p.name)
                seen.add(p.name)
        for iname, nodes, line in s.insts:
            low = line.lower()
            if ("res_high_po" in low or "res_generic" in low) \
                    and iname in design.passives and iname not in seen:
                if "vfb" not in {n.lower() for n in nodes}:
                    out.append(iname)
    return out


def _seeds(base: LdoDesign, pass_keys: Set[str], n: int) -> List[LdoDesign]:
    """Warm design first, then structured then random perturbations of it."""
    def scale_m(d: LdoDesign, keys, k) -> LdoDesign:
        geoms = dict(d.geoms)
        for g in keys:
            l_nm, nfin, m = geoms[g]
            mult_max = M_MAX if g in pass_keys else NON_PASS_M_MAX
            geoms[g] = (l_nm, nfin,
                        max(1, min(mult_max, int(round(m * k)))))
        return replace(d, geoms=geoms)

    def scale_p(d: LdoDesign, pred, k) -> LdoDesign:
        p = dict(d.passives)
        for key in p:
            if pred(key):
                p[key] = p[key] * k
        return replace(d, passives=p)

    is_cap = lambda k: k.lower().startswith(("c", "xc"))
    is_ib = lambda k: k == "ibias"

    def set_p(d: LdoDesign, key, val) -> LdoDesign:
        if key not in d.passives:
            return d
        p = dict(d.passives)
        p[key] = val
        return replace(d, passives=p)

    out = [base,
           scale_m(base, pass_keys, 4.0),
           scale_m(base, pass_keys, 8.0),
           scale_p(scale_m(base, pass_keys, 4.0), is_ib, 4.0),
           scale_p(base, is_cap, 4.0),
           scale_p(base, is_cap, 0.25),
           scale_p(scale_p(base, is_cap, 2.0), is_ib, 8.0),
           # Low-impedance divider: output pole dominant at both loads.
           set_p(base, "div_scale", 0.01),
           set_p(scale_p(base, is_ib, 4.0), "div_scale", 0.03)]
    # Vt flavor is not on any search axis, and it is the headroom lever on the
    # low-rail nodes: a 0.65 V error amp that cannot bias with svt devices
    # converges with ulvt.  Seed every flavor the tech ships.
    for f in VT_FLAVORS:
        if f != base.vt:
            out.insert(1, replace(base, vt=f))
    # set_p returns the base unchanged when there is no divider; drop dups.
    seen, dedup = set(), []
    for d in out:
        key = json.dumps({"vt": d.vt, "vo": sorted(d.vt_overrides.items()),
                          "g": {k: list(v) for k, v in sorted(d.geoms.items())},
                          "p": sorted(d.passives.items())}, default=str)
        if key not in seen:
            seen.add(key)
            dedup.append(d)
    out = dedup

    rng = random.Random(20260731)
    while len(out) < n:
        geoms = {}
        for g, (l_nm, nfin, m) in base.geoms.items():
            if rng.random() < 0.4:
                l_nm = float(rng.choice(L_CHOICES))
                nfin = int(rng.choice(NFIN_CHOICES))
                scales = [0.25, 0.5, 2, 4, 8] if g in pass_keys \
                    else [0.25, 0.5, 2, 4]
                mult_max = M_MAX if g in pass_keys else NON_PASS_M_MAX
                m = max(1, min(mult_max, int(m * rng.choice(scales))))
            geoms[g] = (l_nm, nfin, m)
        passives = dict(base.passives)
        for k in passives:
            if is_cap(k) or is_ib(k):
                passives[k] = passives[k] * rng.choice([0.25, 0.5, 1, 2, 4])
        out.append(replace(base, geoms=geoms, passives=passives))
    return out[:n]


def polish(name: str, max_evals: int = 400, n_seeds: int = 12,
           fine: bool = False) -> dict:
    """*fine* searches only from the shipped point with small steps -- for a
    design within a fraction of a dB of its bar, where coarse moves fall off
    the ridge and reseeding is counterproductive."""
    subs, top = parse_generic(SKY / name / "netlist.spice")
    sub = next(s for s in subs if s.mos)
    out_nodes = [p.lower() for p in sub.ports if _OUT.match(p)]
    has_div = has_resistive_divider(subs, out_nodes)
    _n, info = wiring(sub.ports, "vdd", "vo", has_divider=has_div)

    base = load_design(name)
    for net, bname in info["vbias"]:
        base.passives.setdefault(f"bias_{bname}", 0.5 * VDD)
    if info["ibias"]:
        base.passives.setdefault("ibias", IBIAS)

    keys = list(base.geoms)
    pkeys = [k for k in base.passives
             if k.startswith("bias_") or k in ("vref", "ibias")
             or k.lower().startswith(("c", "xc"))]
    pkeys += _comp_resistors(subs, base)

    # The divider's ratio sets the output level together with vref and stays
    # put; its *impedance* is a free stability knob.  Loading the output with
    # a low-ohm divider flattens Rout across the load range, which is what
    # makes the 100 nF output pole dominant at both extremes -- the lever
    # ldo_1's uncompensated three-stage loop needs.
    div_legs = _divider_legs(subs, base)
    div_base = {leg: base.passives[leg] for leg in div_legs}
    if div_legs:
        base.passives.setdefault("div_scale", 1.0)
        pkeys.append("div_scale")

    def materialize(d: LdoDesign) -> LdoDesign:
        """Apply div_scale onto the divider legs; written designs carry 1.0."""
        if not div_legs:
            return d
        p = dict(d.passives)
        k = max(1e-4, p.pop("div_scale", 1.0))
        for leg in div_legs:
            p[leg] = div_base[leg] * k
        p["div_scale"] = 1.0
        return replace(d, passives=p)

    work = ROOT / "work" / "ldo_polish" / name
    out = ROOT / "ldo" / name
    work.mkdir(parents=True, exist_ok=True)

    def evaluate(d: LdoDesign) -> Tuple[CandidateKey, Metrics]:
        try:
            dm = materialize(d)
            emit(work, subs, top, dm, sub.name, sub.ports, out_nodes)
            built = decks(sub, dm, has_div)
        except Exception as exc:
            errors = [f"setup: {type(exc).__name__}: {exc}"]
            return (len(errors), len(ldo_report({})), BAD, BAD), {}

        metrics: Metrics = {}
        errors: List[str] = []

        def run_stage(order: List[str], timeout: float) -> None:
            for deck in order:
                text, control = built[deck]
                (work / deck).write_text(text)
                try:
                    metrics.update(run_deck_auto(
                        work / deck, control, work, deck.replace(".cir", ""),
                        timeout=timeout))
                except SimError as exc:
                    errors.append(f"{deck}: {str(exc).splitlines()[0]}")
                except Exception as exc:
                    errors.append(f"{deck}: {type(exc).__name__}: {exc}")

        run_stage(FAST_ORDER, 25)
        if not errors and ldo_core_score(metrics) == 0.0:
            # Stage two adds only the line/PSRR/transient decks; never pay to
            # repeat the core analyses for a candidate already known viable.
            run_stage(SLOW_ORDER, 60)
        else:
            # A staged candidate has not completed the audit.  Count every
            # skipped slow deck in its search key so a partially evaluated
            # core point cannot outrank a fully evaluated candidate merely
            # because it reports zero simulator errors on the decks it ran.
            errors.extend(f"{deck}: skipped until core gates pass"
                          for deck in SLOW_ORDER)
        return _search_key(metrics, errors), metrics

    pass_keys = {m.group for s in subs for m in s.mos
                 if is_pass_device(m, out_nodes)}

    t0 = time.time()
    best: Optional[LdoDesign] = None
    best_key: CandidateKey = (sys.maxsize, sys.maxsize, float("inf"),
                              float("inf"))
    seeds = [base] if fine else _seeds(base, pass_keys, n_seeds)
    if not fine and sub.name.lower() == "ldo_folded_cascode":
        source_seed = initial_design(subs, out_nodes)
        source_seed.passives = _passive_defaults(subs)
        source_seed.vref = VOUT_NOM
        source_seed.passives.update({
            "bias_vb1": 0.5 * VDD,
            "bias_vb2": 0.0125 * VDD,
            "vref": VOUT_NOM,
        })
        seeds = ([source_seed] + seeds)[:max(1, n_seeds)]
    evals = 0
    for seed in seeds:
        key, _metrics = evaluate(seed)
        evals += 1
        if key < best_key:
            best, best_key = seed, key
        if best_key[0] == 0 and best_key[1] == 0 and best_key[2] == 0.0:
            break
    if best is None:
        raise RuntimeError("LDO polish produced no candidate")

    step = ([1.35, 1.3, 1.5] * len(keys) + [1.25] * len(pkeys)) if fine \
        else [2.0, 1.6, 3.0] * len(keys) + [1.5] * len(pkeys)
    while evals < max_evals and best_key[2] > 0 and max(step) > 1.05:
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
                key, _metrics = evaluate(cand)
                evals += 1
                if key < best_key:
                    best, best_key = cand, key
                    improved = True
                    vec = _vec(best, keys, pkeys)
                    break
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]
    elapsed = time.time() - t0

    # Full bench on the candidate, in the work directory.
    best_mat = materialize(best)
    emit(work, subs, top, best_mat, sub.name, sub.ports, out_nodes)
    final, errors = {}, []
    built = decks(sub, best_mat, has_div)
    for deck in DECK_ORDER:
        text, control = built[deck]
        (work / deck).write_text(text)
        try:
            final.update(run_deck_auto(work / deck, control, work,
                                       deck.replace(".cir", ""), timeout=600))
        except SimError as exc:
            errors.append(f"{deck}: {str(exc).splitlines()[0]}")

    new_score = ldo_score(final)
    new_key = _acceptance_key(final, errors)
    prev = json.loads((out / "result.json").read_text())
    prev_score = prev.get("score", float("inf"))
    prev_metrics = cast(Metrics, prev.get("metrics", {}))
    raw_prev_errors = prev.get("errors", [])
    prev_errors = cast(List[str], raw_prev_errors) \
        if isinstance(raw_prev_errors, list) else [str(raw_prev_errors)]
    prev_key = _acceptance_key(prev_metrics, prev_errors)
    improved = new_key < prev_key

    if improved:
        emit(out, subs, top, best_mat, sub.name, sub.ports, out_nodes)
        for deck in DECK_ORDER:
            text, control = built[deck]
            (out / deck).write_text(text)
        result = {"design": name, "score": new_score, "evals": evals,
                  "evals_seconds": round(elapsed, 1), "metrics": final,
                  "pass": ldo_report(final), "errors": errors,
                  "acceptance_key": list(new_key),
                  "polished_from": prev_score,
                  "polished_from_key": list(prev_key)}
        (out / "result.json").write_text(
            json.dumps(result, indent=2, default=str))
    else:
        result = dict(prev, polish_rejected=new_score,
                      polish_rejected_key=list(new_key))
    result["improved"] = improved
    return result


def _one(args):
    name, evals = args
    try:
        return polish(name, evals, fine=FINE)
    except Exception as exc:
        return {"design": name, "error": f"{type(exc).__name__}: {exc}",
                "pass": {}, "metrics": {}}


FINE = False


def main() -> None:
    global FINE
    args = list(sys.argv[1:])
    if "--fine" in args:
        FINE = True
        args.remove("--fine")
    evals = int(args[0]) if args and args[0].isdigit() else 400
    names = [a for a in args if not a.isdigit()] or DEFAULT_NAMES
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
                  f"pass={sum(p.values())}/{len(p) or 1} "
                  f"{'IMPROVED' if r.get('improved') else 'kept previous'}",
                  flush=True)
    outp = ROOT / "results" / "ldo_polish.json"
    outp.write_text(json.dumps(sorted(results, key=lambda r: r["design"]),
                               indent=2, default=str))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
