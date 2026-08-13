"""Warm-start refinement for the two sub-threshold references.

Both landed partial for level/TC reasons that sit exactly on the knob the
first pass did not have: a *per-group* Vt flavor.  A sub-threshold reference
cancels temperature terms between device groups, and the size of the cancelled
residue -- and the DC level it lands on -- depends on the Vt difference
between those groups.  ``size_vref`` swept one global flavor; here each group
picks its own.

* dual_output: vref1 at 75 mV (floor is 80 mV) with TC 518 ppm/C (bar 500).
* three_output: vref1 at 792 mV on the 0.8 V rail (ceiling is 700 mV).

Search is the same pattern loop over (L, NFIN, m) per group, with a per-group
Vt sweep folded into each round, warm-started from the shipped design.  The
search optimises against a margined bar (TC 350 ppm/C, level 0.10..0.64 V) so
the result lands inside the reported one.  The shipped directory is replaced
only when the full-sweep score improves.
"""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom_port import parse_generic
from meas import SimError, run_deck
from size_vref import (BAD, DC_CONTROL, DC_CONTROL_FAST, L_CHOICES,
                       NFIN_CHOICES, PORTABLE, ROOT, SKY, VrefDesign, _unvec,
                       _vec, emit, outputs_of, supply_nets, vref_report,
                       vref_score)

from pycmg_lib import VT_FLAVORS as _TECH_VTS
VT_FLAVORS = [v for v in ("svt", "lvt", "ulvt", "hvt") if v in _TECH_VTS]

# Margined bar for the search phase (reporting stays at 500 ppm/C, 0.08-0.70).
TC_S = 350.0
LO_S, HI_S = 0.10, 0.64


def search_score(m: Dict, outs: List[str]) -> float:
    pen = 0.0
    for o in outs:
        tc, v25 = m.get(f"{o}_tc"), m.get(f"{o}_at25")
        if tc is None or v25 is None:
            pen += 20.0
            continue
        pen += max(0.0, (abs(tc) - TC_S) / 500.0)
        if v25 < LO_S:
            pen += (LO_S - v25) / 0.05
        if v25 > HI_S:
            pen += (v25 - HI_S) / 0.05
    return pen


def dev_key(m) -> str:
    return m.name.lstrip("xX").lower()


def load_design(name: str, mos=None) -> VrefDesign:
    """Shipped design; with *mos*, re-keyed per device (template groups weld
    unrelated devices together -- see emit's docstring).  PRESEED_JSON
    overrides the source file for hand-seeded starts."""
    import os
    preseed = os.environ.get("PRESEED_JSON")
    path = Path(preseed) if preseed \
        else ROOT / "voltage_reference" / name / "design.json"
    d = json.loads(path.read_text())
    if mos is None:
        return VrefDesign(
            vdd=d["vdd"], vt=d["vt"],
            geoms={k: (float(v[0]), int(v[1]), int(v[2]))
                   for k, v in d["geoms"].items()},
            vts=dict(d.get("vts", {})),
        )
    geoms, vts = {}, {}
    for m in mos:
        k = dev_key(m)
        g = d["geoms"].get(k) or d["geoms"][m.group]
        geoms[k] = (float(g[0]), int(g[1]), int(g[2]))
        vts[k] = d.get("vts", {}).get(k) or d.get("vts", {}).get(m.group) \
            or d["vt"]
    return VrefDesign(vdd=d["vdd"], vt=d["vt"], geoms=geoms, vts=vts)


def _rand_around(base: VrefDesign, rng: random.Random) -> VrefDesign:
    geoms = {}
    for k, (l_nm, nfin, m) in base.geoms.items():
        if rng.random() < 0.4:
            l_nm = float(rng.choice(L_CHOICES))
            nfin = int(rng.choice(NFIN_CHOICES))
            m = max(1, min(4000, int(m * rng.choice([0.25, 0.5, 2, 4]))))
        geoms[k] = (l_nm, nfin, m)
    vts = {k: (rng.choice(VT_FLAVORS) if rng.random() < 0.3
               else base.vts.get(k, base.vt))
           for k in base.geoms}
    return replace(base, geoms=geoms, vts=vts)


def polish(name: str, max_evals: int = 700, n_seeds: int = 8,
           per_device: bool = False) -> dict:
    subs, _top = parse_generic(SKY / name / "netlist.spice")
    mos = [m for s in subs for m in s.mos]
    outs = outputs_of(subs)
    vdd_net, gnd_net = supply_nets(subs)

    key_fn = dev_key if per_device else None
    warm = load_design(name, mos if per_device else None)
    keys = list(warm.geoms)

    work = ROOT / "work" / "vref_polish" / name
    out = ROOT / "voltage_reference" / name
    work.mkdir(parents=True, exist_ok=True)

    def evaluate(d: VrefDesign, control: str, timeout: float):
        try:
            emit(work, subs, d, name, vdd_net, gnd_net, outs, key_fn=key_fn)
            m = run_deck(work / "tb_dc.cir", control, work, "dc",
                         timeout=timeout)
        except Exception:
            return BAD, {}
        return search_score(m, outs), m

    rng = random.Random(20260731)
    seeds = [warm] + [_rand_around(warm, rng) for _ in range(n_seeds - 1)]
    t0 = time.time()
    best, best_score, best_m = None, float("inf"), {}
    for seed in seeds:
        s, m = evaluate(seed, DC_CONTROL_FAST, 30)
        if s < best_score:
            best, best_score, best_m = seed, s, m
        if best_score <= 0:
            break
    evals = len(seeds)

    step = [2.0, 1.6, 2.5] * len(keys)
    while evals < max_evals and best_score > 0 and max(step) > 1.05:
        improved = False
        vec = _vec(best, keys)
        for i in range(len(vec)):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                cv = list(vec)
                cv[i] *= step[i] ** direction
                cand = _unvec(cv, best, keys)
                cand = replace(cand, vts=dict(best.vts))
                if _vec(cand, keys) == _vec(best, keys):
                    continue
                s, m = evaluate(cand, DC_CONTROL_FAST, 30)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    vec = _vec(best, keys)
                    break
        # Per-group Vt sweep: try every alternative flavor for each group.
        for k in keys:
            if evals >= max_evals:
                break
            cur = best.vts.get(k, best.vt)
            for vt in VT_FLAVORS:
                if vt == cur or evals >= max_evals:
                    continue
                vts = dict(best.vts)
                vts[k] = vt
                cand = replace(best, vts=vts)
                s, m = evaluate(cand, DC_CONTROL_FAST, 30)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    break
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]
    elapsed = time.time() - t0

    # Full 0.5 C sweep on the candidate, in the work directory.
    emit(work, subs, best, name, vdd_net, gnd_net, outs, key_fn=key_fn)
    err = ""
    try:
        final = run_deck(work / "tb_dc.cir", DC_CONTROL, work, "dc",
                         timeout=900)
    except SimError as exc:
        final, err = {}, str(exc).splitlines()[0]

    new_score = vref_score(final, outs) if final else BAD
    prev = json.loads((out / "result.json").read_text())
    prev_score = prev.get("score", float("inf"))
    improved = new_score < prev_score - 1e-12

    if improved:
        emit(out, subs, best, name, vdd_net, gnd_net, outs, key_fn=key_fn)
        result = {"design": name, "outputs": outs, "evals": evals,
                  "evals_seconds": round(elapsed, 1), "score": new_score,
                  "metrics": final, "pass": vref_report(final, outs),
                  "error": err, "polished_from": prev_score}
        (out / "result.json").write_text(
            json.dumps(result, indent=2, default=str))
    else:
        result = dict(prev, polish_rejected=new_score)
    result["improved"] = improved
    return result


PER_DEVICE = False


def _one(args):
    name, evals = args
    try:
        return polish(name, evals, per_device=PER_DEVICE)
    except Exception as exc:
        return {"design": name, "error": f"{type(exc).__name__}: {exc}",
                "pass": {}, "metrics": {}}


def main() -> None:
    global PER_DEVICE
    args = list(sys.argv[1:])
    if "--per-device" in args:
        PER_DEVICE = True
        args.remove("--per-device")
    evals = int(args[0]) if args and args[0].isdigit() else 700
    names = [a for a in args if not a.isdigit()] or PORTABLE
    results = []
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        futs = {pool.submit(_one, (n, evals)): n for n in names}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            p = r.get("pass", {})
            print(f"{r['design']:34s} score={r.get('score', -1):8.3f} "
                  f"pass={sum(p.values())}/{len(p) or 1} "
                  f"{'IMPROVED' if r.get('improved') else 'kept previous'}",
                  flush=True)
    outp = ROOT / "results" / "vref_polish.json"
    outp.write_text(json.dumps(sorted(results, key=lambda r: r["design"]),
                               indent=2, default=str))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
