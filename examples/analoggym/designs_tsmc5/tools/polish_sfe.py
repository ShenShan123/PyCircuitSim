"""Per-device refinement for the sensing front ends that landed partial.

The first pass groups devices that share a source geometry.  For most sensors
that is right, but the AnalogGym templates behind the three partials ship
*every* device at the same placeholder ``l=180n w=2u`` -- the RL environment
they come from sizes each device independently, and collapsing them into one
matched group leaves the whole sensor three knobs and forced symmetry.  A
stack of identical devices produces almost no |Delta|Vgs, which is exactly the
failure measured:

* front_end_25_6T: 4.3 mV of output, 27 uV/C -- no usable voltage develops.
* front_end_31_3T: good slope (5.2 mV/C) but the 25 C point sits at 23 mV,
  under the 50 mV floor.
* front_end_42_2: in range but 158 uV/C against the 300 uV/C floor.

This pass keys geometry per device and lets each device pick its own Vt
flavor (a Vt difference between stacked devices is a designable level/slope
term).  Search optimises against a margined bar so results land inside the
reported one; the shipped directory is replaced only when the official score
improves.
"""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom_port import GMos, parse_generic
from meas import SimError
from sfe import (DC_CONTROL, DC_CONTROL_FAST, SfeDesign, SfeTargets, emit,
                 group_key, ports_roles, run_sfe_dc, sfe_report, sfe_score)
from size_sfe import BAD, L_CHOICES, NFIN_CHOICES, ROOT, SKY, _unvec, _vec

DEFAULT_NAMES = ["front_end_25_6T_schematic", "front_end_31_3T_schematic",
                 "front_end_42_2_2015_REF_schematic"]
from pycmg_lib import VT_FLAVORS as _TECH_VTS
VT_FLAVORS = [v for v in ("svt", "lvt", "ulvt", "hvt") if v in _TECH_VTS]

# Margined bar for the search phase; reporting stays at SFE_TARGETS.
SEARCH_TARGETS = SfeTargets(lsb_min=4e-4, lsb_max=5e-3,
                            vout_lo=0.08, vout_hi=0.65, pp_max=0.55)


def dev_key(m: GMos) -> str:
    """Per-device geometry key: the device name."""
    return m.name.lstrip("xX").lower()


def load_warm(name: str, mos: List[GMos]) -> SfeDesign:
    """Shipped design re-keyed per device (each device starts at its group)."""
    d = json.loads((ROOT / "sensing_front_end" / name /
                    "design.json").read_text())
    geoms = {}
    vts = {}
    for m in mos:
        k = dev_key(m)
        # A design.json from a previous polish run is already keyed per device.
        g = d["geoms"].get(k) or d["geoms"][group_key(m)]
        geoms[k] = (float(g[0]), int(g[1]), int(g[2]))
        vts[k] = d.get("vts", {}).get(k, d["vt"])
    return SfeDesign(vdd=d["vdd"], vt=d["vt"], geoms=geoms, vts=vts)


def _ladder(base: SfeDesign, keys: List[str], ratio: float) -> SfeDesign:
    """Break the symmetry: place device multiplicities on a geometric ladder."""
    geoms = dict(base.geoms)
    for i, k in enumerate(keys):
        l_nm, nfin, m = geoms[k]
        geoms[k] = (l_nm, nfin,
                    max(1, min(4000, int(round(m * ratio ** i)))))
    return replace(base, geoms=geoms)


def _seeds(warm: SfeDesign, keys: List[str], n: int) -> List[SfeDesign]:
    out = [warm,
           _ladder(warm, keys, 4.0),
           _ladder(warm, keys, 0.25)]
    # Vt-alternating stacks: adjacent devices in different flavors.
    for pair in [pq for pq in (("svt", "lvt"), ("hvt", "svt"),
                               ("hvt", "ulvt"))
                 if pq[0] in VT_FLAVORS and pq[1] in VT_FLAVORS]:
        vts = {k: pair[i % 2] for i, k in enumerate(keys)}
        out.append(replace(warm, vts=vts))
    rng = random.Random(20260731)
    while len(out) < n:
        geoms = {}
        for k, (l_nm, nfin, m) in warm.geoms.items():
            geoms[k] = (float(rng.choice(L_CHOICES)),
                        int(rng.choice(NFIN_CHOICES)),
                        max(1, min(4000,
                                   int(m * rng.choice([0.25, 0.5, 1, 2, 4])))))
        vts = {k: rng.choice(VT_FLAVORS) for k in keys}
        out.append(replace(warm, geoms=geoms, vts=vts))
    return out[:n]


def polish(name: str, max_evals: int = 600, n_seeds: int = 14) -> dict:
    subs, _top = parse_generic(SKY / name / "netlist.spice")
    sub = next(s for s in subs if s.mos)
    mos = [m for s in subs for m in s.mos]
    keys = [dev_key(m) for m in mos]

    work = ROOT / "work" / "sfe_polish" / name
    out = ROOT / "sensing_front_end" / name
    work.mkdir(parents=True, exist_ok=True)
    warm = load_warm(name, mos)

    def evaluate(d: SfeDesign, control: str):
        try:
            emit(work, subs, d, sub.ports, sub.name, key_fn=dev_key)
            # Full-sweep runner: the score's smoothness terms need the
            # per-point dump (coarser grid in the search, same definitions).
            m, _warn = run_sfe_dc(work / "tb_dc.cir", control, work, "dc",
                                  timeout=180)
        except Exception:
            return BAD, {}
        return sfe_score(m, SEARCH_TARGETS), m

    t0 = time.time()
    best, best_score, best_m = None, float("inf"), {}
    for seed in _seeds(warm, keys, n_seeds):
        s, m = evaluate(seed, DC_CONTROL_FAST)
        if s < best_score:
            best, best_score, best_m = seed, s, m
        if best_score <= 0:
            break
    evals = n_seeds

    # Resolution ladder.  Monotonicity violations that only exist on the fine
    # reporting grid (0.5 C) are invisible on the 5 C search grid: a search
    # that stops at coarse score 0 exits without ever seeing the failure the
    # gate reads.  When the current rung's score reaches 0, re-anchor the
    # incumbent on the next finer grid and keep searching; the verdict below
    # is always the full 0.5 C sweep.
    rungs = [DC_CONTROL_FAST, "dc temp -20 120 1", DC_CONTROL]
    rung = 0

    step = [2.0, 2.0, 3.0] * len(keys)
    while evals < max_evals and max(step) > 1.05:
        if best_score <= 0:
            if rung + 1 >= len(rungs):
                break
            rung += 1
            best_score, best_m = evaluate(best, rungs[rung])
            evals += 1
            step = [2.0, 2.0, 3.0] * len(keys)
            continue
        improved = False
        vec = _vec(best, keys)
        for i in range(len(vec)):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                cv = list(vec)
                cv[i] *= step[i] ** direction
                cand = _unvec(cv, keys, best.vdd, best.vt)
                cand = replace(cand, vts=dict(best.vts))
                if _vec(cand, keys) == _vec(best, keys):
                    continue
                s, m = evaluate(cand, rungs[rung])
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    vec = _vec(best, keys)
                    break
        # Per-device Vt sweep.
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
                s, m = evaluate(cand, rungs[rung])
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    break
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]
    elapsed = time.time() - t0

    # Full 0.5 C sweep in the work directory, then keep the better design.
    emit(work, subs, best, sub.ports, sub.name, key_fn=dev_key)
    try:
        final, warns = run_sfe_dc(work / "tb_dc.cir", DC_CONTROL, work, "dc",
                                  timeout=600)
        err = "; ".join(warns)
    except SimError as exc:
        final, err = {}, str(exc).split("\n")[0]

    new_score = sfe_score(final) if final else BAD
    prev = json.loads((out / "result.json").read_text())
    prev_score = prev.get("score", float("inf"))
    improved = new_score < prev_score - 1e-12

    if improved:
        emit(out, subs, best, sub.ports, sub.name, key_fn=dev_key)
        result = {"design": name, "score": new_score, "evals": evals,
                  "evals_seconds": round(elapsed, 1), "metrics": final,
                  "pass": sfe_report(final), "error": err,
                  "polished_from": prev_score}
        (out / "result.json").write_text(
            json.dumps(result, indent=2, default=str))
    else:
        result = dict(prev, polish_rejected=new_score)
    result["improved"] = improved
    return result


def _one(args):
    name, evals = args
    try:
        return polish(name, evals)
    except Exception as exc:
        return {"design": name, "error": f"{type(exc).__name__}: {exc}",
                "pass": {}, "metrics": {}}


def main() -> None:
    evals = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
        else 600
    names = [a for a in sys.argv[1:] if not a.isdigit()] or DEFAULT_NAMES
    results = []
    with ProcessPoolExecutor(max_workers=len(names)) as pool:
        futs = {pool.submit(_one, (n, evals)): n for n in names}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            p = r.get("pass", {})
            print(f"{r['design']:38s} score={r.get('score', -1):8.3f} "
                  f"pass={sum(p.values())}/{len(p) or 1} "
                  f"{'IMPROVED' if r.get('improved') else 'kept previous'}",
                  flush=True)
    outp = ROOT / "results" / "sfe_polish.json"
    outp.write_text(json.dumps(sorted(results, key=lambda r: r["design"]),
                               indent=2, default=str))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
