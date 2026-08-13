"""Size every AnalogGym sensing front end for TSMC16.

Knobs are per matched group: channel length, fin count and multiplicity.  A
sensor has two to four groups, so the search space stays small enough for the
same deterministic pattern search the amplifiers use.
"""

from __future__ import annotations

import json
import sys
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom_port import parse_generic
from meas import SimError
from pycmg_lib import snap_l, snap_nfin
from sfe import (DC_CONTROL, DC_CONTROL_FAST, SfeDesign, emit, group_key,
                 groups_of, initial_design, ports_roles, run_sfe_dc,
                 sfe_report, sfe_score)

ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT.parent / "designs" / "sensing_front_end"

# Sensors that are PTAT stacks; SMCNR_SE_2st_AMP is an amplifier and is handled
# by the amplifier bench instead.
SKIP = {"SMCNR_SE_2st_AMP"}

from pycmg_lib import L_CHOICES_LONG, NFIN_CHOICES as _NFIN, VT_FLAVORS

L_CHOICES = list(L_CHOICES_LONG)
NFIN_CHOICES = list(_NFIN)
BAD = 1e3


def _vec(design: SfeDesign, keys: List[str]) -> List[float]:
    out: List[float] = []
    for k in keys:
        l_nm, nfin, mult = design.geoms[k]
        out += [l_nm, float(nfin), float(mult)]
    return out


def _unvec(vec: List[float], keys: List[str], vdd: float, vt: str) -> SfeDesign:
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for i, k in enumerate(keys):
        l_nm = min(L_CHOICES, key=lambda c: abs(c - vec[3 * i]))
        nfin = min(NFIN_CHOICES, key=lambda c: abs(c - vec[3 * i + 1]))
        mult = max(1, min(4000, int(round(vec[3 * i + 2]))))
        geoms[k] = (float(l_nm), int(nfin), mult)
    return SfeDesign(vdd=vdd, geoms=geoms, vt=vt)


def _seeded(base: SfeDesign, seed: int) -> SfeDesign:
    """Perturb a starting point deterministically.

    Pattern search is a local method: from one start it finds one basin.  The
    sensors that stall do so at a genuine local minimum, not for want of
    iterations, so extra evals from the same seed buy nothing -- a different
    starting geometry does.
    """
    if seed == 0:
        return base
    rng = random.Random(seed)
    geoms = {}
    for k, (l_nm, nfin, mult) in base.geoms.items():
        geoms[k] = (float(rng.choice(L_CHOICES)),
                    int(rng.choice(NFIN_CHOICES)),
                    max(1, min(4000, int(mult * rng.choice([0.25, 0.5, 1, 2, 4, 8])))))
    return SfeDesign(vdd=base.vdd, geoms=geoms,
                     vt=rng.choice([v for v in ("svt", "lvt", "ulvt", "hvt")
                                    if v in VT_FLAVORS]))


def size(name: str, max_evals: int = 250, quiet: bool = True,
         restarts: int = 1) -> dict:
    subs, _top = parse_generic(SKY / name / "netlist.spice")
    sub = next(s for s in subs if s.mos)
    mos = [m for s in subs for m in s.mos]
    keys = groups_of(mos)
    gnd, vdd_p, out_p = ports_roles(sub.ports)

    work = ROOT / "work" / "sensing_front_end" / name
    out = ROOT / "sensing_front_end" / name
    base = initial_design(mos)

    def evaluate(design: SfeDesign, where: Path, control: str):
        try:
            emit(where, subs, design, sub.ports, sub.name)
            # Full-sweep runner: the score's smoothness terms need the
            # per-point dump, so the search sees the same gates the report
            # applies (coarser grid, same definitions).
            m, _warn = run_sfe_dc(where / "tb_dc.cir", control, where, "dc",
                                  timeout=180)
        except (SimError, Exception):
            return BAD, {}
        return sfe_score(m), m

    t0 = time.time()
    best, best_score, best_meas = None, float("inf"), {}
    for seed in range(restarts):
        start = _seeded(base, seed)
        s0, m0 = evaluate(start, work, DC_CONTROL_FAST)
        if s0 < best_score:
            best, best_score, best_meas = start, s0, m0
        if best_score <= 0:
            break
    evals = restarts

    # Pattern search over (L, NFIN, m) per group, plus the supply.
    step = [2.0, 2.0, 3.0] * len(keys)
    vt_options = [v for v in ("svt", "lvt", "ulvt", "hvt") if v in VT_FLAVORS]
    while evals < max_evals and best_score > 0 and max(step) > 1.05:
        improved = False
        vec = _vec(best, keys)
        for i in range(len(vec)):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                cand_vec = list(vec)
                cand_vec[i] *= step[i] ** direction
                cand = _unvec(cand_vec, keys, best.vdd, best.vt)
                if _vec(cand, keys) == _vec(best, keys):
                    continue
                s, m = evaluate(cand, work, DC_CONTROL_FAST)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_meas = cand, s, m
                    improved = True
                    vec = _vec(best, keys)
                    break
        # Threshold flavour changes the sub-threshold slope wholesale, so it is
        # tried as its own move rather than as a continuous knob.
        for vt in vt_options:
            if evals >= max_evals or vt == best.vt:
                continue
            cand = SfeDesign(vdd=best.vdd, geoms=dict(best.geoms), vt=vt)
            s, m = evaluate(cand, work, DC_CONTROL_FAST)
            evals += 1
            if s < best_score - 1e-9:
                best, best_score, best_meas = cand, s, m
                improved = True
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]

    elapsed = time.time() - t0
    emit(out, subs, best, sub.ports, sub.name)
    try:
        final, warns = run_sfe_dc(out / "tb_dc.cir", DC_CONTROL, out, "dc",
                                  timeout=600)
        err = "; ".join(warns)
    except SimError as exc:
        final, err = {}, str(exc).split("\n")[0]

    result = {"design": name, "score": sfe_score(final) if final else BAD,
              "evals_seconds": round(elapsed, 1), "evals": evals,
              "metrics": final, "pass": sfe_report(final), "error": err}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _one(args):
    name, evals = args
    try:
        return size(name, evals, restarts=12)
    except Exception as exc:
        return {"design": name, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    evals = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    names = sorted(d.name for d in SKY.iterdir()
                   if d.is_dir() and d.name not in SKIP)
    results = []
    with ProcessPoolExecutor(max_workers=min(len(names), 16)) as pool:
        futs = {pool.submit(_one, (n, evals)): n for n in names}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("error") and not r.get("metrics"):
                print(f"[{len(results):2d}/{len(names)}] {r['design']:36s} "
                      f"ERROR {r['error'][:70]}", flush=True)
            else:
                p = r["pass"]
                print(f"[{len(results):2d}/{len(names)}] {r['design']:36s} "
                      f"score={r['score']:7.3f} pass={sum(p.values())}/{len(p)} "
                      f"lsb={r['metrics'].get('lsb_25_75c')}", flush=True)
    out = ROOT / "results" / "sfe_sizing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(results, key=lambda r: r["design"]),
                              indent=2, default=str))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
