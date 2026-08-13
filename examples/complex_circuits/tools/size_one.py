"""Size one amplifier and write the result. ``python size_one.py <design> [evals]``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acstab import run_deck_auto
from amp_spec import initial_design, report, score
from build_amp import FULL_DECKS, tran_control, write_design
from meas import SimError
from size_amp import Knobs, apply_knobs, make_evaluator, pattern_search
from skyparse import parse_netlist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pycmg_lib import ROOT                              # noqa: E402
SKY = ROOT.parent / "designs" / "amplifier"


def size(name: str, max_evals: int = 300, quiet: bool = False) -> dict:
    topo = parse_netlist(SKY / name / "netlist.spice")
    base = initial_design(topo.roles, topo.passive_vars)
    work = ROOT / "work" / "amplifier" / name
    out = ROOT / "amplifier" / name

    log = None if quiet else (lambda s: print(s, flush=True))
    evaluate = make_evaluator(topo, base, work)
    t0 = time.time()
    knobs, best, _ = pattern_search(evaluate, Knobs(), max_evals=max_evals,
                                    log=log)
    elapsed = time.time() - t0

    design = apply_knobs(base, knobs)
    write_design(out, topo, design)

    merged: dict = {}
    errors = []
    for deck, control in FULL_DECKS + [("tb_tran.cir", tran_control(design))]:
        try:
            merged.update(run_deck_auto(out / deck, control, out,
                                        deck.replace(".cir", ""), timeout=600))
        except SimError as exc:
            errors.append(f"{deck}: {exc}".split("\n")[0])

    result = {
        "design": name, "score": best, "evals_seconds": round(elapsed, 1),
        "knobs": knobs.__dict__, "metrics": merged,
        "pass": report(merged), "errors": errors,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    res = size(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    m, p = res["metrics"], res["pass"]
    print(f"\n{res['design']}  score={res['score']:.3f}  "
          f"({res['evals_seconds']}s)")
    for k in ("dcgain", "gain_bandwidth_product", "phase_in_deg", "pm_true",
              "cmrrdc", "dcpsrp", "dcpsrn", "power", "vos25",
              "sr_rise", "sr_fall"):
        print(f"  {k:24s} {m.get(k)}")
    print("  pass:", {k: v for k, v in p.items()})
    if res["errors"]:
        print("  errors:", res["errors"])
