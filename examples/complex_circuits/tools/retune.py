"""Re-tune every design the carried-over sizes left unhealthy in this tree.

Reads each design's ``result.json`` (written by ``finalize.py`` after
``port_tech.py``), and for every design with a failing gate runs the
category's warm-started refinement loop -- the same searches the TSMC16 tree
used -- in parallel, one process per design.  A refined design replaces the
ported one only when its full-bench score improves, so a re-tune can never
make a healthy design worse.

    python3 tools/retune.py [category ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pycmg_lib import ROOT                              # noqa: E402

# Search-depth multiplier.  The polish searches are deterministic from a fixed
# seed, so a second pass at the same budget replays the first; post_pipeline.sh
# sets AG_EVALS_SCALE=2 so pass 2 extends the trajectory instead.
SCALE = float(os.environ.get("AG_EVALS_SCALE", "1"))

CATEGORIES = ["amplifier", "ldo", "sensing_front_end", "voltage_reference",
              "charge_pump"]


def failing(category: str) -> List[str]:
    out = []
    cdir = ROOT / category
    if not cdir.is_dir():
        return out
    for d in sorted(cdir.iterdir()):
        rj = d / "result.json"
        if not (d / "netlist.spice").exists():
            continue
        if not rj.exists():
            out.append(d.name)
            continue
        try:
            r = json.loads(rj.read_text())
        except json.JSONDecodeError:
            out.append(d.name)
            continue
        verdict = r.get("pass") or {}
        if not verdict or not all(verdict.values()):
            out.append(d.name)
    return out


def _tune(args: Tuple[str, str]) -> Dict:
    category, name = args
    t0 = time.time()
    try:
        if category == "amplifier":
            from polish_amp import polish
            r = polish(name, int(500 * SCALE))
        elif category == "ldo":
            from polish_ldo import polish
            r = polish(name, int(400 * SCALE))
        elif category == "sensing_front_end":
            import sfe_amp
            if name == sfe_amp.NAME:
                # The category's amplifier has a fixed design point; its
                # re-tune is a rebuild from design.json plus a fresh verdict.
                r = sfe_amp.rebuild()
            else:
                from polish_sfe import polish
                r = polish(name, int(600 * SCALE))
        elif category == "voltage_reference":
            from polish_vref import polish
            d = json.loads((ROOT / "voltage_reference" / name /
                            "design.json").read_text())
            # A design.json from the per-device TSMC16 polish is keyed by
            # device name; matched-group keys contain '|'.
            per_dev = not any("|" in k for k in d["geoms"])
            r = polish(name, int(700 * SCALE), per_device=per_dev)
        elif category == "charge_pump":
            from size_cp import size
            r = size(int(200 * SCALE))
        else:
            raise ValueError(category)
        p = r.get("pass") or {}
        return {"category": category, "design": name,
                "pass": f"{sum(p.values())}/{len(p) or 1}",
                "score": r.get("score"),
                "seconds": round(time.time() - t0, 1)}
    except Exception as exc:
        return {"category": category, "design": name,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1)}


def main() -> None:
    wanted = sys.argv[1:] or CATEGORIES
    jobs = [(cat, name) for cat in wanted for name in failing(cat)]
    if not jobs:
        print("everything already healthy")
        return
    print(f"re-tuning {len(jobs)}: "
          + ", ".join(f"{c}/{n}" for c, n in jobs), flush=True)

    done = []
    with ProcessPoolExecutor(max_workers=min(len(jobs), 20)) as pool:
        futs = {pool.submit(_tune, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            done.append(r)
            msg = r.get("error") or f"pass={r['pass']} score={r['score']}"
            print(f"[{len(done):2d}/{len(jobs)}] {r['category']:18s} "
                  f"{r['design']:36s} {msg} ({r['seconds']}s)", flush=True)

    outp = ROOT / "results" / "retune.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(done, indent=2, default=str))
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
