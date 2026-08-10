"""Size every AnalogGym amplifier for TSMC16, one process per design."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from size_one import ROOT, SKY, size


def _one(args):
    name, evals = args
    try:
        return size(name, evals, quiet=True)
    except Exception as exc:                      # keep one failure local
        return {"design": name, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    evals = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    names = sorted(d.name for d in SKY.iterdir() if d.is_dir())
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=min(len(names), 20)) as pool:
        futs = {pool.submit(_one, (n, evals)): n for n in names}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if "error" in res:
                print(f"[{len(results):2d}/{len(names)}] {res['design']:24s} "
                      f"ERROR {res['error']}", flush=True)
            else:
                p = res["pass"]
                print(f"[{len(results):2d}/{len(names)}] {res['design']:24s} "
                      f"score={res['score']:7.3f}  "
                      f"pass={sum(p.values())}/{len(p)}  "
                      f"({res['evals_seconds']}s)", flush=True)

    out = ROOT / "results" / "amplifier_sizing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(results, key=lambda r: r["design"]),
                              indent=2, default=str))
    print(f"\n{len(results)} designs in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
