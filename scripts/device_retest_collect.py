#!/usr/bin/env python3
"""V6.6.2 re-test — summarize finalist device-level suites (DC / tran / device-AC)
from results/recipe_bench/device_iso/<recipe>/<tech>_<suite>.log into
results/recipe_bench/DEVICE_RETEST.md (per plan §5: device NRMSE/MRE/R²/MaxErr
must not regress beyond noise vs clean)."""
from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "recipe_bench" / "device_iso"
TECHS = ["tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16"]
NUM = r"([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?|nan|n/a)"


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def parse_sweep(text: str) -> dict:
    """SUMMARY TABLE rows: cfg | sweep | NRMSE% | MRE% | R2 | MaxErr | Status."""
    rows = []
    for m in re.finditer(
            rf"\|\s*(\w+)\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}(?:uA|mV)\s*\|\s*(PASS|FAIL|ERROR)", text):
        rows.append(dict(nrmse=_f(m.group(2)), mre=_f(m.group(3)), r2=_f(m.group(4)),
                         maxerr=_f(m.group(5)), status=m.group(6)))
    if not rows:
        return {}
    nr = [r["nrmse"] for r in rows if r["nrmse"] is not None]
    return dict(n=len(rows),
                n_pass=sum(1 for r in rows if r["status"] == "PASS"),
                nrmse_mean=round(mean(nr), 2), nrmse_max=round(max(nr), 2),
                mre_mean=round(mean(r["mre"] for r in rows if r["mre"] is not None), 2),
                r2_min=round(min(r["r2"] for r in rows if r["r2"] is not None), 4))


def parse_ac(text: str) -> dict:
    rows = []
    for m in re.finditer(
            rf"AC \| (\w+) \| {NUM} \| {NUM} \| {NUM} \| {NUM} \| (PASS|FAIL)", text):
        rows.append(dict(dev=m.group(1), gain0=_f(m.group(2)), f3db=_f(m.group(3)),
                         mag=_f(m.group(4)), status=m.group(6)))
    if not rows:
        return {}
    return dict(n=len(rows), n_pass=sum(1 for r in rows if r["status"] == "PASS"),
                gain0_mean=round(mean(r["gain0"] for r in rows if r["gain0"] is not None), 2),
                mag_mean=round(mean(r["mag"] for r in rows if r["mag"] is not None), 1),
                f3db=[r["f3db"] for r in rows])


def md(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    recipes = sorted(p.name for p in BASE.iterdir() if p.is_dir())
    recipes = [r for r in ["clean", "corro15", "crit10", "crit15", "crit30", "csob"]
               if r in recipes] + [r for r in recipes if r not in
               ("clean", "corro15", "crit10", "crit15", "crit30", "csob")]
    data = {}
    for r in recipes:
        data[r] = {}
        for t in TECHS:
            d = {}
            for suite, key in (("verify_nn_multi_tech_dc", "dc"),
                               ("verify_nn_multi_tech_tran", "tran")):
                p = BASE / r / f"{t}_{suite}.log"
                if p.exists():
                    d[key] = parse_sweep(p.read_text(errors="replace"))
            p = BASE / r / f"{t}_verify_nn_ac.log"
            if p.exists():
                d["ac"] = parse_ac(p.read_text(errors="replace"))
            data[r][t] = d

    L = ["# V6.6.2 re-test — finalist device-level accuracy", ""]
    for key, title in (("dc", "Device DC sweeps (Id–Vgs over L/NFIN/VT)"),
                       ("tran", "Inverter VTC + transient sweeps")):
        L += [f"## {title} — mean NRMSE % (max) · pass", ""]
        rows = []
        for t in TECHS:
            row = [t]
            for r in recipes:
                s = data[r][t].get(key) or {}
                row.append(f"{s.get('nrmse_mean', '—')} ({s.get('nrmse_max', '—')}) "
                           f"{s.get('n_pass', '—')}/{s.get('n', '—')}" if s else "—")
            rows.append(row)
        L += [md(["Tech"] + recipes, rows), ""]
        L += [f"### {title} — mean MRE % / min R²", ""]
        rows = []
        for t in TECHS:
            row = [t]
            for r in recipes:
                s = data[r][t].get(key) or {}
                row.append(f"{s.get('mre_mean', '—')} / {s.get('r2_min', '—')}" if s else "—")
            rows.append(row)
        L += [md(["Tech"] + recipes, rows), ""]

    L += ["## Device CS-amp AC — mean gain0 err dB · mean magNRMSE % · pass", ""]
    rows = []
    for t in TECHS:
        row = [t]
        for r in recipes:
            s = data[r][t].get("ac") or {}
            row.append(f"{s.get('gain0_mean', '—')}dB · {s.get('mag_mean', '—')}% · "
                       f"{s.get('n_pass', '—')}/{s.get('n', '—')}" if s else "—")
        rows.append(row)
    L += [md(["Tech"] + recipes, rows), ""]

    out = ROOT / "results" / "recipe_bench" / "DEVICE_RETEST.md"
    out.write_text("\n".join(L))
    print(f"Wrote {out}")
    for r in recipes:
        dcs = [data[r][t].get("dc", {}).get("nrmse_mean") for t in TECHS]
        dcs = [v for v in dcs if v is not None]
        acp = sum(data[r][t].get("ac", {}).get("n_pass", 0) for t in TECHS)
        acn = sum(data[r][t].get("ac", {}).get("n", 0) for t in TECHS)
        print(f"  {r:8s}: devDC meanNRMSE {round(mean(dcs), 2) if dcs else '—'}%   devAC {acp}/{acn}")


if __name__ == "__main__":
    main()
