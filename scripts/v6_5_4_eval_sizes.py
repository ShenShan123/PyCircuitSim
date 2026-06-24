#!/usr/bin/env python3
"""Evaluate the FRESH S/M/L/XL checkpoints (uniform clean recipe) against the 16
complex gates and select the BEST SIZE per tech (per-tech config, not unified).

Unlike the (rejected) stale-checkpoint bake-off, the only candidates here are the
4 freshly-trained capacity tiers per tech — `tsmc{X}_dn_{small,medium,large,xl}`.
Each (tech, size) is pinned via the env override (prefix form) and run through all
4 gates, CPU-pinned, in an isolated results dir. Per tech we pick the size with
the most gate wins (tie-break: passes opamp, passes ring, lower capacity, best
aggregate margin).
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "external_compact_models" / "bsimar" / "checkpoints"
NGSPICE = str(ROOT / "tools" / "ngspice-45.2" / "bin" / "ngspice")
PY = "/data1/shenshan/.conda/envs/pycircuitsim/bin/python"
TECHS = ["tsmc5", "tsmc7", "tsmc12", "tsmc16"]
SIZES = ["small", "medium", "large", "xl"]
GATES = ["opamp", "ring_osc", "switchcap", "sram_snm"]
GLABEL = {"opamp": "opamp", "ring_osc": "ring", "switchcap": "swcap", "sram_snm": "sram"}
TUC = {"tsmc5": "TSMC5", "tsmc7": "TSMC7", "tsmc12": "TSMC12", "tsmc16": "TSMC16"}
TOL = {"opamp": 10.0, "ring_osc": 5.0, "switchcap": 5.0, "sram_snm": 100.0}
METRIC_RE = {
    "opamp": re.compile(r"gain error = ([\d.]+)%"),
    "ring_osc": re.compile(r"period error = ([\d.]+)%"),
    "switchcap": re.compile(r"charge err=([\d.]+)% of VDD"),
    "sram_snm": re.compile(r"SNMerr=([\d.]+)%"),
}
PASS_RE = re.compile(r"(\d+)/1\b")
CAP = {"small": 0, "medium": 1, "large": 2, "xl": 3}


def run_task(tech, size, gate):
    cand = f"{tech}_dn_{size}"
    env = dict(os.environ)
    env["PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS"] = f"{cand}_nmos"
    env["PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS"] = f"{cand}_pmos"
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", NGSPICE_BIN=NGSPICE)
    wd = Path("/dev/shm/v6_5_4_evalsizes") / f"{tech}_{size}_{gate}"
    wd.mkdir(parents=True, exist_ok=True)
    env["PYCIRCUITSIM_COMPLEX_RESULTS"] = str(wd)
    try:
        p = subprocess.run([PY, f"tests/verify_complex_{gate}.py", "--tech", TUC[tech]],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
        out = p.stdout
    except subprocess.TimeoutExpired:
        return {"tech": tech, "size": size, "gate": gate, "passed": False, "metric": None}
    passed = False
    for m in PASS_RE.finditer(out):
        passed = (m.group(1) == "1")
    mm = METRIC_RE[gate].search(out)
    return {"tech": tech, "size": size, "gate": gate, "passed": passed,
            "metric": float(mm.group(1)) if mm else None}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    tasks = [(t, s, g) for t in TECHS for s in SIZES for g in GATES
             if (CK / f"{t}_dn_{s}_nmos_best.pt").exists()]
    print(f"[eval] {len(tasks)} tasks, workers={workers}", flush=True)
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_task, *t) for t in tasks]
        for f in as_completed(futs):
            r = f.result(); results.append(r); done += 1
            print(f"[{done}/{len(tasks)}] {r['tech']:6s} {r['size']:6s} "
                  f"{GLABEL[r['gate']]:5s} {'P' if r['passed'] else 'F'} m={r['metric']}", flush=True)

    agg = {}
    for r in results:
        a = agg.setdefault((r["tech"], r["size"]), {"passed": {}, "metric": {}})
        a["passed"][r["gate"]] = r["passed"]; a["metric"][r["gate"]] = r["metric"]

    out = ROOT / "results" / "v6_5_4_retrain"; out.mkdir(parents=True, exist_ok=True)
    def margin(a):
        return sum((a["metric"].get(g) or 0) / TOL[g] for g in GATES)
    def key(t, s):
        a = agg[(t, s)]
        npass = sum(1 for g in GATES if a["passed"].get(g))
        return (-npass, 0 if a["passed"].get("opamp") else 1,
                0 if a["passed"].get("ring_osc") else 1, CAP[s], margin(a))
    chosen, total, lines = {}, 0, ["# V6.5.4 fresh-retrain per-size evaluation", ""]
    for t in TECHS:
        sizes = [s for s in SIZES if (t, s) in agg]
        sizes.sort(key=lambda s: key(t, s))
        best = sizes[0]; chosen[t] = best
        a = agg[(t, best)]; npass = sum(1 for g in GATES if a["passed"].get(g)); total += npass
        lines += [f"## {t} — best size: **{best}** ({npass}/4)", "",
                  "| size | npass | opamp | ring | swcap | sram |",
                  "|---|---|---|---|---|---|"]
        for s in sizes:
            aa = agg[(t, s)]; np_ = sum(1 for g in GATES if aa["passed"].get(g))
            cell = lambda g: ("**P**" if aa["passed"].get(g) else "F") + (
                f" {aa['metric'][g]:.2f}" if aa["metric"].get(g) is not None else "")
            star = " ⭐" if s == best else ""
            lines.append(f"| {s}{star} | {np_}/4 | {cell('opamp')} | {cell('ring_osc')} "
                         f"| {cell('switchcap')} | {cell('sram_snm')} |")
        lines.append("")
    lines += [f"## Best-size-per-tech mix: **{total}/16**", "",
              "| tech | best size | gates |", "|---|---|---|"]
    for t in TECHS:
        a = agg[(t, chosen[t])]
        gv = " ".join(f"{GLABEL[g]}{'✓' if a['passed'].get(g) else '✗'}" for g in GATES)
        lines.append(f"| {t} | {chosen[t]} | {gv} |")
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "chosen_sizes.json").write_text(json.dumps(
        {t: {"size": chosen[t], "passed": agg[(t, chosen[t])]["passed"],
             "metric": agg[(t, chosen[t])]["metric"]} for t in TECHS}, indent=2))
    (out / "all_sizes.json").write_text(json.dumps(
        [{"tech": t, "size": s, **agg[(t, s)]} for t in TECHS for s in SIZES if (t, s) in agg], indent=2))
    print("\n".join(lines))
    print(f"\n[eval] best-size-per-tech aggregate {total}/16 -> {out/'REPORT.md'}")


if __name__ == "__main__":
    main()
