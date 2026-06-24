#!/usr/bin/env python3
"""Final V6.5.4 selection: evaluate the Stage-B seed variants, merge with the
base-size results (results/v6_5_4_retrain/all_sizes.json), and pick the overall
best config per tech (size, or large+seed). Writes the final REPORT + the
installable mix.
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "external_compact_models" / "bsimar" / "checkpoints"
NGSPICE = str(ROOT / "tools" / "ngspice-45.2" / "bin" / "ngspice")
PY = "/data1/shenshan/.conda/envs/pycircuitsim/bin/python"
OUT = ROOT / "results" / "v6_5_4_retrain"
TECHS = ["tsmc5", "tsmc7", "tsmc12", "tsmc16"]
SEED_TECHS = ["tsmc5", "tsmc7", "tsmc16"]
SEEDS = [7, 17, 31]
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
# capacity rank for tie-break (prefer standard tiers / lower capacity)
def cap_rank(cand: str) -> int:
    for i, tag in enumerate(["_dn_medium", "_dn_large", "_dn_small", "_dn_xl", "_dn_lgs"]):
        if tag in cand:
            return i
    return 9


def run_task(tech, cand, gate):
    env = dict(os.environ)
    env["PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS"] = f"{cand}_nmos"
    env["PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS"] = f"{cand}_pmos"
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", NGSPICE_BIN=NGSPICE)
    wd = Path("/dev/shm/v6_5_4_evalseeds") / f"{cand}_{gate}"
    wd.mkdir(parents=True, exist_ok=True)
    env["PYCIRCUITSIM_COMPLEX_RESULTS"] = str(wd)
    try:
        p = subprocess.run([PY, f"tests/verify_complex_{gate}.py", "--tech", TUC[tech]],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
        out = p.stdout
    except subprocess.TimeoutExpired:
        return {"tech": tech, "cand": cand, "gate": gate, "passed": False, "metric": None}
    passed = False
    for m in PASS_RE.finditer(out):
        passed = (m.group(1) == "1")
    mm = METRIC_RE[gate].search(out)
    return {"tech": tech, "cand": cand, "gate": gate, "passed": passed,
            "metric": float(mm.group(1)) if mm else None}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    # seed-variant candidates
    seed_cands = []
    for t in SEED_TECHS:
        for s in SEEDS:
            c = f"{t}_dn_lgs{s}"
            if (CK / f"{c}_nmos_best.pt").exists():
                seed_cands.append((t, c))
    tasks = [(t, c, g) for (t, c) in seed_cands for g in GATES]
    print(f"[eval-seeds] {len(tasks)} tasks", flush=True)
    agg = {}  # (tech, cand) -> {passed, metric}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_task, *t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            r = f.result(); done += 1
            a = agg.setdefault((r["tech"], r["cand"]), {"passed": {}, "metric": {}})
            a["passed"][r["gate"]] = r["passed"]; a["metric"][r["gate"]] = r["metric"]
            print(f"[{done}/{len(tasks)}] {r['tech']:6s} {r['cand']:18s} "
                  f"{GLABEL[r['gate']]:5s} {'P' if r['passed'] else 'F'} m={r['metric']}", flush=True)

    # merge base-size results
    base = json.loads((OUT / "all_sizes.json").read_text())
    for b in base:
        cand = f"{b['tech']}_dn_{b['size']}"
        agg[(b["tech"], cand)] = {"passed": b["passed"], "metric": b["metric"]}

    def margin(a):
        return sum((a["metric"].get(g) or 0) / TOL[g] for g in GATES)
    def npass(a):
        return sum(1 for g in GATES if a["passed"].get(g))
    def key(tc):
        a = agg[tc]
        return (-npass(a), 0 if a["passed"].get("opamp") else 1,
                0 if a["passed"].get("ring_osc") else 1, cap_rank(tc[1]), margin(a))

    lines = ["# V6.5.4 FINAL — fresh retrain, best config per tech (size + seed)", ""]
    chosen, total = {}, 0
    for t in TECHS:
        cands = sorted([tc for tc in agg if tc[0] == t], key=key)
        best = cands[0]; chosen[t] = best[1]; total += npass(agg[best])
        lines += [f"## {t} — best: **{best[1]}** ({npass(agg[best])}/4)", "",
                  "| candidate | npass | opamp | ring | swcap | sram |",
                  "|---|---|---|---|---|---|"]
        for tc in cands:
            a = agg[tc]
            cell = lambda g: ("**P**" if a["passed"].get(g) else "F") + (
                f" {a['metric'][g]:.2f}" if a["metric"].get(g) is not None else "")
            star = " ⭐" if tc == best else ""
            lines.append(f"| `{tc[1]}`{star} | {npass(a)}/4 | {cell('opamp')} | "
                         f"{cell('ring_osc')} | {cell('switchcap')} | {cell('sram_snm')} |")
        lines.append("")
    lines += [f"## FINAL best-config-per-tech mix: **{total}/16**", "",
              "| tech | config | gates |", "|---|---|---|"]
    for t in TECHS:
        a = agg[(t, chosen[t])]
        gv = " ".join(f"{GLABEL[g]}{'✓' if a['passed'].get(g) else '✗'}" for g in GATES)
        lines.append(f"| {t} | `{chosen[t]}` | {gv} |")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines))
    (OUT / "final_mix.json").write_text(json.dumps(
        {t: {"cand": chosen[t], "passed": agg[(t, chosen[t])]["passed"],
             "metric": agg[(t, chosen[t])]["metric"]} for t in TECHS}, indent=2))
    print("\n".join(lines))
    print(f"\n[eval-seeds] FINAL {total}/16 -> {OUT/'FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()
