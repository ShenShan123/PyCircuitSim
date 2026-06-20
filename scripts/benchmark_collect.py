#!/usr/bin/env python3
"""Benchmark Phase E — aggregate the per-(size,tech) NN test logs into a report.

Parses results/benchmark_sml/<size>/<tech>/<suite>.log (captured stdout of the 6
NN suites) and emits:
  - results/benchmark_sml/REPORT.md   (human-facing tables)
  - results/benchmark_sml/benchmark_data.json  (machine-readable)

Device suites (verify_nn_multi_tech_{dc,tran}) print a SUMMARY TABLE with
NRMSE%/MRE%/R2/MaxErr per config; we aggregate per device. Complex suites print a
gate line (period/gain/charge/SNM err -> PASS/FAIL) plus a waveform metric line.
Regexes are tolerant; raw gate lines are preserved in the report.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "benchmark_sml"
SIZES = ["small", "medium", "large"]
TECHS = ["tsmc5", "tsmc7", "tsmc12", "tsmc16"]
DEV_SUITES = ["verify_nn_multi_tech_dc", "verify_nn_multi_tech_tran"]
CPX_SUITES = ["verify_complex_ring_osc", "verify_complex_opamp",
              "verify_complex_sram_snm", "verify_complex_switchcap"]

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# ── parametric summary-table row: "label | sweep | nrmse | mre | r2 | maxerrUNIT | status"
def parse_device_log(text: str) -> dict:
    rows = []
    for ln in text.splitlines():
        if ln.count(" | ") < 5:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 7:
            continue
        label, sweep = parts[0], parts[1]
        m = re.match(rf"({NUM})", parts[2])
        if not m:                       # header / separator row
            continue
        try:
            nrmse = float(parts[2]); mre = float(parts[3]); r2 = float(parts[4])
        except ValueError:
            continue
        me = re.match(rf"({NUM})\s*([a-zA-Z%]*)", parts[5])
        maxerr = float(me.group(1)) if me else float("nan")
        unit = me.group(2) if me else ""
        status = parts[6]
        dev = "nmos" if "nmos" in label.lower() else ("pmos" if "pmos" in label.lower() else "all")
        rows.append(dict(label=label, sweep=sweep, dev=dev, nrmse=nrmse, mre=mre,
                         r2=r2, maxerr=maxerr, unit=unit,
                         passed=status.upper().startswith("PASS")))
    out = {}
    for dev in ("nmos", "pmos", "all"):
        drows = [r for r in rows if r["dev"] == dev]
        if not drows:
            continue
        base = next((r for r in drows if "base" in r["label"].lower()), drows[0])
        out[dev] = dict(
            n=len(drows), n_pass=sum(r["passed"] for r in drows),
            nrmse_mean=round(mean(r["nrmse"] for r in drows), 2),
            nrmse_med=round(median(r["nrmse"] for r in drows), 2),
            nrmse_max=round(max(r["nrmse"] for r in drows), 2),
            mre_mean=round(mean(r["mre"] for r in drows), 2),
            r2_min=round(min(r["r2"] for r in drows), 4),
            maxerr_max=round(max(r["maxerr"] for r in drows), 3),
            unit=drows[0]["unit"],
            base_nrmse=round(base["nrmse"], 2), base_mre=round(base["mre"], 2),
            base_r2=round(base["r2"], 5), base_maxerr=round(base["maxerr"], 3),
        )
    res = re.search(r"RESULT:\s*(.+)", text)
    return {"by_dev": out, "result": res.group(1).strip() if res else "?"}

WAVE = re.compile(rf"MRE=({NUM})%\s+R2=({NUM})\s+NRMSE=({NUM})%\s+MaxErr=({NUM})\s*([a-zA-Z]*)")

def _wave(text: str):
    m = WAVE.search(text)
    if not m:
        return None
    return dict(mre=float(m.group(1)), r2=float(m.group(2)),
                nrmse=float(m.group(3)), maxerr=float(m.group(4)), unit=m.group(5))

def parse_complex_log(suite: str, text: str) -> dict:
    d = {"wave": _wave(text), "gate": None, "headline": "", "raw": []}
    pat = {
        "verify_complex_ring_osc": rf"period error\s*=\s*({NUM})%\s*->\s*(PASS|FAIL)",
        "verify_complex_opamp": rf"gain error\s*=\s*({NUM})%\s+trip shift\s*=\s*({NUM})mV\s*->\s*(PASS|FAIL)",
        "verify_complex_switchcap": rf"charge err\s*=\s*({NUM})%.*?->\s*(PASS|FAIL)",
    }.get(suite)
    for ln in text.splitlines():
        if "->" in ln and ("PASS" in ln or "FAIL" in ln):
            d["raw"].append(ln.strip())
        if "SNM" in ln and ("DN" in ln or "NG" in ln):
            d["raw"].append(ln.strip())
    if pat:
        m = re.search(pat, text)
        if m:
            d["gate"] = m.groups()[-1]
            if suite == "verify_complex_opamp":
                d["headline"] = f"gain_err={m.group(1)}% trip_shift={m.group(2)}mV"
            elif suite == "verify_complex_ring_osc":
                d["headline"] = f"period_err={m.group(1)}%"
            elif suite == "verify_complex_switchcap":
                d["headline"] = f"charge_err={m.group(1)}%"
    if suite == "verify_complex_sram_snm":
        snm = re.search(rf"NG SNM=({NUM})mV\s+DN SNM=({NUM})mV", text)
        if snm:
            d["headline"] = f"NG_SNM={snm.group(1)}mV DN_SNM={snm.group(2)}mV"
        # overall gate: PASS only if every "-> PASS/FAIL" gate in the log passed
        verdicts = re.findall(r"->\s*(PASS|FAIL)", text)
        if verdicts:
            d["gate"] = "PASS" if all(v == "PASS" for v in verdicts) else "FAIL"
    # fallback overall gate from any explicit verdict
    if d["gate"] is None:
        vs = re.findall(r"->\s*(PASS|FAIL)", text)
        d["gate"] = ("PASS" if vs and all(v == "PASS" for v in vs) else ("FAIL" if vs else "?"))
    return d

def resolver_line(text: str) -> str:
    m = re.search(r"\[NN-resolver\].*?->\s*(\S+)", text)
    return m.group(1) if m else "?"

def load():
    data = {}
    for size in SIZES:
        data[size] = {}
        for tech in TECHS:
            d = {"dev": {}, "cpx": {}, "ckpt": None}
            for suite in DEV_SUITES:
                p = BASE / size / tech / f"{suite}.log"
                if p.exists():
                    t = p.read_text(errors="replace")
                    d["dev"][suite] = parse_device_log(t)
                    d["ckpt"] = d["ckpt"] or resolver_line(t)
            for suite in CPX_SUITES:
                p = BASE / size / tech / f"{suite}.log"
                if p.exists():
                    t = p.read_text(errors="replace")
                    d["cpx"][suite] = parse_complex_log(suite, t)
                    d["ckpt"] = d["ckpt"] or resolver_line(t)
            data[size][tech] = d
    return data

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

CPX_SHORT = {"verify_complex_ring_osc": "ring_osc", "verify_complex_opamp": "opamp",
             "verify_complex_sram_snm": "sram_snm", "verify_complex_switchcap": "switchcap"}

def report(data) -> str:
    L = ["# DirectNet (LEVEL=73) capacity benchmark — small / medium / large",
         "",
         "All checkpoints trained on ONE identical clean recipe "
         "(`--apply-filter off --swa-mode ema --seed 42`); capacity is the only "
         "variable. Datasets = full Vth + geometry grid per tech "
         "(`--variants all`, inv-trip + subvt-off overlays). Ground truth = NGSPICE "
         "BSIM-CMG (LEVEL=72), repo ngspice-45.2, CPU-pinned.",
         "",
         "Sizes: small=128x3 (~0.06M p) / medium=256x5 (~0.4M p) / large=384x6 (~0.9M p).",
         ""]

    # ── cross-size headline ──
    L += ["## Cross-size summary", ""]
    srows = []
    for size in SIZES:
        gates = npass = 0
        nrmses = []
        for tech in TECHS:
            d = data[size][tech]
            for suite in CPX_SUITES:
                c = d["cpx"].get(suite)
                if c:
                    gates += 1
                    npass += 1 if c["gate"] == "PASS" else 0
            for suite in DEV_SUITES:
                dv = d["dev"].get(suite)
                if dv:
                    for dev in dv["by_dev"].values():
                        nrmses.append(dev["nrmse_mean"])
        dev_nrmse = round(mean(nrmses), 2) if nrmses else None
        srows.append([size, f"{npass}/{gates}" if gates else "—",
                      dev_nrmse if dev_nrmse is not None else "—"])
    L += [md_table(["Size", "Complex gates PASS", "Device mean-NRMSE% (all sweeps)"], srows), ""]

    # ── per-size detail ──
    for size in SIZES:
        L += [f"## Size = {size}", ""]
        # device table
        L += ["### Device-level parametric sweeps (per tech)", "",
              "DC = Id-Vgs over L/NFIN/VT; INV = inverter VTC+transient. "
              "Values: baseline NRMSE% / mean NRMSE% over all sweep configs; "
              "MRE% (mean); R2 (min); pass-rate.", ""]
        hdr = ["Tech", "Suite", "Dev", "base NRMSE%", "mean NRMSE%", "mean MRE%",
               "min R2", "max MaxErr", "Pass"]
        rows = []
        for tech in TECHS:
            d = data[size][tech]
            for suite in DEV_SUITES:
                tag = "DC" if suite.endswith("dc") else "INV"
                dv = d["dev"].get(suite)
                if not dv:
                    rows.append([tech, tag, "—", "—", "—", "—", "—", "—", "—"]); continue
                for dev, s in dv["by_dev"].items():
                    rows.append([tech, tag, dev, s["base_nrmse"], s["nrmse_mean"],
                                 s["mre_mean"], s["r2_min"],
                                 f"{s['maxerr_max']}{s['unit']}",
                                 f"{s['n_pass']}/{s['n']}"])
        L += [md_table(hdr, rows), ""]
        # complex table
        L += ["### Complex circuits (per tech)", "",
              "Gate verdict + headline + waveform NRMSE%/R2.", ""]
        chdr = ["Tech", "Circuit", "Gate", "Headline", "NRMSE%", "R2", "MaxErr"]
        crows = []
        for tech in TECHS:
            d = data[size][tech]
            for suite in CPX_SUITES:
                c = d["cpx"].get(suite)
                short = CPX_SHORT[suite]
                if not c:
                    crows.append([tech, short, "—", "—", "—", "—", "—"]); continue
                w = c["wave"]
                crows.append([tech, short, c["gate"], c["headline"] or "—",
                              w["nrmse"] if w else "—", w["r2"] if w else "—",
                              (f"{w['maxerr']}{w['unit']}" if w else "—")])
        L += [md_table(chdr, crows), ""]
        # checkpoints used
        L += ["<details><summary>checkpoints resolved</summary>", ""]
        for tech in TECHS:
            L.append(f"- {tech}: `{data[size][tech]['ckpt']}`")
        L += ["", "</details>", ""]
    return "\n".join(L)

def main():
    data = load()
    (BASE / "benchmark_data.json").write_text(json.dumps(data, indent=1))
    (BASE / "REPORT.md").write_text(report(data))
    print(f"Wrote {BASE/'REPORT.md'} and benchmark_data.json")
    # quick console tally
    for size in SIZES:
        g = p = 0
        for tech in TECHS:
            for suite in CPX_SUITES:
                c = data[size][tech]["cpx"].get(suite)
                if c:
                    g += 1; p += 1 if c["gate"] == "PASS" else 0
        print(f"  {size:7s}: complex gates {p}/{g}")

if __name__ == "__main__":
    main()
