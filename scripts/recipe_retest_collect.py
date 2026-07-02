#!/usr/bin/env python3
"""V6.6.2 re-test — aggregate the isolated full-matrix gate logs of EVERY on-disk
recipe into a detailed continuous-accuracy comparison (per plan §5: accuracy-first,
the X/16 count is a derived summary line).

Inputs
  results/recipe_bench/gate_iso/<recipe>/<tech>_<circuit>.log   (gate_matrix_iso.sh)
  results/recipe_bench/opamp_def/<recipe>/{opamp,ring_osc}.txt  (opamp_sweep_def.sh)

Outputs
  results/recipe_bench/RETEST_ACCURACY.md   (human-facing tables)
  results/recipe_bench/retest_data.json     (machine-readable)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "results" / "recipe_bench" / "gate_iso"
OPDEF = ROOT / "results" / "recipe_bench" / "opamp_def"

TECHS = ["tsmc5", "tsmc7", "tsmc12", "tsmc16"]
CIRCS = ["ring_osc", "opamp", "sram_snm", "switchcap"]
RECIPE_ORDER = [
    "clean",
    "invtripft", "invtrip",
    "cor", "corft", "corrft", "corroft", "corro15",
    "crit10", "crit15", "crit15m", "crit15h", "crit20", "crit30",
    "csob", "cs7", "csobekv", "ekv", "sob",
    "s7", "s17", "s123",
]

NUM = r"([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?|nan)"


def _f(x):
    if x == "nan":
        return None  # solver diverged — keep the FAIL verdict, drop the value
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_ring(text: str) -> dict:
    d = {}
    m = re.search(rf"NG per\(ps\).*?\n-+\n(\w+)\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*(\w+)", text, re.S)
    if m:
        d.update(ng_period_ps=_f(m.group(2)), dn_period_ps=_f(m.group(3)),
                 period_err_pct=_f(m.group(4)), nrmse_pct=_f(m.group(5)),
                 r2=_f(m.group(6)), status=m.group(7))
    w = re.search(rf"waveform: MRE={NUM}%\s+R2={NUM}\s+NRMSE={NUM}%\s+MaxErr={NUM}mV", text)
    if w:
        d.update(mre_pct=_f(w.group(1)), maxerr_mv=_f(w.group(4)))
    return d


def parse_opamp(text: str) -> dict:
    d = {}
    m = re.search(rf"GainErr%.*?\n-+\n(\w+)\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}mV\s*\|\s*{NUM}\s*\|\s*(\w+)", text, re.S)
    if m:
        d.update(ng_gain=_f(m.group(2)), dn_gain=_f(m.group(3)),
                 gain_err_pct=_f(m.group(4)), trip_shift_mv=_f(m.group(5)),
                 nrmse_pct=_f(m.group(6)), status=m.group(7))
    w = re.search(rf"Vout curve: MRE={NUM}%\s+R2={NUM}\s+NRMSE={NUM}%\s+MaxErr={NUM}mV", text)
    if w:
        d.update(mre_pct=_f(w.group(1)), r2=_f(w.group(2)), maxerr_mv=_f(w.group(4)))
    return d


def parse_sram(text: str) -> dict:
    d = {"corners": []}
    for m in re.finditer(
            rf"(\w+)\s*\|\s*(\d+)\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*(\w+)", text):
        d["corners"].append(dict(nfin=int(m.group(2)), ng_snm_mv=_f(m.group(3)),
                                 dn_snm_mv=_f(m.group(4)), snm_err_pct=_f(m.group(5)),
                                 min_qb_mv=_f(m.group(6)), nrmse_pct=_f(m.group(7)),
                                 gate=m.group(8)))
    g = re.search(r"GATE:\s*(\w+)\s*\(all-positive:\s*(\w+)\)", text)
    if g:
        d["status"] = g.group(1)
        d["all_positive"] = g.group(2)
    if d["corners"]:
        d["nrmse_max_pct"] = max(c["nrmse_pct"] for c in d["corners"] if c["nrmse_pct"] is not None)
        d["nrmse_mean_pct"] = round(mean(c["nrmse_pct"] for c in d["corners"] if c["nrmse_pct"] is not None), 2)
        d["snm_err_max_pct"] = max(c["snm_err_pct"] for c in d["corners"] if c["snm_err_pct"] is not None)
    return d


def parse_switchcap(text: str) -> dict:
    d = {}
    m = re.search(rf"ChgErr%.*?\n-+\n(\w+)\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*{NUM}\s*\|\s*(\w+)", text, re.S)
    if m:
        d.update(ng_chg_v=_f(m.group(2)), dn_chg_v=_f(m.group(3)),
                 charge_err_pct=_f(m.group(4)), droop_pct_alw=_f(m.group(5)),
                 nrmse_pct=_f(m.group(6)), status=m.group(7))
    w = re.search(rf"waveform: MRE={NUM}%\s+R2={NUM}\s+NRMSE={NUM}%\s+MaxErr={NUM}mV", text)
    if w:
        d.update(mre_pct=_f(w.group(1)), r2=_f(w.group(2)), maxerr_mv=_f(w.group(4)))
    return d


def parse_opamp_ac(text: str) -> dict:
    NA = NUM[:-1] + "|n/a)"  # railed OP → no unity-gain crossing → "n/a"
    d = {}
    m = re.search(
        rf"(\w+)\s*\|\s*dc_gain_err=\s*{NA}dB\s*\|\s*gbw_ratio=\s*{NA}\s*\|\s*"
        rf"pm_err=\s*{NA}deg\s*\|\s*magNRMSE=\s*{NA}%\s*\|\s*(\w+)", text)
    if m:
        d.update(dc_gain_err_db=_f(m.group(2)), gbw_ratio=_f(m.group(3)),
                 pm_err_deg=_f(m.group(4)), mag_nrmse_pct=_f(m.group(5)),
                 status=m.group(6))
    if "OP-MISBIAS" in text:
        d["op_misbias"] = True
    return d


PARSERS = {"ring_osc": parse_ring, "opamp": parse_opamp,
           "sram_snm": parse_sram, "switchcap": parse_switchcap,
           "opamp_ac": parse_opamp_ac}


def load_gate(recipes):
    data = {}
    for r in recipes:
        data[r] = {}
        summ = {}
        sf = GATE / r / "SUMMARY.txt"
        if sf.exists():
            for line in sf.read_text().splitlines():
                m = re.match(r"(\S+)\s+(\S+)\s+\|\s+rc=(\d+)\s+(PASS|FAIL)", line)
                if m:
                    summ[(m.group(1).lower(), m.group(2))] = (m.group(4), int(m.group(3)))
        for t in TECHS:
            data[r][t] = {}
            for c in CIRCS + ["opamp_ac"]:
                p = GATE / r / f"{t}_{c}.log"
                cell = {"present": p.exists()}
                if p.exists():
                    cell.update(PARSERS[c](p.read_text(errors="replace")))
                v = summ.get((t, c))
                if v:
                    cell["gate"] = v[0]
                data[r][t][c] = cell
    return data


def load_opdef(recipes):
    """(recipe, TECH, circ) -> {omp: (rc, err%)} from opamp_def txt files."""
    out = {}
    for r in recipes:
        for c in ("opamp", "ring_osc"):
            f = OPDEF / r / f"{c}.txt"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                m = re.match(rf"(\S+)\s+{c}\s+OMP(\d+)\s+rc=(\d+)\s*\|\s*(.*)", line)
                if not m:
                    continue
                e = re.search(rf"(?:gain|period) error = {NUM}%", m.group(4))
                out.setdefault((r, m.group(1).lower(), c), {})[int(m.group(2))] = \
                    (int(m.group(3)), _f(e.group(1)) if e else None)
    return out


def md(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def fmt(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def cell_headline(c, circ):
    if not c.get("present"):
        return "—"
    if circ == "ring_osc":
        e, s = c.get("period_err_pct"), c.get("status", "?")
        return f"{fmt(e)} {s}"
    if circ == "opamp":
        e, s = c.get("gain_err_pct"), c.get("status", "?")
        return f"{fmt(e)} {s}"
    if circ == "switchcap":
        e, s = c.get("charge_err_pct"), c.get("status", "?")
        return f"{fmt(e)}/{fmt(c.get('droop_pct_alw'), 0)} {s}"
    e, s = c.get("nrmse_max_pct"), c.get("status", "?")
    return f"{fmt(e)} {s}"


def verdict(cell):
    """Gate verdict: exit-code line if present, else the log's own SUMMARY status."""
    return cell.get("gate") or cell.get("status")


def pass16(data, r):
    n = 0
    for t in TECHS:
        for c in CIRCS:
            if verdict(data[r][t][c]) == "PASS":
                n += 1
    return n


def strict16(data, opdef, r):
    """all-OMP strict count: opamp/ring must pass on OMP 1,2,4; sram/sc from gate."""
    n = 0
    for t in TECHS:
        for c in CIRCS:
            if c in ("opamp", "ring_osc"):
                runs = opdef.get((r, t, c))
                if runs and len(runs) >= 3 and all(rc == 0 for rc, _ in runs.values()):
                    n += 1
            elif verdict(data[r][t][c]) == "PASS":
                n += 1
    return n


def det_class(runs):
    """Classify an OMP-sweep cell: detPASS / detFAIL / FLIP."""
    if not runs or len(runs) < 3:
        return "?"
    rcs = [rc for rc, _ in runs.values()]
    if all(rc == 0 for rc in rcs):
        return "detPASS"
    if all(rc != 0 for rc in rcs):
        return "detFAIL"
    return "FLIP"


def main():
    recipes = [r for r in RECIPE_ORDER if (GATE / r).is_dir()]
    extra = sorted(p.name for p in GATE.iterdir() if p.is_dir() and p.name not in recipes)
    recipes += extra
    data = load_gate(recipes)
    opdef = load_opdef(recipes)

    L = ["# V6.6.2 re-test — full-recipe continuous-accuracy comparison",
         "",
         "Every on-disk uniform recipe at the `large` tier, re-gated on the authoritative "
         "`verify_complex_*` gates (CPU-pinned, isolated dirs, NGSPICE BSIM-CMG ground truth). "
         "Accuracy-first per plan §5: continuous metrics lead, X/16 is derived. "
         "`clean` = production control.",
         ""]

    # headline counts
    L += ["## Derived summary — pass counts", ""]
    rows = []
    for r in recipes:
        rows.append([r, f"{pass16(data, r)}/16", f"{strict16(data, opdef, r)}/16"])
    L += [md(["Recipe", "single-run OMP=1", "strict all-OMP (opamp/ring ∈ {1,2,4})"], rows), ""]

    # per-circuit headline tables
    HEAD = {"ring_osc": "period_err % (gate ≤5)", "opamp": "gain_err % (gate ≤10)",
            "sram_snm": "max lobe-NRMSE % (gate ≤10 + positivity)",
            "switchcap": "charge_err %VDD / droop %allowance (gates ≤5 / ≤100)"}
    for circ in CIRCS:
        L += [f"## {circ} — {HEAD[circ]}", ""]
        rows = []
        for t in TECHS:
            row = [t]
            for r in recipes:
                row.append(cell_headline(data[r][t][circ], circ))
            rows.append(row)
        L += [md(["Tech"] + recipes, rows), ""]

    # waveform fidelity (NRMSE) per circuit
    L += ["## Waveform / locus NRMSE % (all circuits, lower = better)", ""]
    for circ in CIRCS:
        rows = []
        for t in TECHS:
            row = [t]
            for r in recipes:
                c = data[r][t][circ]
                row.append(fmt(c.get("nrmse_max_pct") if circ == "sram_snm" else c.get("nrmse_pct")))
            rows.append(row)
        L += [f"### {circ}", "", md(["Tech"] + recipes, rows), ""]

    # opamp AC (if run)
    if any(data[r][t]["opamp_ac"].get("present") for r in recipes for t in TECHS):
        L += ["## Opamp open-loop AC — dc_gain_err dB / GBW ratio / PM err ° (gate ≤3dB, [0.6,1.67], ≤15°)", ""]
        rows = []
        for t in TECHS:
            row = [t]
            for r in recipes:
                c = data[r][t]["opamp_ac"]
                if not c.get("present") or c.get("dc_gain_err_db") is None:
                    row.append("—")
                else:
                    mb = "*" if c.get("op_misbias") else ""
                    row.append(f"{fmt(c['dc_gain_err_db'], 1)}/{fmt(c['gbw_ratio'], 2)}/"
                               f"{fmt(c['pm_err_deg'], 0)} {c.get('status', '?')}{mb}")
            rows.append(row)
        L += [md(["Tech"] + recipes, rows),
              "", "`*` = OP-MISBIAS (NN opamp output railed at the linearization point).", ""]

    # OMP determinism
    L += ["## OMP∈{1,2,4} determinism (opamp + ring) — detPASS / detFAIL / FLIP", "",
          "FLIP = multistable coin-flip (unbankable, §9 discipline #3). "
          "Cell shows class + per-OMP headline err%.", ""]
    for circ in ("opamp", "ring_osc"):
        rows = []
        for t in TECHS:
            row = [t]
            for r in recipes:
                runs = opdef.get((r, t, circ))
                cls = det_class(runs)
                if runs:
                    errs = "/".join(fmt(runs[o][1], 1) if o in runs else "?" for o in (1, 2, 4))
                    row.append(f"{cls} {errs}")
                else:
                    row.append("—")
            rows.append(row)
        L += [f"### {circ}", "", md(["Tech"] + recipes, rows), ""]

    # aggregate per-recipe scores (deterministic-aware)
    L += ["## Aggregate accuracy per recipe", "",
          "ring/SC/SRAM aggregates are means over the 4 techs (deterministic gates). "
          "The opamp columns count OMP-deterministic passes only (FLIP cells are "
          "unbankable, excluded from detPASS; mean gain_err is over detPASS cells). "
          "tsmc7-opamp (structural non-existence, fails everywhere) is excluded from "
          "the opamp mean so it doesn't drown the comparison.", ""]
    rows = []
    for r in recipes:
        ring_errs = [data[r][t]["ring_osc"].get("period_err_pct") for t in TECHS]
        ring_errs = [v for v in ring_errs if v is not None]
        sc_chg = [data[r][t]["switchcap"].get("charge_err_pct") for t in TECHS]
        sc_chg = [v for v in sc_chg if v is not None]
        sc_droop = [data[r][t]["switchcap"].get("droop_pct_alw") for t in TECHS]
        sc_droop = [v for v in sc_droop if v is not None]
        sram_nrmse = [data[r][t]["sram_snm"].get("nrmse_max_pct") for t in TECHS]
        sram_nrmse = [v for v in sram_nrmse if v is not None]
        det_pass = 0
        det_errs = []
        for t in TECHS:
            runs = opdef.get((r, t, "opamp"))
            if det_class(runs) == "detPASS":
                det_pass += 1
                if t != "tsmc7":
                    det_errs += [e for _, e in runs.values() if e is not None]
        ring_det = sum(1 for t in TECHS
                       if det_class(opdef.get((r, t, "ring_osc"))) == "detPASS")
        rows.append([
            r,
            fmt(mean(ring_errs)) if ring_errs else "—",
            f"{ring_det}/4",
            f"{det_pass}/4",
            fmt(mean(det_errs)) if det_errs else "—",
            fmt(mean(sram_nrmse)) if sram_nrmse else "—",
            fmt(mean(sc_chg)) if sc_chg else "—",
            fmt(max(sc_droop), 0) if sc_droop else "—",
        ])
    L += [md(["Recipe", "ring mean period_err%", "ring detPASS", "opamp detPASS",
              "opamp mean err% (det, excl tsmc7)", "SRAM mean maxNRMSE%",
              "SC mean charge_err%", "SC max droop %alw"], rows), ""]

    out = ROOT / "results" / "recipe_bench"
    (out / "retest_data.json").write_text(json.dumps(
        {"gate": data,
         "opdef": {f"{r}|{t}|{c}": v for (r, t, c), v in opdef.items()}}, indent=1))
    (out / "RETEST_ACCURACY.md").write_text("\n".join(L))
    print(f"Recipes: {recipes}")
    print(f"Wrote {out/'RETEST_ACCURACY.md'} and retest_data.json")
    for r in recipes:
        print(f"  {r:10s}: OMP1 {pass16(data, r):2d}/16   strict {strict16(data, opdef, r):2d}/16")


if __name__ == "__main__":
    main()
