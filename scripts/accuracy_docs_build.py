#!/usr/bin/env python3
"""Rebuild the generated halves of docs/accuracy/ from measured data.

Two of the pivot documents are machine-built so their tables cannot drift from
the evidence. Their prose lives in `scripts/accuracy_doc_templates/*.md.in`;
this script replaces the markers:

  by-tech.md   <!--CENSUS-->        per-tech cell-difficulty census
               <!--MATRIX:TSMCn-->  that tech's column of every re-gated group
               <!--DEVICE-->        per-tech device DC / transient / AC tables
  by-scale.md  <!--V710-DEV-->      per-tier device DC / transient tables
               <!--V710-AC-->       per-tier device-AC and opamp-AC tables

Sources: `results/a3_regate/REPORT.md` (V6.13.0 complex matrix, 28 groups) and
`results/v710_regate/data.json` (V7.1.0 device + AC re-gate).

Run after any re-gate:

    python scripts/v710_regate_collect.py     # -> results/v710_regate/data.json
    python scripts/accuracy_docs_build.py
    python scripts/v710_regate_control.py     # must stay at 0 disagreements

Everything else under docs/accuracy/ is hand-written.
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TPL = ROOT / "scripts" / "accuracy_doc_templates"
DOCS = ROOT / "docs" / "accuracy"

CIRCS = ["ring_osc", "opamp", "sram_snm", "switchcap"]
TECHS = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]
FAM = {"dn": "DirectNet", "tf": "BSIM-AR", "pfn": "PFN"}
TIERS = ["small", "medium", "large", "xl"]

# The V6.13.0 groups, in report order: DirectNet, BSIM-AR, PFN.
ORDER = ["dn/clean/small", "dn/clean/medium", "dn/v660clean/large", "dn/crit30f/large",
         "dn/csob/large", "dn/clean/xl", "dn/corroft/xl", "dn/crit10/xl", "dn/crit15m/xl",
         "tf/clean/small", "tf/clean/medium", "tf/clean/large", "tf/clean/xl",
         "tf/corroft/medium", "tf/corro15/medium", "tf/corroft/large", "tf/crit15m/large",
         "tf/crit30/large", "tf/invtrip/large", "tf/corroft/xl", "tf/crit15m/xl",
         "tf/crit30/xl", "tf/corro15/xl", "tf/csob/xl",
         "pfn/clean/small", "pfn/clean/medium", "pfn/clean/large"]

# Device tables list every stem the V7.1.0 pass touched, tiers first.
DEV_ROWS = [("dn", "small"), ("dn", "medium"), ("dn", "large"), ("dn", "xl"),
            ("dn", "v660clean_large"), ("dn", "csob_large"), ("dn", "corroft_xl"),
            ("dn", "crit10_xl"), ("dn", "crit15m_xl"),
            ("tf", "small"), ("tf", "medium"), ("tf", "large"), ("tf", "xl"),
            ("tf", "corroft_medium"),
            ("pfn", "small"), ("pfn", "medium"), ("pfn", "large"), ("pfn", "xl")]


# ── sources ─────────────────────────────────────────────────────────────────
def load_a3() -> dict:
    G, cur = {}, None
    for line in (ROOT / "results/a3_regate/REPORT.md").read_text().splitlines():
        m = re.match(r"^### `([^`]+)` — (\d+)/16", line)
        if m:
            cur = m.group(1); G[cur] = {}; continue
        m = re.match(r"^\| (TSMC\d+) \| (.+) \|$", line)
        if m and cur:
            for circ, val in zip(CIRCS, [p.strip() for p in m.group(2).split("|")]):
                G[cur][f"{m.group(1)}/{circ}"] = val
    return G


def load_v710() -> dict:
    p = ROOT / "results/v710_regate/data.json"
    return json.loads(p.read_text()) if p.exists() else {}


A3, V710 = load_a3(), load_v710()


def cell(tag, var, suite, tech, key=None):
    e = V710.get(tag, {}).get(var, {}).get(suite, {}).get(tech, {}).get("omp1")
    if not e:
        return None
    return e if key is None else e.get(key)


# ── by-tech blocks ──────────────────────────────────────────────────────────
def census() -> str:
    out = ["| tech | ring_osc | opamp | sram_snm | switchcap | all 108 cells |",
           "|---|---|---|---|---|---|"]
    for t in TECHS:
        c = [sum(1 for g in ORDER if A3[g].get(f"{t}/{k}", "").startswith("PASS"))
             for k in CIRCS]
        out.append(f"| **{t}** | {c[0]}/27 | {c[1]}/27 | {c[2]}/27 | {c[3]}/27 | "
                   f"**{sum(c)}/108** |")
    return "\n".join(out)


def matrix(tech: str) -> str:
    out = ["| checkpoint group | ring_osc | opamp | sram_snm | switchcap |",
           "|---|---|---|---|---|"]
    for g in ORDER:
        out.append(f"| `{g}` | " + " | ".join(
            A3[g].get(f"{tech}/{c}", "—") for c in CIRCS) + " |")
    return "\n".join(out)


def dev_by_tech(suite: str, label: str) -> str:
    out = [f"**{label}**", "", "| family / tier | " + " | ".join(TECHS) + " |",
           "|---|" + "---|" * len(TECHS)]
    any_row = False
    for tag, var in DEV_ROWS:
        vals = [cell(tag, var, suite, t) for t in TECHS]
        if not any(v and "n" in v for v in vals):
            continue
        any_row = True
        cells = []
        for v in vals:
            if not v or "n" not in v:
                cells.append("—")
            else:
                cells.append(f"{v['mean_nrmse']:.2f}" +
                             ("" if v["n_pass"] == v["n"] else f" ({v['n_pass']}/{v['n']})"))
        out.append(f"| {FAM[tag]} `{var}` | " + " | ".join(cells) + " |")
    return "\n".join(out) if any_row else ""


def _ac_why(r: dict) -> str:
    if r["status"] == "PASS":
        return "✓"
    why = []
    if float(r["gain0_err_db"]) > 1.5:
        why.append(f"gain {r['gain0_err_db']} dB")
    f3 = float(r["f3db_ratio"])
    if not 0.7 <= f3 <= 1.43:
        why.append(f"f3db {f3:.2f}")
    if float(r["mag_nrmse_pct"]) > 10:
        why.append(f"mag {r['mag_nrmse_pct']} %")
    return "✗ " + ", ".join(why) if why else "✗"


def ac_by_tech() -> str:
    out = ["**Device CS-amp AC — NMOS / PMOS verdicts**", "",
           "| family / tier | " + " | ".join(TECHS) + " |",
           "|---|" + "---|" * len(TECHS)]
    for tag, var in DEV_ROWS:
        cells, any_c = [], False
        for t in TECHS:
            e = cell(tag, var, "verify_nn_ac", t)
            if not e or "nmos" not in e:
                cells.append("—"); continue
            any_c = True
            cells.append(" / ".join(
                _ac_why(e[d]) if e.get(d) else "—" for d in ("nmos", "pmos")))
        if any_c:
            out.append(f"| {FAM[tag]} `{var}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ── by-scale blocks ─────────────────────────────────────────────────────────
def ac_by_tier() -> str:
    out = ["**Device CS-amp AC (gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7,1.43], "
           "magNRMSE ≤10 %)**", "",
           "| family | tier | pass | per-tech (n = NMOS, p = PMOS) |", "|---|---|---|---|"]
    for tag in ("dn", "tf", "pfn"):
        for var in TIERS:
            n = p = 0; det = []
            for t in TECHS:
                e = cell(tag, var, "verify_nn_ac", t)
                if not e:
                    det.append(f"{t}: —"); continue
                s = []
                for d in ("nmos", "pmos"):
                    r = e.get(d)
                    if r:
                        n += 1; p += r["status"] == "PASS"
                        s.append(f"{d[0]}{'✓' if r['status'] == 'PASS' else '✗'}")
                det.append(f"{t}: {' '.join(s)}")
            if n:
                out.append(f"| {FAM[tag]} | {var} | **{p}/{n}** | " + " · ".join(det) + " |")

    out += ["", "**Opamp open-loop AC (gate: DC-gain err ≤3 dB, GBW ratio ∈[0.6,1.67], "
            "PM err ≤15°, and a non-railed NN OP; magNRMSE reported, not gated). "
            "The number shown is the DC-gain error**", "",
            "| family | tier | pass | TSMC5 | TSMC7 | TSMC12 | TSMC16 |",
            "|---|---|---|---|---|---|---|"]
    for tag in ("dn", "tf", "pfn"):
        for var in TIERS:
            n = p = 0; cells = []
            for t in TECHS:
                e = cell(tag, var, "verify_complex_opamp_ac", t)
                if not e:
                    cells.append("—"); continue
                n += 1; p += e["rc"] == "0"
                cells.append(f"{'PASS' if e['rc'] == '0' else 'FAIL'} "
                             f"{e.get('dc_gain_err_db', '—')} dB")
            if n:
                out.append(f"| {FAM[tag]} | {var} | **{p}/{n}** | " + " | ".join(cells) + " |")
    return "\n".join(out)


def dev_by_tier() -> str:
    blocks = []
    for suite, label in (("verify_nn_multi_tech_dc",
                          "Parametric DC — `verify_nn_multi_tech_dc`, 55 configs"),
                         ("verify_nn_multi_tech_tran",
                          "Parametric transient — `verify_nn_multi_tech_tran`, 64 configs")):
        out = [f"**{label}**", "",
               "| family | tier | pass | mean NRMSE % — TSMC5 / 7 / 12 / 16 |",
               "|---|---|---|---|"]
        for tag in ("dn", "tf", "pfn"):
            for var in TIERS:
                g = {t: cell(tag, var, suite, t) for t in TECHS}
                g = {t: e for t, e in g.items() if e and "n" in e}
                if len(g) == 4:
                    out.append(f"| {FAM[tag]} | {var} | "
                               f"{sum(e['n_pass'] for e in g.values())}/"
                               f"{sum(e['n'] for e in g.values())} | " +
                               " / ".join(f"{g[t]['mean_nrmse']:.2f}" for t in TECHS) + " |")
        blocks.append("\n".join(out))

    out = ["**Non-tier (recipe) stems measured in the same pass**", "",
           "| stem | device AC | opamp AC | DC | tran |", "|---|---|---|---|---|"]
    for tag, var in DEV_ROWS:
        if var in TIERS:
            continue
        acn = acp = oan = oap = 0
        for t in TECHS:
            e = cell(tag, var, "verify_nn_ac", t)
            if e:
                for d in ("nmos", "pmos"):
                    if e.get(d):
                        acn += 1; acp += e[d]["status"] == "PASS"
            o = cell(tag, var, "verify_complex_opamp_ac", t)
            if o:
                oan += 1; oap += o["rc"] == "0"
        f = lambda s: (lambda g: f"{sum(e['n_pass'] for e in g.values())}/"
                                 f"{sum(e['n'] for e in g.values())}" if len(g) == 4 else "—")(
            {t: cell(tag, var, s, t) for t in TECHS
             if cell(tag, var, s, t) and "n" in cell(tag, var, s, t)})
        if acn or oan:
            out.append(f"| `{tag}/{var}` | {acp}/{acn} | {oap}/{oan} | "
                       f"{f('verify_nn_multi_tech_dc')} | {f('verify_nn_multi_tech_tran')} |")
    blocks.append("\n".join(out))
    return "\n\n".join(blocks)


# ── build ───────────────────────────────────────────────────────────────────
def main() -> int:
    t = (TPL / "by-tech.md.in").read_text()
    t = t.replace("<!--CENSUS-->", census())
    for tech in TECHS:
        t = t.replace(f"<!--MATRIX:{tech}-->", matrix(tech))
    t = t.replace("<!--DEVICE-->", "\n\n".join(x for x in (
        dev_by_tech("verify_nn_multi_tech_dc",
                    "Parametric DC — mean Id-Vgs NRMSE % per tech "
                    "(config fails in brackets)"),
        dev_by_tech("verify_nn_multi_tech_tran",
                    "Parametric transient — mean NRMSE % per tech"),
        ac_by_tech()) if x))
    (DOCS / "by-tech.md").write_text(t)

    s = (TPL / "by-scale.md.in").read_text()
    s = s.replace("<!--V710-DEV-->", dev_by_tier()).replace("<!--V710-AC-->", ac_by_tier())
    (DOCS / "by-scale.md").write_text(s)

    print(f"[docs-build] by-tech.md + by-scale.md rebuilt "
          f"({sum(len(v) for v in V710.values())} V7.1.0 stems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
