#!/usr/bin/env python3
"""V7.3.0 — build the six accuracy reports from measurement, not transcription.

Emits two files per model family into docs/accuracy/:

    {DirectNet-L73, BSIM-AR-L74, PFN-L75}-{clean, recipes}.md

The prose lives in scripts/accuracy_doc_templates/*.md.in and the tables are
substituted into `<!--MARKER-->` slots, so no table in the reports can drift
from the evidence. Markers:

    <!--HEADLINE-->     per-tier /20 strict summary
    <!--TESTCASE-->     one tier x tech matrix per complex testcase
    <!--BYTECH-->       per-tech roll-up across tiers and testcases
    <!--BYSCALE-->      per-scale roll-up across techs and testcases
    <!--DEVICE-->       parametric DC / transient, device AC, opamp AC
    <!--RECIPES-->      recipe x tier gate table (recipes reports only)
    <!--RECIPEDELTA-->  per-testcase recipe-vs-clean deltas (recipes reports)

Sources, oldest to newest — a cell measured by a later pass wins:

    results/a3_regate/REPORT.md    V6.13.0, complex matrix, single-run
    results/v710_regate/data.json  V7.1.0, device + AC + strict OMP
    results/v730_regate/data.json  V7.3.0, this campaign

Run after any re-gate:

    python scripts/v710_regate_collect.py --root results/v730_regate
    python scripts/v730_docs_build.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
TPL = ROOT / "scripts" / "accuracy_doc_templates"
DOCS = ROOT / "docs" / "accuracy"

# TSMC6 is TSMC7 relabelled and stays so (methodology.md §7). V7.3.0 folds it
# into the headline anyway, by explicit decision, so every total here is /20
# rather than the /16 every earlier report used. No total below is comparable
# to a pre-V7.3.0 total without rescaling.
TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
CIRCS = ["ring_osc", "opamp", "sram_snm", "switchcap"]
CIRC_LABEL = {
    "ring_osc": ("Ring oscillator", "period error %, gate ≤5 %"),
    "opamp": ("Two-stage Miller opamp (DC)", "open-loop gain error %, gate ≤10 %"),
    "sram_snm": ("6T SRAM read SNM", "worst lobe NRMSE %, gate ≤10 % and all lobes positive"),
    "switchcap": ("Switched-capacitor cell", "charge error % of VDD, gate ≤5 %"),
}
TIERS = ["small", "medium", "large", "xl"]
FAM = {"dn": "DirectNet", "tf": "BSIM-AR", "pfn": "PFN"}
FILE_STEM = {"dn": "DirectNet-L73", "tf": "BSIM-AR-L74", "pfn": "PFN-L75"}
STRICT_OMP = ("omp1", "omp2", "omp4")

# Clean control per family; DirectNet's large slot carries crit30f in
# production, so its clean row is the v660clean archive — except on TSMC6,
# which never had a curriculum applied.
CLEAN = {
    "dn": {"small": "small", "medium": "medium",
           "large": "v660clean_large", "xl": "xl"},
    "tf": {t: t for t in TIERS},
    "pfn": {t: t for t in TIERS},
}
CLEAN_OVERRIDE = {("dn", "large", "TSMC6"): "large"}

RECIPES: Dict[str, List[Tuple[str, str]]] = {
    "dn": [("crit30f", "large"), ("csob", "large"),
           ("corroft", "xl"), ("crit15m", "xl")],
    "tf": [("corroft", "medium"), ("corro15", "medium"),
           ("corroft", "large"), ("crit15m", "large"), ("crit30", "large"),
           ("corroft", "xl"), ("corro15", "xl"),
           ("crit15m", "xl"), ("crit30", "xl")],
    "pfn": [("corroft", "small")],
}


# ── evidence ────────────────────────────────────────────────────────────────
def load_a3() -> Dict[Tuple[str, str, str, str], Tuple[str, float]]:
    """V6.13.0 complex matrix, parsed out of its markdown report.

    Group headings read `### \\`dn/clean/large\\` — 15/16`; the clean/large
    group is the production slot, whose variant name differs from the tier.
    """
    src = ROOT / "results/a3_regate/REPORT.md"
    if not src.exists():
        return {}
    out: Dict[Tuple[str, str, str, str], Tuple[str, float]] = {}
    cur: Optional[Tuple[str, str]] = None
    for line in src.read_text().splitlines():
        m = re.match(r"^### `([a-z]+)/([a-z0-9]+)/(\w+)` — ", line)
        if m:
            tag, recipe, tier = m.groups()
            variant = tier if recipe == "clean" else f"{recipe}_{tier}"
            cur = (tag, variant)
            continue
        m = re.match(r"^\| (TSMC\d+) \| (.+) \|$", line)
        if m and cur:
            vals = [p.strip() for p in m.group(2).split("|")]
            for circ, v in zip(CIRCS, vals):
                mm = re.match(r"(PASS|FAIL)\s+([\d.]+)%", v)
                if mm:
                    out[(cur[0], cur[1], m.group(1), circ)] = (
                        mm.group(1), float(mm.group(2)))
    return out


def load_json(name: str) -> Dict:
    p = ROOT / "results" / name / "data.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


A3 = load_a3()
PASSES = [("V7.1.0", load_json("v710_regate")), ("V7.3.0", load_json("v730_regate"))]


def raw(tag: str, variant: str, suite: str, tech: str) -> Optional[Dict]:
    """Newest pass that measured this cell, or None."""
    for _, data in reversed(PASSES):
        e = data.get(tag, {}).get(variant, {}).get(suite, {}).get(tech)
        if e:
            return e
    return None


def at(tag: str, variant: str, suite: str, tech: str, omp: str = "omp1"):
    e = raw(tag, variant, suite, tech)
    return e.get(omp) if e else None


def clean_variant(tag: str, tier: str, tech: str) -> str:
    return CLEAN_OVERRIDE.get((tag, tier, tech), CLEAN[tag][tier])


def strict(tag: str, variant: str, circ: str, tech: str
           ) -> Tuple[Optional[str], Optional[float]]:
    """Strict verdict for one complex cell: PASS only at every thread count.

    ring_osc and opamp sit on multistable fixed points, so a single-run pass on
    a cell that flips is an artifact (methodology.md §3). sram_snm and
    switchcap are deterministic under the thread pin and read from one run.
    A cell measured only by V6.13.0 falls back to its single-run verdict and is
    reported as such, never silently promoted to strict.
    """
    suite = f"verify_complex_{circ}"
    e = raw(tag, variant, suite, tech)
    if e:
        omps = STRICT_OMP if circ in ("ring_osc", "opamp") else ("omp1",)
        got = [e[o] for o in omps if o in e]
        if got:
            rcs = {g["rc"] for g in got}
            metric = got[0].get("metric")
            if len(got) < len(omps):
                return ("PARTIAL", metric)
            if rcs == {"0"}:
                return ("PASS", metric)
            if "0" in rcs:
                return ("FLIP", metric)
            return ("FAIL", metric)
    a = A3.get((tag, variant, tech, circ))
    return (a[0], a[1]) if a else (None, None)


# The headline metric each testcase reports, and the threshold it is judged
# against. Two gates have a second criterion the headline number cannot show:
# switchcap also bounds the hold droop, and sram_snm also requires every lobe
# to be positive. A cell can therefore FAIL with its headline metric inside
# the threshold — which reads as a contradiction unless it is marked.
GATE_THRESHOLD = {"ring_osc": 5.0, "opamp": 10.0, "sram_snm": 10.0,
                  "switchcap": 5.0}
SECOND_CRITERION = {"switchcap": "hold droop", "sram_snm": "lobe positivity"}


def verdict_mark(v: Optional[str], metric: Optional[float],
                 circ: Optional[str] = None) -> str:
    if v is None:
        return "—"
    num = f" {metric:.2f}%" if metric is not None else ""
    if (v == "FAIL" and metric is not None and circ in SECOND_CRITERION
            and metric <= GATE_THRESHOLD[circ]):
        num += "†"
    return {"PASS": f"**PASS**{num}", "FAIL": f"FAIL{num}",
            "FLIP": f"⚡FLIP{num}", "PARTIAL": f"…{num}"}.get(v, f"{v}{num}")


# ── table builders ──────────────────────────────────────────────────────────
def _groups(tag: str, recipes: bool) -> List[Tuple[str, str]]:
    """(label, variant-resolver-key) pairs for the report being built."""
    if recipes:
        return [(f"`{r}`@{t}", f"{r}_{t}") for r, t in RECIPES[tag]]
    return [(t, t) for t in TIERS]


def _variant(tag: str, key: str, tech: str, recipes: bool) -> str:
    return key if recipes else clean_variant(tag, key, tech)


def headline(tag: str, recipes: bool) -> str:
    rows = ["| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |",
            "|---|---|---|---|---|---|---|---|"]
    for label, key in _groups(tag, recipes):
        # Per-circuit denominators are counted, not divided out of the total:
        # a partly-measured group has different denominators per circuit, and
        # dividing would quietly report a pass rate against the wrong base.
        per = {c: [0, 0] for c in CIRCS}
        flips, open_cells = 0, []
        for circ in CIRCS:
            for tech in TECHS:
                v, _ = strict(tag, _variant(tag, key, tech, recipes), circ, tech)
                if v is None:
                    continue
                per[circ][1] += 1
                if v == "PASS":
                    per[circ][0] += 1
                else:
                    open_cells.append(f"{tech.lower()}-{circ}")
                    flips += v == "FLIP"
        tot = sum(n for _, n in per.values())
        if not tot:
            continue
        n_pass = sum(p for p, _ in per.values())
        cells = " | ".join(f"{per[c][0]}/{per[c][1]}" for c in CIRCS)
        oc = ", ".join(open_cells) if open_cells else "—"
        rows.append(f"| {label} | **{n_pass}/{tot}** | {cells} | {flips} | {oc} |")
    return "\n".join(rows)


def testcase_tables(tag: str, recipes: bool) -> str:
    out: List[str] = []
    for circ in CIRCS:
        title, gate = CIRC_LABEL[circ]
        out += [f"#### {title}", "", f"*Verdict is the gate's exit code; the number is the {gate}.*", "",
                "| group | " + " | ".join(TECHS) + " |",
                "|---|" + "---|" * len(TECHS)]
        dagger = False
        for label, key in _groups(tag, recipes):
            cells = []
            for tech in TECHS:
                v, m = strict(tag, _variant(tag, key, tech, recipes), circ, tech)
                mark = verdict_mark(v, m, circ)
                dagger |= mark.endswith("†")
                cells.append(mark)
            if any(c != "—" for c in cells):
                out.append(f"| {label} | " + " | ".join(cells) + " |")
        if dagger:
            out += ["", f"† failed on **{SECOND_CRITERION[circ]}**, the half of this "
                        "gate the headline number does not show — the metric above is "
                        "inside its threshold."]
        out.append("")
    return "\n".join(out).rstrip()


def by_tech_rollup(tag: str, recipes: bool) -> str:
    out = ["| tech | " + " | ".join(c.replace("_", "\\_") for c in CIRCS) +
           " | all cells |", "|---|" + "---|" * (len(CIRCS) + 1)]
    for tech in TECHS:
        per, tot = [], 0
        for circ in CIRCS:
            p = n = 0
            for _, key in _groups(tag, recipes):
                v, _ = strict(tag, _variant(tag, key, tech, recipes), circ, tech)
                if v is None:
                    continue
                n += 1
                p += v == "PASS"
            per.append((p, n))
            tot += n
        if not tot:
            continue
        tp = sum(p for p, _ in per)
        out.append(f"| **{tech}** | " + " | ".join(f"{p}/{n}" for p, n in per) +
                   f" | **{tp}/{tot}** |")
    return "\n".join(out)


def by_scale_rollup(tag: str, recipes: bool) -> str:
    out = ["| group | " + " | ".join(TECHS) + " | all |",
           "|---|" + "---|" * (len(TECHS) + 1)]
    for label, key in _groups(tag, recipes):
        cells, tp, tn = [], 0, 0
        for tech in TECHS:
            p = n = 0
            for circ in CIRCS:
                v, _ = strict(tag, _variant(tag, key, tech, recipes), circ, tech)
                if v is None:
                    continue
                n += 1
                p += v == "PASS"
            cells.append(f"{p}/{n}" if n else "—")
            tp += p
            tn += n
        if tn:
            out.append(f"| {label} | " + " | ".join(cells) + f" | **{tp}/{tn}** |")
    return "\n".join(out)


def _ac_mark(r: Optional[Dict]) -> str:
    if not r:
        return "—"
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


def device_tables(tag: str, recipes: bool) -> str:
    out: List[str] = []
    for suite, title, unit in (
            ("verify_nn_multi_tech_dc", "Parametric DC — `verify_nn_multi_tech_dc`",
             "mean Id-Vgs NRMSE %, config fails in brackets"),
            ("verify_nn_multi_tech_tran", "Parametric transient — `verify_nn_multi_tech_tran`",
             "mean NRMSE %")):
        out += [f"**{title}** *({unit})*", "",
                "| group | " + " | ".join(TECHS) + " | pass |",
                "|---|" + "---|" * (len(TECHS) + 1)]
        for label, key in _groups(tag, recipes):
            cells, p, n = [], 0, 0
            for tech in TECHS:
                e = at(tag, _variant(tag, key, tech, recipes), suite, tech)
                if not e or "mean_nrmse" not in e:
                    cells.append("—")
                    continue
                p += e["n_pass"]
                n += e["n"]
                extra = "" if e["n_pass"] == e["n"] else f" ({e['n_pass']}/{e['n']})"
                cells.append(f"{e['mean_nrmse']:.2f}{extra}")
            if n:
                out.append(f"| {label} | " + " | ".join(cells) + f" | {p}/{n} |")
        out.append("")

    out += ["**Device CS-amp AC** — NMOS / PMOS "
            "*(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*", "",
            "| group | " + " | ".join(TECHS) + " | pass /10 |",
            "|---|" + "---|" * (len(TECHS) + 1)]
    for label, key in _groups(tag, recipes):
        cells, p, n = [], 0, 0
        for tech in TECHS:
            e = at(tag, _variant(tag, key, tech, recipes), "verify_nn_ac", tech)
            if not e or "nmos" not in e:
                cells.append("—")
                continue
            for d in ("nmos", "pmos"):
                if e.get(d):
                    n += 1
                    p += e[d]["status"] == "PASS"
            cells.append(" / ".join(_ac_mark(e.get(d)) for d in ("nmos", "pmos")))
        if n:
            out.append(f"| {label} | " + " | ".join(cells) + f" | **{p}/{n}** |")

    out += ["", "**Opamp open-loop AC** — DC-gain error "
            "*(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*", "",
            "| group | " + " | ".join(TECHS) + " | pass /5 |",
            "|---|" + "---|" * (len(TECHS) + 1)]
    for label, key in _groups(tag, recipes):
        cells, p, n = [], 0, 0
        for tech in TECHS:
            e = at(tag, _variant(tag, key, tech, recipes),
                   "verify_complex_opamp_ac", tech)
            if not e:
                cells.append("—")
                continue
            n += 1
            ok = e["rc"] == "0"
            p += ok
            cells.append(f"{'**PASS**' if ok else 'FAIL'} {e.get('dc_gain_err_db', '—')} dB")
        if n:
            out.append(f"| {label} | " + " | ".join(cells) + f" | **{p}/{n}** |")
    return "\n".join(out)


def recipe_delta(tag: str) -> str:
    """Each recipe's cell-level gain and loss against clean at the same tier.

    Reported as named cells rather than a net count: a recipe that banks two
    opamps and drops two rings is not "unchanged", and the noise floor makes
    the net number the least trustworthy part of the comparison.
    """
    out = ["| recipe | tier | cells gained vs clean | cells lost vs clean | net |",
           "|---|---|---|---|---|"]
    for recipe, tier in RECIPES[tag]:
        gained, lost = [], []
        for tech in TECHS:
            for circ in CIRCS:
                cv, _ = strict(tag, clean_variant(tag, tier, tech), circ, tech)
                rv, _ = strict(tag, f"{recipe}_{tier}", circ, tech)
                if cv is None or rv is None:
                    continue
                if rv == "PASS" and cv != "PASS":
                    gained.append(f"{tech.lower()}-{circ}")
                elif cv == "PASS" and rv != "PASS":
                    lost.append(f"{tech.lower()}-{circ}")
        if not gained and not lost:
            net = "0"
        else:
            net = f"{len(gained) - len(lost):+d}"
        out.append(f"| `{recipe}` | {tier} | {', '.join(gained) or '—'} | "
                   f"{', '.join(lost) or '—'} | **{net}** |")
    return "\n".join(out)


FAMILY_META = {
    "dn": ("73", "**production**", "1.5 ms @ `large`"),
    "tf": ("74", "higher fidelity", "61.5 ms @ `medium`"),
    "pfn": ("75", "research", "15.6 ms @ `small`"),
}


def _score(tag: str, key: str, recipes: bool) -> Tuple[int, int]:
    p = n = 0
    for tech in TECHS:
        for circ in CIRCS:
            v, _ = strict(tag, _variant(tag, key, tech, recipes), circ, tech)
            if v is None:
                continue
            n += 1
            p += v == "PASS"
    return p, n


def _best(tag: str, recipes: bool) -> Tuple[str, int, int]:
    """Highest strict pass *fraction*; ties go to the cheaper group.

    A fraction, not a count, because recipe groups are scored /16 (no TSMC6
    checkpoints) while clean groups are /20 — comparing the raw counts would
    hand the win to whichever group happened to be measured on more techs.
    """
    best = ("—", 0, 0)
    for label, key in _groups(tag, recipes):
        p, n = _score(tag, key, recipes)
        if n and (not best[2] or p / n > best[1] / best[2]):
            best = (label, p, n)
    return best


def scoreboard(_tag=None, _recipes=None) -> str:
    out = ["| LEVEL | family | role | best clean tier | best recipe | CPU cost |",
           "|---|---|---|---|---|---|"]
    for tag in ("dn", "tf", "pfn"):
        lvl, role, cost = FAMILY_META[tag]
        cl, cp, cn = _best(tag, False)
        rl, rp, rn = _best(tag, True)
        c = f"`{cl}` **{cp}/{cn}**" if cn else "—"
        r = f"{rl} **{rp}/{rn}**" if rn else "*(none trained)*"
        out.append(f"| {lvl} | **{FAM[tag]}** | {role} | {c} | {r} | {cost} |")
    out.append("")
    out.append("Strict = passes at OMP ∈ {1, 2, 4}. Clean groups score **/20** "
               "(4 circuits × 5 techs); recipe groups score **/16** because "
               "curriculum checkpoints exist for the four original techs only. "
               "Compare the fractions, not the counts.")
    return "\n".join(out)


BUILDERS = {
    "SCOREBOARD": lambda tag, recipes: scoreboard(),
    "HEADLINE": headline,
    "TESTCASE": testcase_tables,
    "BYTECH": by_tech_rollup,
    "BYSCALE": by_scale_rollup,
    "DEVICE": device_tables,
    "RECIPEDELTA": lambda tag, recipes: recipe_delta(tag),
}


def build(tag: str, recipes: bool, check: bool) -> bool:
    kind = "recipes" if recipes else "clean"
    tpl = TPL / f"{FILE_STEM[tag]}-{kind}.md.in"
    if not tpl.exists():
        print(f"  no template for {FILE_STEM[tag]}-{kind}, skipped")
        return True
    text = tpl.read_text()
    for marker, fn in BUILDERS.items():
        token = f"<!--{marker}-->"
        if token in text:
            text = text.replace(token, fn(tag, recipes))
    dest = DOCS / f"{FILE_STEM[tag]}-{kind}.md"
    if check:
        same = dest.exists() and dest.read_text() == text
        print(f"  {dest.name}: {'up to date' if same else 'STALE'}")
        return same
    dest.write_text(text)
    print(f"  wrote {dest.relative_to(ROOT)}  ({len(text.splitlines())} lines)")
    return True


def build_readme(check: bool) -> bool:
    tpl = TPL / "README.md.in"
    if not tpl.exists():
        return True
    text = tpl.read_text().replace("<!--SCOREBOARD-->", scoreboard())
    dest = DOCS / "README.md"
    if check:
        same = dest.exists() and dest.read_text() == text
        print(f"  {dest.name}: {'up to date' if same else 'STALE'}")
        return same
    dest.write_text(text)
    print(f"  wrote {dest.relative_to(ROOT)}  ({len(text.splitlines())} lines)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the six accuracy reports")
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed files match the evidence; "
                         "do not write. Exit 1 if any is stale.")
    args = ap.parse_args()
    ok = True
    for tag in ("dn", "tf", "pfn"):
        for recipes in (False, True):
            ok &= build(tag, recipes, args.check)
    ok &= build_readme(args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
