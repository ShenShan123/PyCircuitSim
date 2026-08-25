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

Available sources (each rendered report is pinned to one complete pass):

    results/a3_regate/REPORT.md    V6.13.0, complex matrix, single-run
    results/v710_regate/data.json  V7.1.0, device + AC + strict OMP
    results/v730_regate/data.json  V7.3.0, recipes + PFN campaign
    results/v740_regate/data.json  V7.4.0, clean DN + BSIM-AR rebuild
    results/v7516_clean/data.json   V7.5.16, clean DirectNet/BSIM-AR re-gate

Run after any re-gate:

    python scripts/v710_regate_collect.py --root results/v730_regate
    python scripts/v730_docs_build.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import pathlib
import re
from typing import Dict, Generator, List, Optional, Tuple

if __package__:
    from .v710_regate_collect import is_verdict
else:
    from v710_regate_collect import is_verdict

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
REPORT_SUITES: Dict[str, Tuple[str, ...]] = {
    "verify_complex_ring_osc": STRICT_OMP,
    "verify_complex_opamp": STRICT_OMP,
    "verify_complex_sram_snm": ("omp1",),
    "verify_complex_switchcap": ("omp1",),
    "verify_nn_multi_tech_dc": ("omp1",),
    "verify_nn_multi_tech_tran": ("omp1",),
    "verify_nn_ac": ("omp1",),
    "verify_complex_opamp_ac": ("omp1",),
}

_REPORT_PAYLOAD_KEYS: Dict[str, Tuple[str, ...]] = {
    "verify_complex_ring_osc": ("metric",),
    "verify_complex_opamp": ("metric",),
    "verify_complex_sram_snm": ("metric",),
    "verify_complex_switchcap": ("metric",),
    "verify_nn_multi_tech_dc": (
        "n", "n_pass", "mean_nrmse", "max_nrmse", "mean_mre", "min_r2",
        "max_error",
    ),
    "verify_nn_multi_tech_tran": (
        "n", "n_pass", "mean_nrmse", "max_nrmse", "mean_mre", "min_r2",
        "max_error",
    ),
    "verify_complex_opamp_ac": (
        "dc_gain_err_db", "gbw_ratio", "pm_err_deg", "mag_nrmse_pct", "status",
    ),
}
_DEVICE_AC_PAYLOAD_KEYS = (
    "gain0_err_db", "f3db_ratio", "mag_nrmse_pct", "status",
)

# Clean control per family. V7.4.0 retrained DirectNet and BSIM-AR from scratch
# on the clean recipe into the production slots at every tier, so clean@large is
# simply `large` — the v660clean archive detour (needed while the DN `large`
# slot carried the crit30f curriculum) no longer applies.
CLEAN = {
    "dn": {t: t for t in TIERS},
    "tf": {t: t for t in TIERS},
    "pfn": {t: t for t in TIERS},
}
CLEAN_OVERRIDE: Dict[Tuple[str, str, str], str] = {}

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
PASSES = [("V7.1.0", load_json("v710_regate")), ("V7.3.0", load_json("v730_regate")),
          ("V7.4.0", load_json("v740_regate")),
          ("V7.4.2", load_json("v742_regate")),
          ("2026-08-19 recheck", load_json("simple_recheck_24c181a")),
          ("V7.5.16", load_json("v7516_clean"))]
PASS_DATA = dict(PASSES)
ACTIVE_PASS: Optional[str] = None

# Every report is rendered from one coherent campaign. A later partial pass is
# never allowed to backfill itself from older cells and overwrite a complete
# published report.
# DirectNet and BSIM-AR use one complete V7.5.16 campaign. PFN remains pinned
# to its retained V7.3.0 report; a stopped partial retrain must not replace it.
# The builder never backfills a partial current campaign from older cells.
REPORT_PASS: Dict[Tuple[str, bool], str] = {
    ("dn", False): "V7.5.16",
    ("tf", False): "V7.5.16",
    ("pfn", False): "V7.3.0",
    ("dn", True): "V7.3.0",
    ("tf", True): "V7.3.0",
    ("pfn", True): "V7.3.0",
}

CURRENT_CLEAN_TAGS = ("dn", "tf")

# The new hardware does not carry the raw V7.3 recipe/PFN trees. These digests
# make the retained rendered reports immutable and keep --check meaningful.
PRESERVED_REPORT_SHA256: Dict[Tuple[str, bool], str] = {
    ("dn", False): "b4037e7f303a5f680bea3380efec6c022aa9a23132e5009ed5c9d43c7b34d1d3",
    ("dn", True): "cf5e72584234e4c4e15b3759bfc09d1cbbdef4742e5fe664cd501c000a241805",
    ("tf", False): "de54d9da78b0eddc66ffef97f487a04c0d64ea6cc9e26d2cc18ae65694590618",
    ("tf", True): "c79255aabecd3e9b1b320e403e747d790482046a829b1c40c44299d6bd758a2e",
    ("pfn", False): "b9be502ed26b3d380cfaceb2ba938e907ac4dddb4de1a77884b35e4b9832c104",
    ("pfn", True): "e48c24415a76876dcd1c62bb249fdddee772a7bc30fb1c60260b333cfdb554e7",
}
PRESERVED_README_SHA256 = (
    "c7b2743dbfaab7adddfb8abd19ea76b3a4a75264aa7b49f152d69e7f358b8f69"
)


@contextmanager
def evidence_pass(version: str) -> Generator[None, None, None]:
    """Temporarily pin all table lookups to one campaign pass."""
    global ACTIVE_PASS
    previous = ACTIVE_PASS
    ACTIVE_PASS = version
    try:
        yield
    finally:
        ACTIVE_PASS = previous


def raw(tag: str, variant: str, suite: str, tech: str) -> Optional[Dict]:
    """Measured cell from the active report pass, or newest legacy fallback."""
    if ACTIVE_PASS is not None:
        entry = (PASS_DATA.get(ACTIVE_PASS, {}).get(tag, {}).get(variant, {})
                 .get(suite, {}).get(tech))
        return entry if entry else None
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
    for suite, title, error_unit in (
            ("verify_nn_multi_tech_dc", "Parametric DC — `verify_nn_multi_tech_dc`",
             "µA"),
            ("verify_nn_multi_tech_tran", "Parametric transient — `verify_nn_multi_tech_tran`",
             "mV")):
        out += [f"**{title}** *(mean NRMSE % / mean MRE % / min R² / "
                f"max error {error_unit}; config fails in brackets)*", "",
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
                max_error = e["max_error"] * (1e6 if suite.endswith("_dc") else 1.0)
                cells.append(
                    f"{e['mean_nrmse']:.2f} / {e['mean_mre']:.2f} / "
                    f"{e['min_r2']:.3f} / {max_error:.3g}{extra}"
                )
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
            "*(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, "
            "valid refined reference and converged NN OP)*", "",
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

# Raw campaign trees are local evidence and may be absent on another clone.
# These values are the durable published fallback for the README only; the
# rendered labels still identify the code state that produced each result.
HISTORICAL_CLEAN: Dict[str, Tuple[str, str, int, int]] = {
    "dn": ("2026-08-19", "large", 12, 20),
    "tf": ("2026-08-19", "small", 13, 20),
    "pfn": ("V7.3", "small", 14, 20),
}
HISTORICAL_RECIPE: Dict[str, Tuple[str, int, int]] = {
    "dn": ("`crit15m`@xl", 19, 20),
    "tf": ("`corroft`@medium", 20, 20),
    "pfn": ("`corroft`@small", 14, 20),
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
    """Highest strict pass fraction in the pinned pass; cheaper tie wins."""
    best = ("—", 0, 0)
    for label, key in _groups(tag, recipes):
        p, n = _score(tag, key, recipes)
        if n and (not best[2] or p / n > best[1] / best[2]):
            best = (label, p, n)
    return best


def _report_result_complete(suite: str, result: object) -> bool:
    """Whether one verdict includes the metrics its report table consumes."""
    if not isinstance(result, dict) or not is_verdict(result):
        return False
    if suite == "verify_nn_ac":
        for device in ("nmos", "pmos"):
            payload = result.get(device)
            if not isinstance(payload, dict) or any(
                payload.get(key) is None for key in _DEVICE_AC_PAYLOAD_KEYS
            ):
                return False
        return True
    required = _REPORT_PAYLOAD_KEYS.get(suite)
    return required is not None and all(result.get(key) is not None
                                        for key in required)


def _matrix_complete_in_pass(tag: str, recipes: bool, version: str) -> bool:
    """Whether one pass fully measured every table cell in this report."""
    data = PASS_DATA.get(version, {})
    if not data:
        return False
    for _, key in _groups(tag, recipes):
        for tech in TECHS:
            variant = _variant(tag, key, tech, recipes)
            for suite, required in REPORT_SUITES.items():
                entry = (data.get(tag, {}).get(variant, {})
                         .get(suite, {}).get(tech))
                if not entry or any(
                    omp not in entry
                    or not _report_result_complete(suite, entry[omp])
                    for omp in required
                ):
                    return False
    return True


def scoreboard(_tag=None, _recipes=None) -> str:
    out = ["| LEVEL | family | role | current / best clean | historical best recipe | CPU cost |",
           "|---|---|---|---|---|---|"]
    for tag in ("dn", "tf", "pfn"):
        lvl, role, cost = FAMILY_META[tag]
        clean_version = REPORT_PASS[(tag, False)]
        clean_complete = _matrix_complete_in_pass(tag, False, clean_version)
        recipe_complete = _matrix_complete_in_pass(tag, True, "V7.3.0")
        if clean_complete:
            with evidence_pass(clean_version):
                cl, cp, cn = _best(tag, False)
            clean_source_version = clean_version.removesuffix(" recheck")
        else:
            clean_source_version, cl, cp, cn = HISTORICAL_CLEAN[tag]
        if recipe_complete:
            with evidence_pass("V7.3.0"):
                rl, rp, rn = _best(tag, True)
        else:
            rl, rp, rn = HISTORICAL_RECIPE[tag]
        if tag == "dn" and clean_complete:
            with evidence_pass(clean_version):
                served_pass, served_total = _score(tag, "large", False)
            c = (f"{clean_source_version} `large` "
                 f"**{served_pass}/{served_total}** served; "
                 f"`{cl}` **{cp}/{cn}** best")
        else:
            c = f"{clean_source_version} `{cl}` **{cp}/{cn}**"
        r = f"V7.3 {rl} **{rp}/{rn}**"
        out.append(f"| {lvl} | **{FAM[tag]}** | {role} | {c} | {r} | {cost} |")
    out.append("")
    out.append("Strict = passes at OMP ∈ {1, 2, 4}. " + denominator_note(None, None))
    return "\n".join(out)


def _techs_measured(tag: str, recipes: bool) -> List[str]:
    """Techs with at least one measured complex cell in this report's groups."""
    got = []
    for tech in TECHS:
        for _, key in _groups(tag, recipes):
            if any(strict(tag, _variant(tag, key, tech, recipes), c, tech)[0]
                   for c in CIRCS):
                got.append(tech)
                break
    return got


def denominator_note(tag: Optional[str], recipes: Optional[bool]) -> str:
    """State the denominator the tables actually use, never the intended one.

    TSMC6 recipe checkpoints are trained in V7.3.0, so a recipe report is /16
    before that wave lands and /20 after. Writing either number into the prose
    guarantees it is wrong half the time; deriving it cannot be.
    """
    if tag is None:
        scopes = [(t, r) for t in ("dn", "tf", "pfn") for r in (False, True)]
    else:
        scopes = [(tag, bool(recipes))]
    sizes = {len(_techs_measured(t, r)) * 4 for t, r in scopes}
    sizes.discard(0)
    if sizes == {20}:
        return ("Totals are **/20** — 4 circuits × 5 techs, TSMC6 included "
                "(`methodology.md` §2). Earlier reports scored /16 over four "
                "techs, so a "
                "/20 total here and a /16 total there can be the same "
                "measurement.")
    if not sizes:
        return "No complex cells measured yet."
    lo, hi = min(sizes), max(sizes)
    span = f"**/{lo}**" if lo == hi else f"**/{lo}–/{hi}**"
    return (f"Totals are {span}, against the /20 these reports target "
            f"(4 circuits × 5 techs): some groups have no TSMC6 checkpoint "
            f"measured yet. **Compare the fractions, not the counts.**")


BUILDERS = {
    "SCOREBOARD": lambda tag, recipes: scoreboard(),
    "DENOM": denominator_note,
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
    version = REPORT_PASS[(tag, recipes)]
    complete = _matrix_complete_in_pass(tag, recipes, version)
    dest = DOCS / f"{FILE_STEM[tag]}-{kind}.md"
    if not complete:
        expected = PRESERVED_REPORT_SHA256.get((tag, recipes))
        actual = (hashlib.sha256(dest.read_bytes()).hexdigest()
                  if dest.exists() else None)
        preserved = expected is not None and actual == expected
        state = "checksum verified" if preserved else "MISSING OR DRIFTED"
        print(f"  {dest.name}: {version} raw matrix incomplete, {state}")
        return preserved
    with evidence_pass(version):
        text = tpl.read_text()
        for marker, fn in BUILDERS.items():
            token = f"<!--{marker}-->"
            if token in text:
                text = text.replace(token, fn(tag, recipes))
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
    dest = DOCS / "README.md"
    incomplete = [
        f"{tag}@{REPORT_PASS[(tag, False)]}"
        for tag in CURRENT_CLEAN_TAGS
        if not _matrix_complete_in_pass(
            tag, False, REPORT_PASS[(tag, False)]
        )
    ]
    if incomplete:
        actual = (hashlib.sha256(dest.read_bytes()).hexdigest()
                  if dest.exists() else None)
        preserved = actual == PRESERVED_README_SHA256
        state = "checksum verified" if preserved else "MISSING OR DRIFTED"
        print(f"  {dest.name}: clean matrix incomplete "
              f"({', '.join(incomplete)}), {state}")
        return preserved
    text = tpl.read_text().replace("<!--SCOREBOARD-->", scoreboard())
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
    # A campaign lands one family at a time. The per-report completeness guard
    # prevents partial regeneration; --only remains a convenient way to limit
    # output to the family currently being worked on.
    ap.add_argument("--only", default=None,
                    help="Comma list of families to build (dn,tf,pfn). "
                         "Default: all. The README scoreboard spans families, "
                         "so it is skipped unless all are built.")
    ap.add_argument("--recipes", choices=["both", "clean", "recipes"],
                    default="both", help="Which report of each family to build")
    args = ap.parse_args()

    tags = ([t.strip() for t in args.only.split(",")] if args.only
            else ["dn", "tf", "pfn"])
    unknown = [t for t in tags if t not in FAM]
    if unknown:
        ap.error(f"unknown family {unknown}; choose from {sorted(FAM)}")
    kinds = {"both": (False, True), "clean": (False,), "recipes": (True,)}[args.recipes]

    ok = True
    for tag in tags:
        for recipes in kinds:
            ok &= build(tag, recipes, args.check)
    if args.only is None and args.recipes == "both":
        ok &= build_readme(args.check)
    else:
        print("[docs] README scoreboard skipped (partial build)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
