#!/usr/bin/env python3
"""Build active accuracy reports from measurement, not transcription.

Emits clean reports for the two full-terminal NN families into
``docs/accuracy/``:

    {DirectNet-L75, BSIM-AR-L76}-clean.md

The prose lives in scripts/accuracy_doc_templates/*.md.in and the tables are
substituted into `<!--MARKER-->` slots, so no table in the reports can drift
from the evidence. Markers:

    <!--HEADLINE-->     per-tier /20 strict summary
    <!--TESTCASE-->     one tier x tech matrix per simple-v1 testcase
    <!--BYTECH-->       per-tech roll-up across tiers and testcases
    <!--BYSCALE-->      per-scale roll-up across techs and testcases
    <!--DEVICE-->       parametric DC / transient, device AC, opamp AC

Available sources (each rendered report is pinned to one complete pass):

    results/v766_full_clean/data.json V7.6.6, full-terminal clean re-gate
    results/v770_full_clean/data.json V7.7.0, full-terminal clean re-gate

Run after any re-gate:

    python scripts/v710_regate_collect.py --root results/v770_full_clean
    python scripts/v730_docs_build.py --campaign v770_full_clean
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import pathlib
import sys
from typing import Dict, Generator, List, Optional, Tuple

if __package__:
    from .v710_regate_collect import is_verdict
else:
    from v710_regate_collect import is_verdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common.simple_circuit_catalog import (  # noqa: E402
    SIMPLE_V1,
    cases,
)

TPL = ROOT / "scripts" / "accuracy_doc_templates"
DOCS = ROOT / "docs" / "accuracy"

# TSMC6 is TSMC7 relabelled and stays so (methodology.md §7). V7.3.0 folds it
# into the headline anyway, by explicit decision, so every total here is /20
# rather than the /16 every earlier report used. No total below is comparable
# to a pre-V7.3.0 total without rescaling.
TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
SIMPLE_V1_CASES = cases(score_version=SIMPLE_V1)
CIRCS = [case.result_key for case in SIMPLE_V1_CASES]
SUITE_FOR_CIRC = {
    case.result_key: case.campaign_suite
    for case in SIMPLE_V1_CASES
}
CIRC_LABEL = {
    case.result_key: (
        case.report_label or case.label,
        case.report_gate_text
        or f"{case.gate_metric}, gate {case.gate_condition}",
    )
    for case in SIMPLE_V1_CASES
}
TIERS = ["small", "medium", "large", "xl"]
FAM = {
    "dnf": "DirectNet-Full", "tff": "BSIM-AR-Full",
}
FILE_STEM = {
    "dnf": "DirectNet-L75", "tff": "BSIM-AR-L76",
}
STRICT_OMP = ("omp1", "omp2", "omp4")
REPORT_SUITES: Dict[str, Tuple[str, ...]] = {
    **{
        case.campaign_suite: tuple(f"omp{value}" for value in case.omp_threads)
        for case in SIMPLE_V1_CASES
    },
    "verify_nn_multi_tech_dc": ("omp1",),
    "verify_nn_multi_tech_tran": ("omp1",),
    "verify_nn_ac": ("omp1",),
    "verify_circuit_opamp_ac": ("omp1",),
}

_REPORT_PAYLOAD_KEYS: Dict[str, Tuple[str, ...]] = {
    **{
        case.campaign_suite: ("metric",)
        for case in SIMPLE_V1_CASES
    },
    "verify_nn_multi_tech_dc": (
        "n", "n_pass", "mean_nrmse", "max_nrmse", "mean_mre", "min_r2",
        "max_error",
    ),
    "verify_nn_multi_tech_tran": (
        "n", "n_pass", "mean_nrmse", "max_nrmse", "mean_mre", "min_r2",
        "max_error",
    ),
    "verify_circuit_opamp_ac": (
        "dc_gain_err_db", "gbw_ratio", "pm_err_deg", "mag_nrmse_pct", "status",
    ),
}
_DEVICE_AC_PAYLOAD_KEYS = (
    "gain0_err_db", "f3db_ratio", "mag_nrmse_pct", "status",
)

CLEAN = {
    "dnf": {t: t for t in TIERS},
    "tff": {t: t for t in TIERS},
}
CLEAN_OVERRIDE: Dict[Tuple[str, str, str], str] = {}

RECIPES: Dict[str, List[Tuple[str, str]]] = {
    "dnf": [],
    "tff": [],
}


# ── evidence ────────────────────────────────────────────────────────────────
def load_json(name: str) -> Dict:
    p = ROOT / "results" / name / "data.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


PASSES = [("V7.6.1", load_json("v761_directnet_full_clean")),
          ("V7.6.1 combined", load_json("v761_full_clean")),
          ("V7.6.2", load_json("v762_directnet_full_clean")),
          ("V7.6.6", load_json("v766_full_clean")),
          ("V7.7.0", load_json("v770_full_clean")),
          ("V7.7.1", load_json("v771_full_clean"))]
PASS_DATA = dict(PASSES)
ACTIVE_PASS: Optional[str] = None
CAMPAIGN_EVIDENCE: Dict[str, Tuple[str, int, int]] = {
    "V7.6.1": ("v761_directnet_full_clean", 240, 120),
    "V7.6.1 combined": ("v761_full_clean", 480, 280),
    "V7.6.2": ("v762_directnet_full_clean", 240, 120),
    "V7.6.6": ("v766_full_clean", 480, 280),
    "V7.7.0": ("v770_full_clean", 600, 280),
    "V7.7.1": ("v771_full_clean", 600, 280),
}

# Every report is rendered from one coherent campaign. A later partial pass is
# never allowed to backfill itself from older cells and overwrite a complete
# published report.
REPORT_PASS: Dict[Tuple[str, bool], str] = {
    ("dnf", False): "V7.6.6",
    ("tff", False): "V7.6.6",
}

CURRENT_CLEAN_TAGS = ("dnf", "tff")
README_REQUIRED_CLEAN_TAGS = ("dnf", "tff")

# A preserved report digest is allowed only when its raw campaign is absent.
PRESERVED_REPORT_SHA256: Dict[Tuple[str, bool], str] = {
    ("dnf", False): "16e0a268af73c328c97a986a64452066c19ff633884e53d12f8a5905aa524c00",
    ("tff", False): "52770afe913622f9ef5621794347314aa78f23b99b1d3f1b5b7391b4d9996f18",
}
PRESERVED_README_SHA256 = (
    "0760e8b6a999af1831faec9946b4058095f24a791563e3496528624ca6b26b4a"
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


def at(
    tag: str,
    variant: str,
    suite: str,
    tech: str,
    omp: str = "omp1",
) -> Optional[Dict]:
    e = raw(tag, variant, suite, tech)
    return e.get(omp) if e else None


def clean_variant(tag: str, tier: str, tech: str) -> str:
    return CLEAN_OVERRIDE.get((tag, tier, tech), CLEAN[tag][tier])


def strict(tag: str, variant: str, circ: str, tech: str
           ) -> Tuple[Optional[str], Optional[float]]:
    """Strict verdict for one simple-v1 cell at every required thread count.

    ring_osc and opamp sit on multistable fixed points, so a single-run pass on
    a cell that flips is an artifact (methodology.md §3). sram_snm and
    switchcap are deterministic under the thread pin and read from one run.
    A cell measured only by V6.13.0 falls back to its single-run verdict and is
    reported as such, never silently promoted to strict.
    """
    suite = SUITE_FOR_CIRC[circ]
    e = raw(tag, variant, suite, tech)
    if e:
        omps = STRICT_OMP if circ in ("ring_osc", "opamp") else ("omp1",)
        got = [e[o] for o in omps if o in e]
        if got:
            rcs = {g["rc"] for g in got}
            metric = got[0].get("metric")
            if len(got) < len(omps):
                return ("PARTIAL", metric)
            if any(g.get("status") == "ERROR" for g in got):
                return ("ERROR", metric)
            if rcs == {"0"}:
                return ("PASS", metric)
            if "0" in rcs:
                return ("FLIP", metric)
            return ("FAIL", metric)
    return None, None


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
    rows = [
        "| group | strict /20 | " + " | ".join(CIRCS)
        + " | flips | open cells |",
        "|---|" + "---|" * (len(CIRCS) + 3),
    ]
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
                f"max error {error_unit}; passing/total configs in parentheses)*", "",
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
                   "verify_circuit_opamp_ac", tech)
            if not e:
                cells.append("—")
                continue
            n += 1
            ok = e["rc"] == "0"
            p += ok
            if e.get("status") == "ERROR":
                cells.append("ERROR")
            else:
                cells.append(
                    f"{'**PASS**' if ok else 'FAIL'} "
                    f"{e.get('dc_gain_err_db', '—')} dB"
                )
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
    "dnf": ("75", "**default**", "see report"),
    "tff": ("76", "autoregressive alternative", "see report"),
}

# Raw campaign trees are local evidence and may be absent on another clone.
# These values are the durable published fallback for the README only; the
# rendered labels still identify the code state that produced each result.
HISTORICAL_CLEAN: Dict[str, Tuple[str, str, int, int]] = {
    "dnf": ("V7.6.6", "large", 20, 20),
    "tff": ("V7.6.1", "—", 0, 0),
}
HISTORICAL_CLEAN_TEXT = {
    "dnf": "V7.6.6 `large` **20/20** simple-circuit matrix",
    "tff": "no complete five-technology clean matrix",
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
    if suite.startswith("verify_circuit_"):
        error = result.get("error")
        if result.get("status") == "ERROR":
            return (int(result["rc"]) != 0 and isinstance(error, str)
                    and bool(error.strip()))
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


def _campaign_provenance_complete(version: str) -> bool:
    """Whether one campaign has a complete immutable evidence set."""
    directory, expected_jobs, expected_artifacts = CAMPAIGN_EVIDENCE[version]
    root = ROOT / "results" / directory
    manifest_path = root / "campaign_manifest.json"
    provenance_path = root / "collection_provenance.json"
    data_path = root / "data.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        data_digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return False
    checkpoint_hashes = manifest.get("checkpoint_sha256")
    return (
        manifest.get("job_count") == expected_jobs
        and manifest.get("source_dirty") is False
        and isinstance(checkpoint_hashes, dict)
        and len(checkpoint_hashes) == expected_artifacts
        and all(checkpoint_hashes.values())
        and bool(manifest.get("pdk_sha256"))
        and provenance.get("campaign_manifest_sha256") == manifest_digest
        and provenance.get("data_sha256") == data_digest
        and provenance.get("source_commit") == manifest.get("source_commit")
    )


def _matrix_complete_in_pass(tag: str, recipes: bool, version: str) -> bool:
    """Whether one pass fully measured every table cell in this report."""
    if (version in CAMPAIGN_EVIDENCE
            and not _campaign_provenance_complete(version)):
        return False
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


def scoreboard(
    _tag: Optional[str] = None,
    _recipes: Optional[bool] = None,
) -> str:
    out = ["| LEVEL | family | role | current evidence | CPU cost |",
           "|---|---|---|---|---|"]
    for tag in ("dnf", "tff"):
        lvl, role, cost = FAMILY_META[tag]
        clean_version = REPORT_PASS[(tag, False)]
        clean_complete = _matrix_complete_in_pass(tag, False, clean_version)
        if clean_complete:
            with evidence_pass(clean_version):
                cl, cp, cn = _best(tag, False)
            clean_source_version = clean_version.removesuffix(" recheck")
        else:
            clean_source_version, cl, cp, cn = HISTORICAL_CLEAN[tag]
        if not clean_complete and tag in HISTORICAL_CLEAN_TEXT:
            c = HISTORICAL_CLEAN_TEXT[tag]
        else:
            c = f"{clean_source_version} `{cl}` **{cp}/{cn}**"
        out.append(f"| {lvl} | **{FAM[tag]}** | {role} | {c} | {cost} |")
    out.append("")
    out.append("Strict = passes at OMP ∈ {1, 2, 4}. " + denominator_note(None, None))
    return "\n".join(out)


def _techs_measured(tag: str, recipes: bool) -> List[str]:
    """Techs with at least one measured simple-v1 cell in these groups."""
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
        scopes = [("dnf", False), ("tff", False)]
    else:
        scopes = [(tag, bool(recipes))]
    sizes = {len(_techs_measured(t, r)) * len(CIRCS) for t, r in scopes}
    sizes.discard(0)
    if sizes == {20}:
        return (f"Totals are **/20** — {len(CIRCS)} circuits × 5 techs, "
                "TSMC6 included "
                "(`methodology.md` §2). Earlier reports scored /16 over four "
                "techs, so a "
                "/20 total here and a /16 total there can be the same "
                "measurement.")
    if not sizes:
        return "No simple-v1 cells measured yet."
    lo, hi = min(sizes), max(sizes)
    span = f"**/{lo}**" if lo == hi else f"**/{lo}–/{hi}**"
    return (f"Totals are {span}, against the /20 these reports target "
            f"({len(CIRCS)} circuits × 5 techs): some groups have no TSMC6 "
            "checkpoint "
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
    "PROVENANCE": lambda tag, recipes: campaign_provenance(tag),
}


def campaign_provenance(tag: str) -> str:
    """Render immutable provenance for a generated clean report."""
    version = REPORT_PASS[(tag, False)]
    evidence = CAMPAIGN_EVIDENCE.get(version)
    if evidence is None:
        return f"Evidence pass: {version}."
    directory, _expected_jobs, _expected_artifacts = evidence
    root = ROOT / "results" / directory
    manifest_path = root / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return (
        f"Evidence pass: **{version}**. Campaign manifest SHA-256 "
        f"`{digest}` pins gate commit `{manifest['source_commit']}`, "
        f"{manifest['job_count']} jobs, and "
        f"{len(manifest['checkpoint_sha256'])} checkpoint artifacts. "
        f"Raw evidence: `results/{directory}/`."
    )


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
        for tag in README_REQUIRED_CLEAN_TAGS
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
    ap = argparse.ArgumentParser(description="Build full-terminal accuracy reports")
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed files match the evidence; "
                         "do not write. Exit 1 if any is stale.")
    # A campaign lands one family at a time. The per-report completeness guard
    # prevents partial regeneration; --only remains a convenient way to limit
    # output to the family currently being worked on.
    ap.add_argument("--only", default=None,
                    help="Comma list of families to build (dnf,tff). "
                         "Default: all. The README scoreboard spans families, "
                         "so it is skipped unless all are built.")
    campaigns = {
        directory: version
        for version, (directory, _jobs, _artifacts) in CAMPAIGN_EVIDENCE.items()
    }
    ap.add_argument(
        "--campaign", choices=sorted(campaigns),
        help="Campaign directory under results/. Explicit selection requires "
             "complete metrics and provenance; it never uses preserved reports.",
    )
    args = ap.parse_args()

    tags = ([t.strip() for t in args.only.split(",")] if args.only
            else ["dnf", "tff"])
    unknown = [t for t in tags if t not in FAM]
    if unknown:
        ap.error(f"unknown family {unknown}; choose from {sorted(FAM)}")
    if args.campaign:
        version = campaigns[args.campaign]
        PASS_DATA[version] = load_json(args.campaign)
        incomplete = [
            tag for tag in tags
            if not _matrix_complete_in_pass(tag, False, version)
        ]
        if incomplete:
            print(f"[docs] {args.campaign}: incomplete metrics or provenance "
                  f"for {', '.join(incomplete)}; reports were not changed")
            return 1
        for tag in tags:
            REPORT_PASS[(tag, False)] = version
    ok = True
    for tag in tags:
        ok &= build(tag, False, args.check)
    if args.only is None:
        ok &= build_readme(args.check)
    else:
        print("[docs] README scoreboard skipped (partial build)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
