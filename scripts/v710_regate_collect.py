#!/usr/bin/env python3
"""Collect the V7.1.0 re-gate (scripts/v710_regate.sh) into tables + JSON.

Reads ``results/v710_regate/<tag>/<variant>/<tech>/<suite>.omp<n>.log``, whose
verdict is the trailing ``===V710_DONE rc=N===`` marker, and extracts each
suite's headline metric so the accuracy docs can be rebuilt from measurements
rather than transcription.

Emits:
  * ``REPORT.md`` — per (tag, variant): device AC, opamp open-loop AC, device
    DC/transient, the simple-v1 4x5 matrix, strict OMP{1,2,4} verdicts, and
    separate simple-v2 diagnostic rows.
  * ``data.json`` — the same, machine-readable.

Usage: python scripts/v710_regate_collect.py [--root results/v710_regate]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common.gate_result import parse_result_markers  # noqa: E402
from tests.common.simple_circuit_catalog import (  # noqa: E402
    SIMPLE_V1,
    SIMPLE_V2,
    cases,
)

TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
SIMPLE_V1_CASES = cases(score_version=SIMPLE_V1)
SIMPLE_V2_CASES = cases(score_version=SIMPLE_V2)
CIRCS = [case.result_key for case in SIMPLE_V1_CASES]
SUITE_BY_RESULT = {
    case.result_key: case.campaign_suite
    for case in SIMPLE_V1_CASES
}
FAMILY = {
    "dn": "DirectNet (L73)",
    "tf": "BSIM-AR (L74)",
    "dnf": "DirectNet-Full (L75)",
    "tff": "BSIM-AR-Full (L76)",
}

_RC = re.compile(r"===V710_DONE rc=(\S+)===")
_PROVENANCE = re.compile(r"===V710_PROVENANCE sha256=([0-9a-f]{64})===")
_AC_ROW = re.compile(
    r"^\s*AC \| (\w+)_(nmos|pmos) \| (\S+) \| (\S+) \| (\S+) \| (\S+) \| (\S+)", re.M)
_OPAMP_AC = re.compile(
    r"^\s*(TSMC\d+)\s*\| dc_gain_err=\s*(\S+)dB \| gbw_ratio=\s*(\S+) \| "
    r"pm_err=\s*(\S+)deg \| magNRMSE=\s*(\S+)% \| (\w+)", re.M)
_SUMMARY_ERROR_ROW = re.compile(
    r"^\s*(TSMC\d+)\s*\|(?:[^|\n]*\|)*\s*ERROR\s+[—-]\s+(.+?)\s*$",
    re.M,
)
_DEV_ROW = re.compile(
    r"^\s+(TSMC\d+_\S+)\s+NRMSE=\s*(\S+)%\s+MRE=\s*(\S+)%\s+R2=\s*(\S+)\s+"
    r"MaxErr=(\S+)\s+(PASS|FAIL|ERROR)", re.M)
_DEV_ERROR_ROW = re.compile(r"^\s+(TSMC\d+_\S+)\s+ERROR:", re.M)
# SRAM's gate metric is the worst lobe NRMSE over the NFIN corners — column 7
# of its summary table, not the first "NRMSE=" the log happens to print.
_SRAM_ROW = re.compile(
    r"^\s*TSMC\d+\s*\|\s*\d+\s*\|\s*[\d.-]+\s*\|\s*[\d.-]+\s*\|\s*[\d.-]+\s*\|"
    r"\s*[\d.-]+\s*\|\s*([\d.]+)\s*\|\s*(PASS|FAIL)", re.M)
_CIRC_METRIC = {
    "ring_osc": re.compile(r"period error\s*=\s*(-?[\d.]+)\s*%"),
    "opamp": re.compile(r"gain error\s*=\s*(-?[\d.]+)\s*%"),
    "switchcap": re.compile(r"charge err=\s*(-?[\d.]+)%\s*of VDD"),
}


def read(p: Path) -> Optional[str]:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_manifest_digest(
    root: Path,
    require_manifest: bool = False,
) -> Optional[str]:
    manifest = root / "campaign_manifest.json"
    if not manifest.is_file():
        if require_manifest:
            raise ValueError(f"campaign manifest is missing: {manifest}")
        return None
    return _sha256(manifest)


def rc_of(txt: str) -> Optional[str]:
    # Two dispatchers can race onto the same job if one was launched while the
    # other already had it in flight; the tell is two completion markers in one
    # log. Such a log is a mixture of two runs and must be re-run, not parsed.
    markers = [line for line in txt.splitlines()
               if line.startswith("===V710_DONE")]
    if len(markers) > 1:
        return "RACED"
    if not markers:
        return None
    match = _RC.fullmatch(markers[0])
    return match.group(1) if match else None


def _summary_error(txt: str, tech: str) -> Optional[str]:
    """Return a caught per-tech error from a suite's summary table."""
    for match in _SUMMARY_ERROR_ROW.finditer(txt):
        if match.group(1).upper() == tech.upper():
            return match.group(2)
    return None


def is_verdict(entry: object) -> bool:
    """Whether an entry reached a scientific PASS/FAIL verdict."""
    if not isinstance(entry, dict):
        return False
    try:
        rc = int(entry.get("rc"))
    except (TypeError, ValueError):
        return False
    return 0 <= rc < 126 and rc != 124


def collect(root: Path, require_manifest: bool = False) -> Dict:
    expected_provenance = campaign_manifest_digest(root, require_manifest)
    out: Dict[str, Dict] = {}
    for log in sorted(root.glob("*/*/*/*.omp*.log")):
        tech_dir = log.parent.name
        variant = log.parent.parent.name
        tag = log.parent.parent.parent.name
        suite, _, omp = log.name[:-4].partition(".omp")
        txt = read(log)
        if txt is None:
            continue
        if expected_provenance is not None:
            observed = _PROVENANCE.findall(txt)
            if observed != [expected_provenance]:
                raise ValueError(
                    f"campaign provenance mismatch in {log}: "
                    f"expected one {expected_provenance}, got {observed}"
                )
        rc = rc_of(txt)
        if rc is None:
            continue  # still running
        g = out.setdefault(tag, {}).setdefault(variant, {})
        cell = g.setdefault(suite, {}).setdefault(tech_dir.upper(), {})
        entry: Dict = {"rc": rc}
        structured = parse_result_markers(txt)

        if structured:
            entry["results"] = structured
            statuses = {str(item.get("status", "")).upper()
                        for item in structured}
            if "ERROR" in statuses:
                entry["status"] = "ERROR"
                entry["error"] = "; ".join(
                    str(item.get("error", ""))
                    for item in structured if item.get("status") == "error"
                )
            elif "FAIL" in statuses:
                entry["status"] = "FAIL"
            elif "DIAGNOSTIC" in statuses:
                entry["status"] = "DIAGNOSTIC"
            else:
                entry["status"] = "PASS"
            explicit_metrics = [
                item.get("metrics", {}).get("metric")
                for item in structured
                if isinstance(item.get("metrics"), dict)
                and item.get("metrics", {}).get("metric") is not None
            ]
            nrmse_values = [
                item.get("metrics", {}).get("nrmse_pct")
                for item in structured
                if isinstance(item.get("metrics"), dict)
                and item.get("metrics", {}).get("nrmse_pct") is not None
            ]
            values = explicit_metrics or nrmse_values
            if values:
                entry["metric"] = max(float(value) for value in values)
        elif suite == "verify_nn_ac":
            for m in _AC_ROW.finditer(txt):
                entry[m.group(2)] = {
                    "gain0_err_db": m.group(3), "f3db_ratio": m.group(4),
                    "mag_nrmse_pct": m.group(5), "phase_inband_deg": m.group(6),
                    "status": m.group(7)}
        elif suite == "verify_circuit_opamp_ac":
            for m in _OPAMP_AC.finditer(txt):
                entry.update(dc_gain_err_db=m.group(2), gbw_ratio=m.group(3),
                             pm_err_deg=m.group(4), mag_nrmse_pct=m.group(5),
                             status=m.group(6))
        elif suite.startswith("verify_nn_multi_tech"):
            rows = [(m.group(1), float(m.group(2)), float(m.group(3)),
                     float(m.group(4)), float(m.group(5)), m.group(6))
                    for m in _DEV_ROW.finditer(txt)]
            error_labels = {m.group(1) for m in _DEV_ERROR_ROW.finditer(txt)}
            labels = {row[0] for row in rows} | error_labels
            if labels:
                entry["n"] = len(labels)
                entry["n_pass"] = sum(1 for r in rows if r[5] == "PASS")
                entry["rows"] = {label: {"status": "ERROR"}
                                 for label in error_labels}
                entry["rows"].update(
                    {row[0]: {"nrmse": row[1], "mre": row[2], "r2": row[3],
                              "max_error": row[4], "status": row[5]}
                     for row in rows}
                )
            if rows:
                entry["mean_nrmse"] = round(sum(r[1] for r in rows) / len(rows), 3)
                entry["max_nrmse"] = round(max(r[1] for r in rows), 3)
                entry["mean_mre"] = round(sum(r[2] for r in rows) / len(rows), 3)
                entry["min_r2"] = round(min(r[3] for r in rows), 5)
                entry["max_error"] = max(r[4] for r in rows)
        else:  # circuit benchmarks
            circ = suite.replace("verify_circuit_", "")
            if circ == "sram_snm":
                vals = [float(m.group(1)) for m in _SRAM_ROW.finditer(txt)]
                if vals:
                    entry["metric"] = max(vals)
            else:
                pat = _CIRC_METRIC.get(circ)
                if pat:
                    vals = pat.findall(txt)
                    if vals:
                        entry["metric"] = float(vals[-1])
        if not structured and suite.startswith("verify_circuit_"):
            error = _summary_error(txt, tech_dir)
            if error is not None:
                entry.update(status="ERROR", error=error)
        cell[f"omp{omp}"] = entry
    return out


def _verdict(cell: Dict, omp: str = "omp1") -> str:
    e = cell.get(omp)
    if not e:
        return "—"
    if not is_verdict(e):
        return "INVALID"
    if e.get("status") == "ERROR":
        return "ERROR"
    if e["rc"] == "0":
        return "PASS"
    return "FAIL"


def _strict(cell: Dict) -> str:
    vs = [_verdict(cell, f"omp{n}") for n in (1, 2, 4)]
    if any(v == "—" for v in vs):
        return "partial"
    if any(v == "INVALID" for v in vs):
        return "INVALID"
    if any(v == "ERROR" for v in vs):
        return "ERROR"
    if all(v == "PASS" for v in vs):
        return "PASS"
    if all(v == "FAIL" for v in vs):
        return "FAIL"
    return "FLIP"


def render(data: Dict) -> str:
    L: List[str] = ["# Clean re-gate — device suites, AC and strict OMP",
                    "",
                    "Every number below is measured at the current HEAD (post gds sign +",
                    "guard fix, post V7.0.x perf work, opt-in perf flags OFF), CPU-pinned,",
                    "repo ngspice, per-job isolated results dir. Verdict = suite exit code.",
                    ""]
    for tag in ("dn", "tf", "dnf", "tff"):
        if tag not in data:
            continue
        L += [f"## {FAMILY[tag]}", ""]
        for variant, g in sorted(data[tag].items()):
            L += [f"### `{tag}/{variant}`", ""]

            if "verify_nn_ac" in g:
                cells = g["verify_nn_ac"]
                npass = ntot = 0
                rows = []
                for t in TECHS:
                    c = cells.get(t, {}).get("omp1")
                    if not c:
                        rows.append(f"| {t} | — | — |")
                        continue
                    if not is_verdict(c):
                        rows.append(f"| {t} | INVALID | INVALID |")
                        continue
                    for dev in ("nmos", "pmos"):
                        d = c.get(dev)
                        if d:
                            ntot += 1
                            npass += d["status"] == "PASS"
                    n = c.get("nmos", {}); p = c.get("pmos", {})
                    rows.append(
                        f"| {t} | {n.get('status','—')} "
                        f"(gain0 {n.get('gain0_err_db','—')} dB, f3db {n.get('f3db_ratio','—')}, "
                        f"mag {n.get('mag_nrmse_pct','—')} %) | {p.get('status','—')} "
                        f"(gain0 {p.get('gain0_err_db','—')} dB, f3db {p.get('f3db_ratio','—')}, "
                        f"mag {p.get('mag_nrmse_pct','—')} %) |")
                L += [f"**Device CS-amp AC: {npass}/{ntot}**", "",
                      "| tech | NMOS | PMOS |", "|---|---|---|", *rows, ""]

            if "verify_circuit_opamp_ac" in g:
                cells = g["verify_circuit_opamp_ac"]
                rows, npass, ntot = [], 0, 0
                for t in TECHS:
                    c = cells.get(t, {}).get("omp1")
                    if not c:
                        rows.append(f"| {t} | — | | | | |")
                        continue
                    if not is_verdict(c):
                        rows.append(f"| {t} | INVALID | | | | |")
                        continue
                    ntot += 1
                    npass += c["rc"] == "0"
                    rows.append(
                        f"| {t} | {c.get('status', 'FAIL' if c['rc'] != '0' else 'PASS')} "
                        f"| {c.get('dc_gain_err_db','—')} | {c.get('gbw_ratio','—')} "
                        f"| {c.get('pm_err_deg','—')} | {c.get('mag_nrmse_pct','—')} |")
                L += [f"**Opamp open-loop AC: {npass}/{ntot}** "
                      "(gate: dc_gain_err ≤3 dB, GBW ratio ∈[0.6,1.67], PM err ≤15°; "
                      "magNRMSE reported, not gated)", "",
                      "| tech | verdict | dc_gain_err dB | GBW ratio | PM err ° | magNRMSE % |",
                      "|---|---|---|---|---|---|", *rows, ""]

            for suite, label in (("verify_nn_multi_tech_dc", "Parametric DC (Id-Vgs)"),
                                 ("verify_nn_multi_tech_tran", "Parametric transient")):
                if suite not in g:
                    continue
                cells = g[suite]
                rows, npass, ntot = [], 0, 0
                for t in TECHS:
                    c = cells.get(t, {}).get("omp1")
                    if not c:
                        rows.append(f"| {t} | — | | | | | |")
                        continue
                    if not is_verdict(c):
                        rows.append(f"| {t} | INVALID | | | | | |")
                        continue
                    if "n" not in c:
                        rows.append(f"| {t} | — | | | | | |")
                        continue
                    npass += c["n_pass"]; ntot += c["n"]
                    max_error = c["max_error"] * (1e6 if suite.endswith("_dc") else 1.0)
                    rows.append(f"| {t} | {c['n_pass']}/{c['n']} | {c['mean_nrmse']} "
                                f"| {c['max_nrmse']} | {c['mean_mre']} "
                                f"| {c['min_r2']} | {max_error:.6g} |")
                error_unit = "µA" if suite.endswith("_dc") else "mV"
                L += [f"**{label}: {npass}/{ntot} configs**", "",
                      "| tech | pass | mean NRMSE % | max NRMSE % | mean MRE % "
                      f"| min R² | max error {error_unit} |",
                      "|---|---|---|---|---|---|---|", *rows, ""]

            have_circ = [c for c in CIRCS if SUITE_BY_RESULT[c] in g]
            if have_circ:
                rows, npass, ntot = [], 0, 0
                for t in TECHS:
                    cs = []
                    for c in CIRCS:
                        cell = g.get(SUITE_BY_RESULT[c], {}).get(t, {})
                        if not cell:
                            cs.append("—"); continue
                        v = _verdict(cell)
                        m = cell.get("omp1", {}).get("metric")
                        if v in ("PASS", "FAIL"):
                            ntot += 1
                            npass += v == "PASS"
                        cs.append(f"{v}" + (f" {m:.2f}%" if m is not None else ""))
                    rows.append(f"| {t} | " + " | ".join(cs) + " |")
                L += [f"**Simple-v1 matrix (single-run OMP=1): {npass}/{ntot}**", "",
                      "| tech | " + " | ".join(CIRCS) + " |",
                      "|---|" + "---|" * len(CIRCS), *rows, ""]

                srows, spass, stot, flips = [], 0, 0, 0
                for t in TECHS:
                    cs = []
                    for c in CIRCS:
                        cell = g.get(SUITE_BY_RESULT[c], {}).get(t, {})
                        if not cell:
                            cs.append("—"); continue
                        s = _strict(cell) if c in ("opamp", "ring_osc") else _verdict(cell)
                        if s in ("PASS", "FAIL", "FLIP"):
                            stot += 1
                            spass += s == "PASS"
                            flips += s == "FLIP"
                        cs.append(s)
                    srows.append(f"| {t} | " + " | ".join(cs) + " |")
                L += [f"**Strict OMP∈{{1,2,4}} (opamp+ring swept; sram/switchcap "
                      f"deterministic): {spass}/{stot}, {flips} FLIP**", "",
                      "| tech | " + " | ".join(CIRCS) + " |",
                      "|---|" + "---|" * len(CIRCS), *srows, ""]

            diagnostic_cases = [
                case for case in SIMPLE_V2_CASES if case.campaign_suite in g
            ]
            if diagnostic_cases:
                rows: List[str] = []
                for case in diagnostic_cases:
                    cells = g[case.campaign_suite]
                    for tech in TECHS:
                        entry = cells.get(tech, {}).get("omp1")
                        if entry is None:
                            rows.append(f"| {case.case_id} | {tech} | — | — |")
                            continue
                        status = entry.get("status", "INVALID")
                        metric = entry.get("metric")
                        metric_text = "—" if metric is None else f"{metric:.3f}"
                        rows.append(
                            f"| {case.case_id} | {tech} | {status} | "
                            f"{metric_text} |"
                        )
                L += [
                    "**Simple-v2 nominal held-out topology diagnostics (not "
                    "included in the simple-v1 score)**",
                    "",
                    "| case | tech | outcome | worst NRMSE % |",
                    "|---|---|---|---|",
                    *rows,
                    "",
                ]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/v710_regate", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--require-manifest", action="store_true")
    a = ap.parse_args()
    try:
        data = collect(a.root, require_manifest=a.require_manifest)
    except ValueError as exc:
        print(f"[v710-collect] ERROR: {exc}")
        return 2
    out = a.out or (a.root / "REPORT.md")
    out.write_text(render(data))
    data_path = a.root / "data.json"
    data_path.write_text(json.dumps(data, indent=1, sort_keys=True))
    manifest_digest = campaign_manifest_digest(a.root)
    if manifest_digest is not None:
        manifest = json.loads((a.root / "campaign_manifest.json").read_text())
        (a.root / "collection_provenance.json").write_text(json.dumps({
            "campaign_manifest_sha256": manifest_digest,
            "data_sha256": _sha256(data_path),
            "source_commit": manifest["source_commit"],
        }, indent=2, sort_keys=True) + "\n")
    n = sum(len(s) for t in data.values() for v in t.values() for s in v.values())
    print(f"[v710-collect] {n} suite-cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
