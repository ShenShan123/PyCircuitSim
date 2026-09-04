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
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common.gate_result import (  # noqa: E402
    GateResult,
    parse_result_markers,
    result_exit_code,
)
from tests.common.simple_circuit_catalog import (  # noqa: E402
    SIMPLE_V1,
    SIMPLE_V2,
    cases,
)

TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
SIMPLE_V1_CASES = cases(score_version=SIMPLE_V1)
SIMPLE_V2_CASES = cases(score_version=SIMPLE_V2)
SIMPLE_V2_BY_SUITE = {case.campaign_suite: case for case in SIMPLE_V2_CASES}
CATALOG_BY_SUITE = {case.campaign_suite: case for case in (
    *SIMPLE_V1_CASES, *SIMPLE_V2_CASES,
)}
CIRCS = [case.result_key for case in SIMPLE_V1_CASES]
SUITE_BY_RESULT = {
    case.result_key: case.campaign_suite
    for case in SIMPLE_V1_CASES
}
FAMILY = {
    "dnf": "DirectNet-Full (L75)",
    "tff": "BSIM-AR-Full (L76)",
}
MODEL_BY_TAG = {
    "dnf": (75, "DirectNet-Full"),
    "tff": (76, "BSIM-AR-Full"),
}
STRUCTURED_SUITES = {
    *CATALOG_BY_SUITE,
    "verify_device_integrity",
    "verify_terminal_integrity",
    "verify_nn_subckt",
    "verify_nn_ac",
    "verify_circuit_opamp_ac",
    "verify_nn_multi_tech_dc",
    "verify_nn_multi_tech_tran",
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


def log_provenance_error(text: str, expected: Optional[str]) -> str:
    """Return any missing, duplicated, or mismatched campaign log marker."""
    if expected is None:
        return ""
    observed = _PROVENANCE.findall(text)
    if observed != [expected]:
        return f"expected one {expected}, got {observed}"
    return ""


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
    if entry.get("result_complete") is False:
        return False
    try:
        rc = int(entry.get("rc"))
    except (TypeError, ValueError):
        return False
    return rc in {0, 1}


def _structured_schema_error(
    result: Dict,
    *,
    expected_provenance: Optional[str],
) -> str:
    """Validate one marker as complete, provenance-bearing evidence."""
    required = {
        "case_id", "tech", "corner", "analysis", "role", "status",
        "metrics", "domain", "error", "reference_converged",
        "candidate_converged", "control_converged", "partial",
        "execution_state", "error_kind", "model_family", "model_level",
        "checkpoint_pins", "campaign_manifest_sha256", "thread_settings",
    }
    missing = sorted(required - set(result))
    if missing:
        return f"structured row is missing fields: {missing}"
    try:
        row = GateResult(**result)
    except (TypeError, ValueError) as exc:
        return f"invalid GateResult row: {exc}"
    expected_families = {
        75: "DirectNet-Full",
        76: "BSIM-AR-Full",
    }
    if expected_families.get(row.model_level) != row.model_family:
        return (
            "structured row provenance lacks a valid model family/level: "
            f"{row.model_family!r}/{row.model_level!r}"
        )
    if not isinstance(row.checkpoint_pins, dict):
        return "structured row provenance checkpoint_pins must be an object"
    required_threads = {"omp", "mkl", "torch"}
    if not isinstance(row.thread_settings, dict) or \
            set(row.thread_settings) != required_threads:
        return (
            "structured row provenance requires omp/mkl/torch thread settings"
        )
    if expected_provenance is not None \
            and row.campaign_manifest_sha256 != expected_provenance:
        return (
            "structured row campaign provenance mismatch: "
            f"expected {expected_provenance}, got "
            f"{row.campaign_manifest_sha256 or '<empty>'}"
        )
    if row.status == "error" and not row.error:
        return "structured error row has no error message"
    if not isinstance(row.metrics, dict) or not isinstance(row.domain, dict):
        return "structured metrics and domain must be objects"
    if row.status != "error":
        if row.analysis == "derived":
            derived_errors = {
                name: value for name, value in row.domain.items()
                if name.endswith("_db_error")
            }
            if not derived_errors or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in derived_errors.values()
            ):
                return "structured derived metric is missing or non-finite"
        elif not row.metrics:
            return "structured characterized row has no numeric evidence"
        for name, value in row.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                return f"structured metric {name!r} is not finite numeric evidence"
    return ""


def structured_contract_error(
    suite: str,
    tech: str,
    results: List[Dict],
    *,
    expected_provenance: Optional[str] = None,
    completion_rc: Optional[str] = None,
    campaign_tag: Optional[str] = None,
    campaign_variant: Optional[str] = None,
    expected_omp: Optional[int] = None,
) -> str:
    """Return why a structured catalog result set is incomplete."""
    schema_errors = [
        error
        for result in results
        for error in (
            _structured_schema_error(
                result,
                expected_provenance=expected_provenance,
            ),
        )
        if error
    ]
    if schema_errors:
        return "; ".join(schema_errors)
    if campaign_tag is not None and campaign_tag in MODEL_BY_TAG:
        expected_model = MODEL_BY_TAG[campaign_tag]
        observed_models = {
            (int(result["model_level"]), str(result["model_family"]))
            for result in results
        }
        if observed_models != {expected_model}:
            return (
                f"structured model provenance disagrees with campaign tag "
                f"{campaign_tag!r}: expected {expected_model}, "
                f"got {sorted(observed_models)}"
            )
        if campaign_variant is not None:
            expected_pins = {
                polarity: f"{tech.lower()}_{campaign_tag}_"
                f"{campaign_variant}_{polarity}"
                for polarity in ("nmos", "pmos")
            }

            def _pin_stem(raw: object) -> str:
                name = Path(str(raw)).name
                return (
                    name[:-len("_best.pt")]
                    if name.endswith("_best.pt") else name
                )

            for result in results:
                pins = result["checkpoint_pins"]
                observed_pins = {
                    polarity: _pin_stem(pins.get(polarity, ""))
                    for polarity in ("nmos", "pmos")
                }
                if observed_pins != expected_pins:
                    return (
                        "structured checkpoint provenance disagrees with "
                        f"campaign cell: expected {expected_pins}, "
                        f"got {observed_pins}"
                    )
    if expected_omp is not None:
        expected_threads = {
            "omp": expected_omp,
            "mkl": expected_omp,
            "torch": expected_omp,
        }
        wrong_threads = [
            result["thread_settings"] for result in results
            if result["thread_settings"] != expected_threads
        ]
        if wrong_threads:
            return (
                f"structured thread settings disagree with omp{expected_omp}: "
                f"{wrong_threads}"
            )
    if completion_rc is not None:
        try:
            observed_rc = int(completion_rc)
        except (TypeError, ValueError):
            return f"invalid structured completion code: {completion_rc!r}"
        expected_rc = result_exit_code([GateResult(**result) for result in results])
        if observed_rc != expected_rc:
            return (
                f"completion code {observed_rc} disagrees with structured "
                f"rows (expected {expected_rc})"
            )
    case = CATALOG_BY_SUITE.get(suite)
    if case is None:
        expected_pairs: List[tuple[str, str]] = []
        expected_corners: Dict[str, str] = {}
        if suite == "verify_device_integrity":
            from tests.common.circuit_benchmarks import BENCH
            from tests.common.device_integrity import build_sweeps

            expected_pairs = [
                (f"device_{spec.suite}", f"{device}_{spec.label}")
                for device in ("nmos", "pmos")
                for spec in build_sweeps(BENCH[tech.upper()], device)
            ]
            expected_corners = {
                analysis: "nominal" for _case_id, analysis in expected_pairs
            }
        elif suite == "verify_terminal_integrity":
            from tests.common.circuit_benchmarks import BENCH
            from tests.common.terminal_integrity import (
                terminal_biases, terminal_sweeps,
            )

            expected_pairs = [
                ("terminal_currents", f"{device}_{sweep.name}")
                for device in ("nmos", "pmos")
                for sweep in terminal_sweeps(BENCH[tech.upper()], device)
            ] + [
                ("terminal_capacitance", f"{device}_{bias.name}")
                for device in ("nmos", "pmos")
                for bias in terminal_biases(BENCH[tech.upper()], device)
            ]
            expected_corners = {
                analysis: "nominal" for _case_id, analysis in expected_pairs
            }
        elif suite == "verify_nn_subckt":
            from tests.common.subcircuit_catalog import SUBCKT_ANALYSES

            expected_pairs = [
                ("nn_subckt", analysis.name) for analysis in SUBCKT_ANALYSES
            ]
            expected_corners = {
                analysis.name: "nominal" for analysis in SUBCKT_ANALYSES
            }
        elif suite == "verify_nn_ac":
            expected_pairs = [
                ("nn_ac", device) for device in ("nmos", "pmos")
            ]
            expected_corners = {
                device: "nominal" for _case_id, device in expected_pairs
            }
        elif suite == "verify_circuit_opamp_ac":
            expected_pairs = [("opamp_ac", "open_loop")]
            expected_corners = {"open_loop": "nominal"}
        elif suite == "verify_nn_multi_tech_dc":
            from tests.common.nn_sweep import (
                build_dc_parametric,
                make_dc_baseline,
            )

            expected_pairs = [
                ("nn_parametric_dc", config.label)
                for device in ("nmos", "pmos")
                for config in (
                    make_dc_baseline(tech.upper(), device),
                    *build_dc_parametric(tech.upper(), device),
                )
            ]
            expected_corners = {
                config.label: config.sweep_type
                for device in ("nmos", "pmos")
                for config in (
                    make_dc_baseline(tech.upper(), device),
                    *build_dc_parametric(tech.upper(), device),
                )
            }
        elif suite == "verify_nn_multi_tech_tran":
            from tests.common.nn_sweep import (
                build_inv_parametric,
                make_inv_baseline,
            )

            expected_pairs = [
                ("nn_parametric_inverter", config.label)
                for analysis in ("vtc", "tran")
                for config in (
                    make_inv_baseline(tech.upper(), analysis),
                    *build_inv_parametric(tech.upper(), analysis),
                )
            ]
            expected_corners = {
                config.label: config.sweep_type
                for analysis in ("vtc", "tran")
                for config in (
                    make_inv_baseline(tech.upper(), analysis),
                    *build_inv_parametric(tech.upper(), analysis),
                )
            }
        if not expected_pairs:
            return ""
        observed_pairs = [
            (str(result.get("case_id", "")), str(result.get("analysis", "")))
            for result in results
        ]
        if sorted(observed_pairs) != sorted(expected_pairs):
            missing = sorted(set(expected_pairs) - set(observed_pairs))
            unexpected = sorted(set(observed_pairs) - set(expected_pairs))
            return (
                f"suite result markers are incomplete; missing result markers="
                f"{missing}, unexpected={unexpected}, "
                f"expected_count={len(expected_pairs)}, "
                f"observed_count={len(observed_pairs)}"
            )
        wrong_tech = [result.get("tech") for result in results
                      if str(result.get("tech", "")).upper() != tech.upper()]
        if wrong_tech:
            return f"suite result marker technology mismatch: {wrong_tech}"
        wrong_corners = [
            (
                result.get("analysis"),
                result.get("corner"),
                expected_corners.get(str(result.get("analysis", ""))),
            )
            for result in results
            if result.get("corner")
            != expected_corners.get(str(result.get("analysis", "")))
        ]
        if wrong_corners:
            return f"suite result marker corner mismatch: {wrong_corners}"
        expected_role = (
            "qualification"
            if suite in {
                "verify_nn_ac",
                "verify_circuit_opamp_ac",
                "verify_nn_multi_tech_dc",
                "verify_nn_multi_tech_tran",
            }
            else "diagnostic"
        )
        wrong_role = [result.get("role") for result in results
                      if result.get("role") != expected_role]
        if wrong_role:
            return f"suite result marker role mismatch: {wrong_role}"
        if suite == "verify_nn_ac":
            required_metrics = {
                "gain0_db_error", "bandwidth_ratio", "nrmse_pct",
                "phase_error_deg", "mag_nrmse_pct", "mre_pct", "r2",
                "max_err",
            }
        elif suite == "verify_circuit_opamp_ac":
            required_metrics = {
                "dc_gain_db_error", "gbw_ratio",
                "phase_margin_error_deg", "mag_nrmse_pct", "mre_pct", "r2",
                "nrmse_pct", "max_err",
            }
        else:
            required_metrics = {"mre_pct", "r2", "nrmse_pct", "max_err"}
        for result in results:
            if result.get("status") == "error":
                continue
            metrics = result.get("metrics", {})
            missing_metrics = sorted(required_metrics - set(metrics))
            if missing_metrics:
                return (
                    "suite result row is missing required metrics: "
                    f"{missing_metrics}"
                )
        return ""
    expected = [analysis.name for analysis in case.analyses]
    if case.derived_metrics:
        expected.append("derived")
    observed = [str(result.get("analysis", "")) for result in results]
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    duplicates = sorted({name for name in observed if observed.count(name) > 1})
    if missing or unexpected or duplicates or len(observed) != len(expected):
        return (
            f"catalog result markers are incomplete; missing result markers="
            f"{missing}, unexpected={unexpected}, duplicates={duplicates}, "
            f"expected_count={len(expected)}, observed_count={len(observed)}"
        )
    wrong_case = [result.get("case_id") for result in results
                  if result.get("case_id") != case.case_id]
    wrong_tech = [result.get("tech") for result in results
                  if str(result.get("tech", "")).upper() != tech.upper()]
    corners = {str(result.get("corner", "")) for result in results}
    if wrong_case or wrong_tech or len(corners) != 1 or "" in corners:
        return (
            f"catalog result marker identity mismatch: cases={wrong_case}, "
            f"techs={wrong_tech}, corners={sorted(corners)}"
        )
    wrong_role = [result.get("role") for result in results
                  if result.get("role") != case.role]
    if wrong_role:
        return f"catalog result marker role mismatch: {wrong_role}"
    if case.score_version == SIMPLE_V2:
        from tests.common.simple_circuit_harness import validate_analysis_metrics

        specs = {analysis.name: analysis for analysis in case.analyses}
        for result in results:
            name = str(result["analysis"])
            if result["status"] == "error":
                continue
            if name == "derived":
                required_derived = {
                    key
                    for metric in case.derived_metrics
                    for key in (
                        f"{metric}_db_test",
                        f"{metric}_db_ref",
                        f"{metric}_db_error",
                    )
                }
                domain = result["domain"]
                invalid_derived = sorted(
                    key for key in required_derived
                    if key not in domain
                    or isinstance(domain[key], bool)
                    or not isinstance(domain[key], (int, float))
                    or not math.isfinite(float(domain[key]))
                )
                if invalid_derived:
                    return (
                        f"{case.case_id} derived metrics are invalid: "
                        f"{invalid_derived}"
                    )
                continue
            try:
                validate_analysis_metrics(
                    specs[name], result["metrics"], result["domain"],
                )
            except ValueError as exc:
                return f"{case.case_id}/{name} required metrics invalid: {exc}"
        if not any(result["status"] == "error" for result in results):
            produced = {
                key
                for result in results
                for payload in (result["metrics"], result["domain"])
                for key in payload
            }
            missing_metrics = sorted(set(case.required_metrics) - produced)
            if missing_metrics:
                return (
                    f"{case.case_id} rows are missing required metrics: "
                    f"{missing_metrics}"
                )
    elif any(
        result["status"] != "error" and "metric" not in result["metrics"]
        for result in results
    ):
        return f"{case.case_id} qualification row has no gate metric"
    return ""


def _add_structured_nn_summary(
    entry: Dict,
    suite: str,
    results: List[Dict],
) -> None:
    """Retain the historical report view while structured rows own validity."""
    if suite == "verify_nn_ac":
        for result in results:
            device = str(result["analysis"])
            metrics = result.get("metrics", {})
            entry[device] = {
                "gain0_err_db": metrics.get("gain0_db_error", "—"),
                "f3db_ratio": metrics.get("bandwidth_ratio", "—"),
                "mag_nrmse_pct": metrics.get("mag_nrmse_pct", "—"),
                "phase_inband_deg": metrics.get("phase_error_deg", "—"),
                "status": str(result["status"]).upper(),
            }
        return
    if suite == "verify_circuit_opamp_ac":
        result = results[0]
        metrics = result.get("metrics", {})
        entry.update(
            dc_gain_err_db=metrics.get("dc_gain_db_error", "—"),
            gbw_ratio=metrics.get("gbw_ratio", "—"),
            pm_err_deg=metrics.get("phase_margin_error_deg", "—"),
            mag_nrmse_pct=metrics.get("mag_nrmse_pct", "—"),
            status=str(result["status"]).upper(),
        )
        return
    if not suite.startswith("verify_nn_multi_tech"):
        return
    entry["n"] = len(results)
    entry["n_pass"] = sum(result["status"] == "pass" for result in results)
    entry["rows"] = {}
    numeric: List[Dict] = []
    for result in results:
        label = str(result["analysis"])
        metrics = result.get("metrics", {})
        status = str(result["status"]).upper()
        row = {"status": status}
        if status != "ERROR":
            row.update({
                "nrmse": float(metrics["nrmse_pct"]),
                "mre": float(metrics["mre_pct"]),
                "r2": float(metrics["r2"]),
                "max_error": float(metrics["max_err"]),
            })
            numeric.append(row)
        entry["rows"][label] = row
    if numeric:
        entry["mean_nrmse"] = round(
            sum(row["nrmse"] for row in numeric) / len(numeric), 3,
        )
        entry["max_nrmse"] = round(
            max(row["nrmse"] for row in numeric), 3,
        )
        entry["mean_mre"] = round(
            sum(row["mre"] for row in numeric) / len(numeric), 3,
        )
        entry["min_r2"] = round(min(row["r2"] for row in numeric), 5)
        entry["max_error"] = max(row["max_error"] for row in numeric)


def _add_legacy_nn_summary(entry: Dict, suite: str, text: str) -> None:
    """Parse old logs for display only; callers keep them invalid evidence."""
    if suite == "verify_nn_ac":
        for match in _AC_ROW.finditer(text):
            entry[match.group(2)] = {
                "gain0_err_db": match.group(3),
                "f3db_ratio": match.group(4),
                "mag_nrmse_pct": match.group(5),
                "phase_inband_deg": match.group(6),
                "status": match.group(7),
            }
        return
    if suite == "verify_circuit_opamp_ac":
        for match in _OPAMP_AC.finditer(text):
            entry.update(
                dc_gain_err_db=match.group(2),
                gbw_ratio=match.group(3),
                pm_err_deg=match.group(4),
                mag_nrmse_pct=match.group(5),
                status=match.group(6),
            )
        return
    if not suite.startswith("verify_nn_multi_tech"):
        return
    rows = [
        (
            match.group(1), float(match.group(2)), float(match.group(3)),
            float(match.group(4)), float(match.group(5)), match.group(6),
        )
        for match in _DEV_ROW.finditer(text)
    ]
    error_labels = {match.group(1) for match in _DEV_ERROR_ROW.finditer(text)}
    labels = {row[0] for row in rows} | error_labels
    if labels:
        entry["n"] = len(labels)
        entry["n_pass"] = sum(row[5] == "PASS" for row in rows)
        entry["rows"] = {
            label: {"status": "ERROR"} for label in error_labels
        }
        entry["rows"].update({
            row[0]: {
                "nrmse": row[1], "mre": row[2], "r2": row[3],
                "max_error": row[4], "status": row[5],
            }
            for row in rows
        })
    if rows:
        entry["mean_nrmse"] = round(
            sum(row[1] for row in rows) / len(rows), 3,
        )
        entry["max_nrmse"] = round(max(row[1] for row in rows), 3)
        entry["mean_mre"] = round(
            sum(row[2] for row in rows) / len(rows), 3,
        )
        entry["min_r2"] = round(min(row[3] for row in rows), 5)
        entry["max_error"] = max(row[4] for row in rows)


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
        provenance_error = log_provenance_error(txt, expected_provenance)
        if provenance_error:
            raise ValueError(
                f"campaign provenance mismatch in {log}: {provenance_error}"
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
            contract_error = structured_contract_error(
                suite,
                tech_dir,
                structured,
                expected_provenance=expected_provenance,
                completion_rc=rc,
                campaign_tag=tag,
                campaign_variant=variant,
                expected_omp=int(omp) if omp.isdigit() else None,
            )
            entry["result_complete"] = not bool(contract_error)
            if contract_error:
                entry.update(status="ERROR", error=contract_error)
            statuses = {str(item.get("status", "")).upper()
                        for item in structured}
            if contract_error:
                pass
            elif "ERROR" in statuses:
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
            if not contract_error:
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
                _add_structured_nn_summary(entry, suite, structured)
            case = SIMPLE_V2_BY_SUITE.get(suite)
            if case is not None and not contract_error:
                by_analysis = {
                    str(item.get("analysis", "")): item for item in structured
                }
                headlines: Dict[str, Dict[str, object]] = {}
                for analysis in case.analyses:
                    item = by_analysis[analysis.name]
                    payload = {
                        **item.get("metrics", {}),
                        **item.get("domain", {}),
                    }
                    headlines[analysis.name] = {
                        "metric": analysis.headline_metric,
                        "value": payload.get(analysis.headline_metric),
                    }
                entry["headline_metrics"] = headlines
        elif suite in STRUCTURED_SUITES:
            entry.update(
                result_complete=False,
                status="ERROR",
                error="known structured suite emitted no result markers",
            )
            _add_legacy_nn_summary(entry, suite, txt)
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
    for tag in ("dnf", "tff"):
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
                    if "max_error" not in c:
                        rows.append(
                            f"| {t} | {c['n_pass']}/{c['n']} | — | — | — | — | — |"
                        )
                        continue
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
                        headlines = entry.get("headline_metrics", {})
                        metric_text = "; ".join(
                            f"{name}:{item['metric']}={item['value']:.4g}"
                            for name, item in headlines.items()
                            if item.get("value") is not None
                        ) or "—"
                        rows.append(
                            f"| {case.case_id} | {tech} | {status} | "
                            f"{metric_text} |"
                        )
                L += [
                    "**Simple-v2 nominal held-out topology diagnostics (not "
                    "included in the simple-v1 score)**",
                    "",
                    "| case | tech | outcome | headline metrics |",
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
