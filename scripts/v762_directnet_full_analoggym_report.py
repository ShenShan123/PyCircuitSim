#!/usr/bin/env python3
"""Validate and aggregate one five-technology DirectNet-Full AnalogGym pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.complex_circuits.pycircuitsim_bench import campaign


TECHS: Tuple[str, ...] = ("tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16")
FAMILIES: Tuple[str, ...] = ("ac", "dc_source", "dc_temp", "tran")


def _read_json(path: Path) -> Dict[str, Any]:
    """Read one required JSON object or fail loudly."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON value using one stable encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _empty_counts() -> Dict[str, int]:
    return {
        "physical": 0,
        "scored": 0,
        "passed": 0,
        "quarantined": 0,
        "py_failures": 0,
        "ng_failures": 0,
        "metric_agree": 0,
        "metric_measured": 0,
        "missing_py": 0,
        "not_comparable": 0,
    }


def _add_counts(target: Dict[str, int], source: Mapping[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def _row_counts(verdict: Mapping[str, Any]) -> Dict[str, int]:
    measured = int(verdict.get("measured", 0))
    missing_py = int(verdict.get("missing_py", 0))
    not_comparable = int(verdict.get("not_comparable", 0))
    whole_quarantine = (
        measured == 0 and missing_py == 0 and not_comparable > 0
    )
    passed = (
        verdict.get("ran") is True
        and verdict.get("ng_ran") is True
        and verdict.get("engine_ok") is True
        and measured > 0
        and int(verdict.get("agree", 0)) == measured
        and missing_py == 0
    )
    return {
        "physical": 1,
        "scored": int(not whole_quarantine),
        "passed": int(passed),
        "quarantined": int(whole_quarantine),
        "py_failures": int(verdict.get("ran") is not True),
        "ng_failures": int(verdict.get("ng_ran") is not True),
        "metric_agree": int(verdict.get("agree", 0)),
        "metric_measured": measured,
        "missing_py": missing_py,
        "not_comparable": not_comparable,
    }


def _stable_checkpoint_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    model = row.get("py_model")
    if not isinstance(model, dict):
        raise ValueError("campaign row has no DirectNet-Full model provenance")
    checkpoints = model.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("campaign row has no checkpoint provenance")
    result: Dict[str, Any] = {}
    for device in ("nmos", "pmos"):
        record = checkpoints.get(device)
        if not isinstance(record, dict):
            raise ValueError(f"campaign row has no {device} checkpoint record")
        result[device] = {
            key: record.get(key)
            for key in (
                "stem", "checkpoint_sha256", "norm_sha256",
                "completion_sha256",
            )
        }
    return result


def _collect_root(root: Path) -> Dict[str, Any]:
    provenance = _read_json(root / "campaign_provenance.json")
    tech = str(provenance.get("tech", "")).lower()
    expected_provenance = {
        "families": list(FAMILIES),
        "model_level": 75,
        "checkpoint_size": "large",
        "refine": True,
    }
    mismatches = [
        key for key, value in expected_provenance.items()
        if provenance.get(key) != value
    ]
    if tech not in TECHS or mismatches:
        detail = ", ".join(mismatches) or f"unknown technology {tech!r}"
        raise ValueError(f"invalid campaign provenance in {root}: {detail}")

    code_commit = str(provenance.get("code_commit", ""))
    expected_rows: Dict[Path, Tuple[str, str]] = {}
    for family in FAMILIES:
        for category, design, deck in campaign.corpus(tech, [family]):
            path = root / f"{tech}_{category}_{design}_{Path(deck).stem}.json"
            if path in expected_rows:
                raise ValueError(f"duplicate expected AnalogGym cell: {path}")
            expected_rows[path] = (family, deck)
    actual_rows = set(root.glob(f"{tech}_*.json"))
    missing = set(expected_rows) - actual_rows
    extra = actual_rows - set(expected_rows)
    if missing or extra:
        raise ValueError(
            f"{tech} inventory mismatch: {len(missing)} missing, "
            f"{len(extra)} unexpected"
        )

    totals = _empty_counts()
    by_family = {family: _empty_counts() for family in FAMILIES}
    rows: Dict[Path, Dict[str, Any]] = {}
    checkpoints: Optional[Dict[str, Any]] = None
    osdi_hashes: set[str] = set()
    ngspice_hashes: set[str] = set()
    modelcard_hashes: set[str] = set()
    for path, (family, deck) in expected_rows.items():
        row = _read_json(path)
        category = str(row.get("category", ""))
        policy = campaign._campaign_policy(category, deck, True)
        if not campaign._model_matches_row(
            row, tech, 75, "large", code_commit, policy,
        ):
            raise ValueError(f"stale or mixed campaign row: {path}")
        verdict = row.get("verdict")
        if not isinstance(verdict, dict):
            raise ValueError(f"campaign row has no verdict: {path}")
        counts = _row_counts(verdict)
        _add_counts(totals, counts)
        _add_counts(by_family[family], counts)
        rows[path] = row

        row_checkpoints = _stable_checkpoint_record(row)
        if checkpoints is None:
            checkpoints = row_checkpoints
        elif checkpoints != row_checkpoints:
            raise ValueError(f"mixed checkpoint provenance: {path}")
        reference = row["ground_truth"]
        osdi_hashes.add(str(reference["osdi"]["sha256"]))
        ngspice_hashes.add(str(reference["ngspice"]["sha256"]))
        modelcard_hashes.add(str(reference["modelcard"]["sha256"]))

    if len(osdi_hashes) != 1 or len(ngspice_hashes) != 1:
        raise ValueError(f"mixed reference provenance in {root}")
    voltage = campaign._aggregate_voltage_error(rows)
    return {
        "tech": tech,
        "root": str(root.resolve()),
        "code_commit": code_commit,
        "counts": totals,
        "families": by_family,
        "voltage": voltage,
        "checkpoints": checkpoints,
        "ground_truth": {
            "osdi_sha256": next(iter(osdi_hashes)),
            "ngspice_sha256": next(iter(ngspice_hashes)),
            "modelcard_sha256": sorted(modelcard_hashes),
        },
    }


def build_aggregate(roots: Iterable[Path]) -> Dict[str, Any]:
    """Build one checksum-bound aggregate from exactly five campaign roots."""
    roots_list = list(roots)
    if len(roots_list) != len(TECHS):
        raise ValueError(f"expected exactly {len(TECHS)} roots")
    campaigns = [_collect_root(root) for root in roots_list]
    by_tech = {str(item["tech"]): item for item in campaigns}
    if set(by_tech) != set(TECHS) or len(by_tech) != len(campaigns):
        raise ValueError("roots must contain each required technology exactly once")
    commits = {str(item["code_commit"]) for item in campaigns}
    osdi = {str(item["ground_truth"]["osdi_sha256"]) for item in campaigns}
    ngspice = {
        str(item["ground_truth"]["ngspice_sha256"]) for item in campaigns
    }
    if len(commits) != 1 or len(osdi) != 1 or len(ngspice) != 1:
        raise ValueError("campaign roots mix code or reference provenance")

    totals = _empty_counts()
    by_family = {family: _empty_counts() for family in FAMILIES}
    for item in campaigns:
        _add_counts(totals, item["counts"])
        for family in FAMILIES:
            _add_counts(by_family[family], item["families"][family])
    if totals["physical"] != 255:
        raise ValueError(f"expected 255 physical cells, got {totals['physical']}")
    if totals["quarantined"] != 7 or totals["scored"] != 248:
        raise ValueError(
            "tracked denominator changed: expected 248 scored + 7 quarantined, "
            f"got {totals['scored']} + {totals['quarantined']}"
        )

    payload: Dict[str, Any] = {
        "schema": "directnet-full-analoggym-aggregate/v1",
        "code_commit": next(iter(commits)),
        "model": {"family": "directnet-full", "level": 75, "size": "large"},
        "execution": {
            "cpu_only": True,
            "omp_threads": 1,
            "mkl_threads": 1,
            "torch_threads": 1,
            "families": list(FAMILIES),
            "refine": True,
        },
        "ground_truth": {
            "family": "BSIM-CMG", "level": 72,
            "osdi_sha256": next(iter(osdi)),
            "ngspice_sha256": next(iter(ngspice)),
        },
        "totals": totals,
        "families": by_family,
        "technologies": {tech: by_tech[tech] for tech in TECHS},
    }
    return {**payload, "aggregate_sha256": _canonical_sha256(payload)}


def _score(value: Mapping[str, Any]) -> str:
    return f"{int(value['passed'])}/{int(value['scored'])}"


def _tech_label(tech: str) -> str:
    return tech.upper().replace("TSMC", "TSMC")


def _render_tables(aggregate: Mapping[str, Any]) -> str:
    technologies = aggregate["technologies"]
    totals = aggregate["totals"]
    families = aggregate["families"]
    lines = [
        "| tech | AC | dc_source | dc_temp | transient | total | Py failures | NG failures |",
        "|---|:--:|:--:|:--:|:--:|:--:|--:|--:|",
    ]
    family_keys = ("ac", "dc_source", "dc_temp", "tran")
    for tech in TECHS:
        item = technologies[tech]
        cells = [_score(item["families"][family]) for family in family_keys]
        lines.append(
            f"| {_tech_label(tech)} | " + " | ".join(cells)
            + f" | **{_score(item['counts'])}**"
            + f" | {item['counts']['py_failures']}"
            + f" | {item['counts']['ng_failures']} |"
        )
    lines.append(
        "| **all** | "
        + " | ".join(f"**{_score(families[family])}**" for family in family_keys)
        + f" | **{_score(totals)}**"
        + f" | **{totals['py_failures']}** | **{totals['ng_failures']}** |"
    )
    lines += [
        "",
        "| tech | comparable metric cells agreeing | missing Py values | quarantined metric cells |",
        "|---|---:|---:|---:|",
    ]
    for tech in TECHS:
        counts = technologies[tech]["counts"]
        lines.append(
            f"| {_tech_label(tech)} | {counts['metric_agree']}/"
            f"{counts['metric_measured']} | {counts['missing_py']} | "
            f"{counts['not_comparable']} |"
        )
    lines.append(
        f"| **all** | **{totals['metric_agree']}/{totals['metric_measured']}**"
        f" | **{totals['missing_py']}** | **{totals['not_comparable']}** |"
    )
    return "\n".join(lines)


def _render_voltage_table(aggregate: Mapping[str, Any]) -> str:
    lines = [
        "| technology | rows with state data | samples | MRE | R² | NRMSE | max abs error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tech in TECHS:
        voltage = aggregate["technologies"][tech]["voltage"]
        if voltage is None:
            lines.append(f"| {_tech_label(tech)} | 0 | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {_tech_label(tech)} | {int(voltage['decks'])} | "
            f"{int(voltage['n'])} | {100.0 * voltage['mre']:.4g}% | "
            f"{voltage['r2']:.6g} | {100.0 * voltage['nrmse']:.4g}% | "
            f"{voltage['max_error']:.6g} V |"
        )
    return "\n".join(lines)


def render_accuracy_section(aggregate: Mapping[str, Any]) -> str:
    """Render the generated AnalogGym section for the accuracy report."""
    totals = aggregate["totals"]
    return "\n".join([
        "## AnalogGym complex-circuit accuracy",
        "",
        "The production-sized `large` checkpoint pair for each technology was "
        "evaluated over the tracked 255-row basket in "
        "`examples/complex_circuits/`. NGSPICE 45.2 used LEVEL=72 as ground "
        "truth; DirectNet-Full used LEVEL=75 on CPU with one OpenMP, MKL, and "
        "Torch thread. Seven whole-deck invalid examples remain quarantined, "
        f"leaving **{totals['scored']} scored decks. DirectNet-Full passes "
        f"{totals['passed']}/{totals['scored']}.**",
        "",
        _render_tables(aggregate),
        "",
        "Voltage-state metrics use only rows that produced comparable "
        "operating-point or DC-sweep samples; incomplete rows remain failures "
        "in the deck denominator.",
        "",
        _render_voltage_table(aggregate),
        "",
        f"Aggregate SHA-256 `{aggregate['aggregate_sha256']}` pins simulation "
        f"commit `{aggregate['code_commit']}`, the five checkpoint pairs, "
        "normalization and completion hashes, modelcards, the LEVEL=72 OSDI "
        "binary, NGSPICE, and per-deck fidelity policy. Detailed circuit "
        "evidence and quarantine ownership are in "
        "[`examples/complex_circuits/RESULTS_TSMC.md`](../../examples/complex_circuits/RESULTS_TSMC.md). "
        "The upstream AnalogGym source tree is absent, so this is a complete "
        "tracked-deck rerun, not a refreshed source-topology audit.",
    ])


def render_results_section(aggregate: Mapping[str, Any]) -> str:
    """Render the generated DirectNet-Full section for RESULTS_TSMC.md."""
    totals = aggregate["totals"]
    return "\n".join([
        "## DirectNet-Full LEVEL=75 production qualification (2026-08-28 clean rerun)",
        "",
        "The regenerated full-terminal `large` checkpoint pair for each "
        "technology was evaluated on all 255 tracked deck cells, with NGSPICE "
        "LEVEL=72 as ground truth. Scored inference was CPU-only with one "
        "OpenMP, MKL, and Torch thread. Seven whole-deck invalid examples "
        f"remain quarantined, leaving the established `/{totals['scored']}` "
        "denominator.",
        "",
        f"**Verdict: do not promote LEVEL=75. DirectNet-Full passes "
        f"{totals['passed']}/{totals['scored']} decks.** Production promotion "
        "also remains blocked on a refreshed source-tree campaign and the "
        "separately deferred seed/performance protocols.",
        "",
        _render_tables(aggregate),
        "",
        "Per-technology voltage-state metrics are owned by "
        "[`docs/accuracy/DirectNet-L75-clean.md`](../../docs/accuracy/DirectNet-L75-clean.md). "
        "Partial numeric data never convert an incomplete deck into a pass.",
        "",
        f"Provenance: aggregate `{aggregate['aggregate_sha256']}` at simulation "
        f"commit `{aggregate['code_commit']}`. Every row pins its checkpoint, "
        "normalization, completion marker, modelcard, OSDI, NGSPICE executable, "
        "stride, and refinement policy. The upstream AnalogGym source tree "
        "remains absent, so this evaluates the tracked generated decks and is "
        "not a refreshed source-topology audit.",
    ])


def _replace_section(
    path: Path,
    start: str,
    end: str,
    replacement: str,
    *,
    check: bool,
) -> None:
    text = path.read_text()
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        raise ValueError(f"generated section boundary missing in {path}")
    rendered = text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]
    if check:
        if rendered != text:
            raise ValueError(f"generated report section is stale: {path}")
    elif rendered != text:
        path.write_text(rendered)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accuracy-report", action="append", type=Path)
    parser.add_argument("--results-report", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        aggregate = build_aggregate(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text() != rendered:
            parser.error(f"aggregate is missing or stale: {args.out}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    try:
        for report in args.accuracy_report or []:
            _replace_section(
                report,
                "## AnalogGym complex-circuit accuracy",
                "## Qualification verdict",
                render_accuracy_section(aggregate),
                check=args.check,
            )
        if args.results_report is not None:
            _replace_section(
                args.results_report,
                "## DirectNet-Full LEVEL=75 production qualification",
                "## DirectNet LEVEL=73 on the curated basket",
                render_results_section(aggregate),
                check=args.check,
            )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"[analoggym-aggregate] {aggregate['totals']['passed']}/"
        f"{aggregate['totals']['scored']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
