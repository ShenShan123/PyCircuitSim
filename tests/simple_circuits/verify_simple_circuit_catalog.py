#!/usr/bin/env python3
"""Pure contract checks for the versioned simple-circuit catalog."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import (  # noqa: E402
    EXAMPLES_DIR, SIMPLE_DECKS, SUBCIRCUIT_DECKS,
    deck_tokens, render_deck_text,
)
from tests.common.gate_result import GateResult, parse_result_markers  # noqa: E402
from tests.common.simple_circuit_catalog import (  # noqa: E402
    CASES, DIAGNOSTIC, QUALIFICATION, SIMPLE_V1, SIMPLE_V2, cases,
)


def main() -> int:
    failures: list[str] = []

    raw_decks = sorted(
        path.relative_to(EXAMPLES_DIR)
        for path in EXAMPLES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".cir", ".sp", ".spice"}
    )
    if raw_decks:
        failures.append(
            "examples contains materialized decks instead of .spice.tmpl files: "
            + ", ".join(map(str, raw_decks))
        )

    artifact_suffixes = {".cir", ".sp", ".raw", ".csv", ".json", ".log", ".png"}
    tests_root = PROJECT_ROOT / "tests"
    test_artifacts = sorted(
        path.relative_to(tests_root)
        for path in tests_root.rglob("*")
        if path.is_file() and path.suffix.lower() in artifact_suffixes
    )
    result_directories = sorted(
        path.relative_to(tests_root)
        for path in tests_root.rglob("*")
        if path.is_dir() and "results" in path.name.lower()
    )
    if test_artifacts or result_directories:
        offenders = [*map(str, result_directories), *map(str, test_artifacts)]
        failures.append(
            "tests contains persistent simulation artifacts; move them to results/: "
            + ", ".join(offenders)
        )

    v1 = cases(score_version=SIMPLE_V1)
    v2 = cases(score_version=SIMPLE_V2)
    if len(v1) != 4 or any(case.role != QUALIFICATION for case in v1):
        failures.append("simple-v1 must remain exactly four qualification cases")
    if len(v2) < 10 or any(case.role != DIAGNOSTIC for case in v2):
        failures.append("simple-v2 must contain the held-out diagnostic ladder")
    if any(case.training_use != "held_out" for case in v2):
        failures.append("every simple-v2 case must remain held out from training")
    if any(not case.gate_metric or not case.gate_condition for case in v1):
        failures.append("simple-v1 qualification metadata must declare its gate")
    analysis_kinds = {analysis.kind for case in v2 for analysis in case.analyses}
    if analysis_kinds != {"dc", "tran", "ac"}:
        failures.append(
            f"simple-v2 must exercise DC/transient/AC; got {analysis_kinds}")

    seen: set[str] = set()
    catalog_paths: set[Path] = set()
    for case in CASES:
        if case.case_id in seen:
            failures.append(f"duplicate case ID: {case.case_id}")
        seen.add(case.case_id)
        relative = case.template
        path = SIMPLE_DECKS / relative
        catalog_paths.add(path)
        if not path.is_file():
            failures.append(f"{case.case_id}: missing template {relative}")
            continue
        text = path.read_text()
        tokens = deck_tokens(text)
        for token in ("ANALYSIS", "MODEL_SETUP"):
            if token not in tokens:
                failures.append(f"{relative}: must declare <{token}>")
        if case.score_version == SIMPLE_V2:
            expected_body_tokens = {
                "nmos": ("BODY_N", "BODY_N_NODE"),
                "pmos": ("BODY_P", "BODY_P_NODE"),
            }
            for device in case.device_kinds:
                alternatives = expected_body_tokens[device]
                if not any(token in tokens for token in alternatives):
                    failures.append(
                        f"{relative}: {case.case_id} {device} body-bias "
                        f"corner needs one of {alternatives}"
                    )

    standalone: dict[Path, dict[str, str]] = {
        SIMPLE_DECKS / "inverter.spice.tmpl": {
            "MODEL_SETUP": (
                ".model n NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)\n"
                ".model p PMOS (LEVEL=73 TECH=tsmc5 VT=lvt)"
            ),
            "TEMP": "27", "VDD": "0.65", "INPUT_SPEC": "0",
            "N_PREFIX": "M", "P_PREFIX": "M",
            "N_DEVICE": "n L=16n NFIN=2",
            "P_DEVICE": "p L=20n NFIN=3",
            "OUTPUT_LOAD": "Cload out 0 5f", "INITIAL_CONDITION": "",
            "ANALYSIS": ".op",
        },
        SIMPLE_DECKS / "rc_lowpass.spice.tmpl": {
            "TEMP": "27", "INPUT_DC": "0", "INPUT_AC": "1",
            "INPUT_PHASE": "0", "RESISTANCE": "1k",
            "CAPACITANCE": "1p", "ANALYSIS": ".ac dec 1 1 10",
        },
        SIMPLE_DECKS / "common_source.spice.tmpl": {
            "MODEL_SETUP": ".model n NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)",
            "TEMP": "27", "VDD": "0.65", "INPUT_SPEC": "DC=0.3 AC=1 0",
            "LOAD_NETWORK": "Rd vdd out 50k", "BULK_NETWORK": "",
            "DEVICE_PREFIX": "M", "SOURCE_NODE": "0", "BULK_NODE": "0",
            "DEVICE": "n L=16n NFIN=2", "OUTPUT_LOAD": "",
            "ANALYSIS": ".ac dec 1 1 10",
        },
        EXAMPLES_DIR / "single_devices" / "mosfet.spice.tmpl": {
            "MODEL_SETUP": ".model n NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)",
            "TEMP": "27", "DRAIN_BIAS": "Vd d 0 0.5",
            "GATE_BIAS": "Vg g 0 0.3", "SOURCE_BIAS": "Vs s 0 0",
            "BULK_BIAS": "Vb b 0 0", "DEVICE_NAME": "Mdut",
            "DRAIN_NODE": "d", "GATE_NODE": "g", "SOURCE_NODE": "s",
            "BULK_NODE": "b", "DEVICE": "n L=16n NFIN=2",
            "EXTRA_DEVICES": "", "LOAD": "",
            "ANALYSIS": ".op",
        },
        SUBCIRCUIT_DECKS / "rc_ladder_flat.spice.tmpl": {
            "TEMP": "27", "INPUT_SPEC": "PULSE 0 1 1n 1p 1p 5n 12n",
            "R1": "1k", "C1": "1p", "R2": "2k", "C2": "2p",
            "ANALYSIS": ".tran 50p 10n",
        },
        SUBCIRCUIT_DECKS / "rc_ladder_hierarchical.spice.tmpl": {
            "TEMP": "27", "INPUT_SPEC": "PULSE 0 1 1n 1p 1p 5n 12n",
            "R1": "1k", "R_DEFAULT": "999", "C1": "1p", "C2": "2p",
            "ANALYSIS": ".tran 50p 10n",
        },
        SUBCIRCUIT_DECKS / "rc_lowpass_hierarchical.spice.tmpl": {
            "TEMP": "27", "INPUT_DC": "0", "INPUT_AC": "1",
            "INPUT_PHASE": "0", "RESISTANCE": "1k",
            "CAPACITANCE": "1p", "ANALYSIS": ".ac dec 1 1 10",
        },
        SUBCIRCUIT_DECKS / "resistor_tree_flat.spice.tmpl": {
            "TEMP": "27", "VTOP": "2", "R1": "3k", "R2": "6k",
            "CMID": "2p", "RLOAD": "1k", "ANALYSIS": ".op",
        },
        SUBCIRCUIT_DECKS / "resistor_tree_hierarchical.spice.tmpl": {
            "TEMP": "27", "VTOP": "2", "GAIN": "3", "RBASE": "1k",
            "RBASE_DOUBLE": "2k", "CMID": "2p", "RLOAD": "1k",
            "ANALYSIS": ".op",
        },
        SUBCIRCUIT_DECKS / "ic_hierarchical.spice.tmpl": {
            "TEMP": "27", "VIN": "0", "VIC": "0.75", "R1": "1e6",
            "R2": "1e6", "CHOLD": "1p", "MID_IC": "0.2",
            "ANALYSIS": ".tran 1n 20n uic",
        },
        SUBCIRCUIT_DECKS / "inverter_hierarchical.spice.tmpl": {
            "MODEL_SETUP": (
                ".model n NMOS (LEVEL=72)\n.model p PMOS (LEVEL=72)"
            ),
            "TEMP": "27", "VDD": "0.7",
            "INPUT_SPEC": "PULSE 0 0.7 0.5n 0.1n 0.1n 1n 2.2n",
            "NFP": "10", "NFN": "10", "P_PREFIX": "M", "N_PREFIX": "M",
            "P_DEVICE": "p L=30n TFIN=6.5n",
            "N_DEVICE": "n L=30n TFIN=6.5n",
            "OUTPUT_LOAD": "Cload out 0 10f",
            "INITIAL_CONDITION": ".ic V(o)=0.7",
            "ANALYSIS": ".tran 10p 5n",
        },
        SUBCIRCUIT_DECKS / "inverter_buffer_flat.spice.tmpl": {
            "MODEL_SETUP": '.include "model.lib"', "TEMP": "27",
            "VDD": "0.7",
            "INPUT_SPEC": "PULSE(0 0.7 0.5n 0.1n 0.1n 1n 2.2n)",
            "P_PREFIX": "N", "N_PREFIX": "N",
            "P_DEVICE": "p", "N_DEVICE": "n",
            "OUTPUT_LOAD": "Cload out 0 10f",
            "INITIAL_CONDITION": ".ic V(mid)=0.7 V(out)=0",
            "ANALYSIS": ".tran 10p 5n uic",
        },
        SUBCIRCUIT_DECKS / "inverter_buffer_hierarchical.spice.tmpl": {
            "MODEL_SETUP": (
                ".model n NMOS (LEVEL=72)\n.model p PMOS (LEVEL=72)"
            ),
            "TEMP": "27", "VDD": "0.7",
            "INPUT_SPEC": "PULSE 0 0.7 0.5n 0.1n 0.1n 1n 2.2n",
            "NFN": "10", "NFP": "10", "P_PREFIX": "M", "N_PREFIX": "M",
            "P_DEVICE": "p L=30n TFIN=6.5n",
            "N_DEVICE": "n L=30n TFIN=6.5n",
            "OUTPUT_LOAD": "Cload out 0 10f", "OUT_IC": "0",
            "ANALYSIS": ".tran 10p 5n",
        },
    }
    all_templates = set(EXAMPLES_DIR.rglob("*.spice.tmpl"))
    unowned = all_templates - catalog_paths - set(standalone)
    if unowned:
        failures.append(
            "templates lack a catalog or standalone rendering contract: "
            + ", ".join(str(path.relative_to(EXAMPLES_DIR))
                        for path in sorted(unowned))
        )
    for path, substitutions in standalone.items():
        if not path.is_file():
            failures.append(
                f"missing standalone template {path.relative_to(EXAMPLES_DIR)}"
            )
            continue
        try:
            rendered = render_deck_text(
                path.read_text(), substitutions, source_name=path.name,
            )
            if deck_tokens(rendered):
                failures.append(f"{path.name}: unresolved standalone token")
        except (KeyError, ValueError) as exc:
            failures.append(f"{path.name}: standalone render failed: {exc}")

    try:
        render_deck_text("* t\nV1 a 0 <VDD>\n.end\n", {"VDD": "0.8"})
    except Exception as exc:  # noqa: BLE001
        failures.append(f"strict in-memory renderer rejected a valid deck: {exc}")
    try:
        render_deck_text("* t\nV1 a 0 <VDD>\n.end\n", {"VDD": "0.8", "X": "1"})
        failures.append("strict in-memory renderer accepted an unused token")
    except KeyError:
        pass

    record = GateResult(
        case_id="source_follower", tech="TSMC5", corner="nominal",
        analysis="nmos", role=DIAGNOSTIC, status="diagnostic",
        metrics={"nrmse_pct": float("nan")},
    )
    parsed = parse_result_markers(record.marker())
    if len(parsed) != 1 or parsed[0]["metrics"]["nrmse_pct"] is not None:
        failures.append("GateResult marker is not strict-JSON round-trippable")

    try:
        from tests.common.circuit_benchmarks import BENCH
        from tests.common.simple_circuit_harness import (
            CORNERS, render_case_decks, topology_mismatch,
        )
        from tests.common.circuit_sweep import _shared_variants
        expected_corners = {
            "nominal", "temp_cold", "temp_hot", "vdd_low", "vdd_high",
            "body_reverse", "pn_n3p2", "pn_n2p3", "joint_hot_lowvdd",
        }
        if set(CORNERS) != expected_corners:
            failures.append(
                f"simple-v2 corner matrix changed unexpectedly: {list(CORNERS)}")
        temperatures = _shared_variants("TSMC5", "temp")
        joint = _shared_variants("TSMC5", "joint")
        if [item[0].temperature_c for item in temperatures] != [-25.0, 125.0]:
            failures.append("legacy simple sweep does not expose cold/hot corners")
        if len(joint) != 1 or joint[0][0].nfin != 3 \
                or joint[0][0].effective_nfin_p != 2:
            failures.append("legacy simple sweep joint corner is incomplete")
        for case in v2:
            for analysis in case.analyses:
                candidate, reference = render_case_decks(
                    case, analysis, BENCH["TSMC12"], CORNERS["nominal"],
                    baked_lib=Path("/tmp/catalog_fake_bsimcmg.lib"),
                )
                if deck_tokens(candidate) or deck_tokens(reference):
                    failures.append(
                        f"{case.case_id}/{analysis.name}: unresolved token")
                mismatch = topology_mismatch(candidate, reference)
                if mismatch:
                    failures.append(
                        f"{case.case_id}/{analysis.name}: {mismatch}")
    except (ImportError, KeyError, ValueError) as exc:
        failures.append(f"render/topology interface unavailable: {exc}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {len(v1)} simple-v1 + {len(v2)} simple-v2 cases catalogued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
