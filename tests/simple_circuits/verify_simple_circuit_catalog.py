#!/usr/bin/env python3
"""Pure contract checks for the versioned simple-circuit catalog."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.base import (  # noqa: E402
    SIMPLE_DECKS, deck_tokens, render_deck_text,
)
from tests.common.gate_result import GateResult, parse_result_markers  # noqa: E402
from tests.common.simple_circuit_catalog import (  # noqa: E402
    CASES, DIAGNOSTIC, QUALIFICATION, SIMPLE_V1, SIMPLE_V2, cases,
)


def main() -> int:
    failures: list[str] = []

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
    for case in CASES:
        if case.case_id in seen:
            failures.append(f"duplicate case ID: {case.case_id}")
        seen.add(case.case_id)
        for relative in (case.candidate_deck, case.reference_deck):
            path = SIMPLE_DECKS / relative
            if not path.is_file():
                failures.append(f"{case.case_id}: missing deck {relative}")
                continue
            text = path.read_text()
            tokens = deck_tokens(text)
            if "ANALYSIS" not in tokens:
                failures.append(f"{relative}: must declare <ANALYSIS>")
            if case.score_version == SIMPLE_V2:
                expected_body_tokens = {
                    "nmos": "BODY_N", "pmos": "BODY_P",
                }
                for device in case.device_kinds:
                    token = expected_body_tokens[device]
                    if token not in tokens:
                        failures.append(
                            f"{relative}: {case.case_id} {device} body-bias "
                            f"corner needs <{token}>"
                        )

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
