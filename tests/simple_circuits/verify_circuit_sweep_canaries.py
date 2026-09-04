#!/usr/bin/env python3
"""Pure deck-rendering and topology-parity canaries for simple circuits.

Candidate and LEVEL=72 experiments are rendered from one shared template under
``circuit_templates``. This check compares the actual rendered
connectivity for every catalog analysis and also exercises the topology-
changing ring sweep; no simulator or checkpoint is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH, BENCH_TECHS, RingOscParams, directnet_ringosc,
    ngspice_ringosc,
)
from tests.common.base import (  # noqa: E402
    deck_tokens, parse_no_options, template_deck, render_deck_text,
)
from tests.common.simple_circuit_catalog import CASES  # noqa: E402
from tests.common.simple_circuit_harness import (  # noqa: E402
    CORNERS, analysis_applies_to_corner, render_case_decks, topology_mismatch,
)


def main() -> int:
    failures: list[str] = []
    checked = 0
    fake_baked = PROJECT_ROOT / "results" / "_topology_fake_bsimcmg.lib"
    for tech_name in BENCH_TECHS:
        bt = BENCH[tech_name]
        for case in CASES:
            for corner_name, corner in CORNERS.items():
                for analysis in case.analyses:
                    if not analysis_applies_to_corner(case, analysis, bt, corner):
                        continue
                    checked += 1
                    try:
                        candidate, reference = render_case_decks(
                            case, analysis, bt, corner,
                            baked_lib=fake_baked,
                        )
                        mismatch = topology_mismatch(candidate, reference)
                    except Exception as exc:  # noqa: BLE001
                        mismatch = f"{type(exc).__name__}: {exc}"
                    if mismatch:
                        failures.append(
                            f"{tech_name}/{case.case_id}/{corner_name}/"
                            f"{analysis.name}: {mismatch}")

        # Each declared stage count owns a template. Render both adapters from
        # the selected file and prove parity, not only at the baseline.
        for n_stages in (3, 5, 7, 9):
            params = RingOscParams(n_stages=n_stages)
            candidate = directnet_ringosc(bt, params, params.tstop)
            reference = ngspice_ringosc(
                bt, params, fake_baked, params.tstop,
            )["body"]
            mismatch = topology_mismatch(candidate, reference)
            if mismatch:
                failures.append(
                    f"{tech_name}/ring_osc/n_stages={n_stages}: {mismatch}")

        inverter_path = template_deck("inverter.spice.tmpl")
        inverter_template = inverter_path.read_text()
        common = {
            "VDD": f"{bt.vdd:g}", "TEMP": f"{bt.temperature_c:g}",
            "OUTPUT_LOAD": "", "INITIAL_CONDITION": "",
            "INPUT_SPEC": "0", "ANALYSIS": ".dc Vin 0 1 0.1",
        }
        candidate_values = {
            **common, "MODEL_SETUP": ".model nmos_nn NMOS (LEVEL=73)\n"
            ".model pmos_nn PMOS (LEVEL=73)",
            "N_PREFIX": "M", "P_PREFIX": "M",
            "N_DEVICE": (
                f"nmos_nn L={bt.l_nmos * 1e9:g}n NFIN={bt.nfin}"
            ),
            "P_DEVICE": (
                f"pmos_nn L={bt.l_pmos * 1e9:g}n "
                f"NFIN={bt.effective_nfin_p}"
            ),
        }
        reference_values = {
            **common, "MODEL_SETUP": '.include "/tmp/pair.lib"',
            "N_PREFIX": "N", "P_PREFIX": "N",
            "N_DEVICE": bt.nmos_model, "P_DEVICE": bt.pmos_model,
        }
        for analysis_name in ("dc", "tran"):
            if analysis_name == "tran":
                candidate_values.update({
                    "INPUT_SPEC": "PULSE 0 1 0.2n 50p 50p 1n 2.1n",
                    "OUTPUT_LOAD": "Cload out 0 1f",
                    "INITIAL_CONDITION": ".ic V(out)=1",
                    "ANALYSIS": ".tran 2p 3n uic",
                })
                reference_values.update({
                    "INPUT_SPEC": "PULSE(0 1 0.2n 50p 50p 1n 2.1n)",
                    "OUTPUT_LOAD": "Cload out 0 1f",
                    "INITIAL_CONDITION": ".ic V(out)=1",
                    "ANALYSIS": ".tran 2p 3n uic",
                })
            rendered = [
                render_deck_text(
                    inverter_template,
                    {token: values[token]
                     for token in deck_tokens(inverter_template)},
                    source_name=inverter_path.name,
                )
                for values in (candidate_values, reference_values)
            ]
            mismatch = topology_mismatch(rendered[0], rendered[1])
            if mismatch:
                failures.append(
                    f"{tech_name}/inverter/{analysis_name}: {mismatch}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(
        f"PASS: {checked} applicable technology/case/analysis/corner renders "
        "plus ring stage-count variants "
        "and inverter gate pairs have identical topology")
    return 0


if __name__ == "__main__":
    parse_no_options(__doc__ or "")
    raise SystemExit(main())
