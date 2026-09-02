#!/usr/bin/env python3
"""Pure deck-rendering and topology-parity canaries for simple circuits.

Candidate and LEVEL=72 experiments are rendered from their paired files under
``examples/simple_circuits``. This check compares the actual rendered
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
from tests.common.base import SIMPLE_DECKS, deck_tokens, render_deck_text  # noqa: E402
from tests.common.simple_circuit_catalog import CASES  # noqa: E402
from tests.common.simple_circuit_harness import (  # noqa: E402
    CORNERS, render_case_decks, topology_mismatch,
)


def main() -> int:
    failures: list[str] = []
    fake_baked = PROJECT_ROOT / "results" / "_topology_fake_bsimcmg.lib"
    for tech_name in BENCH_TECHS:
        bt = BENCH[tech_name]
        for case in CASES:
            for corner_name, corner in CORNERS.items():
                for analysis in case.analyses:
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

        # The stage-count sweep is the only topology-changing legacy
        # dimension. Render both adapters from the same ring template and
        # prove parity at every declared odd count, not only at the baseline.
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

        inverter_values = {
            "VDD": f"{bt.vdd:g}", "LN": f"{bt.l_nmos * 1e9:g}n",
            "LP": f"{bt.l_pmos * 1e9:g}n", "NFN": str(bt.nfin),
            "NFP": str(bt.effective_nfin_p), "NMOS_PARAMS": "LEVEL=73",
            "PMOS_PARAMS": "LEVEL=73", "TEMP": f"{bt.temperature_c:g}",
            "ANALYSIS": ".dc Vin 0 1 0.1", "BAKED_NMOS": "/tmp/n.lib",
            "BAKED_PMOS": "/tmp/p.lib", "NMOS": bt.nmos_model,
            "PMOS": bt.pmos_model, "TD": "0.2n", "TR": "50p",
            "TF": "50p", "PW": "1n", "PER": "2.1n", "CLOAD": "1f",
        }
        for candidate_name, reference_name in (
            ("directnet_inverter_dc.sp", "bsimcmg_inverter_dc.cir"),
            ("nn_inverter_tran.sp", "bsimcmg_inverter_tran.cir"),
        ):
            rendered = []
            for filename in (candidate_name, reference_name):
                template = (SIMPLE_DECKS / filename).read_text()
                substitutions = {
                    token: inverter_values[token] for token in deck_tokens(template)
                }
                rendered.append(render_deck_text(
                    template, substitutions, source_name=filename,
                ))
            mismatch = topology_mismatch(rendered[0], rendered[1])
            if mismatch:
                failures.append(
                    f"{tech_name}/inverter/{candidate_name}: {mismatch}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(
        f"PASS: {len(BENCH_TECHS)} technologies × {len(CASES)} catalog "
        f"cases × {len(CORNERS)} corners plus ring stage-count variants "
        "and inverter gate pairs have identical topology")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
