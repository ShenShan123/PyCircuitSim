"""Fail-closed selection contracts for standalone verification gates."""
from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.perf.verify_latch_basin_gpu import main as latch_basin_main
from tests.simple_circuits.verify_bsimcmg_tran_comprehensive import (
    main as bsimcmg_tran_main,
)
from tests.simple_circuits.verify_circuit_opamp import main as opamp_main
from tests.simple_circuits.verify_circuit_ring_osc import main as ring_main
from tests.simple_circuits.verify_circuit_sram_snm import main as sram_main
from tests.simple_circuits.verify_circuit_sweep import main as circuit_sweep_main
from tests.simple_circuits.verify_circuit_switchcap import main as switchcap_main
from tests.simple_circuits.verify_multi_tech import main as multi_tech_main
from tests.simple_circuits.verify_nn_subckt import main as nn_subckt_main
from tests.single_devices.verify_bsimcmg_dc_comprehensive import (
    main as bsimcmg_dc_main,
)
from tests.single_devices.verify_nn_lifted_source_dc import (
    main as lifted_source_main,
)

GateMain = Callable[[list[str] | None], int]


@pytest.mark.parametrize(
    ("entry_point", "argv"),
    (
        (bsimcmg_dc_main, ["--sweep", "vt,unknown"]),
        (bsimcmg_dc_main, ["--device", "nmos,unknown"]),
        (bsimcmg_dc_main, ["--tech", "TSMC5,TSMC5"]),
        (bsimcmg_dc_main, ["--tech", "TSMC5,"]),
        (bsimcmg_tran_main, ["--sweep", "vt,unknown"]),
        (bsimcmg_tran_main, ["--tech", "TSMC5,TSMC5"]),
        (multi_tech_main, ["--analysis", "dc,unknown"]),
        (multi_tech_main, ["--analysis", "dc,dc"]),
        (multi_tech_main, ["--tech", "TSMC5,TSMC5"]),
        (lifted_source_main, ["--techs", ""]),
        (lifted_source_main, ["--techs", "TSMC5,TSMC5"]),
        (nn_subckt_main, ["--analysis", "dc,unknown"]),
        (nn_subckt_main, ["--analysis", "dc,dc"]),
        (nn_subckt_main, ["--analysis", "dc,"]),
        (latch_basin_main, ["--tech", "TSMC5,unknown"]),
        (latch_basin_main, ["--tech", "TSMC5,TSMC5"]),
    ),
)
def test_parametric_gate_rejects_invalid_or_duplicate_selection(
    entry_point: GateMain,
    argv: list[str],
) -> None:
    """A bad axis must not run only the valid subset and shrink a denominator."""
    with pytest.raises(SystemExit) as exc_info:
        entry_point(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "entry_point",
    (opamp_main, ring_main, sram_main, switchcap_main),
)
@pytest.mark.parametrize("tech", ("", "TSMC5,TSMC5", "TSMC5,unknown"))
def test_qualification_gate_rejects_invalid_technology_selection(
    entry_point: GateMain,
    tech: str,
) -> None:
    """Qualification totals require a non-empty set of unique known nodes."""
    with pytest.raises(SystemExit) as exc_info:
        entry_point(["--tech", tech])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("nfin", ("", "two", "0", "2,2", "2,"))
def test_sram_gate_rejects_invalid_fin_count_selection(nfin: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        sram_main(["--nfin", nfin])

    assert exc_info.value.code == 2


def test_circuit_sweep_answers_top_level_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert circuit_sweep_main(["--help"]) == 0
    assert "usage:" in capsys.readouterr().out.lower()
