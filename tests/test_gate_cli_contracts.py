"""Fail-closed selection contracts for standalone verification gates."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.simple_circuits.verify_bsimcmg_tran_comprehensive import (
    main as bsimcmg_tran_main,
)
from tests.simple_circuits.verify_circuit_opamp import main as opamp_main
from tests.simple_circuits.verify_circuit_opamp_ac import main as opamp_ac_main
from tests.simple_circuits.verify_circuit_topologies import (
    main as topologies_main,
)
from tests.simple_circuits.verify_nn_ac import main as nn_ac_main
from tests.simple_circuits.verify_nn_multi_tech_tran import (
    main as multi_tech_tran_main,
)
from tests.single_devices.verify_device_integrity import (
    main as device_integrity_main,
)
from tests.single_devices.verify_nn_multi_tech_dc import (
    main as multi_tech_dc_main,
)
from tests.single_devices.verify_terminal_integrity import (
    main as terminal_integrity_main,
)
from tests.simple_circuits.verify_circuit_ring_osc import main as ring_main
from tests.simple_circuits.verify_circuit_sram_snm import main as sram_main
from tests.simple_circuits.verify_circuit_sweep import main as circuit_sweep_main
from tests.simple_circuits.verify_circuit_switchcap import main as switchcap_main
from tests.simple_circuits.verify_multi_tech import main as multi_tech_main
from tests.simple_circuits.verify_nn_subckt import main as nn_subckt_main
from tests.single_devices.verify_bsimcmg_dc_comprehensive import (
    main as bsimcmg_dc_main,
)
from tests.single_devices.verify_data_geometry_coverage import (
    main as geometry_coverage_main,
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
        # The campaign driver passes ``--tech``; the canary must read it.
        (lifted_source_main, ["--tech", "TSMC5,"]),
        (lifted_source_main, ["--tech", "TSMC5,unknown"]),
        (geometry_coverage_main, ["--data-dir", "/nonexistent/datasets"]),
        (nn_subckt_main, ["--analysis", "dc,unknown"]),
        (nn_subckt_main, ["--analysis", "dc,dc"]),
        (nn_subckt_main, ["--analysis", "dc,"]),
        # V7.7.2: the former hand-rolled selectors dropped an empty field, so
        # ``--tech "TSMC5,"`` ran the valid subset and exited 0.
        (device_integrity_main, ["--tech", "TSMC5,"]),
        (device_integrity_main, ["--device", "nmos,nmos"]),
        (device_integrity_main, ["--corner", "nominal,unknown"]),
        (terminal_integrity_main, ["--tech", "TSMC5,"]),
        (terminal_integrity_main, ["--device", "nmos,unknown"]),
        (topologies_main, ["--tech", "TSMC5,"]),
        (topologies_main, ["--case", "current_mirror,"]),
        (topologies_main, ["--case", "ring_osc"]),        # simple-v1, not v2
        (topologies_main, ["--corner", "nominal,nominal"]),
        (nn_ac_main, ["--tech", "TSMC5,"]),
        (nn_ac_main, ["--device", "nmos,"]),
        (opamp_ac_main, ["--tech", "TSMC5,"]),
        (multi_tech_dc_main, ["--tech", "TSMC5,"]),
        (multi_tech_dc_main, ["--tech", "ASAP7"]),
        (multi_tech_tran_main, ["--tech", "TSMC5,"]),
        (multi_tech_tran_main, ["--analysis", "vtc,"]),
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


def _sram_marker_after_corner_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    failing_stage: str,
) -> tuple[int, dict[str, Any]]:
    """Run the SRAM gate with one engine failing at a single NFIN corner."""
    import numpy as np

    import tests.simple_circuits.verify_circuit_sram_snm as sram_gate
    from tests.common.gate_result import parse_result_markers

    def lobe(bt: Any, _nfin: int, _work_dir: Path) -> dict[str, Any]:
        q = np.linspace(0.0, bt.vdd, 61)
        return {"q": q, "qb": bt.vdd - q}

    def fail(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("DC operating point did not converge")

    monkeypatch.setattr(sram_gate, "RESULTS_BASE", tmp_path)
    monkeypatch.setattr(
        sram_gate, "ngspice_lobe", fail if failing_stage == "reference" else lobe)
    monkeypatch.setattr(
        sram_gate, "directnet_lobe", fail if failing_stage == "candidate" else lobe)
    monkeypatch.setattr(
        sram_gate, "force_ic_probe",
        lambda *_args, **_kwargs: {"state1": True, "state0": True})
    code = sram_gate.main(["--tech", "TSMC16", "--nfin", "5"])
    markers = parse_result_markers(capsys.readouterr().out)
    assert len(markers) == 1
    return code, markers[0]


def test_sram_candidate_corner_failure_is_attributed_to_candidate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """NGSPICE converged, so a failed NN corner must not be a reference error.

    The marker used to derive both convergence flags from "any corner
    errored", which labelled BSIM-AR's own NFIN=5 failure as ``reference``.
    """
    code, marker = _sram_marker_after_corner_failure(
        monkeypatch, capsys, tmp_path, "candidate")

    assert code == 1
    assert marker["error_kind"] == "candidate"
    assert marker["reference_converged"] is True
    assert marker["candidate_converged"] is False


def test_sram_reference_corner_failure_is_attributed_to_reference(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """A failed NGSPICE corner remains a reference error."""
    _code, marker = _sram_marker_after_corner_failure(
        monkeypatch, capsys, tmp_path, "reference")

    assert marker["error_kind"] == "reference"
    assert marker["reference_converged"] is False


def test_circuit_sweep_answers_top_level_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert circuit_sweep_main(["--help"]) == 0
    assert "usage:" in capsys.readouterr().out.lower()
