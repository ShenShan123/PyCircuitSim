"""Regression contracts for the catalog-driven circuit experiment seam."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import Resistor, VoltageSource
from pycircuitsim.solver import ACSolver
from tests.common.base import control_deck, template_deck
from tests.common.circuit_benchmarks import BENCH, BENCH_TECHS
from tests.common.gate_result import GateResult, result_exit_code
from tests.common.simple_circuit_catalog import AnalysisSpec, SIMPLE_V1, cases
from tests.common.simple_circuit_harness import (
    CORNERS,
    CandidateConvergenceError,
    CandidateSupportError,
    RunSpec,
    Trace,
    analysis_axis_limits,
    analysis_applies_to_corner,
    applicable_analyses,
    clock_hold_samples,
    compare_traces,
    connectivity_signature,
    get_case_baked_modelcard,
    physical_deck_mismatch,
    render_case_control_deck,
    render_case_decks,
    topology_signature,
    topology_mismatch,
    validate_analysis_metrics,
)


SIMPLE_V1_RENDER_SHA256 = {
    "ring_osc/TSMC5/oscillation/candidate": "aac8e6630c50ec4935b6a3e5aa9421c85b4dd933679d4622b7df871ebd32109f",
    "ring_osc/TSMC5/oscillation/reference": "da6e10f2d1595eacf5fed709761057143d676c3dbe5ddae54af04b80ea9720d5",
    "ring_osc/TSMC6/oscillation/candidate": "40924c02f9e9d1144fd0324756ff67f670091a5d7669026708c2bd14c85961f2",
    "ring_osc/TSMC6/oscillation/reference": "7ead6db6f53c18345bf1c40e79290cf9f26431a295c9eff47e7388af1ebf087f",
    "ring_osc/TSMC7/oscillation/candidate": "2151c283d5427934a82258aec33a78cd5ec0c1dee289070d33162bdc77769718",
    "ring_osc/TSMC7/oscillation/reference": "cb460189e28b273b4fd74c2e867ea21d0431458225c80ea475b654188ac40862",
    "ring_osc/TSMC12/oscillation/candidate": "a8a13a3e58b0a1f5bcdb79a4de60f76e2e8195db3ffc845eb37a7917a816a39c",
    "ring_osc/TSMC12/oscillation/reference": "5aee490637000e72646c6cb604c391d586e95b8d2fb94241030751fc30ba8298",
    "ring_osc/TSMC16/oscillation/candidate": "ade73239fc8fb39ef89e34ee4ef3af0ed8a46663f30eed25f1df2f42d7cbd88e",
    "ring_osc/TSMC16/oscillation/reference": "42723c573b4392e1f450b6f9c32b4a3a3e33e2213fde2df6977cf83c21ed1706",
    "opamp/TSMC5/transfer/candidate": "93aa3a269635a291293be90f0c5f0416cc18dc3b30a30df1dead3d40364733db",
    "opamp/TSMC5/transfer/reference": "87c58c7f441e633c005106142da736b3c51bc6d4e997350a052e7c524f6fd0f4",
    "opamp/TSMC6/transfer/candidate": "15adbd182aea480c2a1708b1b768bf7ffb7afa5777c5b83c8f909981b0a22635",
    "opamp/TSMC6/transfer/reference": "b13f1ed340b1aeb38a933b6a1bae368675b3dfdc004d255a191357b361771932",
    "opamp/TSMC7/transfer/candidate": "7f7398774997b44a5868dc5b89a50cd2e1e759f469ba377bf5b2cfd5d116a097",
    "opamp/TSMC7/transfer/reference": "151f6996af0c6f13b594f9f29ff04615b75b161945c9fa9f85c899081e04c39a",
    "opamp/TSMC12/transfer/candidate": "2eeb64592f4c27c8f3eafea3ce26d7a58a56f77e25a36967b91f8630e9507a98",
    "opamp/TSMC12/transfer/reference": "d326f56c577952758cf15b94297eaef221b47e6b2ba8bd815a15c8613ba68e07",
    "opamp/TSMC16/transfer/candidate": "d7185d6e3c3a7bb37579a949f9eda9b06bfabe82d85b966e919b7e5addcbb386",
    "opamp/TSMC16/transfer/reference": "7006cce697bea9a058b760643f94bd021a8511a5f72939c22673d15994d202cd",
    "sram_snm/TSMC5/read_snm/candidate": "85170258d6f372f44c0e8a5a672eae1cc50b71014313789bfad1517ad0adcb90",
    "sram_snm/TSMC5/read_snm/reference": "d45d7c492d90d6edb78603cc0ccc322a2925fe3cf5da57c62d8c535b0a590fb4",
    "sram_snm/TSMC6/read_snm/candidate": "b17bd3013ceed40f80f0c82f8e4d9271a7570bd240b5cb19195dc224f0edea03",
    "sram_snm/TSMC6/read_snm/reference": "e3417280bb466c6f91905f8529ab47c0fd23ff4f8ca86f3e3a5bc7b22d3b32e3",
    "sram_snm/TSMC7/read_snm/candidate": "254ce8cce5200b65dc35379df52f130df2fdf6903140298d13548fe7c119eb97",
    "sram_snm/TSMC7/read_snm/reference": "4bd587d58f267c5cfe1778ff86ddd8dbc8fb7134e83e39e7d83104961af4c04f",
    "sram_snm/TSMC12/read_snm/candidate": "59d0715c731308f4a65828e7a82774b603905ff0501eec702ddb9b7eadd7e948",
    "sram_snm/TSMC12/read_snm/reference": "b783a6838fd06268916dc827a821cc58616cc0ea2fa68f584bd9fb2031188361",
    "sram_snm/TSMC16/read_snm/candidate": "fa2805eadc1df5525dcc57a5fc43c5338f13a9f8973d89f87c0f754648a4344f",
    "sram_snm/TSMC16/read_snm/reference": "4da8b84a262cfdbefc9ecdc1c44b7b2897797a9fe0d23aa281c7418b0a4a3c0d",
    "switchcap/TSMC5/sample_hold/candidate": "e50b4d25a3971b5b69b96650db657f215e5f0fc64f98c88d0b911322496ad773",
    "switchcap/TSMC5/sample_hold/reference": "c114c2b6363f44355aef77e5f0f076f16a1fbba1f2135875008127f6c470cbe8",
    "switchcap/TSMC6/sample_hold/candidate": "833625fc018a72f6e358c3fd294dec6b2c3e1d44fde5168a29045003a10658c9",
    "switchcap/TSMC6/sample_hold/reference": "c915b52daf0a7ae23f112a0cf39a413a63933ecc3247c9c2c43e2a1f0759d730",
    "switchcap/TSMC7/sample_hold/candidate": "0790033de8712c16b7a30ffedfa1ac303f9bc7ef17e8442f057caf6bb5e519bc",
    "switchcap/TSMC7/sample_hold/reference": "b06c4815da0ef0ccf7388256b8762d1deacf74886bd303836714bc5569d89f81",
    "switchcap/TSMC12/sample_hold/candidate": "63a14b6891347cc29d4ede8bb40ad7a896fe21197d5275ad9025a03b16876395",
    "switchcap/TSMC12/sample_hold/reference": "33cf30dca3e3472edeed9a10e1762422fe4550f90d92a83fa391fe863deee86a",
    "switchcap/TSMC16/sample_hold/candidate": "6447e4e7cb29874f15bfa9246ff8272c35c71ccec2e745093ce9ee836a02ea82",
    "switchcap/TSMC16/sample_hold/reference": "fe5db7d06db2f08a193155d7edc24e30c9322b33ceabe3b923c38a76280b707b",
}


def test_simple_v1_rendered_decks_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Published simple-v1 cells must remain byte-identical."""
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    observed: dict[str, str] = {}
    for case in cases(score_version=SIMPLE_V1):
        for tech_name in BENCH_TECHS:
            for analysis in case.analyses:
                candidate, reference = render_case_decks(
                    case,
                    analysis,
                    BENCH[tech_name],
                    CORNERS["nominal"],
                    baked_lib=Path(f"/frozen/{tech_name.lower()}.lib"),
                )
                prefix = f"{case.case_id}/{tech_name}/{analysis.name}"
                observed[f"{prefix}/candidate"] = hashlib.sha256(
                    candidate.encode()
                ).hexdigest()
                observed[f"{prefix}/reference"] = hashlib.sha256(
                    reference.encode()
                ).hexdigest()

    assert observed == SIMPLE_V1_RENDER_SHA256


@pytest.mark.parametrize(
    ("reference_converged", "candidate_converged", "partial"),
    ((False, True, False), (True, False, False), (True, False, True)),
)
def test_incomplete_execution_cannot_be_reported_as_characterized(
    reference_converged: bool,
    candidate_converged: bool,
    partial: bool,
) -> None:
    """A diagnostic verdict is valid only for a complete physical solve."""
    with pytest.raises(ValueError, match="incomplete execution"):
        GateResult(
            case_id="case",
            tech="TSMC12",
            corner="nominal",
            analysis="tran",
            role="diagnostic",
            status="diagnostic",
            reference_converged=reference_converged,
            candidate_converged=candidate_converged,
            partial=partial,
        )


def test_explicit_complete_state_cannot_override_incomplete_execution() -> None:
    with pytest.raises(ValueError, match="complete execution"):
        GateResult(
            case_id="case",
            tech="TSMC12",
            corner="nominal",
            analysis="analysis",
            role="diagnostic",
            status="diagnostic",
            candidate_converged=False,
            partial=True,
            execution_state="complete",
        )


def test_error_result_names_execution_state_and_origin() -> None:
    result = GateResult(
        case_id="case",
        tech="TSMC12",
        corner="nominal",
        analysis="tran",
        role="diagnostic",
        status="error",
        error="timestep failed",
        reference_converged=True,
        candidate_converged=False,
        partial=True,
        execution_state="partial",
        error_kind="candidate",
    )

    assert result.payload()["execution_state"] == "partial"
    assert result.payload()["error_kind"] == "candidate"


@pytest.mark.parametrize(
    ("axis", "message"),
    (
        (np.asarray([0.0, 0.5, 0.4, 1.0]), "strictly monotonic"),
        (np.asarray([0.0, 0.5, 0.5, 1.0]), "strictly monotonic"),
        (np.asarray([0.0, 0.5, 0.8]), "does not reach requested stop"),
    ),
)
def test_trace_rejects_incomplete_or_nonmonotonic_axes(
    axis: np.ndarray,
    message: str,
) -> None:
    trace = Trace("time", axis, {"v(out)": axis.copy()})

    with pytest.raises(ValueError, match=message):
        trace.validate(expected_start=0.0, expected_stop=1.0)


def test_trace_accepts_complete_decreasing_dc_axis() -> None:
    trace = Trace(
        "sweep",
        np.asarray([1.0, 0.5, 0.0]),
        {"v(out)": np.asarray([0.0, 0.5, 1.0])},
    )

    trace.validate(expected_start=1.0, expected_stop=0.0)


def test_dc_trace_allows_only_one_rounding_step_before_requested_stop() -> None:
    accepted = Trace(
        "sweep",
        np.asarray([0.0, 0.4, 0.795]),
        {"v(out)": np.asarray([0.0, 0.2, 0.3])},
    )
    rejected = Trace(
        "sweep",
        np.asarray([0.0, 0.4, 0.79]),
        {"v(out)": np.asarray([0.0, 0.2, 0.3])},
    )

    accepted.validate(
        expected_start=0.0,
        expected_stop=0.8,
        endpoint_tolerance=0.0050000001,
    )
    with pytest.raises(ValueError, match="does not reach requested stop"):
        rejected.validate(
            expected_start=0.0,
            expected_stop=0.8,
            endpoint_tolerance=0.0050000001,
        )


def test_transient_trace_allows_first_internal_substep() -> None:
    trace = Trace(
        "time",
        np.asarray([5e-14, 1e-12, 5e-12]),
        {"v(out)": np.asarray([0.0, 0.1, 0.2])},
    )

    trace.validate(
        expected_start=0.0,
        expected_stop=5e-12,
        endpoint_tolerance=5e-12,
    )


def test_trace_rejects_internal_gaps_larger_than_the_declared_step() -> None:
    trace = Trace(
        "sweep",
        np.asarray([0.0, 0.8]),
        {"v(out)": np.asarray([0.0, 1.0])},
    )

    with pytest.raises(ValueError, match="gap"):
        trace.validate(
            expected_start=0.0,
            expected_stop=0.8,
            endpoint_tolerance=0.0050000001,
            max_step=0.005,
        )


def test_trace_allows_at_most_one_missing_dc_endpoint() -> None:
    trace = Trace(
        "sweep",
        np.arange(0.005, 0.8, 0.005),
        {"v(out)": np.arange(0.005, 0.8, 0.005)},
    )

    with pytest.raises(ValueError, match="points"):
        trace.validate(
            expected_start=0.0,
            expected_stop=0.8,
            endpoint_tolerance=0.0050000001,
            max_step=0.005,
            minimum_points=160,
        )


def test_device_trace_rejects_nonmonotonic_or_incomplete_sweeps() -> None:
    from tests.common.device_integrity import DeviceTrace

    nonmonotonic = DeviceTrace(
        np.asarray([0.0, 0.2, 0.1]),
        np.asarray([0.0, 1.0, 2.0]),
    )
    incomplete = DeviceTrace(
        np.asarray([0.0, 0.4, 0.8]),
        np.asarray([0.0, 1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="monotonic"):
        nonmonotonic.validate()
    with pytest.raises(ValueError, match="gap"):
        incomplete.validate(
            expected_start=0.0,
            expected_stop=0.8,
            endpoint_tolerance=0.0050000001,
            max_step=0.005,
        )


def test_pmos_subthreshold_metrics_start_at_zero_gate_overdrive() -> None:
    from tests.common.device_integrity import (
        SweepSpec,
        suite_metrics,
        validate_device_metrics,
    )

    grid = np.linspace(-0.44, 0.0, 101)
    reference = 1e-12 * 10.0 ** (np.abs(grid) / 0.08)
    spec = SweepSpec(
        suite="subthreshold",
        label="idvg_log",
        device="pmos",
        axis="vgs",
        start=0.0,
        stop=-0.44,
        step=-0.0044,
        vds=-0.4,
    )

    metrics, domain = suite_metrics(
        spec,
        grid,
        reference,
        reference,
        vdd=0.8,
    )

    assert domain["ioff_ref_a"] == pytest.approx(1e-12)
    assert domain["ss_ref_mv_dec"] == pytest.approx(80.0, rel=0.02)
    validate_device_metrics(spec, metrics, domain)

    domain["ss_ref_mv_dec"] = None
    with pytest.raises(ValueError, match="ss_ref_mv_dec"):
        validate_device_metrics(spec, metrics, domain)


def test_pmos_output_metrics_end_at_maximum_drain_overdrive() -> None:
    from tests.common.device_integrity import SweepSpec, suite_metrics

    grid = np.linspace(-0.8, -0.02, 40)
    current = -np.minimum(np.abs(grid) / 0.2, 1.0)
    spec = SweepSpec(
        suite="output",
        label="vgs1.00",
        device="pmos",
        axis="vds",
        start=-0.02,
        stop=-0.8,
        step=-0.02,
        vgs=-0.8,
    )

    _metrics, domain = suite_metrics(
        spec,
        grid,
        current,
        current,
        vdd=0.8,
    )

    assert domain["idsat_ref_a"] == pytest.approx(-1.0)
    assert domain["knee_vds_ref_v"] < 0.3


def test_switching_metrics_sample_clock_defined_hold_windows() -> None:
    axis = np.arange(9, dtype=float)
    clock = np.asarray([0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    storage = np.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=float)

    samples = clock_hold_samples(axis, storage, clock, level=0.5)

    np.testing.assert_allclose(samples, [2.0, 4.0])


@pytest.mark.parametrize(
    ("kind", "card", "expected"),
    (
        ("dc", "dc Vin 0.8 0 -0.005", (0.8, 0.0)),
        ("tran", "tran 5p 12n uic", (0.0, 12e-9)),
        ("ac", "ac dec 10 1k 100g", (1e3, 100e9)),
        ("op", "op", (0.0, 0.0)),
    ),
)
def test_analysis_axis_limits_parse_the_requested_experiment(
    kind: str,
    card: str,
    expected: tuple[float, float],
) -> None:
    spec = AnalysisSpec("analysis", kind, card, ("v(out)",))

    assert analysis_axis_limits(spec) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("domain", "message"),
    (
        ({}, "missing required metrics"),
        (
            {
                "period_test_s": 1e-9,
                "period_ref_s": 1e-9,
                "period_error_pct": float("nan"),
            },
            "non-finite required metrics",
        ),
    ),
)
def test_metric_contract_rejects_missing_or_nonfinite_events(
    domain: dict[str, float],
    message: str,
) -> None:
    analysis = AnalysisSpec(
        "oscillation",
        "tran",
        "tran 1p 10n uic",
        ("v(out)",),
        metric_profile="ring_osc",
    )
    metrics = {
        "mre_pct": 0.0,
        "r2": 1.0,
        "nrmse_pct": 0.0,
        "max_err": 0.0,
    }

    with pytest.raises(ValueError, match=message):
        validate_analysis_metrics(analysis, metrics, domain)


@pytest.mark.parametrize("invalid", (None, "not-a-number", True))
def test_metric_contract_rejects_nonnumeric_required_values(
    invalid: object,
) -> None:
    analysis = AnalysisSpec(
        "trace",
        "dc",
        "dc Vin 0 1 0.1",
        ("v(out)",),
    )
    metrics = {
        "mre_pct": 0.0,
        "r2": 1.0,
        "nrmse_pct": invalid,
        "max_err": 0.0,
    }

    with pytest.raises(ValueError, match="numeric"):
        validate_analysis_metrics(analysis, metrics, {})


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("Vdd vdd 0 0.8", "Vdd vdd 0 0.7"),
        ("Rload out 0 10k", "Rload out 0 11k"),
        ("Cload out 0 2f", "Cload out 0 3f"),
        (".temp 27", ".temp 125"),
        (".ic V(out)=0.8", ".ic V(out)=0"),
        (".options rshunt=1e12", ".options rshunt=1e9"),
    ),
)
def test_physical_parity_detects_value_changes(old: str, new: str) -> None:
    deck = (
        "* parity\n"
        "Vdd vdd 0 0.8\n"
        "Rload out 0 10k\n"
        "Cload out 0 2f\n"
        "Mn out in 0 0 n L=16n NFIN=2\n"
        ".temp 27\n"
        ".ic V(out)=0.8\n"
        ".options rshunt=1e12\n"
        ".tran 1p 2n uic\n"
        ".end\n"
    )

    assert topology_mismatch(deck, deck.replace(old, new))


def test_physical_parity_normalizes_equivalent_pulse_syntax() -> None:
    candidate = "Vclk clk 0 PULSE 0 1 1n 1p 1p 2n 4n\n.end\n"
    reference = "Vclk clk 0 PULSE(0 1 1n 1p 1p 2n 4n)\n.end\n"

    assert topology_mismatch(candidate, reference) == ""


def test_physical_parity_preserves_independent_source_identity() -> None:
    candidate = "Va a 0 0\nVb b 0 0\n.end\n"
    reference = "Vb a 0 0\nVa b 0 0\n.end\n"

    assert "physical deck mismatch" in topology_mismatch(candidate, reference)


@pytest.mark.parametrize(
    ("old", "new"),
    (("L=16n", "L=24n"), ("NFIN=2", "NFIN=5"), ("VT=svt", "VT=lvt")),
)
def test_physical_manifest_detects_candidate_model_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    case = next(item for item in cases() if item.case_id == "current_mirror")
    analysis = case.analyses[0]
    candidate, reference = render_case_decks(
        case,
        analysis,
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/tsmc12.lib"),
    )
    resolved = replace(analysis, card="dc Voutn 0 0.8 0.005")
    assert physical_deck_mismatch(
        candidate,
        reference,
        resolved,
        BENCH["TSMC12"],
        baked_lib=Path("/frozen/tsmc12.lib"),
    ) == ""
    if old == "VT=svt":
        old = "VT=svt"
        assert old in candidate
    else:
        assert old in candidate

    assert physical_deck_mismatch(
        candidate.replace(old, new, 1),
        reference,
        resolved,
        BENCH["TSMC12"],
        baked_lib=Path("/frozen/tsmc12.lib"),
    )


def test_single_device_physical_parity_detects_geometry_drift(
    tmp_path: Path,
) -> None:
    from tests.common.device_integrity import build_sweeps, render_device_decks

    bt = BENCH["TSMC12"]
    spec = build_sweeps(bt, "nmos")[0]
    baked = tmp_path / "baked.lib"
    candidate, reference = render_device_decks(
        spec,
        bt,
        baked_lib=baked,
        level=75,
    )
    analysis = AnalysisSpec(
        spec.label,
        "dc",
        f"dc {spec.sweep_source} {spec.start:.12g} "
        f"{spec.stop:.12g} {spec.step:.12g}",
        ("i(Vds)",),
    )

    assert "geometry mismatch" in physical_deck_mismatch(
        candidate.replace("L=16n", "L=24n"),
        reference,
        analysis,
        bt,
        baked_lib=baked,
        model_level=75,
        device_kinds=("nmos",),
    )


def test_run_spec_captures_model_pins_threads_and_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setenv("PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS", "nmos-pin")
    monkeypatch.setenv("PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS", "pmos-pin")
    monkeypatch.setenv("PYCIRCUITSIM_CAMPAIGN_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.setenv("PYCIRCUITSIM_TORCH_THREADS", "4")

    spec = RunSpec.from_environment()

    assert spec.model_level == 75
    assert spec.model_family == "DirectNet-Full"
    assert dict(spec.checkpoint_pins) == {
        "nmos": "nmos-pin",
        "pmos": "pmos-pin",
    }
    assert spec.campaign_manifest_sha256 == "a" * 64
    assert (spec.omp_threads, spec.mkl_threads, spec.torch_threads) == (2, 3, 4)


def test_run_spec_rejects_a_missing_explicit_checkpoint(tmp_path: Path) -> None:
    spec = RunSpec(
        75,
        "DirectNet-Full",
        checkpoint_pins=(("nmos", "missing_nmos"),),
    )

    with pytest.raises(FileNotFoundError, match="explicit nmos checkpoint"):
        spec.validate_checkpoint_pins(tmp_path)


def test_gate_result_carries_explicit_model_provenance() -> None:
    result = GateResult(
        case_id="case",
        tech="TSMC12",
        corner="nominal",
        analysis="dc",
        role="diagnostic",
        status="diagnostic",
        model_family="BSIM-AR-Full",
        model_level=76,
        checkpoint_pins={"nmos": "n-pin", "pmos": "p-pin"},
        campaign_manifest_sha256="b" * 64,
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )

    payload = result.payload()
    assert payload["model_family"] == "BSIM-AR-Full"
    assert payload["model_level"] == 76
    assert payload["checkpoint_pins"] == {"nmos": "n-pin", "pmos": "p-pin"}
    assert payload["campaign_manifest_sha256"] == "b" * 64


def test_ac_solver_exposes_complex_voltage_source_currents() -> None:
    circuit = Circuit()
    circuit.add_component(
        VoltageSource("V1", ["in", "0"], 0.0, ac_magnitude=1.0),
    )
    circuit.add_component(Resistor("R1", ["in", "0"], 1_000.0))

    result = ACSolver(circuit, dc_solution={"in": 0.0, "0": 0.0}).solve(
        np.asarray([1e3, 1e6]),
    )

    np.testing.assert_allclose(result["i(V1)"], [-1e-3, -1e-3])


def test_distinct_template_files_do_not_duplicate_a_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    signatures: dict[object, str] = {}
    for case in cases():
        candidate, _ = render_case_decks(
            case,
            case.analyses[0],
            BENCH["TSMC12"],
            CORNERS["nominal"],
            baked_lib=Path("/frozen/tsmc12.lib"),
        )
        signature = tuple(sorted(connectivity_signature(candidate).items()))
        owner = signatures.setdefault(signature, case.template)
        assert owner == case.template, (
            f"{case.template} duplicates topology owned by {owner}"
        )


def test_simple_v2_has_one_campaign_case_per_template() -> None:
    owners: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for case in cases(score_version="simple-v2"):
        previous = owners.setdefault(case.template, case.case_id)
        if previous != case.case_id:
            duplicates.append((case.template, previous, case.case_id))

    assert duplicates == []


def test_catalog_has_no_duplicate_physical_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    seen: dict[object, str] = {}
    for case in cases():
        for analysis in case.analyses:
            candidate, _ = render_case_decks(
                case,
                analysis,
                BENCH["TSMC12"],
                CORNERS["nominal"],
                baked_lib=Path("/frozen/tsmc12.lib"),
            )
            cards = tuple(
                line.lower() for line in candidate.splitlines()
                if line.lower().startswith((".op", ".dc", ".tran", ".ac"))
            )
            key = (
                tuple(sorted(topology_signature(candidate).items())),
                cards,
                analysis.signals,
                analysis.metric_profile,
            )
            identity = f"{case.case_id}/{analysis.name}"
            assert key not in seen, f"{identity} duplicates {seen.get(key)}"
            seen[key] = identity


def test_diode_load_high_impedance_analysis_owns_line_sensitivity() -> None:
    case = next(item for item in cases() if item.case_id == "diode_load")
    analysis = next(item for item in case.analyses if item.name == "load_high")

    assert analysis.metric_profile == "mos_reference"
    assert {"vref_error_v", "line_sensitivity_error_pct"} <= set(
        case.required_metrics
    )


@pytest.mark.parametrize(
    ("case_id", "tier"),
    (
        ("sram_snm", "L2_stages"),
        ("transmission_gate_dc", "L2_stages"),
        ("transmission_gate_hold", "L3_blocks"),
    ),
)
def test_cases_are_tiered_by_the_crutches_the_deck_supplies(
    case_id: str,
    tier: str,
) -> None:
    case = next(item for item in cases() if item.case_id == case_id)

    assert case.tier == tier
    assert (Path("circuit_templates") / tier / case.template).is_file()


def test_passive_control_is_outside_the_model_ladder() -> None:
    assert control_deck("rc_lowpass.spice.tmpl").parent.name == "controls"
    with pytest.raises(FileNotFoundError):
        template_deck("rc_lowpass.spice.tmpl")


def test_inverter_is_a_coupled_device_stage() -> None:
    assert template_deck("inverter.spice.tmpl").parent.name == "L2_stages"


@pytest.mark.parametrize(
    ("case_id", "analysis_name", "corner_name", "expected"),
    (
        ("source_follower", "nmos", "pn_n3p2", True),
        ("source_follower", "nmos", "pn_n2p3", False),
        ("source_follower", "pmos", "pn_n3p2", False),
        ("source_follower", "pmos", "pn_n2p3", True),
        ("diffpair_active", "steering", "pn_n2p3", False),
        ("opamp_rejection", "differential_ac", "body_reverse", False),
        ("current_mirror", "nmos", "temp_hot", True),
    ),
)
def test_corner_applicability_tracks_the_observed_device_roles(
    case_id: str,
    analysis_name: str,
    corner_name: str,
    expected: bool,
) -> None:
    case = next(item for item in cases() if item.case_id == case_id)
    analysis = next(item for item in case.analyses if item.name == analysis_name)

    assert analysis_applies_to_corner(
        case,
        analysis,
        BENCH["TSMC12"],
        CORNERS[corner_name],
    ) is expected


def test_case_matrix_omits_analysis_level_noop_rows() -> None:
    case = next(item for item in cases() if item.case_id == "source_follower")

    selected = applicable_analyses(
        case,
        BENCH["TSMC12"],
        CORNERS["pn_n2p3"],
    )

    assert [analysis.name for analysis in selected] == ["pmos"]


@pytest.mark.parametrize(
    ("device", "corner", "expected"),
    (
        ("nmos", "pn_n3p2", True),
        ("nmos", "pn_n2p3", False),
        ("pmos", "pn_n3p2", False),
        ("pmos", "pn_n2p3", True),
        ("nmos", "lp_16", False),
        ("pmos", "ln_20", False),
        ("nmos", "body_reverse", False),
        ("pmos", "body_reverse", False),
    ),
)
def test_device_integrity_corner_matrix_omits_noop_rows(
    device: str,
    corner: str,
    expected: bool,
) -> None:
    from tests.common.device_integrity import device_corner_applies

    assert device_corner_applies(
        BENCH["TSMC12"], device, CORNERS[corner]
    ) is expected


def test_device_integrity_skips_fully_unsupported_corner_before_applying_it(
    tmp_path: Path,
) -> None:
    from tests.common.device_integrity import run_device_suites

    assert run_device_suites(
        BENCH["TSMC7"],
        CORNERS["vt_alternate"],
        tmp_path,
        level=75,
    ) == []


def test_terminal_integrity_sweeps_the_same_corner_matrix_as_device_integrity(
    tmp_path: Path,
) -> None:
    """The charge surface must be reachable at the corners the current one is.

    Currents and the 4x4 transcapacitance matrix were nominal-only through
    V7.6.8, so a full-terminal family could be exact at 27 C and wrong at
    125 C with no row to say so. The two single-device gates now share one
    applicability rule, and a corner that is a no-op for a polarity still
    creates no denominator row.
    """
    from tests.common.device_integrity import device_corner_applies
    from tests.common.terminal_integrity import (
        run_terminal_integrity, terminal_corner_applies,
    )

    for device in ("nmos", "pmos"):
        for corner in CORNERS.values():
            assert terminal_corner_applies(
                BENCH["TSMC12"], device, corner,
            ) is device_corner_applies(BENCH["TSMC12"], device, corner)

    # TSMC7 has no alternate VT pair, so the corner cannot be applied at all
    # and must produce nothing rather than an unlabelled nominal repeat.
    assert run_terminal_integrity(
        BENCH["TSMC7"], ["nmos", "pmos"], tmp_path,
        RunSpec(model_level=75, model_family="DirectNet-Full"),
        CORNERS["vt_alternate"],
    ) == []


def test_terminal_integrity_rows_carry_the_requested_corner_label() -> None:
    """A stressed row that still reported ``nominal`` would corrupt collection."""
    import inspect

    from tests.common import terminal_integrity

    source = inspect.getsource(terminal_integrity)
    assert 'corner="nominal"' not in source
    assert source.count("corner=corner.name") == 4


def test_declared_nn_stress_matrix_includes_vt_and_independent_geometry() -> None:
    assert {
        "vt_alternate", "vt_asymmetric", "ln_20", "lp_16", "nfin_high",
    } <= set(CORNERS)

    from tests.common.simple_circuit_harness import apply_corner

    base = BENCH["TSMC12"]
    alternate = apply_corner(base, CORNERS["vt_alternate"])
    asymmetric = apply_corner(base, CORNERS["vt_asymmetric"])
    assert alternate.effective_nmos_vt == alternate.effective_pmos_vt == "lvt"
    assert asymmetric.effective_nmos_vt == "svt"
    assert asymmetric.effective_pmos_vt == "lvt"
    assert apply_corner(base, CORNERS["ln_20"]).l_nmos == pytest.approx(20e-9)
    assert apply_corner(base, CORNERS["lp_16"]).l_pmos == pytest.approx(16e-9)
    high = apply_corner(base, CORNERS["nfin_high"])
    assert (high.nfin, high.effective_nfin_p) == (5, 5)


def test_device_integrity_is_part_of_campaign_coverage() -> None:
    from scripts.v710_regate_jobs import DEVICE_SUITES
    from scripts.v730_coverage import NON_SIMPLE_SUITES

    assert "verify_device_integrity" in DEVICE_SUITES
    assert "verify_device_integrity" in NON_SIMPLE_SUITES


def test_collector_rejects_incomplete_catalog_result_markers(
    tmp_path: Path,
) -> None:
    from scripts.v710_regate_collect import collect, is_verdict

    suite = "verify_circuit_topologies__current_mirror"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    marker = GateResult(
        case_id="current_mirror",
        tech="TSMC12",
        corner="nominal",
        analysis="nmos",
        role="diagnostic",
        status="diagnostic",
        metrics={"mre_pct": 0.0, "r2": 1.0,
                 "nrmse_pct": 0.0, "max_err": 0.0},
        model_family="DirectNet-Full",
        model_level=75,
        checkpoint_pins={
            "nmos": "tsmc12_dnf_large_nmos",
            "pmos": "tsmc12_dnf_large_pmos",
        },
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )
    log.write_text(marker.marker() + "\n===V710_DONE rc=0===\n")

    entry = collect(tmp_path)["dnf"]["large"][suite]["TSMC12"]["omp1"]

    assert entry["result_complete"] is False
    assert "missing result markers" in entry["error"]
    assert not is_verdict(entry)


def test_known_structured_suite_cannot_complete_without_markers(
    tmp_path: Path,
) -> None:
    from scripts import v730_coverage as coverage
    from scripts.v710_regate_collect import collect, is_verdict

    suite = "verify_circuit_topologies__inverter_chain"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    log.write_text("===V710_DONE rc=0===\n")

    entry = collect(tmp_path)["dnf"]["large"][suite]["TSMC12"]["omp1"]
    scanned = coverage.scan_logs(tmp_path)

    assert entry["result_complete"] is False
    assert "no result markers" in entry["error"]
    assert not is_verdict(entry)
    assert next(iter(scanned.values())) is None


def test_collector_rejects_structured_rows_without_provenance(
    tmp_path: Path,
) -> None:
    from scripts.v710_regate_collect import collect, is_verdict

    suite = "verify_circuit_topologies__inverter_chain"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    marker = GateResult(
        case_id="inverter_chain",
        tech="TSMC12",
        corner="nominal",
        analysis="fo4",
        role="diagnostic",
        status="diagnostic",
        metrics={
            "mre_pct": 0.0,
            "r2": 1.0,
            "nrmse_pct": 0.0,
            "max_err": 0.0,
        },
    )
    log.write_text(marker.marker() + "\n===V710_DONE rc=0===\n")

    entry = collect(tmp_path)["dnf"]["large"][suite]["TSMC12"]["omp1"]

    assert entry["result_complete"] is False
    assert "provenance" in entry["error"]
    assert not is_verdict(entry)


def test_collector_matches_each_row_to_the_campaign_manifest(
    tmp_path: Path,
) -> None:
    from scripts import v730_coverage as coverage
    from scripts.v710_regate_collect import collect, is_verdict

    manifest = tmp_path / "campaign_manifest.json"
    manifest.write_text('{"campaign":"test"}\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    suite = "verify_circuit_topologies__inverter_chain"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    marker = GateResult(
        case_id="inverter_chain",
        tech="TSMC12",
        corner="nominal",
        analysis="fo4",
        role="diagnostic",
        status="diagnostic",
        metrics={
            "mre_pct": 0.0,
            "r2": 1.0,
            "nrmse_pct": 0.0,
            "max_err": 0.0,
            "delay_error_pct": 0.0,
            "rise_fall_error_pct": 0.0,
            "amplitude_error_pct": 0.0,
            "phase_aligned_nrmse_pct": 0.0,
        },
        model_family="DirectNet-Full",
        model_level=75,
        checkpoint_pins={
            "nmos": "tsmc12_dnf_large_nmos",
            "pmos": "tsmc12_dnf_large_pmos",
        },
        campaign_manifest_sha256="",
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )
    log.write_text(
        f"===V710_PROVENANCE sha256={digest}===\n"
        + marker.marker()
        + "\n===V710_DONE rc=0===\n"
    )

    entry = collect(tmp_path, require_manifest=True)["dnf"]["large"][suite][
        "TSMC12"
    ]["omp1"]
    scanned = coverage.scan_logs(tmp_path)

    assert entry["result_complete"] is False
    assert "campaign provenance mismatch" in entry["error"]
    assert not is_verdict(entry)
    assert next(iter(scanned.values())) is None


def test_infrastructure_exit_is_not_a_scientific_verdict() -> None:
    from scripts.v710_regate_collect import is_verdict

    assert not is_verdict({"rc": "2"})


def test_parametric_circuit_sweep_cannot_exit_green_with_error_rows() -> None:
    from tests.common.circuit_sweep import exit_code

    results = [
        {"status": "pass", "config": SimpleNamespace(tech_key="TSMC12")},
        {"status": "error", "config": SimpleNamespace(tech_key="TSMC12")},
    ]

    assert exit_code(results, ["TSMC12"]) == 2


def test_collector_rejects_completion_code_that_disagrees_with_rows(
    tmp_path: Path,
) -> None:
    from scripts.v710_regate_collect import collect, is_verdict

    suite = "verify_circuit_topologies__inverter_chain"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    marker = GateResult(
        case_id="inverter_chain",
        tech="TSMC12",
        corner="nominal",
        analysis="fo4",
        role="diagnostic",
        status="diagnostic",
        metrics={"mre_pct": 0.0, "r2": 1.0,
                 "nrmse_pct": 0.0, "max_err": 0.0},
        model_family="DirectNet-Full",
        model_level=75,
        checkpoint_pins={
            "nmos": "tsmc12_dnf_large_nmos",
            "pmos": "tsmc12_dnf_large_pmos",
        },
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )
    log.write_text(marker.marker() + "\n===V710_DONE rc=1===\n")

    entry = collect(tmp_path)["dnf"]["large"][suite]["TSMC12"]["omp1"]

    assert entry["result_complete"] is False
    assert "completion code" in entry["error"]
    assert not is_verdict(entry)


def test_collector_rejects_characterized_row_without_numeric_evidence(
    tmp_path: Path,
) -> None:
    from scripts.v710_regate_collect import collect, is_verdict

    suite = "verify_circuit_topologies__inverter_chain"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    marker = GateResult(
        case_id="inverter_chain",
        tech="TSMC12",
        corner="nominal",
        analysis="fo4",
        role="diagnostic",
        status="diagnostic",
        model_family="DirectNet-Full",
        model_level=75,
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )
    log.write_text(marker.marker() + "\n===V710_DONE rc=0===\n")

    entry = collect(tmp_path)["dnf"]["large"][suite]["TSMC12"]["omp1"]

    assert entry["result_complete"] is False
    assert "numeric evidence" in entry["error"]
    assert not is_verdict(entry)


def test_collector_rejects_model_family_that_disagrees_with_campaign_tag(
    tmp_path: Path,
) -> None:
    from scripts.v710_regate_collect import collect, is_verdict

    suite = "verify_circuit_topologies__inverter_chain"
    log = tmp_path / "dnf" / "large" / "tsmc12" / f"{suite}.omp1.log"
    log.parent.mkdir(parents=True)
    marker = GateResult(
        case_id="inverter_chain",
        tech="TSMC12",
        corner="nominal",
        analysis="fo4",
        role="diagnostic",
        status="diagnostic",
        metrics={"mre_pct": 0.0, "r2": 1.0,
                 "nrmse_pct": 0.0, "max_err": 0.0},
        model_family="BSIM-AR-Full",
        model_level=76,
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )
    log.write_text(marker.marker() + "\n===V710_DONE rc=0===\n")

    entry = collect(tmp_path)["dnf"]["large"][suite]["TSMC12"]["omp1"]

    assert entry["result_complete"] is False
    assert "campaign tag" in entry["error"]
    assert not is_verdict(entry)


def test_structured_contract_binds_role_metrics_pins_and_threads() -> None:
    from scripts.v710_regate_collect import structured_contract_error

    base = GateResult(
        case_id="inverter_chain",
        tech="TSMC12",
        corner="nominal",
        analysis="fo4",
        role="diagnostic",
        status="diagnostic",
        metrics={
            "mre_pct": 0.0,
            "r2": 1.0,
            "nrmse_pct": 0.0,
            "max_err": 0.0,
            "delay_error_pct": 0.0,
            "rise_fall_error_pct": 0.0,
            "amplitude_error_pct": 0.0,
            "phase_aligned_nrmse_pct": 0.0,
        },
        model_family="DirectNet-Full",
        model_level=75,
        checkpoint_pins={
            "nmos": "tsmc12_dnf_large_nmos",
            "pmos": "tsmc12_dnf_large_pmos",
        },
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    ).payload()
    kwargs = {
        "campaign_tag": "dnf",
        "campaign_variant": "large",
        "expected_omp": 1,
    }

    wrong_role = {**base, "role": "qualification", "status": "pass"}
    assert "role" in structured_contract_error(
        "verify_circuit_topologies__inverter_chain",
        "TSMC12",
        [wrong_role],
        **kwargs,
    )

    wrong_metrics = {**base, "metrics": {"garbage": 1.0}}
    assert "required metrics" in structured_contract_error(
        "verify_circuit_topologies__inverter_chain",
        "TSMC12",
        [wrong_metrics],
        **kwargs,
    )

    wrong_threads = {
        **base,
        "thread_settings": {"omp": 2, "mkl": 2, "torch": 2},
    }
    assert "thread" in structured_contract_error(
        "verify_circuit_topologies__inverter_chain",
        "TSMC12",
        [wrong_threads],
        **kwargs,
    )

    wrong_pins = {**base, "checkpoint_pins": {}}
    assert "checkpoint" in structured_contract_error(
        "verify_circuit_topologies__inverter_chain",
        "TSMC12",
        [wrong_pins],
        **kwargs,
    )


def test_structured_contract_rejects_nonfinite_derived_metric() -> None:
    from scripts.v710_regate_collect import structured_contract_error

    derived = GateResult(
        case_id="diffpair_ideal",
        tech="TSMC12",
        corner="nominal",
        analysis="derived",
        role="diagnostic",
        status="diagnostic",
        domain={"cmrr_db_error": None},
        model_family="DirectNet-Full",
        model_level=75,
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    ).payload()

    assert "derived metric" in structured_contract_error(
        "verify_circuit_topologies__diffpair_ideal",
        "TSMC12",
        [derived],
    )


def test_cascode_ac_scores_each_polarity_and_headlines_the_worse() -> None:
    """The complementary cascode's small-signal gain must reach the row.

    Until V7.6.9 ``cascode_ac`` was the only AC profile with an empty metric
    contract, so a cascode whose NMOS branch had lost half its gain reported
    the same trace NRMSE as one that had not. Averaging the two branches is
    equally wrong: NMOS and PMOS are separate NN checkpoints, so one healthy
    polarity must not cover for the other.
    """
    analysis = next(
        item
        for case in cases() if case.case_id == "cascode_stack"
        for item in case.analyses if item.name == "gain"
    )
    frequencies = np.asarray([1e3, 1e4, 1e5])
    reference = Trace(
        "frequency", frequencies,
        {"v(nac)": np.full(3, 20.0 + 0j), "v(pac)": np.full(3, 10.0 + 0j)},
        reference=True,
    )
    candidate = Trace(
        "frequency", frequencies,
        {"v(nac)": np.full(3, 10.0 + 0j), "v(pac)": np.full(3, 9.5 + 0j)},
    )

    _, domain = compare_traces(candidate, reference, analysis, vdd=0.8)

    assert domain["nmos_gain_error_pct"] == pytest.approx(50.0)
    assert domain["pmos_gain_error_pct"] == pytest.approx(5.0)
    assert domain["cascode_gain_worst_error_pct"] == pytest.approx(50.0)
    assert analysis.headline_metric == "cascode_gain_worst_error_pct"
    validate_analysis_metrics(analysis, {
        "mre_pct": 0.0, "r2": 1.0, "nrmse_pct": 0.0, "max_err": 0.0,
    }, domain)


def test_catalog_declares_domain_headline_metrics() -> None:
    expected = {
        ("ring_osc_supply", "oscillation"): "period_error_pct",
        ("sram6t_modes", "write_margin"): "write_trip_error_v",
        ("ldo_regulator", "load_step"): "load_droop_error_v",
        ("current_mirror", "nmos_iref"): "iref_worst_ratio_error_pct",
    }
    observed = {
        (case.case_id, analysis.name): analysis.headline_metric
        for case in cases()
        for analysis in case.analyses
        if (case.case_id, analysis.name) in expected
    }

    assert observed == expected


def test_run_case_derives_exact_cmrr_and_psrr_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejection ratios use 20*log10 and preserve candidate/reference error."""
    import tests.common.simple_circuit_harness as harness

    case = next(item for item in cases() if item.case_id == "opamp_rejection")
    gains = {
        "differential_ac": ("diff", 100.0, 200.0),
        "common_mode_ac": ("cm", 1.0, 4.0),
        "supply_ac": ("supply", 0.1, 0.4),
    }

    def _result(
        _case: object,
        analysis: AnalysisSpec,
        *_args: object,
        **_kwargs: object,
    ) -> GateResult:
        stem, test_gain, reference_gain = gains[analysis.name]
        return GateResult(
            case_id=case.case_id,
            tech="TSMC12",
            corner="nominal",
            analysis=analysis.name,
            role="diagnostic",
            status="diagnostic",
            metrics={
                "mre_pct": 0.0,
                "r2": 1.0,
                "nrmse_pct": 0.0,
                "max_err": 0.0,
            },
            domain={
                f"{stem}_gain_test": test_gain,
                f"{stem}_gain_ref": reference_gain,
                f"{stem}_gain_error_pct": 0.0,
            },
            model_family="DirectNet-Full",
            model_level=75,
            thread_settings={"omp": 1, "mkl": 1, "torch": 1},
        )

    monkeypatch.setattr(harness, "run_case_analysis", _result)

    results = harness.run_case(
        case,
        BENCH["TSMC12"],
        CORNERS["nominal"],
        tmp_path,
        run_spec=RunSpec(75, "DirectNet-Full"),
    )
    derived = next(result for result in results if result.analysis == "derived")

    assert derived.domain["cmrr_db_test"] == pytest.approx(40.0)
    assert derived.domain["cmrr_db_ref"] == pytest.approx(
        20.0 * np.log10(50.0)
    )
    assert derived.domain["cmrr_db_error"] == pytest.approx(
        40.0 - 20.0 * np.log10(50.0)
    )
    assert derived.domain["psrr_db_test"] == pytest.approx(60.0)
    assert derived.domain["psrr_db_ref"] == pytest.approx(
        20.0 * np.log10(500.0)
    )
    assert derived.domain["psrr_db_error"] == pytest.approx(
        60.0 - 20.0 * np.log10(500.0)
    )


@pytest.mark.parametrize(
    ("error_kind", "expected"),
    (
        ("candidate", 1),
        ("infrastructure", 2),
        ("reference", 2),
        ("result_schema", 2),
    ),
)
def test_result_exit_code_separates_scientific_and_infrastructure_errors(
    error_kind: str,
    expected: int,
) -> None:
    result = GateResult(
        case_id="case",
        tech="TSMC12",
        corner="nominal",
        analysis="analysis",
        role="diagnostic",
        status="error",
        error="failed",
        execution_state="error",
        error_kind=error_kind,
    )

    assert result_exit_code([result]) == expected


def test_terminal_current_metrics_report_kcl_and_each_terminal() -> None:
    from tests.common.terminal_integrity import (
        terminal_current_metrics,
        validate_sweep_lengths,
    )

    reference = {
        "d": np.asarray([-2.0, -3.0]),
        "g": np.asarray([0.2, 0.3]),
        "s": np.asarray([1.7, 2.5]),
        "b": np.asarray([0.1, 0.2]),
    }
    candidate = {name: values.copy() for name, values in reference.items()}
    candidate["g"] += 0.1

    metrics, domain = terminal_current_metrics(candidate, reference)

    assert metrics["max_err"] == pytest.approx(0.1)
    assert domain["reference_kcl_max_a"] == pytest.approx(0.0)
    assert domain["candidate_kcl_max_a"] == pytest.approx(0.1)
    assert domain["gate_current_max_error_a"] == pytest.approx(0.1)
    validate_sweep_lengths(candidate, reference, expected_points=3)


def test_terminal_sweep_rejects_more_than_one_missing_endpoint() -> None:
    from tests.common.terminal_integrity import validate_sweep_lengths

    rows = {terminal: np.zeros(9) for terminal in ("d", "g", "s", "b")}
    with pytest.raises(ValueError, match="incomplete"):
        validate_sweep_lengths(rows, rows, expected_points=11)


def test_ac_terminal_currents_form_the_transcapacitance_matrix() -> None:
    from tests.common.terminal_integrity import capacitance_from_admittance

    capacitance = np.asarray([
        [3.0, -1.0, -1.0, -1.0],
        [-1.0, 3.0, -1.0, -1.0],
        [-1.0, -1.0, 3.0, -1.0],
        [-1.0, -1.0, -1.0, 3.0],
    ]) * 1e-15
    frequency = 1e6
    admittance = 1j * 2.0 * np.pi * frequency * capacitance

    observed = capacitance_from_admittance(admittance, frequency)

    np.testing.assert_allclose(observed, capacitance, atol=1e-30)
    np.testing.assert_allclose(observed.sum(axis=0), 0.0, atol=1e-30)
    np.testing.assert_allclose(observed.sum(axis=1), 0.0, atol=1e-30)


def test_terminal_source_current_is_negated_to_device_admittance() -> None:
    from tests.common.terminal_integrity import (
        device_admittance_from_source_currents,
    )

    frequency = 1e6
    capacitance = 1e-12
    source_current = -1j * 2.0 * np.pi * frequency * capacitance

    observed = device_admittance_from_source_currents(
        np.asarray([source_current]),
    )

    assert observed[0].imag / (2.0 * np.pi * frequency) == pytest.approx(
        capacitance
    )


def test_terminal_renderer_declares_full_model_level() -> None:
    from tests.common.terminal_integrity import (
        render_terminal_sweep_decks,
        terminal_sweeps,
    )

    candidate, _reference, _card = render_terminal_sweep_decks(
        BENCH["TSMC12"],
        "nmos",
        terminal_sweeps(BENCH["TSMC12"], "nmos")[0],
        baked_lib=Path("/frozen/tsmc12.lib"),
        level=76,
    )

    assert "LEVEL=76" in candidate
    assert "FAMILY=" not in candidate


@pytest.mark.parametrize(
    ("case_id", "template", "analyses"),
    (
        (
            "common_source_nn",
            "common_source.spice.tmpl",
            {"nmos_fixed", "pmos_fixed", "nmos_floating", "pmos_floating"},
        ),
        (
            "inverter_energy",
            "inverter.spice.tmpl",
            {"vtc", "switching"},
        ),
    ),
)
def test_standalone_active_templates_are_catalog_experiments(
    case_id: str,
    template: str,
    analyses: set[str],
) -> None:
    case = next(item for item in cases() if item.case_id == case_id)

    assert case.template == template
    assert {analysis.name for analysis in case.analyses} == analyses


def test_common_source_bandwidth_analyses_render_a_measurable_load_pole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    case = next(item for item in cases() if item.case_id == "common_source_nn")

    for analysis in case.analyses:
        candidate, reference = render_case_decks(
            case,
            analysis,
            BENCH["TSMC12"],
            CORNERS["nominal"],
            baked_lib=Path("/frozen/tsmc12.lib"),
        )
        assert "Cload out 0 10f" in candidate
        assert "Cload out 0 10f" in reference


def test_sram_exercises_both_storage_states_and_write_directions() -> None:
    case = next(item for item in cases() if item.case_id == "sram6t_modes")

    assert {analysis.name for analysis in case.analyses} >= {
        "hold", "hold_state0", "read", "read_state0",
        "write", "write_state0", "write_margin", "write_margin_state0",
    }


def test_l4_systems_include_closed_loop_small_signal_diagnostics() -> None:
    expected = {
        "unity_gain_buffer": {"closed_loop_ac"},
        "ota_5t_buffer": {"closed_loop_ac"},
        "ldo_regulator": {"supply_ac", "output_impedance_ac"},
    }

    for case_id, names in expected.items():
        case = next(item for item in cases() if item.case_id == case_id)
        assert {analysis.name for analysis in case.analyses} >= names


def test_active_load_diffpair_bridges_both_input_polarities() -> None:
    case = next(item for item in cases() if item.case_id == "diffpair_active_load")

    assert case.tier == "L3_blocks"
    assert {analysis.name for analysis in case.analyses} == {
        "nmos_steering", "pmos_steering", "nmos_ac", "pmos_ac",
    }
    assert template_deck(case.template, tier=case.tier).is_file()


def test_active_load_topology_is_owned_by_its_template() -> None:
    case = next(item for item in cases() if item.case_id == "diffpair_active_load")
    text = template_deck(case.template, tier=case.tier).read_text()

    assert "<ACTIVE_LOAD_STAGE>" not in text
    assert "<INPUT_PREFIX>in_l" in text
    assert "<LOAD_PREFIX>load_d" in text


def test_active_load_template_receives_analysis_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    case = next(item for item in cases() if item.case_id == "diffpair_active_load")
    analysis = next(item for item in case.analyses if item.name == "nmos_ac")

    candidate, _ = render_case_decks(
        case,
        analysis,
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/roles.lib"),
    )

    assert "Vinp inp 0 DC=0.44 AC=1 0" in candidate
    assert "Vinn inn 0 DC=0.44 AC=0 0" in candidate


def test_bias_fanout_provides_a_real_device_count_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    fanout_cases = [
        case for case in cases()
        if case.case_id.startswith("bias_tree_fanout_")
    ]
    counts = []
    for case in fanout_cases:
        template = template_deck(case.template, tier=case.tier).read_text()
        assert "<BIAS_BRANCHES>" not in template
        candidate, _ = render_case_decks(
            case,
            case.analyses[0],
            BENCH["TSMC12"],
            CORNERS["nominal"],
            baked_lib=Path("/frozen/tsmc12.lib"),
        )
        counts.append(sum(
            count for item, count in connectivity_signature(candidate).items()
            if item[0] == "mos"
        ))

    assert [case.case_id for case in fanout_cases] == [
        "bias_tree_fanout_3t",
        "bias_tree_fanout_5t",
        "bias_tree_fanout_9t",
        "bias_tree_fanout_17t",
    ]
    assert counts == [3, 5, 9, 17]


def test_large_feedback_proxy_exceeds_the_old_ten_device_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    case = next(
        item for item in cases() if item.case_id == "multistage_buffer_12t"
    )
    candidate, _ = render_case_decks(
        case,
        case.analyses[0],
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/tsmc12.lib"),
    )
    count = sum(
        value for item, value in connectivity_signature(candidate).items()
        if item[0] == "mos"
    )

    assert case.tier == "L4_systems"
    assert count == 12
    assert {analysis.kind for analysis in case.analyses} == {"tran", "ac"}


def test_generated_cascode_bias_is_exercised_for_both_polarities() -> None:
    """An NMOS-only self-bias ladder cannot expose a PMOS model asymmetry."""
    observed = {
        tuple(case.device_kinds): case.analyses[0].signals[0]
        for case in cases()
        if case.case_id.startswith("self_biased_cascode")
    }

    assert observed == {
        ("nmos",): "i(Voutn)",
        ("pmos",): "i(Voutp)",
    }


def test_pmos_generated_cascode_bias_uses_a_passive_load_line() -> None:
    """An ideal current can force a two-diode PMOS rail below ground."""
    case = next(
        item for item in cases()
        if item.case_id == "self_biased_cascode_pmos"
    )
    candidate, reference = render_case_decks(
        case,
        case.analyses[0],
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/tsmc12.lib"),
    )

    assert "Rrefp pc 0 100k" in candidate
    assert "Rrefp pc 0 100k" in reference
    assert "Irefp" not in candidate


def test_named_device_roles_render_independent_geometry_and_reference_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    case = next(item for item in cases() if item.case_id == "diffpair_active_load")
    candidate, reference = render_case_decks(
        case,
        case.analyses[0],
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/roles.lib"),
    )
    p_candidate, p_reference = render_case_decks(
        case,
        case.analyses[1],
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/roles.lib"),
    )

    assert "n_input_nn L=16n NFIN=3" in candidate
    assert "p_input_nn L=20n NFIN=3" in p_candidate
    assert "n_load_nn L=16n NFIN=2" in p_candidate
    assert "p_load_nn L=20n NFIN=2" in candidate
    assert "v768_diffpair_active_load_n_input" in reference
    assert "v768_diffpair_active_load_p_input" in p_reference
    assert physical_deck_mismatch(
        candidate,
        reference,
        case.analyses[0],
        BENCH["TSMC12"],
        baked_lib=Path("/frozen/roles.lib"),
        case=case,
    ) == ""


def test_named_device_roles_build_distinct_osdi_models(tmp_path: Path) -> None:
    case = next(item for item in cases() if item.case_id == "diffpair_active_load")

    library = get_case_baked_modelcard(case, BENCH["TSMC12"], tmp_path)
    text = library.read_text().lower()

    for role in ("n_input", "p_input", "n_load", "p_load"):
        assert f".model v768_diffpair_active_load_{role} " in text


def test_nn_hierarchy_renderer_uses_selected_family_and_instance_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.simple_circuits.verify_nn_subckt import render_candidate_pair

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "76")
    spec = RunSpec.from_environment()
    flat, hierarchical = render_candidate_pair(BENCH["TSMC12"], spec)

    assert "LEVEL=76" in flat and "LEVEL=76" in hierarchical
    assert "TECH=tsmc12 VT=svt" in flat
    assert "Xbuf in out vdd buf" in hierarchical
    assert "NFIN=NFP" in hierarchical and "NFIN=NFN" in hierarchical


def test_nn_hierarchy_gate_covers_dc_transient_and_ac() -> None:
    """Flattening can be correct in one solver path and broken in another."""
    from tests.simple_circuits.verify_nn_subckt import SUBCKT_ANALYSES

    assert {analysis.kind for analysis in SUBCKT_ANALYSES} == {
        "dc", "tran", "ac",
    }
    assert all(analysis.signals == ("v(out)", "v(mid)")
               for analysis in SUBCKT_ANALYSES)


@pytest.mark.parametrize(
    ("failure", "execution_state", "error_kind"),
    (
        (KeyError("candidate results carry no v(out)"),
         "infrastructure_error", "infrastructure"),
        (CandidateConvergenceError(
            "candidate AC operating point did not converge"),
         "nonconverged", "candidate"),
        (CandidateSupportError("outside certified support"),
         "error", "candidate"),
    ),
)
def test_nn_hierarchy_classifies_only_convergence_as_a_scientific_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    execution_state: str,
    error_kind: str,
) -> None:
    import tests.simple_circuits.verify_nn_subckt as subckt

    analysis = subckt.SUBCKT_ANALYSES[0]
    reference = Trace(
        "sweep",
        np.asarray([0.0, 0.8]),
        {"v(out)": np.asarray([0.0, 0.8]),
         "v(mid)": np.asarray([0.8, 0.0])},
        reference=True,
    )
    monkeypatch.setattr(subckt, "get_baked_modelcard",
                        lambda *_args, **_kwargs: tmp_path / "baked.lib")
    monkeypatch.setattr(subckt, "render_candidate_pair",
                        lambda *_args, **_kwargs: ("flat", "hierarchical"))
    monkeypatch.setattr(subckt, "render_reference",
                        lambda *_args, **_kwargs: "reference")
    monkeypatch.setattr(subckt, "physical_deck_mismatch",
                        lambda *_args, **_kwargs: "")
    monkeypatch.setattr(subckt, "parse_netlist", lambda *_args: object())
    monkeypatch.setattr(subckt, "flattened_candidate_mismatch",
                        lambda *_args: "")
    monkeypatch.setattr(subckt, "run_reference_trace",
                        lambda *_args, **_kwargs: reference)

    def _fail_candidate(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(subckt, "run_candidate_trace", _fail_candidate)

    result = subckt.run_nn_subckt_analysis(
        BENCH["TSMC12"], analysis, tmp_path, RunSpec(75, "DirectNet-Full"),
    )

    assert result.execution_state == execution_state
    assert result.error_kind == error_kind


def test_level72_control_renders_the_same_physical_experiment() -> None:
    case = next(item for item in cases() if item.case_id == "diode_load")
    analysis = case.analyses[1]
    resolved_analysis = replace(
        analysis,
        card=analysis.card.replace("<VDD>", f"{BENCH['TSMC12'].vdd:g}"),
    )
    baked = Path("/frozen/tsmc12.lib")
    candidate, reference = render_case_decks(
        case, analysis, BENCH["TSMC12"], CORNERS["nominal"], baked_lib=baked,
    )
    control = render_case_control_deck(
        case, analysis, BENCH["TSMC12"], CORNERS["nominal"], baked_lib=baked,
    )

    assert "LEVEL=72" in control
    assert "TECH=" not in control and "VT=" not in control
    assert physical_deck_mismatch(
        control,
        reference,
        resolved_analysis,
        BENCH["TSMC12"],
        baked_lib=baked,
        case=case,
        model_level=72,
        control=True,
    ) == ""
    assert topology_mismatch(candidate, reference) == ""

    wrong_geometry = control.replace("L=16n", "L=24n", 1)
    assert "geometry mismatch" in physical_deck_mismatch(
        wrong_geometry,
        reference,
        resolved_analysis,
        BENCH["TSMC12"],
        baked_lib=baked,
        case=case,
        model_level=72,
        control=True,
    )


def test_explicit_run_spec_owns_rendered_model_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    case = next(item for item in cases() if item.case_id == "diode_load")
    run_spec = RunSpec(76, "BSIM-AR-Full")

    candidate, _ = render_case_decks(
        case,
        case.analyses[1],
        BENCH["TSMC12"],
        CORNERS["nominal"],
        baked_lib=Path("/frozen/tsmc12.lib"),
        model_level=run_spec.model_level,
    )

    assert "LEVEL=76" in candidate
    assert "FAMILY=" not in candidate


def test_campaign_decks_render_the_selected_model_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.common.nn_sweep import active_nn_contract
    from tests.simple_circuits.verify_nn_ac import _directnet_deck

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")

    deck = _directnet_deck(
        BENCH["TSMC12"],
        "nmos",
        0.4,
        "ac dec 1 1e3 1e6",
    )

    assert "LEVEL=75" in deck
    assert "FAMILY=" not in deck
    assert active_nn_contract() == (75, "directnet_full")


def test_topology_cli_prints_derived_rows_without_trace_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.simple_circuits import verify_circuit_topologies as cli

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setattr(cli, "RESULTS_BASE", tmp_path)
    derived = GateResult(
        case_id="diffpair_ideal",
        tech="TSMC12",
        corner="nominal",
        analysis="derived",
        role="diagnostic",
        status="diagnostic",
        domain={"cmrr_db_error": 1.25},
        model_family="DirectNet-Full",
        model_level=75,
        checkpoint_pins={},
        campaign_manifest_sha256="",
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )
    monkeypatch.setattr(cli, "run_case", lambda *_args, **_kwargs: [derived])

    assert cli.main([
        "--case", "diffpair_ideal",
        "--tech", "TSMC12",
        "--corner", "nominal",
    ]) == 0
    assert "cmrr_db_error=1.25" in capsys.readouterr().out


def test_topology_cli_skips_unsupported_cells_without_dropping_valid_ones(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.simple_circuits import verify_circuit_topologies as cli

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setattr(cli, "RESULTS_BASE", tmp_path)

    def characterized(case: object, tech: object, corner: object, *_args: object,
                      **_kwargs: object) -> list[GateResult]:
        return [GateResult(
            case_id="diode_load",
            tech=getattr(tech, "name"),
            corner=getattr(corner, "name"),
            analysis="op_nominal",
            role="diagnostic",
            status="diagnostic",
            metrics={"mre_pct": 0.0, "r2": 1.0,
                     "nrmse_pct": 0.0, "max_err": 0.0},
            model_family="DirectNet-Full",
            model_level=75,
            thread_settings={"omp": 1, "mkl": 1, "torch": 1},
        )]

    monkeypatch.setattr(cli, "run_case", characterized)

    assert cli.main([
        "--case", "diode_load",
        "--tech", "TSMC5,TSMC7",
        "--corner", "vt_alternate",
    ]) == 0
    output = capsys.readouterr().out
    assert "diode_load / TSMC5 / vt_alternate" in output
    assert "diode_load / TSMC7 / vt_alternate NOT-APPLICABLE" in output


def test_simple_circuit_geometry_inventory_skips_unsupported_corners_and_roles(
) -> None:
    from tests.single_devices.verify_data_geometry_coverage import (
        _simple_circuit_geometries,
    )

    points = _simple_circuit_geometries()
    coordinates = {
        (point.tech, point.dev, point.vt, point.length, point.nfin)
        for point in points
    }

    assert ("tsmc12", "nmos", "svt", 16e-9, 4.0) in coordinates
    assert ("tsmc12", "nmos", "svt", 16e-9, 6.0) in coordinates
    assert not any(
        point.tech == "tsmc7" and "vt_alternate" in point.label
        for point in points
    )


def test_transient_op_convergence_guard_rejects_failed_retry() -> None:
    from tests.common.circuit_benchmarks import _require_converged_op

    solution = {"out": 0.4}
    assert _require_converged_op(
        SimpleNamespace(_last_solve_converged=True),
        solution,
        "DC OP fast path",
    ) is solution
    with pytest.raises(RuntimeError, match="DC OP GMIN retry did not converge"):
        _require_converged_op(
            SimpleNamespace(_last_solve_converged=False),
            solution,
            "DC OP GMIN retry",
        )


@pytest.mark.parametrize(
    ("module_name", "dimension_flag", "dimension"),
    [
        ("tests.single_devices.verify_nn_multi_tech_dc", "--device", "nmos"),
        ("tests.simple_circuits.verify_nn_multi_tech_tran", "--analysis", "vtc"),
    ],
)
def test_parametric_cli_banner_names_forced_family(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module_name: str,
    dimension_flag: str,
    dimension: str,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setattr(module, "run_nn_multi_tech", lambda *_args: [object()])
    monkeypatch.setattr(
        module,
        "print_nn_summary_table",
        lambda *_args, **_kwargs: {"pass": 1, "fail": 0, "error": 0},
    )
    monkeypatch.setattr(module, "save_nn_summary_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "plot_nn_summary_bar", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "sweep_gate_results", lambda *_args, **_kwargs: [
        GateResult(
            case_id="stub", tech="TSMC12", corner="nominal",
            analysis="stub", role="qualification", status="pass",
            metrics={"nrmse_pct": 0.0},
        ),
    ])

    assert module.main(["--tech", "TSMC12", dimension_flag, dimension]) == 0
    assert "DirectNet-Full (LEVEL=75)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("module_name", "argv"),
    [
        ("tests.single_devices.verify_nn_multi_tech_dc", ["--tech", ""]),
        (
            "tests.single_devices.verify_nn_multi_tech_dc",
            ["--tech", "TSMC12,TSMC12"],
        ),
        (
            "tests.simple_circuits.verify_nn_multi_tech_tran",
            ["--analysis", ""],
        ),
        (
            "tests.simple_circuits.verify_nn_multi_tech_tran",
            ["--analysis", "vtc,vtc"],
        ),
    ],
)
def test_parametric_clis_reject_empty_and_duplicate_axes(
    module_name: str,
    argv: list[str],
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc_info:
        module.main(argv)
    assert exc_info.value.code == 2


def test_flattened_hierarchy_parity_checks_resolved_device_geometry() -> None:
    from tests.simple_circuits.verify_nn_subckt import (
        flattened_candidate_mismatch,
    )

    class PMOS:
        def __init__(self, nodes: list[str], nfin: float = 2.0) -> None:
            self.nodes = nodes
            self.L = 20e-9
            self.NFIN = nfin
            self.m = 1.0
            self.temperature = 300.15

    flat = SimpleNamespace(
        circuit=SimpleNamespace(
            components=[PMOS(["mid", "in", "vdd", "vdd"])],
            initial_conditions={"mid": 0.8},
        ),
        analysis_type="tran",
        analysis_params={"tstep": 2e-12, "tstop": 4e-9, "uic": True},
        models={"pmos_nn": {"type": "PMOS", "params": {"LEVEL": 75.0}}},
        _temperature_kelvin=300.15,
    )
    hierarchical = SimpleNamespace(
        circuit=SimpleNamespace(
            components=[PMOS(["Xbuf.m", "in", "vdd", "vdd"])],
            initial_conditions={"Xbuf.m": 0.8},
        ),
        analysis_type=flat.analysis_type,
        analysis_params=flat.analysis_params,
        models=flat.models,
        _temperature_kelvin=flat._temperature_kelvin,
    )

    assert flattened_candidate_mismatch(flat, hierarchical) == ""
    hierarchical.circuit.components[0].NFIN = 3.0
    assert "resolved hierarchy mismatch" in flattened_candidate_mismatch(
        flat, hierarchical,
    )


def test_nn_ac_cli_emits_one_provenance_row_per_requested_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.common.gate_result import parse_result_markers
    from tests.simple_circuits import verify_nn_ac as cli

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setattr(cli, "RESULTS_BASE", tmp_path)
    monkeypatch.setattr(cli, "_print_result", lambda _result: None)

    def result(tech: str, device: str) -> dict[str, object]:
        return {
            "tech": tech,
            "device": device,
            "passed": True,
            "op_ok": True,
            "op_valid": True,
            "ng_vbias": 0.3,
            "nn_vbias": 0.31,
            "ng_vout": 0.4,
            "nn_vout": 0.4,
            "m": {
                "gain0_db_err": 0.1,
                "f3db_ratio": 1.0,
                "mag_nrmse": 0.01,
                "phase_maxerr_inband_deg": 2.0,
            },
            "aggregate": {
                "mre_pct": 1.0,
                "r2": 0.99,
                "nrmse_pct": 1.0,
                "max_err": 0.01,
            },
        }

    monkeypatch.setattr(cli, "eval_cs_amp", result)

    assert cli.main([
        "--tech", "TSMC12", "--device", "nmos,pmos",
    ]) == 0
    rows = parse_result_markers(capsys.readouterr().out)
    assert [(row["case_id"], row["analysis"]) for row in rows] == [
        ("nn_ac", "nmos"),
        ("nn_ac", "pmos"),
    ]
    assert {(row["model_family"], row["model_level"]) for row in rows} == {
        ("DirectNet-Full", 75),
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--tech", ""],
        ["--tech", "TSMC12,TSMC12"],
        ["--device", ""],
        ["--device", "nmos,nmos"],
    ],
)
def test_nn_ac_cli_rejects_empty_and_duplicate_axes(argv: list[str]) -> None:
    from tests.simple_circuits import verify_nn_ac as cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 2


def test_parametric_error_row_preserves_candidate_convergence_stage() -> None:
    from tests.common.nn_sweep import sweep_gate_results

    cfg = SimpleNamespace(
        tech_key="TSMC12",
        sweep_type="cload",
        label="TSMC12_tran_cload_100fF",
        swept={"cload_fF": 100},
    )
    rows = sweep_gate_results(
        [{
            "config": cfg,
            "passed": False,
            "error": "RuntimeError: transient did not converge",
            "reference_converged": True,
            "candidate_converged": False,
            "error_kind": "candidate",
        }],
        RunSpec(75, "DirectNet-Full"),
        case_id="nn_parametric_inverter",
        max_error_unit="mV",
    )

    assert len(rows) == 1
    assert rows[0].reference_converged is True
    assert rows[0].candidate_converged is False
    assert rows[0].execution_state == "nonconverged"
    assert rows[0].error_kind == "candidate"


def test_opamp_ac_cli_emits_provenance_bound_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.common.gate_result import parse_result_markers
    from tests.simple_circuits import verify_circuit_opamp_ac as cli

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setattr(cli, "_print_result", lambda _result: None)
    monkeypatch.setattr(cli, "eval_opamp_ac", lambda tech: {
        "tech": tech,
        "passed": True,
        "op_ok": True,
        "reference_op_valid": True,
        "nn_op_valid": True,
        "ng_trip": 0.4,
        "nn_trip": 0.41,
        "ng_vout": 0.4,
        "nn_vout": 0.4,
        "m": {
            "gain0_db_err": 0.1,
            "gbw_ratio": 1.0,
            "pm_err": 2.0,
            "mag_nrmse": 0.01,
        },
        "aggregate": {
            "mre_pct": 1.0,
            "r2": 0.99,
            "nrmse_pct": 1.0,
            "max_err": 0.01,
        },
    })

    assert cli.main(["--tech", "TSMC12"]) == 0
    rows = parse_result_markers(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "opamp_ac"
    assert rows[0]["analysis"] == "open_loop"
    assert rows[0]["model_family"] == "DirectNet-Full"


def test_ngspice_ac_adapter_rejects_a_truncated_reference_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.common import circuit_ac

    expected = np.geomspace(1e3, 1e6, 7)
    monkeypatch.setattr(
        circuit_ac,
        "run_ngspice_ac",
        lambda *_args, **_kwargs: (
            expected[:-1],
            np.ones(expected.size - 1, dtype=complex),
        ),
    )

    with pytest.raises(RuntimeError, match="frequency count"):
        circuit_ac.run_ngspice_ac_baked(
            ["V1 in 0 AC=1"],
            tmp_path,
            "truncated",
            "ac dec 2 1e3 1e6",
            "out",
            expected,
        )


def test_parametric_curve_validator_rejects_a_truncated_reference() -> None:
    from tests.common.nn_sweep import validate_curve_trace

    axis = np.arange(0.0, 0.8, 0.1)
    with pytest.raises(RuntimeError, match="stops at"):
        validate_curve_trace(
            axis,
            np.zeros_like(axis),
            expected_start=0.0,
            expected_stop=0.8,
            max_step=0.1,
            label="reference VTC",
        )


def test_legacy_circuit_sweep_keeps_configs_after_baseline_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.common import circuit_sweep

    baseline = circuit_sweep.make_baseline("ringosc", "TSMC12")
    corners = [
        replace(baseline, sweep_type="cload", config_name=f"corner-{index}")
        for index in (1, 2)
    ]
    called: list[str] = []

    def run(config: object, _work_dir: Path) -> dict[str, object]:
        called.append(config.config_name)
        if config.config_name == "baseline":
            return circuit_sweep._err(config, "baseline failed")
        return circuit_sweep._result(
            config,
            True,
            {"mre_pct": 0.0, "r2": 1.0,
             "nrmse_pct": 0.0, "max_err": 0.0},
            {"metric": "period_err%", "value": 0.0},
        )

    monkeypatch.setitem(circuit_sweep._RUNNERS, "ringosc", run)
    monkeypatch.setattr(
        circuit_sweep, "verify_checkpoint_pin", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        circuit_sweep, "build_parametric", lambda *_args: corners,
    )

    results = circuit_sweep.run_circuit_multi_tech(
        "ringosc", ["TSMC12"], "cload", tmp_path,
    )

    assert called == ["baseline", "corner-1", "corner-2"]
    assert [result["config"].config_name for result in results] == called


def test_legacy_circuit_sweep_rejects_partial_or_truncated_transient() -> None:
    from tests.common.circuit_sweep import validate_transient_trace

    axis = np.arange(0.0, 1.0e-9, 0.1e-9)
    with pytest.raises(RuntimeError, match="partial transient"):
        validate_transient_trace(
            axis,
            np.zeros_like(axis),
            expected_stop=1.0e-9,
            max_step=0.1e-9,
            label="candidate ring",
            partial=True,
        )
    with pytest.raises(RuntimeError, match="requested stop"):
        validate_transient_trace(
            axis[:-1],
            np.zeros_like(axis[:-1]),
            expected_stop=1.0e-9,
            max_step=0.1e-9,
            label="reference switchcap",
        )


def test_legacy_circuit_sweep_rejects_bad_dc_axes() -> None:
    from tests.common.circuit_sweep import validate_dc_trace

    with pytest.raises(RuntimeError, match="expected at least"):
        validate_dc_trace(
            np.array([0.0, 0.1, 0.2]),
            np.zeros(3),
            expected_start=0.0,
            expected_stop=0.5,
            max_step=0.1,
            label="SRAM reference",
        )
    with pytest.raises(RuntimeError, match="strictly monotonic"):
        validate_dc_trace(
            np.array([0.0, 0.1, 0.05, 0.2, 0.3, 0.4, 0.5]),
            np.zeros(7),
            expected_start=0.0,
            expected_stop=0.5,
            max_step=0.1,
            label="opamp candidate",
        )
    with pytest.raises(RuntimeError, match="expected at least"):
        validate_dc_trace(
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.zeros(4),
            expected_start=0.0,
            expected_stop=0.5,
            max_step=0.1,
            label="both endpoints missing",
        )


def test_legacy_circuit_sweep_rejects_duplicate_technologies(
    tmp_path: Path,
) -> None:
    from tests.common.circuit_sweep import run_circuit_multi_tech

    with pytest.raises(ValueError, match="unique"):
        run_circuit_multi_tech(
            "ringosc", ["TSMC12", "TSMC12"], "cload", tmp_path,
        )


def test_complete_nonoscillating_ring_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.common import circuit_sweep

    cfg = circuit_sweep.make_baseline("ringosc", "TSMC12")
    p = cfg.stim
    time = np.arange(0.0, p.tstop + p.tstep * 0.5, p.tstep)
    values = np.zeros_like(time)
    periods = iter([0.05e-9, float("nan")])
    monkeypatch.setattr(
        circuit_sweep, "get_baked_modelcard", lambda *_args, **_kwargs: tmp_path / "x",
    )
    monkeypatch.setattr(circuit_sweep, "ngspice_ringosc", lambda *_args: {
        "body": "", "signals": "v(n5)", "analysis": "tran 2p 1.2n uic",
    })
    monkeypatch.setattr(
        circuit_sweep,
        "run_ngspice_wrdata",
        lambda *_args, **_kwargs: np.column_stack((time, values)),
    )
    monkeypatch.setattr(circuit_sweep, "period_from_wave", lambda *_args: next(periods))
    monkeypatch.setattr(circuit_sweep, "directnet_ringosc", lambda *_args: "* deck\n")
    monkeypatch.setattr(
        circuit_sweep,
        "run_directnet_transient",
        lambda *_args: ({"time": time, "n5": values}, False, ""),
    )

    result = circuit_sweep.run_single_ringosc(cfg, tmp_path)

    assert result["status"] == "error"
    assert "did not oscillate" in result["error"]
    assert np.isnan(result["nrmse_pct"])


def test_near_zero_reference_snm_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.common import circuit_sweep

    cfg = circuit_sweep.make_baseline("sram", "TSMC12")
    axis = np.arange(0.0, cfg.bt.vdd + 0.0025, cfg.stim.dc_step)
    monkeypatch.setattr(
        circuit_sweep, "get_baked_modelcard", lambda *_args, **_kwargs: tmp_path / "x",
    )
    monkeypatch.setattr(circuit_sweep, "ngspice_sram_lobe", lambda *_args: {
        "body": "", "signals": "v(qb)", "analysis": "dc Vq 0 0.8 0.005",
    })
    monkeypatch.setattr(
        circuit_sweep,
        "run_ngspice_wrdata",
        lambda *_args, **_kwargs: np.column_stack((axis, axis)),
    )
    monkeypatch.setattr(circuit_sweep, "snm_from_lobes", lambda *_args: 0.0)

    result = circuit_sweep.run_single_sram(cfg, tmp_path)

    assert result["status"] == "error"
    assert "reference SNM is not measurable" in result["error"]
