"""Contracts for the non-compact-model core devices and solver knobs.

These questions were owned by five V7.5 NGSPICE gates — ``verify_inductor``,
``verify_current_source_ngspice``, ``verify_cmg_set_temperature``,
``verify_tran_branch_current`` and ``verify_tran_gear2`` — that were deleted
without a replacement. The V7.6.9 harness audit found the result: ``Inductor``,
``TransientSolver(integration_method=...)`` and in-place ``set_temperature``
were shipped features with no test anywhere in ``tests/``, while
``tests/common/core_gates.py`` still advertised gates for them.

They come back as hermetic contracts rather than as NGSPICE gates on purpose.
Every one of them is a parser or solver seam whose failure mode is structural —
a device silently dropped, a DC short in a transient run, a stale
voltage-keyed cache after a temperature change — so an in-process assertion
answers the same question as a simulator comparison, runs in the collected
unit suite, and cannot rot behind a missing binary. Where a number is checked
it is checked against the closed form of a *linear* network, which is an
independent reference; nothing here compares a compact model against itself.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import (
    CurrentSource, Inductor, Resistor, VoltageSource,
)
from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF
from pycircuitsim.parser import Parser
from pycircuitsim.solver import ACSolver, DCSolver, TransientSolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data.contracts import FULL_TERMINAL_OUTPUT_COLUMN_ORDER
from neural_network.data.normalize import NormStats
from neural_network.models.direct_net import DirectNet


def _parse(text: str, tmp_path: Path) -> Parser:
    """Parse one deck through the file entry point the gates use."""
    deck = tmp_path / "deck.sp"
    deck.write_text(text)
    parser = Parser()
    parser.parse_file(str(deck))
    return parser


@pytest.fixture()
def nn_device(tmp_path: Path) -> NMOS_DNF:
    """A LEVEL=75 device on a deterministic in-tmpdir checkpoint.

    Temperature rebinding is a base-class contract shared by every NN family,
    so a synthetic checkpoint answers it without pinning the test to whichever
    trained artifacts happen to be on this machine.
    """
    model = DirectNet(
        input_dim=7, hidden_dim=8, n_layers=1,
        output_dim=len(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
        num_tech_codes=2, tech_embed_dim=2, tech_embed_dropout=0.0,
        unknown_code_id=1,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias[0] = 1.0
    checkpoint = tmp_path / "core_probe_best.pt"
    torch.save(model.state_dict(), checkpoint)
    NormStats(
        mode="zscore",
        input_mean=np.zeros(7, dtype=np.float64),
        input_std=np.ones(7, dtype=np.float64),
        input_min=np.asarray(
            [-0.8, -0.8, -0.8, -0.8, 0.0, 0.0, 200.0], dtype=np.float64),
        input_max=np.asarray(
            [0.8, 0.8, 0.8, 0.8, 8.0, 1e-6, 500.0], dtype=np.float64),
        output_mean=np.zeros(6, dtype=np.float64),
        output_std=np.asarray([1e-4] * 3 + [1e-15] * 3, dtype=np.float64),
        output_columns=list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
    ).save(str(tmp_path / "core_probe_norm.npz"))
    norm_path = tmp_path / "core_probe_norm.npz"
    marker = {
        "family": "directnet-full",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "normalization": norm_path.name,
        "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "output_columns": list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
    }
    checkpoint.with_suffix(".pt.complete").write_text(json.dumps(marker))
    return NMOS_DNF(
        "Mprobe", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )


# ---------------------------------------------------------------------------
# Inductor — DC short, AC branch reactance, loud transient rejection
# ---------------------------------------------------------------------------
def test_inductor_card_reaches_the_circuit_as_an_inductor(
    tmp_path: Path,
) -> None:
    """An unrecognized device letter must never parse to a missing device."""
    parser = _parse(
        "* RL divider\n"
        "V1 in 0 DC=1 AC=1 0\n"
        "R1 in out 1k\n"
        "L1 out 0 1e-3\n"
        ".ac dec 10 1 1e6\n"
        ".end\n",
        tmp_path,
    )

    inductors = [component for component in parser.circuit.components
                 if isinstance(component, Inductor)]
    assert [item.name for item in inductors] == ["L1"]
    assert inductors[0].inductance == pytest.approx(1e-3)
    assert inductors[0].nodes == ["out", "0"]


@pytest.mark.parametrize(
    ("nodes", "inductance"),
    (
        (["a", "b", "c"], 1e-3),
        (["a"], 1e-3),
        (["a", "b"], 0.0),
        (["a", "b"], -1e-3),
    ),
)
def test_inductor_rejects_malformed_geometry(
    nodes: list[str],
    inductance: float,
) -> None:
    with pytest.raises(ValueError):
        Inductor("L1", nodes, inductance)


def test_inductor_is_a_dc_short() -> None:
    """The DC-short/AC-open feedback break is the whole point of the device."""
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "out"], 1_000.0))
    circuit.add_component(Inductor("L1", ["out", "0"], 1e-3))

    solution = DCSolver(circuit).solve(skip_header=True)

    assert solution["out"] == pytest.approx(0.0, abs=1e-9)


def test_inductor_carries_its_reactance_on_its_own_branch_row() -> None:
    """v(out)/v(in) of an RL divider must follow jwL/(R + jwL) exactly.

    A linear network has a closed form, so this is an independent reference,
    not PyCircuitSim checked against itself. A branch row that dropped the
    reactance would leave the DC short in place and return 0 V at every
    frequency — which is exactly the silent wrong answer the AC stamp exists
    to prevent.
    """
    resistance, inductance = 1_000.0, 1e-3
    circuit = Circuit()
    circuit.add_component(
        VoltageSource("V1", ["in", "0"], 0.0, ac_magnitude=1.0),
    )
    circuit.add_component(Resistor("R1", ["in", "out"], resistance))
    circuit.add_component(Inductor("L1", ["out", "0"], inductance))

    frequencies = np.asarray([1e3, 1e5, 1e7])
    result = ACSolver(
        circuit, dc_solution={"in": 0.0, "out": 0.0, "0": 0.0},
    ).solve(frequencies)

    omega = 2.0 * np.pi * frequencies
    expected = (1j * omega * inductance) / (resistance + 1j * omega * inductance)
    np.testing.assert_allclose(result["out"], expected, rtol=1e-9, atol=1e-12)


def test_inductor_in_a_transient_circuit_is_loud() -> None:
    """A DC short accepted by a transient run would be a plausible wrong answer."""
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "out"], 1_000.0))
    circuit.add_component(Inductor("L1", ["out", "0"], 1e-3))

    with pytest.raises(NotImplementedError, match="DC/AC only"):
        TransientSolver(circuit, t_stop=1e-6, dt=1e-8)


# ---------------------------------------------------------------------------
# Transient integration method
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ("auto", "gear2", "trap"))
def test_declared_integration_methods_are_accepted(method: str) -> None:
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "0"], 1_000.0))

    solver = TransientSolver(
        circuit, t_stop=1e-6, dt=1e-8, integration_method=method,
    )

    assert solver.integration_method == method


@pytest.mark.parametrize("method", ("gear", "bdf2", "GEAR2", ""))
def test_unknown_integration_method_is_rejected_before_solving(
    method: str,
) -> None:
    """A typo must not silently fall back to the default integrator."""
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "0"], 1_000.0))

    with pytest.raises(ValueError, match="integration_method"):
        TransientSolver(
            circuit, t_stop=1e-6, dt=1e-8, integration_method=method,
        )


# ---------------------------------------------------------------------------
# Independent current source sign convention
# ---------------------------------------------------------------------------
def test_current_source_drives_current_out_of_its_first_node() -> None:
    """SPICE's ``I<name> n+ n- <value>`` pushes current from n+ through to n-.

    An RC or diode-connected bias deck that got this backwards would still
    converge and still look plausible; only the sign of every biased node
    would be wrong.
    """
    circuit = Circuit()
    circuit.add_component(CurrentSource("I1", ["0", "out"], 1e-3))
    circuit.add_component(Resistor("R1", ["out", "0"], 1_000.0))

    solution = DCSolver(circuit).solve(skip_header=True)

    assert solution["out"] == pytest.approx(1.0, rel=1e-9)


def test_current_source_card_keeps_the_netlist_node_order(
    tmp_path: Path,
) -> None:
    """Node order is the sign, so the parser may not normalize it away.

    The value is written ``1e-3`` rather than ``1m`` on purpose: ``m`` is milli
    to NGSPICE and mega to this parser (see
    ``tests/test_deck_engine_compatibility.py``).
    """
    parser = _parse(
        "* current bias\n"
        "I1 0 out 1e-3\n"
        "R1 out 0 1k\n"
        ".op\n"
        ".end\n",
        tmp_path,
    )

    sources = [component for component in parser.circuit.components
               if isinstance(component, CurrentSource)]
    assert [item.nodes for item in sources] == [["0", "out"]]
    assert sources[0].value == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# Transient branch currents
# ---------------------------------------------------------------------------
def test_transient_reports_a_branch_current_for_every_committed_step() -> None:
    """``solver.source_currents`` is what becomes the ``i(V*)`` trace column.

    ``circuit_benchmarks`` lifts this mapping into ``i(<name>)`` only when it
    holds one finite value per returned time point, so a short or NaN-padded
    row silently drops the supply-current signal that ``ring_supply``,
    ``bias_fanout_op`` and the LDO load step are scored on. Index 0 carries the
    pre-transient operating point by construction and is excluded here.
    """
    circuit = Circuit()
    circuit.add_component(VoltageSource("V1", ["in", "0"], 1.0))
    circuit.add_component(Resistor("R1", ["in", "0"], 1_000.0))

    solver = TransientSolver(circuit, t_stop=1e-8, dt=1e-9)
    result = solver.solve()

    assert set(solver.source_currents) == {"V1"}
    current = solver.source_currents["V1"]
    assert len(current) == len(result["time"])
    assert np.all(np.isfinite(current[1:]))
    np.testing.assert_allclose(current[1:], -1e-3, rtol=1e-6)


# ---------------------------------------------------------------------------
# In-place temperature rebinding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "deck",
    (
        "* temperature first\n.temp -25\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n",
        "* temperature last\nV1 in 0 1\nR1 in 0 1k\n.temp -25\n.op\n.end\n",
    ),
)
def test_temp_card_is_order_independent_and_celsius(
    deck: str,
    tmp_path: Path,
) -> None:
    """``.temp`` is Celsius and binds regardless of where the card sits.

    Card order matters because existing devices are rebound in place while
    later constructors read ``_temperature_kelvin`` directly; a deck that
    declared its temperature after its devices must not run at 27 C. The
    Celsius-to-Kelvin conversion is also rounded, so ``-25 C`` lands exactly on
    248.15 K rather than 248.14999999999998 and is not spuriously rejected by a
    checkpoint certified down to that bound.
    """
    parser = _parse(deck, tmp_path)

    assert parser._temperature_kelvin == pytest.approx(248.15, abs=0.0)


def test_temp_card_rejects_a_kelvin_value_and_a_malformed_card(
    tmp_path: Path,
) -> None:
    """A deck that meant 300 K but wrote Celsius must fail, not run at 573 K."""
    with pytest.raises(ValueError, match="exceed 200 K"):
        _parse("* cold\n.temp -300\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n",
               tmp_path)
    with pytest.raises(ValueError, match="Invalid .temp syntax"):
        _parse("* two values\n.temp 27 85\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n",
               tmp_path)


def test_set_temperature_rebinds_geometry_and_drops_the_stale_cache(
    nn_device: "NMOS_DNF",
) -> None:
    """A temperature change at unchanged voltages must not return a stale value.

    ``_eval_cache`` is keyed on the terminal voltages alone and temperature
    lives in the constant geometry tensor, so a model that rebinds temperature
    without clearing the cache answers the new temperature with the old
    current. That is a silent wrong answer, and it is what every ``temp_cold``
    / ``temp_hot`` corner row depends on not happening.
    """
    voltages = {"d": 0.5, "g": 0.6, "s": 0.0, "b": 0.0}
    nn_device.get_terminal_stamp(voltages)
    assert nn_device._eval_cache is not None
    before = nn_device._raw_inputs(voltages).copy()

    nn_device.set_temperature(398.15)

    assert nn_device.temperature == pytest.approx(398.15)
    assert nn_device._eval_cache is None
    assert nn_device._q_prev is None
    assert not np.array_equal(nn_device._raw_inputs(voltages), before)


def test_set_temperature_rejects_a_celsius_value(nn_device: "NMOS_DNF") -> None:
    """``temp_C`` passed where ``temp_K`` is expected must fail, not run cold."""
    with pytest.raises(ValueError, match="Kelvin"):
        nn_device.set_temperature(125.0)
