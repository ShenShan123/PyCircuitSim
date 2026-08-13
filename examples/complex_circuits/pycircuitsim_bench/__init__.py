"""Run AnalogGym's decks through PyCircuitSim and score them against NGSPICE.

The AnalogGym decks (``examples/complex_circuits/designs_tsmc{5,6,7,12,16}/``) are
ngspice OSDI decks: geometry lives in per-geometry ``.model`` cards rather than
on the instance line, devices carry the ``N`` prefix and an ``m=`` multiplier,
values are ``.param`` expressions, the metrics are ``.meas`` statements, and the
analysis command is not in the deck at all — ``tools/finalize.py`` injects it.
This package bridges that to PyCircuitSim in three pieces:

``translate``
    Deck on disk -> a PyCircuitSim netlist plus an :class:`AnalysisPlan` list.
``measure``
    ``.meas`` statements -> metrics, evaluated in Python over a
    :class:`SweepResult`. One code path scores BOTH simulators, so a
    PyCircuitSim/NGSPICE difference can never be a difference in measurement
    semantics.
``run_compare``
    Drives both simulators over one deck and reports the metrics side by side.

This module is the frozen contract between those three: the dataclasses they
pass each other and the deck/control tables copied from ``finalize.py``. Renaming
anything here breaks all three at once, so treat the names as fixed.

Ground truth is always NGSPICE on the identical BSIM-CMG (LEVEL=72) OSDI model.
No NN model family (LEVEL=73/74/75) is involved anywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: Bumped when the on-disk JSON result layout changes incompatibly.
SCHEMA_VERSION: str = "1"


@dataclass(frozen=True)
class Signal:
    """A measurement's signal reference, e.g. ``vdb(opout)`` -> func/arg."""

    func: str
    arg: str


@dataclass(frozen=True)
class Condition:
    """A ``WHEN <signal>=<value>`` crossing condition."""

    signal: Signal
    value: float
    edge: Optional[str] = None
    nth: int = 1
    td: float = 0.0


@dataclass(frozen=True)
class MeasCard:
    """One parsed ``.meas``/``.measure`` statement.

    ``kind`` is the measurement form (``find_at``, ``when``, ``max``, ``avg``,
    ``param``, ...); ``analysis`` is the ``dc``/``ac``/``tran`` it belongs to.
    A card is only evaluated against a result of its own analysis.
    """

    name: str
    analysis: str
    kind: str
    signal: Optional[Signal] = None
    at: Optional[float] = None
    cond: Optional[Condition] = None
    window: Optional[Tuple[Optional[float], Optional[float]]] = None
    expr: Optional[str] = None
    raw: str = ""


@dataclass(frozen=True)
class AnalysisPlan:
    """One analysis to run, parsed out of an injected control string.

    ``kind`` is ``ac``, ``tran``, ``dc_source`` or ``dc_temp``. Only the fields
    belonging to that kind are populated. ``dc_temp`` sweeps temperature, which
    PyCircuitSim expresses by rebinding devices per point rather than by a
    ``.dc`` directive, which is why a plan is data and not a directive line.
    """

    kind: str
    control: str
    label: str = ""
    f_start: Optional[float] = None
    f_stop: Optional[float] = None
    n_per_decade: Optional[int] = None
    source: Optional[str] = None
    start: Optional[float] = None
    stop: Optional[float] = None
    step: Optional[float] = None
    t_step: Optional[float] = None
    t_stop: Optional[float] = None
    t_max: Optional[float] = None


@dataclass(frozen=True)
class DeckOptions:
    """``.options``/``.option`` values a deck sets as convergence aids.

    These eight keys were the complete set present across all 880 decks of
    the pre-V7.5.6 corpus, so they remain complete over the 410 decks the
    curated basket ships.
    """

    cshunt: Optional[float] = None
    rshunt: Optional[float] = None
    reltol: Optional[float] = None
    vntol: Optional[float] = None
    abstol: Optional[float] = None
    gmin: Optional[float] = None
    method: Optional[str] = None
    maxord: Optional[int] = None


@dataclass
class TranslatedDeck:
    """Everything ``translate`` extracted from one deck.

    ``multipliers`` is keyed ``"<subckt>:<name>"`` inside a subcircuit and by the
    bare emitted name at top level — a bare-name dict silently loses devices,
    because the charge pump defines the same instance name in two subcircuits.
    """

    tech: str
    category: str
    design: str
    deck: str
    design_dir: Path
    netlist_text: str
    modelcard_path: Path
    plans: List[AnalysisPlan]
    meas: List[MeasCard]
    nodesets: Dict[str, float]
    ic: Dict[str, float]
    params: Dict[str, float]
    options: DeckOptions
    devices: int
    multipliers: Dict[str, float]
    stability: Optional[Tuple[str, str, str]]
    temp_c: Optional[float]
    warnings: List[str]


@dataclass
class SweepResult:
    """One analysis' output, in the shape ``measure`` scores.

    ``x`` is the sweep axis (frequency, temperature in Celsius, the swept source
    value, or time); ``v``/``i`` map lower-cased names to per-point arrays,
    complex for ``kind == "ac"``. ``ok`` flags per-point solver success, so a
    partially converged sweep can still be measured over the points that did
    solve — ``None`` means every point is good. ``source`` is
    ``"pycircuitsim"`` or ``"ngspice"``.
    """

    kind: str
    x_name: str
    x: np.ndarray
    v: Dict[str, np.ndarray]
    i: Dict[str, np.ndarray]
    ok: Optional[np.ndarray]
    source: str
    meta: Dict[str, Any] = field(default_factory=dict)


#: Measured metrics by lower-cased name. ``None`` is ngspice's ``failed``: a
#: real outcome (a gain that never crosses 0 dB has no GBW), not an error.
MetricDict = Dict[str, Optional[float]]


#: Analysis control per (category, deck), verbatim from
#: ``tools/finalize.py`` CONTROLS. ``None`` means the control is
#: computed per design (LDO sweep limits from ``design.json``, the amplifier
#: transient step from the design's target GBW).
CONTROLS: Dict[str, Dict[str, Optional[str]]] = {
    "amplifier": {
        "tb_gain.cir": "ac dec 20 0.1 10G",
        "tb_cmrr.cir": "ac dec 20 0.1 10G",
        "tb_psrrp.cir": "ac dec 20 0.1 10G",
        "tb_psrrn.cir": "ac dec 20 0.1 10G",
        "tb_dc.cir": "dc temp 125 -40 -0.1",
        "tb_tran.cir": None,
    },
    "ldo": {
        "tb_load.cir": None,
        "tb_line_max.cir": None,
        "tb_line_min.cir": None,
        "tb_loop_max.cir": "ac dec 20 0.1 1G",
        "tb_loop_min.cir": "ac dec 20 0.1 1G",
        "tb_psrr_max.cir": "ac dec 20 0.1 1G",
        "tb_psrr_min.cir": "ac dec 20 0.1 1G",
        "tb_tran.cir": "tran 20n 44u",
    },
    "sensing_front_end": {"tb_dc.cir": "dc temp -20 120 0.5",
                          "tb_ac.cir": "ac dec 20 0.1 1G"},
    "voltage_reference": {"tb_dc.cir": "dc temp -40 125 0.5"},
    "charge_pump": {"tb_tran.cir": "tran 2p 200n"},
}

#: Decks whose loop gain feeds the wrap-aware stability metrics
#: (``tools/acstab.py``): deck -> (output node, GBW metric name, suffix).
STABILITY_DECKS: Dict[str, Tuple[str, str, str]] = {
    "tb_gain.cir": ("opout", "gain_bandwidth_product", ""),
    "tb_ac.cir": ("vout", "gain_bandwidth_product", ""),
    "tb_loop_max.cir": ("vo", "gbw_max", "_max"),
    "tb_loop_min.cir": ("vo", "gbw_min", "_min"),
}

#: ``finalize.py``'s two-sweep recovery for an amplifier ``tb_dc`` whose single
#: 165 C sweep loses its Newton branch: measure hot and cold from 25 C outward
#: and recombine.
AMP_TB_DC_RECOVERY: Tuple[Tuple[str, str], ...] = (
    ("hot", "dc temp 25 125 0.1"),
    ("cold", "dc temp 25 -40 -0.1"),
)

#: Last-resort narrow window when even the two-sweep recovery diverges.
AMP_TB_DC_NARROW: str = "dc temp 30 20 -5"


class TranslateError(RuntimeError):
    """A deck could not be translated into something PyCircuitSim can run."""


class SimFailure(RuntimeError):
    """A simulator did not produce a usable result for an analysis."""


__all__ = [
    "SCHEMA_VERSION",
    "Signal",
    "Condition",
    "MeasCard",
    "AnalysisPlan",
    "DeckOptions",
    "TranslatedDeck",
    "SweepResult",
    "MetricDict",
    "CONTROLS",
    "STABILITY_DECKS",
    "AMP_TB_DC_RECOVERY",
    "AMP_TB_DC_NARROW",
    "TranslateError",
    "SimFailure",
]
