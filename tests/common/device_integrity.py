"""Single-device integrity harness: the surfaces the DC gate never scores.

``verify_nn_multi_tech_dc`` sweeps Id-Vgs on a linear axis at one fixed
``Vds = 0.5*VDD``.  That leaves four device behaviours ungated, and every one
of them is upstream of a circuit failure the accuracy reports still carry
open:

``output``
    Id-Vds at a family of gate biases.  ``gds`` is never compared against
    ground truth anywhere else in the suite, yet output resistance sets the
    gain of every amplifier and the opamp fixed point is ``gds``-dominated.

``subthreshold``
    Id-Vgs scored in decades.  A linear-axis NRMSE at half supply is dominated
    by strong inversion, so the leakage floor is invisible - and switched-
    capacitor hold droop and SRAM lobe positivity are subthreshold effects.

``linear``
    Id-Vds through ``Vds = 0``.  Triode on-resistance, origin symmetry, and
    the wrong-sign leakage that destabilises Newton all live here.

``derivative``
    ``gm``, ``gds`` and ``gmb`` against ground truth.  The existing Jacobian
    probe differentiates the network against finite differences *of itself*;
    that is self-consistency, not accuracy.

Ground truth is NGSPICE on the identical BSIM-CMG LEVEL=72 OSDI model, from
the same ``circuit_templates/L0_devices/mosfet.spice.tmpl`` source the candidate
renders.  Only the compact model changes.

Sign convention
---------------
Both engines expose the drain source branch current as ``i(Vds)`` with the
*same* sign, so this module measures ``id = -i(Vds)`` on both sides: one
definition, no per-engine correction.  (The legacy device gates instead pair
NGSPICE ``-i(Vds)`` against the PyCircuitSim device current ``i(Mdut)``, which
is the same number reached by two different routes.)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tests.common.base import (
    DEVICE_DECKS, OSDI_PATH, render_template, run_ngspice_subprocess,
)
from tests.common.circuit_benchmarks import (
    BenchTech, full_metrics, get_baked_modelcard, parse_netlist,
    run_directnet_dc_sweep,
)
from tests.common.gate_result import GateResult
from tests.common.simple_circuit_catalog import AnalysisSpec
from tests.common.simple_circuit_harness import (
    Corner, RunSpec, apply_corner, physical_deck_mismatch,
)

TEMPLATE = DEVICE_DECKS / "mosfet.spice.tmpl"

#: Suites in the order a failure should be read: a broken output
#: characteristic explains a broken derivative, not the other way round.
SUITES: Tuple[str, ...] = ("output", "subthreshold", "linear", "derivative")

DEVICE_KINDS: Tuple[str, ...] = ("nmos", "pmos")


# ---------------------------------------------------------------------------
# Sweep declarations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepSpec:
    """One paired single-device sweep.

    Every voltage is stated in the device's own frame and already carries its
    polarity sign, so a PMOS spec holds negative gate and drain biases and is
    swept downwards.  The source terminal is ground in both decks.
    """

    suite: str
    label: str
    device: str
    axis: str            # "vgs" | "vds" | "vbs" — which bias is swept
    start: float
    stop: float
    step: float
    vgs: float = 0.0
    vds: float = 0.0
    vbs: float = 0.0
    channel: str = ""    # derivative suite: gm | gds | gmb

    def __post_init__(self) -> None:
        if self.device not in DEVICE_KINDS:
            raise ValueError(f"unknown device kind {self.device!r}")
        if self.axis not in ("vgs", "vds", "vbs"):
            raise ValueError(f"unknown sweep axis {self.axis!r}")
        if self.step == 0.0:
            raise ValueError("sweep step cannot be zero")
        if (self.stop - self.start) * self.step <= 0.0:
            raise ValueError(
                f"{self.label}: step {self.step:g} never reaches {self.stop:g} "
                f"from {self.start:g}")

    @property
    def sweep_source(self) -> str:
        """Name of the swept independent source in the rendered deck."""
        return {"vgs": "Vgs", "vds": "Vds", "vbs": "Vbs"}[self.axis]

    @property
    def result_key(self) -> str:
        return f"{self.suite}_{self.device}_{self.label}"


def _sign(device: str) -> float:
    """Polarity of the device's own bias frame."""
    return 1.0 if device == "nmos" else -1.0


def device_corner_applies(
    base_bt: BenchTech,
    device: str,
    corner: Corner,
) -> bool:
    """Whether a corner changes a field exercised by this device polarity."""
    if device not in DEVICE_KINDS:
        raise ValueError(f"unknown device kind {device!r}")
    if corner.name == "nominal":
        return True
    # The integrity matrix already has an explicit Vbs derivative sweep. The
    # circuit-only body rail corner is not consumed by these source-relative
    # device decks and must not create a duplicate nominal row.
    if corner.body_reverse_frac:
        return False
    try:
        stressed = apply_corner(base_bt, corner)
    except ValueError:
        return False
    if stressed.vdd != base_bt.vdd \
            or stressed.temperature_c != base_bt.temperature_c:
        return True
    if device == "nmos":
        return (
            stressed.l_nmos != base_bt.l_nmos
            or stressed.nfin != base_bt.nfin
            or stressed.effective_nmos_vt != base_bt.effective_nmos_vt
        )
    return (
        stressed.l_pmos != base_bt.l_pmos
        or stressed.effective_nfin_p != base_bt.effective_nfin_p
        or stressed.effective_pmos_vt != base_bt.effective_pmos_vt
    )


def build_sweeps(bt: BenchTech, device: str) -> Tuple[SweepSpec, ...]:
    """Return every declared integrity sweep for one technology and polarity.

    The windows are fractions of the technology supply rather than absolute
    volts, so a 0.65 V and a 0.80 V node are asked the same physical question.
    """
    s = _sign(device)
    v = bt.vdd
    specs: List[SweepSpec] = []

    # --- output characteristics: Id-Vds at four gate biases -----------------
    # Starting just off the origin keeps this suite about saturation; the
    # triode knee and Vds=0 behaviour belong to the `linear` suite.
    for frac in (0.45, 0.60, 0.80, 1.00):
        specs.append(SweepSpec(
            suite="output", label=f"vgs{frac:.2f}", device=device,
            axis="vds", start=s * 0.02 * v, stop=s * v, step=s * 0.02 * v,
            vgs=s * frac * v, vds=0.0, vbs=0.0,
        ))

    # --- subthreshold: decades of Id below threshold ------------------------
    specs.append(SweepSpec(
        suite="subthreshold", label="idvg_log", device=device,
        axis="vgs", start=0.0, stop=s * 0.55 * v, step=s * 0.005 * v,
        vds=s * 0.5 * v, vbs=0.0,
    ))

    # --- linear region: Id-Vds straddling the origin ------------------------
    # The reverse half is deliberately included.  The only reverse bias the
    # existing DC matrix covers is a single -0.25*VDD point, so the sign
    # symmetry of the model around Vds=0 has never been scored.
    specs.append(SweepSpec(
        suite="linear", label="ron", device=device,
        axis="vds", start=-s * 0.15 * v, stop=s * 0.15 * v, step=s * 0.01 * v,
        vgs=s * v, vbs=0.0,
    ))

    # --- derivatives: gm, gds, gmb against ground truth ---------------------
    specs.append(SweepSpec(
        suite="derivative", label="gm", device=device, channel="gm",
        axis="vgs", start=s * 0.30 * v, stop=s * v, step=s * 0.01 * v,
        vds=s * 0.5 * v, vbs=0.0,
    ))
    specs.append(SweepSpec(
        suite="derivative", label="gds", device=device, channel="gds",
        axis="vds", start=s * 0.05 * v, stop=s * v, step=s * 0.01 * v,
        vgs=s * 0.8 * v, vbs=0.0,
    ))
    # Reverse body bias only: forward body bias would forward-bias the
    # source/bulk junction and is outside the certified training corridor.
    specs.append(SweepSpec(
        suite="derivative", label="gmb", device=device, channel="gmb",
        axis="vbs", start=-s * 0.30 * v, stop=0.0, step=s * 0.01 * v,
        vgs=s * 0.8 * v, vds=s * 0.5 * v,
    ))
    return tuple(specs)


# ---------------------------------------------------------------------------
# Deck rendering
# ---------------------------------------------------------------------------
def _bias_card(name: str, node: str, value: float, swept: bool) -> str:
    """Emit one independent bias source; a swept source starts at its origin."""
    return f"{name} {node} 0 {0.0 if swept else value:.12g}"


def render_device_decks(
    spec: SweepSpec,
    bt: BenchTech,
    *,
    baked_lib: Path,
    level: int,
) -> Tuple[str, str]:
    """Render the candidate and LEVEL=72 decks for one identical sweep."""
    is_pmos = spec.device == "pmos"
    model = bt.pmos_model if is_pmos else bt.nmos_model
    length = bt.l_pmos if is_pmos else bt.l_nmos
    nfin = bt.effective_nfin_p if is_pmos else bt.nfin
    vt = bt.effective_pmos_vt if is_pmos else bt.effective_nmos_vt

    shared = {
        "TEMP": f"{bt.temperature_c:.12g}",
        "DRAIN_BIAS": _bias_card("Vds", "d", spec.vds, spec.axis == "vds"),
        "GATE_BIAS": _bias_card("Vgs", "g", spec.vgs, spec.axis == "vgs"),
        "SOURCE_BIAS": "",
        "BULK_BIAS": _bias_card("Vbs", "b", spec.vbs, spec.axis == "vbs"),
        "DRAIN_NODE": "d", "GATE_NODE": "g",
        "SOURCE_NODE": "0", "BULK_NODE": "b",
        "EXTRA_DEVICES": "", "LOAD": "",
    }
    # Device names differ by engine prefix only; the second character carries
    # polarity so both decks produce the same topology signature.
    stem = "pdut" if is_pmos else "dut"
    card = (f"dc {spec.sweep_source} {spec.start:.12g} {spec.stop:.12g} "
            f"{spec.step:.12g}")
    family = {
        75: " FAMILY=directnet-full",
        76: " FAMILY=bsimar-full",
    }.get(level, "")

    reference = render_template(TEMPLATE, {
        **shared,
        "MODEL_SETUP": f'.include "{baked_lib}"',
        "DEVICE_NAME": f"N{stem}", "DEVICE": model,
        # The reference analysis is issued inside the NGSPICE control block, so
        # the template slot stays empty and cannot double-run the sweep.
        "ANALYSIS": "",
    })
    device_kind = "PMOS" if is_pmos else "NMOS"
    candidate = render_template(TEMPLATE, {
        **shared,
        "MODEL_SETUP": (
            f".model {spec.device}_nn {device_kind} "
            f"(LEVEL={level}{family} "
            f"TECH={bt.nn_tech} VT={vt})"
        ),
        "DEVICE_NAME": f"M{stem}",
        "DEVICE": f"{spec.device}_nn L={length * 1e9:g}n NFIN={nfin}",
        "ANALYSIS": f".{card}",
    })
    return candidate, reference


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceTrace:
    """One accepted single-device sweep in the shared ``id`` convention."""

    axis: np.ndarray
    current: np.ndarray
    converged: bool = True

    def validate(
        self,
        *,
        expected_start: Optional[float] = None,
        expected_stop: Optional[float] = None,
        endpoint_tolerance: float = 0.0,
        max_step: Optional[float] = None,
        minimum_points: Optional[int] = None,
    ) -> None:
        axis = np.asarray(self.axis, dtype=float)
        current = np.asarray(self.current, dtype=float)
        if axis.ndim != 1 or axis.size < 3:
            raise ValueError("device sweep needs at least three points")
        if minimum_points is not None and axis.size < minimum_points:
            raise ValueError(
                f"device sweep has {axis.size} points; expected at least "
                f"{minimum_points}"
            )
        if current.shape != axis.shape:
            raise ValueError("device sweep axis/current length mismatch")
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(current)):
            raise ValueError("device sweep contains NaN/Inf")
        delta = np.diff(axis)
        if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
            raise ValueError("device sweep axis must be strictly monotonic")
        if max_step is not None:
            if not math.isfinite(max_step) or max_step <= 0.0:
                raise ValueError("device sweep maximum step must be positive")
            largest = float(np.max(np.abs(delta)))
            if largest > max_step * (1.0 + 1e-9) + 1e-18:
                raise ValueError(
                    f"device sweep axis gap {largest:g} exceeds declared "
                    f"step {max_step:g}"
                )
        span = abs(float(axis[-1] - axis[0]))
        atol = max(1e-18, span * 1e-9, endpoint_tolerance)
        if expected_start is not None and not np.isclose(
            axis[0], expected_start, rtol=1e-9, atol=atol,
        ):
            raise ValueError(
                f"device sweep starts at {axis[0]:g}, not {expected_start:g}"
            )
        if expected_stop is not None and not np.isclose(
            axis[-1], expected_stop, rtol=1e-9, atol=atol,
        ):
            raise ValueError(
                f"device sweep stops at {axis[-1]:g}, not {expected_stop:g}"
            )


def run_reference_sweep(
    deck: str, spec: SweepSpec, work_dir: Path, tag: str,
) -> DeviceTrace:
    """Run the accepted LEVEL=72 trajectory for one rendered sweep."""
    work_dir.mkdir(parents=True, exist_ok=True)
    deck_path = work_dir / f"ngspice_{tag}.cir"
    csv_path = work_dir / f"ngspice_{tag}.csv"
    log_path = work_dir / f"ngspice_{tag}.log"
    runner_path = work_dir / f"ngspice_{tag}_runner.cir"
    deck_path.write_text(deck)
    card = (f"dc {spec.sweep_source} {spec.start:.12g} {spec.stop:.12g} "
            f"{spec.step:.12g}")
    runner_path.write_text(
        f"* NGSPICE device-integrity runner ({tag})\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {deck_path}\n"
        "set filetype=ascii\n"
        "set wr_vecnames\n"
        f"{card}\n"
        f"wrdata {csv_path} i(Vds)\n"
        ".endc\n.end\n"
    )
    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)
    rows = [[float(value) for value in line.split()]
            for line in lines[1:] if line.strip()]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise RuntimeError(
            f"NGSPICE wrdata width {data.shape} is not one real vector")
    trace = DeviceTrace(data[:, 0], -data[:, 1])
    trace.validate(
        expected_start=spec.start,
        expected_stop=spec.stop,
        endpoint_tolerance=abs(spec.step) * (1.0 + 1e-9),
        max_step=abs(spec.step),
        minimum_points=max(
            int(round(abs((spec.stop - spec.start) / spec.step))), 3,
        ),
    )
    return trace


def run_candidate_sweep(
    deck: str, spec: SweepSpec, work_dir: Path, tag: str,
) -> DeviceTrace:
    """Run the NN adapter over the identical sweep."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"candidate_{tag}.sp"
    path.write_text(deck)
    logging.disable(logging.CRITICAL)
    try:
        parser = parse_netlist(path)
        results = run_directnet_dc_sweep(path, work_dir, tag)
    finally:
        logging.disable(logging.NOTSET)
    key = next((name for name in results if name.lower() == "i(vds)"), "")
    if not key:
        raise KeyError(
            f"candidate results carry no i(Vds); keys={sorted(results)}")
    current = -np.asarray(results[key], dtype=float)
    params = parser.analysis_params
    axis = (float(params["start"])
            + float(params["step"]) * np.arange(current.size, dtype=float))
    trace = DeviceTrace(axis, current)
    trace.validate(
        expected_start=spec.start,
        expected_stop=spec.stop,
        endpoint_tolerance=abs(spec.step) * (1.0 + 1e-9),
        max_step=abs(spec.step),
        minimum_points=max(
            int(round(abs((spec.stop - spec.start) / spec.step))), 3,
        ),
    )
    return trace


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _common_grid(candidate: DeviceTrace, reference: DeviceTrace) -> np.ndarray:
    lo = max(float(np.min(candidate.axis)), float(np.min(reference.axis)))
    hi = min(float(np.max(candidate.axis)), float(np.max(reference.axis)))
    if not hi > lo:
        raise ValueError("candidate/reference sweep axes do not overlap")
    count = min(max(min(candidate.axis.size, reference.axis.size), 16), 400)
    return np.linspace(lo, hi, count)


def _interpolate(target: np.ndarray, trace: DeviceTrace) -> np.ndarray:
    order = np.argsort(trace.axis)
    return np.interp(target, trace.axis[order], trace.current[order])


def _relative_error(test: float, reference: float) -> float:
    if not np.isfinite(test) or not np.isfinite(reference) \
            or abs(reference) < 1e-30:
        return float("nan")
    return abs(test - reference) / abs(reference) * 100.0


def _slope(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Matched central difference — the identical stencil on both engines."""
    return np.gradient(values, axis)


def _decades(current: np.ndarray, floor: float) -> np.ndarray:
    return np.log10(np.maximum(np.abs(current), floor))


def _subthreshold_window(
    grid: np.ndarray, reference: np.ndarray,
) -> np.ndarray:
    """Reference-defined exponential window.

    The window is derived from the *reference* only.  Letting the candidate
    influence it would allow a broken model to select the range that flatters
    it, which is the failure mode a fitted-window slope metric invites.
    """
    magnitude = np.abs(reference)
    floor = float(magnitude[0])
    ceiling = float(np.max(magnitude))
    if not np.isfinite(floor) or floor <= 0.0 or ceiling <= floor * 10.0:
        return np.zeros(grid.shape, dtype=bool)
    lo = floor * 10.0
    hi = min(ceiling * 1e-3, floor * 1e5)
    if hi <= lo:
        hi = math.sqrt(lo * ceiling)
    return (magnitude >= lo) & (magnitude <= hi)


def _slope_mv_per_decade(
    grid: np.ndarray, current: np.ndarray, window: np.ndarray, floor: float,
) -> float:
    if int(np.count_nonzero(window)) < 4:
        return float("nan")
    decades = _decades(current[window], floor)
    axis = np.abs(grid[window])
    if float(np.ptp(decades)) < 0.5:
        return float("nan")
    slope = float(np.polyfit(decades, axis, 1)[0])
    return abs(slope) * 1e3


def _zero_crossing_value(grid: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(0.0, grid, values)) if grid[0] <= 0.0 <= grid[-1] \
        else float("nan")


def suite_metrics(
    spec: SweepSpec,
    grid: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    vdd: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Return the aggregate and suite-specific metrics for one sweep."""
    metrics: Dict[str, float] = dict(full_metrics(candidate, reference))
    domain: Dict[str, object] = {}

    if spec.suite == "output":
        order = np.argsort(np.abs(grid))
        drive = np.abs(grid[order])
        test_current = candidate[order]
        ref_current = reference[order]
        window = drive >= 0.5 * vdd
        if int(np.count_nonzero(window)) >= 3:
            gds_test = float(np.median(np.abs(
                _slope(drive, test_current)[window]
            )))
            gds_ref = float(np.median(np.abs(
                _slope(drive, ref_current)[window]
            )))
            domain.update(
                gds_sat_test_s=gds_test, gds_sat_ref_s=gds_ref,
                gds_sat_error_pct=_relative_error(gds_test, gds_ref),
            )
        idsat_test = float(test_current[-1])
        idsat_ref = float(ref_current[-1])
        domain.update(
            idsat_test_a=idsat_test, idsat_ref_a=idsat_ref,
            idsat_error_pct=_relative_error(idsat_test, idsat_ref),
        )
        # Saturation knee: where the current first reaches 90% of its value at
        # the top of the sweep.  A model that saturates too early or too late
        # misplaces every compliance limit built on it.
        knees = []
        for values in (test_current, ref_current):
            target = 0.9 * abs(values[-1])
            reached = np.flatnonzero(np.abs(values) >= target)
            knees.append(float(drive[reached[0]]) if reached.size
                         else float("nan"))
        domain.update(knee_vds_test_v=knees[0], knee_vds_ref_v=knees[1],
                      knee_vds_error_v=abs(knees[0] - knees[1]))

    elif spec.suite == "subthreshold":
        # Put both polarities in increasing |Vgs| order. The shared comparison
        # grid is numerically ascending, which means a PMOS trace otherwise
        # starts at its on-state and silently reports that current as Ioff.
        order = np.argsort(np.abs(grid))
        drive = np.abs(grid[order])
        test_current = candidate[order]
        ref_current = reference[order]
        floor = max(float(np.max(np.abs(ref_current))) * 1e-12, 1e-18)
        test_dec = _decades(test_current, floor)
        ref_dec = _decades(ref_current, floor)
        span = float(np.ptp(ref_dec))
        errors = np.abs(test_dec - ref_dec)
        domain.update(
            decades_spanned=span,
            max_decade_error=float(np.max(errors)),
            log_decade_nrmse_pct=(
                float(np.sqrt(np.mean(errors ** 2)) / span * 100.0)
                if span > 1e-9 else float("nan")),
            ioff_test_a=float(abs(test_current[0])),
            ioff_ref_a=float(abs(ref_current[0])),
            ioff_error_pct=_relative_error(
                float(abs(test_current[0])), float(abs(ref_current[0]))),
        )
        window = _subthreshold_window(drive, ref_current)
        ss_test = _slope_mv_per_decade(
            drive, test_current, window, floor,
        )
        ss_ref = _slope_mv_per_decade(
            drive, ref_current, window, floor,
        )
        domain.update(
            ss_window_points=int(np.count_nonzero(window)),
            ss_test_mv_dec=ss_test, ss_ref_mv_dec=ss_ref,
            ss_error_pct=_relative_error(ss_test, ss_ref),
        )

    elif spec.suite == "linear":
        # Two on-resistances, deliberately. The near-origin fit straddles
        # Vds=0 and therefore measures the shape of the curve *through* the
        # origin, including forward/reverse symmetry. The forward fit uses
        # only the conducting half. Reporting one number would conflate "the
        # triode resistance is wrong" with "the origin crossing is wrong",
        # and on the measured checkpoints those differ by an order of
        # magnitude.
        polarity = _sign(spec.device)
        windows = {
            "ron": np.abs(grid) <= 0.05 * vdd,
            "ron_forward": polarity * grid > 0,
        }
        for stem, mask in windows.items():
            if int(np.count_nonzero(mask)) < 3:
                continue
            conductances = [
                float(np.polyfit(grid[mask], values[mask], 1)[0])
                for values in (candidate, reference)
            ]
            ron = [1.0 / g if abs(g) > 1e-30 else float("nan")
                   for g in conductances]
            domain.update(**{
                f"{stem}_test_ohm": ron[0], f"{stem}_ref_ohm": ron[1],
                f"{stem}_error_pct": _relative_error(ron[0], ron[1]),
                f"{stem}_points": int(np.count_nonzero(mask)),
            })
        zero_test = _zero_crossing_value(grid, candidate)
        zero_ref = _zero_crossing_value(grid, reference)
        # Physics pins Id(Vds=0)=0.  A model that does not is injecting current
        # into a shorted device, which no circuit metric would attribute here.
        domain.update(
            zero_offset_test_a=abs(zero_test), zero_offset_ref_a=abs(zero_ref),
            zero_offset_error_a=abs(abs(zero_test) - abs(zero_ref)),
        )

    elif spec.suite == "derivative":
        test_slope = _slope(grid, candidate)
        ref_slope = _slope(grid, reference)
        derivative = full_metrics(test_slope, ref_slope)
        domain.update(
            deriv_channel=spec.channel,
            deriv_nrmse_pct=derivative["nrmse_pct"],
            deriv_mre_pct=derivative["mre_pct"],
            deriv_r2=derivative["r2"],
            deriv_max_err=derivative["max_err"],
            deriv_median_test_s=float(np.median(np.abs(test_slope))),
            deriv_median_ref_s=float(np.median(np.abs(ref_slope))),
        )
        # A sign disagreement in gds is not a magnitude error: AGENTS.md floors
        # the stamped drain conductance precisely because a negative derivative
        # destabilises Newton.  Report it separately from NRMSE.
        agreement = np.sign(test_slope) == np.sign(ref_slope)
        domain["deriv_sign_agreement_pct"] = float(
            np.count_nonzero(agreement) / agreement.size * 100.0)

    return metrics, domain


_DEVICE_METRIC_CONTRACTS: Dict[str, Tuple[str, ...]] = {
    "output": (
        "gds_sat_error_pct", "idsat_error_pct", "knee_vds_error_v",
    ),
    "subthreshold": (
        "max_decade_error", "log_decade_nrmse_pct", "ioff_error_pct",
        "ss_test_mv_dec", "ss_ref_mv_dec", "ss_error_pct",
    ),
    "linear": (
        "ron_error_pct", "ron_forward_error_pct", "zero_offset_error_a",
    ),
    "derivative": (
        "deriv_nrmse_pct", "deriv_mre_pct", "deriv_r2",
        "deriv_max_err", "deriv_sign_agreement_pct",
    ),
}


def validate_device_metrics(
    spec: SweepSpec,
    metrics: Dict[str, float],
    domain: Dict[str, object],
) -> None:
    """Require every quantity promised by a device suite to be finite."""
    payload: Dict[str, object] = {**metrics, **domain}
    required = (
        "mre_pct", "r2", "nrmse_pct", "max_err",
        *_DEVICE_METRIC_CONTRACTS[spec.suite],
    )
    invalid = [
        name for name in required
        if name not in payload
        or isinstance(payload[name], (bool, np.bool_))
        or not isinstance(
            payload[name], (int, float, np.integer, np.floating),
        )
        or not np.isfinite(payload[name])
    ]
    if invalid:
        raise ValueError(f"missing or non-finite device metrics: {invalid}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_sweep(
    spec: SweepSpec,
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    *,
    level: int,
    reference_repeats: int = 1,
    run_spec: Optional[RunSpec] = None,
) -> GateResult:
    """Run one paired sweep and return a structured, schema-stable result."""
    if reference_repeats < 1:
        raise ValueError("reference_repeats must be >= 1")
    resolved_run_spec = run_spec or RunSpec.from_environment()
    if resolved_run_spec.model_level != level:
        raise ValueError(
            f"run spec LEVEL={resolved_run_spec.model_level} != requested {level}"
        )
    provenance = resolved_run_spec.result_fields()
    bt = apply_corner(base_bt, corner)
    reference_converged = False
    candidate_converged = False
    try:
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        candidate_deck, reference_deck = render_device_decks(
            spec, bt, baked_lib=baked, level=level,
        )
        analysis = AnalysisSpec(
            spec.label,
            "dc",
            f"dc {spec.sweep_source} {spec.start:.12g} "
            f"{spec.stop:.12g} {spec.step:.12g}",
            ("i(Vds)",),
            device_kinds=(spec.device,),
        )
        mismatch = physical_deck_mismatch(
            candidate_deck,
            reference_deck,
            analysis,
            bt,
            baked_lib=baked,
            model_level=level,
            device_kinds=(spec.device,),
        )
        if mismatch:
            raise ValueError(mismatch)

        references = [
            run_reference_sweep(
                reference_deck, spec, work_dir, f"{spec.result_key}_ref{index}",
            )
            for index in range(reference_repeats)
        ]
        reference_converged = True
        candidate = run_candidate_sweep(
            candidate_deck, spec, work_dir, spec.result_key,
        )
        candidate_converged = candidate.converged

        grid = _common_grid(candidate, references[0])
        test = _interpolate(grid, candidate)
        truth = _interpolate(grid, references[0])
        metrics, domain = suite_metrics(spec, grid, test, truth, vdd=bt.vdd)
        validate_device_metrics(spec, metrics, domain)
        domain["reference_repeats"] = len(references)
        if len(references) > 1:
            worst = 0.0
            for extra in references[1:]:
                repeat_grid = _common_grid(extra, references[0])
                worst = max(worst, full_metrics(
                    _interpolate(repeat_grid, extra),
                    _interpolate(repeat_grid, references[0]),
                )["nrmse_pct"])
            domain["reference_repeat_nrmse_pct"] = worst
        domain["sweep"] = {
            "axis": spec.axis, "start": spec.start, "stop": spec.stop,
            "step": spec.step, "vgs": spec.vgs, "vds": spec.vds,
            "vbs": spec.vbs, "points": int(grid.size),
        }
        return GateResult(
            case_id=f"device_{spec.suite}", tech=bt.name, corner=corner.name,
            analysis=f"{spec.device}_{spec.label}", role="diagnostic",
            status="diagnostic", metrics=metrics, domain=domain,
            reference_converged=True, candidate_converged=candidate.converged,
            **provenance,
        )
    except Exception as exc:  # noqa: BLE001 — an error row keeps its denominator slot
        return GateResult(
            case_id=f"device_{spec.suite}", tech=bt.name, corner=corner.name,
            analysis=f"{spec.device}_{spec.label}", role="diagnostic",
            status="error", error=f"{type(exc).__name__}: {exc}",
            reference_converged=reference_converged,
            candidate_converged=candidate_converged,
            execution_state=(
                "reference_error" if not reference_converged
                else "nonconverged" if "converg" in str(exc).lower()
                else "infrastructure_error"
            ),
            error_kind=(
                "reference" if not reference_converged
                else "candidate" if "converg" in str(exc).lower()
                else "infrastructure"
            ),
            **provenance,
        )


def run_device_suites(
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    *,
    level: int,
    suites: Sequence[str] = SUITES,
    devices: Sequence[str] = DEVICE_KINDS,
    reference_repeats: int = 1,
) -> List[GateResult]:
    """Run every requested suite/polarity sweep for one technology corner."""
    unknown = [name for name in suites if name not in SUITES]
    if unknown:
        raise ValueError(f"unknown suites {unknown}; available: {list(SUITES)}")
    unknown_devices = [name for name in devices if name not in DEVICE_KINDS]
    if unknown_devices:
        raise ValueError(f"unknown device kinds {unknown_devices}")
    selected_devices = [
        device for device in devices
        if device_corner_applies(base_bt, device, corner)
    ]
    if not selected_devices:
        return []
    bt = apply_corner(base_bt, corner)
    results: List[GateResult] = []
    run_spec = RunSpec.from_environment()
    for device in selected_devices:
        for spec in build_sweeps(bt, device):
            if spec.suite not in suites:
                continue
            results.append(run_sweep(
                spec, base_bt, corner,
                work_dir / spec.suite / device,
                level=level, reference_repeats=reference_repeats,
                run_spec=run_spec,
            ))
    return results
