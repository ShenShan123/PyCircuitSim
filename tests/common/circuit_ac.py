"""Shared infrastructure for the AC (small-signal) circuit benchmark.

V6.5 — extends the DC/transient circuit harness (``circuit_benchmarks.py``) into the
frequency domain so DirectNet (LEVEL=73) AC accuracy can be gated against NGSPICE
BSIM-CMG (LEVEL=72) ground truth — never a self-defined transfer function
(AGENTS.md Validation rule).

AC linearizes about a DC operating point, so it is only physically meaningful for
circuits with a stable amplifying OP. This module supports the two such circuits:

  * the device-level common-source amplifier   (verify_nn_ac.py)
  * the two-stage Miller opamp, open loop       (verify_circuit_opamp_ac.py)

The free-running ring oscillator (astable, no stable OP) and the 6T SRAM
(bistable) are deliberately EXCLUDED — neither has a defensible NGSPICE ``.ac``
ground truth.

The AC metric primitives (decade grid, NGSPICE `.ac` runner, complex interp,
mag/phase agreement, −3 dB corner) live in ``tests/simple_circuits/verify_ac.py`` and are reused
here verbatim (single code path). This module adds:

  * ``run_directnet_ac``      — NN-aware DC OP (retry path) + ACSolver sweep,
  * ``run_ngspice_ac_baked``  — NGSPICE `.ac` on a baked BSIM-CMG body,
  * ``ac_metrics_extended``   — base metrics + GBW / unity-gain / phase-margin,
  * ``ac_freq_grid``          — the standard CS-amp / opamp decade grids.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Single-thread torch BEFORE any inference (v664 P0) — same rationale and
# PYCIRCUITSIM_TORCH_THREADS override as tests/common/circuit_benchmarks.py: the AC gate
# linearizes about the same basin-fragile OP the DC gates solve.
try:
    import torch

    torch.set_num_threads(int(os.environ.get("PYCIRCUITSIM_TORCH_THREADS", "1")))
except (ImportError, ValueError):  # pragma: no cover
    pass

# Reuse the AC primitives from the shipping single-point AC gate (DRY — one
# code path for the complex-MNA / NGSPICE-runner / metric logic). verify_ac has
# no import side effects (its main() is __main__-guarded).
from tests.simple_circuits.verify_ac import (  # noqa: E402
    dec_frequencies,
    run_ngspice_ac,
    _interp_to,
    ac_metrics,
    minus3db_corner,
)


# ---------------------------------------------------------------------------
# Frequency grids
# ---------------------------------------------------------------------------
def ac_freq_grid(kind: str = "cs_amp") -> Tuple[np.ndarray, str]:
    """Return (frequencies, ac_card) for a circuit class.

    ``cs_amp``: 1 kHz–1 THz @ 20/decade (matches verify_ac.py L2).
    ``opamp`` : 10 Hz–100 GHz @ 10/decade — wide enough to capture the DC-gain
                plateau AND the unity-gain crossing of a two-stage opamp.
    """
    if kind == "opamp":
        f1, f2, n = 1e1, 1e11, 10
    else:  # cs_amp
        f1, f2, n = 1e3, 1e12, 20
    freqs = dec_frequencies(n, f1, f2)
    ac_card = f"ac dec {n} {f1:g} {f2:g}"
    return freqs, ac_card


# ---------------------------------------------------------------------------
# NGSPICE BSIM-CMG (LEVEL=72) AC runner — ground truth
# ---------------------------------------------------------------------------
def run_ngspice_ac_baked(
    body_lines: List[str],
    work_dir: Path,
    tag: str,
    ac_card: str,
    out_node: str,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Run NGSPICE `.ac` on a BSIM-CMG body and return complex V(out_node)
    interpolated onto ``frequencies``.

    ``body_lines`` is the circuit body (``.include "{baked}"``, ``.temp 27``,
    sources, device lines with NO instance params, caps) following the
    ``circuit_benchmarks.py`` baking convention. Delegates to
    ``verify_ac.run_ngspice_ac``
    which writes a one-vector ``wrdata`` (columns [freq, real, imag] — no
    rad/deg ambiguity).
    """
    deck_text = "\n".join(body_lines) + "\n.end\n"
    f_ng, v_ng = run_ngspice_ac(deck_text, work_dir, tag, ac_card, out_node)
    return _interp_to(frequencies, f_ng, v_ng)


# ---------------------------------------------------------------------------
# DirectNet (LEVEL=73) AC runner — NN-aware DC operating point
# ---------------------------------------------------------------------------
def run_directnet_ac(
    deck_text: str,
    work_dir: Path,
    tag: str,
    frequencies: np.ndarray,
    out_node: str,
    dc_op: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """Parse a DirectNet deck, solve (or accept) the DC OP, run AC.

    Returns ``(vout_complex, op_converged, dc_op)`` where ``dc_op`` is the DC
    operating-point voltage dict (so a low gain can be attributed to OP drift vs
    derivative error).

    If ``dc_op`` is provided it is used verbatim as the linearization point
    (op_converged=True) — used by the opamp gate to linearize about the SAME
    DC-sweep operating point its gain gate uses (the basin-fragile opamp OP can
    re-solve into a different basin otherwise). When ``dc_op`` is None the OP is
    solved here via ``_solve_dc_with_retry`` (GMIN + pseudo-transient fallback),
    NOT a bare DCSolver — the NN value surface needs the homotopy retries. The AC
    linearization is about exactly this OP, so OP fidelity == AC fidelity;
    ``op_converged`` is surfaced so a non-converged OP is reported honestly.
    """
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, ACSolver
    from pycircuitsim.simulation import _circuit_has_nn, _solve_dc_with_retry

    work_dir.mkdir(parents=True, exist_ok=True)
    deck_path = work_dir / f"directnet_{tag}.sp"
    deck_path.write_text(deck_text)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(deck_path))
        circuit = parser.circuit
        if dc_op is not None:
            dc_solution = dict(dc_op)
            dc_solution.setdefault("0", 0.0)
            op_converged = True
        else:
            has_nn = _circuit_has_nn(circuit)
            guess = circuit.initial_conditions or None

            def _solve(use_gmin: bool):
                s = DCSolver(circuit, initial_guess=guess,
                             use_source_stepping=True, use_gmin_stepping=use_gmin)
                return s, s.solve()

            solver, dc_solution = _solve_dc_with_retry(circuit, has_nn, _solve)
            op_converged = bool(getattr(solver, "_last_solve_converged", True))

        ac_solver = ACSolver(circuit, dc_solution=dc_solution)
        ac_results = ac_solver.solve(frequencies)
    finally:
        logging.disable(logging.NOTSET)

    vout = np.asarray(ac_results[out_node], dtype=complex)
    dc_op_out = {k: float(v) for k, v in dc_solution.items()
                 if k not in ("0", "GND")}
    return vout, op_converged, dc_op_out


# ---------------------------------------------------------------------------
# Extended AC metrics — base agreement + multi-stage figures of merit
# ---------------------------------------------------------------------------
def _phase_deg_unwrapped(v: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.angle(v)))


def _unity_gain_freq(freq: np.ndarray, v: np.ndarray) -> float:
    """First downward 0 dB (|H|=1) crossing, log-f interpolated.

    Returns NaN if the DC gain is already ≤ 0 dB (no crossing from above) or if
    |H| never falls to 0 dB inside the band (fail-to-measure — never
    extrapolate).
    """
    mag_db = 20.0 * np.log10(np.abs(v) + 1e-30)
    if mag_db[0] <= 0.0:
        return float("nan")
    below = np.where(mag_db <= 0.0)[0]
    if below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(freq[0])
    # linear interpolation in (log f, dB) between the last >0 and first ≤0 point
    lf0, lf1 = np.log10(freq[i - 1]), np.log10(freq[i])
    d0, d1 = mag_db[i - 1], mag_db[i]
    lf = lf0 + (0.0 - d0) * (lf1 - lf0) / (d1 - d0)
    return float(10.0 ** lf)


def _phase_at(freq: np.ndarray, ph: np.ndarray, f0: float) -> float:
    """Log-f interpolate an (already unwrapped) phase array at f0."""
    if not np.isfinite(f0):
        return float("nan")
    return float(np.interp(np.log10(f0), np.log10(freq), ph))


def phase_margin_deg(freq: np.ndarray, v: np.ndarray) -> float:
    """Phase margin = 180° − (phase lag from DC to the unity-gain frequency).

    Uses only the phase DROP from DC, so it is independent of the ±180° branch
    np.angle returns for an inverting amplifier — and is computed identically for
    the NN and NGSPICE sides, so ``pm_err`` is convention-robust. NaN if there is
    no in-band unity-gain crossing.
    """
    fu = _unity_gain_freq(freq, v)
    if not np.isfinite(fu):
        return float("nan")
    ph = _phase_deg_unwrapped(v)
    ph_u = _phase_at(freq, ph, fu)
    return 180.0 - (ph[0] - ph_u)


def inband_phase_maxerr_deg(
    freq: np.ndarray, v_test: np.ndarray, v_ref: np.ndarray,
    db_below_peak: float = 3.0,
) -> float:
    """Max passband phase error via the complex transfer RATIO ``v_test/v_ref``.

    Computed as ``max |angle(v_test / v_ref)|`` over the passband (|v_ref| within
    ``db_below_peak`` of its peak — up to the −3 dB corner). The ratio cancels
    the shared response, so a matching response gives ~0° regardless of the
    ±180° branch np.angle assigns an inverting amplifier — no unwrap, no
    co-alignment, no high-frequency-tail artifact. A nonzero value is a genuine
    pole-location (charge-derivative cap) difference.
    """
    mag_r = np.abs(v_ref)
    thresh = mag_r.max() * (10.0 ** (-db_below_peak / 20.0))
    mask = mag_r >= thresh
    if not np.any(mask):
        return float("nan")
    ratio = v_test[mask] / v_ref[mask]
    return float(np.max(np.abs(np.rad2deg(np.angle(ratio)))))


def beyond_corner_phase_err_deg(
    freq: np.ndarray, v_test: np.ndarray, v_ref: np.ndarray,
) -> Tuple[float, float]:
    """Phase error in the band ABOVE the −3 dB corner (G4 visibility metric).

    DIAGNOSTIC ONLY — complements ``inband_phase_maxerr_deg`` (which masks to the
    passband, UP TO the −3 dB corner, and is therefore structurally BLIND to the
    Cgd-feedforward RHP-zero phase). The NN does not reproduce that feedforward
    lag, and it manifests AT/BEYOND the corner — exactly the band the in-band
    metric drops. This measures the NN-vs-NGSPICE phase disagreement there.

    The corner is located via ``minus3db_corner`` on the REFERENCE (NGSPICE)
    response; the error band is ``[f3db_ref, fstop]``. As in the in-band metric,
    the error is taken on the complex transfer RATIO ``v_test / v_ref``, which
    cancels the shared response and the ±180° inverting branch (no unwrap /
    co-alignment / HF-tail artifact).

    Returns ``(maxerr_deg, rmse_deg)``. Returns ``(nan, nan)`` when the reference
    −3 dB corner is not found (no roll-off in band) or fewer than 2 grid points
    lie above it (too few points to characterize the tail) — never raises.
    """
    f3_r = minus3db_corner(freq, v_ref)
    if not np.isfinite(f3_r):
        return float("nan"), float("nan")
    mask = freq >= f3_r
    if int(np.count_nonzero(mask)) < 2:
        return float("nan"), float("nan")
    ratio = v_test[mask] / v_ref[mask]
    err = np.abs(np.rad2deg(np.angle(ratio)))
    return float(np.max(err)), float(np.sqrt(np.mean(err ** 2)))


def ac_metrics_extended(
    freq: np.ndarray, v_test: np.ndarray, v_ref: np.ndarray,
) -> Dict[str, float]:
    """Base mag/phase agreement (``verify_ac.ac_metrics``) plus multi-stage FOMs
    computed identically for BOTH sides.

    ``v_test`` and ``v_ref`` must already be on the common grid ``freq``.
    """
    m = dict(ac_metrics(freq, v_test, v_ref))
    m["phase_maxerr_inband_deg"] = inband_phase_maxerr_deg(freq, v_test, v_ref)
    # G4 visibility (diagnostic, not gated): HF-tail phase ABOVE the −3 dB corner,
    # where the in-band passband mask above is blind to the Cgd RHP-zero lag.
    bc_max, bc_rmse = beyond_corner_phase_err_deg(freq, v_test, v_ref)
    m["phase_maxerr_beyond_corner_deg"] = bc_max
    m["phase_rmse_beyond_corner_deg"] = bc_rmse

    f3_t = minus3db_corner(freq, v_test)
    f3_r = minus3db_corner(freq, v_ref)
    m["f3db_test"] = f3_t
    m["f3db_ref"] = f3_r
    m["f3db_ratio"] = (f3_t / f3_r) if (np.isfinite(f3_t) and np.isfinite(f3_r)
                                        and f3_r > 0) else float("nan")

    gbw_t = _unity_gain_freq(freq, v_test)
    gbw_r = _unity_gain_freq(freq, v_ref)
    m["gbw_test"] = gbw_t
    m["gbw_ref"] = gbw_r
    m["gbw_ratio"] = (gbw_t / gbw_r) if (np.isfinite(gbw_t) and np.isfinite(gbw_r)
                                         and gbw_r > 0) else float("nan")

    pm_t = phase_margin_deg(freq, v_test)
    pm_r = phase_margin_deg(freq, v_ref)
    m["pm_test"] = pm_t
    m["pm_ref"] = pm_r
    m["pm_err"] = (abs(pm_t - pm_r) if (np.isfinite(pm_t) and np.isfinite(pm_r))
                   else float("nan"))

    m["gain0_db_err"] = abs(m["gain0_db"] - m["gain0_db_ref"])
    return m


def fmt_ratio(x: float) -> str:
    return f"{x:.3g}" if np.isfinite(x) else "n/a"


def fmt_hz(x: float) -> str:
    return f"{x:.3g}" if np.isfinite(x) else "n/a"
