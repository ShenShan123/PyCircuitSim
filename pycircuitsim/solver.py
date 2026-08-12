"""
DC and Transient Solvers for linear and non-linear circuits using Modified Nodal Analysis (MNA).

This module implements:
1. DCSolver: Solves for the DC operating point of circuits
2. TransientSolver: Performs time-domain analysis using Backward Euler integration

Both solvers use MNA formulation to construct and solve the circuit equations:

    [G  B] [v]     [i]
    [    ] [ ] =   [ ]
    [C  D] [j]     [e]

Where:
    - G: Conductance matrix from passive components
    - B/C: Voltage source connection matrices
    - v: Node voltages (unknown)
    - j: Voltage source currents (unknown)
    - i: Current source vector (known)
    - e: Voltage source values (known)

The solver handles:
- Linear resistors (conductance stamping)
- Voltage sources (augmented matrix with B and C blocks)
- Current sources (RHS vector stamping)
- Non-linear MOSFETs (Newton-Raphson iteration)
- Capacitors (Backward Euler companion model for transient analysis)
"""
import functools
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, issparse
from scipy.sparse.linalg import spsolve, splu
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import VoltageSource, Capacitor
from pycircuitsim.logger import Logger, IterationInfo

# V7.2.0 Phase 4a' — opt-in sparse-LU column ordering
# (plan §5.2). scipy's spsolve default (COLAMD) can blow up fill on these
# structurally-symmetric MNA matrices; MMD_AT_PLUS_A holds fill flat.
# PERTURBING: a different pivot order rounds differently, so this ships
# default-off and gates with the Phase-3 flag bundle (§8.4). Unset (the
# default) leaves the spsolve path byte-identical. Unknown values raise
# at first use — no silent fallback (audit §B6 silent-green class).
_MNA_ORDERING = os.environ.get("PYCIRCUITSIM_MNA_ORDERING", "").strip()
_VALID_ORDERINGS = ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")
if _MNA_ORDERING and _MNA_ORDERING not in _VALID_ORDERINGS:
    raise ValueError(
        f"PYCIRCUITSIM_MNA_ORDERING={_MNA_ORDERING!r} not one of "
        f"{_VALID_ORDERINGS}")


class SimStepLimit(Exception):
    """A transient sub-interval exceeded its sub-piece budget (V7.5.0).

    Deliberately NOT a RuntimeError: the adaptive-march retry ladder in
    ``TransientSolver.solve`` catches RuntimeError to shrink the local
    timestep, and a budget blow-out must escape that ladder, not feed it.
    """


def _create_mna_matrix(size: int) -> lil_matrix:
    """Create a sparse MNA matrix (LIL format for efficient element-by-element assembly)."""
    return lil_matrix((size, size), dtype=np.float64)


def _solve_mna(mna_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve MNA system Ax=b, using sparse solver if matrix is sparse.

    Raises:
        np.linalg.LinAlgError: if the matrix is singular or the solve returns a
            non-finite vector.

    ``scipy.sparse.linalg.spsolve`` does NOT raise on a singular matrix — it
    emits a ``MatrixRankWarning`` and returns NaN. Left unchecked, those NaNs
    flow into the convergence tests, where every ``dv >= threshold`` comparison
    is False, so the solvers declare convergence and write NaN to the waveform,
    CSV and plot. The dense path already raises, so normalizing the sparse path
    onto the same contract makes the surrounding ``except LinAlgError`` handlers
    (which were written for it) actually reachable.
    """
    if issparse(mna_matrix):
        if _MNA_ORDERING:
            # Phase 4a' opt-in: explicit LU with the chosen column
            # ordering. splu raises on exactly-singular input; the
            # non-finite check below still guards near-singular results,
            # mirroring the spsolve path's contract.
            try:
                lu = splu(mna_matrix.tocsc(), permc_spec=_MNA_ORDERING)
                solution = lu.solve(rhs)
            except RuntimeError as e:  # singular factorisation
                raise np.linalg.LinAlgError(
                    f"Singular MNA matrix (splu/{_MNA_ORDERING}): {e}"
                ) from e
        else:
            solution = spsolve(mna_matrix.tocsr(), rhs)
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                "Singular MNA matrix: sparse solve returned a non-finite "
                "solution (check for floating nodes, duplicate voltage sources, "
                "or a node connected only through GMIN)."
            )
        return solution
    return np.linalg.solve(mna_matrix, rhs)


# Phase 6b — absolute KCL-residual floor for the residual-norm
# acceptance gate. A non-physical fixed point (no DC equilibrium) has a
# residual on the order of a device current (µA–mA); a SPICE-converged
# iterate's residual is orders of magnitude smaller. 1e-6 A sits in the
# gap: well above any converged inverter point, well below a stalled one.
_RESID_ABS_FLOOR: float = 1e-6


def _mna_residual_inf(mna_matrix, rhs: np.ndarray, x: np.ndarray) -> float:
    """Return ‖rhs − A·x‖∞ — the MNA residual at iterate ``x``.

    V6.4 Phase 6b. The NR fixed point of this solver satisfies A·x = rhs
    (full re-stamp formulation: the solve returns absolute voltages, not
    deltas). A small `Δv` between iterations does NOT by itself prove a
    physical solution — a stalled / oscillating iterate can have tiny
    `Δv` yet a large residual. This infinity-norm residual is used as an
    OR-gate in the convergence test and to guard averaged-solution
    acceptance, so a non-physical fixed point is rejected.

    Args:
        mna_matrix: assembled MNA matrix A (sparse LIL or dense).
        rhs: assembled RHS vector b.
        x: candidate solution vector (absolute node/branch values).

    Returns:
        ‖rhs − A·x‖∞ as a finite float; ``inf`` if the product is not finite.
    """
    if issparse(mna_matrix):
        ax = mna_matrix.tocsr().dot(x)
    else:
        ax = mna_matrix.dot(x)
    resid = rhs - ax
    if not np.all(np.isfinite(resid)):
        return float("inf")
    return float(np.max(np.abs(resid)))


def _lm_augment(mna_matrix, lam: float):
    """Return a copy of ``mna_matrix`` with λ added to its diagonal.

    V6.4 Phase 6a — Levenberg-Marquardt damping. Adding ``λ·I`` to the
    MNA Jacobian biases the Newton step toward steepest-descent, which
    re-establishes descent in ‖F(x)‖ when a pure Newton step overshoots.
    The fixed point is unchanged: at convergence the step → 0 so the
    ``λ·I`` term contributes nothing to the accepted solution — LM only
    changes the *path*, never the converged answer.

    Args:
        mna_matrix: assembled MNA matrix (sparse LIL or dense). Not mutated.
        lam: non-negative damping parameter λ.

    Returns:
        A new matrix equal to ``mna_matrix + λ·I``, same type as input.
    """
    if issparse(mna_matrix):
        augmented = mna_matrix.tocsr().copy()
        augmented.setdiag(augmented.diagonal() + lam)
        return augmented
    augmented = mna_matrix.copy()
    n = augmented.shape[0]
    augmented[np.diag_indices(n)] += lam
    return augmented

# --- Module-level MOSFET helpers (used by both DCSolver and TransientSolver) ---

@functools.lru_cache(maxsize=None)
def _mosfet_types() -> tuple:
    """Return tuple of all MOSFET classes (BSIM-CMG, NN, BSIM-AR).

    The tuple is fixed once the model modules are importable, but this is
    called once per component per stamp — 4462 times in a 70-point NN DC
    sweep before V7.0.1, each re-running four ``try: import`` blocks.
    Memoized: the classes are module-level singletons, so the tuple is a
    process-lifetime constant.
    """
    types = []
    try:
        from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
        types.extend([NMOS_CMG, PMOS_CMG])
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN
        types.extend([NMOS_NN, PMOS_NN])
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_bsimar import NMOS_BSIMAR, PMOS_BSIMAR
        types.extend([NMOS_BSIMAR, PMOS_BSIMAR])
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_pfn import NMOS_PFN, PMOS_PFN
        types.extend([NMOS_PFN, PMOS_PFN])
    except ImportError:
        pass
    return tuple(types)


@functools.lru_cache(maxsize=None)
def _pmos_types() -> tuple:
    """Return tuple of all PMOS classes (BSIM-CMG, DirectNet, BSIM-AR).

    The tuple is fixed once the model modules are importable, but this is
    called once per component per stamp — 4462 times in a 70-point NN DC
    sweep before V7.0.1, each re-running four ``try: import`` blocks.
    Memoized: the classes are module-level singletons, so the tuple is a
    process-lifetime constant.
    """
    types = []
    try:
        from pycircuitsim.models.mosfet_cmg import PMOS_CMG
        types.append(PMOS_CMG)
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_directnet import PMOS_NN
        types.append(PMOS_NN)
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_bsimar import PMOS_BSIMAR
        types.append(PMOS_BSIMAR)
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_pfn import PMOS_PFN
        types.append(PMOS_PFN)
    except ImportError:
        pass
    return tuple(types)


@functools.lru_cache(maxsize=None)
def _nn_mosfet_types() -> tuple:
    """Return tuple of NN MOSFET classes eligible for the batched
    forward+Jacobian pre-warm (plan Phase 5): LEVEL=73 DirectNet and —
    since V6.8 — LEVEL=74 BSIMAR. LEVEL=72 BSIM-CMG stays per-device.

    BSIMAR's AR-inference forward is batch-row independent (verified:
    column-sum grad == per-row grad to float noise), and
    ``_MOSFETNNBase.batch_eval`` is a pure ``_eval_cache`` pre-warm with the
    per-device path as fallback — so batching it is the same float-noise
    trade documented for DirectNet (``NN_BATCHED_EVAL=0`` opt-out applies).
    For the ~8-forwards-per-eval AR loop the per-checkpoint grouping is the
    single biggest CPU-gate speedup (e.g. ring-osc: 10 devices → 2 groups).
    """
    types: list = []
    try:
        from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN
        types.extend([NMOS_NN, PMOS_NN])
    except ImportError:
        pass
    try:
        from pycircuitsim.models.mosfet_bsimar import NMOS_BSIMAR, PMOS_BSIMAR
        types.extend([NMOS_BSIMAR, PMOS_BSIMAR])
    except ImportError:
        pass
    # V6.9: LEVEL=75 TabPFN — one-shot forward, row-independent by
    # construction (queries only cross-attend to the frozen context), so
    # the batched pre-warm applies unchanged.
    try:
        from pycircuitsim.models.mosfet_pfn import NMOS_PFN, PMOS_PFN
        types.extend([NMOS_PFN, PMOS_PFN])
    except ImportError:
        pass
    return tuple(types)


def _is_mosfet(component) -> bool:
    """Check if component is any MOSFET variant."""
    return isinstance(component, _mosfet_types())


def _is_nn_mosfet(component) -> bool:
    """Check if component is a LEVEL=73 DirectNet MOSFET.

    R / C / BSIM-CMG (LEVEL=72) stay on the per-device stamping path;
    only DirectNet devices are collected into the batched eval.
    """
    return isinstance(component, _nn_mosfet_types())


def _is_pmos(component) -> bool:
    """Check if component is any PMOS variant."""
    return isinstance(component, _pmos_types())


def _has_non_linear(circuit: Circuit) -> bool:
    """Check if circuit contains non-linear components (MOSFETs)."""
    return any(_is_mosfet(c) for c in circuit.components)


def _topo_key(circuit: Circuit):
    """Cache key for component-membership scans (V7.2.0 Phase 4a).

    ``Circuit._topo_version`` is bumped by ``add_component`` and by every
    in-repo direct ``components`` mutation via ``invalidate_topology``;
    ``len(components)`` is belt-and-braces against an un-hooked mutation
    (every real add/remove pattern changes the length at the point where
    the next solve begins).
    """
    return (getattr(circuit, "_topo_version", None), len(circuit.components))


def _nn_mosfets(circuit: Circuit) -> list:
    """The circuit's NN MOSFETs (LEVEL>=73), cached per topology state.

    The per-NR-iteration ``[c for c in components if _is_nn_mosfet(c)]``
    scan was pure per-device Python executed ~620x per write op (plan
    Phase 4a hoist). Membership only changes when the component list
    does, so the list is cached on the circuit keyed by ``_topo_key``.
    """
    key = _topo_key(circuit)
    cached = getattr(circuit, "_pcs_nn_list_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    lst = [c for c in circuit.components if _is_nn_mosfet(c)]
    circuit._pcs_nn_list_cache = (key, lst)
    return lst


def _has_nn_device(circuit: Circuit) -> bool:
    """Check if circuit contains any NN compact-model device (LEVEL>=73).

    Cached per topology state (Phase 4a) — this is called several times
    per NR iteration.
    """
    key = _topo_key(circuit)
    cached = getattr(circuit, "_pcs_has_nn_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    from pycircuitsim.models.mosfet_directnet import _MOSFETNNBase
    val = any(isinstance(c, _MOSFETNNBase) for c in circuit.components)
    circuit._pcs_has_nn_cache = (key, val)
    return val


def _has_full_stamp_device(circuit: Circuit) -> bool:
    """Any device with the V7.5.0 full 4-terminal stamp (BSIM-CMG L72)?

    Cached per topology state like ``_has_nn_device`` — consulted by the
    oscillation-average acceptance gates each time NR exhausts.
    """
    key = _topo_key(circuit)
    cached = getattr(circuit, "_pcs_has_l72_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    val = any(hasattr(c, "get_terminal_stamp") for c in circuit.components)
    circuit._pcs_has_l72_cache = (key, val)
    return val


def _require_nn_caps(circuit: Circuit) -> None:
    """Tell every NN device in ``circuit`` that this analysis reads caps.

    Only ``TransientSolver`` and ``ACSolver`` consume MOSFET
    capacitances. Marking them here lets a ``.dc`` / ``.op`` run skip the
    qg / qd autograd sweeps that produce those caps — two of the three
    backward passes, ~2x on the NN eval path (V7.0.1). A no-op for
    circuits with no NN device, and ``get_capacitances`` self-heals if
    some other caller needs caps anyway.
    """
    from pycircuitsim.models.mosfet_nn import _MOSFETNNBase
    _MOSFETNNBase.require_caps(circuit.components)


def _batch_eval_nn_mosfets(
    circuit: Circuit, voltages: Dict[str, float],
) -> None:
    """Pre-warm every DirectNet MOSFET's eval cache with one stacked
    forward+Jacobian call per checkpoint (plan Phase 5).

    Collects all LEVEL=73 DirectNet devices and hands them to
    ``_MOSFETNNBase.batch_eval``, which groups by checkpoint and issues a
    single batched NN forward + ``autograd.grad`` per group. The
    subsequent per-device ``_stamp_mosfet_dc`` then hits the warm cache.
    A no-op when the circuit has no DirectNet device.

    Accuracy note: for a group of ONE device the batched path is exactly
    bit-identical to the per-device path (it is the same GEMV). For a
    group of N>1 the stacked GEMM accumulates in a different order than N
    separate GEMVs — a ~1e-8 last-bit difference in the NN output. On a
    plain device that is pure floating-point noise; in a high-gain
    circuit (opamp, ring oscillator) the gain can amplify it into a
    visible node-voltage / startup-phase shift, though the engineering
    metrics (DC gain, oscillation period/amplitude) are preserved. Set
    ``NN_BATCHED_EVAL=0`` to force the per-device path when exact
    multi-device bit-identity is required. The inverter (always 1 NMOS +
    1 PMOS → group-of-one) is bit-identical either way.
    """
    if os.environ.get("NN_BATCHED_EVAL", "1") == "0":
        return  # opt out → per-device _eval fallback (bit-identical)
    nn_mosfets = _nn_mosfets(circuit)  # cached scan (Phase 4a)
    if not nn_mosfets:
        return
    from pycircuitsim.models.mosfet_nn import _MOSFETNNBase
    _MOSFETNNBase.batch_eval(nn_mosfets, voltages)


def _stamp_mosfet_dc(
    mosfet,
    mna_matrix: np.ndarray,
    rhs: np.ndarray,
    node_map: Dict[str, int],
    voltages: Dict[str, float],
    gmin: float,
    limit: bool = True,
) -> None:
    """Stamp MOSFET conductance and NR current source to MNA matrix.

    Shared by DCSolver._stamp_mosfet and TransientSolver._stamp_mosfet_transient.
    The NR linearization stamps g_ds, g_m, g_mb conductances and the equivalent
    current source i_eq = I_leaving(V0) - g_ds*V_ds0 - g_m*V_gs0 - g_mb*V_bs0.

    Devices exposing ``nr_limit_voltages`` (LEVEL=72) are EVALUATED at the
    limited bias V0' that method returns, and the whole companion —
    conductances AND i_eq — linearizes about V0' (V7.5.0, SPICE-style
    damped limiting). The extrapolated line keeps the true derivative, so
    the fixed point is unchanged; the limiter is the identity for small
    steps, so circuits that never stray stamp bit-identically.
    ``limit=False`` (residual probes) evaluates at ``voltages`` as given.
    """
    drain, gate, source, bulk = mosfet.nodes

    limiter = getattr(mosfet, "nr_limit_voltages", None) if limit else None
    if limiter is not None:
        voltages = limiter(voltages)
        # Even the limited bias can defeat the OSDI internal-node solve
        # for some geometries. Bisect back toward the last evaluable
        # anchor until the model evaluates; the convergence guard sees
        # _nr_limited=True, so such an iteration is never accepted.
        for _ in range(8):
            try:
                mosfet.get_conductance(voltages)
                break
            except RuntimeError:
                retreat = mosfet.nr_retreat_voltages()
                if retreat is None:
                    raise
                voltages = retreat

    # --- Full 4-terminal stamp (V7.5.0, LEVEL=72) ---
    # The classic 3-conductance companion below linearizes only the
    # channel: its Jacobian comes from the gm/gds/gmb opvars and its
    # current routes drain-to-source. BSIM-CMG terminal currents also
    # carry body-junction and gate-leakage components — dominant at high
    # temperature — that the opvars know nothing about, so circuit NR
    # cycles around a residual its Jacobian cannot see. Devices exposing
    # get_terminal_stamp() stamp all four KCL rows from the condensed
    # OSDI Jacobian instead, exactly as NGSPICE loads the model.
    full_stamp = getattr(mosfet, "get_terminal_stamp", None)
    if full_stamp is not None:
        i_out, g4 = full_stamp(voltages)
        idx = [node_map.get(n) if n not in ("0", "GND") else None
               for n in mosfet.nodes]
        v_eval = [voltages.get(n, 0.0) for n in mosfet.nodes]
        # SPICE GMIN across d-s and both body junctions (d-b, s-b):
        # keeps every terminal row invertible when the device is off.
        for a, b in ((0, 2), (0, 3), (2, 3)):
            g4[a, a] += gmin
            g4[b, b] += gmin
            g4[a, b] -= gmin
            g4[b, a] -= gmin
        for t in range(4):
            row = idx[t]
            if row is None:
                continue
            i_eq = i_out[t]
            for j in range(4):
                i_eq -= g4[t, j] * v_eval[j]
                col = idx[j]
                if col is not None:
                    mna_matrix[row, col] += g4[t, j]
            rhs[row] -= i_eq
        return

    # Get conductances (3-tuple: g_ds, g_m, g_mb)
    g_ds, g_m, g_mb = mosfet.get_conductance(voltages)

    i_ds = mosfet.calculate_current(voltages)
    g_ds = max(g_ds, gmin)  # SPICE GMIN floor

    # --- Stamp conductances ---
    # g_ds between drain and source
    if drain != "0" and drain in node_map:
        d_idx = node_map[drain]
        mna_matrix[d_idx, d_idx] += g_ds
    if source != "0" and source in node_map:
        s_idx = node_map[source]
        mna_matrix[s_idx, s_idx] += g_ds
    if drain != "0" and drain in node_map and source != "0" and source in node_map:
        d_idx, s_idx = node_map[drain], node_map[source]
        mna_matrix[d_idx, s_idx] -= g_ds
        mna_matrix[s_idx, d_idx] -= g_ds

    # g_m transconductance (VCCS: gate controls drain current)
    if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
        mna_matrix[node_map[drain], node_map[gate]] += g_m
    if drain != "0" and drain in node_map and source != "0" and source in node_map:
        mna_matrix[node_map[drain], node_map[source]] -= g_m
    if gate != "0" and gate in node_map and source != "0" and source in node_map:
        mna_matrix[node_map[source], node_map[gate]] -= g_m
    if source != "0" and source in node_map:
        mna_matrix[node_map[source], node_map[source]] += g_m

    # g_mb bulk transconductance: i_d = gmb * (v_b - v_s)
    # Full 4-entry VCCS stamp (matching AC solver pattern at lines 2002-2023).
    if abs(g_mb) > 1e-12 and bulk != source:
        # Stamp for drain equation
        if bulk != "0" and bulk in node_map and drain != "0" and drain in node_map:
            mna_matrix[node_map[drain], node_map[bulk]] += g_mb
        if source != "0" and source in node_map and drain != "0" and drain in node_map:
            mna_matrix[node_map[drain], node_map[source]] -= g_mb
        # Stamp for source equation
        if bulk != "0" and bulk in node_map and source != "0" and source in node_map:
            mna_matrix[node_map[source], node_map[bulk]] -= g_mb
        if source != "0" and source in node_map:
            mna_matrix[node_map[source], node_map[source]] += g_mb

    # --- Stamp NR equivalent current source to RHS ---
    v_d, v_g = voltages.get(drain, 0.0), voltages.get(gate, 0.0)
    v_s, v_b = voltages.get(source, 0.0), voltages.get(bulk, 0.0)
    v_ds, v_gs, v_bs = v_d - v_s, v_g - v_s, v_b - v_s

    # Convert to "leaving drain" convention:
    # NMOS: i_ds positive = leaving drain; PMOS: i_ds positive = INTO drain
    i_leaving = -i_ds if _is_pmos(mosfet) else i_ds
    i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs

    if drain != "0" and drain in node_map:
        rhs[node_map[drain]] -= i_eq
    if source != "0" and source in node_map:
        rhs[node_map[source]] += i_eq


# ─────────────────────────────────────────────────────────────────────
# V7.2.0 Phase 3b — batched COO stamping for NN MOSFETs (opt-in)
#
# PYCIRCUITSIM_BATCHED_STAMP=1 replaces the per-device lil_matrix writes
# for LEVEL>=73 devices with one COO assembly per NR iteration:
# A = lil.tocsr() + coo.tocsr(). LEVEL=72 (OSDI) and all passives stay
# on the scalar path. PERTURBING: summing duplicate coordinates in
# CSR-conversion order is a different float accumulation order than
# sequential lil writes — deterministic run-to-run, but not bit-equal to
# the scalar path — so this ships default-off and gates with the §8.4
# flag bundle. The two §4.3 hazards are handled explicitly below:
# max(g_ds, gmin) is applied per device BEFORE accumulation, and the
# dynamic abs(g_mb) > 1e-12 stamp mask reproduces the scalar branch.
# ─────────────────────────────────────────────────────────────────────
_BATCHED_STAMP = os.environ.get("PYCIRCUITSIM_BATCHED_STAMP", "0") == "1"

#: V7.5.0 diagnostic: PYCIRCUITSIM_NR_TRACE=1 prints one line per NR
#: iteration (worst node, delta, active limiters) in the transient loop.
_NR_TRACE = os.environ.get("PYCIRCUITSIM_NR_TRACE", "0") == "1"


def _nn_stamp_indices(circuit: Circuit, node_map: Dict[str, int]):
    """Static per-NN-device index arrays for the batched stamp.

    Topology-invariant, so cached on the circuit keyed by ``_topo_key``
    (``node_map`` is itself topology-cached, so the key covers it).
    Ground ("0"/"GND") and absent nodes map to -1.
    """
    key = _topo_key(circuit)
    cached = getattr(circuit, "_pcs_stamp_idx_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    devs = _nn_mosfets(circuit)
    n = len(devs)
    d_i = np.empty(n, dtype=np.int64)
    g_i = np.empty(n, dtype=np.int64)
    s_i = np.empty(n, dtype=np.int64)
    b_i = np.empty(n, dtype=np.int64)
    sign = np.ones(n)          # +1 NMOS, -1 PMOS ("leaving drain" flip)
    b_ne_s = np.empty(n, dtype=bool)
    for i, m in enumerate(devs):
        dn, gn, sn, bn = m.nodes
        d_i[i] = node_map.get(dn, -1)
        g_i[i] = node_map.get(gn, -1)
        s_i[i] = node_map.get(sn, -1)
        b_i[i] = node_map.get(bn, -1)
        if _is_pmos(m):
            sign[i] = -1.0
        b_ne_s[i] = bn != sn
    data = (devs, d_i, g_i, s_i, b_i, sign, b_ne_s)
    circuit._pcs_stamp_idx_cache = (key, data)
    return data


def _stamp_nn_mosfets_batched(
    circuit: Circuit,
    node_map: Dict[str, int],
    matrix_size: int,
    rhs: np.ndarray,
    voltages: Dict[str, float],
    v_arr: np.ndarray,
    gmin: float,
    tran_solver=None,
):
    """Batched replacement for the per-NN-device stamp loop.

    Accumulates the NR current sources into ``rhs`` (np.add.at, device
    order) and returns the NN devices' conductance/transcap block as one
    CSR matrix to be added to the lil-born CSR — or ``None`` when the
    circuit has no NN device. When ``tran_solver`` is given, the
    charge-companion (transcap) block of ``_stamp_mosfet_transient`` is
    included; otherwise this is the DC stamp only.

    The value expressions reproduce ``_stamp_mosfet_dc`` /
    ``_stamp_mosfet_transient`` term for term; only the accumulation
    order of coincident matrix entries differs (the documented
    perturbation this flag gates).
    """
    from scipy.sparse import coo_matrix

    idx_data = _nn_stamp_indices(circuit, node_map)
    devs, d_i, g_i, s_i, b_i, sign, b_ne_s = idx_data
    n = len(devs)
    if n == 0:
        return None

    # Padded node-voltage vector: index -1 lands on the appended 0.0,
    # matching ``voltages.get(node, 0.0)`` for ground/absent nodes.
    v_pad = np.concatenate([v_arr, np.zeros(1)])

    gds = np.empty(n)
    gm = np.empty(n)
    gmb = np.empty(n)
    ids = np.empty(n)
    for i, m in enumerate(devs):
        gds[i], gm[i], gmb[i] = m.get_conductance(voltages)
        ids[i] = m.calculate_current(voltages)
    gds = np.maximum(gds, gmin)   # §4.3: per-device floor BEFORE any sum

    v_d = v_pad[d_i]
    v_g = v_pad[g_i]
    v_s = v_pad[s_i]
    v_b = v_pad[b_i]
    v_ds = v_d - v_s
    v_gs = v_g - v_s
    v_bs = v_b - v_s

    i_leaving = sign * ids
    i_eq = i_leaving - gds * v_ds - gm * v_gs - gmb * v_bs

    d_ok = d_i >= 0
    g_ok = g_i >= 0
    s_ok = s_i >= 0
    b_ok = b_i >= 0
    ds_ok = d_ok & s_ok

    np.add.at(rhs, d_i[d_ok], -i_eq[d_ok])
    np.add.at(rhs, s_i[s_ok], i_eq[s_ok])

    rows: list = []
    cols: list = []
    vals: list = []

    def _add(r: np.ndarray, c: np.ndarray, v: np.ndarray,
             valid: np.ndarray) -> None:
        if valid.any():
            rows.append(r[valid])
            cols.append(c[valid])
            vals.append(v[valid])

    # gds between drain and source (4 entries)
    _add(d_i, d_i, gds, d_ok)
    _add(s_i, s_i, gds, s_ok)
    _add(d_i, s_i, -gds, ds_ok)
    _add(s_i, d_i, -gds, ds_ok)
    # g_m VCCS (4 entries)
    _add(d_i, g_i, gm, g_ok & d_ok)
    _add(d_i, s_i, -gm, ds_ok)
    _add(s_i, g_i, -gm, g_ok & s_ok)
    _add(s_i, s_i, gm, s_ok)
    # g_mb VCCS (4 entries, dynamic mask mirrors the scalar branch)
    mb = (np.abs(gmb) > 1e-12) & b_ne_s
    _add(d_i, b_i, gmb, mb & b_ok & d_ok)
    _add(d_i, s_i, -gmb, mb & ds_ok)
    _add(s_i, b_i, -gmb, mb & b_ok & s_ok)
    _add(s_i, s_i, gmb, mb & s_ok)

    # ── transcap companion block (transient only) ────────────────────
    if tran_solver is not None:
        act = [i for i, m in enumerate(devs)
               if getattr(m, "_q_prev", None) is not None]
        if act:
            k = len(act)
            dt = tran_solver._current_dt
            method = getattr(tran_solver, "_integration_method", "trap")
            qg = np.empty(k)
            qd = np.empty(k)
            cgg = np.empty(k)
            cgd = np.empty(k)
            cgs = np.empty(k)
            cdg = np.empty(k)
            cdd = np.empty(k)
            h_g = np.empty(k)
            h_d = np.empty(k)
            coeff = np.empty(k)
            for j, i in enumerate(act):
                m = devs[i]
                charges = m.get_charges(voltages)
                caps = m.get_capacitances(voltages)
                qg[j] = charges["qg"]
                qd[j] = charges["qd"]
                cgg[j] = caps.get("cgg", 0.0)
                cgd[j] = caps.get("cgd", 0.0)
                cgs[j] = caps.get("cgs", 0.0)
                cdg[j] = caps.get("cdg", 0.0)
                cdd[j] = caps.get("cdd", 0.0)
                q_prev = m._q_prev
                q_prev2 = getattr(m, "_q_prev2", None)
                if method == 'bdf2' and q_prev2 is not None:
                    c_j = 1.5 / dt
                    h_g[j] = (2.0 / dt) * q_prev["qg"] - (0.5 / dt) * q_prev2["qg"]
                    h_d[j] = (2.0 / dt) * q_prev["qd"] - (0.5 / dt) * q_prev2["qd"]
                elif method == 'trap' or (method == 'bdf2' and q_prev2 is None):
                    c_j = 2.0 / dt
                    h_g[j] = c_j * q_prev["qg"] + getattr(m, '_i_prev_gate', 0.0)
                    h_d[j] = c_j * q_prev["qd"] + getattr(m, '_i_prev_drain', 0.0)
                else:
                    c_j = 1.0 / dt
                    h_g[j] = c_j * q_prev["qg"]
                    h_d[j] = c_j * q_prev["qd"]
                coeff[j] = c_j

            a = np.asarray(act)
            da, ga, sa = d_i[a], g_i[a], s_i[a]
            vg_a, vd_a, vs_a = v_pad[ga], v_pad[da], v_pad[sa]
            i_g_cap = coeff * qg - h_g
            i_d_cap = coeff * qd - h_d
            cds = -(cdg + cdd)
            csg = -(cgg + cdg)
            csd = -(cgd + cdd)
            css = -(cgs + cds)
            if os.environ.get("NN_SYMMETRIC_CAPS", "0") == "1":
                cgd = cdg = 0.5 * (cgd + cdg)
                cgs = csg = 0.5 * (cgs + csg)
                cds = csd = 0.5 * (cds + csd)
                css = -(cgs + cds)
            scale = coeff

            ga_ok = ga >= 0
            da_ok = da >= 0
            sa_ok = sa >= 0
            _add(ga, ga, scale * cgg, ga_ok)
            _add(ga, da, scale * cgd, ga_ok & da_ok)
            _add(ga, sa, scale * cgs, ga_ok & sa_ok)
            _add(da, ga, scale * cdg, da_ok & ga_ok)
            _add(da, da, scale * cdd, da_ok)
            _add(da, sa, scale * cds, da_ok & sa_ok)
            _add(sa, ga, scale * csg, sa_ok & ga_ok)
            _add(sa, da, scale * csd, sa_ok & da_ok)
            _add(sa, sa, scale * css, sa_ok)

            e_g = i_g_cap - scale * (cgg * vg_a + cgd * vd_a + cgs * vs_a)
            e_d = i_d_cap - scale * (cdg * vg_a + cdd * vd_a + cds * vs_a)
            e_s = -(e_g + e_d)
            np.add.at(rhs, ga[ga_ok], -e_g[ga_ok])
            np.add.at(rhs, da[da_ok], -e_d[da_ok])
            np.add.at(rhs, sa[sa_ok], -e_s[sa_ok])

    if not rows:
        return None
    coo = coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(matrix_size, matrix_size))
    return coo.tocsr()


class DCSolver:
    """
    DC Solver for linear and non-linear circuits using Modified Nodal Analysis.

    The DCSolver constructs the MNA matrix and solves for the DC operating
    point of a circuit. It handles linear components (resistors, voltage
    sources, current sources) and non-linear components (MOSFETs) using
    Newton-Raphson iteration.

    Attributes:
        circuit: Circuit object containing components and topology
        tolerance: Convergence tolerance for Newton-Raphson
        max_iterations: Maximum Newton-Raphson iterations
    """

    def __init__(self, circuit: Circuit, tolerance: float = 1e-9, max_iterations: int = 50,
                 output_file: Optional[Path] = None, initial_guess: Optional[Dict[str, float]] = None,
                 logger: Optional[Logger] = None, use_source_stepping: bool = True,
                 source_stepping_steps: int = 20,
                 damping_factor: float = 1.0,
                 reltol: float = 1e-4, vntol: float = 1e-7, gmin: float = 1e-12,
                 use_gmin_stepping: bool = False, force_ic: bool = False,
                 dv_limit: Optional[float] = None):
        """
        Initialize the DC Solver.

        Args:
            circuit: Circuit object to solve
            tolerance: Convergence tolerance for Newton-Raphson (default: 1e-9)
            max_iterations: Maximum Newton-Raphson iterations (default: 50)
            output_file: Optional path to output log file (.lis file)
            initial_guess: Optional initial voltage guess for Newton-Raphson (dictionary of node->voltage)
            logger: Optional external Logger instance for logging (reuses existing logger)
            use_source_stepping: Enable source stepping homotopy (default: True)
            source_stepping_steps: Number of source stepping steps (default: 20)
            damping_factor: Initial damping factor for Newton-Raphson (default: 1.0, 0.5 = aggressive damping)
            reltol: Relative convergence tolerance (default: 1e-4, tighter than SPICE 1e-3)
            vntol: Absolute voltage tolerance (default: 1e-7 V, tighter than SPICE 1e-6)
            gmin: Minimum MOSFET channel conductance (SPICE GMIN, default: 1e-12 S)
            use_gmin_stepping: Enable DC GMIN stepping for bistable convergence (default: False)
            force_ic: Enforce .ic as voltage constraints, not just initial guess (default: False)
            dv_limit: Per-iteration, per-node |ΔV| trust-region cap in volts
                (default None = OFF, historical behaviour). There is otherwise
                NO voltage limiting on the LEVEL=72 path: `mosfet_cmg.py` hands
                raw node voltages to OSDI, so one bad Newton step (measured:
                a 2 µA current source into a node still at the 1e-12 S gds
                floor asks for ΔV ≈ 2e6 V) reaches the compact model as
                `g=-505225 V` and OSDI's internal-node solve raises. Capping
                the step is SPICE's own answer to this. NN circuits (LEVEL
                73/74/75) already cap at one supply rail unconditionally; a
                value here overrides that cap for them too.
                PERTURBING, hence default-off: the cap changes the Newton PATH
                (not the fixed point), so it is not bit-identical on a circuit
                where it engages.

                MEASURED DEAD END, kept here so it is not retried: also
                ramping the CURRENT sources with the source-stepping homotopy
                (source stepping scales only VoltageSource, so the full bias
                current is applied while the rails sit at 1/N of nominal) does
                NOT fix that blow-up on its own — the same OSDI raise comes
                back with g = -25261 V instead of -505225 V, i.e. 1/20 of the
                step, still 4 orders of magnitude past the rail. Limiting the
                step is the fix; scaling the excitation only rescales it.
        """
        self.circuit = circuit
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.output_file = output_file
        self.logger = logger  # Use external logger if provided
        self.initial_guess = initial_guess
        self.use_source_stepping = use_source_stepping
        self.source_stepping_steps = source_stepping_steps
        self.damping_factor = damping_factor
        self.reltol = reltol
        self.vntol = vntol
        self.gmin = gmin
        self.use_gmin_stepping = use_gmin_stepping
        self.force_ic = force_ic
        self.dv_limit = dv_limit
        self.last_solution: Optional[Dict[str, float]] = None
        self._owns_logger = False  # Track if we created the logger (for cleanup)
        # V5 Phase A retry-design: True if the last `solve()` reached
        # SPICE convergence AND the returned voltage vector is finite.
        # The simulation orchestrator inspects this to decide whether
        # to retry with GMIN stepping enabled.
        self._last_solve_converged: bool = False
        # V6.4.6 Phase 1: KCL-residual telemetry for the force_ic path so
        # the SRAM probe can gate acceptance on a TRUE convergence check
        # (the released solution's MNA residual) instead of a stale flag.
        # These are set only on the force_ic release path; they stay None
        # on every other solve so a test reading them can tell whether the
        # value is live.
        self._last_dc_residual: Optional[float] = None
        self._last_dc_resid_threshold: Optional[float] = None
        self._last_residual_ok: Optional[bool] = None

    def __enter__(self):
        """
        Enter the context manager and initialize the logger.

        Returns:
            DCSolver instance
        """
        if self.logger is None and self.output_file:
            # Create new logger only if we don't have one and output_file is specified
            netlist_name = getattr(self.circuit, 'netlist', 'Unknown')
            self.logger = Logger(netlist_name, self.output_file)
            self.logger.__enter__()
            self._owns_logger = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and close the logger.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if self.logger and self._owns_logger:
            self.logger.__exit__(exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions

    def solve(self, skip_header: bool = False) -> Dict[str, float]:
        """
        Solve the circuit for DC operating point.

        This method checks if the circuit contains non-linear components (MOSFETs).
        - If linear: constructs the MNA matrix and solves directly
        - If non-linear: uses Newton-Raphson iteration

        The MNA matrix has size (num_nodes + num_voltage_sources) x (num_nodes + num_voltage_sources).

        Args:
            skip_header: If True, skip logging header (for use in DC sweep)

        Returns:
            Dictionary mapping node names to voltage values (including ground at 0V)

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
            RuntimeError: If Newton-Raphson fails to converge
        """
        # Log header and circuit summary if logger is available
        if self.logger and not skip_header:
            self.logger.log_header("DC Operating Point Analysis", {})
            num_nodes = len(self.circuit.get_nodes())
            num_vsources = self.circuit.count_voltage_sources()
            self.logger.log_circuit_summary(
                component_count=len(self.circuit.components),
                node_count=num_nodes,
                vsource_count=num_vsources
            )

        # Check if circuit has non-linear components
        has_non_linear = self._has_non_linear_components()

        if has_non_linear:
            # Use Newton-Raphson for non-linear circuits.
            #
            # `_solve_newton` restores the source-stepping ramp on its way out,
            # but only on the path that RETURNS: a raise (NR blow-up, singular
            # matrix, a compact model rejecting a wild voltage) leaves every
            # VoltageSource at 1/N of nominal. The next solve on the same
            # circuit then reads those as its "original" values and ramps them
            # DOWN AGAIN — measured on the AnalogGym amplifier bench: one
            # failed solve, and the following one converged happily with
            # vdd = 0.65/400 V. A silent wrong answer that survives across
            # every remaining point of a sweep, so the snapshot is restored
            # here on the exception path. The success path is untouched.
            _saved_v = [(c, c.voltage) for c in self.circuit.components
                        if isinstance(c, VoltageSource)]
            try:
                solution = self._solve_newton()
            except BaseException:
                for _comp, _value in _saved_v:
                    _comp.voltage = _value
                raise
        else:
            # Direct solve for linear circuits
            solution = self._solve_linear()
            # Linear solve either succeeds or raises — if we got here,
            # converged.
            self._last_solve_converged = True

        # Store the solution for potential reuse
        self.last_solution = solution.copy()

        return solution

    def _has_non_linear_components(self) -> bool:
        """Check if circuit contains non-linear components (MOSFETs)."""
        return _has_non_linear(self.circuit)

    def _solve_linear(self) -> Dict[str, float]:
        """
        Solve linear circuit directly using MNA.

        Returns:
            Dictionary mapping node names to voltage values

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
        """
        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Initialize MNA matrix and RHS vector
        mna_matrix = _create_mna_matrix(matrix_size)
        rhs = np.zeros(matrix_size)

        # Stamp conductances (G matrix) and current sources (RHS)
        for component in self.circuit.components:
            component.stamp_conductance(mna_matrix, node_map)
            component.stamp_rhs(rhs, node_map)

        # Handle voltage sources (B and C matrices)
        self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=None)

        # Solve the linear system
        try:
            solution = _solve_mna(mna_matrix, rhs)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Circuit is singular or unsolvable. Check for floating nodes or short circuits."
            ) from e

        # Extract node voltages from solution
        voltages = self._extract_voltages(solution, nodes)

        # Extract and store voltage source currents
        self._store_source_currents(solution, nodes)

        # Log iteration for linear circuit (single iteration)
        if self.logger:
            # Calculate device currents
            currents = {}
            for comp in self.circuit.components:
                try:
                    current = comp.calculate_current(voltages)
                    currents[comp.name] = current
                except (NotImplementedError, AttributeError):
                    # Skip components that don't support current calculation
                    pass

            # Create iteration info
            iter_info = IterationInfo(
                iteration=0,
                voltages=voltages.copy(),
                deltas={},  # No deltas for linear solve
                currents=currents,
                conductances={}  # No conductances for linear solve
            )
            self.logger.log_iteration(point_num=0, iter_info=iter_info)

            # Log convergence
            self.logger.log_convergence(
                point_num=0,
                converged=True,
                iterations=1,
                tolerance=0.0  # Linear solve has exact solution
            )

        return voltages

    def _apply_gmin_stepping(self, mna_matrix, node_map: Dict[str, int], gmin: float) -> None:
        """Add minimum conductance from each node to ground for convergence aid."""
        for node, idx in node_map.items():
            mna_matrix[idx, idx] += gmin

    def _solve_newton(self) -> Dict[str, float]:
        """
        Solve non-linear circuit using Newton-Raphson iteration.

        Features:
        - Source stepping homotopy for improved convergence
        - GMIN stepping (opt-in) for bistable circuits (SRAM latches)
        - Adaptive damping with supply-relative thresholds
        - Oscillation detection with averaged-solution acceptance
        - Hard .ic mode (force_ic) via temporary voltage source constraints

        Returns:
            Dictionary mapping node names to voltage values

        Raises:
            RuntimeError: If Newton-Raphson fails to converge
            np.linalg.LinAlgError: If the circuit matrix is singular
        """
        # Reset damping to default for each solve
        self.damping_factor = 1.0

        # V7.5.0: forget NR-limiting anchors from a previous solve — the
        # first iteration then evaluates at the (window-clamped) initial
        # guess, which for sweep continuations is the previous solution.
        for component in self.circuit.components:
            reset = getattr(component, "reset_nr_limits", None)
            if reset is not None:
                reset()

        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)

        # --- Force IC: add temporary voltage source constraints ---
        _ic_temp_sources: List[VoltageSource] = []
        if self.force_ic and self.initial_guess:
            # Find nodes already constrained by existing voltage sources
            vs_constrained = set()
            for comp in self.circuit.components:
                if isinstance(comp, VoltageSource):
                    if comp.nodes[1] in ("0", "GND"):
                        vs_constrained.add(comp.nodes[0])
                    elif comp.nodes[0] in ("0", "GND"):
                        vs_constrained.add(comp.nodes[1])
            # Add temp VS for IC nodes not already constrained
            for node_name, voltage in self.initial_guess.items():
                if (node_name not in ("0", "GND")
                        and node_name not in vs_constrained
                        and node_name in node_map):
                    vs = VoltageSource(f"_V_ic_{node_name}", [node_name, "0"], voltage)
                    self.circuit.components.append(vs)
                    _ic_temp_sources.append(vs)
            if _ic_temp_sources:
                self.circuit.invalidate_topology()

        num_voltage_sources = self.circuit.count_voltage_sources()
        matrix_size = num_nodes + num_voltage_sources

        # Store original voltage source values for source stepping
        original_voltages = []
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                original_voltages.append(component.voltage)

        # Estimate supply voltage for supply-relative damping threshold
        max_vs_voltage = max((abs(v) for v in original_voltages), default=1.0) or 1.0

        # --- GMIN stepping schedule ---
        # V5 Phase A retry-design (2026-05-07): reduced from the 4-level
        # schedule [1e-6, 1e-8, 1e-10, self.gmin] to a 2-level schedule
        # [1e-8, self.gmin] when retry is invoked. Source stepping is
        # tried once per level, so 4 levels x 5 steps = 20 NR sweeps per
        # GMIN-on solve, which dominated the verify wall-time when GMIN
        # was default-on. 2 levels keeps the homotopy useful for the
        # cells that need it (TSMC5 BSIMAR-M VTC trip-point overflow)
        # while halving the slow-path cost.
        if self.use_gmin_stepping:
            if _has_nn_device(self.circuit):
                # V5 Phase A 2-level schedule (NN VTC trip-point path) —
                # unchanged, so the NN retry cost stays what was measured.
                gmin_schedule = [1e-8, self.gmin]
            else:
                # V7.5.0 — BSIM-CMG hard starts (hot-temperature operating
                # points with forward body junctions) need a real homotopy:
                # a large node shunt nearly linearizes the first solve and
                # each decade walks the solution down by continuation, as
                # NGSPICE's dynamic gmin stepping does. The 2-level
                # schedule jumped from a solved 1e-8 world straight to
                # 1e-12 and lost the 125 C amplifier start.
                gmin_schedule = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
                                 1e-8, 1e-9, 1e-10, 1e-11, self.gmin]
        else:
            gmin_schedule = [self.gmin]

        voltages: Dict[str, float] = {}
        final_converged = False
        _gmin_last_good: Optional[Dict[str, float]] = None

        for gmin_level in gmin_schedule:
            # Source stepping: gradually increase voltage source values
            num_steps = self.source_stepping_steps if (self._has_non_linear_components() and self.use_source_stepping) else 1
            for step in range(num_steps):
                # Scale voltage sources
                scale = (step + 1) / num_steps
                vs_idx = 0
                for component in self.circuit.components:
                    if isinstance(component, VoltageSource):
                        component.voltage = original_voltages[vs_idx] * scale
                        vs_idx += 1

                # Initial guess
                if step == 0 and gmin_level == gmin_schedule[0]:
                    if self.initial_guess is not None:
                        voltages = {node: 0.0 for node in nodes}
                        for node, voltage in self.initial_guess.items():
                            if node in voltages:
                                voltages[node] = voltage
                    else:
                        voltages = {node: 0.0 for node in nodes}
                voltages["0"] = 0.0
                voltages["GND"] = 0.0

                # --- Adaptive damping state ---
                damping = 1.0
                prev_max_delta = float('inf')
                stuck_counter = 0
                voltage_history: List[Dict[str, float]] = []

                # V7.5.0: fresh NR-limiting anchors per NR sweep — a stale
                # anchor from a failed gmin level / source step distorts
                # the first evaluations of this one (see the transient
                # solver's reset for the measured failure mode).
                for component in self.circuit.components:
                    reset = getattr(component, "reset_nr_limits", None)
                    if reset is not None:
                        reset()

                # --- Levenberg-Marquardt damping state (Phase 6a) ---
                # lm_lambda is the current λ; prev_residual is ‖F(x)‖∞ at
                # the last accepted iterate. LM only engages when a Newton
                # step fails to reduce the residual.
                lm_lambda = 0.0
                prev_residual = float('inf')

                # V7.2.0 Phase 2d — per-iteration node state as a vector.
                # ``v_arr`` mirrors ``voltages[node] for node in nodes``
                # and is maintained by the update step below; the
                # VS-constrained set is topology-static, so it is built
                # once per source step, not once per NR iteration.
                n_nodes = len(nodes)
                v_arr = np.fromiter(
                    (voltages[nd] for nd in nodes), np.float64, n_nodes)
                vs_constrained_nodes = set()
                for component in self.circuit.components:
                    if isinstance(component, VoltageSource):
                        pos_node = component.nodes[0]
                        neg_node = component.nodes[1]
                        if neg_node == "0":
                            vs_constrained_nodes.add(pos_node)
                        elif pos_node == "0":
                            vs_constrained_nodes.add(neg_node)
                vs_mask = np.fromiter(
                    (nd in vs_constrained_nodes for nd in nodes),
                    bool, n_nodes)

                # Newton-Raphson iteration for this source step
                nr_converged = False
                max_change = 0.0
                for iteration in range(self.max_iterations // num_steps):
                    # Initialize MNA matrix and RHS vector
                    mna_matrix = _create_mna_matrix(matrix_size)
                    rhs = np.zeros(matrix_size)

                    # Stamp linear components (resistors, current sources)
                    for component in self.circuit.components:
                        if not _is_mosfet(component):
                            component.stamp_conductance(mna_matrix, node_map)
                            component.stamp_rhs(rhs, node_map)

                    # Batched DirectNet (LEVEL=73) forward+Jacobian: one
                    # stacked NN call per checkpoint pre-warms the eval
                    # cache so the per-device stamps below hit it. Perf
                    # only — see _batch_eval_nn_mosfets for the accuracy
                    # note (exact for group-of-one, e.g. the inverter).
                    _batch_eval_nn_mosfets(self.circuit, voltages)

                    # Stamp MOSFET conductances and currents.
                    # Phase 3b (opt-in): NN devices go through one COO
                    # assembly; LEVEL=72/others stay scalar.
                    nn_extra = None
                    if _BATCHED_STAMP:
                        nn_extra = _stamp_nn_mosfets_batched(
                            self.circuit, node_map, matrix_size, rhs,
                            voltages, v_arr, self.gmin)
                        for component in self.circuit.components:
                            if (_is_mosfet(component)
                                    and not _is_nn_mosfet(component)):
                                self._stamp_mosfet(
                                    component, mna_matrix, rhs,
                                    node_map, voltages)
                    else:
                        for component in self.circuit.components:
                            if _is_mosfet(component):
                                self._stamp_mosfet(component, mna_matrix, rhs, node_map, voltages)

                    # Handle voltage sources (B and C matrices)
                    self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, voltages=voltages)

                    # Apply node-level GMIN (conductance from each node to ground)
                    if gmin_level > self.gmin:
                        self._apply_gmin_stepping(mna_matrix, node_map, gmin_level)

                    # V7.2.0 Phase 4a — one LIL→CSR conversion per
                    # iteration, shared by the residual, the solve, and
                    # the LM ladder (``tocsr`` on a CSR input is a
                    # no-op, so the callees are unchanged).
                    mna_solve = (
                        mna_matrix.tocsr() if issparse(mna_matrix)
                        else mna_matrix)
                    if nn_extra is not None:
                        mna_solve = mna_solve + nn_extra

                    # Current iterate as a full MNA vector (node voltages in
                    # the leading slots; branch-current slots left at 0 —
                    # they only scale the residual, not the descent test).
                    current_iterate = np.zeros(matrix_size)
                    current_iterate[:n_nodes] = v_arr

                    # MNA residual ‖b − A·v‖∞ at the current iterate
                    # (Phase 6b): the nonlinear KCL mismatch before the
                    # Newton step. LM uses this to detect overshoot.
                    iter_residual = _mna_residual_inf(mna_solve, rhs, current_iterate)

                    # Solve the MNA system
                    try:
                        solution = _solve_mna(mna_solve, rhs)
                    except (np.linalg.LinAlgError, RuntimeError) as e:
                        raise np.linalg.LinAlgError(
                            f"Circuit matrix is singular at source step {step + 1}, iteration {iteration + 1}. "
                            f"Check circuit topology or initial guess."
                        ) from e

                    # --- Levenberg-Marquardt damping (Phase 6a) ---
                    # If the residual did not decrease vs the previous
                    # accepted iterate, the pure Newton step overshot.
                    # Add λ·I to the Jacobian and re-solve, scaling λ ×10
                    # (Nielsen rule) until the candidate's residual
                    # improves; shrink λ ÷3 on acceptance. Sits ALONGSIDE
                    # the trust-region rail cap below — it does not
                    # replace it. The fixed point is unchanged: λ only
                    # reshapes the step direction, never the converged
                    # answer.
                    if (_has_nn_device(self.circuit)
                            and np.isfinite(prev_residual)
                            and iter_residual > prev_residual + 1e-30):
                        lm_lambda = lm_lambda if lm_lambda > 0.0 else 1e-9
                        for _ in range(8):
                            try:
                                cand = _solve_mna(_lm_augment(mna_solve, lm_lambda), rhs)
                            except (np.linalg.LinAlgError, RuntimeError):
                                lm_lambda *= 10.0
                                continue
                            cand_residual = _mna_residual_inf(mna_solve, rhs, cand)
                            if cand_residual < iter_residual:
                                solution = cand
                                break
                            lm_lambda *= 10.0
                        else:
                            # No λ in the ladder helped — keep the plain
                            # Newton step and let damping handle it.
                            lm_lambda = 0.0
                    else:
                        # Step accepted on its own merit — relax λ.
                        lm_lambda = lm_lambda / 3.0 if lm_lambda > 1e-12 else 0.0

                    # V5' trust-region: cap NN per-iteration |ΔV| at one supply rail to kill NR runaway.
                    # (Phase 2d: vectorised; same per-element arithmetic.)
                    #
                    # `dv_limit` opens the SAME cap to non-NN circuits (and
                    # overrides the rail for NN ones) — see __init__. With
                    # dv_limit None the cap value and the arithmetic below are
                    # exactly what they were, so the NN path is bit-identical.
                    if self.dv_limit is not None:
                        dv_cap = self.dv_limit
                    elif _has_nn_device(self.circuit):
                        dv_cap = max_vs_voltage
                    else:
                        dv_cap = None
                    if dv_cap is not None:
                        d_head = solution[:n_nodes] - v_arr
                        solution[:n_nodes] = np.where(
                            d_head > dv_cap,
                            v_arr + dv_cap,
                            np.where(
                                d_head < -dv_cap,
                                v_arr - dv_cap,
                                solution[:n_nodes]))

                    # Calculate deltas (undamped, for adaptive damping)
                    sol_head = solution[:n_nodes]
                    max_delta = (
                        float(np.max(np.abs(sol_head - v_arr)))
                        if n_nodes else 0.0)

                    if _NR_TRACE and n_nodes and (
                            iteration % 50 == 0
                            or iteration >= self.max_iterations // num_steps - 6):
                        worst_i = int(np.argmax(np.abs(sol_head - v_arr)))
                        limited = [c.name for c in self.circuit.components
                                   if getattr(c, "_nr_limited", False)]
                        print(f"[DCNR g={gmin_level:.0e}] it={iteration} "
                              f"node={nodes[worst_i]} v {v_arr[worst_i]:+.4f}->"
                              f"{sol_head[worst_i]:+.4f} dmax={max_delta:.3e} "
                              f"resid={iter_residual:.3e} lim={limited[:6]}")

                    # Track voltage history for oscillation detection
                    voltage_snapshot = dict(zip(nodes, sol_head))
                    voltage_history.append(voltage_snapshot)
                    if len(voltage_history) > 5:
                        voltage_history.pop(0)

                    # --- Adaptive damping ---
                    improvement_ratio = max_delta / (prev_max_delta + 1e-15)

                    if improvement_ratio > 0.9 and iteration > 15:
                        stuck_counter += 1
                        if stuck_counter >= 2:
                            damping = max(0.1, damping * 0.8)
                            stuck_counter = 0
                    elif improvement_ratio < 0.5:
                        damping = min(1.0, damping * 1.1)
                        stuck_counter = 0
                    else:
                        stuck_counter = 0

                    # Supply-relative large-delta damping
                    if max_delta > 0.5 * max_vs_voltage:
                        damping = min(damping, 0.5)
                    elif max_delta < 0.05 * max_vs_voltage:
                        damping = min(1.0, damping * 1.2)

                    prev_max_delta = max_delta
                    # Record this iterate's residual for the next LM
                    # descent test (Phase 6a).
                    prev_residual = iter_residual

                    # Update voltages with damping (Phase 2d: vectorised;
                    # VS-constrained nodes take the solution directly,
                    # free nodes the damped blend — same expressions).
                    new_v = np.where(
                        vs_mask, sol_head,
                        damping * sol_head + (1.0 - damping) * v_arr)
                    deltas_arr = np.abs(new_v - v_arr)
                    max_change = (
                        float(np.max(deltas_arr)) if n_nodes else 0.0)
                    voltages.update(zip(nodes, new_v))
                    v_arr = new_v

                    # Log iteration if logger is available
                    if self.logger:
                        currents = {}
                        conductances = {}
                        # Pre-warm the batched NN cache for the post-update
                        # voltages so the per-device current/conductance
                        # reads below hit it (perf — the logger block
                        # re-evaluates every MOSFET at the damped iterate).
                        _batch_eval_nn_mosfets(self.circuit, voltages)
                        for comp in self.circuit.components:
                            try:
                                current = comp.calculate_current(voltages)
                                currents[comp.name] = current
                                if _is_mosfet(comp):
                                    g_ds, g_m, g_mb = comp.get_conductance(voltages)
                                    conductances[comp.name] = {"gm": g_m, "gds": g_ds, "gmb": g_mb}
                            except (NotImplementedError, AttributeError):
                                pass

                        iter_info = IterationInfo(
                            iteration=iteration,
                            voltages=voltages.copy(),
                            deltas=dict(zip(nodes, deltas_arr)),
                            currents=currents,
                            conductances=conductances
                        )
                        self.logger.log_iteration(point_num=0, iter_info=iter_info)

                    # Check convergence: SPICE-standard RELTOL + VNTOL
                    # (Phase 2d: vectorised — per node,
                    # threshold = vntol + reltol·max(|v_new|, |v_new−dv|),
                    # not converged if any dv ≥ threshold.)
                    if n_nodes:
                        conv_thr = self.vntol + self.reltol * np.maximum(
                            np.abs(new_v), np.abs(new_v - deltas_arr))
                        all_converged = bool(
                            not np.any(deltas_arr >= conv_thr))
                    else:
                        all_converged = True

                    # --- Residual-norm acceptance OR-gate (Phase 6b) ---
                    # The SPICE |ΔV| test alone can declare a stalled
                    # iterate "converged" — small step, but the KCL
                    # residual is still large (a non-physical fixed
                    # point with no DC equilibrium — ring oscillator,
                    # saddle latch). Require the MNA residual to also be
                    # small. iter_residual is ‖b−A·v‖∞ at the pre-step
                    # iterate; when |ΔV| is tiny the post-step voltages
                    # equal that iterate, so it is a valid residual at
                    # the accepted point. This ADDS a gate; it never
                    # relaxes the SPICE criterion.
                    #
                    # The threshold is deliberately generous. A genuine
                    # non-physical fixed point has a residual comparable
                    # to a full device current (µA–mA). A converged
                    # iterate's residual is ‖A·Δ‖ which, at the high-gain
                    # inverter trip, is the tiny step Δ amplified by the
                    # ~20× MNA gain — still far below a device current.
                    # Gating on max(_RESID_ABS_FLOOR, 100·reltol·‖b‖∞)
                    # cleanly separates the two and never misfires on a
                    # SPICE-converged inverter point.
                    if all_converged and _has_nn_device(self.circuit):
                        rhs_scale = float(np.max(np.abs(rhs))) if rhs.size else 0.0
                        resid_threshold = max(_RESID_ABS_FLOOR,
                                              100.0 * self.reltol * rhs_scale)
                        if iter_residual > resid_threshold:
                            all_converged = False

                    # --- NR-limiting acceptance gate (V7.5.0) ---
                    # An iteration whose matrix was stamped about LIMITED
                    # device biases is a linear extrapolation, not the
                    # circuit: never declare convergence on it. The
                    # limiter is the identity near a true solution, so
                    # this only delays acceptance while limiting is live.
                    if all_converged:
                        for component in self.circuit.components:
                            if getattr(component, "_nr_limited", False):
                                all_converged = False
                                break

                    if all_converged:
                        if self.logger:
                            self.logger.log_convergence(
                                point_num=0,
                                converged=True,
                                iterations=iteration + 1,
                                tolerance=max_change
                            )
                        nr_converged = True
                        break

                # --- Oscillation detection (NR exhausted without converging) ---
                if not nr_converged and len(voltage_history) >= 3:
                    max_rel_variance = 0.0
                    avg_voltages: Dict[str, float] = {}
                    for node in nodes:
                        values = [s.get(node, 0.0) for s in voltage_history[-3:]]
                        avg_voltages[node] = sum(values) / 3.0
                        variance = max(values) - min(values)
                        v_abs = max(abs(v) for v in values) if values else 0.0
                        threshold = self.vntol + self.reltol * v_abs
                        max_rel_variance = max(max_rel_variance, variance / (threshold + 1e-30))

                    if max_rel_variance < 10.0:
                        # Oscillating within tolerance — but only accept
                        # the averaged solution if it also satisfies the
                        # MNA residual test (Phase 6b). A small inter-
                        # iteration variance does not prove the average
                        # is a physical fixed point; an oscillator with
                        # no DC equilibrium can satisfy the variance test
                        # while ‖b−A·v‖ stays large. Re-stamp at the
                        # average and check the residual.
                        avg_residual = self._dc_residual_at(
                            avg_voltages, node_map, nodes, num_nodes,
                            matrix_size, gmin_level,
                        )
                        rhs_scale = avg_residual[1]
                        resid_threshold = max(_RESID_ABS_FLOOR, 100.0 * self.reltol * rhs_scale)
                        # V7.5.1: the KCL gate now also covers BSIM-CMG
                        # circuits — a quiet-but-garbage average that gets
                        # accepted poisons every later warm start (measured
                        # on the charge pump: one such commit at t~0 wedged
                        # the transient march at femtosecond scale forever).
                        residual_ok = (not (_has_nn_device(self.circuit)
                                            or _has_full_stamp_device(self.circuit))
                                       or avg_residual[0] <= resid_threshold)
                        # V7.5.0: never average-accept while NR limiting
                        # was clamping a device — the recent iterates are
                        # extrapolations, not near-solutions.
                        if residual_ok:
                            for component in self.circuit.components:
                                if getattr(component, "_nr_limited", False):
                                    residual_ok = False
                                    break
                        if residual_ok:
                            # Oscillating within tolerance — accept averaged solution
                            for node in nodes:
                                voltages[node] = avg_voltages[node]
                            voltages["0"] = 0.0
                            voltages["GND"] = 0.0
                            nr_converged = True

                # V7.5.0 — the verdict of a homotopy is its LAST level at
                # the true GMIN, full sources. The old sticky OR let an
                # intermediate gmin level mark the whole solve converged
                # even when the final level then diverged (observed on the
                # 125 C amplifier: flag True with nodes at -666 V).
                if step == num_steps - 1:
                    final_converged = nr_converged

            # GMIN continuation: use converged solution as initial guess
            # for the next level; a FAILED level restarts the next one
            # from the last good homotopy point instead of its own wreck.
            if final_converged:
                self.initial_guess = voltages.copy()
                _gmin_last_good = voltages.copy()
            elif _gmin_last_good is not None:
                voltages = _gmin_last_good.copy()
            if _NR_TRACE:
                vmax = max((abs(v) for k, v in voltages.items()
                            if k not in ("0", "GND")), default=0.0)
                print(f"[GMIN {gmin_level:.0e}] converged={final_converged} "
                      f"max|V|={vmax:.3f}")

        # Restore original voltage source values
        vs_idx = 0
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                component.voltage = original_voltages[vs_idx]
                vs_idx += 1

        # --- Force IC cleanup ---
        # The solver pins the .ic nodes with temp V-sources, converges
        # (correctly railed), then removes the pins and re-solves
        # UNCONSTRAINED, warm-started at the railed result, returning that.
        #
        # V6.4.6 Phase 1 — probe-hardening (Step 1) ONLY. The early `return`
        # here means the `_last_solve_converged` set at the bottom of
        # `_solve_newton` is NEVER reached on this path → it was stale. The
        # SRAM `force_ic` acceptance then trusted that stale flag plus a
        # rail-proximity check, with NO KCL-residual gate, so a pinned-node
        # artifact could false-PASS (the top Goodhart risk of V6.4.6). We now
        # compute the *released* (unconstrained) solution's KCL residual and
        # set `_last_solve_converged` honestly (finite AND residual within
        # the 100·reltol·‖b‖∞ band), plus expose `_last_dc_residual` /
        # `_last_dc_resid_threshold` so the test gates on a TRUE convergence
        # check, not a stale flag.
        #
        # The plan's Phase-1 Step-2 constraint-continuation homotopy (Norton
        # soft-pin g:1→0, tracking the railed branch P0-A proved exists) was
        # built and KILLED here as a dead end: on ALL 4 techs the railed
        # branch undergoes a fold/turning-point near g*≈1e-5 S and the
        # continuation slides into the symmetric metastable point
        # (q≈qb≈0.60/0.68/0.73), 0/8 — identical to the one-shot release.
        # Root cause: the railed point is a fixed point of the residual
        # ‖b−A·x‖ (≈8.5e-5, P0-A) but is UNSTABLE under the full re-stamp NR
        # map x→A(x)⁻¹b(x). A single re-stamp solve seeded EXACTLY at the
        # literal rail jumps qb 0→0.159 V in one step: the OFF storage node
        # has near-zero conductance to ground, so the step Δqb=residual/g_qb
        # explodes as the soft-pin conductance vanishes. The series-resistor
        # fallback is the Norton dual (R=1/g) and folds at the same R*=1/g*.
        # No single schedule rails any tech → the homotopy was reverted; only
        # the Step-1 probe-hardening ships. See
        # results/v6_4_6/phase1_force_ic_recovery.md.
        if _ic_temp_sources:
            for vs in _ic_temp_sources:
                self.circuit.components.remove(vs)
            self.circuit.invalidate_topology()
            # Re-solve without IC constraints using constrained result as guess
            saved_force_ic = self.force_ic
            self.force_ic = False
            self.initial_guess = voltages.copy()
            num_voltage_sources = self.circuit.count_voltage_sources()
            matrix_size = num_nodes + num_voltage_sources
            try:
                voltages = self._solve_newton()
            finally:
                self.force_ic = saved_force_ic
            # Released-solution KCL residual + honest convergence flag (Step 1).
            residual_inf, rhs_scale = self._dc_residual_at(
                voltages, node_map, nodes, num_nodes, matrix_size, self.gmin)
            resid_threshold = max(
                _RESID_ABS_FLOOR, 100.0 * self.reltol * rhs_scale)
            finite_voltages = all(
                (np.isfinite(v) and abs(v) < 1.0e10)
                for v in voltages.values())
            residual_ok = residual_inf <= resid_threshold
            self._last_dc_residual = float(residual_inf)
            self._last_dc_resid_threshold = float(resid_threshold)
            self._last_residual_ok = bool(residual_ok)
            self._last_solve_converged = bool(finite_voltages and residual_ok)
            return voltages

        # Extract and store voltage source currents from final operating point
        mna_matrix_final = _create_mna_matrix(matrix_size)
        rhs_final = np.zeros(matrix_size)

        for component in self.circuit.components:
            if not _is_mosfet(component):
                component.stamp_conductance(mna_matrix_final, node_map)
                component.stamp_rhs(rhs_final, node_map)

        # Batched DirectNet pre-warm for the final operating-point restamp.
        _batch_eval_nn_mosfets(self.circuit, voltages)

        for component in self.circuit.components:
            if _is_mosfet(component):
                self._stamp_mosfet(component, mna_matrix_final, rhs_final, node_map, voltages)

        self._stamp_voltage_sources(mna_matrix_final, rhs_final, node_map, num_nodes, voltages=voltages)

        try:
            solution_final = _solve_mna(mna_matrix_final, rhs_final)
            self._store_source_currents(solution_final, nodes)
        except (np.linalg.LinAlgError, RuntimeError):
            pass

        # V5 Phase A retry-design: surface convergence + finite-output
        # status so the simulation orchestrator can decide whether to
        # retry with GMIN homotopy on. Bad outputs (NaN / Inf / >1e10)
        # also count as failures because the DC solver does not raise
        # on NR exhaustion — it just returns the last (possibly garbage)
        # voltage vector.
        finite_voltages = all(
            (np.isfinite(v) and abs(v) < 1.0e10)
            for v in voltages.values()
        )
        self._last_solve_converged = bool(final_converged and finite_voltages)

        # V7.5.0 — automatic GMIN-homotopy fallback for BSIM-CMG circuits.
        # A cold plain-NR start can fall into an extrapolation-made hole
        # (measured on the LDO loop bench: a 5 mA load drags the output
        # below ground before the error amp biases up, and NR locks into
        # a fake equilibrium at vo=-3.9 V). The wide gmin ladder walks in
        # from a nearly-linear world and recovers the physical branch
        # (worst node 0.0 V vs NGSPICE, 2.7 s). NN circuits keep their own
        # `_solve_dc_with_retry` orchestration unchanged.
        if (not self._last_solve_converged
                and not self.use_gmin_stepping
                and not getattr(self, "_in_gmin_fallback", False)
                and any(hasattr(c, "get_terminal_stamp")
                        for c in self.circuit.components)):
            # The ladder replaces source stepping (two homotopies at once
            # just splits the NR budget 20 ways) and needs a real
            # iteration budget of its own. If the ladder ALSO fails, the
            # primary iterate is returned — the fallback must never
            # replace a near-solution with its own wreck. The convergence
            # flag stays honest either way.
            self._in_gmin_fallback = True
            prev_src = self.use_source_stepping
            prev_maxit = self.max_iterations
            self.use_gmin_stepping = True
            self.use_source_stepping = False
            self.max_iterations = max(200, prev_maxit)
            try:
                fallback_voltages = self.solve(skip_header=True)
                if self._last_solve_converged:
                    return fallback_voltages
                return voltages
            finally:
                self.use_gmin_stepping = False
                self.use_source_stepping = prev_src
                self.max_iterations = prev_maxit
                self._in_gmin_fallback = False

        return voltages

    def _stamp_mosfet(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        voltages: Dict[str, float],
        limit: bool = True,
    ) -> None:
        """Stamp MOSFET conductance and NR current source to MNA matrix (DC)."""
        _stamp_mosfet_dc(mosfet, mna_matrix, rhs, node_map, voltages, self.gmin,
                         limit=limit)

    def _dc_residual_at(
        self,
        voltages: Dict[str, float],
        node_map: Dict[str, int],
        nodes: List[str],
        num_nodes: int,
        matrix_size: int,
        gmin_level: float,
    ) -> tuple:
        """Re-stamp the DC MNA system at ``voltages`` and return its residual.

        Phase 6b. Used to validate the oscillation-detection averaged
        solution: a small inter-iteration variance does not prove the
        averaged voltages are a physical fixed point. This re-stamps the
        full MNA system (linear + MOSFET + voltage sources) at the
        candidate voltages and reports the infinity-norm KCL residual.

        Args:
            voltages: candidate node-voltage dict.
            node_map: node-name → matrix-index map.
            nodes: ordered non-ground node names.
            num_nodes: number of non-ground nodes.
            matrix_size: full MNA dimension (nodes + voltage sources).
            gmin_level: GMIN level to stamp (matches the active homotopy step).

        Returns:
            ``(residual_inf, rhs_scale)`` — the residual ‖b−A·v‖∞ and the
            RHS infinity-norm used to scale the acceptance threshold.
        """
        mna = _create_mna_matrix(matrix_size)
        rhs = np.zeros(matrix_size)
        for component in self.circuit.components:
            if not _is_mosfet(component):
                component.stamp_conductance(mna, node_map)
                component.stamp_rhs(rhs, node_map)
        for component in self.circuit.components:
            if _is_mosfet(component):
                # limit=False: a residual probe wants the TRUE KCL residual
                # at the candidate voltages, and must not advance the
                # NR-limiting anchor of the surrounding solve.
                self._stamp_mosfet(component, mna, rhs, node_map, voltages,
                                   limit=False)
        self._stamp_voltage_sources(mna, rhs, node_map, num_nodes, voltages=voltages)
        if gmin_level > self.gmin:
            self._apply_gmin_stepping(mna, node_map, gmin_level)
        iterate = np.zeros(matrix_size)
        for idx, node in enumerate(nodes):
            iterate[idx] = voltages[node]
        rhs_scale = float(np.max(np.abs(rhs))) if rhs.size else 0.0
        return _mna_residual_inf(mna, rhs, iterate), rhs_scale

    def _stamp_voltage_sources(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int,
        voltages: Dict[str, float] = None,
    ) -> None:
        """
        Stamp voltage source equations to MNA matrix.

        For each voltage source, we add:
        - B matrix column: connection to node voltages
        - C matrix row: voltage constraint equation
        - RHS entry: voltage source value (for linear) or mismatch (for Newton-Raphson)

        The voltage source equation is: V_pos - V_neg = V_source
        For Newton-Raphson: delta_V_pos - delta_V_neg = V_source - (V_pos_old - V_neg_old)

        Args:
            mna_matrix: MNA matrix to modify (in-place)
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
            voltages: Current voltage estimate (for Newton-Raphson), None for linear solve
        """
        voltage_source_index = 0

        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Get voltage source nodes
                pos_node = component.nodes[0]  # Positive terminal
                neg_node = component.nodes[1]  # Negative terminal
                voltage = component.voltage

                # The row index for this voltage source's equation
                vs_row = num_nodes + voltage_source_index

                # Stamp B matrix (voltage source current flows into nodes)
                if pos_node != "0" and pos_node in node_map:
                    pos_idx = node_map[pos_node]
                    mna_matrix[vs_row, pos_idx] += 1.0
                    mna_matrix[pos_idx, vs_row] += 1.0

                if neg_node != "0" and neg_node in node_map:
                    neg_idx = node_map[neg_node]
                    mna_matrix[vs_row, neg_idx] -= 1.0
                    mna_matrix[neg_idx, vs_row] -= 1.0

                # Stamp voltage source value to RHS
                # Use direct voltage value for companion model consistency.
                # The companion model for MOSFETs solves for V directly,
                # so voltage sources should also use direct form.
                rhs[vs_row] = voltage

                # Move to next voltage source
                voltage_source_index += 1

    def _extract_voltages(self, solution: np.ndarray, nodes: List[str]) -> Dict[str, float]:
        """
        Extract node voltages from solution vector.

        The solution vector contains:
        - First num_nodes entries: node voltages
        - Remaining entries: voltage source currents

        Args:
            solution: Solution vector from np.linalg.solve
            nodes: List of non-ground node names

        Returns:
            Dictionary mapping node names to voltages (including ground)
        """
        voltages = {}

        # Extract node voltages (first num_nodes entries)
        for idx, node in enumerate(nodes):
            voltages[node] = float(solution[idx])

        # Add ground node (reference voltage)
        voltages["0"] = 0.0
        voltages["GND"] = 0.0

        return voltages

    def _store_source_currents(self, solution: np.ndarray, nodes: List[str]) -> None:
        """
        Extract and store voltage source currents from solution vector.

        The solution vector contains voltage source currents after the node voltages.
        This method extracts those currents and stores them in the VoltageSource objects
        so they can be retrieved via calculate_current().

        Args:
            solution: Solution vector from np.linalg.solve
            nodes: List of non-ground node names
        """
        num_nodes = len(nodes)
        vs_idx = 0

        # Iterate through circuit components to find voltage sources in order
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Extract current from solution vector (after node voltages)
                current_idx = num_nodes + vs_idx
                if current_idx < len(solution):
                    current = float(solution[current_idx])
                    # Store current in the voltage source object
                    if hasattr(component, 'set_current'):
                        component.set_current(current)
                vs_idx += 1

    def get_last_solution(self) -> Optional[Dict[str, float]]:
        """
        Get the last computed solution from this solver.

        Returns:
            Dictionary mapping node names to voltages, or None if solve() hasn't been called yet
        """
        return self.last_solution

    def __repr__(self) -> str:
        """String representation of the solver."""
        return (
            f"DCSolver(circuit={self.circuit}, "
            f"tolerance={self.tolerance}, "
            f"max_iterations={self.max_iterations})"
        )


class TransientSolver:
    """
    Transient Solver for time-domain analysis using Backward Euler integration.

    The TransientSolver performs time-domain simulation of circuits with capacitors.
    It uses the Backward Euler method to discretize capacitors into companion models
    (equivalent conductance and current source) at each timestep.

    Algorithm:
    1. Perform DC analysis at t=0 to find initial conditions
    2. For each timestep:
       a. Update capacitor companion models (G_eq = C/dt, I_eq = G_eq * V_prev)
       b. Solve DC circuit at current timestep
       c. Update capacitor voltages for next timestep
       d. Store results

    Attributes:
        circuit: Circuit object containing components and topology
        t_stop: Stop time for simulation in seconds
        dt: Timestep size in seconds
    """

    def __init__(self, circuit: Circuit, t_stop: float, dt: float,
                 initial_guess: Optional[Dict[str, float]] = None,
                 debug: bool = False,
                 use_gmin_stepping: bool = True,
                 gmin_initial: float = 1e-8,
                 gmin_final: float = 1e-12,
                 gmin_steps: int = 5,
                 use_pseudo_transient: bool = True,
                 pseudo_transient_steps: int = 3,
                 pseudo_transient_cap: float = 1e-12,
                 nr_tolerance: float = 1e-7,
                 reltol: float = 1e-4, vntol: float = 1e-7, gmin: float = 1e-12,
                 max_substeps: int = 1, lte_safety_factor: float = 0.5,
                 integration_method: str = 'auto',
                 dv_limit: Optional[float] = None,
                 refine_output: bool = False):
        """
        Initialize the Transient Solver.

        Args:
            circuit: Circuit object to simulate
            t_stop: Stop time for simulation in seconds
            dt: Timestep size in seconds (must be positive)
            initial_guess: Optional initial voltage guess from DC operating point
            debug: Enable debug logging for convergence diagnostics
            use_gmin_stepping: Enable Gmin stepping for difficult convergence (default: True)
            gmin_initial: Initial Gmin value for stepping (default: 1e-8 S)
            gmin_final: Final Gmin value (default: 1e-12 S)
            gmin_steps: Number of Gmin stepping steps (default: 5)
            use_pseudo_transient: Enable pseudo-transient initialization (default: True)
            pseudo_transient_steps: Number of initial timesteps with pseudo-capacitance (default: 3)
            pseudo_transient_cap: Artificial capacitance value in Farads (default: 1e-12 F)
            nr_tolerance: Newton-Raphson convergence tolerance (default: 1e-7 V)
            reltol: Relative convergence tolerance (default: 1e-4, tighter than SPICE 1e-3)
            vntol: Absolute voltage tolerance (default: 1e-7 V, tighter than SPICE 1e-6)
            gmin: Minimum MOSFET channel conductance (SPICE GMIN, default: 1e-12 S)
            max_substeps: Max LTE-adaptive sub-steps per output interval (1=disabled, default: 1)
            lte_safety_factor: LTE acceptance threshold (default: 0.5)
            integration_method: 'auto' (default) reproduces the historical
                ladder BE(step 1) -> Trap(step 2+) -> BDF-2 on stiffness;
                'gear2' keeps BE for step 1 (as ngspice does) and then pins
                Gear-2/BDF-2 for every later step with the stiffness trip
                disabled — for decks carrying `.options method=gear maxord=2`,
                where trapezoid ringing corrupts slew/overshoot metrics.
            dv_limit: Per-iteration, per-node |ΔV| trust-region cap in volts,
                applied at EVERY timestep (default None = OFF, historical
                behaviour). The transient twin of ``DCSolver(dv_limit=...)``;
                see there for why it exists and why it is default-off.
            refine_output: V7.5.2 — LTE-driven local refinement ON THE
                OUTPUT AXIS (default False = OFF, byte-identical; env
                ``PYCIRCUITSIM_TRAN_REFINE=1`` also enables). The fixed
                output grid records only interval endpoints, so fast
                events between grid points (the AnalogGym charge pump's
                ±4 µA / ~10 ps switching-current spike) are invisible to
                any downstream ``.meas`` no matter how accurately the
                march resolved them — NGSPICE saves every LTE-accepted
                internal point, so its .meas sees the spike. When on:
                every committed march piece is emitted into the returned
                waveform (time axis becomes non-uniform; the fixed grid
                points all remain present exactly); PULSE source corners
                become breakpoints (piece boundaries land on them, the
                following piece restarts small and integrates BE, as
                NGSPICE restarts after a breakpoint — this is what stops
                trapezoidal corner ringing); and each committed piece is
                LTE-checked (trapezoidal 3rd-divided-difference estimate,
                NGSPICE TRTOL=7) with depth-1 rollback: a piece whose
                LTE exceeds tolerance is rejected, device charge/voltage
                histories restored, and re-marched with a smaller dt.

        Raises:
            ValueError: If dt or t_stop is not positive, or integration_method
                is not one of {'auto', 'gear2'}
            NotImplementedError: If the circuit contains an Inductor (DC/AC
                only — a silent DC short in a transient run would be a wrong
                answer, so it must be loud)
        """
        if dt <= 0:
            raise ValueError(f"Timestep dt must be positive, got {dt}")
        if t_stop <= 0:
            raise ValueError(f"Stop time t_stop must be positive, got {t_stop}")
        if integration_method not in ('auto', 'gear2'):
            raise ValueError(
                f"Unknown integration_method '{integration_method}'. "
                "Supported: 'auto' (BE->Trap->BDF-2 on stiffness), 'gear2'")

        from pycircuitsim.models.passive import Inductor
        for _component in circuit.components:
            if isinstance(_component, Inductor):
                raise NotImplementedError(
                    f"Inductor is DC/AC only; {_component.name} has no "
                    "transient companion model")

        self.circuit = circuit
        self.t_stop = t_stop
        self.dt = dt
        self.initial_guess = initial_guess
        self.debug = debug

        # V7.0.1 — transient stamps the MOSFET capacitances every NR
        # iteration, so the NN devices must compute their qg / qd autograd
        # Jacobians. Declared once here; a `.dc` / `.op` run never declares
        # it and skips those two backward passes entirely.
        _require_nn_caps(circuit)

        # Gmin stepping parameters
        self.use_gmin_stepping = use_gmin_stepping
        self.gmin_initial = gmin_initial
        self.gmin_final = gmin_final
        self.gmin_steps = gmin_steps

        # Pseudo-transient initialization parameters
        self.use_pseudo_transient = use_pseudo_transient
        self.pseudo_transient_steps = pseudo_transient_steps
        self.pseudo_transient_cap = pseudo_transient_cap

        # Newton-Raphson convergence tolerance
        self.nr_tolerance = nr_tolerance

        # SPICE-standard convergence parameters
        self.reltol = reltol
        self.vntol = vntol
        self.gmin = gmin

        # LTE adaptive sub-stepping parameters
        self.max_substeps = max_substeps
        self.lte_safety_factor = lte_safety_factor

        # Active internal timestep (may differ from self.dt during sub-stepping)
        self._current_dt = dt

        # Requested integrator policy ('auto' | 'gear2'); the per-step method
        # actually in force is self._integration_method below.
        self.integration_method = integration_method

        # Per-iteration |ΔV| trust-region cap (None = off); see __init__ docs.
        self.dv_limit = dv_limit

        # Integration method: 'be', 'trap', or 'bdf2'
        self._integration_method = 'be'

        # C6a — per-committed-step branch currents of every VoltageSource
        # (name -> array aligned with the returned "time" vector). Populated
        # by solve(); deliberately NOT added to solve()'s returned dict,
        # because run_transient builds its CSV columns from that dict.
        self.source_currents: Dict[str, np.ndarray] = {}

        # Number of committed steps whose branch currents could not be read
        # from a solution vector (the oscillation-averaged acceptance path has
        # none): those entries are NaN, never a plausible-looking zero.
        self._branch_current_gaps = 0

        # Voltage-source tail (solution[num_nodes:]) of the last accepted
        # sub-step solve, or None when the step was accepted without one.
        self._last_solution_tail: Optional[np.ndarray] = None

        # Store pseudo-capacitor references for cleanup
        self._pseudo_capacitors: List = []

        # V5 Phase A — A3: dt-halve fallback event log. Each entry is a
        # dict with {step, sub_idx, sim_time, halve_num, dt_before, dt_after,
        # error_msg}. Read by verification scripts after a transient run
        # to flag cells that needed >1 halving.
        self._dt_halve_events: List[Dict] = []

        # V7.5.2 — opt-in LTE-driven output refinement (see __init__ docs).
        self.refine_output = (refine_output or
                              os.environ.get("PYCIRCUITSIM_TRAN_REFINE",
                                             "0") == "1")
        # NGSPICE's transient truncation-error tolerance multiplier.
        self._refine_trtol = 7.0

    def _collect_breakpoints(self) -> List[float]:
        """PULSE source corner times in (0, t_stop), sorted, deduplicated.

        Corners per period k: td + k*per + {0, tr, tr+pw, tr+pw+tf} — the
        four slope discontinuities of the PULSE waveform. Only used by the
        refine_output mode; a deck without PULSE sources yields [].
        """
        from pycircuitsim.models.passive import (PulseVoltageSource,
                                                 PulseCurrentSource)
        bps: set = set()
        for c in self.circuit.components:
            if not isinstance(c, (PulseVoltageSource, PulseCurrentSource)):
                continue
            corners = (0.0, c.tr, c.tr + c.pw, c.tr + c.pw + c.tf)
            k = 0
            while c.td + k * c.per < self.t_stop:
                for corner in corners:
                    t = c.td + k * c.per + corner
                    if 0.0 < t < self.t_stop:
                        bps.add(t)
                k += 1
        return sorted(bps)

    def _snapshot_tran_state(self) -> List[tuple]:
        """Depth-1 snapshot of every device's committed transient history.

        Captures exactly the state the commit path mutates (Capacitor
        ``update_voltage``: v_prev/v_prev2/_i_prev; MOSFET
        ``update_charge_state``: _q_prev/_q_prev2/_v_prev_tran/_i_prev_*),
        so a piece rejected on LTE can be un-committed. Refine mode only.
        """
        snap: List[tuple] = []
        for c in self.circuit.components:
            if isinstance(c, Capacitor):
                snap.append((c, "cap", c.v_prev, c.v_prev2, c._i_prev))
            elif _is_mosfet(c) and hasattr(c, "update_charge_state"):
                qp = getattr(c, "_q_prev", None)
                qp2 = getattr(c, "_q_prev2", None)
                vpt = getattr(c, "_v_prev_tran", None)
                snap.append((
                    c, "mos",
                    qp.copy() if qp is not None else None,
                    qp2.copy() if qp2 is not None else None,
                    vpt.copy() if vpt is not None else None,
                    getattr(c, "_i_prev_gate", 0.0),
                    getattr(c, "_i_prev_drain", 0.0),
                    getattr(c, "_i_prev_source", 0.0),
                    getattr(c, "_i_prev_bulk", 0.0)))
        return snap

    def _restore_tran_state(self, snap: List[tuple]) -> None:
        """Undo one committed piece (see _snapshot_tran_state)."""
        for entry in snap:
            c, kind = entry[0], entry[1]
            if kind == "cap":
                c.v_prev, c.v_prev2, c._i_prev = entry[2], entry[3], entry[4]
            else:
                (c._q_prev, c._q_prev2, c._v_prev_tran, c._i_prev_gate,
                 c._i_prev_drain, c._i_prev_source, c._i_prev_bulk) = entry[2:]

    def _refine_lte_ratio(self, hist: List[tuple], t_new: float,
                          v_new: np.ndarray) -> float:
        """Worst-node LTE / (TRTOL·tol) for the candidate point.

        Trapezoidal per-step LTE = (h³/12)·|v'''|, with v''' estimated as
        6× the third divided difference over the last three ACCEPTED fine
        points plus the candidate (non-uniform spacing handled exactly).
        Ratio > 1 means the piece exceeds NGSPICE-equivalent truncation
        tolerance (TRTOL=7) and should be re-marched with a smaller dt.
        """
        (t0, v0), (t1, v1), (t2, v2) = hist[-3], hist[-2], hist[-1]
        h = t_new - t2
        if h <= 0.0:
            return 0.0
        dd01 = (v1 - v0) / (t1 - t0)
        dd12 = (v2 - v1) / (t2 - t1)
        dd23 = (v_new - v2) / h
        dd012 = (dd12 - dd01) / (t2 - t0)
        dd123 = (dd23 - dd12) / (t_new - t1)
        dd0123 = np.abs(dd123 - dd012) / (t_new - t0)
        lte = 0.5 * h ** 3 * dd0123          # (h³/12)·|6·DD3|
        tol = self._refine_trtol * (
            self.reltol * np.maximum(np.abs(v_new), np.abs(v2)) + self.vntol)
        return float(np.max(lte / tol))

    def _has_non_linear_components(self) -> bool:
        """Check if circuit contains non-linear components (MOSFETs)."""
        return _has_non_linear(self.circuit)

    def _has_nn_devices(self) -> bool:
        """Return True if the circuit contains any NN compact-model device
        (LEVEL >= 73). Used to gate the V5 Phase A dt-halve fallback so
        BSIM-CMG (LEVEL=72) transients keep their existing behaviour.
        """
        return _has_nn_device(self.circuit)

    def _add_pseudo_capacitors(self) -> None:
        """Add pseudo-capacitors scaled to circuit capacitance for initialization."""
        from pycircuitsim.models.passive import Capacitor

        # Auto-detect max circuit capacitance
        max_circuit_cap = 0.0
        for component in self.circuit.components:
            if isinstance(component, Capacitor) and not component.name.startswith("_pseudo_"):
                max_circuit_cap = max(max_circuit_cap, component.capacitance)

        # Scale pseudo-cap: 5x the largest circuit cap, or use user-specified value
        if max_circuit_cap > 0 and self.pseudo_transient_cap > 10 * max_circuit_cap:
            effective_cap = 5.0 * max_circuit_cap
            if self.debug:
                print(f"  Auto-scaling pseudo-cap: {self.pseudo_transient_cap:.2e} -> "
                      f"{effective_cap:.2e} (5x max circuit cap {max_circuit_cap:.2e})")
        else:
            effective_cap = self.pseudo_transient_cap

        nodes = self.circuit.get_nodes()
        pseudo_cap_idx = 0
        for node in nodes:
            cap = Capacitor(f"_pseudo_{pseudo_cap_idx}", [node, "0"], effective_cap)
            self.circuit.components.append(cap)
            self._pseudo_capacitors.append(cap)
            pseudo_cap_idx += 1
        if self._pseudo_capacitors:
            self.circuit.invalidate_topology()

    def _remove_pseudo_capacitors(self) -> None:
        """
        Remove pseudo-capacitors added for pseudo-transient initialization.
        """
        for cap in self._pseudo_capacitors:
            if cap in self.circuit.components:
                self.circuit.components.remove(cap)
        if self._pseudo_capacitors:
            self.circuit.invalidate_topology()
        self._pseudo_capacitors.clear()

    def _apply_gmin_stepping(self, mna_matrix: np.ndarray, node_map: Dict[str, int], gmin: float) -> None:
        """
        Apply Gmin stepping by adding minimum conductance to all nodes.

        Args:
            mna_matrix: MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
            gmin: Current Gmin value to apply
        """
        # Add gmin from each non-ground node to ground
        for node, idx in node_map.items():
            mna_matrix[idx, idx] += gmin

    def _solve_timestep_newton(
        self,
        nodes: List[str],
        node_map: Dict[str, int],
        num_nodes: int,
        num_voltage_sources: int,
        initial_voltages: Dict[str, float],
        time: float,
        step_index: int = 0,
        use_gmin: bool = True
    ) -> Dict[str, float]:
        """
        Solve circuit at a single timestep using Newton-Raphson iteration.

        This method is used for non-linear circuits (with MOSFETs).
        It iteratively linearizes the circuit equations until convergence.

        Args:
            nodes: List of non-ground node names
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
            num_voltage_sources: Number of voltage sources
            initial_voltages: Initial voltage guess from previous timestep
            time: Current simulation time
            step_index: Current timestep index (for Gmin stepping)

        Returns:
            Dictionary mapping node names to voltage values

        Raises:
            RuntimeError: If Newton-Raphson fails to converge
        """
        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # C6a — drop any tail from an earlier sub-step so a failed/averaged
        # solve can never commit stale branch currents.
        self._last_solution_tail = None

        # V7.5.0: fresh NR-limiting anchors for THIS solve. An anchor left
        # at the limiting window by a failed attempt would distort the
        # first evaluations of the retry near the (good) starting point,
        # and the charge companion amplifies any eval distortion by 1/dt —
        # measured as ~1e2 A phantom residuals at the operating point.
        for component in self.circuit.components:
            reset = getattr(component, "reset_nr_limits", None)
            if reset is not None:
                reset()

        # Use previous timestep's voltages as initial guess
        voltages = initial_voltages.copy()

        # Newton-Raphson parameters (aligned with DC solver)
        tolerance = self.nr_tolerance
        max_iterations = 200  # Increased from 100 for difficult convergence

        # Calculate Gmin value for this timestep (if enabled)
        gmin = self.gmin_final
        if use_gmin and self.use_gmin_stepping and step_index < self.gmin_steps:
            # Exponential decay from gmin_initial to gmin_final
            alpha = step_index / (self.gmin_steps - 1) if self.gmin_steps > 1 else 1.0
            gmin = self.gmin_initial * (1 - alpha) + self.gmin_final * alpha
            if self.debug:
                print(f"  Gmin stepping: step {step_index}, gmin = {gmin:.2e}")

        # Start with full damping (no damping); reduce if needed during iteration
        damping = 1.0

        # Track previous max_delta for adaptive damping
        prev_max_delta = float('inf')
        stuck_counter = 0  # Count iterations with minimal improvement

        # --- Levenberg-Marquardt damping state (Phase 6a) ---
        lm_lambda = 0.0
        prev_residual = float('inf')

        # Track recent voltages for oscillation detection
        voltage_history = []

        # Debug: Track convergence behavior (if enabled)
        debug_log = [] if self.debug else None

        # V7.2.0 Phase 2d — per-iteration node state as a vector (see
        # the DC loop); the VS-constrained set is topology-static, so it
        # is built once per timestep, not once per NR iteration.
        n_nodes = len(nodes)
        v_arr = np.fromiter(
            (voltages[nd] for nd in nodes), np.float64, n_nodes)
        vs_constrained_nodes = set()
        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                pos_node = component.nodes[0]
                neg_node = component.nodes[1]
                # If one terminal is ground, the other is constrained
                if neg_node == "0":
                    vs_constrained_nodes.add(pos_node)
                elif pos_node == "0":
                    vs_constrained_nodes.add(neg_node)
        vs_mask = np.fromiter(
            (nd in vs_constrained_nodes for nd in nodes), bool, n_nodes)

        # V7.5.1 — fail-fast bookkeeping. A step that is going to fail
        # used to burn all `max_iterations` before the adaptive march cut
        # dt; on switching edges (charge pump) that is hundreds of
        # 200-iteration NR runs per edge. Track the best |ΔV| seen in a
        # trailing window; a stagnant far-from-converged iterate bails to
        # the dt-cut early. Failure-path only: any iterate still making
        # 2x progress per window — and any within 100x of tolerance — is
        # untouched, so converging trajectories are bit-identical.
        _ff_window = 30
        _ff_best = float("inf")
        _ff_best_at = 0

        for iteration in range(max_iterations):
            # Build MNA matrix and RHS
            mna_matrix = _create_mna_matrix(matrix_size)
            rhs = np.zeros(matrix_size)

            # Stamp linear components (resistors, capacitors)
            for component in self.circuit.components:
                if not _is_mosfet(component):
                    component.stamp_conductance(mna_matrix, node_map)
                    self._stamp_component_rhs(component, rhs, node_map, time)

            # Stamp voltage sources (with time-varying support)
            self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, time, voltages)

            # Batched DirectNet (LEVEL=73) forward+Jacobian: one stacked
            # NN call per checkpoint pre-warms the eval cache so the
            # per-device transient stamps below hit it. Perf only — see
            # _batch_eval_nn_mosfets for the accuracy note.
            _batch_eval_nn_mosfets(self.circuit, voltages)

            # Stamp MOSFETs at current voltage estimate.
            # Phase 3b (opt-in): NN devices go through one COO assembly
            # (DC + transcap companion); LEVEL=72/others stay scalar.
            nn_extra = None
            if _BATCHED_STAMP:
                nn_extra = _stamp_nn_mosfets_batched(
                    self.circuit, node_map, matrix_size, rhs,
                    voltages, v_arr, self.gmin, tran_solver=self)
                for component in self.circuit.components:
                    if (_is_mosfet(component)
                            and not _is_nn_mosfet(component)):
                        self._stamp_mosfet_transient(
                            component, mna_matrix, rhs, node_map, voltages)
            else:
                for component in self.circuit.components:
                    if _is_mosfet(component):
                        self._stamp_mosfet_transient(component, mna_matrix, rhs, node_map, voltages)

            # Apply Gmin stepping (if enabled)
            if gmin > self.gmin_final:
                self._apply_gmin_stepping(mna_matrix, node_map, gmin)

            # V7.2.0 Phase 4a — one LIL→CSR conversion per iteration,
            # shared by the residual, the solve, and the LM ladder.
            mna_solve = (
                mna_matrix.tocsr() if issparse(mna_matrix) else mna_matrix)
            if nn_extra is not None:
                mna_solve = mna_solve + nn_extra

            # Current iterate as a full MNA vector (Phase 6a/6b).
            current_iterate = np.zeros(matrix_size)
            current_iterate[:n_nodes] = v_arr

            # MNA residual ‖b−A·v‖∞ at the current iterate (Phase 6b).
            iter_residual = _mna_residual_inf(mna_solve, rhs, current_iterate)

            # Solve for voltage updates
            try:
                solution = _solve_mna(mna_solve, rhs)
            except (np.linalg.LinAlgError, RuntimeError):
                raise RuntimeError(
                    f"Circuit matrix is singular at t={time:.6e}s during Newton-Raphson iteration {iteration+1}"
                )

            # --- Levenberg-Marquardt damping (Phase 6a) ---
            # When the residual fails to decrease, the Newton step
            # overshot — add λ·I to the Jacobian and re-solve, scaling λ
            # ×10 (Nielsen rule) until the candidate residual improves,
            # ÷3 on acceptance. Sits alongside the trust-region rail cap
            # below; the fixed point is unchanged.
            if (_has_nn_device(self.circuit)
                    and np.isfinite(prev_residual)
                    and iter_residual > prev_residual + 1e-30):
                lm_lambda = lm_lambda if lm_lambda > 0.0 else 1e-9
                for _ in range(8):
                    try:
                        cand = _solve_mna(_lm_augment(mna_solve, lm_lambda), rhs)
                    except (np.linalg.LinAlgError, RuntimeError):
                        lm_lambda *= 10.0
                        continue
                    cand_residual = _mna_residual_inf(mna_solve, rhs, cand)
                    if cand_residual < iter_residual:
                        solution = cand
                        break
                    lm_lambda *= 10.0
                else:
                    lm_lambda = 0.0
            else:
                lm_lambda = lm_lambda / 3.0 if lm_lambda > 1e-12 else 0.0

            # V5' trust-region: cap NN per-iteration |ΔV| at one supply rail to kill NR runaway.
            # (Phase 2d: vectorised; same per-element arithmetic.)
            #
            # `dv_limit` opens the same cap to non-NN circuits — the DC
            # counterpart, same rationale (see DCSolver.__init__), needed at
            # every timestep and not only at the operating point because a
            # transient NR step blows up the same way. With dv_limit None the
            # cap and the arithmetic are unchanged.
            if self.dv_limit is not None:
                vdd_cap = self.dv_limit
            elif _has_nn_device(self.circuit):
                vdd_cap = max(
                    (abs(c.voltage) for c in self.circuit.components if isinstance(c, VoltageSource)),
                    default=1.0,
                ) or 1.0
            else:
                vdd_cap = None
            if vdd_cap is not None:
                d_head = solution[:n_nodes] - v_arr
                solution[:n_nodes] = np.where(
                    d_head > vdd_cap, v_arr + vdd_cap,
                    np.where(d_head < -vdd_cap, v_arr - vdd_cap,
                             solution[:n_nodes]))

            # Extract voltages from solution (matches DC solver approach)
            # Solution contains NEW voltages, not deltas (due to MNA
            # formulation). Phase 2d: vectorised; the old per-node
            # ``deltas`` dict was write-only in this loop and is gone.
            sol_head = solution[:n_nodes]
            dv_arr = np.abs(sol_head - v_arr)
            max_delta = float(np.max(dv_arr)) if n_nodes else 0.0

            if _NR_TRACE and n_nodes:
                worst_i = int(np.argmax(dv_arr))
                limited = [c.name for c in self.circuit.components
                           if getattr(c, "_nr_limited", False)]
                print(f"[NR t={time:.3e}] it={iteration} node={nodes[worst_i]} "
                      f"v {v_arr[worst_i]:+.4f}->{sol_head[worst_i]:+.4f} "
                      f"dmax={max_delta:.3e} resid={iter_residual:.3e} "
                      f"lim={limited[:6]}")

            # V7.5.1 fail-fast (see the bookkeeping above the loop).
            if max_delta < _ff_best * 0.5:
                _ff_best = max_delta
                _ff_best_at = iteration
            elif (iteration - _ff_best_at >= _ff_window
                  and iteration >= 2 * _ff_window
                  and max_delta > max(100.0 * tolerance, 1e-2)):
                # The 1e-2 V floor keeps this strictly above the
                # oscillation-average acceptance ceiling (10x the
                # vntol + reltol·|v| threshold, sub-mV on these rails),
                # so a fast-switcher that would be accepted by averaging
                # is never stolen from that path.
                raise RuntimeError(
                    f"Newton-Raphson stagnant at t={time:.6e}s: no 2x "
                    f"progress in {iteration - _ff_best_at} iterations "
                    f"(max delta {max_delta:.2e}); failing fast to the "
                    f"timestep-cut ladder")

            # Track voltage history for oscillation detection (store last 5 iterations)
            voltage_snapshot = dict(zip(nodes, sol_head))
            voltage_history.append(voltage_snapshot)
            if len(voltage_history) > 5:
                voltage_history.pop(0)

            # DEBUG: Log first few and last few iterations
            if self.debug and (iteration < 5 or iteration >= max_iterations - 5):
                debug_log.append(f"  Iter {iteration}: max_delta={max_delta:.6e}, damping={damping:.2f}, gmin={gmin:.2e}")

            # Check convergence: SPICE-standard RELTOL + VNTOL
            # (Phase 2d: vectorised — dv = |sol−old| against
            # vntol + reltol·max(|sol|, |old|), any violation fails.)
            if n_nodes:
                conv_thr = self.vntol + self.reltol * np.maximum(
                    np.abs(sol_head), np.abs(v_arr))
                all_converged = bool(not np.any(dv_arr >= conv_thr))
            else:
                all_converged = True

            # --- Residual-norm acceptance OR-gate (Phase 6b) ---
            # Reject a stalled iterate: small |ΔV| but large KCL
            # residual is a non-physical fixed point. Adds a gate on top
            # of the SPICE |ΔV| test, never relaxes it. The threshold is
            # generous (see the DC analog) so it never misfires on a
            # SPICE-converged timestep.
            if all_converged and _has_nn_device(self.circuit):
                rhs_scale = float(np.max(np.abs(rhs))) if rhs.size else 0.0
                resid_threshold = max(_RESID_ABS_FLOOR, 100.0 * self.reltol * rhs_scale)
                if iter_residual > resid_threshold:
                    all_converged = False

            # --- NR-limiting acceptance gate (V7.5.0) ---
            # Same rule as the DC loop: a timestep stamped about limited
            # device biases is an extrapolation — do not accept it.
            if all_converged:
                for component in self.circuit.components:
                    if getattr(component, "_nr_limited", False):
                        all_converged = False
                        break

            if all_converged:
                # Converged! Use new voltages directly
                voltages.update(zip(nodes, sol_head))
                # C6a — keep the voltage-source branch-current tail of the
                # accepted iterate; solve() reads it when it commits the step.
                self._last_solution_tail = np.array(solution[num_nodes:])
                self._last_nr_iterations = iteration + 1
                if self.debug and debug_log is not None and len(debug_log) > 0:
                    print(f"\nDEBUG: Converged at t={time:.6e}s after {iteration+1} iterations")
                break

            # Adaptive damping: adjust based on convergence behavior
            improvement_ratio = max_delta / (prev_max_delta + 1e-12)

            # Reduce damping aggressively if not converging well
            if improvement_ratio > 0.9 and iteration > 3:
                # Stuck or oscillating: reduce damping more
                stuck_counter += 1
                if stuck_counter >= 2:
                    damping = max(0.25, damping * 0.8)  # Reduce damping
                    stuck_counter = 0
            elif improvement_ratio < 0.5:
                # Good progress: increase damping
                damping = min(1.0, damping * 1.1)
                stuck_counter = 0
            else:
                stuck_counter = 0

            # Apply damping based on voltage deltas (match DC solver threshold)
            if max_delta >= 1.0:
                damping = min(damping, 0.5)  # Force damping if deltas are very large
            elif max_delta < 0.1:
                damping = 1.0  # No damping needed for small deltas

            prev_max_delta = max_delta
            # Record residual for the next LM descent test (Phase 6a).
            prev_residual = iter_residual

            # Update voltages with damping (match DC solver approach;
            # Phase 2d: vectorised — VS-constrained nodes take the
            # solution directly, free nodes the damped blend).
            new_v = np.where(
                vs_mask, sol_head,
                damping * sol_head + (1.0 - damping) * v_arr)
            voltages.update(zip(nodes, new_v))
            v_arr = new_v
        else:
            # Did not converge - check if it's "good enough"
            # For fast-switching circuits, accept solution if oscillating around stable point
            # Check if we're oscillating (voltages bouncing between similar values)
            if len(voltage_history) >= 3:
                # Calculate average of last 3 iterations
                avg_voltages = {}
                for node in nodes:
                    sum_v = 0.0
                    for snapshot in voltage_history[-3:]:
                        sum_v += snapshot.get(node, 0.0)
                    avg_voltages[node] = sum_v / 3.0

                # Check oscillation: variance relative to SPICE tolerance
                max_rel_variance = 0.0
                for node in nodes:
                    values = [s.get(node, 0.0) for s in voltage_history[-3:]]
                    variance = max(values) - min(values)
                    v_abs = max(abs(v) for v in values) if values else 0.0
                    threshold = self.vntol + self.reltol * v_abs
                    max_rel_variance = max(max_rel_variance, variance / (threshold + 1e-30))

                # Accept if oscillation is within 10x convergence
                # tolerance AND the averaged solution also passes the
                # MNA residual test (Phase 6b) — a small variance does
                # not prove the average is a physical fixed point.
                residual_ok = True
                if max_rel_variance < 10.0 and (
                        _has_nn_device(self.circuit)
                        or _has_full_stamp_device(self.circuit)):
                    # V7.5.1: gate extended to BSIM-CMG circuits — an
                    # averaged garbage point that commits here corrupts the
                    # charge history and, through the 1/dt companion, every
                    # subsequent piece of the march (see the DC twin).
                    avg_resid, rhs_scale = self._transient_residual_at(
                        avg_voltages, node_map, nodes, num_nodes,
                        matrix_size, gmin, time,
                    )
                    resid_threshold = max(_RESID_ABS_FLOOR, 100.0 * self.reltol * rhs_scale)
                    residual_ok = avg_resid <= resid_threshold
                # V7.5.0: never average-accept while NR limiting was
                # clamping a device — those iterates are extrapolations.
                if residual_ok:
                    for component in self.circuit.components:
                        if getattr(component, "_nr_limited", False):
                            residual_ok = False
                            break

                if max_rel_variance < 10.0 and residual_ok:
                    if self.debug:
                        print(f"  WARNING: Newton-Raphson oscillating at t={time:.6e}s")
                        print(f"  Max variance = {max_rel_variance:.2e} (accepting averaged solution)")
                    # Use averaged voltages
                    for node in nodes:
                        voltages[node] = avg_voltages[node]
                    voltages["0"] = 0.0
                    voltages["GND"] = 0.0
                    # C6a — there is no solution vector behind an averaged
                    # acceptance, so this step has no branch currents.
                    self._last_solution_tail = None
                    return voltages

            # Not good enough - print debug log and raise error
            if self.debug and debug_log is not None and len(debug_log) > 0:
                print(f"\nDEBUG: Convergence failure at t={time:.6e}s:")
                for log_line in debug_log:
                    print(log_line)
                print(f"\n  Final voltages:")
                for node in nodes[:5]:  # Print first 5 nodes
                    print(f"    {node}: {voltages[node]:.6f}V")

            raise RuntimeError(
                f"Newton-Raphson failed to converge at t={time:.6e}s after {max_iterations} iterations. "
                f"Final max delta: {max_delta:.2e} (tolerance: {tolerance:.2e})"
            )

        # Add ground nodes
        voltages["0"] = 0.0
        voltages["GND"] = 0.0

        return voltages

    def _stamp_mosfet_transient(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        voltages: Dict[str, float],
        limit: bool = True,
    ) -> None:
        """Stamp MOSFET conductance/current (DC part) + charge-based capacitance for transient."""
        # DC conductance + NR current source stamping (shared with DCSolver)
        _stamp_mosfet_dc(mosfet, mna_matrix, rhs, node_map, voltages, self.gmin,
                         limit=limit)

        # The charge companion below must linearize about the SAME bias
        # the resistive companion evaluated at: when NR limiting clamped
        # this device (V7.5.0), pick up the limited mapping it cached.
        # Identity case returns the original dict, so this is bit-neutral
        # for every unclamped iteration and for all non-L72 devices.
        if limit:
            v_lim = getattr(mosfet, "_nr_v_eval", None)
            if v_lim is not None:
                voltages = v_lim

        # --- Charge-based intrinsic capacitance stamping ---
        # Supports BE, Trapezoidal, and BDF-2 integration methods.
        # Theory: I_t(n+1) = coeff * Q_t(n+1) - history_terms
        drain, gate, source, bulk = mosfet.nodes

        # V7.5.1 — full 4-terminal charge companion for LEVEL=72. The 3x3
        # block below rebuilds a transcap matrix from the 5 SPICE cap
        # variables plus charge-conservation shortcuts; for floating-bulk
        # devices that produces SIGN-FLIPPED off-diagonals (measured
        # stamped +0.758 S vs true -0.758 S at dt=1e-15 on the charge
        # pump), and a wrong-signed Jacobian at small dt makes every
        # Newton iteration AMPLIFY the error ~15x. Devices exposing
        # get_charge_stamp stamp the condensed reactive OSDI Jacobian
        # directly — true dQ/dV, bulk row and column included.
        full_q = getattr(mosfet, "get_charge_stamp", None)
        if full_q is not None:
            if getattr(mosfet, "_q_prev", None) is None:
                return
            q4, c4 = full_q(voltages)
            dt = self._current_dt
            method = getattr(self, '_integration_method', 'trap')
            qp = mosfet._q_prev
            qp2 = getattr(mosfet, "_q_prev2", None)
            keys = ("qd", "qg", "qs", "qb")
            iprev = (getattr(mosfet, "_i_prev_drain", 0.0),
                     getattr(mosfet, "_i_prev_gate", 0.0),
                     getattr(mosfet, "_i_prev_source", 0.0),
                     getattr(mosfet, "_i_prev_bulk", 0.0))
            if method == 'bdf2' and qp2 is not None:
                coeff = 1.5 / dt
                hist = [(2.0 / dt) * qp[k] - (0.5 / dt) * qp2[k] for k in keys]
            elif method == 'trap' or (method == 'bdf2' and qp2 is None):
                coeff = 2.0 / dt
                hist = [coeff * qp[k] + ip for k, ip in zip(keys, iprev)]
            else:  # 'be'
                coeff = 1.0 / dt
                hist = [coeff * qp[k] for k in keys]
            idx = [node_map.get(n) if n not in ("0", "GND") else None
                   for n in mosfet.nodes]
            v_eval = [voltages.get(n, 0.0) for n in mosfet.nodes]
            for t in range(4):
                row = idx[t]
                if row is None:
                    continue
                e_t = coeff * q4[t] - hist[t]
                for j in range(4):
                    e_t -= coeff * c4[t, j] * v_eval[j]
                    col = idx[j]
                    if col is not None:
                        mna_matrix[row, col] += coeff * c4[t, j]
                rhs[row] -= e_t
            return

        if hasattr(mosfet, '_q_prev') and mosfet._q_prev is not None:
            charges = mosfet.get_charges(voltages)
            caps = mosfet.get_capacitances(voltages)
            dt = self._current_dt

            # Select integration method coefficients
            method = getattr(self, '_integration_method', 'trap')
            if method == 'bdf2' and hasattr(mosfet, '_q_prev2') and mosfet._q_prev2 is not None:
                coeff = 1.5 / dt
                h_g = (2.0 / dt) * mosfet._q_prev["qg"] - (0.5 / dt) * mosfet._q_prev2["qg"]
                h_d = (2.0 / dt) * mosfet._q_prev["qd"] - (0.5 / dt) * mosfet._q_prev2["qd"]
            elif method == 'trap' or (method == 'bdf2' and mosfet._q_prev2 is None):
                # Trapezoidal (or fallback when BDF-2 history not yet available)
                coeff = 2.0 / dt
                h_g = coeff * mosfet._q_prev["qg"] + getattr(mosfet, '_i_prev_gate', 0.0)
                h_d = coeff * mosfet._q_prev["qd"] + getattr(mosfet, '_i_prev_drain', 0.0)
            else:
                # Backward Euler: no i_prev history term
                coeff = 1.0 / dt
                h_g = coeff * mosfet._q_prev["qg"]
                h_d = coeff * mosfet._q_prev["qd"]

            # Terminal voltages at current NR iterate
            v_g = voltages.get(gate, 0.0)
            v_d = voltages.get(drain, 0.0)
            v_s = voltages.get(source, 0.0)

            # Capacitive currents at NR iterate V0
            i_g_cap = coeff * charges["qg"] - h_g
            i_d_cap = coeff * charges["qd"] - h_d
            # Source by charge conservation: i_s = -(i_g + i_d)

            # Jacobian entries: dI_t/dV_j = coeff * C_tj
            cgg = caps.get("cgg", 0.0)
            cgd = caps.get("cgd", 0.0)
            cgs = caps.get("cgs", 0.0)
            cdg = caps.get("cdg", 0.0)
            cdd = caps.get("cdd", 0.0)
            # Derived from charge conservation on each terminal
            cds = -(cdg + cdd)
            csg = -(cgg + cdg)
            csd = -(cgd + cdd)
            css = -(cgs + cds)

            # Optional C-stamp symmetrization (Phase 2a, gated by env var).
            # cgd vs cdg (and cgs/csg, cds/csd) are independent MLP outputs
            # that can drift asymmetric; under BDF-2 that seeds spurious
            # damping/growth on oscillators. Replacing each conjugate pair
            # with its mean restores reciprocity. Default off → bit-identical.
            if os.environ.get("NN_SYMMETRIC_CAPS", "0") == "1":
                cgd = cdg = 0.5 * (cgd + cdg)
                cgs = csg = 0.5 * (cgs + csg)
                cds = csd = 0.5 * (cds + csd)
                # Keep the source self-term charge-conserving after the
                # off-diagonal source entries were replaced.
                css = -(cgs + cds)

            scale = coeff

            # Node indices (None if ground)
            g_idx = node_map.get(gate) if gate != "0" else None
            d_idx = node_map.get(drain) if drain != "0" else None
            s_idx = node_map.get(source) if source != "0" else None

            # --- Stamp Jacobian (conductance matrix) ---
            # Gate row: dI_g/dV_g, dI_g/dV_d, dI_g/dV_s
            if g_idx is not None:
                mna_matrix[g_idx, g_idx] += scale * cgg
                if d_idx is not None:
                    mna_matrix[g_idx, d_idx] += scale * cgd
                if s_idx is not None:
                    mna_matrix[g_idx, s_idx] += scale * cgs

            # Drain row: dI_d/dV_g, dI_d/dV_d, dI_d/dV_s
            if d_idx is not None:
                if g_idx is not None:
                    mna_matrix[d_idx, g_idx] += scale * cdg
                mna_matrix[d_idx, d_idx] += scale * cdd
                if s_idx is not None:
                    mna_matrix[d_idx, s_idx] += scale * cds

            # Source row: dI_s/dV_g, dI_s/dV_d, dI_s/dV_s
            if s_idx is not None:
                if g_idx is not None:
                    mna_matrix[s_idx, g_idx] += scale * csg
                if d_idx is not None:
                    mna_matrix[s_idx, d_idx] += scale * csd
                mna_matrix[s_idx, s_idx] += scale * css

            # --- Stamp RHS (NR constant) ---
            # e_t = I_t(V0) - Σ_j(scale * C_tj * V0_j)
            e_g = i_g_cap - scale * (cgg * v_g + cgd * v_d + cgs * v_s)
            e_d = i_d_cap - scale * (cdg * v_g + cdd * v_d + cds * v_s)
            e_s = -(e_g + e_d)  # Charge conservation

            if g_idx is not None:
                rhs[g_idx] -= e_g
            if d_idx is not None:
                rhs[d_idx] -= e_d
            if s_idx is not None:
                rhs[s_idx] -= e_s

    def _transient_residual_at(
        self,
        voltages: Dict[str, float],
        node_map: Dict[str, int],
        nodes: List[str],
        num_nodes: int,
        matrix_size: int,
        gmin: float,
        time: float,
    ) -> tuple:
        """Re-stamp the transient MNA system at ``voltages``; return its residual.

        Phase 6b. Validates the oscillation-detection averaged solution
        in the transient NR loop: small inter-iteration variance does
        not prove the averaged voltages satisfy KCL (an oscillator with
        no DC equilibrium can fool the variance test). Re-stamps the
        full transient system — linear components, voltage sources at
        ``time``, and charge-based MOSFET stamps — at the candidate
        voltages and reports the infinity-norm residual.

        Args:
            voltages: candidate node-voltage dict.
            node_map: node-name → matrix-index map.
            nodes: ordered non-ground node names.
            num_nodes: number of non-ground nodes.
            matrix_size: full MNA dimension.
            gmin: GMIN level to stamp (matches the active stepping level).
            time: simulation time for time-varying voltage sources.

        Returns:
            ``(residual_inf, rhs_scale)``.
        """
        mna = _create_mna_matrix(matrix_size)
        rhs = np.zeros(matrix_size)
        for component in self.circuit.components:
            if not _is_mosfet(component):
                component.stamp_conductance(mna, node_map)
                component.stamp_rhs(rhs, node_map)
        self._stamp_voltage_sources(mna, rhs, node_map, num_nodes, time, voltages)
        for component in self.circuit.components:
            if _is_mosfet(component):
                # limit=False: residual probes must see the true KCL residual
                # and must not advance the NR-limiting anchor (V7.5.0).
                self._stamp_mosfet_transient(component, mna, rhs, node_map,
                                             voltages, limit=False)
        if gmin > self.gmin_final:
            self._apply_gmin_stepping(mna, node_map, gmin)
        iterate = np.zeros(matrix_size)
        for idx, node in enumerate(nodes):
            iterate[idx] = voltages[node]
        rhs_scale = float(np.max(np.abs(rhs))) if rhs.size else 0.0
        return _mna_residual_inf(mna, rhs, iterate), rhs_scale

    def solve(self) -> Dict[str, np.ndarray]:
        """
        Perform transient analysis from t=0 to t=t_stop.

        This method:
        1. Performs DC analysis at t=0 to find initial operating point
        2. Iterates through timesteps, updating capacitor companion models
        3. Solves circuit at each timestep using DC solver
        4. Returns time series of node voltages

        Returns:
            Dictionary containing:
                - "time": numpy array of time points
                - node names: numpy arrays of voltages at each time point

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular (unsolvable)
            RuntimeError: If DC solver fails to converge
        """
        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # V7.5.0: fresh NR-limiting anchors for this analysis. They then
        # persist ACROSS timesteps (SPICE semantics: the anchor is the
        # previous accepted evaluation), not per step.
        for component in self.circuit.components:
            reset = getattr(component, "reset_nr_limits", None)
            if reset is not None:
                reset()

        # Calculate number of timesteps.
        #
        # A bare ceil(t_stop/dt) is wrong when the quotient is an exact
        # integer mathematically but lands a hair above it in IEEE-754:
        # 5e-9/1e-11 == 500.00000000000006, so ceil returned 501, the loop ran
        # one extra step whose time clamped to t_stop, and the final sample was
        # duplicated (502 points for a 500-interval sweep). Snap to the nearest
        # integer when the quotient is within a relative epsilon of it; ceil
        # only a genuinely partial final step.
        ratio = self.t_stop / self.dt
        nearest = round(ratio)
        if abs(ratio - nearest) <= 1e-9 * max(1.0, abs(ratio)):
            num_intervals = int(nearest)
        else:
            num_intervals = int(np.ceil(ratio))
        num_steps = num_intervals + 1

        # Initialize storage arrays
        time = np.zeros(num_steps)
        voltages_over_time = {node: np.zeros(num_steps) for node in nodes}

        # C6a — per-step voltage-source branch currents, in the SAME ordinal
        # order as _stamp_voltage_sources / _store_source_currents walks the
        # component list. Index 0 is the pre-transient state (t=0): there is no
        # transient solve behind it, so it carries whatever the caller's DC
        # operating-point solve stored on the source (0.0 if there was none).
        vsources = [c for c in self.circuit.components
                    if isinstance(c, VoltageSource)]
        self.source_currents = {
            c.name: np.full(num_steps, np.nan) for c in vsources}
        self._branch_current_gaps = 0
        for c in vsources:
            self.source_currents[c.name][0] = c.calculate_current({})

        # V5 Phase A — A3.2: track the highest committed step so the
        # verify_nn_dc_tran inverter-tran runner can recover a partial
        # waveform when NR exhausts mid-transient (turns ERROR row into
        # numeric FAIL row).
        self._last_committed_step = 0
        self._partial_time = time
        self._partial_voltages = voltages_over_time

        # Step 1: Initial conditions from capacitor voltages
        # For transient analysis, we use the capacitor's initial voltage (v_prev)
        # instead of doing a DC solve (which would give steady-state, not transient)

        # Build initial voltage estimate based on capacitor v_prev values
        initial_voltages = {"0": 0.0, "GND": 0.0}

        # Use initial_guess if provided (from DC operating point)
        if self.initial_guess is not None:
            for node, voltage in self.initial_guess.items():
                if node not in ["0", "GND"]:
                    initial_voltages[node] = voltage

            # Initialize capacitor v_prev from DC operating point
            # This is critical: capacitors must start with their DC voltage,
            # otherwise the transient analysis will have incorrect initial conditions
            for component in self.circuit.components:
                if isinstance(component, Capacitor):
                    node_i, node_j = component.nodes[0], component.nodes[1]
                    v_i = self.initial_guess.get(node_i, 0.0)
                    v_j = self.initial_guess.get(node_j, 0.0)
                    # v_prev is the voltage across the capacitor (V_i - V_j)
                    component.v_prev = v_i - v_j

        # For each capacitor, estimate the node voltages based on v_prev
        for component in self.circuit.components:
            if isinstance(component, Capacitor):
                node_i, node_j = component.nodes[0], component.nodes[1]

                # If one node is ground, the other is at v_prev
                if node_j == "0" or node_j == "GND":
                    initial_voltages[node_i] = component.v_prev
                elif node_i == "0" or node_i == "GND":
                    initial_voltages[node_j] = -component.v_prev
                else:
                    # Both nodes are non-ground: we can't determine individual voltages
                    # from just the difference, so set them to 0 for now
                    # The first timestep will correct this
                    if node_i not in initial_voltages:
                        initial_voltages[node_i] = 0.0
                    if node_j not in initial_voltages:
                        initial_voltages[node_j] = 0.0

        # For any remaining nodes, set to 0V
        for node in nodes:
            if node not in initial_voltages:
                initial_voltages[node] = 0.0

        # Initialize MOSFET charge state for intrinsic capacitance tracking
        #
        # V7.2.0 Phase 2t (opt-in): warm every NN device's eval cache with
        # one batched call before the per-device init_charge_state loop —
        # the same cold batch-1 pattern as the per-step commit below, paid
        # once at t=0. PERTURBING (batched GEMM rows differ from single-row
        # evals in float32), hence default-off; see the commit-loop note.
        if os.environ.get("PYCIRCUITSIM_TRAN_BATCH_COMMIT", "0") == "1":
            _batch_eval_nn_mosfets(self.circuit, initial_voltages)
        for component in self.circuit.components:
            if _is_mosfet(component) and hasattr(component, 'init_charge_state'):
                component.init_charge_state(initial_voltages)

        # Store initial voltages
        time[0] = 0.0
        for node in nodes:
            voltages_over_time[node][0] = initial_voltages.get(node, 0.0)

        # Debug: Print initial voltages
        if self.debug:
            print(f"Initial transient voltages:")
            for node in sorted(nodes)[:5]:
                print(f"  V{node} = {initial_voltages.get(node, 0.0):.4f} V")

        # Step 2: Add pseudo-capacitors if enabled AND no DC OP provided
        # If a valid DC operating point was provided as initial_guess, skip
        # pseudo-transient and Gmin stepping — they create startup artifacts.
        has_dc_op = self.initial_guess is not None and len(self.initial_guess) > 0
        if has_dc_op:
            # DC OP provides correct initial conditions; convergence aids not needed
            effective_use_pseudo = False
            effective_use_gmin = False
            if self.debug:
                print(f"DC operating point provided — skipping pseudo-transient and Gmin stepping")
        else:
            effective_use_pseudo = self.use_pseudo_transient
            effective_use_gmin = self.use_gmin_stepping

        if effective_use_pseudo and self._has_non_linear_components():
            if self.debug:
                print(f"Adding pseudo-capacitors for better DC convergence (first {self.pseudo_transient_steps} steps)")
            self._add_pseudo_capacitors()

        # Step 3: Adaptive time-stepping with LTE-based sub-stepping
        # V7.5.0: the retry ladder is an adaptive march (see below); the
        # local piece may shrink to sub_dt·2⁻²⁴ before a step is declared
        # unconvergable. (The V5 Phase A 4/5-halve caps applied to the old
        # fixed-target ladder, which made the companions stiffer instead
        # of the step smaller and is gone.)
        has_non_linear = self._has_non_linear_components()
        is_nn_circuit = self._has_nn_devices()

        # LTE-adaptive sub-stepping: uses constructor parameters
        adaptive_substeps = 1
        max_substeps = self.max_substeps
        lte_safety_factor = self.lte_safety_factor

        # Stiffness tracking for BDF-2 auto-switching
        _stiff_switched = False  # Once True, stays on BDF-2

        # V7.5.2 — refine_output state (see __init__ docs). All of it must
        # live ACROSS output intervals: the adaptive piece size, the LTE
        # history and the breakpoint cursor survive interval boundaries,
        # or the refinement would re-seed ringing at every grid point.
        refine = self.refine_output
        _r = 0.0
        if refine:
            _bps = self._collect_breakpoints()
            _bp_idx = 0
            _v_init_vec = np.array([voltages_over_time[n][0] for n in nodes])
            _lte_hist: List[tuple] = [(0.0, _v_init_vec)]
            _dense_t: List[float] = [0.0]
            _dense_v: List[np.ndarray] = [_v_init_vec]
            _dense_i: List[Optional[np.ndarray]] = [None]
            _refine_dt_carry = self.dt
            _force_be_piece = False
            _RESTART_FRAC = 2.0 ** -6  # piece restart after a breakpoint

        for step in range(1, num_steps):
            # Current output time
            current_time = step * self.dt
            time[step] = min(current_time, self.t_stop)

            # Remove pseudo-capacitors after specified steps
            if effective_use_pseudo and step == self.pseudo_transient_steps + 1:
                if self.debug:
                    print(f"Removing pseudo-capacitors at step {step}")
                self._remove_pseudo_capacitors()

            # Integration method selection.
            #   'auto'  : BE (step 1) → Trap (step 2+) → BDF-2 (on stiffness)
            #   'gear2' : BE (step 1, as ngspice does) → BDF-2 pinned from
            #             step 2 on, stiffness trip irrelevant. Matches
            #             `.options method=gear maxord=2` decks, where the
            #             trapezoid's ringing corrupts exactly the measured
            #             quantities (slew crossing time, over/undershoot).
            if step == 1:
                self._integration_method = 'be'
            elif self.integration_method == 'gear2':
                self._integration_method = 'bdf2'
            elif _stiff_switched:
                self._integration_method = 'bdf2'
            else:
                self._integration_method = 'trap'

            # Set capacitor integration flags
            use_trap = self._integration_method == 'trap'
            self._use_trap_for_charges = use_trap
            for component in self.circuit.components:
                if isinstance(component, Capacitor):
                    component._use_trapezoidal = use_trap
                    component._method = self._integration_method
            # V7.5.2 — the step-level method; refine-mode pieces may
            # deviate to BE right after a breakpoint and must know what
            # to restore for the pieces that follow.
            _step_method = self._integration_method

            # Starting voltages for this output interval
            current_voltages = {}
            for node in nodes:
                current_voltages[node] = voltages_over_time[node][step - 1]
            current_voltages["0"] = 0.0
            current_voltages["GND"] = 0.0

            # Sub-step within this output interval
            n_subs = adaptive_substeps
            sub_dt = self.dt / n_subs

            for sub_idx in range(n_subs):
                sub_time = time[step - 1] + (sub_idx + 1) * sub_dt
                self._current_dt = sub_dt

                # Retry loop for NR convergence failures.
                #
                # V7.5.0 — a failed attempt now CUTS THE LOCAL TIMESTEP
                # and marches through the interval adaptively: halve the
                # piece from the last committed time on failure, double
                # it back (up to sub_dt) after success. The old ladder
                # halved the companion dt while keeping the SAME target
                # time, which is not a smaller step at all — the cap
                # companions then demand the full interval's dV inside
                # dt/2^n, so every "halving" made the system STIFFER
                # (measured on the AnalogGym charge pump: residual pinned
                # at ~2e3 A at "minimum dt"). A bistable element mid-
                # transition (the same deck's quench comparator) needs
                # fs-scale local steps for the capacitive terms to
                # dominate its loop gain, exactly as NGSPICE's timestep
                # control provides. A step that converges on the first
                # attempt takes the exact old code path, so never-failing
                # circuits are byte-identical.
                t_now = sub_time - sub_dt
                dt_try = min(sub_dt, _refine_dt_carry) if refine else sub_dt
                min_piece_dt = sub_dt * 2.0 ** -24
                pieces_done = 0
                halvings = 0
                _rejects = 0
                _MAX_PIECES = 4096

                while True:
                    remaining = sub_time - t_now
                    piece_dt = dt_try if dt_try < remaining else remaining
                    t_k = sub_time if piece_dt >= remaining else t_now + piece_dt
                    _on_bp = False
                    if refine:
                        # Advance past any breakpoint at/behind the march
                        # position, then clamp the piece so it LANDS on the
                        # next one instead of integrating across the slope
                        # discontinuity.
                        while (_bp_idx < len(_bps)
                               and _bps[_bp_idx] <= t_now + min_piece_dt):
                            _bp_idx += 1
                        if _bp_idx < len(_bps) and _bps[_bp_idx] <= t_k:
                            _on_bp = True
                            if _bps[_bp_idx] < t_k:
                                t_k = _bps[_bp_idx]
                                piece_dt = t_k - t_now
                        # BE for the piece FOLLOWING a breakpoint: the
                        # trapezoid's current history from before the
                        # corner is inconsistent across the discontinuity
                        # and is what seeds the corner ringing (NGSPICE
                        # likewise restarts at order 1 after breakpoints).
                        piece_method = ('be' if _force_be_piece
                                        else _step_method)
                        if self._integration_method != piece_method:
                            self._integration_method = piece_method
                            _piece_trap = piece_method == 'trap'
                            for component in self.circuit.components:
                                if isinstance(component, Capacitor):
                                    component._use_trapezoidal = _piece_trap
                                    component._method = piece_method
                    try:
                        self._current_dt = piece_dt

                        # Update capacitor companion models
                        for component in self.circuit.components:
                            if isinstance(component, Capacitor):
                                component.get_companion_model(piece_dt, component.v_prev)

                        # Solve for node voltages
                        if has_non_linear:
                            timestep_voltages = self._solve_timestep_newton(
                                nodes=nodes,
                                node_map=node_map,
                                num_nodes=num_nodes,
                                num_voltage_sources=num_voltage_sources,
                                initial_voltages=current_voltages,
                                time=t_k,
                                step_index=step - 1,
                                use_gmin=effective_use_gmin
                            )
                        else:
                            matrix_size = num_nodes + num_voltage_sources
                            mna_matrix = _create_mna_matrix(matrix_size)
                            rhs = np.zeros(matrix_size)

                            for component in self.circuit.components:
                                component.stamp_conductance(mna_matrix, node_map)
                                self._stamp_component_rhs(
                                    component, rhs, node_map, t_k)

                            self._stamp_voltage_sources(mna_matrix, rhs, node_map, num_nodes, t_k)

                            try:
                                solution = _solve_mna(mna_matrix, rhs)
                            except (np.linalg.LinAlgError, RuntimeError) as e:
                                raise np.linalg.LinAlgError(
                                    f"Circuit matrix is singular at t={t_k:.6f}s. "
                                    f"Check for floating nodes or short circuits."
                                ) from e

                            timestep_voltages = {}
                            for idx, node in enumerate(nodes):
                                timestep_voltages[node] = float(solution[idx])
                            timestep_voltages["0"] = 0.0
                            timestep_voltages["GND"] = 0.0
                            # C6a — branch-current tail of this linear solve.
                            self._last_solution_tail = np.array(
                                solution[num_nodes:])

                        # Piece succeeded — commit state
                        if refine:
                            _pre_commit = self._snapshot_tran_state()
                        for component in self.circuit.components:
                            if isinstance(component, Capacitor):
                                component.update_voltage(timestep_voltages)

                        # V7.2.0 Phase 2t (opt-in): batch the commit-path
                        # eval. The solved voltages were never evaluated by
                        # the NR-loop batch eval (it warms the *iterate*,
                        # not the accepted solution), so every device below
                        # runs a cache-cold batch-1 forward + backwards in
                        # get_charges — measured 75-85% of transient wall
                        # at 4x4..16x16 (plan §2.4t). One batched eval at
                        # timestep_voltages makes those hits warm.
                        # PERTURBING: a batched GEMM row != the single-row
                        # eval at the last float32 bit, and the committed
                        # charge history feeds every later step through the
                        # companion model — so default OFF behind
                        # PYCIRCUITSIM_TRAN_BATCH_COMMIT=1, gated with the
                        # other perturbing phases (§8.4). Flag-unset path
                        # is byte-identical. _require_nn_caps ran at solver
                        # entry, so the warmed cache carries the charges.
                        if os.environ.get(
                                "PYCIRCUITSIM_TRAN_BATCH_COMMIT", "0") == "1":
                            _batch_eval_nn_mosfets(
                                self.circuit, timestep_voltages)

                        for component in self.circuit.components:
                            if _is_mosfet(component) and hasattr(component, 'update_charge_state'):
                                terminal_currents = {}
                                if hasattr(component, '_q_prev') and component._q_prev is not None:
                                    charges_new = component.get_charges(timestep_voltages)
                                    dt_eff = piece_dt
                                    method = self._integration_method
                                    if method == 'bdf2' and hasattr(component, '_q_prev2') and component._q_prev2 is not None:
                                        coeff = 1.5 / dt_eff
                                        h_g = (2.0 / dt_eff) * component._q_prev["qg"] - (0.5 / dt_eff) * component._q_prev2["qg"]
                                        h_d = (2.0 / dt_eff) * component._q_prev["qd"] - (0.5 / dt_eff) * component._q_prev2["qd"]
                                    elif method == 'trap':
                                        coeff = 2.0 / dt_eff
                                        h_g = coeff * component._q_prev["qg"] + getattr(component, '_i_prev_gate', 0.0)
                                        h_d = coeff * component._q_prev["qd"] + getattr(component, '_i_prev_drain', 0.0)
                                    else:  # 'be'
                                        coeff = 1.0 / dt_eff
                                        h_g = coeff * component._q_prev["qg"]
                                        h_d = coeff * component._q_prev["qd"]
                                    terminal_currents["i_gate"] = coeff * charges_new["qg"] - h_g
                                    terminal_currents["i_drain"] = coeff * charges_new["qd"] - h_d
                                    # V7.5.1: the full 4-terminal charge
                                    # companion needs source/bulk history too.
                                    if hasattr(component, "get_charge_stamp"):
                                        for key, name in (("qs", "i_source"),
                                                          ("qb", "i_bulk")):
                                            if method == 'bdf2' and hasattr(component, '_q_prev2') and component._q_prev2 is not None:
                                                h_t = (2.0 / dt_eff) * component._q_prev[key] - (0.5 / dt_eff) * component._q_prev2[key]
                                            elif method == 'trap':
                                                h_t = coeff * component._q_prev[key] + getattr(
                                                    component, f"_i_prev_{name[2:]}", 0.0)
                                            else:
                                                h_t = coeff * component._q_prev[key]
                                            terminal_currents[name] = coeff * charges_new[key] - h_t
                                component.update_charge_state(timestep_voltages, terminal_currents)

                        if refine:
                            _v_vec = np.array(
                                [timestep_voltages[n] for n in nodes])
                            _r = 0.0
                            if (len(_lte_hist) >= 3
                                    and piece_dt > min_piece_dt
                                    and _rejects < 8):
                                _r = self._refine_lte_ratio(
                                    _lte_hist, t_k, _v_vec)
                                if _r > 1.0:
                                    # Reject: un-commit the piece and
                                    # re-march it with a smaller dt
                                    # (NGSPICE-style truncation-error
                                    # timestep control; _rejects caps a
                                    # pathological reject loop).
                                    self._restore_tran_state(_pre_commit)
                                    _rejects += 1
                                    dt_try = max(
                                        piece_dt * max(
                                            0.25, 0.9 * _r ** (-1.0 / 3.0)),
                                        min_piece_dt)
                                    continue
                            _rejects = 0
                            _lte_hist.append((t_k, _v_vec))
                            if len(_lte_hist) > 4:
                                _lte_hist.pop(0)
                            _dense_t.append(float(t_k))
                            _dense_v.append(_v_vec)
                            _tail = self._last_solution_tail
                            _dense_i.append(None if _tail is None
                                            else np.asarray(_tail).copy())

                        # Advance the march; leave when the interval is done.
                        current_voltages = timestep_voltages
                        t_now = t_k
                        pieces_done += 1
                        if refine:
                            _force_be_piece = False
                            if _on_bp:
                                _bp_idx += 1
                                _force_be_piece = True
                                dt_try = max(sub_dt * _RESTART_FRAC,
                                             min_piece_dt)
                            else:
                                # LTE-scaled growth, at most 2x per piece.
                                _grow = 2.0
                                if _r > 0.0:
                                    _grow = min(2.0, max(
                                        1.0, 0.9 * _r ** (-1.0 / 3.0)))
                                dt_try = min(piece_dt * _grow, sub_dt)
                            _refine_dt_carry = dt_try
                        if t_k >= sub_time:
                            break
                        if pieces_done >= _MAX_PIECES:
                            raise SimStepLimit(
                                f"Timestep at t={sub_time:.2e}s needed more "
                                f"than {_MAX_PIECES} sub-pieces")
                        if not refine:
                            # Recover the piece size after success.
                            dt_try = piece_dt * 2.0
                            if dt_try > sub_dt:
                                dt_try = sub_dt

                    except RuntimeError as e:
                        halvings += 1
                        dt_try = piece_dt * 0.5
                        if dt_try < min_piece_dt:
                            raise RuntimeError(
                                f"Failed to converge at t={sub_time:.2e}s even with minimum dt. "
                                f"Original error: {e}"
                            ) from e
                        # V5 Phase A — A3: log every dt-halve event so
                        # verification scripts can flag cells that needed
                        # >1 halving (escalates as a model-fit issue).
                        # V7.5.0: sim_time is the local march position, and
                        # dt_after the piece the march will retry with.
                        self._dt_halve_events.append({
                            "step": step,
                            "sub_idx": sub_idx,
                            "sim_time": float(t_now),
                            "halve_num": halvings,
                            "dt_before": float(piece_dt),
                            "dt_after": float(dt_try),
                            "is_nn_circuit": bool(is_nn_circuit),
                            "error_msg": str(e),
                        })
                        if self.debug:
                            print(f"  WARNING: Convergence failed at t={t_now + piece_dt:.2e}s, "
                                  f"retrying from t={t_now:.2e}s with dt={dt_try:.2e}s")

            # Store at output point
            for node in nodes:
                voltages_over_time[node][step] = current_voltages[node]
            # C6a — commit this step's branch currents (NaN + a counted gap
            # when the accepted solve produced no solution vector).
            tail = self._last_solution_tail
            if tail is None:
                self._branch_current_gaps += 1
            else:
                for vs_idx, comp in enumerate(vsources):
                    if vs_idx < len(tail):
                        self.source_currents[comp.name][step] = float(tail[vs_idx])
            # V5 Phase A — A3.2: track committed step for partial-recovery.
            self._last_committed_step = step

            # Stiffness detection: if NR took > 20 iterations, switch to BDF-2
            # (disabled under 'gear2', which is already pinned to BDF-2).
            if (not _stiff_switched and has_non_linear and step > 2
                    and self.integration_method != 'gear2'
                    and getattr(self, '_last_nr_iterations', 0) > 20):
                _stiff_switched = True
                if self.debug:
                    print(f"  Stiffness detected at step {step} (NR iters={self._last_nr_iterations}) -> switching to BDF-2")

            # LTE estimation for adaptive sub-stepping (need >= 3 output points)
            if step >= 2:
                max_lte_ratio = 0.0
                for node in nodes:
                    v_np1 = voltages_over_time[node][step]
                    v_n = voltages_over_time[node][step - 1]
                    v_nm1 = voltages_over_time[node][step - 2]
                    d2v = abs(v_np1 - 2.0 * v_n + v_nm1)
                    lte = d2v / 12.0  # Trapezoidal LTE coefficient
                    threshold = self.vntol + self.reltol * max(abs(v_np1), abs(v_n))
                    if threshold > 0:
                        max_lte_ratio = max(max_lte_ratio, lte / threshold)

                # Account for current sub-stepping: effective error ~ raw / n^2
                # (Trapezoidal order 2: global error is O(h^2), h = dt/n)
                effective_lte = max_lte_ratio / (adaptive_substeps ** 2)

                # Compute optimal sub-steps: n = ceil(sqrt(raw_lte / threshold))
                if effective_lte > lte_safety_factor:
                    optimal_n = int(np.ceil(np.sqrt(max_lte_ratio / lte_safety_factor)))
                    adaptive_substeps = min(max(optimal_n, adaptive_substeps), max_substeps)
                    if self.debug:
                        print(f"  LTE={max_lte_ratio:.1f} eff={effective_lte:.2f} at t={current_time:.2e}s -> substeps={adaptive_substeps}")
                elif effective_lte < lte_safety_factor / 8 and adaptive_substeps > 1:
                    adaptive_substeps = max(adaptive_substeps // 2, 1)
                    if self.debug:
                        print(f"  LTE={max_lte_ratio:.1f} eff={effective_lte:.2f} at t={current_time:.2e}s -> substeps={adaptive_substeps}")

        # Prepare results dictionary
        if refine:
            # V7.5.2 — emit every committed piece (non-uniform time axis;
            # the fixed grid points all remain present exactly). Branch
            # currents follow the same axis; index 0 keeps the
            # pre-transient value, exactly like the fixed-grid path.
            dense_time = np.asarray(_dense_t)
            dense_mat = np.vstack(_dense_v)
            results = {"time": dense_time}
            for j, node in enumerate(nodes):
                results[node] = dense_mat[:, j]
            n_dense = len(_dense_t)
            sc = {c.name: np.full(n_dense, np.nan) for c in vsources}
            self._branch_current_gaps = 0
            for c in vsources:
                sc[c.name][0] = c.calculate_current({})
            for k in range(1, n_dense):
                tail_k = _dense_i[k]
                if tail_k is None:
                    self._branch_current_gaps += 1
                    continue
                for vs_idx, comp in enumerate(vsources):
                    if vs_idx < len(tail_k):
                        sc[comp.name][k] = float(tail_k[vs_idx])
            self.source_currents = sc
            return results

        results = {"time": time}
        for node in nodes:
            results[node] = voltages_over_time[node]

        return results

    def _stamp_voltage_sources(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int,
        time: float = 0.0,
        voltages: Dict[str, float] = None,
    ) -> None:
        """
        Stamp voltage source equations to MNA matrix.

        For each voltage source, we add:
        - B matrix column: connection to node voltages
        - C matrix row: voltage constraint equation
        - RHS entry: voltage source value (for linear) or mismatch (for Newton-Raphson)

        The voltage source equation is: V_pos - V_neg = V_source
        For Newton-Raphson: delta_V_pos - delta_V_neg = V_source - (V_pos_old - V_neg_old)

        Args:
            mna_matrix: MNA matrix to modify (in-place)
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
            time: Current simulation time (for time-varying sources)
            voltages: Current voltage estimate (for Newton-Raphson mismatch computation)
        """
        from pycircuitsim.models.passive import PulseVoltageSource

        voltage_source_index = 0

        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Get voltage source nodes
                pos_node = component.nodes[0]  # Positive terminal
                neg_node = component.nodes[1]  # Negative terminal

                # Get voltage value (support time-varying sources)
                if isinstance(component, PulseVoltageSource):
                    voltage_target = component.get_voltage_at_time(time)
                else:
                    voltage_target = component.voltage

                # The row index for this voltage source's equation
                vs_row = num_nodes + voltage_source_index

                # Stamp B matrix (voltage source current flows into nodes)
                if pos_node != "0" and pos_node in node_map:
                    pos_idx = node_map[pos_node]
                    mna_matrix[vs_row, pos_idx] += 1.0
                    mna_matrix[pos_idx, vs_row] += 1.0

                if neg_node != "0" and neg_node in node_map:
                    neg_idx = node_map[neg_node]
                    mna_matrix[vs_row, neg_idx] -= 1.0
                    mna_matrix[neg_idx, vs_row] -= 1.0

                # Stamp voltage source value to RHS
                # Use direct voltage value for companion model consistency.
                # The companion model for MOSFETs solves for V directly,
                # so voltage sources should also use direct form.
                # NOTE: Previous implementation used voltage_target - (v_pos - v_neg)
                # which caused oscillation in Newton-Raphson. The correct formulation
                # (matching DC solver) is to use the direct voltage value.
                rhs[vs_row] = voltage_target

                # Move to next voltage source
                voltage_source_index += 1

    def _stamp_component_rhs(
        self,
        component,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        time: float,
    ) -> None:
        """Stamp a non-voltage-source component's RHS at the current time.

        Identical to ``component.stamp_rhs`` except that a PULSE current
        source is evaluated AT ``time`` — symmetric to the PulseVoltageSource
        special case in ``_stamp_voltage_sources``. Every other component
        (resistor, capacitor companion, DC current source) is time-less, so
        this is a pure pass-through for them.

        Args:
            component: Component to stamp
            rhs: RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            time: Current simulation time in seconds
        """
        from pycircuitsim.models.passive import PulseCurrentSource

        if isinstance(component, PulseCurrentSource):
            component.stamp_rhs_at_time(rhs, node_map, time)
        else:
            component.stamp_rhs(rhs, node_map)

    def __repr__(self) -> str:
        """String representation of the solver."""
        return (
            f"TransientSolver(circuit={self.circuit}, "
            f"t_stop={self.t_stop}, "
            f"dt={self.dt})"
        )


class ACSolver:
    """
    AC (small-signal frequency domain) Solver for linear and linearized circuits.

    The ACSolver performs small-signal AC analysis by:
    1. Computing DC operating point using DCSolver
    2. Linearizing the circuit around the operating point
    3. Building complex MNA matrix (with capacitances and transconductances)
    4. Sweeping frequency and computing complex node voltages

    Algorithm:
    1. DC analysis to find operating point (all AC sources = 0)
    2. For each frequency:
       a. Build complex admittance matrix Y = G + jwC
       b. Stamp MOSFET small-signal parameters (gm, gds, Cgs, Cgd)
       c. Stamp AC sources to RHS
       d. Solve Y * V = I for complex voltages
       e. Store magnitude and phase

    Attributes:
        circuit: Circuit object containing components and topology
        dc_solution: DC operating point voltages (computed once)
    """

    def __init__(self, circuit: Circuit, dc_solution: Optional[Dict[str, float]] = None):
        """
        Initialize the AC Solver.

        Args:
            circuit: Circuit object to analyze
            dc_solution: Optional pre-computed DC operating point (if None, will compute)
        """
        self.circuit = circuit
        self.dc_solution = dc_solution
        # V7.0.1 — the small-signal stamp reads the full transcapacitance
        # block, so the NN devices need their charge Jacobians (see
        # TransientSolver.__init__).
        _require_nn_caps(circuit)

    def solve(self, frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform AC analysis over a range of frequencies.

        This method:
        1. Computes DC operating point (if not provided)
        2. For each frequency, solves the small-signal circuit
        3. Returns complex voltages at each node for each frequency

        Args:
            frequencies: Array of frequencies in Hz

        Returns:
            Dictionary containing:
                - "frequency": numpy array of frequencies (Hz)
                - node names: numpy arrays of complex voltages at each frequency

        Raises:
            np.linalg.LinAlgError: If the circuit matrix is singular
            RuntimeError: If DC operating point fails to converge
        """
        # Step 1: Compute DC operating point if not provided
        if self.dc_solution is None:
            from pycircuitsim.solver import DCSolver
            dc_solver = DCSolver(self.circuit)
            with dc_solver:
                self.dc_solution = dc_solver.solve()

        # Get circuit topology
        nodes = self.circuit.get_nodes()
        node_map = self.circuit.get_node_map()
        num_nodes = len(nodes)
        num_voltage_sources = self.circuit.count_voltage_sources()

        # Matrix size: num_nodes + num_voltage_sources
        matrix_size = num_nodes + num_voltage_sources

        # Initialize storage arrays for results
        num_freqs = len(frequencies)
        voltages_over_freq = {node: np.zeros(num_freqs, dtype=complex) for node in nodes}
        voltages_over_freq["frequency"] = frequencies

        # Precompute small-signal MOSFET parameters ONCE at the DC operating
        # point. gm/gds/gmb and the terminal capacitances are linearizations
        # about the (fixed) operating point, so they do not depend on
        # frequency. Evaluating the — possibly torch-backed (LEVEL=73) or
        # OSDI-backed (LEVEL=72) — compact model once per device, instead of
        # once per (device, frequency), keeps the sweep cheap.
        mosfet_ss = self._precompute_mosfet_small_signal()

        # Step 2: Frequency sweep
        for freq_idx, freq in enumerate(frequencies):
            omega = 2 * np.pi * freq

            # Build complex MNA matrix: Y = G + jwC
            mna_matrix = np.zeros((matrix_size, matrix_size), dtype=complex)
            rhs = np.zeros(matrix_size, dtype=complex)

            # Stamp linear components (resistors, capacitors)
            for component in self.circuit.components:
                if not _is_mosfet(component):
                    self._stamp_component_ac(component, mna_matrix, rhs, node_map, omega)

            # Stamp MOSFETs (small-signal model: gm, gds, gmb + transcapacitances)
            for component in self.circuit.components:
                if _is_mosfet(component):
                    self._stamp_mosfet_ac(
                        component, mna_matrix, node_map, omega,
                        mosfet_ss[id(component)])

            # Stamp voltage sources (DC sources become short circuits, AC sources become AC stimulus)
            self._stamp_voltage_sources_ac(mna_matrix, rhs, node_map, num_nodes, omega)

            # Solve the complex linear system
            try:
                solution = np.linalg.solve(mna_matrix, rhs)
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(
                    f"Circuit matrix is singular at f={freq:.3e} Hz. "
                    f"Check circuit topology or AC sources."
                ) from e

            # Extract complex node voltages from solution
            for idx, node in enumerate(nodes):
                voltages_over_freq[node][freq_idx] = complex(solution[idx])

        return voltages_over_freq

    def _stamp_component_ac(
        self,
        component,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        omega: float
    ) -> None:
        """
        Stamp a passive component for AC analysis.

        For AC analysis:
        - Resistors: stamp conductance G (same as DC)
        - Capacitors: stamp admittance jwC (frequency-dependent)
        - Voltage sources: handled separately
        - Current sources: stamp to RHS (AC current sources not yet supported)

        Args:
            component: Component to stamp
            mna_matrix: Complex MNA matrix to modify (in-place)
            rhs: Complex RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            omega: Angular frequency (2*pi*f) in rad/s
        """
        from pycircuitsim.models.passive import Resistor, Capacitor, CurrentSource

        if isinstance(component, Resistor):
            # Resistor: stamp conductance (real, frequency-independent)
            node_i, node_j = component.nodes[0], component.nodes[1]
            g = component.conductance

            # Stamp exactly as in DC analysis
            if node_i != "0" and node_i in node_map:
                idx_i = node_map[node_i]
                mna_matrix[idx_i, idx_i] += g

                if node_j != "0" and node_j in node_map:
                    idx_j = node_map[node_j]
                    mna_matrix[idx_i, idx_j] -= g
                    mna_matrix[idx_j, idx_i] -= g

            if node_j != "0" and node_j in node_map:
                idx_j = node_map[node_j]
                mna_matrix[idx_j, idx_j] += g

        elif isinstance(component, Capacitor):
            # Capacitor: stamp admittance Y_C = jwC (imaginary, frequency-dependent)
            node_i, node_j = component.nodes[0], component.nodes[1]
            y_c = 1j * omega * component.capacitance

            # Stamp same pattern as resistor, but with complex admittance
            if node_i != "0" and node_i in node_map:
                idx_i = node_map[node_i]
                mna_matrix[idx_i, idx_i] += y_c

                if node_j != "0" and node_j in node_map:
                    idx_j = node_map[node_j]
                    mna_matrix[idx_i, idx_j] -= y_c
                    mna_matrix[idx_j, idx_i] -= y_c

            if node_j != "0" and node_j in node_map:
                idx_j = node_map[node_j]
                mna_matrix[idx_j, idx_j] += y_c

        elif isinstance(component, CurrentSource):
            # AC current source: stamp the complex phasor I = mag * e^{j*phase}
            # to the RHS. The DC bias current is set to zero in small-signal
            # AC analysis (independent DC sources are suppressed), so only the
            # ac_magnitude contributes. Sign convention matches the DC
            # CurrentSource.stamp_rhs (NGSPICE): the current is DRAWN out of
            # node_i (the + terminal) and pushed into node_j.
            ac_mag = getattr(component, "ac_magnitude", 0.0)
            if ac_mag != 0.0:
                ac_phase_rad = np.deg2rad(getattr(component, "ac_phase", 0.0))
                i_ac = ac_mag * np.exp(1j * ac_phase_rad)
                node_i, node_j = component.nodes[0], component.nodes[1]
                if node_i != "0" and node_i in node_map:
                    rhs[node_map[node_i]] -= i_ac
                if node_j != "0" and node_j in node_map:
                    rhs[node_map[node_j]] += i_ac

        # VoltageSource handled separately in _stamp_voltage_sources_ac

    def _precompute_mosfet_small_signal(self) -> Dict[int, tuple]:
        """Evaluate every MOSFET's small-signal parameters once at the OP.

        Returns a dict keyed by ``id(mosfet)`` → ``(g_ds, g_m, g_mb, caps)``
        where ``caps`` is the ``{cgg, cgd, cgs, cdg, cdd}`` dict from
        ``mosfet.get_capacitances`` (empty dict if the model does not expose
        capacitances). These are linearizations about the fixed DC operating
        point and so are frequency-independent — computing them once avoids a
        per-frequency compact-model re-evaluation (important for the
        torch-backed LEVEL=73 model).
        """
        ss: Dict[int, tuple] = {}
        for component in self.circuit.components:
            if not _is_mosfet(component):
                continue
            # V7.5.2 — full 4-terminal small-signal block for LEVEL=72.
            # The 3-conductance + 3x3-cap path below linearizes only the
            # channel about (g, d, s): junction/gate-leakage conductances
            # are invisible (the DC solve stopped trusting those opvars in
            # V7.5.0 for exactly that reason), and the 3x3 cap expansion
            # stamps SIGN-FLIPPED transcap off-diagonals for devices whose
            # bulk is not tied to the source rail (the V7.5.1 transient
            # defect, same hazard). Devices exposing get_terminal_stamp
            # carry Y = G4 + jw*C4 from the condensed OSDI Jacobians,
            # evaluated once at the DC OP — exactly NGSPICE's AC load.
            if hasattr(component, "get_terminal_stamp"):
                _, g4 = component.get_terminal_stamp(self.dc_solution)
                _, c4 = component.get_charge_stamp(self.dc_solution)
                # NO external gmin here: the DC stamp's gmin across
                # d-s/d-b/s-b is a Newton convergence aid; NGSPICE's AC
                # load carries only the model's own Jacobian (the OSDI
                # model handles gmin internally via $simparam). An extra
                # 1e-12 S across the junctions injects a measurable
                # spurious signal into high-impedance bulk nodes
                # (measured: 6% on a 100k-tied NMOS bulk, and a fake
                # 2.5e-7 V response on a PMOS bulk NGSPICE holds at zero).
                ss[id(component)] = ("full4", g4, c4)
                continue
            conductance_result = component.get_conductance(self.dc_solution)
            if len(conductance_result) == 2:
                g_ds, g_m = conductance_result
                g_mb = 0.0
            else:
                g_ds, g_m, g_mb = conductance_result
            # SPICE GMIN floor for numerical stability (matches DC stamping).
            g_ds = max(g_ds, 1e-12)
            if hasattr(component, "get_capacitances"):
                caps = component.get_capacitances(self.dc_solution)
            else:
                caps = {}
            ss[id(component)] = (g_ds, g_m, g_mb, caps)
        return ss

    def _stamp_cap_ac(
        self,
        mna_matrix: np.ndarray,
        node_map: Dict[str, int],
        gate: str,
        drain: str,
        source: str,
        caps: Dict[str, float],
        omega: float,
    ) -> None:
        """Stamp the small-signal MOSFET transcapacitances as jω·C admittance.

        The compact models expose the source-referenced, SPICE-sign-convention
        condensed capacitances {cgg, cgd, cdg, cdd} (PyCMG `_condense_caps`,
        matching NGSPICE's `@n1[cXX]` operating-point variables). These form
        the 2-port (gate, drain) capacitance matrix referenced to source:

            I_g = jω [  cgg·(Vg−Vs) − cgd·(Vd−Vs) ]
            I_d = jω [ −cdg·(Vg−Vs) + cdd·(Vd−Vs) ]
            I_s = −(I_g + I_d)               (charge conservation / KCL)

        Embedding into the nodal 3×3 over {g, d, s} gives a matrix whose rows
        and columns each sum to zero, stamped here (× jω). At ω→0 the stamp
        vanishes, so AC reduces to the resistive small-signal model at DC.
        """
        cgg = caps.get("cgg", 0.0)
        cgd = caps.get("cgd", 0.0)
        cdg = caps.get("cdg", 0.0)
        cdd = caps.get("cdd", 0.0)
        if cgg == 0.0 and cgd == 0.0 and cdg == 0.0 and cdd == 0.0:
            return
        jw = 1j * omega
        # Nodal 3×3 (rows/cols sum to zero):
        #            gate        drain        source
        # gate   [  cgg        -cgd          cgd-cgg          ]
        # drain  [ -cdg         cdd          cdg-cdd          ]
        # source [  cdg-cgg     cgd-cdd      cgg-cgd-cdg+cdd  ]
        entries = (
            (gate,   gate,   cgg),
            (gate,   drain, -cgd),
            (gate,   source, cgd - cgg),
            (drain,  gate,  -cdg),
            (drain,  drain,  cdd),
            (drain,  source, cdg - cdd),
            (source, gate,   cdg - cgg),
            (source, drain,  cgd - cdd),
            (source, source, cgg - cgd - cdg + cdd),
        )
        for row_node, col_node, val in entries:
            if val == 0.0:
                continue
            if row_node == "0" or row_node not in node_map:
                continue
            if col_node == "0" or col_node not in node_map:
                continue
            mna_matrix[node_map[row_node], node_map[col_node]] += jw * val

    def _stamp_mosfet_ac(
        self,
        mosfet,
        mna_matrix: np.ndarray,
        node_map: Dict[str, int],
        omega: float,
        ss: tuple,
    ) -> None:
        """
        Stamp MOSFET small-signal model for AC analysis.

        Small-signal MOSFET model includes:
        - gm: transconductance (gate to drain)
        - gds: output conductance (drain to source)
        - gmb: bulk transconductance (bulk to drain, if applicable)
        - Cgg/Cgd/Cdg/Cdd: the source-referenced transcapacitance matrix
          (Miller-coupled gate-drain feedback + gate/drain self-capacitance),
          stamped as jω·C by `_stamp_cap_ac`.

        Args:
            mosfet: MOSFET component (NMOS or PMOS)
            mna_matrix: Complex MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
            omega: Angular frequency (2*pi*f) in rad/s
            ss: precomputed (g_ds, g_m, g_mb, caps) at the DC operating point
        """
        # Get MOSFET terminals
        drain = mosfet.nodes[0]
        gate = mosfet.nodes[1]
        source = mosfet.nodes[2]
        bulk = mosfet.nodes[3]

        # V7.5.2 — full 4-terminal admittance for LEVEL=72 (see
        # _precompute_mosfet_small_signal): Y[t,j] = G4[t,j] + jw*C4[t,j]
        # over terminal order [d, g, s, b], ground rows/cols dropped.
        if ss[0] == "full4":
            _, g4, c4 = ss
            jw = 1j * omega
            idx = [node_map.get(n) if n not in ("0", "GND") else None
                   for n in mosfet.nodes]
            for t in range(4):
                row = idx[t]
                if row is None:
                    continue
                for j in range(4):
                    col = idx[j]
                    if col is not None:
                        mna_matrix[row, col] += g4[t, j] + jw * c4[t, j]
            return

        # Small-signal conductances + capacitances (precomputed once at the OP)
        g_ds, g_m, g_mb, caps = ss

        # Stamp conductances (same as DC, but to complex matrix)
        # g_ds between drain and source
        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            mna_matrix[d_idx, d_idx] += g_ds

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_ds

        if drain != "0" and drain in node_map and source != "0" and source in node_map:
            d_idx = node_map[drain]
            s_idx = node_map[source]
            mna_matrix[d_idx, s_idx] -= g_ds
            mna_matrix[s_idx, d_idx] -= g_ds

        # g_m transconductance: i_d = gm * (v_g - v_s)
        # Stamp for drain equation (KCL at drain node)
        if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
            g_idx = node_map[gate]
            d_idx = node_map[drain]
            mna_matrix[d_idx, g_idx] += g_m

        if source != "0" and source in node_map and drain != "0" and drain in node_map:
            s_idx = node_map[source]
            d_idx = node_map[drain]
            mna_matrix[d_idx, s_idx] -= g_m

        # Stamp for source equation (KCL at source node: current into source = -i_d)
        if gate != "0" and gate in node_map and source != "0" and source in node_map:
            g_idx = node_map[gate]
            s_idx = node_map[source]
            mna_matrix[s_idx, g_idx] -= g_m

        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_m

        # g_mb bulk transconductance: i_d = gmb * (v_b - v_s)
        if abs(g_mb) > 1e-12 and bulk != source:
            # Stamp for drain equation
            if bulk != "0" and bulk in node_map and drain != "0" and drain in node_map:
                b_idx = node_map[bulk]
                d_idx = node_map[drain]
                mna_matrix[d_idx, b_idx] += g_mb

            if source != "0" and source in node_map and drain != "0" and drain in node_map:
                s_idx = node_map[source]
                d_idx = node_map[drain]
                mna_matrix[d_idx, s_idx] -= g_mb

            # Stamp for source equation
            if bulk != "0" and bulk in node_map and source != "0" and source in node_map:
                b_idx = node_map[bulk]
                s_idx = node_map[source]
                mna_matrix[s_idx, b_idx] -= g_mb

            if source != "0" and source in node_map:
                s_idx = node_map[source]
                mna_matrix[s_idx, s_idx] += g_mb

        # Frequency-dependent small-signal capacitances (Cgg/Cgd/Cdg/Cdd) —
        # the Miller-coupled gate↔drain feedback that sets the device roll-off.
        self._stamp_cap_ac(mna_matrix, node_map, gate, drain, source, caps, omega)

    def _stamp_voltage_sources_ac(
        self,
        mna_matrix: np.ndarray,
        rhs: np.ndarray,
        node_map: Dict[str, int],
        num_nodes: int,
        omega: float = 0.0
    ) -> None:
        """
        Stamp voltage sources for AC analysis.

        For AC analysis:
        - DC voltage sources become SHORT CIRCUITS (V_ac = 0)
        - AC voltage sources provide AC stimulus (V_ac = magnitude * e^(j*phase))

        The voltage source stamping adds:
        - B/C matrix blocks (same as DC)
        - RHS: AC magnitude with phase for AC sources, 0 for DC-only sources

        An Inductor is a 0 V source (DC short) that additionally carries its
        reactance on its OWN branch row, turning that row into
        ``V_pos - V_neg - jwL*I_L = 0`` — an open circuit at high frequency.

        Args:
            mna_matrix: Complex MNA matrix to modify (in-place)
            rhs: Complex RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
            num_nodes: Number of non-ground nodes
            omega: Angular frequency (2*pi*f) in rad/s — only used by Inductor
        """
        from pycircuitsim.models.passive import Inductor

        voltage_source_index = 0

        for component in self.circuit.components:
            if isinstance(component, VoltageSource):
                # Get voltage source nodes
                pos_node = component.nodes[0]
                neg_node = component.nodes[1]

                # The row index for this voltage source's equation
                vs_row = num_nodes + voltage_source_index

                # Stamp B/C matrix (same as DC analysis)
                if pos_node != "0" and pos_node in node_map:
                    pos_idx = node_map[pos_node]
                    mna_matrix[vs_row, pos_idx] += 1.0
                    mna_matrix[pos_idx, vs_row] += 1.0

                if neg_node != "0" and neg_node in node_map:
                    neg_idx = node_map[neg_node]
                    mna_matrix[vs_row, neg_idx] -= 1.0
                    mna_matrix[neg_idx, vs_row] -= 1.0

                # Stamp AC stimulus to RHS
                # Convert AC magnitude and phase to complex phasor
                ac_mag = component.ac_magnitude
                ac_phase_deg = component.ac_phase
                ac_phase_rad = np.deg2rad(ac_phase_deg)

                # Complex phasor: V = magnitude * e^(j*phase)
                v_ac = ac_mag * np.exp(1j * ac_phase_rad)

                rhs[vs_row] = v_ac

                # Inductor: add the branch reactance to its own diagonal, so
                # the branch row reads V_pos - V_neg - jwL*I_L = 0. (The
                # branch current I_L is oriented pos -> neg through the
                # device by the B/C incidence rows above.)
                if isinstance(component, Inductor):
                    mna_matrix[vs_row, vs_row] -= 1j * omega * component.inductance

                # Move to next voltage source
                voltage_source_index += 1

    def __repr__(self) -> str:
        """String representation of the solver."""
        return f"ACSolver(circuit={self.circuit})"
