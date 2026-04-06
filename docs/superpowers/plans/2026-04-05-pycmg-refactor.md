# PyCMG Test Harness & model.py Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~500 lines of duplicated test boilerplate and DRY the Schur complement math in `model.py`, while keeping all 280 existing tests green.

**Architecture:** Extract shared test infrastructure (bias points, DC comparison, skip markers) into `conftest.py` and `helpers.py`, then slim each test file to use the shared helpers. In `model.py`, extract a `_schur_condense()` static method used by both `_condense_capacitance()` and `get_jacobian_matrix()`.

**Tech Stack:** Python 3.10, pytest, numpy, PyCMG (ctypes OSDI wrapper)

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `pycmg/model.py:296-433` | Extract `_schur_condense()` from duplicated Schur complement logic |
| Modify | `tests/conftest.py` | Add `requires_osdi` marker, `standard_bias_points()`, re-export `OSDI_PATH` |
| Modify | `tests/helpers.py` | Add `run_dc_comparison()` unified helper |
| Modify | `tests/test_dc_regions.py` | Replace inline comparison with `run_dc_comparison()` |
| Modify | `tests/test_dc_jacobian.py` | Import bias points from conftest instead of local defs |
| Modify | `tests/test_body_bias.py` | Replace inline comparison with `run_dc_comparison()` |
| Modify | `tests/test_temperature.py` | Replace inline comparison with `run_dc_comparison()` |
| Modify | `tests/test_vt_variants.py` | Use shared `standard_bias_points()` + `run_dc_comparison()` |

All work is within `external_compact_models/PyCMG/`.

---

### Task 1: Extract `_schur_condense()` in model.py

**Files:**
- Modify: `pycmg/model.py:296-433`

The Schur complement math (extract sub-matrices from full NxN, solve G_ii, compute G_ee - G_ei @ G_ii^-1 @ G_ie) is duplicated between `_condense_capacitance()` (complex-valued, lines 296-333) and `get_jacobian_matrix()` (real-valued, lines 369-433). Extract a shared static method.

- [ ] **Step 1: Add `_schur_condense()` static method**

Insert after `_build_full_jacobian` (line 294), before `_condense_capacitance`:

```python
@staticmethod
def _schur_condense(full: np.ndarray,
                    external: List[int],
                    internal: List[int]) -> np.ndarray:
    """Schur complement condensation: reduce NxN matrix to external-only.

    Computes: M_ee - M_ei @ M_ii^{-1} @ M_ie

    Works for both real and complex matrices (capacitance uses complex
    Y = G + jωC; resistive Jacobian uses real G).

    Args:
        full: NxN matrix (real or complex)
        external: indices of external (terminal) nodes
        internal: indices of internal nodes

    Returns:
        ne×ne condensed matrix (same dtype as input), or None if
        the internal node matrix is singular (LinAlgError).
    """
    ne = len(external)
    ni = len(internal)
    dtype = full.dtype

    m_ee = np.zeros((ne, ne), dtype=dtype)
    for r in range(ne):
        for c in range(ne):
            m_ee[r, c] = full[external[r], external[c]]

    if ni == 0:
        return m_ee

    m_ei = np.zeros((ne, ni), dtype=dtype)
    m_ie = np.zeros((ni, ne), dtype=dtype)
    m_ii = np.zeros((ni, ni), dtype=dtype)
    for r in range(ne):
        for c in range(ni):
            m_ei[r, c] = full[external[r], internal[c]]
    for r in range(ni):
        for c in range(ne):
            m_ie[r, c] = full[internal[r], external[c]]
        for c in range(ni):
            m_ii[r, c] = full[internal[r], internal[c]]

    try:
        m_ie_sol = np.linalg.solve(m_ii, m_ie)
    except np.linalg.LinAlgError:
        # Return None to signal failure — callers handle fallback
        return None

    return m_ee - m_ei @ m_ie_sol
```

- [ ] **Step 2: Rewrite `_condense_capacitance()` to use `_schur_condense()`**

Replace the body of `_condense_capacitance` (lines 296-333) with:

```python
@staticmethod
def _condense_capacitance(g_full: np.ndarray,
                          c_full: np.ndarray,
                          external: List[int],
                          internal: List[int]) -> np.ndarray:
    ne = len(external)
    c_condensed = np.zeros((ne, ne), dtype=float)
    if ne == 0:
        return c_condensed
    # Build complex admittance Y = G + jωC (ω=1 for capacitance extraction)
    y_full = g_full.astype(complex) + 1j * c_full.astype(complex)
    y_condensed = Instance._schur_condense(y_full, external, internal)
    if y_condensed is None:
        return c_condensed  # zeros on singular internal matrix (matches old behavior)
    return np.imag(y_condensed).astype(float)
```

- [ ] **Step 3: Rewrite `get_jacobian_matrix()` condensation to use `_schur_condense()`**

Replace lines 397-433 in `get_jacobian_matrix` with:

```python
    # Build full NxN resistive Jacobian from OSDI
    g_full = self._build_full_jacobian(self._sim, self._sim.jacobian_resist)

    # Condense to external-only using Schur complement
    ext = self._sim.terminal_indices
    intn = self._sim.internal_indices
    g_condensed = self._schur_condense(g_full, ext, intn)

    if g_condensed is None:
        # Fallback: return external-only block negated (matches old behavior)
        ne = len(ext)
        g_ee = np.zeros((ne, ne))
        for r in range(ne):
            for c in range(ne):
                g_ee[r, c] = g_full[ext[r], ext[c]]
        return -g_ee

    # Negate: OSDI jacobian_resist stores dF/dV where F is KCL residual
    # (current into node). Terminal currents use I = -F, so dI/dV = -dF/dV.
    return -g_condensed
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/test_api.py tests/test_dc_jacobian.py tests/test_ac_caps.py -v --tb=short -x`

Expected: All PASS (these are the tests that exercise Jacobian and capacitance paths).

- [ ] **Step 5: Commit**

```bash
git add pycmg/model.py
git commit -m "refactor(model): extract _schur_condense() to DRY Schur complement math"
```

---

### Task 2: Add shared test infrastructure to conftest.py and helpers.py

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/helpers.py`

- [ ] **Step 1: Add `requires_osdi` marker to conftest.py**

Add at the end of conftest.py (after `pytest_runtest_makereport`):

```python
# Shared skip marker — replaces per-file @pytest.mark.skipif(not OSDI_PATH.exists(), ...)
requires_osdi = pytest.mark.skipif(
    not OSDI_PATH.exists(), reason="missing OSDI build artifact"
)
```

- [ ] **Step 2: Add `standard_bias_points()` to conftest.py**

Add after the `requires_osdi` marker. This replaces duplicate bias point generators in `test_dc_regions.py`, `test_dc_jacobian.py`, and `test_vt_variants.py`:

```python
def standard_bias_points(
    vdd: float, device_type: str, regions: str = "all",
) -> dict[str, dict[str, float]]:
    """Canonical bias points for DC verification tests.

    Args:
        vdd: Supply voltage
        device_type: "nmos" or "pmos"
        regions: "all" for 3 regions, or a specific region name

    Returns:
        Dict mapping region name -> terminal voltage dict {d, g, s, e}
    """
    if device_type == "nmos":
        points = {
            "off":        {"d": vdd,       "g": 0.0,       "s": 0.0, "e": 0.0},
            "linear":     {"d": 0.3 * vdd, "g": vdd,       "s": 0.0, "e": 0.0},
            "saturation": {"d": vdd,       "g": 0.8 * vdd, "s": 0.0, "e": 0.0},
        }
    else:
        # ve=0 exercises deep reverse body bias (Vbs = -Vdd)
        points = {
            "off":        {"d": 0.0,       "g": vdd,       "s": vdd, "e": 0.0},
            "linear":     {"d": 0.7 * vdd, "g": 0.0,       "s": vdd, "e": 0.0},
            "saturation": {"d": 0.0,       "g": 0.2 * vdd, "s": vdd, "e": 0.0},
        }
    if regions == "all":
        return points
    return {regions: points[regions]}


REGION_NAMES = ["off", "linear", "saturation"]
```

- [ ] **Step 3: Add `run_dc_comparison()` to helpers.py**

Add at the end of `helpers.py`. This is the unified DC comparison helper that replaces ~485 lines across 5 test files:

```python
def run_dc_comparison(
    tech_name: str,
    device_type: str,
    bias: Dict[str, float],
    tag: str,
    *,
    outputs: Optional[List[str]] = None,
    temp_c: float = 27.0,
    temp_k: Optional[float] = None,
    check_off_state: bool = False,
    check_ids: bool = False,
    rel_tol: float = REL_TOL,
    abs_tol_override: Optional[Dict[str, float]] = None,
) -> None:
    """Run PyCMG eval_dc vs NGSPICE OP comparison for a single bias point.

    Replaces the repeated pattern: get_tech_modelcard → run_ngspice_op →
    Model → Instance → eval_dc → assert_close for each output.

    Args:
        tech_name: Key from conftest ALL_TECHNOLOGIES
        device_type: "nmos" or "pmos"
        bias: Terminal voltages {"d": ..., "g": ..., "s": ..., "e": ...}
        tag: Unique tag for NGSPICE output directory
        outputs: List of output keys to compare (default: id, gm, gds, gmb, qg, qd, qs, qb)
        temp_c: Temperature in Celsius (for NGSPICE)
        temp_k: Temperature in Kelvin (for PyCMG). If None, derived from temp_c.
        check_off_state: If True, use relaxed leakage-ratio comparison for id
        check_ids: If True, also verify ids = id - is consistency
        rel_tol: Relative tolerance override
        abs_tol_override: Per-output absolute tolerance overrides {key: tol}
    """
    # Avoid circular import: conftest imports from helpers, so import lazily
    from tests.conftest import ALL_TECHNOLOGIES, get_tech_modelcard

    tech = ALL_TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, device_type)
    vdd = tech["vdd"]

    if temp_k is None:
        temp_k = temp_c + 273.15

    if outputs is None:
        outputs = ["id", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        bias["d"], bias["g"], bias["s"], bias["e"],
        temp_c=temp_c, tag=tag,
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    kwargs: Dict[str, Any] = {"params": inst_params}
    if abs(temp_k - 300.15) > 0.01:
        kwargs["temperature"] = temp_k
    inst = Instance(model, **kwargs)
    py = inst.eval_dc(bias)

    prefix = f"{tech_name}/{device_type}/{tag}"

    for key in outputs:
        kw: Dict[str, Any] = {"rel_tol": rel_tol}
        if abs_tol_override and key in abs_tol_override:
            kw["abs_tol"] = abs_tol_override[key]
        assert_close(f"{prefix}/{key}", py[key], ng[key], **kw)

    if check_off_state:
        _leakage_floor = 1e-12
        if abs(ng["id"]) > _leakage_floor and abs(py["id"]) > _leakage_floor:
            ratio = abs(py["id"] / ng["id"])
            assert 0.1 < ratio < 10.0, (
                f"{prefix}: PyCMG/NGSPICE off-state id ratio {ratio:.2f} outside [0.1, 10.0]"
            )

    if check_ids:
        ids_from_components = py["id"] - py["is"]
        assert abs(py["ids"] - ids_from_components) < 1e-15, (
            f"{prefix}: ids ({py['ids']:.3e}) != id - is ({ids_from_components:.3e})"
        )
        ng_ids = ng["id"] - ng["is"]
        assert_close(f"{prefix}/ids", py["ids"], ng_ids, rel_tol=rel_tol)
```

Also add the import at the top of `helpers.py`:

```python
from pycmg import Model, Instance
```

- [ ] **Step 4: Run a quick smoke test to verify imports resolve**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -c "from tests.helpers import run_dc_comparison; print('OK')"`

Expected: `OK` (no import errors)

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/helpers.py
git commit -m "refactor(tests): add shared requires_osdi marker, standard_bias_points, run_dc_comparison"
```

---

### Task 3: Refactor test_dc_regions.py

**Files:**
- Modify: `tests/test_dc_regions.py`

Replace 168 lines (two near-identical test functions + duplicate bias point generators) with ~40 lines using shared helpers.

- [ ] **Step 1: Rewrite test_dc_regions.py**

```python
"""
DC Operating Region Tests

Verifies model accuracy across voltage-ratio-defined operating regions
for both NMOS and PMOS devices across all 5 technologies.

Run: pytest tests/test_dc_regions.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import (
    TECH_NAMES, requires_osdi, standard_bias_points, REGION_NAMES,
)


DC_OUTPUTS = ["id", "ig", "is", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]


@requires_osdi
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("region", REGION_NAMES)
def test_dc_region(tech_name: str, device: str, region: str) -> None:
    """Test DC currents and derivatives match NGSPICE in operating region."""
    from tests.conftest import TECHNOLOGIES
    vdd = TECHNOLOGIES[tech_name]["vdd"]
    bias = standard_bias_points(vdd, device)[region]

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"region_{tech_name}_{device}_{region}",
            outputs=DC_OUTPUTS,
            check_off_state=(region == "off"),
            check_ids=(region != "off"),
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run tests to verify**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/test_dc_regions.py -v --tb=short -x`

Expected: 30 tests PASS (5 techs × 2 devices × 3 regions). Same test count as before.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dc_regions.py
git commit -m "refactor(tests): slim test_dc_regions.py using shared helpers"
```

---

### Task 4: Refactor test_body_bias.py

**Files:**
- Modify: `tests/test_body_bias.py`

Replace two near-identical NMOS/PMOS functions with one parametrized function.

- [ ] **Step 1: Rewrite test_body_bias.py**

```python
"""
Body Bias Verification Tests

Verifies model accuracy with non-zero body bias (bulk terminal voltage)
for both NMOS and PMOS devices across all 5 technologies.

Run: pytest tests/test_body_bias.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import TECHNOLOGIES, TECH_NAMES, requires_osdi

BIAS_TYPES = ["reverse", "forward"]
BODY_OUTPUTS = ["id", "gm", "gds", "gmb", "ie", "qg", "qd"]


def _body_bias(vdd: float, device: str, bias_type: str) -> dict[str, float]:
    """Build bias dict for body-bias test."""
    if device == "nmos":
        ve = -0.1 if bias_type == "reverse" else 0.1
        return {"d": vdd / 2, "g": vdd / 2, "s": 0.0, "e": ve}
    else:
        ve = vdd + 0.1 if bias_type == "reverse" else vdd - 0.1
        return {"d": vdd * 0.3, "g": vdd * 0.3, "s": vdd, "e": ve}


@requires_osdi
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("bias_type", BIAS_TYPES)
def test_body_bias(tech_name: str, device: str, bias_type: str) -> None:
    """Test DC outputs match NGSPICE with non-zero body bias."""
    vdd = TECHNOLOGIES[tech_name]["vdd"]
    bias = _body_bias(vdd, device, bias_type)

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"body_bias_{tech_name}_{device}_{bias_type}",
            outputs=BODY_OUTPUTS,
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run tests to verify**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/test_body_bias.py -v --tb=short -x`

Expected: 20 tests PASS (5 techs × 2 devices × 2 bias types). Same count as before.

- [ ] **Step 3: Commit**

```bash
git add tests/test_body_bias.py
git commit -m "refactor(tests): slim test_body_bias.py using shared helpers"
```

---

### Task 5: Refactor test_vt_variants.py

**Files:**
- Modify: `tests/test_vt_variants.py`

Remove local `_nmos_ops`/`_pmos_ops` duplicates and `_compare_dc` helper. Use shared `standard_bias_points()` and `run_dc_comparison()`.

- [ ] **Step 1: Rewrite test_vt_variants.py**

```python
"""
Core-Voltage Vt Variant Verification Tests

Verifies PyCMG vs NGSPICE agreement for all core-voltage threshold voltage
variants across technologies.

Run: pytest tests/test_vt_variants.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import (
    ALL_TECHNOLOGIES, CORE_VT_NAMES, requires_osdi,
    standard_bias_points, REGION_NAMES,
)

VT_OUTPUTS = ["id", "ig", "is", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]


@requires_osdi
@pytest.mark.parametrize("tech_name", CORE_VT_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("region", REGION_NAMES)
def test_vt_variant(tech_name: str, device: str, region: str) -> None:
    """DC verification for core-voltage Vt variants."""
    vdd = ALL_TECHNOLOGIES[tech_name]["vdd"]
    bias = standard_bias_points(vdd, device)[region]

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"vt_{tech_name}_{device}_{region}",
            outputs=VT_OUTPUTS,
            check_off_state=(region == "off"),
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run tests to verify**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/test_vt_variants.py -v --tb=short -x`

Expected: 96 tests PASS (16 variants × 2 devices × 3 regions). Same count as before.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vt_variants.py
git commit -m "refactor(tests): slim test_vt_variants.py using shared helpers"
```

---

### Task 6: Refactor test_temperature.py

**Files:**
- Modify: `tests/test_temperature.py`

Unify the two separate `test_temperature` and `test_temperature_tsmc` functions into one parametrized function using `run_dc_comparison()`.

- [ ] **Step 1: Rewrite test_temperature.py**

```python
"""
Temperature Verification Tests

Verifies PyCMG temperature handling against NGSPICE ground truth.

Run: pytest tests/test_temperature.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import TECHNOLOGIES, requires_osdi

TEMP_OUTPUTS = ["id", "gm", "gds"]

# ASAP7 covers wide temperature range; TSMC7 skips -40C (convergence issues)
TEMP_CASES = [
    ("ASAP7", -40.0),
    ("ASAP7", 85.0),
    ("ASAP7", 125.0),
    ("TSMC7", 85.0),
    ("TSMC7", 125.0),
]


def _temp_bias(vdd: float, device: str) -> dict[str, float]:
    """Saturation bias point for temperature tests."""
    if device == "nmos":
        return {"d": vdd / 2, "g": vdd / 2, "s": 0.0, "e": 0.0}
    else:
        return {"d": vdd * 0.3, "g": vdd * 0.3, "s": vdd, "e": vdd}


@requires_osdi
@pytest.mark.parametrize("tech_name,temp_c", TEMP_CASES,
                         ids=[f"{t[0]}_T{t[1]}" for t in TEMP_CASES])
@pytest.mark.parametrize("device", ["nmos", "pmos"])
def test_temperature(tech_name: str, temp_c: float, device: str) -> None:
    """Test DC currents and derivatives match NGSPICE at non-default temperatures."""
    vdd = TECHNOLOGIES[tech_name]["vdd"]
    bias = _temp_bias(vdd, device)

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"temp_{tech_name}_{device}_{temp_c}",
            outputs=TEMP_OUTPUTS,
            temp_c=temp_c,
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run tests to verify**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/test_temperature.py -v --tb=short -x`

Expected: 10 tests PASS (5 temp cases × 2 devices). Same count as before.

- [ ] **Step 3: Commit**

```bash
git add tests/test_temperature.py
git commit -m "refactor(tests): slim test_temperature.py using shared helpers"
```

---

### Task 7: Refactor test_dc_jacobian.py

**Files:**
- Modify: `tests/test_dc_jacobian.py`

Replace local `get_nmos_jacobian_op_points`/`get_pmos_jacobian_op_points` with `standard_bias_points()`. Merge NMOS/PMOS into one parametrized test. Keep `compute_numerical_jacobian_central` (it's unique to this file).

- [ ] **Step 1: Rewrite test_dc_jacobian.py**

```python
"""
DC Jacobian Verification Tests

Compares PyCMG's condensed 4x4 analytical Jacobian against NGSPICE's
numerical Jacobian computed via central finite-difference perturbation.

Run: pytest tests/test_dc_jacobian.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from pycmg import Model, Instance
from tests.helpers import (
    OSDI_PATH, run_ngspice_op, assert_close,
    ABS_TOL_G, REL_TOL_JAC,
)
from tests.conftest import (
    TECHNOLOGIES, TECH_NAMES, get_tech_modelcard,
    requires_osdi, standard_bias_points, REGION_NAMES,
)


def compute_numerical_jacobian_central(
    modelcard: Path, model_name: str, inst_params: Dict[str, float],
    op: Dict[str, float], delta: float = 1e-6, temp_c: float = 27.0,
    tag_prefix: str = "jac",
) -> np.ndarray:
    """Compute 4x4 Jacobian via central finite-difference perturbation.

    Uses central differencing for O(delta^2) accuracy:
        J[:,j] = (I(V+delta_j) - I(V-delta_j)) / (2*delta)
    """
    op_keys = ["d", "g", "s", "e"]
    current_keys = ["id", "ig", "is", "ie"]
    n = 4
    J = np.zeros((n, n))

    for j, op_key in enumerate(op_keys):
        fwd_op = dict(op)
        fwd_op[op_key] = op[op_key] + delta
        fwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            fwd_op["d"], fwd_op["g"], fwd_op["s"], fwd_op["e"],
            temp_c, tag=f"{tag_prefix}_fwd_{op_key}",
        )
        fwd_I = np.array([fwd[k] for k in current_keys])

        bwd_op = dict(op)
        bwd_op[op_key] = op[op_key] - delta
        bwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            bwd_op["d"], bwd_op["g"], bwd_op["s"], bwd_op["e"],
            temp_c, tag=f"{tag_prefix}_bwd_{op_key}",
        )
        bwd_I = np.array([bwd[k] for k in current_keys])

        J[:, j] = (fwd_I - bwd_I) / (2.0 * delta)

    return J


@requires_osdi
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("region", REGION_NAMES)
def test_dc_jacobian_full_matrix(tech_name: str, device: str, region: str) -> None:
    """Compare condensed 4x4 Jacobian matrix against NGSPICE numerical Jacobian."""
    tech = TECHNOLOGIES[tech_name]

    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, device)
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")

    vdd = tech["vdd"]
    op = standard_bias_points(vdd, device)[region]

    # NGSPICE: numerical Jacobian via central differencing
    ng_J = compute_numerical_jacobian_central(
        modelcard, model_name, inst_params, op,
        tag_prefix=f"jac_{tech_name}_{device}_{region}",
    )

    # PyCMG: analytical condensed Jacobian
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_J = inst.get_jacobian_matrix(op)

    # Compare each entry
    terminals = ["d", "g", "s", "e"]
    for i, term_i in enumerate(terminals):
        for j, term_j in enumerate(terminals):
            label = f"{tech_name}/{device}/{region}/d(I{term_i})/d(V{term_j})"
            assert_close(
                label, py_J[i, j], ng_J[i, j],
                abs_tol=ABS_TOL_G, rel_tol=REL_TOL_JAC,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: Run tests to verify**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/test_dc_jacobian.py -v --tb=short -x`

Expected: 30 tests PASS (5 techs × 2 devices × 3 regions). Same count as before.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dc_jacobian.py
git commit -m "refactor(tests): slim test_dc_jacobian.py using shared helpers"
```

---

### Task 8: Full test suite regression check

**Files:** None (verification only)

- [ ] **Step 1: Run the complete PyCMG test suite**

Run: `cd external_compact_models/PyCMG && conda run -n pycircuitsim python -m pytest tests/ -v --tb=short 2>&1 | tail -30`

Expected: All ~280 tests PASS, 0 FAIL, 0 ERROR.

- [ ] **Step 2: Verify line count reduction**

Run: `cd external_compact_models/PyCMG && wc -l tests/test_dc_regions.py tests/test_body_bias.py tests/test_vt_variants.py tests/test_temperature.py tests/test_dc_jacobian.py`

Expected: Total lines should be roughly 250 (down from ~770 before refactor, ~67% reduction in these 5 files).

- [ ] **Step 3: Final commit if any cleanup needed**

Only if adjustments were needed during the regression run.

---

## Summary of Changes

| File | Before | After | Delta |
|------|--------|-------|-------|
| `pycmg/model.py` | 2 separate Schur impls (~130 lines) | 1 shared `_schur_condense()` + 2 thin callers (~80 lines) | -50 |
| `tests/conftest.py` | Skip markers in each file | Shared `requires_osdi` + `standard_bias_points()` | +30 |
| `tests/helpers.py` | No comparison helper | `run_dc_comparison()` | +50 |
| `tests/test_dc_regions.py` | 168 lines, 2 functions | ~40 lines, 1 parametrized function | -128 |
| `tests/test_body_bias.py` | 126 lines, 2 functions | ~45 lines, 1 parametrized function | -81 |
| `tests/test_vt_variants.py` | 133 lines, 2 functions + duplicated ops | ~40 lines, 1 parametrized function | -93 |
| `tests/test_temperature.py` | 106 lines, 2 functions | ~50 lines, 1 parametrized function | -56 |
| `tests/test_dc_jacobian.py` | 176 lines, 2 functions + duplicated ops | ~90 lines, 1 parametrized function | -86 |
| **Net** | | | **~-414 lines** |

All 280 tests remain green. Zero behavioral changes.
