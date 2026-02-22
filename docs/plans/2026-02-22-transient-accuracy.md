# Transient Analysis Accuracy Improvement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce transient NRMSE from 2.1% post-settling / 14.2% full-range to < 1% by fixing startup artifact, adding BSIM-CMG intrinsic capacitances, and switching to trapezoidal integration.

**Architecture:** Three independent improvements applied sequentially: (A) reduce pseudo-transient capacitor size to eliminate startup artifact, (B) stamp BSIM-CMG intrinsic capacitances (cgg, cgd, cgs, cdd) as Backward Euler companion models in the MNA matrix, (C) upgrade both external and intrinsic capacitor integration from Backward Euler to Trapezoidal Rule. Each phase is verified against NGSPICE independently.

**Tech Stack:** Python 3, numpy, PyCMG (BSIM-CMG OSDI wrapper), NGSPICE 45.2

---

## Phase A: Fix Startup Artifact

### Task 1: Reduce pseudo-transient capacitor and Gmin stepping parameters

**Files:**
- Modify: `pycircuitsim/solver.py:895-922` (`_add_pseudo_capacitors()`)
- Modify: `tests/verify_bsimcmg_tran.py:233-241` (solver invocation parameters)

**Step 1: Modify `_add_pseudo_capacitors()` to auto-scale pseudo-cap**

In `pycircuitsim/solver.py`, replace the `_add_pseudo_capacitors()` method (line 895) with a version that auto-detects the maximum circuit capacitance and scales the pseudo-cap to 5x that value:

```python
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
```

**Step 2: Update test script to use reduced parameters**

In `tests/verify_bsimcmg_tran.py`, change the TransientSolver invocation (line 233-241):

```python
solver = TransientSolver(
    circuit, t_stop=final_time, dt=time_step,
    initial_guess=op_solution,
    use_gmin_stepping=True,
    gmin_initial=1e-9, gmin_final=1e-12, gmin_steps=5,
    use_pseudo_transient=True,
    pseudo_transient_steps=5, pseudo_transient_cap=1e-12,
    debug=False,
)
```

Key changes:
- `gmin_initial`: 1e-8 -> 1e-9 (10x less aggressive)
- `gmin_steps`: 10 -> 5
- `pseudo_transient_steps`: 10 -> 5
- `pseudo_transient_cap`: 1e-12 stays, but auto-scaling in solver will reduce to 5e-14 (5x 10fF)

**Step 3: Update startup exclusion constant**

In `tests/verify_bsimcmg_tran.py`, change line 75:

```python
STARTUP_EXCLUSION = 0.1e-9  # 0.1ns (reduced: pseudo-caps now auto-scaled)
```

**Step 4: Run verification**

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py
```

Expected: Full-range NRMSE should drop from 14.2% to < 5%. Post-settling NRMSE should remain ~2%.

**Step 5: Commit**

```bash
git add pycircuitsim/solver.py tests/verify_bsimcmg_tran.py
git commit -m "fix: auto-scale pseudo-transient caps to reduce startup artifact"
```

---

## Phase B: Add BSIM-CMG Intrinsic Capacitances

### Task 2: Add charge state tracking to BSIM-CMG device models

**Files:**
- Modify: `pycircuitsim/models/mosfet_cmg.py:39-289` (NMOS_CMG class)
- Modify: `pycircuitsim/models/mosfet_cmg.py:291-466` (PMOS_CMG class)

**Step 1: Add charge state and capacitance integration to NMOS_CMG**

Add these methods and attributes to `NMOS_CMG` class. After the `get_capacitances()` method (line 288), add:

```python
def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
    """Get terminal charges from BSIM-CMG eval_dc().

    Returns:
        Dictionary with keys: qg, qd, qs, qb (Coulombs)
    """
    result = self._eval_dc(voltages)
    return {
        "qg": result.get("qg", 0.0),
        "qd": result.get("qd", 0.0),
        "qs": result.get("qs", 0.0),
        "qb": result.get("qb", 0.0),
    }

def init_charge_state(self, voltages: Dict[str, float]) -> None:
    """Initialize charge state from DC operating point.
    Must be called before transient analysis starts.
    """
    charges = self.get_charges(voltages)
    self._q_prev = charges.copy()
    self._v_prev = {
        "d": voltages.get(self.nodes[0], 0.0),
        "g": voltages.get(self.nodes[1], 0.0),
        "s": voltages.get(self.nodes[2], 0.0),
        "b": voltages.get(self.nodes[3], 0.0),
    }

def update_charge_state(self, voltages: Dict[str, float]) -> None:
    """Update charge state after a converged timestep.
    Called after each timestep to store Q(n) for the next step.
    """
    charges = self.get_charges(voltages)
    self._q_prev = charges.copy()
    self._v_prev = {
        "d": voltages.get(self.nodes[0], 0.0),
        "g": voltages.get(self.nodes[1], 0.0),
        "s": voltages.get(self.nodes[2], 0.0),
        "b": voltages.get(self.nodes[3], 0.0),
    }
```

Also add initialization in `__init__` (after `self._eval_cache = None`):

```python
# Charge state for transient analysis
self._q_prev: Optional[Dict[str, float]] = None
self._v_prev: Optional[Dict[str, float]] = None
```

**Step 2: Add identical methods to PMOS_CMG**

Copy the same `get_charges()`, `init_charge_state()`, `update_charge_state()` methods and `_q_prev`/`_v_prev` attributes to the PMOS_CMG class (after its `get_capacitances()` at line 466).

**Step 3: Commit**

```bash
git add pycircuitsim/models/mosfet_cmg.py
git commit -m "feat: add charge state tracking to BSIM-CMG device models"
```

### Task 3: Stamp intrinsic capacitances in transient solver

**Files:**
- Modify: `pycircuitsim/solver.py:1168-1298` (`_stamp_mosfet_transient()`)
- Modify: `pycircuitsim/solver.py:1300-1500` (`solve()` main loop)

**Step 1: Add intrinsic capacitance stamping to `_stamp_mosfet_transient()`**

After the RHS stamping block in `_stamp_mosfet_transient()` (after line 1298), add this code to stamp intrinsic capacitances:

```python
# --- Intrinsic capacitance stamping (Backward Euler companion model) ---
# Only for BSIM-CMG devices that have capacitance data
try:
    from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
    is_cmg = isinstance(mosfet, (NMOS_CMG, PMOS_CMG))
except ImportError:
    is_cmg = False

if is_cmg and hasattr(mosfet, '_q_prev') and mosfet._q_prev is not None:
    caps = mosfet.get_capacitances(voltages)
    dt = self.dt

    # Cgd: between gate and drain
    cgd = abs(caps.get("cgd", 0.0))
    if cgd > 1e-20 and dt > 0:
        g_cgd = cgd / dt
        # V_prev across Cgd = V_g_prev - V_d_prev
        v_gd_prev = mosfet._v_prev["g"] - mosfet._v_prev["d"]
        i_cgd = g_cgd * v_gd_prev

        # Stamp g_cgd between gate and drain
        if gate != "0" and gate in node_map:
            g_idx = node_map[gate]
            mna_matrix[g_idx, g_idx] += g_cgd
        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            mna_matrix[d_idx, d_idx] += g_cgd
        if gate != "0" and gate in node_map and drain != "0" and drain in node_map:
            g_idx = node_map[gate]
            d_idx = node_map[drain]
            mna_matrix[g_idx, d_idx] -= g_cgd
            mna_matrix[d_idx, g_idx] -= g_cgd

        # Stamp i_cgd to RHS
        if gate != "0" and gate in node_map:
            g_idx = node_map[gate]
            rhs[g_idx] += i_cgd
        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            rhs[d_idx] -= i_cgd

    # Cgs: between gate and source
    cgs = abs(caps.get("cgs", 0.0))
    if cgs > 1e-20 and dt > 0:
        g_cgs = cgs / dt
        v_gs_prev = mosfet._v_prev["g"] - mosfet._v_prev["s"]
        i_cgs = g_cgs * v_gs_prev

        if gate != "0" and gate in node_map:
            g_idx = node_map[gate]
            mna_matrix[g_idx, g_idx] += g_cgs
        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_cgs
        if gate != "0" and gate in node_map and source != "0" and source in node_map:
            g_idx = node_map[gate]
            s_idx = node_map[source]
            mna_matrix[g_idx, s_idx] -= g_cgs
            mna_matrix[s_idx, g_idx] -= g_cgs

        if gate != "0" and gate in node_map:
            g_idx = node_map[gate]
            rhs[g_idx] += i_cgs
        if source != "0" and source in node_map:
            s_idx = node_map[source]
            rhs[s_idx] -= i_cgs

    # Cdd (drain junction): between drain and source
    cdd = abs(caps.get("cdd", 0.0))
    if cdd > 1e-20 and dt > 0:
        g_cdd = cdd / dt
        v_ds_prev = mosfet._v_prev["d"] - mosfet._v_prev["s"]
        i_cdd = g_cdd * v_ds_prev

        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            mna_matrix[d_idx, d_idx] += g_cdd
        if source != "0" and source in node_map:
            s_idx = node_map[source]
            mna_matrix[s_idx, s_idx] += g_cdd
        if drain != "0" and drain in node_map and source != "0" and source in node_map:
            d_idx = node_map[drain]
            s_idx = node_map[source]
            mna_matrix[d_idx, s_idx] -= g_cdd
            mna_matrix[s_idx, d_idx] -= g_cdd

        if drain != "0" and drain in node_map:
            d_idx = node_map[drain]
            rhs[d_idx] += i_cdd
        if source != "0" and source in node_map:
            s_idx = node_map[source]
            rhs[s_idx] -= i_cdd
```

**Step 2: Initialize charge state before transient loop**

In the `solve()` method (around line 1390, after initializing capacitor v_prev), add charge initialization:

```python
# Initialize MOSFET charge state for intrinsic capacitance tracking
for component in self.circuit.components:
    if _is_mosfet(component) and hasattr(component, 'init_charge_state'):
        component.init_charge_state(initial_voltages)
```

**Step 3: Update charge state after each converged timestep**

In the `solve()` method (around line 1475, after `component.update_voltage(timestep_voltages)`), add:

```python
# Update MOSFET charge state for next timestep
for component in self.circuit.components:
    if _is_mosfet(component) and hasattr(component, 'update_charge_state'):
        component.update_charge_state(timestep_voltages)
```

**Step 4: Run verification**

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py
```

Expected: Post-settling NRMSE should drop from ~2% to ~1-1.5%. Max edge error should decrease.

**Step 5: Commit**

```bash
git add pycircuitsim/solver.py pycircuitsim/models/mosfet_cmg.py
git commit -m "feat: stamp BSIM-CMG intrinsic capacitances in transient solver"
```

---

## Phase C: Trapezoidal Integration

### Task 4: Upgrade Capacitor companion model to Trapezoidal Rule

**Files:**
- Modify: `pycircuitsim/models/passive.py:586-775` (Capacitor class)

**Step 1: Add trapezoidal state to Capacitor class**

In `Capacitor.__init__()` (line 611), add after `self._i_eq = 0.0`:

```python
# Trapezoidal integration state
self._i_prev = 0.0  # Current through capacitor at previous timestep
self._use_trapezoidal = True  # Use trapezoidal (True) vs backward Euler (False)
```

**Step 2: Update `get_companion_model()` for trapezoidal integration**

Replace the `get_companion_model()` method (line 649-673) with:

```python
def get_companion_model(self, dt: float, v_prev: float) -> tuple[float, float]:
    """Calculate companion model parameters using Trapezoidal Rule.

    Trapezoidal Rule (2nd order, matches NGSPICE default):
        G_eq = 2*C/dt
        I_eq = G_eq * V_prev + I_prev

    Backward Euler (1st order, fallback):
        G_eq = C/dt
        I_eq = G_eq * V_prev

    Args:
        dt: Timestep size in seconds
        v_prev: Voltage across capacitor at previous timestep

    Returns:
        Tuple of (G_eq, I_eq)
    """
    if self._use_trapezoidal:
        g_eq = 2.0 * self.capacitance / dt
        i_eq = g_eq * v_prev + self._i_prev
    else:
        g_eq = self.capacitance / dt
        i_eq = g_eq * v_prev

    self._g_eq = g_eq
    self._i_eq = i_eq

    return g_eq, i_eq
```

**Step 3: Update `update_voltage()` to track current for trapezoidal**

Replace the `update_voltage()` method (line 736-753) with:

```python
def update_voltage(self, voltages: Dict[str, float]) -> None:
    """Update state after a timestep completes.

    For trapezoidal integration, also computes and stores the capacitor
    current for use in the next timestep's companion model.
    """
    node_i, node_j = self.nodes[0], self.nodes[1]
    v_i = voltages.get(node_i, 0.0)
    v_j = voltages.get(node_j, 0.0)
    v_new = v_i - v_j

    if self._use_trapezoidal:
        # I_cap = G_eq * (V_new - V_prev) - I_prev
        # equivalently: I_cap = 2*C/dt * (V_new - V_prev) - I_prev
        self._i_prev = self._g_eq * (v_new - self.v_prev) - self._i_prev

    self.v_prev = v_new
```

**Step 4: Run verification**

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py
```

Expected: Further NRMSE reduction from trapezoidal integration (second-order accuracy).

**Step 5: Commit**

```bash
git add pycircuitsim/models/passive.py
git commit -m "feat: upgrade capacitor integration from Backward Euler to Trapezoidal Rule"
```

### Task 5: Apply trapezoidal integration to intrinsic MOSFET capacitances

**Files:**
- Modify: `pycircuitsim/models/mosfet_cmg.py` (add `_i_prev_cap` state)
- Modify: `pycircuitsim/solver.py` (`_stamp_mosfet_transient()` intrinsic cap section)

**Step 1: Add trapezoidal current state to NMOS_CMG and PMOS_CMG**

In both `init_charge_state()` methods, add:

```python
# Trapezoidal integration state for intrinsic caps
self._i_prev_cgd = 0.0
self._i_prev_cgs = 0.0
self._i_prev_cdd = 0.0
```

In both `update_charge_state()` methods, receive and store the currents:

```python
def update_charge_state(self, voltages: Dict[str, float],
                        cap_currents: Optional[Dict[str, float]] = None) -> None:
    """Update charge state after a converged timestep."""
    charges = self.get_charges(voltages)
    self._q_prev = charges.copy()
    self._v_prev = {
        "d": voltages.get(self.nodes[0], 0.0),
        "g": voltages.get(self.nodes[1], 0.0),
        "s": voltages.get(self.nodes[2], 0.0),
        "b": voltages.get(self.nodes[3], 0.0),
    }
    if cap_currents is not None:
        self._i_prev_cgd = cap_currents.get("i_cgd", 0.0)
        self._i_prev_cgs = cap_currents.get("i_cgs", 0.0)
        self._i_prev_cdd = cap_currents.get("i_cdd", 0.0)
```

**Step 2: Update intrinsic cap stamping in `_stamp_mosfet_transient()` to use trapezoidal**

Change the intrinsic capacitance stamping code from Task 3 to use trapezoidal:

```python
# For each intrinsic cap (cgd, cgs, cdd):
# Trapezoidal: G_eq = 2*C/dt, I_eq = G_eq * V_prev + I_prev
g_cgd = 2.0 * cgd / dt
i_cgd = g_cgd * v_gd_prev + getattr(mosfet, '_i_prev_cgd', 0.0)
```

(Same pattern for cgs and cdd.)

**Step 3: Compute and pass cap currents after each converged timestep**

In `solve()`, after the timestep converges, compute intrinsic cap currents for the trapezoidal method:

```python
for component in self.circuit.components:
    if _is_mosfet(component) and hasattr(component, 'update_charge_state'):
        # Compute intrinsic cap currents for trapezoidal
        caps = component.get_capacitances(timestep_voltages) if hasattr(component, 'get_capacitances') else {}
        v_d = timestep_voltages.get(component.nodes[0], 0.0)
        v_g = timestep_voltages.get(component.nodes[1], 0.0)
        v_s = timestep_voltages.get(component.nodes[2], 0.0)
        v_prev = component._v_prev if component._v_prev else {"d": 0, "g": 0, "s": 0, "b": 0}
        dt = current_dt

        cap_currents = {}
        cgd = abs(caps.get("cgd", 0.0))
        if cgd > 1e-20:
            g_eq = 2.0 * cgd / dt
            v_gd_new = v_g - v_d
            v_gd_prev = v_prev["g"] - v_prev["d"]
            cap_currents["i_cgd"] = g_eq * (v_gd_new - v_gd_prev) - getattr(component, '_i_prev_cgd', 0.0)

        cgs = abs(caps.get("cgs", 0.0))
        if cgs > 1e-20:
            g_eq = 2.0 * cgs / dt
            v_gs_new = v_g - v_s
            v_gs_prev = v_prev["g"] - v_prev["s"]
            cap_currents["i_cgs"] = g_eq * (v_gs_new - v_gs_prev) - getattr(component, '_i_prev_cgs', 0.0)

        cdd = abs(caps.get("cdd", 0.0))
        if cdd > 1e-20:
            g_eq = 2.0 * cdd / dt
            v_ds_new = v_d - v_s
            v_ds_prev = v_prev["d"] - v_prev["s"]
            cap_currents["i_cdd"] = g_eq * (v_ds_new - v_ds_prev) - getattr(component, '_i_prev_cdd', 0.0)

        component.update_charge_state(timestep_voltages, cap_currents)
```

**Step 4: Run verification**

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py
```

Expected: NRMSE should be < 1% with all three improvements combined.

**Step 5: Commit**

```bash
git add pycircuitsim/solver.py pycircuitsim/models/mosfet_cmg.py
git commit -m "feat: trapezoidal integration for intrinsic MOSFET capacitances"
```

---

## Phase D: Finalize & Verify

### Task 6: Update verification thresholds and run all tests

**Files:**
- Modify: `tests/verify_bsimcmg_tran.py` (update thresholds, reduce exclusion)
- Modify: `CLAUDE.md` (update Phase 5/6 status, verification results)

**Step 1: Update verification script acceptance criteria**

In `tests/verify_bsimcmg_tran.py`, update based on achieved results:
- Reduce `NRMSE_THRESHOLD` if results are good
- Reduce `STARTUP_EXCLUSION` if startup artifact is fixed
- Update comments to reflect new implementation

**Step 2: Run full verification suite**

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_op.py
conda run -n pycircuitsim python tests/verify_bsimcmg_dc.py
conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py
```

All three must still pass. OP and DC results must be unchanged (< 0.1% error). Transient results should show improved NRMSE.

**Step 3: Update CLAUDE.md**

Update the verification results table and add notes about the improvements.

**Step 4: Final commit**

```bash
git add -A
git commit -m "docs: update verification results after transient accuracy improvement"
```
