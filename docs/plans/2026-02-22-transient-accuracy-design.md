# Transient Analysis Accuracy Improvement — Design

**Date:** 2026-02-22
**Goal:** Reduce transient analysis NRMSE from 2.1% (post-settling) / 14.2% (full-range) to as low as practical

## Problem Analysis

### Current Baseline (vs NGSPICE)
| Metric | Value |
|--------|-------|
| Post-settling NRMSE (t >= 0.3ns) | 2.10% |
| Full-range NRMSE | 14.24% |
| Max absolute error | 94.4 mV (13.5% of Vdd) |
| Startup exclusion window | 0.3 ns |

### Error Sources

**1. Startup Artifact (dominates full-range NRMSE)**
- Pseudo-transient uses 1pF caps, 100x larger than 10fF load
- Gmin stepping uses 10 steps with linear decay (not aggressive enough)
- V(out) drops from 0.7V to ~0V during first 10 steps, recovers by ~0.15ns

**2. Missing Intrinsic MOSFET Capacitances (dominates transition errors)**
- PyCMG provides cgg, cgd, cgs, cdg, cdd but they are ignored
- At switching point: total Cgd+Cdd on output = 0.45 fF (4.5% of Cload)
- Cgd (Miller) creates gate-drain feedback — missing from simulation
- Error spikes up to 94mV occur precisely at switching edges

**3. Backward Euler Integration (systematic accuracy loss)**
- First-order method, NGSPICE uses Trapezoidal (second-order)
- Introduces numerical damping that slightly distorts transition shape
- Capacitor companion: G_eq = C/dt, I_eq = G_eq * V_prev

## Design: Three-Phase Fix

### Phase A: Fix Startup Artifact

**Changes to `TransientSolver.__init__()` defaults:**
- `pseudo_transient_cap`: 1e-12 → scale to ~5x the largest circuit capacitance
- `pseudo_transient_steps`: 10 → 5 (fewer steps needed with smaller caps)
- `gmin_initial`: 1e-8 → 1e-9 (less aggressive initial Gmin)
- `gmin_steps`: 10 → 5 (match pseudo-transient steps)

**Algorithm change in `_add_pseudo_capacitors()`:**
- Auto-detect max circuit capacitance, set pseudo_cap = 5x that value
- For 10fF load: pseudo_cap = 50fF (not 1pF = 1000fF)

**Expected impact:** Eliminate startup exclusion window, full-range NRMSE should approach post-settling value (~2%).

### Phase B: Add BSIM-CMG Intrinsic Capacitances

**New method in `mosfet_cmg.py`:**
```python
def get_capacitances(self, voltages: Dict[str, float]) -> Dict[str, float]:
    """Return intrinsic capacitances from BSIM-CMG eval_dc().
    Returns: {'cgg': ..., 'cgd': ..., 'cgs': ..., 'cdg': ..., 'cdd': ...}
    """
```

**New state tracking in `mosfet_cmg.py`:**
- Add `q_prev` dict (previous timestep charges: qg, qd, qs) to each MOSFET instance
- Add `update_charges()` method called after each timestep converges

**Changes to `solver.py` `_stamp_mosfet_transient()`:**
After existing conductance stamping, add capacitance stamping:
```python
# Get intrinsic capacitances (voltage-dependent)
caps = mosfet.get_capacitances(voltages)
# Stamp as Backward Euler companion models
# Cgd between gate and drain, Cgs between gate and source, etc.
```

The capacitance stamp follows the same pattern as the external Capacitor class:
- G_eq = C_ij / dt added to MNA matrix
- I_eq = G_eq * V_prev_across added to RHS

**Terminal capacitance mapping:**
- `cgd` → between gate (node[1]) and drain (node[0])
- `cgs` → between gate (node[1]) and source (node[2])
- `cdd` → between drain (node[0]) and source (node[2]) (drain junction)

**Charge state management:**
- Before first timestep: initialize `q_prev` from DC eval charges
- After each converged timestep: call `update_charges(voltages)`
- Charges needed for trapezoidal integration (Phase C)

**Expected impact:** Reduce max edge error from 94mV to ~50-60mV, NRMSE improvement of ~30-50%.

### Phase C: Trapezoidal Integration

**Changes to `Capacitor.get_companion_model()`:**
Replace Backward Euler with Trapezoidal Rule:
```python
# Backward Euler: G_eq = C/dt,     I_eq = G_eq * V_prev
# Trapezoidal:    G_eq = 2*C/dt,   I_eq = G_eq * V_prev + I_prev
```

Where I_prev = C * (V_current - V_prev) / dt from the previous step, or equivalently:
```python
G_eq = 2 * C / dt
I_eq = G_eq * V_prev + I_prev  # I_prev stored from previous step
```

**Changes needed:**
- `Capacitor` class: add `i_prev` state, update companion model formula
- `TransientSolver._stamp_mosfet_transient()`: apply same trapezoidal formula to intrinsic caps
- Both external and intrinsic capacitors use the same integration method

**Expected impact:** Further 20-40% NRMSE reduction due to second-order accuracy.

## Files Modified

| File | Changes |
|------|---------|
| `pycircuitsim/solver.py` | TransientSolver defaults, _stamp_mosfet_transient() caps, trapezoidal |
| `pycircuitsim/models/mosfet_cmg.py` | get_capacitances(), charge state tracking |
| `pycircuitsim/models/passive.py` | Capacitor trapezoidal companion model |
| `tests/verify_bsimcmg_tran.py` | Update thresholds, remove startup exclusion |

## Verification Strategy

After each phase, run `tests/verify_bsimcmg_tran.py` and measure:
1. Full-range NRMSE (no exclusion window)
2. Post-settling NRMSE (keep for comparison)
3. Max absolute error at transitions
4. Visual waveform comparison

**Success criteria:** Full-range NRMSE < 2%, or at least significant reduction from 14.2%.
