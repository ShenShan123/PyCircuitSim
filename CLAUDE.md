# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture. 
**Primary Goal:** specific support for **Level-1 MOS models** and **PyCMG-wrapped CMG models** (LEVEL=72). 
The simulator must support **Operating Point (OP)**, **DC Sweep**, and **Transient Analysis** for both model types.

**Core Principles:**
* Pure Python with clean, readable code
* Complete decoupling: Solver ↔ Device Models
* Production-grade compact model integration via PyCMG/OSDI
* Basic HSPICE netlist compatibility

## Architecture

### Module Structure
```
pycircuitsim/
├── __init__.py         # Package initialization, exports public API
├── config.py           # Path configuration (OSDI binary, modelcards)
├── simulation.py       # Simulation orchestration (run_simulation, run_dc_sweep, run_transient)
├── parser.py           # Two-pass netlist parsing, .model directive support
├── circuit.py          # Circuit topology management
├── solver.py           # MNA matrix construction, Newton-Raphson solver
├── logger.py           # HSPICE-like .lis output
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── __init__.py
    ├── base.py         # Component abstract base class
    ├── passive.py      # R, C, V, I sources (including PULSE)
    ├── mosfet.py       # Level 1 Shichman-Hodges model
    └── mosfet_cmg.py   # BSIM-CMG FinFET model (LEVEL=72) via PyCMG

models/
├── PyCMG/              # BSIM-CMG OSDI wrapper (git submodule)
│   ├── pycmg/          # Python ctypes-based OSDI interface (Model, Instance)
│   ├── build-deep-verify/osdi/bsimcmg.osdi  # Compiled OSDI binary
│   └── tech_model_cards/ASAP7/              # ASAP7 7nm modelcards
main.py                 # CLI entry point (single main entrance)
examples/*.sp           # Example netlists
results/                # Simulation output (.lis, .csv, .png)
tests/                  # Validation scripts & NGSPICE comparison
```

### Key Algorithms
* **MNA (Modified Nodal Analysis)** - Circuit equation matrix construction
* **Newton-Raphson** - Non-linear circuit solver
* **Backward Euler** - Capacitor integration for transient analysis
* **Source Stepping** - Two-stage analysis for improved convergence

## Supported Features

### Devices
* Passive: R, C
* Active:
  - NMOS/PMOS Level 1 (Shichman-Hodges)
  - NMOS/PMOS Level 72 (BSIM-CMG FinFET via PyCMG)
* Sources: DC voltage/current, PULSE

### Analysis
* `.op` - Operating Point Analysis (Basic DC solution)
* `.dc` - DC Sweep Analysis
* `.tran` - Transient Analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=1 or LEVEL=72)
* `.include` - External library files
* `.ic` - Initial conditions (critical for SRAM/bistable circuits)

## Validation Strategy

**Mandatory Requirement:**
* **Test Case:** An inverter circuit must be used to verify functionality.
* **Analysis:** The inverter must successfully pass **Transient Analysis**.
* **Ground Truth:** All simulation results must be verified against **NGSPICE**.
* **Metric:** Waveforms and operating points must match NGSPICE within reasonable numerical tolerance.

## Status & Roadmap

### Phase 1: Core Implementation ✅ Complete
- [x] MNA matrix construction
- [x] Level 1 MOSFET model (Shichman-Hodges)
- [x] Newton-Raphson solver
- [x] Transient analysis with capacitors

### Phase 2: Enhancements ✅ Complete
- [x] HSPICE-like logging (.lis files)
- [x] Voltage clamping for numerical stability
- [x] Two-stage DC analysis
- [x] Enhanced visualization

### Phase 3: Production Ready ✅ Complete
- [x] Comprehensive numerical validation
- [x] Clean project structure
- [x] Documentation and examples

### Phase 4: BSIM-CMG Integration ✅ Complete (2026-02-09)
- [x] PyCMG OSDI wrapper integration
- [x] BSIM-CMG (LEVEL=72) parser support
- [x] 3-conductance model (gds, gm, gmb) in solver
- [x] ASAP7 7nm modelcard compatibility
- [x] **Critical bug fix: PMOS RHS stamping**
- [x] DC analysis validation (NMOS, PMOS, Inverter)

### Phase 5: Advanced Verification & Robustness
- [x] **Parser Enhancement** (2026-02-12)
    - Added femto (f) and other unit suffixes (T, G, M, m) to UNIT_SUFFIXES
- [x] **Transient Solver Improvements** (2026-02-12)
    - Fixed damping threshold from >=0.1V to >=1.0V (match DC solver)
    - Changed damping condition from > to >= (catch exact 1.0V case)
    - Made debug logging conditional (debug flag in TransientSolver.__init__)
- [x] **PMOS Conductance Fix** (2026-02-12)
    - Fixed PMOS default KP from 20e-6 to -20e-6 (SPICE convention)
    - Fixed abs() usage: abs(K) * abs(v) instead of abs(K * v)
- [x] **Level-1 Capacitance Model** (2026-02-12)
    - Added get_capacitances() method to NMOS/PMOS classes
    - Implements Meyer capacitance model (Cgs, Cgd, Cdb, Csb)
    - Note: Integration into transient solver requires state management (deferred)
- [x] **Transient Stability Improvements** (2026-02-12)
    - [x] Gmin stepping algorithm (exponential decay from 1e-8 to 1e-12)
    - [x] Pseudo-transient initialization (artificial capacitances for first N steps)
    - [x] Adaptive damping with oscillation detection

### Phase 6: PyCMG Update & NGSPICE Verification ✅ Complete (2026-02-21)
- [x] **PyCMG updated to latest** (34 commits: multi-tech, Jacobian, DEVTYPE injection)
- [x] **Path references fixed** (PyCMG → models/PyCMG, ASAP7 dir updated)
- [x] **ASAP7 modelcard name mapping** (parser auto-maps to nmos_rvt/pmos_rvt)
- [x] **calculate_current() bug fix** (use terminal `id` not channel `ids`)
- [x] **RHS stamping unified** (single "current leaving drain" convention)
- [x] **OP verification** — NMOS, PMOS, Inverter vs NGSPICE (all < 0.02% error)
- [x] **DC sweep verification** — Id-Vgs, VTC vs NGSPICE (all < 0.1% NRMSE)
- [x] **Transient verification** — Inverter vs NGSPICE (2.1% NRMSE post-settling)

### Phase 7: Transient Accuracy Improvement ✅ Complete (2026-02-22)
- [x] **Auto-scaled pseudo-caps** — pseudo-cap reduced from 1pF to 5x max circuit cap (50fF for 10fF load)
- [x] **Reduced Gmin stepping** — gmin_initial 1e-8→1e-9, steps 10→5, startup exclusion 0.3ns→0.1ns
- [x] **BSIM-CMG intrinsic capacitances** — Cgd, Cgs, Cdd stamped as companion models in MNA matrix
- [x] **Trapezoidal integration** — Upgraded from Backward Euler (1st order) to Trapezoidal Rule (2nd order)
- [x] **Charge state tracking** — get_charges(), init_charge_state(), update_charge_state() in mosfet_cmg.py
- [x] **Charge state tracking** — get_charges(), init_charge_state(), update_charge_state() in mosfet_cmg.py
- [x] **Skip convergence aids with DC OP** — pseudo-transient and Gmin stepping skipped when valid DC OP is provided
- [x] **Results:** NRMSE 2.1% → **0.23%** (post-settling), full-range 14.2% → **0.29%**, max error 94mV → **9.9mV**

### Phase 8: Comprehensive Transient Test Suite ✅ Complete (2026-02-22)
- [x] **Parametric test framework** — TestConfig dataclass with adaptive timing, 14 unique configurations
- [x] **4-sweep coverage** — VDD (0.5-0.8V), Cload (1-100fF), slew (10-500ps), pulse width (0.2-2.0ns)
- [x] **Automated NGSPICE comparison** — Per-config netlist generation, wrdata parsing, interpolation
- [x] **Summary reporting** — Formatted table, CSV export, color-coded bar chart by sweep type
- [x] **Results:** All 14 configs PASS (NRMSE < 5%), worst case 0.95% (Cload=1fF), best 0.03% (Cload=100fF)

### Future Work
- [ ] **Expanded Test Suite**
    - [ ] NAND/NOR gates
    - [ ] Ring Oscillator (multi-stage transient)
    - [ ] SRAM bitcell (static noise margin)
- [ ] **Adaptive Timestep** — Use local truncation error estimate for automatic timestep control

---

## Quick Start

### Basic Simulation
Create a netlist (`.sp` file) with your circuit. Examples provided in `examples/` directory.

**BSIM-CMG Geometric Parameters:**
- `L` - Channel length (required, in meters e.g., 30n)
- `NFIN` - Number of fins (required, integer or float)
- `TFIN` - Fin thickness (optional, uses modelcard default if omitted)
- `HFIN` - Fin height (optional, uses modelcard default)
- `FPITCH` - Fin pitch (optional, uses modelcard default)

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` / `*_transient.csv` - Waveform data (node voltages + device currents)

## Development Guidelines

### Coding Standards
- Type hints required for all function signatures
- Clear variable names (e.g., `v_gate`, `i_drain`, not `a`, `b`)
- Docstrings for complex algorithms
- Voltage clamping: Vgs ± 5V, Vds ± 10V

### Separation Principle
- **Solver** (`solver.py`) builds MNA matrix, executes Newton-Raphson (no device equations)
- **Device Models** (`models/`) calculate current/conductances from voltages (no matrix operations)
- **Simulation** (`simulation.py`) orchestrates the workflow (parse → solve → visualize)
- All devices inherit from `Component` base class

### Key Numerical Techniques
- Minimum conductance (1µS) prevents singular matrices
- Source stepping (20 steps) improves convergence
- Damping factor (0.5) for large voltage deltas
- Two-stage analysis: DC OP → DC sweep/transient
- Voltage-source-constrained nodes exempt from damping

### Entry Points
- **CLI**: `main.py` - Command-line interface (argparse, error handling)
- **API**: `pycircuitsim.simulation.run_simulation()` - Programmatic access
- **Module**: `pycircuitsim` - Package exports (Circuit, Parser, Visualizer, run_simulation)


## Environment & Tools
* **Conda Environment**: `pycircuitsim` in `/home/shenshan/.conda/envs/pycircuitsim`
* **OpenVAF Compiler:** `/usr/local/bin/openvaf`
* **NGSPICE Simulator:** `/usr/local/ngspice-45.2/bin/ngspice`
* **Build System:** CMake / Make
* **Python Bindings:** PyBind11

## Critical Bugs and Solutions

### CRITICAL: PMOS RHS Stamping & Current Convention (FIXED: 2026-02-09, REVISED: 2026-02-21)

**Severity**: Critical - PMOS circuits produced completely wrong results
**Affected**: Both Level 1 and BSIM-CMG PMOS devices
**Files**: `pycircuitsim/solver.py` `_stamp_mosfet()` and `_stamp_mosfet_transient()`

**Root Cause (original 2026-02-09):**
The solver applied the SAME RHS stamping formula to both NMOS and PMOS.

**Revised Fix (2026-02-21):**
The original fix used separate NMOS/PMOS code paths (`rhs[d] -= i_eq` vs `rhs[d] += i_eq`),
but this was fragile and broke when `calculate_current()` was also fixed. The correct approach
is to unify around the "current leaving drain" convention:

```python
# Convert to "current leaving drain" for MNA
# NMOS calculate_current: positive = current leaving drain (D→S) — already correct
# PMOS calculate_current: positive = current INTO drain — negate for MNA
i_leaving = -i_ds if is_pmos else i_ds

# Newton-Raphson constant (unified for both types)
i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs

# Stamp uniformly (no NMOS/PMOS branch needed)
rhs[d_idx] -= i_eq
rhs[s_idx] += i_eq
```

**Prevention:**
- Never have separate NMOS/PMOS RHS stamping branches — unify via sign convention
- `calculate_current()` and `_stamp_mosfet()` signs must be consistent
- Test: NMOS pulls output DOWN, PMOS pulls output UP, inverter switches correctly

---

### Device Recognition Bug: _is_mosfet() Helper (FIXED: 2026-02-09)

**Severity**: High - BSIM-CMG devices not recognized as MOSFETs
**Affected**: BSIM-CMG NMOS/PMOS (Level 72)
**Files**: `pycircuitsim/solver.py` line 36-44

**Root Cause:**
The `_is_mosfet()` helper only checked for Level 1 types (`NMOS`, `PMOS`), causing BSIM-CMG devices to be treated as linear components (no Newton-Raphson linearization).

**Symptom:**
- Inverter circuit: "Circuit is singular or unsolvable"
- No MNA stamping for BSIM-CMG devices
- Only affects circuits with LEVEL=72 devices

**Solution:**
```python
def _is_mosfet(component):
    """Check if component is a MOSFET (Level 1 or BSIM-CMG)."""
    from pycircuitsim.models.mosfet import NMOS, PMOS
    try:
        from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
        return isinstance(component, (NMOS, PMOS, NMOS_CMG, PMOS_CMG))
    except ImportError:
        return isinstance(component, (NMOS, PMOS))
```

**Prevention:**
- When adding new device types, update ALL device-type checking helpers
- Use try/except for optional imports (e.g., BSIM-CMG may not be installed)
- Grep for existing device type checks: `grep -r "isinstance.*NMOS\|PMOS" pycircuitsim/`

---

### Negative Conductance Stability Issue (FIXED: 2026-02-09)

**Severity**: Medium - Newton-Raphson divergence at extreme voltages
**Affected**: BSIM-CMG devices (can return negative gds)
**Files**: `pycircuitsim/models/mosfet_cmg.py` lines 258, 431

**Root Cause:**
BSIM-CMG model can return negative `gds` (negative differential resistance) at extreme operating points, causing Newton-Raphson oscillation.

**Symptom:**
- Inverter convergence failure
- gds = -2.95e-3 S (negative!)
- Oscillating voltages during iteration

**Solution:**
```python
g_ds = result.get("gds", 0.0)
g_m = result.get("gm", 0.0)
g_mb = result.get("gmb", 0.0)

# IMPORTANT: gds must always be positive (output conductance magnitude)
g_ds = abs(g_ds)

# gm and gmb are SIGNED transconductances - preserve their signs!
# Do NOT apply abs() to gm or gmb
return (g_ds, g_m, g_mb)
```

**Prevention:**
- Only apply `abs()` to **conductances** (gds), not **transconductances** (gm, gmb)
- Transconductances have physical sign meaning (feedback direction)
- Add minimum conductance as backup: `g_ds = max(abs(g_ds), 1e-6)`

---

### Sign Convention Checklist for New Device Models

When integrating new compact models (BSIM4, BSIM-SOI, HSPICE models, etc.):

**1. Understand the model's sign convention:**
- SPICE convention: Positive current = INTO terminal (id > 0 when current enters drain)
- Device convention: Positive current = drain-to-source flow
- PyCMG/OSDI uses: **SPICE convention** for terminal currents (`id`, `ig`, `is`, `ie`)

**2. Use terminal current `id`, NOT channel current `ids`:**
```python
# CRITICAL: Use result["id"] (drain terminal current), NOT result["ids"]
# ids = id - is ≈ 2*id (approximately double the terminal current)
# This caused a subtle 2x current error that was hard to catch

# NMOS: id < 0 when ON (SPICE: current leaves drain)
# calculate_current should return: -result["id"]  (positive = leaving drain)

# PMOS: id > 0 when ON (SPICE: current enters drain)
# calculate_current should return: result["id"]  (positive = entering drain)
```

**3. Check conductance signs:**
```python
# For both NMOS and PMOS:
print(f"gm = {result['gm']}")   # Should be POSITIVE (magnitude)
print(f"gds = {result['gds']}")  # Should be POSITIVE (can be negative at extremes!)
print(f"gmb = {result['gmb']}")  # Should be POSITIVE for normal operation
```

**4. Solver stamping — "current leaving drain" convention:**
```python
# The solver's _stamp_mosfet() uses "current leaving drain" for MNA:
# NMOS: calculate_current returns positive = leaving drain (already correct)
# PMOS: calculate_current returns positive = INTO drain (negate for MNA)
i_leaving = -i_ds if is_pmos else i_ds
i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
rhs[d_idx] -= i_eq  # Same for both NMOS and PMOS
rhs[s_idx] += i_eq
```

**5. Test BOTH device types against NGSPICE:**
- Single NMOS OP (expect drain current matches NGSPICE within 0.1%)
- Single PMOS with load resistor (expect V_drain matches NGSPICE)
- CMOS inverter DC sweep (VTC curve matches NGSPICE)
- CMOS inverter transient (waveform matches NGSPICE within ~2% NRMSE)

**6. Update device-type helpers:**
- Add new types to `_is_mosfet()` in `solver.py`
- Add new types to PMOS check in `_stamp_mosfet()` (both DC and transient)
- Search for all `isinstance(component, ...)` calls

---

### CRITICAL: calculate_current() Used Wrong Current Variable (FIXED: 2026-02-21)

**Severity**: Critical — drain current was approximately 2x the correct value
**Affected**: BSIM-CMG NMOS_CMG and PMOS_CMG
**Files**: `pycircuitsim/models/mosfet_cmg.py` lines ~243 (NMOS) and ~440 (PMOS)

**Root Cause:**
Both `NMOS_CMG.calculate_current()` and `PMOS_CMG.calculate_current()` used `result["ids"]`
(channel current = `id - is ≈ 2*id`) instead of `result["id"]` (drain terminal current).
Since `id ≈ -is` for a MOSFET, `ids = id - is ≈ 2*id`, giving approximately double the
correct terminal current.

**Symptom:**
- Drain current was ~2x what NGSPICE reported
- Hard to catch because the factor-of-2 error was consistent across bias points

**Solution:**
```python
# NMOS: id < 0 (SPICE convention, current leaves drain) → negate for "leaving drain"
return -result["id"]

# PMOS: id > 0 (SPICE convention, current enters drain) → use directly
return result["id"]
```

**Prevention:**
- Always use terminal currents (`id`, `ig`, `is`, `ie`), never channel current (`ids`)
- Verify against NGSPICE OP at a single bias point before running sweeps

---

### ASAP7 Modelcard Name Mismatch (FIXED: 2026-02-21)

**Severity**: High — parser failed to create BSIM-CMG devices
**Affected**: Parser when using ASAP7 modelcards
**Files**: `pycircuitsim/parser.py`, `pycircuitsim/models/mosfet_cmg.py`

**Root Cause:**
Netlists define models as `.model nmos1 NMOS (LEVEL=72)` but ASAP7 modelcards define
them as `.model nmos_rvt nmos (...)`. PyCMG's `Model()` searched for "nmos1" in the
modelcard and failed with "no nmos1 model found".

**Solution:**
- Added `model_card_name` parameter to `NMOS_CMG`/`PMOS_CMG` constructors
- Parser auto-detects ASAP7 modelcard usage and maps to `nmos_rvt`/`pmos_rvt`
- `model_card_name` overrides `model_name` for modelcard lookup

---

### PyCMG Path References (FIXED: 2026-02-21)

**Severity**: High — all BSIM-CMG imports failed
**Affected**: `config.py`, `mosfet_cmg.py`

**Root Cause:**
PyCMG was moved from `PROJECT_ROOT/PyCMG/` to `PROJECT_ROOT/models/PyCMG/` but path
references were never updated. Also, ASAP7 modelcard directory changed from
`tech_model_cards/asap7_pdk_r1p7/models/hspice/` to `tech_model_cards/ASAP7/`.

**Solution:**
- `config.py`: All three path constants updated to include `models/` prefix
- `mosfet_cmg.py`: `PYCMG_PATH` updated to traverse through `models/`

---

## BSIM-CMG NGSPICE Verification Results (2026-02-22)

All verification scripts in `tests/`:

| Test | Script | Metric | Result |
|------|--------|--------|--------|
| NMOS OP | `verify_bsimcmg_op.py` | Rel error | 0.00% |
| PMOS OP | `verify_bsimcmg_op.py` | Rel error | 0.01% |
| Inverter OP (Vin=0) | `verify_bsimcmg_op.py` | Rel error | 0.00% |
| Inverter OP (Vin=0.7) | `verify_bsimcmg_op.py` | Rel error | 0.00% |
| NMOS DC sweep | `verify_bsimcmg_dc.py` | NRMSE | 0.010% |
| PMOS DC sweep | `verify_bsimcmg_dc.py` | NRMSE | 0.014% |
| Inverter VTC | `verify_bsimcmg_dc.py` | NRMSE | 0.002% |
| Inverter Transient | `verify_bsimcmg_tran.py` | NRMSE (post-settling) | **0.23%** |
| Inverter Transient | `verify_bsimcmg_tran.py` | NRMSE (full-range) | **0.29%** |
| Inverter Transient | `verify_bsimcmg_tran.py` | Max error | 9.9 mV (1.4% Vdd) |

**Transient accuracy improvement history (Phase 7, 2026-02-22):**
| Change | Post-settling NRMSE | Full-range NRMSE |
|--------|--------------------:|------------------:|
| Baseline (Phase 6) | 2.10% | 14.24% |
| + Auto-scaled pseudo-caps | 2.05% | 7.13% |
| + Intrinsic capacitances (Cgd, Cgs, Cdd) | 1.46% | — |
| + Trapezoidal integration | 0.62% | 9.82% |
| + Skip convergence aids with DC OP | **0.23%** | **0.29%** |

Run all: `conda run -n pycircuitsim python tests/verify_bsimcmg_op.py && conda run -n pycircuitsim python tests/verify_bsimcmg_dc.py && conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py`

### Comprehensive Transient Verification (Phase 8, 2026-02-22)

14 parametric configurations sweeping VDD, Cload, input slew, and pulse width.
Script: `tests/verify_bsimcmg_tran_comprehensive.py`

| Config | VDD | Cload | tr/tf | pw | NRMSE(%) | MaxErr(mV) | Status |
|--------|-----|-------|-------|----|----------|------------|--------|
| vdd_0p5 | 0.50 | 10fF | 100ps | 0.8ns | 0.17 | 5.2 | PASS |
| vdd_0p6 | 0.60 | 10fF | 100ps | 0.8ns | 0.22 | 10.1 | PASS |
| baseline | 0.70 | 10fF | 100ps | 0.8ns | 0.22 | 9.9 | PASS |
| vdd_0p8 | 0.80 | 10fF | 100ps | 0.8ns | 0.20 | 11.1 | PASS |
| cload_1fF | 0.70 | 1fF | 100ps | 0.8ns | 0.95 | 67.3 | PASS |
| cload_5fF | 0.70 | 5fF | 100ps | 0.8ns | 0.30 | 17.6 | PASS |
| cload_50fF | 0.70 | 50fF | 100ps | 0.8ns | 0.04 | 1.3 | PASS |
| cload_100fF | 0.70 | 100fF | 100ps | 0.8ns | 0.03 | 0.9 | PASS |
| slew_10ps | 0.70 | 10fF | 10ps | 0.8ns | 0.15 | 7.2 | PASS |
| slew_50ps | 0.70 | 10fF | 50ps | 0.8ns | 0.13 | 6.1 | PASS |
| slew_500ps | 0.70 | 10fF | 500ps | 0.8ns | 0.17 | 8.3 | PASS |
| pw_0p2ns | 0.70 | 10fF | 100ps | 0.2ns | 0.31 | 9.9 | PASS |
| pw_0p5ns | 0.70 | 10fF | 100ps | 0.5ns | 0.26 | 9.9 | PASS |
| pw_2p0ns | 0.70 | 10fF | 100ps | 2.0ns | 0.15 | 9.9 | PASS |

Run: `conda run -n pycircuitsim python tests/verify_bsimcmg_tran_comprehensive.py`

---

## References
- **ngspice** - Physics equation verification
- **Xyce** - Architecture patterns for device/solver separation
- **Shichman-Hodges Model** - Level 1 MOSFET compact model
- **BSIM-CMG** - FinFET compact model (LEVEL=72), integrated via PyCMG
- **ASAP7** - https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7.git
- **PyCMG** - https://github.com/ShenShan123/PyCMG.git

## Other Notes
- Git commit for every significant change
- Single main entrance: `main.py` at project root

## Other Tips in This Project
* **Start every complex task in plan mode:** 
    * Pour your energy into the plan for 1-shot the implementation.
    * The moment something goes sideways, just switch back to plan mode and re-plan. Don't keep pushing.
    * Enter plan mode for verification steps, not just for the build.
* **Update CLAUDE.md:**
    * After every correction, update your CLAUDE.md so you don't make that mistake again.
* **Never be lazy:** 
    * Never be lazy in writing the code and running tests.
    * Do NOT use any simplifed equations or self-defined CMG models as reference, ALWAYS use simulation results as ground truth for comparison.
* Use subagents. 
    * Use a second agent to review the plan as a staff engineer.
    * If you want to try multiple solutions, use multiple subagents, git commit to different branches. Roll back and to the main branch and create new branch when the subagent find it's a dead end.
* Enable the "Explanatory" or "Learning" output style in /config to explain the *why* behind its changes.
