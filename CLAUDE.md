# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture. Supports DC/transient analysis with Level 1 MOSFET models and BSIM-CMG FinFET compact models (LEVEL=72).

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

PyCMG/                  # BSIM-CMG OSDI wrapper (external submodule)
main.py                 # CLI entry point (single main entrance)
examples/*.sp           # Example netlists
results/                # Simulation output (.lis, .csv, .png)
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
* `.dc` - DC sweep analysis
* `.tran` - Transient analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=1 or LEVEL=72)
* `.include` - External library files
* `.ic` - Initial conditions (critical for SRAM/bistable circuits)

## Status

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
- [ ] Transient analysis stability (convergence issues remain)

---

## Quick Start

### Basic Simulation
Create a netlist (`.sp` file) with your circuit. Examples provided in `examples/` directory.

```bash
# Set PYTHONPATH if not installed
export PYTHONPATH=/path/to/NN_SPICE:$PYTHONPATH

# Run simulation
python main.py examples/rc_transient.sp
python main.py examples/test_nmos_level1.sp

# Custom circuit
python main.py your_circuit.sp
```

### MOSFET Terminal Order
**Important**: Terminals are `drain gate source bulk`

**Level 1 Models (Shichman-Hodges):**
```spice
* NMOS: drain=output, gate=input, source=GND, bulk=GND
Mn1 3 2 0 0 NMOS_VTL L=1u W=10u

* PMOS: drain=output, gate=input, source=Vdd, bulk=Vdd
Mp1 3 2 1 1 PMOS_VTL L=1u W=20u

.model NMOS_VTL NMOS (LEVEL=1 VTO=0.7 KP=50u)
.model PMOS_VTL PMOS (LEVEL=1 VTO=-0.7 KP=-20u)
```

**BSIM-CMG Models (FinFET LEVEL=72):**
```spice
* NMOS FinFET: 10 fins, 30nm channel length
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* PMOS FinFET: 10 fins, 30nm channel length
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* Use ASAP7 7nm modelcards or custom BSIM-CMG parameters
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)
```

**BSIM-CMG Geometric Parameters:**
- `L` - Channel length (required, in meters e.g., 30n)
- `NFIN` - Number of fins (required, integer or float)
- `TFIN` - Fin thickness (optional, uses modelcard default if omitted)
- `HFIN` - Fin height (optional, uses modelcard default)
- `FPITCH` - Fin pitch (optional, uses modelcard default)

### Python API
```python
from pycircuitsim.simulation import run_simulation

# Run simulation programmatically
run_simulation(
    netlist_path='circuit.sp',
    output_dir='results',
    verbose=True
)
```

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` / `*_transient.csv` - Waveform data (node voltages + device currents)
- `*_dc_sweep.png` / `*_transient.png` - Voltage/current plots

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

## Critical Bugs and Solutions

### CRITICAL: PMOS RHS Stamping Bug (FIXED: 2026-02-09)

**Severity**: Critical - PMOS circuits produced completely wrong results
**Affected**: Both Level 1 and BSIM-CMG PMOS devices
**Files**: `pycircuitsim/solver.py` lines 628-660 (DC), 1024-1050 (transient)

**Root Cause:**
The solver applied the SAME RHS stamping formula to both NMOS and PMOS, but they have **opposite physical current directions**:

- **NMOS:** Current flows OUT OF drain → circuit equation requires `rhs[drain] -= i_eq`
- **PMOS:** Current flows INTO drain → circuit equation requires `rhs[drain] += i_eq` (OPPOSITE!)

**Symptom:**
- PMOS single-device test: V_drain = **-0.381V** (negative voltage, physically impossible!)
- Level 1 PMOS test: V_drain = **-0.102V** (also wrong)
- Expected: V_drain ≈ +0.9V (near Vdd)

**Solution:**
Added device-type checking in `_stamp_mosfet()` and `_stamp_mosfet_transient()`:

```python
# Check if PMOS or NMOS
from pycircuitsim.models.mosfet import PMOS
try:
    from pycircuitsim.models.mosfet_cmg import PMOS_CMG
    is_pmos = isinstance(mosfet, (PMOS, PMOS_CMG))
except ImportError:
    is_pmos = isinstance(mosfet, PMOS)

if is_pmos:
    # PMOS: current INTO drain, OUT OF source
    rhs[d_idx] += i_eq  # OPPOSITE of NMOS
    rhs[s_idx] -= i_eq
else:
    # NMOS: current OUT OF drain, INTO source
    rhs[d_idx] -= i_eq
    rhs[s_idx] += i_eq
```

**Prevention:**
- Always consider PMOS vs NMOS **separately** when stamping MNA equations
- Test both device types in isolation before testing complementary circuits
- Verify physical correctness: PMOS should pull node UP, NMOS should pull DOWN
- Sign conventions matter: SPICE (current INTO terminal) vs circuit solver (current OUT OF terminal)

**Validation After Fix:**
- NMOS: V_drain = 0.116V ✓
- PMOS (BSIM-CMG): V_drain = 0.650V ✓ (was -0.381V)
- PMOS (Level 1): V_drain = 0.104V ✓ (was -0.102V)
- Inverter DC: V_out = 0.026V @ V_in = 0.5V ✓

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
- PyCMG/OSDI uses: **SPICE convention** for `ids`

**2. Check current sign:**
```python
# Test NMOS ON (Vgs > Vth, Vds > 0)
result = instance.eval_dc({"d": 0.5, "g": 1.0, "s": 0.0, "e": 0.0})
print(f"NMOS ON: ids = {result['ids']}")  # Should be NEGATIVE (SPICE: current OUT)

# Test PMOS ON (Vgs < Vth, Vds < 0)
result = instance.eval_dc({"d": 0.0, "g": 0.0, "s": 1.0, "e": 1.0})
print(f"PMOS ON: ids = {result['ids']}")  # Should be POSITIVE (SPICE: current IN)
```

**3. Check conductance signs:**
```python
# For both NMOS and PMOS:
print(f"gm = {result['gm']}")   # Should be POSITIVE (magnitude)
print(f"gds = {result['gds']}")  # Should be POSITIVE (can be negative at extremes!)
print(f"gmb = {result['gmb']}")  # Should be POSITIVE for normal operation
```

**4. Apply correct transformations:**
```python
# pycircuitsim convention: positive i_ds = current OUT OF drain
# Both NMOS and PMOS need negation from SPICE convention:
return -result["ids"]

# Conductances: gds magnitude, gm/gmb preserve sign
return (abs(g_ds), g_m, g_mb)
```

**5. Test BOTH device types:**
- Single NMOS with load resistor (expect V_drain low)
- Single PMOS with load resistor (expect V_drain high)
- CMOS inverter DC (expect switching around Vdd/2)
- Verify voltages are physically reasonable (0 < V < Vdd)

**6. Update device-type helpers:**
- Add new types to `_is_mosfet()` in `solver.py`
- Add new types to PMOS check in `_stamp_mosfet()` (both DC and transient)
- Search for all `isinstance(component, ...)` calls

---

## References
- **ngspice** - Physics equation verification
- **Xyce** - Architecture patterns for device/solver separation
- **Shichman-Hodges Model** - Level 1 MOSFET compact model
- **BSIM-CMG** - FinFET compact model (LEVEL=72), integrated via PyCMG

## Other Notes
- Use conda environment `pycircuitsim`
- Git commit for every significant change
- Single main entrance: `main.py` at project root
- BSIM-CMG integration: Requires PyCMG library with OSDI binary (see `PyCMG/README.md`)
