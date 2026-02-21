# PyCMG Update & BSIM-CMG Verification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update PyCMG to latest version, fix path references, and verify BSIM-CMG OP/DC/transient analyses against NGSPICE.

**Architecture:** Pull latest PyCMG (34 new commits), rebuild OSDI binary, fix config paths (`models/PyCMG` instead of `PyCMG`), then progressively verify OP -> DC sweep -> transient against NGSPICE ground truth using the same `.osdi` binary.

**Tech Stack:** Python 3 (conda env `pycircuitsim`), PyCMG (ctypes OSDI wrapper), NGSPICE 45.2 (`/usr/local/ngspice-45.2/bin/ngspice`), OpenVAF, ASAP7 7nm PDK modelcards.

**Conda env:** `pycircuitsim` at `/home/shenshan/.conda/envs/pycircuitsim`

**NGSPICE binary:** `/usr/local/ngspice-45.2/bin/ngspice`

**OSDI binary:** `models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi`

---

## Task 1: Restore deleted pycircuitsim/models/ directory

The `pycircuitsim/models/` directory was deleted from the working tree. All code imports from `pycircuitsim.models.*`. Restore it from git.

**Files:**
- Restore: `pycircuitsim/models/__init__.py`
- Restore: `pycircuitsim/models/base.py`
- Restore: `pycircuitsim/models/passive.py`
- Restore: `pycircuitsim/models/mosfet.py`
- Restore: `pycircuitsim/models/mosfet_cmg.py`

**Step 1: Restore files from git**

```bash
cd /home/shenshan/NN_SPICE
git checkout -- pycircuitsim/models/
```

**Step 2: Verify import works**

```bash
conda run -n pycircuitsim python -c "from pycircuitsim.simulation import run_simulation; print('Import OK')"
```

Expected: `Import OK`

**Step 3: Commit**

```bash
git add pycircuitsim/models/
git commit -m "fix: restore pycircuitsim/models/ directory deleted from working tree"
```

---

## Task 2: Update PyCMG to latest version

Pull the latest 34 commits from `origin/master` in `models/PyCMG`.

**Files:**
- Update: `models/PyCMG/` (entire subdir via git pull)

**Step 1: Pull latest PyCMG**

```bash
cd /home/shenshan/NN_SPICE/models/PyCMG
https_proxy=http://127.0.0.1:7890 git pull origin master
```

Expected: Fast-forward merge bringing in 34 commits.

**Step 2: Rebuild OSDI binary**

The OSDI binary must be rebuilt after pulling since the Verilog-A source or build system may have changed.

```bash
cd /home/shenshan/NN_SPICE/models/PyCMG
mkdir -p build-deep-verify
cd build-deep-verify
cmake ..
cmake --build . --target osdi
```

Expected: `bsimcmg.osdi` rebuilt at `build-deep-verify/osdi/bsimcmg.osdi`.

**Step 3: Verify OSDI binary exists**

```bash
ls -la /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi
file /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi
```

Expected: File exists, is a shared object (~400KB-3MB).

**Step 4: Verify PyCMG import works**

```bash
conda run -n pycircuitsim python -c "
import sys
sys.path.insert(0, '/home/shenshan/NN_SPICE/models/PyCMG')
from pycmg import Model, Instance
print('PyCMG import OK')
print(dir(Instance))
"
```

Expected: Shows `Model` and `Instance` classes with `eval_dc`, `eval_tran`, `get_jacobian_matrix` methods.

---

## Task 3: Fix path references in config.py

The config expects PyCMG at `PROJECT_ROOT/PyCMG/` but it's at `PROJECT_ROOT/models/PyCMG/`. Also, ASAP7 modelcard path changed from `tech_model_cards/asap7_pdk_r1p7/models/hspice` to `tech_model_cards/ASAP7/` in the new PyCMG.

**Files:**
- Modify: `pycircuitsim/config.py` (lines 15-29)

**Step 1: Edit config.py**

Change the three path constants to include `models/` prefix and update ASAP7 path:

```python
# OLD (line 17):
str(PROJECT_ROOT / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi")
# NEW:
str(PROJECT_ROOT / "models" / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi")

# OLD (line 24):
str(PROJECT_ROOT / "PyCMG" / "tech_model_cards" / "asap7_pdk_r1p7" / "models" / "hspice")
# NEW:
str(PROJECT_ROOT / "models" / "PyCMG" / "tech_model_cards" / "ASAP7")

# OLD (line 29):
str(PROJECT_ROOT / "PyCMG" / "bsim-cmg-va" / "benchmark_test")
# NEW:
str(PROJECT_ROOT / "models" / "PyCMG" / "bsim-cmg-va" / "benchmark_test")
```

**Step 2: Verify paths resolve correctly**

```bash
conda run -n pycircuitsim python -c "
from pycircuitsim.config import BSIMCMG_OSDI_PATH, ASAP7_MODELCARD_DIR, GENERIC_MODELCARD_DIR
import os
print('OSDI:', BSIMCMG_OSDI_PATH)
print('OSDI exists:', os.path.exists(BSIMCMG_OSDI_PATH))
print('ASAP7:', ASAP7_MODELCARD_DIR)
print('ASAP7 exists:', os.path.exists(ASAP7_MODELCARD_DIR))
print('Generic:', GENERIC_MODELCARD_DIR)
print('Generic exists:', os.path.exists(GENERIC_MODELCARD_DIR))
"
```

Expected: All three paths exist and resolve correctly.

---

## Task 4: Fix PyCMG sys.path in mosfet_cmg.py

The `mosfet_cmg.py` file calculates `PYCMG_PATH` relative to itself, going up 3 levels to find `PyCMG/`. After the move, it needs to go to `models/PyCMG/`.

**Files:**
- Modify: `pycircuitsim/models/mosfet_cmg.py` (lines 22-25)

**Step 1: Edit mosfet_cmg.py**

```python
# OLD (line 23):
PYCMG_PATH = Path(__file__).parent.parent.parent / "PyCMG"
# NEW:
PYCMG_PATH = Path(__file__).parent.parent.parent / "models" / "PyCMG"
```

The path traversal: `mosfet_cmg.py` -> `pycircuitsim/models/` -> `pycircuitsim/` -> `NN_SPICE/` -> `NN_SPICE/models/PyCMG/`.

**Step 2: Verify BSIM-CMG model imports work end-to-end**

```bash
conda run -n pycircuitsim python -c "
from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
print('NMOS_CMG:', NMOS_CMG)
print('PMOS_CMG:', PMOS_CMG)
"
```

Expected: Both classes imported successfully.

**Step 3: Commit path fixes**

```bash
cd /home/shenshan/NN_SPICE
git add pycircuitsim/config.py pycircuitsim/models/mosfet_cmg.py
git commit -m "fix: update PyCMG path references to models/PyCMG"
```

---

## Task 5: Fix ASAP7 modelcard filename in parser

The parser's `ASAP7_MODELCARD_FILES` list must match actual filenames in the new `tech_model_cards/ASAP7/` directory. The new directory has `7nm_TT.pm`, `7nm_TT_160803.pm`, `7nm_FF.pm`, `7nm_FF_160803.pm`, `7nm_SS.pm`, `7nm_SS_160803.pm`. The parser currently lists `"7nm_TT_160803.pm"`, `"7nm_FF.pm"`, `"7nm_SS.pm"` — these should still work since files exist. Verify and fix if needed.

**Files:**
- Check: `pycircuitsim/parser.py` (lines 83-87)

**Step 1: Verify modelcard resolution works**

```bash
conda run -n pycircuitsim python -c "
from pycircuitsim.config import ASAP7_MODELCARD_DIR
from pathlib import Path
import os
print('ASAP7 dir:', ASAP7_MODELCARD_DIR)
for f in sorted(os.listdir(ASAP7_MODELCARD_DIR)):
    print(f'  {f}')

# Test the lookup that parser does
ASAP7_MODELCARD_FILES = ['7nm_TT_160803.pm', '7nm_FF.pm', '7nm_SS.pm']
for fname in ASAP7_MODELCARD_FILES:
    p = Path(ASAP7_MODELCARD_DIR) / fname
    print(f'{fname}: exists={p.exists()}')
"
```

Expected: All three files exist. If not, update the list.

**Step 2: Run parser on a BSIM-CMG netlist to verify end-to-end**

```bash
conda run -n pycircuitsim python -c "
from pycircuitsim.parser import Parser
p = Parser()
p.parse_file('examples/bsimcmg_nmos_dc.sp')
print('Components:', len(p.circuit.components))
for c in p.circuit.components:
    print(f'  {c.name}: {type(c).__name__}')
print('Analysis:', p.analysis_type, p.analysis_params)
"
```

Expected: Parses successfully, shows 3 components (Vds, Vgs, Mn1 as NMOS_CMG), analysis type `dc`.

---

## Task 6: Verify OP analysis — NMOS single device

Run a single NMOS at a fixed bias and compare against NGSPICE.

**Files:**
- Create: `tests/verify_bsimcmg_op.py`
- Use: `examples/bsimcmg_nmos_dc.sp` (modify to `.op` only)

**Step 1: Create NGSPICE netlist for NMOS OP**

Create `tests/ngspice_nmos_op.cir`:

```spice
* NMOS OP verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model nmos1 nmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vds drain 0 0.5
Vgs gate 0 0.5
Mn1 drain gate 0 0 nmos1 L=30n NFIN=10

.control
op
print v(drain) v(gate) @mn1[id] @mn1[gm] @mn1[gds]
.endc

.end
```

**Step 2: Run NGSPICE and capture output**

```bash
/usr/local/ngspice-45.2/bin/ngspice -b tests/ngspice_nmos_op.cir 2>&1 | tee tests/ngspice_nmos_op.log
```

Expected: NGSPICE prints node voltages and device parameters.

**Step 3: Run PyCircuitSim on same circuit**

Write `tests/verify_bsimcmg_op.py`:

```python
#!/usr/bin/env python3
"""Verify BSIM-CMG OP analysis against NGSPICE."""
import sys
sys.path.insert(0, "/home/shenshan/NN_SPICE")

from pycircuitsim.parser import Parser
from pycircuitsim.solver import DCSolver

# Parse netlist — we create a simple OP netlist inline
netlist = """* NMOS OP test
Vds 2 0 0.5
Vgs 1 0 0.5
Mn1 2 1 0 0 nmos1 L=30n NFIN=10
.model nmos1 NMOS (LEVEL=72)
.end
"""

# Write temp netlist
with open("/tmp/nmos_op_test.sp", "w") as f:
    f.write(netlist)

parser = Parser()
parser.parse_file("/tmp/nmos_op_test.sp")

solver = DCSolver(parser.circuit)
solution = solver.solve()

print("PyCircuitSim OP Results:")
for node, v in sorted(solution.items()):
    print(f"  V({node}) = {v:.6e} V")

# Also print device currents
for comp in parser.circuit.components:
    try:
        i = comp.calculate_current(solution)
        print(f"  I({comp.name}) = {i:.6e} A")
    except (NotImplementedError, AttributeError):
        pass
```

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_op.py
```

**Step 4: Compare results**

Compare node voltages (should match exactly since they're set by voltage sources) and drain current (should match NGSPICE within 1% relative error).

---

## Task 7: Verify OP analysis — PMOS single device

Same as Task 6 but for PMOS. This validates the PMOS RHS stamping fix and DEVTYPE auto-injection.

**Files:**
- Extend: `tests/verify_bsimcmg_op.py` (add PMOS test)
- Create: `tests/ngspice_pmos_op.cir`

**Step 1: Create NGSPICE PMOS netlist**

```spice
* PMOS OP verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model pmos1 pmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vdd vdd 0 0.7
Vgs gate 0 0.2
Rload drain 0 10k
Mp1 drain gate vdd vdd pmos1 L=30n NFIN=10

.control
op
print v(drain) v(gate) v(vdd) @mp1[id] @mp1[gm] @mp1[gds]
.endc

.end
```

**Step 2: Run NGSPICE**

```bash
/usr/local/ngspice-45.2/bin/ngspice -b tests/ngspice_pmos_op.cir 2>&1 | tee tests/ngspice_pmos_op.log
```

**Step 3: Run PyCircuitSim PMOS OP and compare**

Add PMOS test to `tests/verify_bsimcmg_op.py` with same comparison approach.

---

## Task 8: Verify OP analysis — CMOS inverter

Test complementary NMOS+PMOS in an inverter topology.

**Files:**
- Extend: `tests/verify_bsimcmg_op.py` (add inverter test)
- Create: `tests/ngspice_inverter_op.cir`

**Step 1: Create NGSPICE inverter OP netlist**

```spice
* CMOS Inverter OP verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model nmos1 nmos level=72
.model pmos1 pmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vdd vdd 0 0.7
Vin in 0 0.0

Mp1 out in vdd vdd pmos1 L=30n NFIN=10
Mn1 out in 0 0 nmos1 L=30n NFIN=10

.control
op
print v(out) v(in) v(vdd)
.endc

.end
```

**Step 2: Run NGSPICE and PyCircuitSim, compare Vout**

For Vin=0.0V, expect Vout ~ Vdd (PMOS on, NMOS off).
For Vin=0.7V, expect Vout ~ 0V (PMOS off, NMOS on).

Test both bias points.

**Step 3: Commit OP verification**

```bash
cd /home/shenshan/NN_SPICE
git add tests/verify_bsimcmg_op.py tests/ngspice_*.cir tests/ngspice_*.log
git commit -m "test: add BSIM-CMG OP verification against NGSPICE"
```

---

## Task 9: Verify DC sweep — NMOS Id-Vgs

Sweep Vgs from 0 to 0.7V and compare Id-Vgs curve against NGSPICE.

**Files:**
- Create: `tests/verify_bsimcmg_dc.py`
- Create: `tests/ngspice_nmos_dc.cir`

**Step 1: Create NGSPICE DC sweep netlist**

```spice
* NMOS DC sweep verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model nmos1 nmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vds drain 0 0.5
Vgs gate 0 0.0
Mn1 drain gate 0 0 nmos1 L=30n NFIN=10

.control
dc Vgs 0 0.7 0.01
wrdata /tmp/ngspice_nmos_dc.csv v(drain) i(Vds)
.endc

.end
```

**Step 2: Run NGSPICE**

```bash
/usr/local/ngspice-45.2/bin/ngspice -b tests/ngspice_nmos_dc.cir
```

**Step 3: Run PyCircuitSim DC sweep**

```bash
conda run -n pycircuitsim python main.py examples/bsimcmg_nmos_dc.sp -o results
```

Note: The existing `bsimcmg_nmos_dc.sp` sweeps Vgs 0->1.0V with step 0.02. Adjust to match NGSPICE netlist range (0->0.7V) or adjust NGSPICE to match.

**Step 4: Compare waveforms**

Write `tests/verify_bsimcmg_dc.py` that:
1. Reads NGSPICE CSV output
2. Reads PyCircuitSim CSV output
3. Interpolates to common sweep points
4. Computes RMSE and max absolute error
5. Plots both on same axes for visual comparison

Acceptance criteria: RMSE < 1% of max current, max relative error < 5%.

---

## Task 10: Verify DC sweep — PMOS Id-Vgs

Same structure as Task 9 but for PMOS.

**Files:**
- Extend: `tests/verify_bsimcmg_dc.py`
- Create: `tests/ngspice_pmos_dc.cir`

**Step 1: Create NGSPICE PMOS DC sweep netlist**

PMOS convention: Vgs measured from gate to source (source=Vdd). Sweep Vgs from 0 to -0.7V (i.e. gate from Vdd down to 0).

```spice
* PMOS DC sweep verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model pmos1 pmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vdd vdd 0 0.7
Vgs gate 0 0.7
Rload drain 0 1k
Mp1 drain gate vdd vdd pmos1 L=30n NFIN=10

.control
dc Vgs 0 0.7 0.01
wrdata /tmp/ngspice_pmos_dc.csv v(drain) i(Vdd)
.endc

.end
```

**Step 2: Run both simulators, compare as in Task 9**

---

## Task 11: Verify DC sweep — Inverter VTC

Sweep inverter input from 0 to 0.7V, compare Vout vs Vin transfer curve.

**Files:**
- Extend: `tests/verify_bsimcmg_dc.py`
- Create: `tests/ngspice_inverter_dc.cir`
- Modify: `examples/bsimcmg_inverter_dc.sp` (add .dc sweep)

**Step 1: Create NGSPICE inverter DC sweep netlist**

```spice
* CMOS Inverter VTC verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model nmos1 nmos level=72
.model pmos1 pmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vdd vdd 0 0.7
Vin in 0 0.0

Mp1 out in vdd vdd pmos1 L=30n NFIN=10
Mn1 out in 0 0 nmos1 L=30n NFIN=10

.control
dc Vin 0 0.7 0.01
wrdata /tmp/ngspice_inverter_dc.csv v(out)
.endc

.end
```

**Step 2: Update PyCircuitSim inverter netlist**

Update `examples/bsimcmg_inverter_dc.sp` to add a `.dc` sweep:

```spice
* BSIM-CMG CMOS Inverter DC VTC
Vdd 1 0 0.7
Vin 2 0 0.0
Mp1 3 2 1 1 pmos1 L=30n NFIN=10
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)
.dc Vin 0 0.7 0.01
.end
```

**Step 3: Run both, compare VTC curves**

The VTC should show:
- Vout ~ 0.7V when Vin ~ 0V
- Vout ~ 0V when Vin ~ 0.7V
- Sharp transition around Vin ~ 0.3-0.4V

**Step 4: Commit DC verification**

```bash
cd /home/shenshan/NN_SPICE
git add tests/verify_bsimcmg_dc.py tests/ngspice_*_dc.cir examples/bsimcmg_inverter_dc.sp
git commit -m "test: add BSIM-CMG DC sweep verification against NGSPICE"
```

---

## Task 12: Verify transient analysis — Inverter

This is the most challenging part. Run inverter transient with PULSE input and compare waveforms.

**Files:**
- Create: `tests/verify_bsimcmg_tran.py`
- Create: `tests/ngspice_inverter_tran.cir`
- Use/modify: `examples/bsimcmg_inverter_tran.sp`

**Step 1: Create NGSPICE transient netlist**

```spice
* CMOS Inverter Transient verification
.osdi /home/shenshan/NN_SPICE/models/PyCMG/build-deep-verify/osdi/bsimcmg.osdi

.model nmos1 nmos level=72
.model pmos1 pmos level=72
.include /home/shenshan/NN_SPICE/models/PyCMG/tech_model_cards/ASAP7/7nm_TT_160803.pm

Vdd vdd 0 0.7
Vin in 0 PULSE(0 0.7 0.5n 0.1n 0.1n 0.8n 2n)

Mp1 out in vdd vdd pmos1 L=30n NFIN=10
Mn1 out in 0 0 nmos1 L=30n NFIN=10
Cload out 0 10f

.ic v(out)=0.7

.control
tran 10p 5n
wrdata /tmp/ngspice_inverter_tran.csv v(out) v(in)
.endc

.end
```

**Step 2: Run NGSPICE**

```bash
/usr/local/ngspice-45.2/bin/ngspice -b tests/ngspice_inverter_tran.cir
```

**Step 3: Update PyCircuitSim inverter transient netlist**

Update `examples/bsimcmg_inverter_tran.sp` to use Vdd=0.7V (matching ASAP7):

```spice
* BSIM-CMG CMOS Inverter Transient
Vdd 1 0 0.7
Vin 2 0 PULSE 0 0.7 0.5n 0.1n 0.1n 0.8n 2n
Mp1 3 2 1 1 pmos1 L=30n NFIN=10
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
Cload 3 0 10f
.ic V(3)=0.7
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)
.tran 10p 5n
.end
```

**Step 4: Run PyCircuitSim transient**

```bash
conda run -n pycircuitsim python main.py examples/bsimcmg_inverter_tran.sp -o results -v
```

**Step 5: Compare waveforms**

Write `tests/verify_bsimcmg_tran.py` that:
1. Reads NGSPICE CSV output (time, v(out), v(in))
2. Reads PyCircuitSim transient CSV output
3. Interpolates to common time points
4. Computes RMSE for Vout
5. Plots both on same axes

Acceptance criteria: RMSE < 10% of Vdd for transient (looser than DC due to numerical method differences).

**Step 6: Debug convergence issues if needed**

If transient fails to converge, check:
- Is the initial DC OP solving correctly?
- Are MOSFET capacitances being included in transient? (The solver uses Backward Euler for capacitors)
- Is the time step small enough relative to transition times?
- Are the Newton-Raphson damping/clamping thresholds appropriate?

Known issue from CLAUDE.md: transient convergence struggles with fast switching. The `TransientSolver` has Gmin stepping and pseudo-transient options (enabled by default in `run_transient`).

**Step 7: Commit transient verification**

```bash
cd /home/shenshan/NN_SPICE
git add tests/verify_bsimcmg_tran.py tests/ngspice_inverter_tran.cir examples/bsimcmg_inverter_tran.sp
git commit -m "test: add BSIM-CMG transient verification against NGSPICE"
```

---

## Task 13: Fix any issues found during verification

This is a placeholder task. Based on verification results from Tasks 6-12, there may be bugs to fix in:

- `pycircuitsim/solver.py` — MNA stamping, Newton-Raphson convergence
- `pycircuitsim/models/mosfet_cmg.py` — current/conductance sign conventions
- `pycircuitsim/parser.py` — value parsing, modelcard resolution
- `pycircuitsim/simulation.py` — analysis orchestration

For each bug found:
1. Identify the root cause
2. Write a minimal test that reproduces it
3. Fix the code
4. Verify the fix
5. Commit with descriptive message

---

## Task 14: Final integration test and commit

Run all three analysis types end-to-end and produce a summary.

**Step 1: Run all verifications**

```bash
conda run -n pycircuitsim python tests/verify_bsimcmg_op.py
conda run -n pycircuitsim python tests/verify_bsimcmg_dc.py
conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py
```

**Step 2: Print summary table**

Expected format:
```
BSIM-CMG Verification Summary
==============================
Test                    | Status | Error
NMOS OP (Id)           | PASS   | < 1%
PMOS OP (Id)           | PASS   | < 1%
Inverter OP (Vout)     | PASS   | < 1%
NMOS DC sweep (Id-Vgs) | PASS   | RMSE < 1%
PMOS DC sweep (Id-Vgs) | PASS   | RMSE < 1%
Inverter VTC (Vout)    | PASS   | RMSE < 1%
Inverter Transient     | PASS   | RMSE < 10%
```

**Step 3: Final commit**

```bash
cd /home/shenshan/NN_SPICE
git add -A
git commit -m "feat: update PyCMG and verify BSIM-CMG OP/DC/transient against NGSPICE"
```

---

## Dependency Order

```
Task 1 (restore models/) -> Task 2 (update PyCMG) -> Task 3+4 (fix paths)
                                                         |
                                                         v
                                                    Task 5 (verify parser)
                                                         |
                                                         v
                                          Tasks 6-8 (OP verification, parallel)
                                                         |
                                                         v
                                          Tasks 9-11 (DC verification, parallel)
                                                         |
                                                         v
                                              Task 12 (Transient verification)
                                                         |
                                                         v
                                              Task 13 (Bug fixes as needed)
                                                         |
                                                         v
                                              Task 14 (Final integration)
```

Tasks 6-8 can run in parallel. Tasks 9-11 can run in parallel. Task 12 depends on DC working first (transient starts with DC OP).
