# PyCMG Update & BSIM-CMG Verification Design

## Problem

1. PyCMG at `models/PyCMG` is 34 commits behind `origin/master`
2. Path references in `config.py` and `mosfet_cmg.py` point to `PROJECT_ROOT/PyCMG/` but PyCMG lives at `models/PyCMG/`
3. `pycircuitsim/models/` was deleted from working tree (restored from git)
4. BSIM-CMG OP, DC, and transient analyses need verification against NGSPICE

## Solution

### 1. PyCMG Update
- Pull latest 34 commits in `models/PyCMG`
- Rebuild OSDI binary via cmake
- Fix path references in `config.py` and `mosfet_cmg.py` to use `models/PyCMG`

### 2. API Compatibility
The new PyCMG keeps the same `Model`/`Instance`/`eval_dc`/`eval_tran` API. Key changes that may affect us:
- Modelcard parser now lowercases parameter keys
- DEVTYPE auto-injection for ASAP7 PMOS
- New `get_jacobian_matrix()` method (additive)
- NF/NFIN treated as instance-only parameters

### 3. Verification (progressive)

**Phase A: OP Analysis** — Single NMOS, single PMOS, CMOS inverter at fixed bias points. Compare node voltages and device currents against NGSPICE.

**Phase B: DC Sweep** — NMOS Id-Vgs, PMOS Id-Vgs, inverter VTC (Vout vs Vin). Compare waveforms against NGSPICE using RMSE.

**Phase C: Transient** — Inverter with PULSE input and load capacitor. Compare time-domain waveforms against NGSPICE.

### 4. NGSPICE Ground Truth Method
- Write NGSPICE-compatible netlists loading the same `.osdi` binary
- Run NGSPICE in batch mode with `wrdata` to produce CSV
- Parse CSV and compare against PyCircuitSim output

### 5. Test Circuits
- NMOS single device: Id-Vgs sweep at Vds=0.05V
- PMOS single device: Id-Vgs sweep at Vds=-0.05V (Vsd=0.05V)
- CMOS inverter DC: VTC sweep Vin 0->0.7V
- CMOS inverter transient: PULSE input, 10fF load cap
