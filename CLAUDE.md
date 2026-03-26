# Project: PyCircuitSim

## Overview
Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture.
**Primary Goal:** specific support for **Level-1 MOS models**, **PyCMG-wrapped CMG models** (LEVEL=72), and **NN-based compact models** (LEVEL=73).
The simulator must support **Operating Point (OP)**, **DC Sweep**, and **Transient Analysis** for all model types.

**Core Principles:**
* Pure Python with clean, readable code
* Complete decoupling: Solver <-> Device Models
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
    ├── mosfet_cmg.py   # BSIM-CMG FinFET model (LEVEL=72) via PyCMG
    └── mosfet_nn.py    # NN-based compact model (LEVEL=73) via PyTorch

nn_model/                           # NN training pipeline
├── config.py                       # Hyperparams, paths, tech configs
├── data/
│   ├── generate.py                 # PyCMG bias sweep → .npz datasets
│   ├── normalize.py                # Signed-log + z-score normalization
│   └── dataset.py                  # PyTorch Dataset/DataLoader
├── architecture/
│   ├── direct_loss.py              # DirectNet MLP + DirectLoss (13 outputs)
│   ├── mosfet_net.py               # Dual-head MLP (MOSFETNet, for reference)
│   └── physics_loss.py             # Autograd derivative-supervised loss
├── train.py                        # Training loop (direct13/finetune modes)
└── checkpoints/                    # Saved model weights (.pt + _norm.npz)

external_compact_models/
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
* **Trapezoidal Integration** - 2nd-order time integration for transient analysis (charge-based)
* **Source Stepping** - Two-stage analysis for improved convergence

## Supported Features

### Devices
* Passive: R, C
* Active:
  - NMOS/PMOS Level 1 (Shichman-Hodges)
  - NMOS/PMOS Level 72 (BSIM-CMG FinFET via PyCMG)
  - NMOS/PMOS Level 73 (NN-based compact model via PyTorch)
* Sources: DC voltage/current, PULSE

### Analysis
* `.op` - Operating Point Analysis (Basic DC solution)
* `.dc` - DC Sweep Analysis
* `.tran` - Transient Analysis

### Directives
* `.model` - MOSFET model definitions (LEVEL=1, LEVEL=72, or LEVEL=73)
* `.include` - External library files
* `.ic` - Initial conditions (critical for SRAM/bistable circuits)

## Validation Strategy

* **Test Case:** An inverter circuit must be used to verify functionality.
* **Analysis:** The inverter must successfully pass **Transient Analysis**.
* **Ground Truth:** All simulation results must be verified against **NGSPICE**.
* **Metric:** Waveforms and operating points must match NGSPICE within reasonable numerical tolerance.

## Status

All phases (1-15) are complete. Key milestones:
- **Phases 1-3:** Core simulator (MNA, NR solver, Level-1 MOSFET, transient)
- **Phases 4-6:** BSIM-CMG (LEVEL=72) integration via PyCMG, NGSPICE-verified (<0.02% OP, <0.1% DC)
- **Phases 7-10:** Charge-based transient (0.20% NRMSE vs NGSPICE), 5-tech support (ASAP7, TSMC5/7/12/16), 21-config parametric sweep all PASS
- **Phases 11-12:** NN compact model (LEVEL=73) — training pipeline, autograd conductances, multi-tech DC+transient verified
- **Phases 13-15:** Universal NN v2 — 21 variants across 5 techs, 13-dim input (voltages + 7 process params), 19/21 PASS (ASAP7:SLVT and TSMC7:LVT FAIL on NMOS DC)
- **Leave-one-out transferability** — 8/10 good transfer (gap < 5%), zero-shot avg 4.65% NRMSE, in-dist avg 0.95%

### Future Work
- [ ] **Improved NN Transient Accuracy** — Retrain with `--w-charges 1.5 --w-caps 1.0`, PhysicsLoss for capacitances
- [ ] **Expanded Test Suite** — NAND/NOR gates, Ring Oscillator, SRAM bitcell
- [ ] **Adaptive Timestep** — Local truncation error estimate for automatic timestep control

---

## Quick Start

### Basic Simulation
Create a netlist (`.sp` file). Examples in `examples/`.

**BSIM-CMG Geometric Parameters:** `L` (channel length), `NFIN` (fin count), `TFIN`/`HFIN`/`FPITCH` (optional, uses modelcard defaults).

### NN Model (LEVEL=73)
```bash
# Generate universal data (21 variants, ~815K pts x 2)
conda run -n pycircuitsim python -m nn_model.data.generate --device both --universal
# Train on GPU
conda run -n pycircuitsim python -u -m nn_model.train --device-type nmos --universal --mode direct13 --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda
conda run -n pycircuitsim python -u -m nn_model.train --device-type pmos --universal --mode direct13 --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048 --cuda
```
Checkpoints: Universal -> `universal_{nmos,pmos}_best.pt`, Per-tech -> `{tech}_{nmos,pmos}_best.pt` + `_norm.npz`.
Netlist usage: `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`.
Parser auto-resolves process params from TECH+VT and prefers universal checkpoint when available.
Direct process params: `.model nmos_nn NMOS (LEVEL=73 PHIG=4.41 U0=0.033 VSAT=65370 EOT=1.06e-9 ETA0=0.005 CIT=-9.81e-4 RDSW=15)`.

### Output Files
Results organized in `results/<circuit_name>/<analysis_type>/`:
- `*_simulation.lis` - Detailed iteration log (HSPICE-like)
- `*_dc_sweep.csv` / `*_transient.csv` - Waveform data (node voltages + device currents)

## Testing & Verification

All tests require: `conda activate pycircuitsim`

| Test Suite | Script | What it tests |
|-----------|--------|---------------|
| OP Verification | `verify_bsimcmg_op.py` | NMOS, PMOS, Inverter OP vs NGSPICE (<0.02%) |
| DC Sweep | `verify_bsimcmg_dc.py` | Id-Vgs, VTC vs NGSPICE (<0.1% NRMSE) |
| Transient | `verify_bsimcmg_tran.py` | Inverter pulse vs NGSPICE (<0.5% NRMSE) |
| Comprehensive Transient | `verify_bsimcmg_tran_comprehensive.py` | 21 parametric configs (VDD, Cload, slew, pw, NFIN, P/N ratio) |
| Multi-Tech Transient | `verify_multi_tech_tran.py` | 5 techs, baseline + parametric sweep |
| NN Multi-Tech | `verify_nn_multi_tech.py` | NMOS/PMOS DC + Inverter VTC per tech (<10%/15%) |
| NN Universal v2 | `verify_nn_universal_v2.py` | 21 variants x 3 tests (63 tests) |
| NN Transient | `verify_nn_tran.py` | NN vs NGSPICE transient per tech (<15%) |
| NN Leave-One-Out | `verify_nn_leave_one_out.py` | Zero-shot transferability experiment |

**Quick Sanity Check:**
```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
```

---

## Development Guidelines

### Coding Standards
- Type hints required for all function signatures
- Clear variable names (e.g., `v_gate`, `i_drain`, not `a`, `b`)
- Docstrings for complex algorithms
- Voltage clamping: Vgs +/- 5V, Vds +/- 10V

### Separation Principle
- **Solver** (`solver.py`) builds MNA matrix, executes Newton-Raphson (no device equations)
- **Device Models** (`models/`) calculate current/conductances from voltages (no matrix operations)
- **Simulation** (`simulation.py`) orchestrates the workflow (parse -> solve -> visualize)
- All devices inherit from `Component` base class

### Key Numerical Techniques
- Minimum conductance (1uS) prevents singular matrices
- Source stepping (20 steps) improves convergence
- Damping factor (0.5) for large voltage deltas
- Two-stage analysis: DC OP -> DC sweep/transient
- Voltage-source-constrained nodes exempt from damping

### Entry Points
- **CLI**: `main.py` - Command-line interface (argparse, error handling)
- **API**: `pycircuitsim.simulation.run_simulation()` - Programmatic access
- **Module**: `pycircuitsim` - Package exports (Circuit, Parser, Visualizer, run_simulation)

## Environment & Tools
* **Conda Environment**: `pycircuitsim` in `/home/shenshan/.conda/envs/pycircuitsim`
* **PyTorch:** 2.10.0 (CPU, installed via pip in pycircuitsim env)
* **OpenVAF Compiler:** `/usr/local/bin/openvaf`
* **NGSPICE Simulator:** `/usr/local/ngspice-45.2/bin/ngspice`
* **Build System:** CMake / Make
* **Python Bindings:** PyBind11

---

## Critical Design Rules

These rules were learned from bugs. Violating them causes NR divergence or wrong results.

### Sign Convention for Device Models

When integrating new compact models, follow this checklist:

1. **Use terminal current `id`, NOT channel current `ids`** — `ids = id - is ~ 2*id` (2x error)
2. **NMOS `calculate_current()`**: return `-result["id"]` (positive = current leaving drain)
3. **PMOS `calculate_current()`**: return `result["id"]` (positive = current into drain)
4. **Solver stamping** uses unified "current leaving drain" convention:
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq  # Same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
5. **Conductance signs**: `abs(gds)` always (can be negative at extremes), but preserve gm/gmb signs
6. **Update `_is_mosfet()`** in `solver.py` when adding new device types
7. **Test both NMOS and PMOS** against NGSPICE: single OP, DC sweep, inverter VTC, inverter transient

### NN Model Rules (LEVEL=73)

1. **Jacobian consistency is mandatory** — gm/gds MUST be `torch.autograd.grad(id, V)`, never independent predictions. Without this, NR diverges in multi-device circuits.
2. **PMOS source-relative frame** — Shift all voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`). Training uses Vs=0; in CMOS, PMOS Vs=VDD.
3. **Training range covers NR overshoot** — Margin of +/-VDD beyond operating range, not just +/-0.1V
4. **Voltage clamping** — Clip inputs to training range to prevent extrapolation garbage
5. **Signed-log normalization** — `sign(x) * log10(|x|/floor)` preserves sign across 14-decade range
6. **TSMC asymmetric L** — NMOS L=16nm, PMOS L=20nm; TechConfig uses `L_nmos`/`L_pmos`
7. **ASAP7 modelcard name mapping** — Parser auto-maps netlist names to `nmos_rvt`/`pmos_rvt`

---

## References
- **ngspice** - Physics equation verification
- **Xyce** - Architecture patterns for device/solver separation
- **Shichman-Hodges Model** - Level 1 MOSFET compact model
- **BSIM-CMG** - FinFET compact model (LEVEL=72), integrated via PyCMG
- **ASAP7** - https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7.git
- **PyCMG** - https://github.com/ShenShan123/PyCMG.git

## Project Structure Notes

### Important Path References
- **PyCMG Location**: `/home/shenshan/pycmg-wrapper` (standalone, 21 device variants)
- **BSIM-CMG OSDI Binary**: `/home/shenshan/pycmg-wrapper/build-deep-verify/osdi/bsimcmg.osdi`
- **Modelcards**: `/home/shenshan/pycmg-wrapper/tech_model_cards/` (ASAP7: `ASAP7/`, TSMC: `TSMC{5,7,12,16}/naive/`)
- **Results Output**: `results/<circuit_name>/<analysis_type>/` (`.lis`, `.csv`, `.png`)
- **Examples**: `examples/` (50+ netlists)
- **Test Results**: `tests/verify_*_results/` (generated, not tracked in git)

## Other Tips
* **Start every complex task in plan mode:**
    * Pour your energy into the plan for 1-shot the implementation.
    * The moment something goes sideways, just switch back to plan mode and re-plan. Don't keep pushing.
    * Enter plan mode for verification steps, not just for the build.
* **Update CLAUDE.md:**
    * After every correction, update your CLAUDE.md so you don't make that mistake again.
* **Never be lazy:**
    * Never be lazy in writing the code and running tests.
    * Do NOT use any simplified equations or self-defined CMG models as reference, ALWAYS use simulation results as ground truth for comparison.
* Use subagents.
    * Use a second agent to review the plan as a staff engineer.
    * If you want to try multiple solutions, use multiple subagents, git commit to different branches. Roll back and to the main branch and create new branch when the subagent find it's a dead end.
* Enable the "Explanatory" or "Learning" output style in /config to explain the *why* behind its changes.
