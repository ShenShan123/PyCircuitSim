# Transient Analysis Enhancement Summary (2026-02-12)

## Overview
Implemented several improvements to transient analysis convergence for BSIM-CMG (FinFET) circuits with lower Vdd (1.2V).

## Changes Made

### 1. New Netlists Created
- `examples/bsimcmg_inverter_vdd12.sp` - Vdd=1.2V inverter
- `examples/bsimcmg_inverter_vdd12_slow.sp` - Vdd=1.2V with slower transitions
- `examples/bsimcmg_inverter_vdd15_slow.sp` - Vdd=1.5V with slower transitions
- `examples/bsimcmg_inverter_vdd12_very_slow.sp` - Vdd=1.2V with very slow transitions
- `examples/bsimcmg_inverter_vdd12_dc.sp` - DC sweep verification

### 2. Solver Enhancements (pycircuitsim/solver.py)

#### TransientSolver.__init__() - New Parameters:
- `use_gmin_stepping` (bool, default=True) - Enable Gmin stepping
- `gmin_initial` (float, default=1e-8) - Initial Gmin value
- `gmin_final` (float, default=1e-12) - Final Gmin value
- `gmin_steps` (int, default=5) - Number of Gmin stepping steps
- `use_pseudo_transient` (bool, default=True) - Enable pseudo-transient initialization
- `pseudo_transient_steps` (int, default=3) - Number of initial timesteps with pseudo-capacitance
- `pseudo_transient_cap` (float, default=1e-12) - Artificial capacitance value

#### New Helper Methods:
- `_add_pseudo_capacitors()` - Add pseudo-capacitors from nodes to ground
- `_remove_pseudo_capacitors()` - Clean up pseudo-capacitors after initialization
- `_apply_gmin_stepping()` - Apply Gmin stepping to MNA matrix

#### Newton-Raphson Improvements (_solve_timestep_newton):
- **Relaxed tolerance**: Changed from 1e-9 to 1e-6 for transient (fast-switching circuits)
- **Increased max_iterations**: Changed from 100 to 200
- **Adaptive damping**: Enhanced algorithm that:
  - Reduces damping when stuck or oscillating (improvement_ratio > 0.9)
  - Increases damping when making good progress (improvement_ratio < 0.5)
  - Forces damping (0.5) for large deltas (>= 1.0V)
  - No damping for small deltas (< 0.1V)
- **Oscillation detection**: Tracks voltage history and detects oscillating behavior
- **Oscillation fallback**: If variance of last 3 iterations < 100mV, accept averaged solution

### 3. Simulation Updates (pycircuitsim/simulation.py)
- `run_transient()` now uses the new convergence features by default

## Test Results

### DC Sweep (Working ✅)
- BSIM-CMG inverter DC sweep works correctly with Vdd=1.2V
- File: `examples/bsimcmg_inverter_vdd12_dc.sp`
- Results: Proper inverter VTC with switching around Vdd/2

### Transient Analysis (Partial Success ⚠️)
- **Level-1 MOSFET**: Transient works, but produces only 3 time points (investigation needed)
- **BSIM-CMG**: Very slow due to Newton-Raphson hitting max_iterations (200) at switching points
- **Root Cause**: At critical operating points (V_in ≈ Vth), the Jacobian matrix becomes ill-conditioned
  - MOSFETs are in moderate inversion region
  - Conductances change rapidly with small voltage changes
  - Newton-Raphson oscillates between similar solutions

### Oscillation Detection Effectiveness
- The oscillation detection works (detects < 100mV variance)
- However, many timesteps have variance > 100mV, so fallback doesn't trigger often enough
- Even with 100mV threshold, simulation is extremely slow

## Known Issues

### 1. Transient Performance (Critical)
BSIM-CMG transient analysis is impractically slow for circuits with fast switching inputs:
- Each timestep in switching region: 200 iterations × ~complex MNA operations
- For 10ns simulation with 10ps steps: 1000 timesteps
- Estimated time: Several hours to complete

### 2. Timestep Count Issue
Transient analysis reports suspiciously low time point counts (e.g., "3 time points" for 10ns/10ps simulation).
This suggests a potential bug in:
- Time array initialization
- Loop bounds calculation
- Or the pseudo-capacitor removal logic

### 3. Oscillation Detection Threshold
Current threshold (100mV) may be too loose for precision analog simulation:
- Allows accepting solutions with ~10% error for Vdd=1.2V
- May cause waveform distortion

## Recommendations

### Immediate (Required for Practical Use)
1. **Implement Trap or Gear integration methods** - More stable than Backward Euler for stiff circuits
2. ** timestep control** - Automatically reduce timestep during fast transitions
3. **Damped Newton** - Add trust region or line search to prevent overshoot
4. **Continuation methods** - Use homotopy for initial guess at each timestep

### Short-term (Performance)
1. **Parasitic capacitances** - Implement MOSFET capacitance model (Cgs, Cgd, Cdb, Csb)
   - This provides physical damping during switching
   - Reduces need for artificial convergence tricks
2. **Analytic Jacobian** - BSIM-CMG derivatives may have numerical issues
3. **Event queue** - Only apply Newton-Raphson near switching events

### Long-term (Robustness)
1. **SPICE3-like algorithms** - Implement proven convergence methods from commercial simulators
2. **Hybrid DC-Transient** - Use DC solver for "quasi-static" timesteps
3. **Machine learning** - Learn good initial guesses from previous simulations

## Files Modified
- `/home/shenshan/NN_SPICE/pycircuitsim/solver.py` - TransientSolver enhancements
- `/home/shenshan/NN_SPICE/pycircuitsim/simulation.py` - run_transient() updates
- `/home/shenshan/NN_SPICE/examples/*.sp` - New test netlists

## Next Steps
1. Debug the "3 time points" issue in transient analysis
2. Implement adaptive timestep control
3. Add MOSFET capacitance model integration
4. Consider integrating with NGSPICE for BSIM-CMG transient verification
