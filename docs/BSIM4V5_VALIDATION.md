# BSIM4V5 Model Validation

This document describes the validation plan for PyCircuitSim's BSIM4V5 implementation.

## Validation Status

**Current Status: Pending ngspice installation**

The verification script has been created but requires ngspice to be installed for execution.

## Validation Plan

### 1. Test Circuits

We will validate BSIM4V5 implementation using the following test circuits:

#### CMOS Inverter (DC Sweep)
- **File**: `examples/inverter_bsim4_dc.sp`
- **Purpose**: Verify voltage transfer characteristic (VTC)
- **Metrics**:
  - Switching threshold voltage
  - Noise margins
  - Output voltage levels (VOL, VOH)

#### NMOS Common-Source Amplifier
- **File**: `examples/nmos_amplifier_bsim4.sp`
- **Purpose**: Verify transistor amplification behavior
- **Metrics**:
  - Voltage gain
  - Operating point stability
  - Small-signal parameters (gm, gds)

#### Inverter Chain (Multi-Stage)
- **File**: `examples/inverter_stage_bsim4_dc.sp`
- **Purpose**: Verify signal restoration and propagation
- **Metrics**:
  - Stage-by-stage VTC consistency
  - Signal level restoration

### 2. Reference Simulator

We will use **ngspice** as the reference simulator for validation:

- **Why ngspice?**
  - Open-source and widely used
  - Implements BSIM4V5 (Level 54) standard
  - Well-validated against industry simulators
  - Python-agnostic (no risk of sharing bugs)

- **Installation**:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ngspice

  # macOS
  brew install ngspice

  # Or download from: http://ngspice.sourceforge.net/
  ```

### 3. Verification Metrics

For each test circuit, we will compare:

| Metric | Acceptance Criteria |
|--------|---------------------|
| Mean Absolute Error (MAE) | < 1 mV (excellent), < 10 mV (good) |
| Root Mean Square Error (RMSE) | < 5 mV (excellent), < 20 mV (good) |
| Maximum Error | < 50 mV (excellent), < 100 mV (good) |
| Mean Relative Error | < 1% (excellent), < 5% (good) |

### 4. Verification Script

Run the automated verification script:

```bash
python tests/verify_bsim4_accuracy.py
```

The script will:
1. Run PyCircuitSim simulation
2. Run ngspice simulation
3. Compare results numerically
4. Generate comparison plots
5. Report accuracy metrics

### 5. Expected Validation Results

Based on the BSIM4V5 specification and PTM 45nm model parameters:

#### Inverter VTC
- Switching threshold: ~0.5V (for 1.0V Vdd)
- VOL < 0.1V
- VOH > 0.9V
- Noise margins > 0.2V

#### Amplifier Gain
- Small-signal gain: 5-20× (depending on bias)
- Linear operation range: > 0.3V

### 6. Known Limitations

Current implementation limitations that may affect validation:

1. **Line Continuations**: Multi-line `.model` definitions may not parse correctly
2. **Parameter Coverage**: Only core BSIM4V5 parameters are implemented
3. **Temperature Effects**: Operating at fixed temperature (300K)
4. **Noise**: No noise modeling

### 7. Next Steps

**TODO**: Complete the following validation steps:

1. [ ] Install ngspice
2. [ ] Run verification script: `python tests/verify_bsim4_accuracy.py`
3. [ ] Review comparison plots
4. [ ] Document any discrepancies
5. [ ] Fix any bugs found during validation
6. [ ] Add additional test cases if needed

### 8. Validation Report Template

Once validation is complete, we will create a report with:

```
# BSIM4V5 Validation Report

## Date: [DATE]

## Test Circuits
1. Inverter DC Sweep
2. NMOS Amplifier
3. Inverter Chain

## Results Summary

| Circuit | MAE (mV) | RMSE (mV) | Max Error (mV) | Status |
|---------|----------|-----------|----------------|--------|
| Inverter | X.XX | X.XX | X.XX | ✓/✗ |
| Amplifier | X.XX | X.XX | X.XX | ✓/✗ |
| Chain | X.XX | X.XX | X.XX | ✓/✗ |

## Issues Found
- [List any discrepancies or bugs]

## Recommendations
- [Any improvements needed]
```

## References

- [BSIM4V5 Manual](https://bsim.berkeley.edu/wp-content/uploads/2021/11/BSIM450_Manual_2015.pdf)
- [PTM CMOS 45nm Models](https://ptm.asu.edu/)
- [ngspice Documentation](http://ngspice.sourceforge.net/docs.html)
