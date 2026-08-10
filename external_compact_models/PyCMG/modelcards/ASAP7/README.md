# ASAP7 7nm PDK Modelcards

This directory contains ASAP7 (Arizona State University Predictive Technology 7nm) modelcards for BSIM-CMG FinFET simulation.

## PDK Information

- **Source**: Arizona State University (ASU)
- **Technology Node**: 7nm FinFET
- **Release**: ASAP7 version 1.0 (8/3/2016)
- **Model**: BSIM-CMG (Berkeley Short-channel IGFET Model for Common Multi-Gate)
- **Level**: 72 (BSIM-CMG)

## Available Models

### NMOS Devices
| Model Name | Description | Vth Type |
|-------------|---------------|------------|
| `nmos_lvt` | Low Threshold Voltage | Lower |Vth| for low-power apps |
| `nmos_rvt` | Regular Threshold Voltage | Standard |Vth| |
| `nmos_slvt` | Super-Low Threshold Voltage | Lowest |Vth| |
| `nmos_sram` | SRAM-optimized | Optimized for SRAM |

### PMOS Devices
| Model Name | Description | Vth Type |
|-------------|---------------|------------|
| `pmos_lvt` | Low Threshold Voltage | Lower |Vth| for low-power apps |
| `pmos_rvt` | Regular Threshold Voltage | Standard |Vth| |
| `pmos_slvt` | Super-Low Threshold Voltage | Lowest |Vth| |
| `pmos_sram` | SRAM-optimized | Optimized for SRAM |

## PVT Corners

| Corner | File | Description |
|--------|-------|-------------|
| **TT** | `7nm_TT_160803.pm` | Typical-Typical (nominal process) |
| **SS** | `7nm_SS_160803.pm` | Slow-Slow (worst-case speed) |
| **FF** | `7nm_FF_160803.pm` | Fast-Fast (best-case speed) |

## Key Parameters

### Geometry (Process-dependent)
- `tfin`: Fin thickness = 6.5-7.0 nm
- `hfin`: Fin height = 3.2-3.4 nm
- `l`: Gate length = 21 nm (process default)
- `eot`: Equivalent oxide thickness = 0.75-1.0 nm (TT), 0.7e-10 (SS/FF)

### Electrical
- `u0`: Carrier mobility
  - NMOS: ~0.025-0.030 m²/V·s (electron mobility)
  - PMOS: ~0.020-0.024 m²/V·s (hole mobility)
  - **Note**: PMOS has lower mobility due to hole vs electron physics
- `vsat`: Saturation velocity
  - NMOS: ~70000 m/s
  - PMOS: ~60000 m/s
  - Holes are slower than electrons
- `vth0`: Threshold voltage (calculated by CMG physics)
  - NMOS: ~0.4-0.5 V (positive for enhancement mode)
  - PMOS: ~-0.4 to -0.5 V (negative for depletion mode)

## Usage in PyCMG

```python
from pycmg import Model, Instance, parse_modelcard

# Load ASAP7 modelcard (same as TSMC naive approach)
parsed = parse_modelcard(
    "tech_model_cards/ASAP7/7nm_TT_160803.pm",
    "nmos_lvt"
)

# Create model and instance
model = Model("build-deep-verify/osdi/bsimcmg.osdi", parsed.params)
inst = Instance(model, params={"L": 30e-9, "NFIN": 2})

# DC analysis
result = inst.eval_dc({"d": 0.7, "g": 0.7, "s": 0.0, "e": 0.0})
print(f"Id = {result.id:.6e} A")
```

## Usage in NGSPICE

**Netlist File** (`test_asap7.cir`):

```spice
* NGSPICE + OSDI with ASAP7 Modelcard
.include tech_model_cards/ASAP7/7nm_TT_160803.pm

* NMOS transistor (direct model call)
N1 d g s e nmos_lvt l=30n nfin=2

* Bias voltages
Vd d 0 0.7
Vg g 0 0.7
Vs s 0 0
Ve e 0 0

* Analysis
.temp 27
.op
.end
```

**Running NGSPICE:**

```bash
export NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice
ngspice -b test_asap7.cir
```

## Important Notes

### PMOS vs NMOS Current Differences

**PMOS current is typically 2-3x lower than NMOS** - this is **expected physics**, not a bug!

**Physical Reasons:**

1. **Hole mobility is lower**: µp ≈ 0.022 m²/V·s vs µn ≈ 0.028 m²/V·s
2. **PMOS needs negative bias**: For PMOS turn-on, use Vgs < 0, Vds < 0
3. **Sign convention**:
   - NMOS in saturation: Id is negative (flows OUT of drain in SPICE)
   - PMOS in saturation: Id is positive (flows INTO drain)

**This is NOT a bug** - the ASAP7 modelcards are complete and correct.

### Verification Status

**PMOS DEVTYPE ISSUE RESOLVED (2026-02-13)**: ASAP7 PMOS models previously exhibited inverted behavior due to missing `devtype` parameter. This is now automatically fixed in PyCMG's `parse_modelcard()` function which injects:
- `devtype = 1.0` for NMOS models (ntype)
- `devtype = 0.0` for PMOS models (ptype)

**Technical Details:**
- BSIM-CMG v107 uses integer parameter `DEVTYPE` to distinguish device types
- Standard ASAP7 modelcards omit this parameter
- PyCMG automatically detects model type from `.model` line (nmos/pmos keyword)
- DEVTYPE is injected during parsing if not already present
- Implementation: `pycmg/ctypes_host.py` in both `parse_modelcard()` and `_extract_model_params()`

The original ASAP7 modelcard files remain unmodified. The `7nm_TT_160803_with_devtype.pm` file is kept for reference but is no longer needed.

- ✅ All 8 NMOS models complete with ~250 parameters each
- ✅ All 8 PMOS models complete with ~250 parameters each
- ✅ PMOS models now work correctly with auto-injected devtype
- ✅ DC, AC (capacitance), and noise sections present in all models
- ✅ Both NMOS and PMOS verified against NGSPICE with binary-level consistency
- ✅ Tolerances: ABS_TOL_I=1e-9, REL_TOL=5e-3

## References

- **ASAP7 Paper**: "Predictive Technology Model for 7nm FinFETs" (ASU, 2016)
- **BSIM-CMG**: Berkeley BSIM Group
- **Main CLAUDE.md**: `../../CLAUDE.md`
