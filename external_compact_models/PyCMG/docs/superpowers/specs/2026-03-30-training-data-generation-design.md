# Training Data Generation for NN Compact Models

**Date:** 2026-03-30
**Status:** Final

## Problem Statement

PyCMG provides accurate single-point BSIM-CMG evaluation via OSDI, but lacks a systematic way to generate large-scale training datasets for neural network compact models. Users must manually write loops over voltage, geometry, and temperature dimensions. There is no threshold-aware non-uniform sampling, no multi-technology orchestration, and no structured output format.

## Goals

1. Provide a composable sweep API in `pycmg/sweep.py` for generating DC training data across voltage, geometry, temperature, process variation, and technology dimensions.
2. Provide a one-call convenience wrapper `generate_dataset()` for common workflows.
3. Provide a CLI script `scripts/generate_training_data.py` for command-line dataset generation.
4. Extract the technology registry from `tests/conftest.py` into `pycmg/tech.py` for shared use.
5. Rewrite `README.md` to lead with data generation, fix stale paths, and include examples for ML workflows.

## Non-Goals

- Transient sweep (DC-only for this iteration)
- HDF5 or NumPy binary output (CSV only; users can convert)
- Parallel/multi-threaded evaluation (OSDI ctypes calls are not thread-safe)
- NGSPICE cross-validation of generated data (existing test suite covers correctness)
- Instance reuse optimization via `set_params()` (deferred; fresh Instance per config is correct and simple)

---

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `pycmg/tech.py` | Technology registry: `TechConfig`, `DeviceConfig`, `TECH_REGISTRY` |
| `pycmg/sweep.py` | Sweep engine: `SweepConfig`, `SweepResult`, core functions, convenience wrapper |
| `scripts/generate_training_data.py` | CLI entry point wrapping `generate_dataset()` |

### Modified Files

| File | Change |
|------|--------|
| `pycmg/__init__.py` | Export `generate_dataset`, `SweepConfig`, `SweepResult` |
| `pycmg/model.py` | Add `model_overrides` param to `Instance.__init__()` |
| `tests/conftest.py` | Thin wrapper importing from `pycmg/tech.py` |
| `README.md` | Complete rewrite |

### Data Flow

```
CLI args / Python API
        |
        v
   SweepConfig
        |
        v
  +--------------------------------------+
  | For each (tech, device, L,           |
  |  process_combo, NFIN, temp):         |
  |                                      |
  | 1. resolve_modelcard(device, tech, L)|
  | 2. Instance(..., model_overrides)    |
  | 3. find_threshold()                  | --> coarse Vg sweep -> max(gm) -> Vth
  | 4. build_voltage_grid()              | --> non-uniform Vg (dense near Vth) + uniform Vd
  | 5. eval_dc() loop                    | --> 17 outputs per (Vg, Vd, Ve) point
  +--------------------------------------+
        |
        v
   SweepResult
        |
        v
   to_csv() --> CSV files (28+N columns per row)
```

---

## Module Design: `pycmg/tech.py`

### `DeviceConfig`

```python
@dataclass
class DeviceConfig:
    """Configuration for a single device (one Vt flavor, one polarity)."""
    model_name: str              # e.g., "nmos_rvt", "nch_svt_mac"
    inst_params: dict            # Static-only params: {TFIN, DEVTYPE}. L and NFIN are swept.
    modelcard: str | None = None # Static modelcard for all L (ASAP7)
    pdk_device: str | None = None  # Device name in PDK (TSMC), e.g., "nch_svt_mac"
    _min_l: float | None = None  # Per-device min_l, auto-detected and cached

    def get_min_l(self, pdk_path: str | None = None) -> float:
        """Auto-detect and return the minimum gate length for this device.

        Detection is per-device because min_l varies across devices within the
        same technology (e.g., TSMC12 core nch_svt_mac=16nm vs I/O nch_18_mac=135nm).

        Strategy:
        - TSMC (pdk_device set): scan PDK for min(lmin) across this device's variants.
        - ASAP7 (modelcard set): parse L parameter from the modelcard.
        Result is cached after first call.
        """
        if self._min_l is not None:
            return self._min_l
        if self.pdk_device is not None and pdk_path is not None:
            self._min_l = _scan_pdk_device_min_l(pdk_path, self.pdk_device)
        elif self.modelcard is not None:
            self._min_l = _parse_modelcard_l(self.modelcard)
        else:
            raise RuntimeError(f"Cannot detect min_l for {self.model_name}")
        return self._min_l
```

**L and NFIN are NOT in `inst_params`** — both are swept dimensions. `inst_params` contains only truly static parameters:
- **TFIN**: fin thickness, physically constrained per technology
- **DEVTYPE**: device polarity indicator (1=NMOS, 0=PMOS)

**min_l is per-device, not per-technology.** Within the same TSMC node, different devices have vastly different minimum L:

| TSMC12 Device | min_l | Type |
|---------------|-------|------|
| nch_svt_mac | 16nm | Core |
| nch_hvt_mac | 16nm | Core |
| nch_18_mac | 135nm | I/O 1.8V |
| nch_hv18_mac | 600nm | High-voltage |

**Two modelcard strategies:**
- **ASAP7**: `modelcard` is set. Single file works for all L values (no length binning).
- **TSMC**: `pdk_device` is set. Modelcards are generated on-the-fly from the original PDK lib file using `_extract_model_params()` and `_find_length_variant()` from `pycmg/parser.py`. Any L that falls within a PDK variant's `[lmin, lmax]` range is valid.

### `TechConfig`

```python
@dataclass
class TechConfig:
    """Technology node with all available devices."""
    name: str                              # e.g., "ASAP7", "TSMC7"
    vdd: float                             # Supply voltage (V)
    tfin: float                            # Fin thickness (m)
    devices: dict[str, DeviceConfig]       # key = "nmos_rvt", "pmos_lvt", etc.
    pdk_path: str | None = None            # Path to original PDK lib file (TSMC only)

    def list_devices(self) -> list[str]:
        """Return all available device names."""
        return list(self.devices.keys())

    def get_device(self, name: str) -> DeviceConfig:
        """Look up device by name. Raises KeyError if not found."""
        return self.devices[name]
```

`_scan_pdk_device_min_l(pdk_path, device_name)` and `_parse_modelcard_l(modelcard_path)` are private helpers called only by `DeviceConfig.get_min_l()`.

### `resolve_modelcard()`

```python
def resolve_modelcard(device: DeviceConfig, tech: TechConfig, L: float,
                      cache_dir: str = "build/modelcards") -> str:
    """Resolve or generate a modelcard for the given device and L.

    For ASAP7 (static modelcard): returns device.modelcard directly.

    For TSMC (dynamic from PDK): generates a naive modelcard on-the-fly
    using generate_naive_tsmc_modelcard() from scripts/generate_naive_tsmc.py.
    Generated files are cached in cache_dir/{tech_name}/ to avoid
    regeneration on subsequent calls.

    Args:
        device: DeviceConfig for the target device
        tech: TechConfig for the target technology
        L: Gate length in meters
        cache_dir: Directory for caching generated modelcards

    Returns:
        Absolute path to the modelcard file

    Raises:
        RuntimeError: if no PDK variant matches the requested L
        ValueError: if neither modelcard nor pdk_device is configured
    """
    if device.modelcard is not None:
        return device.modelcard

    if device.pdk_device is not None and tech.pdk_path is not None:
        L_nm = int(L * 1e9)
        cache_path = f"{cache_dir}/{tech.name}/{device.pdk_device}_l{L_nm}nm.l"

        if not os.path.exists(cache_path):
            generate_naive_tsmc_modelcard(
                pdk_path=tech.pdk_path,
                model_type=device.pdk_device.split("_", 1)[0],
                device_type=device.pdk_device.split("_", 1)[1],
                L=L,
                output_path=cache_path,
                tech=tech.name,
            )
        return cache_path

    raise ValueError(f"No modelcard strategy for device {device.model_name}")
```

**Cache strategy:** Generated modelcards are cached in `build/modelcards/{tech}/` (same build directory used by OSDI compilation). Files persist across runs.

### Registry Constants

```python
TECH_REGISTRY: dict[str, TechConfig] = {
    "ASAP7": TechConfig(
        name="ASAP7", vdd=0.9, tfin=6.5e-9,
        pdk_path=None,
        # min_l auto-detected from modelcard L parameter = 21nm
        devices={
            "nmos_rvt": DeviceConfig(
                model_name="nmos_rvt",
                modelcard="modelcards/ASAP7/7nm_TT_160803.pm",
                inst_params={"TFIN": 6.5e-9, "DEVTYPE": 1},
            ),
            "pmos_rvt": DeviceConfig(
                model_name="pmos_rvt",
                modelcard="modelcards/ASAP7/7nm_TT_160803.pm",
                inst_params={"TFIN": 6.5e-9, "DEVTYPE": 0},
            ),
            "nmos_lvt": DeviceConfig(...),
            "pmos_lvt": DeviceConfig(...),
            "nmos_slvt": DeviceConfig(...),
            "pmos_slvt": DeviceConfig(...),
            "nmos_sram": DeviceConfig(...),
            "pmos_sram": DeviceConfig(...),
        }
    ),
    "TSMC5": TechConfig(
        name="TSMC5", vdd=0.65, tfin=6e-9,
        pdk_path="modelcards/TSMC5/cln5_1d2_sp_v1d2_2p2.l",
        # min_l auto-detected from PDK: 6nm (range: 6nm - 135nm)
        devices={
            "nmos_svt": DeviceConfig(
                model_name="nch_svt_mac",
                pdk_device="nch_svt_mac",
                inst_params={"TFIN": 6e-9, "DEVTYPE": 1},
            ),
            "pmos_svt": DeviceConfig(
                model_name="pch_svt_mac",
                pdk_device="pch_svt_mac",
                inst_params={"TFIN": 6e-9, "DEVTYPE": 0},
            ),
            "nmos_lvt": DeviceConfig(...),
            "pmos_lvt": DeviceConfig(...),
            "nmos_ulvt": DeviceConfig(...),
            "pmos_ulvt": DeviceConfig(...),
            "nmos_elvt": DeviceConfig(...),
            "pmos_elvt": DeviceConfig(...),
        }
    ),
    "TSMC7": TechConfig(
        name="TSMC7", vdd=0.75, tfin=6e-9,
        pdk_path="modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l",
        # min_l auto-detected from PDK: 8nm (range: 8nm - 240nm)
        devices={...}
    ),
    "TSMC12": TechConfig(
        name="TSMC12", vdd=0.80, tfin=6e-9,
        pdk_path="modelcards/TSMC12/cln12ffcll_1d8_sp_v1d0_2p4.l",
        # min_l auto-detected from PDK: 16nm (range: 16nm - 240nm)
        devices={...}
    ),
    "TSMC16": TechConfig(
        name="TSMC16", vdd=0.80, tfin=6e-9,
        pdk_path="modelcards/TSMC16/crn16ffcll_1d8_sp_v1d0_2p1.l",
        # min_l auto-detected from PDK: 16nm (range: 16nm - 240nm)
        devices={...}
    ),
}

def get_tech_config(name: str) -> TechConfig:
    """Look up technology by name. Raises KeyError if not found."""
    return TECH_REGISTRY[name]

def list_techs() -> list[str]:
    """Return all available technology names."""
    return list(TECH_REGISTRY.keys())
```

**L sweep with multipliers — example outputs (per-device min_l):**

| Technology | Device | Auto-detected min_l | l_multipliers=[1,2,3,4,5] | Valid L range |
|------------|--------|--------------------|-----------------------------|---------------|
| ASAP7 | nmos_rvt | 21nm | 21, 42, 63, 84, 105nm | any (no binning) |
| TSMC5 | nch_svt_mac | 6nm | 6, 12, 18, 24, 30nm | 6-135nm |
| TSMC5 | nch_12_mac | 54nm | 54, 108nm (3-5x out of range) | 54-135nm |
| TSMC7 | nch_svt_mac | 8nm | 8, 16, 24, 32, 40nm | 8-240nm |
| TSMC7 | nch_18_mac | 135nm | 135, 270nm (3-5x out of range) | 135-240nm |
| TSMC12 | nch_svt_mac | 16nm | 16, 32, 48, 64, 80nm | 16-240nm |
| TSMC12 | nch_hv18_mac | 600nm | 600nm only (2x out of range) | 600nm+ |

Each device auto-detects its own min_l. Out-of-range multiplied L values are skipped with a warning.

**Key design notes:**

- **Dynamic modelcard generation from PDK**: TSMC modelcards are generated on-the-fly from the original PDK lib file. The existing `_find_length_variant()` and `_extract_model_params()` in `pycmg/parser.py` handle the heavy lifting. Any L value within a PDK variant's `[lmin, lmax]` range is valid.
- **Caching**: Generated modelcards are cached in `build/modelcards/{tech}/` to avoid regeneration. The `generate_naive_tsmc_modelcard()` function from `scripts/generate_naive_tsmc.py` will be imported into `pycmg/tech.py`.
- **Per-device auto-detected min_l**: `DeviceConfig.get_min_l(pdk_path)` auto-detects via `_scan_pdk_device_min_l()` (TSMC) or `_parse_modelcard_l()` (ASAP7). Result is cached per-device. This is critical because min_l varies across devices within the same technology (e.g., TSMC12 core=16nm vs I/O=135nm vs HV=600nm).
- **L and NFIN excluded from `inst_params`**: Both are swept dimensions. `inst_params` = `{TFIN, DEVTYPE}` only. For non-sweep use (tests, single-point eval), pass directly to `Instance(params={"L": ..., "NFIN": ...})`.
- **TFIN = 6e-9 for all TSMC nodes**: Matches the verified test suite.
- **TSMC device naming**: Both matched (nmos_svt/pmos_svt) and historical (nmos_svt/pmos_lvt) pairings are available as separate device entries. The `tests/conftest.py` thin wrapper preserves historical pairings for backward compatibility.

---

## Module Design: `pycmg/sweep.py`

### `SweepConfig`

```python
@dataclass
class SweepConfig:
    """Configuration for a multi-dimensional DC sweep."""
    techs: list[str]                                    # ["ASAP7", "TSMC7"] or ["all"]
    devices: dict[str, list[str] | None] | None = None  # Per-tech device selection; None = all
    l_multipliers: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])
    nfins: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    temperatures: list[float] = field(default_factory=lambda: [233.15, 273.15, 300.15, 358.15, 398.15])
    vg_points: int = 50
    vd_points: int = 50
    ve_values: list[float] = field(default_factory=lambda: [0.0])
        # Body bias MAGNITUDE: 0.0 = no body bias (Ve=Vs for both NMOS/PMOS).
    process_vars: dict[str, list[float]] | None = None
        # Process parameter variations (model-level). Keys: OSDI param names.
        # Values: list of values to sweep. Swept as Cartesian product grid.
    dense_ratio: float = 0.6
    n_coarse: int = 30
```

**Design note on `l_multipliers` and `nfins`:**

Both L and NFIN are swept dimensions, excluded from `inst_params`:

- **L**: `L = DeviceConfig.get_min_l() * multiplier`. min_l is auto-detected per-device from the PDK (TSMC) or modelcard (ASAP7). For TSMC, `resolve_modelcard()` generates the correct modelcard per L. Out-of-range L values are skipped with a warning.
- **NFIN**: Swept directly from the `nfins` list. Default `[1, 2, 3]`.
- Both use `list[float]` to support fractional values.

### `SweepResult`

```python
@dataclass
class SweepResult:
    """Structured results from a DC sweep."""
    columns: list[str]                   # Column names
    data: list[list[str | float]]        # Row data (str for tech/device, float for rest)
    metadata: dict                       # Sweep config, timestamp, total rows, etc.
```

### Output Key Order and Column Assembly

```python
OUTPUT_KEYS: list[str] = [
    "id", "ig", "is", "ie", "ids",       # Currents (A)
    "qg", "qd", "qs", "qb",              # Charges (C)
    "gm", "gds", "gmb",                  # Derivatives (S)
    "cgg", "cgd", "cgs", "cdg", "cdd",   # Capacitances (F)
]

GEOM_COLUMNS: list[str] = ["tech", "device", "L", "NFIN", "TFIN", "temp_K"]
VOLTAGE_COLUMNS: list[str] = ["Vg", "Vd", "Vs", "Ve", "Vth"]

def build_all_columns(process_keys: list[str]) -> list[str]:
    """Without process_vars: 28 columns. With N process_vars: 28 + N columns."""
    return GEOM_COLUMNS + sorted(process_keys) + VOLTAGE_COLUMNS + OUTPUT_KEYS
```

### `find_threshold()`

```python
def find_threshold(inst: Instance, vdd: float, device_type: str = "nmos",
                   n_coarse: int = 30) -> float:
    """Find threshold voltage via coarse Vg sweep at Vd=Vdd/2.

    Returns the Vg MAGNITUDE (always positive) where |gm| is maximum.
    For NMOS, this is the actual Vg at threshold.
    For PMOS, this is |Vgs| at threshold (Vdd - Vg_actual).

    The returned value is in "magnitude space" — the same coordinate system
    used by build_voltage_grid() and build_nodes().
    """
```

**Algorithm:**
1. Fix Vd_mag = Vdd/2 (saturation bias)
2. Sweep Vg_mag from 0 to Vdd in `n_coarse` uniform steps
3. Convert Vg_mag to actual terminal voltages via `build_nodes()`
4. Call `inst.eval_dc()` at each point, extract `gm`
5. Return the Vg_mag where |gm| is maximum

**Key: returns magnitude, not actual voltage.** By returning magnitude, the value can be passed directly to `build_voltage_grid()` without any coordinate transformation for either NMOS or PMOS.

### `build_voltage_grid()`

```python
def build_voltage_grid(vdd: float, vth_mag: float, vg_points: int = 50,
                       vd_points: int = 50, dense_ratio: float = 0.6
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Build non-uniform Vg magnitude array (dense near Vth) and uniform Vd array.

    All values are in MAGNITUDE space (0 to Vdd). Use build_nodes() to convert
    to actual terminal voltages for NMOS/PMOS.
    """
```

**Algorithm:**
1. Dense region: `[vth_mag - 0.15*Vdd, vth_mag + 0.15*Vdd]`, clipped to `[0, Vdd]`
2. Allocate `int(vg_points * dense_ratio)` points in dense region (linspace)
3. Allocate remaining points uniformly across `[0, Vdd]`
4. Merge, sort, deduplicate
5. Vd: uniform linspace from 0 to Vdd, `vd_points` points

### `build_nodes()`

```python
def build_nodes(vg_mag: float, vd_mag: float, ve_mag: float,
                vdd: float, device_type: str) -> dict[str, float]:
    """Convert magnitude-space voltages to actual terminal voltages.

    NMOS: terminals directly correspond to magnitudes (Vs=0, Ve=ve_mag).
    PMOS: Vs=Vdd, Vg=Vdd-Vg_mag, Vd=Vdd-Vd_mag, Ve=Vdd-Ve_mag.
    """
    if device_type == "nmos":
        return {"g": vg_mag, "d": vd_mag, "s": 0.0, "e": ve_mag}
    else:
        return {"g": vdd - vg_mag, "d": vdd - vd_mag, "s": vdd, "e": vdd - ve_mag}
```

### `resolve_devices()`

```python
def resolve_devices(device_filter: dict[str, list[str] | None] | None,
                    tech_name: str, tech_config: TechConfig) -> list[str]:
    """Resolve which devices to sweep for a given technology.

    Supports glob matching (e.g., "nmos_*"). Missing devices skipped with warning.
    """
```

**Algorithm:**
1. If `device_filter is None`: return `tech_config.list_devices()`
2. If `tech_name not in device_filter` or value is `None`: return all devices
3. Otherwise, for each entry: glob-match or exact match. Skip missing with warning.
4. Return deduplicated, order-preserved list

**CLI mapping:** `--devices` flat list is applied to ALL specified techs. Missing devices per-tech are skipped with a warning.

### Process Parameter Variation

Process parameters (EOT, TOXP, NBODY, etc.) are **model-level** parameters in OSDI. To apply process overrides, the `Instance` class gets a new `model_overrides` parameter:

```python
class Instance:
    def __init__(self, model: Model, params=None, temperature=300.15,
                 model_overrides: dict[str, float] | None = None):
        ...
        # 1. Apply modelcard params (model-level)
        for key, val in model.modelcard_params.items():
            apply_param(desc, self._inst, model.model, key, val, True)

        # 2. Apply process variation overrides (model-level, overrides modelcard)
        if model_overrides:
            for key, val in model_overrides.items():
                apply_param(desc, self._inst, model.model, key, val, True)

        # 3. Apply instance params (instance-level: L, TFIN, NFIN, DEVTYPE)
        if params:
            for key, val in params.items():
                apply_param(desc, self._inst, model.model, key, val, False)
        ...
```

Order matters: model_overrides after modelcard params ensures overrides take precedence.

**Common process parameters for FinFET variation:**

| Parameter | Description | Typical TT value | Variation range |
|-----------|-------------|-------------------|-----------------|
| `eot` | Equivalent oxide thickness | ~1.0nm | +/-10% |
| `toxp` | Physical oxide thickness | ~2.1nm | +/-10% |
| `nbody` | Channel doping | ~1e22 m^-3 | +/-20% |
| `phig` | Gate work function | ~4.3 eV | +/-50mV |
| `nsd` | Source/drain doping | ~2e26 m^-3 | +/-10% |
| `hfin` | Fin height | ~32nm | +/-5% |

### `sweep_dc()`

```python
def sweep_dc(osdi_path: str, config: SweepConfig,
             verbose: int = 2) -> SweepResult:
    """Run a full multi-dimensional DC sweep.

    Args:
        osdi_path: Path to compiled .osdi binary
        config: SweepConfig defining all sweep dimensions
        verbose: 0=silent, 1=summary, 2=per-config progress

    Returns:
        SweepResult containing all sweep data
    """
```

**Core loop:**
```
resolved_techs = list(TECH_REGISTRY.keys()) if "all" in config.techs else config.techs

# Expand process grid (Cartesian product)
if config.process_vars:
    pkeys = sorted(config.process_vars.keys())
    process_combos = [dict(zip(pkeys, c))
                      for c in itertools.product(*(config.process_vars[k] for k in pkeys))]
else:
    pkeys, process_combos = [], [{}]

data = []

for each tech_name in resolved_techs:
    tech_config = get_tech_config(tech_name)
    device_list = resolve_devices(config.devices, tech_name, tech_config)

    for each device_name in device_list:
        device = tech_config.get_device(device_name)
        device_type = "nmos" if "nmos" in device_name else "pmos"

        # Compute lengths from per-device auto-detected min_l × multipliers
        min_l = device.get_min_l(tech_config.pdk_path)
        lengths = [min_l * m for m in config.l_multipliers]

        for each L in lengths:
            try:
                modelcard_path = resolve_modelcard(device, tech_config, L)
            except RuntimeError as e:
                if verbose >= 2:
                    print(f"  Warning: {tech_name} {device_name} L={L*1e9:.0f}nm: {e}, skipping")
                continue

            model = Model(osdi_path, modelcard_path, device.model_name)

            for each proc_combo in process_combos:
                for each nfin in config.nfins:
                    for each temp in config.temperatures:
                        params = {**device.inst_params, "L": L, "NFIN": nfin}
                        inst = Instance(model, params=params, temperature=temp,
                                        model_overrides=proc_combo or None)

                        # Two-pass threshold detection (per-config: Vth shifts with process)
                        vth_mag = find_threshold(inst, tech_config.vdd, device_type, config.n_coarse)
                        vg_arr, vd_arr = build_voltage_grid(
                            tech_config.vdd, vth_mag, config.vg_points,
                            config.vd_points, config.dense_ratio)

                        for ve in config.ve_values:
                            for vg_mag in vg_arr:
                                for vd_mag in vd_arr:
                                    nodes = build_nodes(vg_mag, vd_mag, ve, tech_config.vdd, device_type)
                                    result = inst.eval_dc(nodes)
                                    row = [
                                        tech_name, device_name,
                                        L, nfin, tech_config.tfin, temp,
                                        *[proc_combo.get(k, 0.0) for k in pkeys],
                                        nodes["g"], nodes["d"], nodes["s"], nodes["e"],
                                        vth_mag,
                                        *[result[k] for k in OUTPUT_KEYS],
                                    ]
                                    data.append(row)

                        if verbose >= 2:
                            proc_str = " ".join(f"{k}={v:.2e}" for k, v in proc_combo.items())
                            print(f"[{count}/{total}] {tech_name} {device_name} "
                                  f"L={L*1e9:.0f}nm NFIN={nfin:.0f} T={temp-273.15:.0f}C "
                                  f"{proc_str} ... Vth={vth_mag:.3f}V ... {n_points} points")

if verbose >= 1:
    print(f"Done. {len(data)} rows across {count} configurations.")

return SweepResult(columns=ALL_COLUMNS, data=data, metadata={...})
```

**Key design notes on sweep loop:**

- **Threshold re-detection per process combo**: `find_threshold()` runs inside the process loop because process variations (e.g., EOT, PHIG) shift Vth. This ensures the non-uniform voltage grid is centered correctly for each process point.
- **Model reuse across process combos**: The same `Model()` object (same modelcard) is reused for all process combos within a (tech, device, L) group. Process overrides are applied per-Instance via `model_overrides`.
- **Column order**: process param columns are inserted between geometry (temp_K) and voltage (Vg) columns, sorted alphabetically by parameter name for deterministic ordering.
- **Model caching**: For ASAP7, `resolve_modelcard()` returns the same static path for all L values. For TSMC, each L generates a different cached modelcard. The loop creates `Model()` inside the L loop for correctness with both cases.

### `to_csv()`

```python
def to_csv(results: SweepResult, output_dir: str,
           split_by: str = "tech") -> list[str]:
    """Write SweepResult to CSV files.

    Args:
        results: SweepResult from sweep_dc()
        output_dir: Output directory path
        split_by: "tech" (one per tech), "device" (one per tech+device), "none" (single file)

    Returns:
        List of written file paths
    """
```

**File naming:**
- `split_by="tech"`: `ASAP7_dc.csv`, `TSMC7_dc.csv`
- `split_by="device"`: `ASAP7_nmos_rvt_dc.csv`, `ASAP7_pmos_rvt_dc.csv`
- `split_by="none"`: `training_data_dc.csv`

**Float formatting:** Use `%.6e` for scientific notation. String columns (tech, device) written as-is.

### `generate_dataset()`

```python
def generate_dataset(
    osdi_path: str,
    techs: list[str] = ["all"],
    devices: dict[str, list[str] | None] | None = None,
    output_dir: str = "./training_data",
    l_multipliers: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0],
    nfins: list[float] = [1.0, 2.0, 3.0],
    temperatures: list[float] = [233.15, 273.15, 300.15, 358.15, 398.15],
    vg_points: int = 50,
    vd_points: int = 50,
    ve_values: list[float] = [0.0],
    process_vars: dict[str, list[float]] | None = None,
    dense_ratio: float = 0.6,
    split_by: str = "tech",
    verbose: int = 2,
) -> list[str]:
    """Generate DC training datasets. Returns list of CSV file paths.

    Convenience wrapper. For fine-grained control, use SweepConfig + sweep_dc() + to_csv().
    n_coarse (threshold detection resolution) is not exposed — use SweepConfig directly.
    """
    config = SweepConfig(
        techs=techs, devices=devices, l_multipliers=l_multipliers,
        nfins=nfins, temperatures=temperatures, vg_points=vg_points,
        vd_points=vd_points, ve_values=ve_values, process_vars=process_vars,
        dense_ratio=dense_ratio,
    )
    result = sweep_dc(osdi_path, config, verbose=verbose)
    return to_csv(result, output_dir, split_by=split_by)
```

---

## CLI Design: `scripts/generate_training_data.py`

```
usage: generate_training_data.py [-h] --osdi PATH
                                  [--tech TECH [TECH ...]]
                                  [--devices DEV [DEV ...]]
                                  [--list-devices]
                                  [--l-multipliers F [F ...]]
                                  [--nfins F [F ...]]
                                  [--temps C [C ...]]
                                  [--vg-points N] [--vd-points N]
                                  [--ve-values V [V ...]]
                                  [--process-var NAME=V1,V2,V3 ...]
                                  [--dense-ratio F]
                                  [--output-dir DIR]
                                  [--split-by {tech,device,none}]
                                  [--verbose | --quiet | --silent]
```

**Key behaviors:**
- `--l-multipliers` specifies L as multiples of auto-detected min_l. Out-of-range values skipped with warning.
- `--process-var` specifies process parameter variations (repeatable). Multiple flags create Cartesian product.
- `--temps` accepts Celsius, converts to Kelvin internally.
- `--devices nmos_*` supports glob-style matching.
- `--list-devices` prints available devices and exits.
- Default verbosity: verbose (all per-config progress).

**Examples:**
```bash
# Minimal: ASAP7, all defaults
python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi --tech ASAP7

# Multi-tech, specific devices
python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 TSMC7 --devices nmos_rvt pmos_rvt

# Custom multipliers
python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 TSMC7 --l-multipliers 1 2 3 4 5 6 7 8

# Process variation: sweep EOT and TOXP (3x3 = 9 combos per device config)
python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 \
    --process-var eot=0.9e-9,1.0e-9,1.1e-9 \
    --process-var toxp=1.8e-9,2.1e-9,2.4e-9

# Full control
python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi \
    --tech all --l-multipliers 1 2 3 4 5 --nfins 1 2 3 \
    --temps -40 0 27 85 125 --vg-points 100 --vd-points 100 \
    --output-dir ./my_data --split-by device --quiet

# List available devices
python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --list-devices
```

---

## CSV Output Schema

### Columns (28 + N_process_vars)

| # | Column | Type | Unit | Description |
|---|--------|------|------|-------------|
| 1 | tech | str | - | Technology name (ASAP7, TSMC5, ...) |
| 2 | device | str | - | Device name (nmos_rvt, pmos_lvt, ...) |
| 3 | L | float | m | Gate length |
| 4 | NFIN | float | - | Number of fins |
| 5 | TFIN | float | m | Fin thickness |
| 6 | temp_K | float | K | Temperature in Kelvin |
| 7..6+N | *(process vars)* | float | *(varies)* | Process params, sorted alphabetically |
| 7+N | Vg | float | V | Gate voltage (actual terminal) |
| 8+N | Vd | float | V | Drain voltage (actual terminal) |
| 9+N | Vs | float | V | Source voltage (actual terminal) |
| 10+N | Ve | float | V | Bulk/extended voltage (actual terminal) |
| 11+N | Vth | float | V | Detected threshold voltage (magnitude, always positive) |
| 12+N | id | float | A | Drain current |
| 13+N | ig | float | A | Gate current |
| 14+N | is | float | A | Source current |
| 15+N | ie | float | A | Bulk current |
| 16+N | ids | float | A | Drain-source current (Id - Is) |
| 17+N | qg | float | C | Gate charge |
| 18+N | qd | float | C | Drain charge |
| 19+N | qs | float | C | Source charge |
| 20+N | qb | float | C | Bulk charge |
| 21+N | gm | float | S | Transconductance |
| 22+N | gds | float | S | Output conductance |
| 23+N | gmb | float | S | Bulk transconductance |
| 24+N | cgg | float | F | Gate self-capacitance |
| 25+N | cgd | float | F | Gate-drain capacitance |
| 26+N | cgs | float | F | Gate-source capacitance |
| 27+N | cdg | float | F | Drain-gate capacitance |
| 28+N | cdd | float | F | Drain self-capacitance |

**Note on Vth column:** Stores threshold voltage in magnitude space (always positive). For PMOS, Vg_actual = Vdd - Vth.

### Dataset Size Estimates

| Configuration | Devices | Configs | Points/Config | Total Rows | ~CSV Size |
|---------------|---------|---------|---------------|------------|-----------|
| ASAP7, defaults | 8 | 8×5L×3N×5T = 600 | 2,500 | 1.5M | ~300MB |
| ASAP7, NMOS only | 4 | 300 | 2,500 | 750K | ~150MB |
| All 5 techs, defaults | ~36 | ~2,700 | 2,500 | 6.75M | ~1.35GB |
| ASAP7, high-res (100x100) | 8 | 600 | 10,000 | 6.0M | ~1.2GB |
| + 3×3 process variation | × | ×9 | × | ×9 | ×9 |

(Default: 5 L multipliers × 3 NFINs × 5 temps = 75 configs/device)

---

## README Rewrite Plan

### Top-Level Structure

```
# PyCMG -- BSIM-CMG Training Data Generator for Neural Network Compact Models

## What is PyCMG?
## Quick Start
  ### 1. Build OSDI Binary
  ### 2. Generate Your First Dataset
  ### 3. Load into Your ML Framework
## Installation
  ### Prerequisites
  ### Building the OSDI Binary
  ### Environment Variables
## Generating Training Data
  ### CLI Usage
  ### Python API: One-Liner
  ### Python API: Composable Pipeline
  ### Notebook Workflow
## Supported Technologies
  ### Base Technologies (Tier 1)
  ### Vt Variants (Tier 2)
## Output Format
  ### CSV Schema
  ### Dataset Size Estimates
  ### Non-Uniform Voltage Sampling
  ### Loading Data for Training
## Advanced Usage
  ### Single-Point Evaluation
  ### Temperature Sweeps
  ### Body Bias Effects
  ### NFIN Scaling
  ### Custom Voltage Grids
  ### PMOS Conventions
  ### Process Variation
## Verification Against NGSPICE
  ### Strategy
  ### Tolerance Specifications
  ### Running the Test Suite
  ### Test Coverage Summary
## Project Structure
## API Reference
  ### pycmg.sweep
  ### pycmg.model
  ### pycmg.parser
  ### pycmg.tech
## License
```

### Key Content Changes

1. **Lead with purpose**: "Generate physically accurate training data for NN compact models"
2. **Quick Start in 3 steps**: build, generate, load into PyTorch/TF
3. **Fix stale paths**: `pycmg/ctypes_host.py` → split modules, `tech_model_cards/` → `modelcards/`, remove `cpp/` reference
4. **ML framework snippets**: pandas, PyTorch Dataset, TensorFlow tf.data, scikit-learn
5. **Condense verification**: move from hero section to supporting section
6. **Updated project structure**: reflect current directory layout
7. **Process variation section**: explain `--process-var` CLI and `process_vars` API

---

## Testing Strategy

### Unit Tests for `pycmg/sweep.py`

Add `tests/test_sweep.py`:

1. **`test_find_threshold_nmos`**: Verify Vth is in reasonable range for ASAP7 NMOS (~0.2-0.5V)
2. **`test_find_threshold_pmos`**: Verify Vth detection works for PMOS (returns magnitude)
3. **`test_build_voltage_grid_density`**: Verify more points near Vth than away
4. **`test_build_voltage_grid_bounds`**: Verify grid covers [0, Vdd]
5. **`test_build_voltage_grid_dedup`**: Verify no duplicate points
6. **`test_sweep_dc_single_config`**: Sweep 1 tech, 1 device, 1 L, 1 NFIN, 1 temp; verify output shape
7. **`test_sweep_dc_pmos_voltages`**: Verify PMOS node voltages are inverted correctly
8. **`test_to_csv_split_by_tech`**: Verify file naming and content
9. **`test_generate_dataset_smoke`**: End-to-end with minimal config
10. **`test_device_selection`**: Verify device filtering with glob matching
11. **`test_resolve_modelcard_asap7`**: Verify ASAP7 returns static modelcard for any L
12. **`test_resolve_modelcard_tsmc_cached`**: Verify TSMC generates on first call, returns cached on second
13. **`test_resolve_modelcard_tsmc_invalid_l`**: Verify RuntimeError for L outside PDK ranges
14. **`test_process_variation`**: Verify model_overrides applied correctly, Vth shifts with EOT

### Integration with Existing Tests

- `tests/conftest.py` imports from `pycmg/tech.py` — verify existing 266 tests still pass
- No changes to existing test logic, only the import source
