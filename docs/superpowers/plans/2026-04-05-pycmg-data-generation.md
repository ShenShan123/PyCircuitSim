# PyCMG Data Generation Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all NN training data generation from `nn_model/` into PyCMG by extending `pycmg/sweep.py` with `.npz` output and NN-specific voltage grids, creating `pycmg/nn_config.py` + `pycmg/nn_generate.py`, and deleting `nn_model/data/generate.py` entirely. Use PyCMG's PDK-defined legal `(L, NFIN)` bins via `DeviceConfig.get_geometry_combos()` for geometry coverage, and extract process parameters on-the-fly from the original modelcards via `Model.modelcard_params` rather than hardcoding them in config.

**Architecture:** `sweep.py` gains three additions (NN output columns, extended voltage grid, `.npz` writer). `nn_generate.py` uses these primitives plus `find_threshold`/`build_nodes` from `sweep.py` to drive the NN-specific sweep loop. For each `(tech, variant, device_type)`, it enumerates PDK-legal `(L, NFIN)` combos, resolves an NFIN-aware modelcard per combo, creates a `Model` to extract the 12 process parameters on-the-fly from `model.modelcard_params`, then sweeps voltages and writes process params explicitly into the dataset's geometry columns. `nn_model/config.py` becomes a thin re-export shim; `nn_model/data/generate.py` is deleted. All existing consumers of `nn_model.config` keep working unchanged.

**Tech Stack:** Python 3.10, PyCMG submodule (pycmg/sweep.py, pycmg/model.py, pycmg/tech.py, pycmg/parser.py), numpy

**Breaking change — geometry format 14→15 columns:** Adding `L` as an explicit geometry feature (user requirement) and process params are now **per-bin accurate** (vary across L/NFIN bins within the same variant). New geometry layout: `[NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]`. NN input dimension: 18→19. All existing checkpoints must be retrained after data regeneration.

**Breaking change — dataset size increase:** TSMC PDKs define many (L, NFIN) bins per variant (e.g., TSMC7 has ~42 combos per device). The dataset will be significantly larger than the current 7-NFIN-per-variant approach, providing better geometry coverage at the cost of longer training time.

---

## Key Design Constraints

1. **Use `sweep.py` primitives.** `nn_generate.py` calls `find_threshold`, `build_voltage_grid`, `build_nodes` from `sweep.py`. Do not re-implement threshold detection or duplicate voltage-grid logic.
2. **Source-relative frame.** NN training always uses Vs=0, Vb=0 for both NMOS and PMOS. NMOS Vg ∈ [−VDD, 2·VDD]; PMOS Vg ∈ [−2·VDD, +VDD] (source-relative, negative to turn on).
3. **`sweep.py` backward compatibility.** All changes to `build_voltage_grid()` default to the current behavior (v_min=0.0, n_dense_mid=0, vth_center=None). Existing `sweep_dc()` is untouched.
4. **`nn_config.py` uses relative imports only.** `from .tech import TECH_REGISTRY`, never from `pycmg.__init__` (avoids loading-order issues).
5. **No shim for `nn_model/data/generate.py`.** File is deleted. Its one consumer (`verify_nn_leave_one_out.py`) has its import updated to `pycmg.nn_generate`.
6. **`sys.path` mutation stays in `nn_model/config.py`.** `pycircuitsim/` modules rely on it to make `pycmg` importable without setting sys.path themselves.
7. **Name collision:** `pycmg/sweep.py` exports `generate_dataset` (CSV); `nn_generate.py` has a different function with the same name. Do NOT re-export `nn_generate.generate_dataset` from `pycmg/__init__.py`. Callers import from `pycmg.nn_generate` directly.
8. **PDK-driven geometry.** For TSMC techs, enumerate legal `(L, NFIN)` combos from `DeviceConfig.get_geometry_combos(pdk_path)`. For ASAP7 (no PDK binning), use a fallback NFIN list `[2, 3, 5, 10, 15, 20, 24]` with the tech's single L value. Filter out `NFIN < 2` always (convergence failures documented in CLAUDE.md).
9. **On-the-fly process params.** Do NOT hardcode ProcessParams per variant. Instead: `resolve_modelcard(dev, tech, L, NFIN)` → `Model(modelcard_path)` → `extract_process_params(model.modelcard_params)`. This produces per-bin-accurate process params that match the actual modelcard used for evaluation.
10. **One Model per (L, NFIN) combo.** Because `model_overrides` writes to a shared buffer (CLAUDE.md: "Instance / Model Isolation"), each `(L, NFIN)` bin needs its own `Model()` created from the bin-specific modelcard. Do not reuse a single Model across bins.

---

## File Map

```
# New files
external_compact_models/PyCMG/pycmg/nn_config.py       ← config: ProcessParams, NNTechConfig, TECH_CONFIGS, extract_process_params()
external_compact_models/PyCMG/pycmg/nn_generate.py     ← sweep loop + .npz generation (PDK-driven L/NFIN)
external_compact_models/PyCMG/scripts/generate_nn_data.py  ← CLI entry point

# Modified files
external_compact_models/PyCMG/pycmg/sweep.py           ← +NN_OUTPUT_COLUMNS, +build_voltage_grid params, +save_npz
external_compact_models/PyCMG/pycmg/__init__.py         ← export nn_config symbols
nn_model/config.py                                      ← thin re-export shim (keep TrainConfig, paths)
nn_model/data/normalize.py                              ← handle 15-col geometry with L feature
tests/verify_nn_leave_one_out.py                        ← update import path

# Deleted files
nn_model/data/generate.py                               ← replaced by pycmg/nn_generate.py
```

---

## Task 1: Extend `pycmg/sweep.py`

Add three things: `NN_OUTPUT_COLUMNS`, extended `build_voltage_grid()`, and `save_npz()`.
No changes to `sweep_dc()`, `to_csv()`, or `SweepConfig`.

**Files:**
- Modify: `external_compact_models/PyCMG/pycmg/sweep.py`

- [ ] **Step 1: Add `NN_OUTPUT_COLUMNS` constant**

Insert after the existing `OUTPUT_KEYS` constant (around line 34):

```python
# NN training target columns — subset of OUTPUT_KEYS (13 of 17)
# Excludes ig, is, ie, ids which are not needed for circuit simulation.
NN_OUTPUT_COLUMNS: List[str] = [
    "id", "gm", "gds", "gmb",
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]
```

- [ ] **Step 2: Extend `build_voltage_grid()` signature**

Replace the current `build_voltage_grid()` function with an extended version that adds three new parameters with backward-compatible defaults:

```python
def build_voltage_grid(
    vdd: float,
    vth_mag: float,
    vg_points: int = 50,
    vd_points: int = 50,
    dense_ratio: float = 0.6,
    voltage_scale: float = 1.0,
    v_min: float = 0.0,
    n_dense_mid: int = 0,
    vth_center: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a non-uniform Vg grid (dense near threshold) and uniform Vd grid.

    Extends the original with three new parameters:
      v_min: Lower voltage bound (default 0.0 = existing behavior). Use -vdd
          for NN source-relative training data that covers NR overshoot.
      n_dense_mid: Extra dense points near the mid-supply crossing
          ((v_min + v_max) / 2). 0 = disabled (default).
      vth_center: Explicit signed threshold center for dense region. When None
          (default), uses ``vth_mag`` (positive, backward-compatible). Pass a
          negative value for PMOS source-relative frame (e.g., -0.2).

    Args:
        vdd: Nominal supply voltage (used for dense region width calculation).
        vth_mag: Threshold voltage magnitude (backward-compat; used when
            vth_center is None).
        vg_points: Total target Vg points (dense + sparse).
        vd_points: Number of Vd points (uniform).
        dense_ratio: Fraction of vg_points in dense threshold region.
        voltage_scale: Upper bound multiplier: v_max = vdd * voltage_scale.
        v_min: Lower voltage bound (signed). Default 0.0.
        n_dense_mid: Extra dense points near the mid-supply crossing.
        vth_center: Signed threshold center. None → use vth_mag.

    Returns:
        Tuple of (vg_array, vd_array), both sorted, no duplicates.
    """
    v_max = vdd * voltage_scale
    _vth = vth_center if vth_center is not None else vth_mag

    # Dense region: ±0.15·Vdd around threshold, clipped to [v_min, v_max]
    dense_lo = max(v_min, _vth - 0.15 * vdd)
    dense_hi = min(v_max, _vth + 0.15 * vdd)

    n_dense = int(vg_points * dense_ratio)
    n_sparse = vg_points - n_dense

    vg_dense = np.linspace(dense_lo, dense_hi, n_dense)
    vg_sparse = np.linspace(v_min, v_max, n_sparse)

    parts = [vg_dense, vg_sparse]
    if n_dense_mid > 0:
        mid = (v_min + v_max) / 2.0   # generalises to signed frame
        mid_lo = mid - 0.15 * vdd
        mid_hi = mid + 0.15 * vdd
        parts.append(np.linspace(mid_lo, mid_hi, n_dense_mid))

    vg_all = np.unique(np.concatenate(parts))

    vd_all = np.linspace(v_min, v_max, vd_points)

    return vg_all, vd_all
```

- [ ] **Step 3: Add `save_npz()` function**

Add after `to_csv()`:

```python
def save_npz(
    inputs: np.ndarray,
    geometry: np.ndarray,
    outputs: np.ndarray,
    output_path: "str | Path",
    metadata: "Optional[Dict[str, object]]" = None,
) -> None:
    """Save NN training arrays to a .npz file.

    The .npz layout is the contract between pycmg data generation and the
    nn_model training pipeline:
      inputs   (N, 4)  — source-relative terminal voltages [Vd, Vg, Vs, Vb]
      geometry (N, 15) — [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW,
                           CFS, TOXP, CGSL, UA, EU]
      outputs  (N, 13) — NN_OUTPUT_COLUMNS order

    Optional metadata keys are saved as ``meta_<key>`` arrays.

    Args:
        inputs: (N, 4) float64 array.
        geometry: (N, 15) float64 array.
        outputs: (N, 13) float64 array.
        output_path: Destination file path (``.npz`` extension added if absent).
        metadata: Optional dict of scalar/array metadata (tech name, VDD, etc.).
    """
    import os
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dict: Dict[str, object] = {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
    }
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v)
    np.savez(output_path, **save_dict)
    size_kb = os.path.getsize(output_path + ".npz" if not output_path.endswith(".npz") else output_path) / 1024
    print(f"  Saved {inputs.shape[0]} samples → {output_path} ({size_kb:.0f} KB)")
```

- [ ] **Step 4: Verify `build_voltage_grid` backward compat with existing tests**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -m pytest tests/test_api.py -v 2>&1 | tail -10
```
Expected: all `test_api.py` tests pass (no regressions from `sweep.py` changes).

- [ ] **Step 5: Verify new voltage grid covers expected ranges**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
from pycmg.sweep import build_voltage_grid, NN_OUTPUT_COLUMNS
# NMOS source-relative range [-VDD, 2*VDD]
vg, vd = build_voltage_grid(0.7, vth_mag=0.2, v_min=-0.7, voltage_scale=2.0, n_dense_mid=5, vg_points=71, vd_points=71)
print('NMOS vg:', vg.min(), '→', vg.max(), '  points:', len(vg))
assert vg.min() < -0.6 and vg.max() > 1.3
# PMOS source-relative range [-2*VDD, +VDD]
vg, vd = build_voltage_grid(0.7, vth_mag=0.2, v_min=-1.4, voltage_scale=1.0, vth_center=-0.2, vg_points=71, vd_points=71)
print('PMOS vg:', vg.min(), '→', vg.max(), '  points:', len(vg))
assert vg.min() < -1.3 and vg.max() > 0.6
print('NN_OUTPUT_COLUMNS:', NN_OUTPUT_COLUMNS)
print('sweep.py ok')
"
```
Expected: prints correct ranges, no assertion errors.

- [ ] **Step 6: Commit**

```bash
cd /home/shenshan/NN_SPICE
git add external_compact_models/PyCMG/pycmg/sweep.py
git commit -m "feat(sweep): add NN_OUTPUT_COLUMNS, extend build_voltage_grid, add save_npz"
```

---

## Task 2: Create `pycmg/nn_config.py`

**Files:**
- Create: `external_compact_models/PyCMG/pycmg/nn_config.py`

All imports are relative (`from .tech import ...`, `from .sweep import ...`). No reference to `nn_model`. `OSDI_PATH` and `PYCMG_DIR` are computed from `__file__` so the submodule is self-contained.

**Key design: no hardcoded ProcessParams.** `NNTechConfig` only stores the training VDD, variant name list, and an optional NFIN fallback for techs without PDK binning (ASAP7). Process parameters are extracted on-the-fly from modelcards during data generation via `extract_process_params()`.

- [ ] **Step 1: Write `pycmg/nn_config.py`**

```python
"""NN compact model configuration: process parameters and technology configs.

Defines ProcessParams, NNTechConfig, and the TECH_CONFIGS registry used by
nn_generate.py for training data generation.

Lives inside PyCMG so all data generation is self-contained in the submodule.
nn_model/config.py re-exports from here for backward compatibility.

Key design: no hardcoded ProcessParams per variant. Instead,
extract_process_params() reads them on-the-fly from the actual modelcard
for each (L, NFIN) bin during data generation. This ensures per-bin accuracy.

Geometry array layout (15 columns):
    [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]

Input feature vector (19 features = 4 voltages + 15 geometry):
    [Vd, Vg, Vs, Vb, log2(NFIN), L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW,
     CFS, TOXP, CGSL, UA, EU]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .tech import TECH_REGISTRY as _PYCMG_REGISTRY, resolve_modelcard, TechConfig
from .sweep import NN_OUTPUT_COLUMNS

# PyCMG root: two levels up from pycmg/nn_config.py  →  PyCMG/
PYCMG_DIR: Path = Path(__file__).resolve().parents[1]
OSDI_PATH: str = str(PYCMG_DIR / "build" / "osdi" / "bsimcmg.osdi")

DEFAULT_TEMPERATURE: float = 300.15  # 27 °C in Kelvin

# Process parameter names used as NN input features (12 total, order fixed)
PROCESS_PARAM_NAMES: List[str] = [
    "PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW",
    "CFS", "TOXP", "CGSL", "UA", "EU",
]

# NN output targets — re-exported from sweep for convenience
OUTPUT_COLUMNS: List[str] = NN_OUTPUT_COLUMNS

# Full NN input feature names (19 total)
INPUT_COLUMNS: List[str] = [
    "Vd", "Vg", "Vs", "Vb",   # 4 terminal voltages (source-relative)
    "NFIN", "L", "T",          # 3 geometric / operating-condition scalars
    *PROCESS_PARAM_NAMES,       # 12 process parameters
]

# Default NFIN values for techs without PDK binning (e.g., ASAP7).
# NFIN=1 excluded: causes convergence failure in certain TSMC variants.
DEFAULT_NFIN_VALUES: List[int] = [2, 3, 5, 10, 15, 20, 24]


@dataclass
class ProcessParams:
    """BSIM-CMG process parameters used as NN input features (12 total).

    These are NOT stored in config — they are extracted on-the-fly from
    modelcards via extract_process_params(). This dataclass is the canonical
    output format for the extraction.
    """
    phig: float   # Gate workfunction [V] — variant discriminator
    u0: float     # Low-field mobility [m²/(V·s)]
    vsat: float   # Saturation velocity [m/s]
    eot: float    # Equivalent oxide thickness [m]
    eta0: float   # DIBL coefficient
    cit: float    # Interface trap charge [F/m]
    rdsw: float   # S/D parasitic resistance [Ω·μm]
    cfs: float    # Fringing capacitance [F/m]
    toxp: float   # Physical oxide thickness [m]
    cgsl: float   # Gate-source overlap capacitance [F/m]
    ua: float     # Mobility degradation coefficient
    eu: float     # Mobility temperature exponent

    def as_array(self) -> List[float]:
        """Ordered list matching PROCESS_PARAM_NAMES (12 elements)."""
        return [self.phig, self.u0, self.vsat, self.eot,
                self.eta0, self.cit, self.rdsw,
                self.cfs, self.toxp, self.cgsl, self.ua, self.eu]

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(PROCESS_PARAM_NAMES, self.as_array()))


def extract_process_params(modelcard_params: Dict[str, float]) -> ProcessParams:
    """Extract the 12 NN process parameters from a parsed modelcard.

    Reads lowercase-keyed parameters from the modelcard (as returned by
    ``Model.modelcard_params`` or ``parse_modelcard().params``) and returns
    a ProcessParams instance. Missing parameters default to 0.0.

    This is the single source of truth for mapping modelcard → NN features.
    Both data generation and inference should use this function.

    Args:
        modelcard_params: Dict of modelcard parameters (lowercase keys),
            e.g., from ``Model(osdi, modelcard, name).modelcard_params``.

    Returns:
        ProcessParams with the 12 NN-relevant process parameters.
    """
    return ProcessParams(
        phig=modelcard_params.get("phig", 0.0),
        u0=modelcard_params.get("u0", 0.0),
        vsat=modelcard_params.get("vsat", 0.0),
        eot=modelcard_params.get("eot", 0.0),
        eta0=modelcard_params.get("eta0", 0.0),
        cit=modelcard_params.get("cit", 0.0),
        rdsw=modelcard_params.get("rdsw", 0.0),
        cfs=modelcard_params.get("cfs", 0.0),
        toxp=modelcard_params.get("toxp", 0.0),
        cgsl=modelcard_params.get("cgsl", 0.0),
        ua=modelcard_params.get("ua", 0.0),
        eu=modelcard_params.get("eu", 0.0),
    )


@dataclass
class NNTechConfig:
    """NN training config that wraps PyCMG's tech registry.

    Stores NN-specific data: training VDD, variant name list, temperature.
    Does NOT store L, NFIN, or ProcessParams — those come from the PDK:
      - Legal (L, NFIN) combos: ``DeviceConfig.get_geometry_combos(pdk_path)``
      - Process params: ``extract_process_params(model.modelcard_params)``

    For ASAP7 (no PDK binning), ``fallback_nfin_values`` provides a default
    NFIN list to pair with the tech's single L value.
    """
    pycmg_name: str
    vdd_train: float
    variant_names: List[str]          # e.g., ["svt", "lvt", "ulvt"]
    temperature: float = DEFAULT_TEMPERATURE
    default_variant: str = ""
    fallback_nfin_values: Optional[List[int]] = None  # for ASAP7 (no PDK binning)

    @property
    def name(self) -> str:
        return self.pycmg_name

    @property
    def pycmg_tech(self) -> TechConfig:
        return _PYCMG_REGISTRY[self.pycmg_name]

    @property
    def vdd(self) -> float:
        return self.vdd_train

    @property
    def tfin(self) -> float:
        return self.pycmg_tech.tfin

    def get_geometry_combos(
        self, device_type: str, variant: str,
    ) -> List[tuple[float, float]]:
        """Return legal (L, NFIN) combinations for this tech/device/variant.

        For TSMC techs: delegates to DeviceConfig.get_geometry_combos(pdk_path),
            which scans the PDK file for bin boundaries.
        For ASAP7 (no PDK binning): uses fallback_nfin_values with the
            device's single L (from modelcard).

        Returns:
            Sorted list of (L, NFIN) tuples with NFIN >= 2.
        """
        dev = self.pycmg_tech.get_device(f"{device_type}_{variant}")
        pdk_path = self.pycmg_tech.pdk_path

        if dev.pdk_device is not None and pdk_path is not None:
            # TSMC: scan PDK for legal combos
            combos = dev.get_geometry_combos(pdk_path=pdk_path)
        else:
            # ASAP7 or other: use fallback NFIN list with device min_l
            min_l = dev.get_min_l(pdk_path)
            nfin_list = self.fallback_nfin_values or DEFAULT_NFIN_VALUES
            combos = [(min_l, float(nfin)) for nfin in nfin_list]

        # Filter out NFIN < 2 (convergence failures: ETA0_i/U0_i go negative)
        return [(L, NFIN) for L, NFIN in combos if NFIN >= 2]

    def get_model_name(self, device_type: str, variant: str) -> str:
        """Get model name from PyCMG registry."""
        dev = self.pycmg_tech.get_device(f"{device_type}_{variant}")
        return dev.model_name

    def resolve_modelcard(
        self, device_type: str, variant: str,
        L: float, NFIN: Optional[float] = None,
    ) -> str:
        """Resolve the modelcard path for a specific (L, NFIN) bin.

        For ASAP7: returns the static modelcard (same for all L/NFIN).
        For TSMC: generates/caches an NFIN-aware naive modelcard.

        Returns:
            Absolute path to the modelcard file.
        """
        dev = self.pycmg_tech.get_device(f"{device_type}_{variant}")
        return resolve_modelcard(dev, self.pycmg_tech, L, NFIN)


# ---------------------------------------------------------------------------
# Technology configs (5 process nodes)
#
# No hardcoded ProcessParams — they are extracted on-the-fly from modelcards
# during data generation. Only training VDD, variant names, and fallback
# NFIN values (for ASAP7) are stored here.
# ---------------------------------------------------------------------------

ASAP7_CONFIG = NNTechConfig(
    pycmg_name="ASAP7",
    vdd_train=0.7,
    variant_names=["rvt", "lvt", "slvt", "sram"],
    default_variant="rvt",
    fallback_nfin_values=[2, 3, 5, 10, 15, 20, 24],  # no PDK binning
)

TSMC5_CONFIG = NNTechConfig(
    pycmg_name="TSMC5",
    vdd_train=0.65,
    variant_names=["svt", "lvt", "ulvt", "elvt"],
    default_variant="svt",
    # TSMC: legal (L, NFIN) combos from PDK, no fallback needed
)

TSMC7_CONFIG = NNTechConfig(
    pycmg_name="TSMC7",
    vdd_train=0.75,
    variant_names=["svt", "lvt", "ulvt"],
    default_variant="svt",
)

TSMC12_CONFIG = NNTechConfig(
    pycmg_name="TSMC12",
    vdd_train=0.80,
    variant_names=["svt", "lvt", "ulvt", "hvt", "lnvt"],
    default_variant="svt",
)

TSMC16_CONFIG = NNTechConfig(
    pycmg_name="TSMC16",
    vdd_train=0.80,
    variant_names=["svt", "lvt", "ulvt", "hvt", "lnvt"],
    default_variant="svt",
)

TECH_CONFIGS: Dict[str, NNTechConfig] = {
    "asap7": ASAP7_CONFIG,
    "tsmc5": TSMC5_CONFIG,
    "tsmc7": TSMC7_CONFIG,
    "tsmc12": TSMC12_CONFIG,
    "tsmc16": TSMC16_CONFIG,
}
```

- [ ] **Step 2: Smoke-test imports**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
from pycmg.nn_config import (
    TECH_CONFIGS, ProcessParams, NNTechConfig, OSDI_PATH,
    OUTPUT_COLUMNS, INPUT_COLUMNS, PROCESS_PARAM_NAMES,
    extract_process_params, DEFAULT_NFIN_VALUES,
)
print('techs:', list(TECH_CONFIGS.keys()))
print('INPUT_COLUMNS (19):', len(INPUT_COLUMNS), INPUT_COLUMNS)
print('OSDI_PATH:', OSDI_PATH)
assert len(INPUT_COLUMNS) == 19, f'Expected 19, got {len(INPUT_COLUMNS)}'
assert 'L' in INPUT_COLUMNS

# Test extract_process_params
dummy = extract_process_params({'phig': 4.5, 'u0': 0.03, 'vsat': 70000.0})
print('extract_process_params:', dummy.phig, dummy.u0, dummy.vsat)
assert dummy.eot == 0.0  # missing params default to 0.0

# Test get_geometry_combos for ASAP7 (fallback)
asap7 = TECH_CONFIGS['asap7']
combos = asap7.get_geometry_combos('nmos', 'rvt')
print(f'ASAP7 nmos_rvt combos ({len(combos)}): {combos[:3]}...')
assert len(combos) == 7  # fallback [2,3,5,10,15,20,24]
assert all(nfin >= 2 for _, nfin in combos)

# Test get_geometry_combos for TSMC7 (PDK-driven)
tsmc7 = TECH_CONFIGS['tsmc7']
combos = tsmc7.get_geometry_combos('nmos', 'svt')
print(f'TSMC7 nmos_svt combos ({len(combos)}): {combos[:5]}...')
assert len(combos) > 7  # should have many more than the old 7 NFIN values
assert all(nfin >= 2 for _, nfin in combos)

print('nn_config.py ok')
"
```
Expected: 5 techs, 19 input columns, ASAP7 has 7 combos (fallback), TSMC7 has many more (PDK-driven), no errors.

- [ ] **Step 3: Test on-the-fly process param extraction**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
from pycmg.nn_config import TECH_CONFIGS, extract_process_params
from pycmg.model import Model

# Test ASAP7: static modelcard, params constant across NFIN
asap7 = TECH_CONFIGS['asap7']
mc_path = asap7.resolve_modelcard('nmos', 'rvt', L=7e-9, NFIN=10)
model = Model('build/osdi/bsimcmg.osdi', mc_path, 'nmos_rvt', 'nmos_rvt')
proc = extract_process_params(model.modelcard_params)
print(f'ASAP7 nmos_rvt: PHIG={proc.phig:.4f}, U0={proc.u0:.4e}, VSAT={proc.vsat:.1f}')

# Test TSMC7: different (L, NFIN) bins may have different params
tsmc7 = TECH_CONFIGS['tsmc7']
combos = tsmc7.get_geometry_combos('nmos', 'svt')
# Compare process params from first and last bin
L1, NFIN1 = combos[0]
L2, NFIN2 = combos[-1]
mc1 = tsmc7.resolve_modelcard('nmos', 'svt', L=L1, NFIN=NFIN1)
mc2 = tsmc7.resolve_modelcard('nmos', 'svt', L=L2, NFIN=NFIN2)
m1 = Model('build/osdi/bsimcmg.osdi', mc1, 'nch_svt_mac', 'nch_svt_mac')
m2 = Model('build/osdi/bsimcmg.osdi', mc2, 'nch_svt_mac', 'nch_svt_mac')
p1 = extract_process_params(m1.modelcard_params)
p2 = extract_process_params(m2.modelcard_params)
print(f'TSMC7 svt bin ({L1*1e9:.0f}nm, NFIN={NFIN1:.0f}): PHIG={p1.phig:.4f}, U0={p1.u0:.4e}')
print(f'TSMC7 svt bin ({L2*1e9:.0f}nm, NFIN={NFIN2:.0f}): PHIG={p2.phig:.4f}, U0={p2.u0:.4e}')
# At least some params should differ between bins (variant overrides global differently)
print('Process params per-bin extraction: OK')
"
```
Expected: ASAP7 shows process params, TSMC7 shows potentially different params per bin. No errors.

- [ ] **Step 4: Commit**

```bash
git add external_compact_models/PyCMG/pycmg/nn_config.py
git commit -m "feat(PyCMG): add pycmg/nn_config.py — PDK-driven geometry combos + on-the-fly process param extraction"
```

---

## Task 3: Create `pycmg/nn_generate.py`

Uses `sweep.py` primitives: `find_threshold` (threshold detection), `build_voltage_grid` (extended, source-relative), `build_nodes` (absolute voltages for threshold detection only), `NN_OUTPUT_COLUMNS`, `save_npz`. The outer sweep loop iterates over PDK-legal `(L, NFIN)` combos and extracts process params on-the-fly from each bin's modelcard.

**Files:**
- Create: `external_compact_models/PyCMG/pycmg/nn_generate.py`

- [ ] **Step 1: Write `pycmg/nn_generate.py`**

```python
"""Generate NN training data (.npz) via PyCMG BSIM-CMG sweeps.

Replaces nn_model/data/generate.py. Uses sweep.py primitives for threshold
detection and voltage grid construction.

Key design: enumerates PDK-legal (L, NFIN) combos per variant and extracts
process parameters on-the-fly from the resolved modelcard for each bin.

Dataset coverage:
  Geometric:          (L, NFIN) from PDK bin boundaries (TSMC) or fallback list (ASAP7)
  Operating cond.:    Vd, Vg, Vs=0, Vb=0 (source-relative); T = tech.temperature
  Process params:     12 params per (L, NFIN) bin, extracted from modelcard on-the-fly
  Voltage range:      NMOS Vg/Vd ∈ [−VDD, 2·VDD]; PMOS ∈ [−2·VDD, +VDD]
  Outputs (13):       id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd

Geometry array layout (N, 15):
    [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# sweep.py primitives — this is where we reuse PyCMG infrastructure
from .sweep import (
    find_threshold,
    build_voltage_grid,
    build_nodes,
    NN_OUTPUT_COLUMNS,
    save_npz,
)
from .model import Model, Instance
from .nn_config import (
    OSDI_PATH,
    NNTechConfig,
    ProcessParams,
    PROCESS_PARAM_NAMES,
    TECH_CONFIGS,
    OUTPUT_COLUMNS,
    extract_process_params,
)


def _create_model_and_instance(
    tech: NNTechConfig,
    device_type: str,
    variant: str,
    L: float,
    NFIN: float,
) -> Tuple[Model, Instance, ProcessParams]:
    """Create a Model + Instance for a specific (L, NFIN) bin and extract process params.

    Each (L, NFIN) bin gets its own Model (because model_overrides write to a
    shared OsdiModel buffer — reusing across bins would corrupt earlier instances).

    The process parameters are extracted on-the-fly from the resolved modelcard's
    parsed parameters. For TSMC, this means per-bin-accurate params (the variant
    overlay differs per L/NFIN bin). For ASAP7, params are constant across all bins.

    Returns:
        (model, instance, process_params) tuple.
    """
    model_name = tech.get_model_name(device_type, variant)
    modelcard_path = tech.resolve_modelcard(device_type, variant, L, NFIN)

    model = Model(
        osdi_path=OSDI_PATH,
        modelcard_path=modelcard_path,
        model_name=model_name,
        model_card_name=model_name,
    )

    # Extract process params on-the-fly from the actual modelcard
    proc = extract_process_params(model.modelcard_params)

    devtype = 1 if device_type == "nmos" else 0
    inst = Instance(
        model=model,
        params={"L": L, "NFIN": NFIN, "TFIN": tech.tfin, "DEVTYPE": devtype},
        temperature=tech.temperature,
    )

    return model, inst, proc


def eval_single_point(
    inst: Instance,
    vd: float,
    vg: float,
    vs: float = 0.0,
    vb: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Evaluate one bias point. Returns None on failure or non-physical result."""
    try:
        result = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": vb})
        out = {k: result[k] for k in NN_OUTPUT_COLUMNS}
        if any(math.isnan(v) or math.isinf(v) for v in out.values()):
            return None
        if abs(out["id"]) > 1.0:   # > 1 A is non-physical for a FinFET
            return None
        return out
    except Exception as e:
        print(f"  WARNING: eval_dc failed at Vd={vd:.3f} Vg={vg:.3f}: {e}")
        return None


def generate_dataset(
    tech: NNTechConfig,
    device_type: str,
    variant_names: Optional[List[str]] = None,
    verbose: bool = True,
    vg_points: int = 71,
    vd_points: int = 71,
    dense_ratio: float = 0.6,
    n_dense_mid: int = 0,
) -> Dict[str, np.ndarray]:
    """Generate training data for one tech/polarity across all variants and legal (L, NFIN) bins.

    For each variant, enumerates PDK-legal (L, NFIN) combos via
    NNTechConfig.get_geometry_combos(). For each combo, resolves the
    NFIN-aware modelcard and extracts process params on-the-fly.

    Uses sweep.py's find_threshold + build_voltage_grid in source-relative frame.
    Geometry stores [NFIN, L, T, 12 process params] (15 columns).

    Args:
        tech: Technology configuration (VDD, variant names).
        device_type: "nmos" or "pmos".
        variant_names: Subset of variants to generate; None = all variants.
        verbose: Print per-(L, NFIN) progress.
        vg_points: Total Vg grid points (dense near Vth + sparse uniform).
        vd_points: Vd grid points (uniform).
        dense_ratio: Fraction of vg_points allocated to the dense Vth region.
        n_dense_mid: Extra dense points near mid-supply (useful for transient).

    Returns:
        Dict with keys "inputs" (N,4), "geometry" (N,15), "outputs" (N,13),
        "metadata".
    """
    vdd = tech.vdd
    is_pmos = device_type == "pmos"

    # Source-relative voltage range:
    #   NMOS: Vg/Vd ∈ [−VDD, 2·VDD]   (v_min=−VDD, voltage_scale=2.0)
    #   PMOS: Vg/Vd ∈ [−2·VDD, +VDD]  (v_min=−2·VDD, voltage_scale=1.0)
    v_min = -2.0 * vdd if is_pmos else -vdd
    voltage_scale = 1.0 if is_pmos else 2.0

    variants_to_use = variant_names or tech.variant_names
    if not variants_to_use:
        raise ValueError(f"No variants for {tech.name}. "
                         f"Available: {tech.variant_names}")

    all_inputs: List[np.ndarray] = []
    all_geometry: List[np.ndarray] = []
    all_outputs: List[np.ndarray] = []
    total_pts = 0
    failed_pts = 0

    for variant_name in variants_to_use:
        # Enumerate PDK-legal (L, NFIN) combos for this variant
        combos = tech.get_geometry_combos(device_type, variant_name)

        if verbose:
            print(f"\n--- {tech.name} {device_type} variant={variant_name} "
                  f"({len(combos)} geometry combos) ---")

        for L, NFIN in combos:
            # Create Model + Instance with NFIN-aware modelcard;
            # extract process params on-the-fly from the modelcard
            try:
                _model, inst, proc = _create_model_and_instance(
                    tech, device_type, variant_name, L, NFIN,
                )
            except Exception as e:
                if verbose:
                    print(f"  SKIP L={L*1e9:.1f}nm NFIN={NFIN:.0f}: {e}")
                continue

            # Build geometry row: [NFIN, L, T, 12 process params]
            geo = np.array([NFIN, L, tech.temperature] + proc.as_array())

            # --- Threshold detection via sweep.py (uses absolute voltages internally) ---
            vth_mag = find_threshold(inst, vdd, device_type)
            # Convert to source-relative threshold center
            vth_center = -vth_mag if is_pmos else vth_mag

            # --- Build source-relative voltage grid via extended build_voltage_grid ---
            vg_arr, vd_arr = build_voltage_grid(
                vdd=vdd,
                vth_mag=vth_mag,
                vg_points=vg_points,
                vd_points=vd_points,
                dense_ratio=dense_ratio,
                voltage_scale=voltage_scale,
                v_min=v_min,
                n_dense_mid=n_dense_mid,
                vth_center=vth_center,
            )

            t0 = time.time()
            bin_pts = 0

            # --- Main grid sweep (source-relative: Vs=0, Vb=0 always) ---
            for vg in vg_arr:
                for vd in vd_arr:
                    result = eval_single_point(inst, vd, vg, 0.0, 0.0)
                    if result is None:
                        failed_pts += 1
                        continue
                    all_inputs.append(np.array([vd, vg, 0.0, 0.0]))
                    all_geometry.append(geo.copy())
                    all_outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))
                    bin_pts += 1

            # --- Zero-bias anchor (3× weight) ---
            result = eval_single_point(inst, 0.0, 0.0)
            if result is not None:
                out_arr = np.array([result[k] for k in NN_OUTPUT_COLUMNS])
                for _ in range(3):
                    all_inputs.append(np.array([0.0, 0.0, 0.0, 0.0]))
                    all_geometry.append(geo.copy())
                    all_outputs.append(out_arr.copy())
                    bin_pts += 1

            # --- Deep cutoff anchors ---
            if is_pmos:
                cutoff_vg = [0.0, 0.05, 0.1]
                cutoff_vd = [0.0, -vdd / 2, -vdd]
            else:
                cutoff_vg = [-0.1, -0.05, 0.0]
                cutoff_vd = [0.0, vdd / 2, vdd]
            for vg_c in cutoff_vg:
                for vd_c in cutoff_vd:
                    result = eval_single_point(inst, vd_c, vg_c)
                    if result is not None:
                        all_inputs.append(np.array([vd_c, vg_c, 0.0, 0.0]))
                        all_geometry.append(geo.copy())
                        all_outputs.append(np.array([result[k] for k in NN_OUTPUT_COLUMNS]))
                        bin_pts += 1

            total_pts += bin_pts
            if verbose:
                elapsed = max(time.time() - t0, 0.001)
                print(f"  L={L*1e9:.1f}nm NFIN={NFIN:.0f}: {bin_pts} pts in {elapsed:.1f}s "
                      f"(vth={vth_mag:.3f}V, PHIG={proc.phig:.4f}, {bin_pts/elapsed:.0f} pts/s)")

    inputs = np.array(all_inputs, dtype=np.float64)    # (N, 4)
    geometry = np.array(all_geometry, dtype=np.float64)  # (N, 15)
    outputs = np.array(all_outputs, dtype=np.float64)  # (N, 13)

    if verbose:
        print(f"\nTotal: {total_pts} pts, {failed_pts} failed")
        print(f"Shapes — inputs: {inputs.shape}, geometry: {geometry.shape}, "
              f"outputs: {outputs.shape}")
        # Print process param ranges across all bins
        print(f"\nProcess parameter ranges across all (L, NFIN) bins:")
        for i, pname in enumerate(PROCESS_PARAM_NAMES):
            col = geometry[:, 3 + i]  # offset 3: skip NFIN, L, T
            unique_vals = np.unique(col)
            if len(unique_vals) <= 5:
                print(f"  {pname:>6s}: {unique_vals}")
            else:
                print(f"  {pname:>6s}: [{col.min():.4e}, {col.max():.4e}] ({len(unique_vals)} unique)")

    # Collect unique L values for metadata
    unique_L = np.unique(geometry[:, 1])

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
        "metadata": {
            "tech_name": tech.name,
            "device_type": device_type,
            "vdd": tech.vdd,
            "L_values": unique_L,
            "temperature": tech.temperature,
            "output_columns": np.array(NN_OUTPUT_COLUMNS),
            "variants": np.array(variants_to_use),
        },
    }


def generate_universal_dataset(
    device_type: str,
    verbose: bool = True,
    vg_points: int = 71,
    vd_points: int = 71,
    dense_ratio: float = 0.6,
    n_dense_mid: int = 0,
) -> Dict[str, np.ndarray]:
    """Concatenate per-tech datasets across all 5 technologies and all variants."""
    all_inputs, all_geometry, all_outputs = [], [], []

    for tech_name, tech in TECH_CONFIGS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {tech_name.upper()} — {len(tech.variant_names)} variants, "
                  f"VDD={tech.vdd}V")
            print(f"{'='*60}")
        data = generate_dataset(tech, device_type, verbose=verbose,
                                vg_points=vg_points, vd_points=vd_points,
                                dense_ratio=dense_ratio, n_dense_mid=n_dense_mid)
        all_inputs.append(data["inputs"])
        all_geometry.append(data["geometry"])
        all_outputs.append(data["outputs"])

    inputs = np.concatenate(all_inputs, axis=0)
    geometry = np.concatenate(all_geometry, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Universal {device_type.upper()}: {inputs.shape[0]:,} total points")
        # Print L coverage
        unique_L = np.unique(geometry[:, 1])
        print(f"Unique L values: {[f'{l*1e9:.1f}nm' for l in unique_L]}")
        # Print NFIN coverage
        unique_NFIN = np.unique(geometry[:, 0])
        print(f"Unique NFIN values: {unique_NFIN.astype(int).tolist()}")
        print(f"{'='*60}")

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
        "metadata": {
            "tech_name": "universal",
            "device_type": device_type,
            "vdd": 0.0,
            "L_values": np.unique(geometry[:, 1]),
            "temperature": 300.15,
            "output_columns": np.array(NN_OUTPUT_COLUMNS),
            "variants": np.array(list(TECH_CONFIGS.keys())),
        },
    }
```

- [ ] **Step 2: Smoke-test imports (no PyCMG eval yet — just import)**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
from pycmg.nn_generate import (
    _create_model_and_instance, eval_single_point,
    generate_dataset, generate_universal_dataset,
)
from pycmg.sweep import save_npz
print('nn_generate imports ok')
"
```
Expected: no import errors.

- [ ] **Step 3: Quick end-to-end smoke test (1 variant, tiny grid)**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
from pycmg.nn_config import TECH_CONFIGS, NNTechConfig
from pycmg.nn_generate import generate_dataset
from pycmg.sweep import save_npz
import numpy as np

# Test with ASAP7 (fallback NFIN, small NFIN list to keep it fast)
tech = TECH_CONFIGS['asap7']
# Patch to use only 1 NFIN for speed
tech_tiny = NNTechConfig(
    pycmg_name=tech.pycmg_name, vdd_train=tech.vdd_train,
    variant_names=['rvt'],
    default_variant='rvt',
    fallback_nfin_values=[10],  # single NFIN
)
data = generate_dataset(tech_tiny, 'nmos', vg_points=5, vd_points=5, verbose=True)
assert data['inputs'].shape[1] == 4,   f'inputs cols: {data[\"inputs\"].shape}'
assert data['geometry'].shape[1] == 15, f'geometry cols: {data[\"geometry\"].shape}'
assert data['outputs'].shape[1] == 13,  f'outputs cols: {data[\"outputs\"].shape}'
print(f'Shapes: inputs={data[\"inputs\"].shape}, geo={data[\"geometry\"].shape}, out={data[\"outputs\"].shape}')

# Check L is stored in geometry col 1
expected_L = 7e-9
assert np.allclose(data['geometry'][:, 1], expected_L), 'L column mismatch'
print('L stored correctly:', data['geometry'][0, 1])

# Check process params are non-zero (extracted from modelcard, not hardcoded)
phig_col = data['geometry'][:, 3]  # PHIG at col 3 (after NFIN, L, T)
assert np.all(phig_col > 0), f'PHIG should be positive: {phig_col[0]}'
print(f'PHIG from modelcard: {phig_col[0]:.4f}')

print('nn_generate smoke test PASSED')
" 2>&1
```
Expected: shapes are correct, L column contains 7e-9, PHIG is non-zero (from modelcard), PASSED.

- [ ] **Step 4: TSMC smoke test (PDK-driven combos)**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import generate_dataset
import numpy as np

# Test TSMC7 with 1 variant, tiny grid — should use PDK-defined (L, NFIN) combos
tech = TECH_CONFIGS['tsmc7']
data = generate_dataset(tech, 'nmos', variant_names=['svt'],
                        vg_points=5, vd_points=5, verbose=True)
print(f'Shapes: inputs={data[\"inputs\"].shape}, geo={data[\"geometry\"].shape}')

# Check multiple L values exist (PDK has multiple L bins)
unique_L = np.unique(data['geometry'][:, 1])
print(f'Unique L values: {[f\"{l*1e9:.1f}nm\" for l in unique_L]}')
assert len(unique_L) >= 1, 'Should have at least 1 L value'

# Check multiple NFIN values
unique_NFIN = np.unique(data['geometry'][:, 0])
print(f'Unique NFIN values: {unique_NFIN.astype(int).tolist()}')

# Check process params vary across bins (for TSMC, different bins have different params)
phig_unique = np.unique(data['geometry'][:, 3])
print(f'Unique PHIG values: {len(phig_unique)} (bins may share or differ)')

print('TSMC PDK-driven generation PASSED')
" 2>&1
```
Expected: multiple L values, multiple NFIN values, shows PDK-driven combos working.

- [ ] **Step 5: Commit**

```bash
git add external_compact_models/PyCMG/pycmg/nn_generate.py
git commit -m "feat(PyCMG): add pycmg/nn_generate.py — PDK-driven NN data generation with on-the-fly process params"
```

---

## Task 4: Create CLI `scripts/generate_nn_data.py`

**Files:**
- Create: `external_compact_models/PyCMG/scripts/generate_nn_data.py`

- [ ] **Step 1: Write the CLI**

```python
#!/usr/bin/env python3
"""Generate NN training data (.npz) from PyCMG BSIM-CMG sweeps.

Usage (from PyCMG root):
    python scripts/generate_nn_data.py --device both --universal
    python scripts/generate_nn_data.py --device nmos --tech asap7
    python scripts/generate_nn_data.py --device both --universal --n-dense-mid 30

Output goes to --data-dir (default: ../../nn_model/data/datasets/).
"""

import argparse
import sys
from pathlib import Path

# Allow running directly from PyCMG root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import generate_dataset, generate_universal_dataset
from pycmg.sweep import save_npz


def _default_data_dir() -> Path:
    """Default output: nn_model/data/datasets/ relative to project root."""
    pycmg_root = Path(__file__).resolve().parents[1]
    project_root = pycmg_root.parents[1]   # PyCMG/ → external_compact_models/ → project root
    return project_root / "nn_model" / "data" / "datasets"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NN training data (.npz) from PyCMG BSIM-CMG"
    )
    parser.add_argument("--device", choices=["nmos", "pmos", "both"], default="nmos")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()) + ["all"],
                        default="asap7")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names (default: all)")
    parser.add_argument("--universal", action="store_true",
                        help="Generate universal dataset across all techs/variants")
    parser.add_argument("--vg-points", type=int, default=71)
    parser.add_argument("--vd-points", type=int, default=71)
    parser.add_argument("--n-dense-mid", type=int, default=0,
                        help="Extra dense points near mid-supply (default: 0)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Output directory for .npz files")
    args = parser.parse_args()

    data_dir = args.data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    devices = ["nmos", "pmos"] if args.device == "both" else [args.device]
    sweep_kw = dict(vg_points=args.vg_points, vd_points=args.vd_points,
                    n_dense_mid=args.n_dense_mid, verbose=True)

    if args.universal:
        for device_type in devices:
            data = generate_universal_dataset(device_type, **sweep_kw)
            out = data_dir / f"universal_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"])
        return

    techs = list(TECH_CONFIGS.values()) if args.tech == "all" \
        else [TECH_CONFIGS[args.tech]]
    variant_names = None if args.variants == "all" \
        else [v.strip() for v in args.variants.split(",")]

    for tech in techs:
        for device_type in devices:
            data = generate_dataset(tech, device_type,
                                    variant_names=variant_names, **sweep_kw)
            out = data_dir / f"{tech.name.lower()}_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test CLI (`--help`)**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python scripts/generate_nn_data.py --help
```
Expected: prints usage without error.

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/PyCMG/scripts/generate_nn_data.py
git commit -m "feat(PyCMG): add scripts/generate_nn_data.py CLI for NN .npz data generation"
```

---

## Task 5: Update `pycmg/__init__.py`

Export NN config types. Do NOT export `nn_generate.generate_dataset` (name collision with `sweep.generate_dataset`).

**Files:**
- Modify: `external_compact_models/PyCMG/pycmg/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

```python
from .model import Model, Instance
from .parser import parse_modelcard, parse_number_with_suffix, scan_pdk_geometry_combos
from .sensitivity import compute_sensitivity, SensitivityResult
from .sweep import generate_dataset, SweepConfig, SweepResult, NN_OUTPUT_COLUMNS, save_npz
from .nn_config import (
    ProcessParams, NNTechConfig,
    TECH_CONFIGS, OUTPUT_COLUMNS, INPUT_COLUMNS, PROCESS_PARAM_NAMES,
    OSDI_PATH, PYCMG_DIR, extract_process_params,
)
# nn_generate is NOT re-exported here: its generate_dataset would shadow
# sweep.generate_dataset. Import from pycmg.nn_generate directly.

__all__ = [
    # Core API
    "Model", "Instance",
    "parse_modelcard", "parse_number_with_suffix", "scan_pdk_geometry_combos",
    "compute_sensitivity", "SensitivityResult",
    "generate_dataset", "SweepConfig", "SweepResult",
    # sweep additions
    "NN_OUTPUT_COLUMNS", "save_npz",
    # NN config
    "ProcessParams", "NNTechConfig",
    "TECH_CONFIGS", "OUTPUT_COLUMNS", "INPUT_COLUMNS", "PROCESS_PARAM_NAMES",
    "OSDI_PATH", "PYCMG_DIR", "extract_process_params",
]
```

- [ ] **Step 2: Smoke-test**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -c "
import pycmg
print('TECH_CONFIGS:', list(pycmg.TECH_CONFIGS.keys()))
print('INPUT_COLUMNS:', pycmg.INPUT_COLUMNS)
print('extract_process_params:', pycmg.extract_process_params)
print('__init__ ok')
"
```
Expected: 5 techs, 19 input columns.

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/PyCMG/pycmg/__init__.py
git commit -m "feat(PyCMG): export NN config symbols from pycmg.__init__"
```

---

## Task 6: Update `nn_model/config.py` to a thin re-export shim

Remove all moved definitions. Keep: `sys.path` injection, path constants (`DATA_DIR`, `CHECKPOINT_DIR`), `TrainConfig`, backward compat aliases. Add `INPUT_COLUMNS` re-export (now 19 features including L).

**Note:** `NNVariantConfig` no longer exists in `pycmg/nn_config.py`. The backward-compat alias `VariantConfig` is removed. Any tests using it must be updated.

**Files:**
- Modify: `nn_model/config.py`

- [ ] **Step 1: Rewrite `nn_model/config.py`**

```python
"""NN compact model configuration.

Process params, tech configs, output/input columns, and OSDI path are now
owned by PyCMG (pycmg/nn_config.py) and re-exported here for backward
compatibility. Path constants and training hyperparameters remain here.

Note: INPUT_COLUMNS now includes "L" (19 features total, up from 18).
Existing checkpoints trained on 18-feature data are incompatible and must
be retrained after regenerating datasets with the new format.

Note: ProcessParams are no longer hardcoded per variant. They are extracted
on-the-fly from modelcards via extract_process_params(). NNVariantConfig
has been removed — variants are identified by name strings.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_MODEL_DIR = PROJECT_ROOT / "nn_model"
CHECKPOINT_DIR = NN_MODEL_DIR / "checkpoints"
DATA_DIR = NN_MODEL_DIR / "data" / "datasets"

# ── Make pycmg importable ─────────────────────────────────────────────────────
# Required: pycircuitsim/parser.py and pycircuitsim/models/mosfet_nn.py import
# from nn_model.config without setting sys.path themselves.
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
_PYCMG_PYPATH = str(PYCMG_DIR)
if _PYCMG_PYPATH not in sys.path:
    sys.path.insert(0, _PYCMG_PYPATH)

# ── Re-export everything from pycmg.nn_config ────────────────────────────────
from pycmg.nn_config import (  # noqa: E402
    OSDI_PATH,
    DEFAULT_TEMPERATURE,
    PROCESS_PARAM_NAMES,
    ProcessParams,
    NNTechConfig,
    TECH_CONFIGS,
    OUTPUT_COLUMNS,
    INPUT_COLUMNS,       # now 19 features (includes "L")
    extract_process_params,
    DEFAULT_NFIN_VALUES,
)

# Backward compat alias used by test files
TechConfig = NNTechConfig


# ── Training hyperparameters (NN-project-specific) ───────────────────────────
@dataclass
class TrainConfig:
    """Training hyperparameters — not part of data generation."""
    batch_size: int = 1024
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    trunk_hidden: int = 128
    trunk_layers: int = 3
    head_hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 50
    w_id: float = 1.0
    w_gm: float = 0.5
    w_gds: float = 0.5
    w_gmb: float = 0.3
    w_charges: float = 0.5
    w_caps: float = 0.3
    w_zero_bias: float = 5.0
```

- [ ] **Step 2: Verify all existing consumers still import correctly**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python -c "
from nn_model.config import (
    TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, OSDI_PATH,
    ProcessParams, PROCESS_PARAM_NAMES, OUTPUT_COLUMNS, INPUT_COLUMNS,
    TrainConfig, DATA_DIR, extract_process_params,
)
print('TECH_CONFIGS:', list(TECH_CONFIGS.keys()))
print('INPUT_COLUMNS (19):', len(INPUT_COLUMNS))
assert len(INPUT_COLUMNS) == 19
assert 'L' in INPUT_COLUMNS
print('TechConfig alias:', TechConfig.__name__)
print('extract_process_params:', extract_process_params)
print('nn_model.config shim ok')
"
```
Expected: 5 techs, 19 input columns, no errors.

- [ ] **Step 3: Commit**

```bash
git add nn_model/config.py
git commit -m "refactor(nn_model): slim config.py to re-export shim backed by pycmg.nn_config (no hardcoded ProcessParams)"
```

---

## Task 7: Update `nn_model/data/normalize.py` for 15-column geometry

The geometry array grows from 14 to 15 columns by inserting `L` at position 1. `_build_combined_input()` must handle the new layout `[NFIN, L, T, process_params...]`.

**Files:**
- Modify: `nn_model/data/normalize.py`

- [ ] **Step 1: Update `_build_combined_input()` in `normalize.py`**

Replace the current method body:

```python
def _build_combined_input(
    self,
    inputs: np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    """Combine voltage inputs with geometry into a single feature vector.

    Geometry column layouts (backward-compatible):
      (N,  2): [NFIN, T]                         — legacy
      (N,  3): [NFIN, T, PHIG]                   — Phase 13
      (N,  9): [NFIN, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW]  — 7-param
      (N, 14): [NFIN, T, <12 process params>]     — 12-param (old universal)
      (N, 15): [NFIN, L, T, <12 process params>]  — 12-param + L (current)

    Returns:
      (N,  6): inputs + [log2(NFIN), T]
      (N,  7): inputs + [log2(NFIN), T, PHIG]
      (N, 13): inputs + [log2(NFIN), T, 7 params]
      (N, 18): inputs + [log2(NFIN), T, 12 params]   (old)
      (N, 19): inputs + [log2(NFIN), L, T, 12 params] (current)
    """
    nfin_log = np.log2(np.clip(geometry[:, 0], 1.0, None))

    if geometry.shape[1] == 15:
        # Current format: [NFIN, L, T, 12 process params]
        L_col      = geometry[:, 1]
        temperature = geometry[:, 2]
        proc_params = geometry[:, 3:]   # (N, 12)
        return np.column_stack([inputs, nfin_log, L_col, temperature, proc_params])

    # Legacy formats: [NFIN, T, ...]
    temperature = geometry[:, 1]
    if geometry.shape[1] >= 9:
        proc_params = geometry[:, 2:]   # (N, 7) or (N, 12)
        return np.column_stack([inputs, nfin_log, temperature, proc_params])
    elif geometry.shape[1] >= 3:
        phig = geometry[:, 2]
        return np.column_stack([inputs, nfin_log, temperature, phig])
    return np.column_stack([inputs, nfin_log, temperature])
```

Also update the docstring of `fit()` and `normalize_inputs()` to list `(N, 15)` as a valid geometry shape.

- [ ] **Step 2: Verify normalizer handles new geometry shape**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python -c "
import numpy as np
from nn_model.data.normalize import Normalizer

N = 100
rng = np.random.default_rng(0)
inputs   = rng.uniform(-0.7, 1.4, (N, 4))
# New 15-col geometry: [NFIN, L, T, 12 process params]
geometry = np.column_stack([
    np.full(N, 10.0),      # NFIN
    np.full(N, 7e-9),      # L
    np.full(N, 300.15),    # T
    rng.normal(0, 1, (N, 12)),  # 12 process params (dummy)
])
outputs = rng.uniform(-1e-4, 1e-4, (N, 13))

norm = Normalizer()
norm.fit(inputs, geometry, outputs)
combined = norm.normalize_inputs(inputs, geometry)
print('Combined shape:', combined.shape)   # expect (100, 19)
assert combined.shape == (N, 19), f'Expected (N,19), got {combined.shape}'
print('normalize.py 15-col geometry ok')
"
```
Expected: combined shape (100, 19).

- [ ] **Step 3: Verify legacy 14-col geometry still works (backward compat)**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python -c "
import numpy as np
from nn_model.data.normalize import Normalizer

N = 50
rng = np.random.default_rng(1)
inputs   = rng.uniform(-0.7, 1.4, (N, 4))
# Old 14-col geometry: [NFIN, T, 12 process params]
geometry = np.column_stack([
    np.full(N, 10.0),
    np.full(N, 300.15),
    rng.normal(0, 1, (N, 12)),
])
outputs = rng.uniform(-1e-4, 1e-4, (N, 13))

norm = Normalizer()
norm.fit(inputs, geometry, outputs)
combined = norm.normalize_inputs(inputs, geometry)
assert combined.shape == (N, 18), f'Expected (N,18), got {combined.shape}'
print('Legacy 14-col geometry still works:', combined.shape)
"
```
Expected: (50, 18) — old format still handled.

- [ ] **Step 4: Commit**

```bash
git add nn_model/data/normalize.py
git commit -m "feat(normalize): handle 15-col geometry [NFIN, L, T, process_params] → 19-feature input"
```

---

## Task 8: Delete `nn_model/data/generate.py` and update `verify_nn_leave_one_out.py`

**Files:**
- Delete: `nn_model/data/generate.py`
- Modify: `tests/verify_nn_leave_one_out.py`

- [ ] **Step 1: Delete `nn_model/data/generate.py`**

```bash
cd /home/shenshan/NN_SPICE
git rm nn_model/data/generate.py
```

- [ ] **Step 2: Update the import in `verify_nn_leave_one_out.py`**

Line 100-102 currently reads:
```python
from nn_model.data.generate import (
    generate_dataset, create_pycmg_instance,
)
```

Replace with:
```python
from pycmg.nn_generate import (
    generate_dataset, eval_single_point,
)
```

Note: `create_pycmg_instance` is replaced by `_create_model_and_instance` (private). If the test needs to create individual instances, use the public API through `generate_dataset()` or call `_create_model_and_instance` directly (accept the private API).

- [ ] **Step 3: Verify the test file imports cleanly**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python -c "
import sys
sys.path.insert(0, 'external_compact_models/PyCMG')
# Simulate the import chain verify_nn_leave_one_out.py uses
from pycmg.nn_generate import generate_dataset, eval_single_point
from nn_model.config import TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, DATA_DIR, OSDI_PATH, PROCESS_PARAM_NAMES, TrainConfig
print('verify_nn_leave_one_out imports ok')
"
```
Expected: no import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/verify_nn_leave_one_out.py
git commit -m "refactor: remove nn_model/data/generate.py; update verify_nn_leave_one_out import to pycmg.nn_generate"
```

---

## Task 9: Integration tests

- [ ] **Step 1: Cross-project import smoke test**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python -c "
# Simulate every unique import pattern found across the codebase
from nn_model.config import TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, OSDI_PATH, PROCESS_PARAM_NAMES, OUTPUT_COLUMNS, INPUT_COLUMNS, TrainConfig, DATA_DIR, extract_process_params
from pycmg.nn_config import TECH_CONFIGS as TC2, ProcessParams, NNTechConfig
from pycmg.nn_generate import generate_dataset, eval_single_point
from pycmg.sweep import NN_OUTPUT_COLUMNS, save_npz, build_voltage_grid
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase  # imports PROCESS_PARAM_NAMES internally

assert list(TC2.keys()) == list(TECH_CONFIGS.keys())
assert len(INPUT_COLUMNS) == 19
assert 'L' in INPUT_COLUMNS
print('All cross-project imports: OK')
"
```
Expected: no errors.

- [ ] **Step 2: Verify PyCMG unit tests still pass**

```bash
cd /home/shenshan/NN_SPICE/external_compact_models/PyCMG
conda run -n pycircuitsim python -m pytest tests/test_api.py -v 2>&1 | tail -15
```
Expected: all tests pass (existing sweep.py tests not broken by additions).

- [ ] **Step 3: Verify BSIM-CMG simulator tests still pass**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python tests/verify_bsimcmg_op.py && \
conda run -n pycircuitsim python tests/verify_bsimcmg_dc.py
```
Expected: same results as before (no regression in simulator).

- [ ] **Step 4: Verify PDK combo enumeration for all techs**

```bash
cd /home/shenshan/NN_SPICE
conda run -n pycircuitsim python -c "
from pycmg.nn_config import TECH_CONFIGS

for tech_name, tech in TECH_CONFIGS.items():
    print(f'\n{tech_name.upper()}:')
    for variant in tech.variant_names:
        for dtype in ['nmos', 'pmos']:
            combos = tech.get_geometry_combos(dtype, variant)
            L_vals = sorted(set(L for L, _ in combos))
            NFIN_vals = sorted(set(int(N) for _, N in combos))
            print(f'  {dtype}_{variant}: {len(combos)} combos, '
                  f'L={[f\"{l*1e9:.0f}nm\" for l in L_vals]}, '
                  f'NFIN={NFIN_vals}')
"
```
Expected: ASAP7 shows 7 combos per device (fallback), TSMC techs show many combos from PDK bins.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "test: confirm pycmg data generation migration — PDK-driven geometry + on-the-fly process params"
```

---

## Task 10: Update CLAUDE.md

- [ ] **Step 1: Update data generation command in project `CLAUDE.md`**

Replace the existing data generation command in the Quick Start section with:
```markdown
# Generate universal data (PyCMG is now the canonical data generator)
# Uses PDK-defined (L, NFIN) bins for geometry coverage and extracts
# process parameters on-the-fly from modelcards (per-bin accurate)
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal

# Optional: add --n-dense-mid 30 for transient accuracy
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal --n-dense-mid 30
```

Update the Status section to note:
- Geometry format is now 15-col `[NFIN, L, T, 12 process params]` (breaking change)
- NN input dimension: 19 features (up from 18); existing checkpoints require retraining
- Data generation fully moved to PyCMG
- L/NFIN combos from PDK bin boundaries (TSMC) or fallback list (ASAP7)
- Process parameters extracted on-the-fly from modelcards (per-bin accurate, not hardcoded)
- Dataset is significantly larger due to full PDK bin coverage

- [ ] **Step 2: Update `external_compact_models/PyCMG/CLAUDE.md`**

Add to the Directory Structure:
```
pycmg/nn_config.py          # NN training config (ProcessParams, NNTechConfig, extract_process_params)
pycmg/nn_generate.py        # NN .npz data generation using PDK-driven (L, NFIN) combos
scripts/generate_nn_data.py # CLI: generates universal_nmos.npz / universal_pmos.npz
```

Add to the Design Principles & Known Constraints:
```
### NN Data Generation
- Legal (L, NFIN) combos come from `DeviceConfig.get_geometry_combos(pdk_path)` for TSMC, or a fallback NFIN list for ASAP7.
- Process parameters are extracted on-the-fly from the resolved modelcard via `extract_process_params(model.modelcard_params)`. They are NOT hardcoded in config.
- Each (L, NFIN) bin gets its own `Model()` instance to avoid shared-buffer corruption (see "Instance / Model Isolation" constraint).
- NFIN < 2 is always filtered out (convergence failures documented above).
```

- [ ] **Step 3: Commit docs**

```bash
git add CLAUDE.md external_compact_models/PyCMG/CLAUDE.md
git commit -m "docs: update CLAUDE.md for PyCMG data generation migration — PDK-driven geometry + on-the-fly process params"
```
