# Training Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a composable DC sweep API that generates training datasets for NN compact models across voltage, geometry, temperature, process variation, and technology dimensions.

**Architecture:** New `pycmg/tech.py` provides a technology registry with per-device min_l auto-detection and on-the-fly modelcard generation from PDK files. New `pycmg/sweep.py` provides the sweep engine with non-uniform voltage sampling (dense near Vth), process variation support, and CSV output. `Instance` gets a `model_overrides` parameter for process variation. A CLI script wraps the API.

**Tech Stack:** Python 3.8+, NumPy, ctypes (OSDI), argparse. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-30-training-data-generation-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pycmg/tech.py` | Create | `DeviceConfig`, `TechConfig`, `resolve_modelcard()`, `TECH_REGISTRY` |
| `pycmg/sweep.py` | Create | `SweepConfig`, `SweepResult`, `find_threshold()`, `build_voltage_grid()`, `build_nodes()`, `resolve_devices()`, `sweep_dc()`, `to_csv()`, `generate_dataset()` |
| `pycmg/model.py` | Modify | Add `model_overrides` param to `Instance.__init__()` |
| `pycmg/__init__.py` | Modify | Export `generate_dataset`, `SweepConfig`, `SweepResult` |
| `scripts/generate_training_data.py` | Create | CLI wrapper with argparse |
| `tests/test_sweep.py` | Create | Unit tests for sweep module |
| `tests/test_tech.py` | Create | Unit tests for tech module |
| `tests/conftest.py` | Modify | Thin wrapper importing from `pycmg/tech.py` |
| `README.md` | Rewrite | Data-generation-focused documentation |

---

## Task 1: Add `model_overrides` to `Instance.__init__()`

This is a small, isolated change that enables process variation. Do it first so the rest of the code can use it.

**Files:**
- Modify: `pycmg/model.py:169-195`
- Test: `tests/test_api.py` (add test) or `tests/test_tech.py` (new file)

- [ ] **Step 1: Write failing test**

Create `tests/test_tech.py` with a test that passes `model_overrides` to `Instance`:

```python
"""Tests for technology registry and model_overrides support."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
ASAP7_MODELCARD = ROOT / "modelcards" / "ASAP7" / "7nm_TT_160803.pm"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
def test_instance_model_overrides():
    """model_overrides should shift device behavior (e.g., changing EOT shifts Id)."""
    from pycmg import Model, Instance

    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    nodes = {"d": 0.45, "g": 0.45, "s": 0.0, "e": 0.0}
    params = {"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0}

    # Baseline: no overrides
    inst_base = Instance(model, params=params)
    result_base = inst_base.eval_dc(nodes)

    # Override EOT (thicker oxide -> less current)
    inst_thick = Instance(model, params=params, model_overrides={"eot": 1.5e-9})
    result_thick = inst_thick.eval_dc(nodes)

    # Thicker oxide should reduce drain current
    assert abs(result_thick["id"]) < abs(result_base["id"]), \
        f"Thicker EOT should reduce Id: base={result_base['id']:.3e}, thick={result_thick['id']:.3e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tech.py::test_instance_model_overrides -v`
Expected: FAIL — `Instance() got an unexpected keyword argument 'model_overrides'`

- [ ] **Step 3: Implement model_overrides in Instance.__init__()**

In `pycmg/model.py`, modify `Instance.__init__()`:

```python
def __init__(self, model: Model, params: Optional[Dict[str, float]] = None,
             temperature: float = 300.15,
             model_overrides: Optional[Dict[str, float]] = None) -> None:
    self._model = model
    self._inst = OsdiInstance(model.descriptor)
    self._temperature = temperature
    if temperature < 200.0:
        warnings.warn(
            f"Temperature {temperature} K is very low (< 200 K). "
            f"Did you pass Celsius instead of Kelvin? "
            f"Use temp_K = temp_C + 273.15 to convert.",
            stacklevel=2,
        )
    self._sim = OsdiSimulation()
    self._connected_terminals = int(model.descriptor.num_terminals)
    # 1. Apply modelcard params (model-level)
    for key, val in model.modelcard_params.items():
        apply_param(model.descriptor, self._inst, model.model, key, val, True)
    # 2. Apply process variation overrides (model-level, overrides modelcard)
    # WARNING: model_overrides writes to shared OsdiModel buffer. Do not reuse
    # an Instance after creating another Instance from the same Model with
    # different model_overrides — the shared buffer will have the latest values.
    # The sweep loop avoids this by using each Instance immediately.
    if model_overrides:
        for key, val in model_overrides.items():
            apply_param(model.descriptor, self._inst, model.model, key, val, True)
    # 3. Apply instance params (instance-level)
    if params:
        for key, val in params.items():
            apply_param(model.descriptor, self._inst, model.model, key, val, False)
    self._model.model.process_params()
    self._inst.bind_simulation(self._sim, model.model, self._connected_terminals, temperature)
    self._has_prev_solve = False
    self._has_prev_q = False
    self._prev_qg = 0.0
    self._prev_qd = 0.0
    self._prev_qs = 0.0
    self._prev_qb = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tech.py::test_instance_model_overrides -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `pytest tests/test_api.py -v`
Expected: All pass (model_overrides defaults to None)

- [ ] **Step 6: Commit**

```bash
git add pycmg/model.py tests/test_tech.py
git commit -m "feat: add model_overrides param to Instance for process variation"
```

---

## Task 2: Create `pycmg/tech.py` — DeviceConfig and TechConfig

Extract and restructure the technology registry from `tests/conftest.py`.

**Files:**
- Create: `pycmg/tech.py`
- Test: `tests/test_tech.py` (extend)

- [ ] **Step 1: Write failing tests for DeviceConfig and TechConfig**

Append to `tests/test_tech.py`:

```python
def test_device_config_asap7():
    """ASAP7 DeviceConfig should have static modelcard, no pdk_device."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["ASAP7"]
    dev = tech.get_device("nmos_rvt")
    assert dev.modelcard is not None
    assert dev.pdk_device is None
    assert "TFIN" in dev.inst_params
    assert "DEVTYPE" in dev.inst_params
    assert "L" not in dev.inst_params
    assert "NFIN" not in dev.inst_params


def test_device_config_tsmc():
    """TSMC DeviceConfig should have pdk_device, no static modelcard."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["TSMC7"]
    dev = tech.get_device("nmos_svt")
    assert dev.modelcard is None
    assert dev.pdk_device == "nch_svt_mac"
    assert "TFIN" in dev.inst_params
    assert "DEVTYPE" in dev.inst_params


def test_tech_config_list_devices():
    """TechConfig should list all registered devices."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["ASAP7"]
    devices = tech.list_devices()
    assert "nmos_rvt" in devices
    assert "pmos_rvt" in devices
    assert len(devices) >= 8  # 4 Vt flavors x 2 polarities


def test_tech_registry_all_techs():
    """Registry should contain all 5 base technologies."""
    from pycmg.tech import TECH_REGISTRY, list_techs
    assert set(list_techs()) >= {"ASAP7", "TSMC5", "TSMC7", "TSMC12", "TSMC16"}


def test_tech_config_pdk_path():
    """TSMC TechConfig should have pdk_path, ASAP7 should not."""
    from pycmg.tech import TECH_REGISTRY
    assert TECH_REGISTRY["ASAP7"].pdk_path is None
    assert TECH_REGISTRY["TSMC7"].pdk_path is not None
    assert "cln7" in TECH_REGISTRY["TSMC7"].pdk_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tech.py -v -k "not model_overrides"`
Expected: FAIL — `ModuleNotFoundError: No module named 'pycmg.tech'`

- [ ] **Step 3: Implement pycmg/tech.py with DeviceConfig, TechConfig, and TECH_REGISTRY**

Create `pycmg/tech.py` with:
- `DeviceConfig` dataclass (model_name, inst_params, modelcard, pdk_device, `_min_l`)
- `TechConfig` dataclass (name, vdd, tfin, devices, pdk_path)
- `TECH_REGISTRY` dict with all 5 base techs
- Each tech populated with all available devices (nmos/pmos × all Vt flavors)
- ASAP7: 8 devices (rvt, lvt, slvt, sram × nmos/pmos), `modelcard` set
- TSMC5: devices with `pdk_device` set (svt, lvt, ulvt, elvt × nmos/pmos)
- TSMC7: svt, lvt, ulvt × nmos/pmos
- TSMC12: svt, lvt, hvt, ulvt, lnvt × nmos/pmos
- TSMC16: svt, lvt, hvt, ulvt, lnvt × nmos/pmos
- `get_tech_config()` and `list_techs()` functions
- All `inst_params` contain only `{TFIN, DEVTYPE}` — no L or NFIN

Reference `tests/conftest.py` lines 27-137 for existing device/tech data to migrate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tech.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add pycmg/tech.py tests/test_tech.py
git commit -m "feat: add pycmg/tech.py with DeviceConfig, TechConfig, and TECH_REGISTRY"
```

---

## Task 3: Per-device min_l auto-detection

Add `DeviceConfig.get_min_l()` with PDK scanning and modelcard parsing.

**Files:**
- Modify: `pycmg/tech.py`
- Test: `tests/test_tech.py` (extend)

- [ ] **Step 1: Write failing tests for min_l detection**

Append to `tests/test_tech.py`:

```python
@pytest.mark.skipif(not ASAP7_MODELCARD.exists(), reason="ASAP7 modelcard missing")
def test_min_l_asap7():
    """ASAP7 min_l should be auto-detected from modelcard L parameter (21nm)."""
    from pycmg.tech import TECH_REGISTRY
    dev = TECH_REGISTRY["ASAP7"].get_device("nmos_rvt")
    min_l = dev.get_min_l()
    assert abs(min_l - 21e-9) < 1e-12, f"Expected 21nm, got {min_l*1e9:.1f}nm"


@pytest.mark.skipif(
    not (ROOT / "modelcards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l").exists(),
    reason="TSMC7 PDK missing"
)
def test_min_l_tsmc7_core():
    """TSMC7 core device min_l should be 8nm."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["TSMC7"]
    dev = tech.get_device("nmos_svt")
    min_l = dev.get_min_l(tech.pdk_path)
    assert abs(min_l - 8e-9) < 1e-12, f"Expected 8nm, got {min_l*1e9:.1f}nm"


@pytest.mark.skipif(
    not (ROOT / "modelcards" / "TSMC5" / "cln5_1d2_sp_v1d2_2p2.l").exists(),
    reason="TSMC5 PDK missing"
)
def test_min_l_tsmc5_core():
    """TSMC5 core device min_l should be 6nm."""
    from pycmg.tech import TECH_REGISTRY
    tech = TECH_REGISTRY["TSMC5"]
    dev = tech.get_device("nmos_svt")
    min_l = dev.get_min_l(tech.pdk_path)
    assert abs(min_l - 6e-9) < 1e-12, f"Expected 6nm, got {min_l*1e9:.1f}nm"


def test_min_l_cached():
    """get_min_l() should cache result after first call."""
    from pycmg.tech import TECH_REGISTRY
    dev = TECH_REGISTRY["ASAP7"].get_device("nmos_rvt")
    dev._min_l = None  # Reset cache
    l1 = dev.get_min_l()
    l2 = dev.get_min_l()
    assert l1 == l2
    assert dev._min_l is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tech.py -v -k "min_l"`
Expected: FAIL — `AttributeError: 'DeviceConfig' object has no attribute 'get_min_l'`

- [ ] **Step 3: Implement get_min_l() with _scan_pdk_device_min_l and _parse_modelcard_l**

In `pycmg/tech.py`, add:
- `_parse_modelcard_l(modelcard_path: str) -> float`: reads first `.model` block, extracts `l = <value>`
- `_scan_pdk_device_min_l(pdk_path: str, device_name: str) -> float`: scans all numbered variants of the given device, returns `min(lmin)` across all variants
- `DeviceConfig.get_min_l(pdk_path=None) -> float`: dispatches to the above, caches in `_min_l`

For `_scan_pdk_device_min_l`, reuse the same parsing approach as `_find_length_variant()` in `pycmg/parser.py` (regex-based `.model` block parsing with lmin/lmax extraction).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tech.py -v -k "min_l"`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add pycmg/tech.py tests/test_tech.py
git commit -m "feat: add per-device min_l auto-detection from PDK and modelcard"
```

---

## Task 4: `resolve_modelcard()` — on-the-fly generation

Generate TSMC modelcards dynamically from PDK, with caching.

**Files:**
- Modify: `pycmg/tech.py`
- Test: `tests/test_tech.py` (extend)

- [ ] **Step 1: Write failing tests for resolve_modelcard**

Append to `tests/test_tech.py`:

```python
def test_resolve_modelcard_asap7():
    """ASAP7 should return static modelcard path for any L."""
    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech = TECH_REGISTRY["ASAP7"]
    dev = tech.get_device("nmos_rvt")
    path = resolve_modelcard(dev, tech, 21e-9)
    assert "7nm_TT_160803.pm" in path
    # Same modelcard for different L
    path2 = resolve_modelcard(dev, tech, 42e-9)
    assert path == path2


@pytest.mark.skipif(
    not (ROOT / "modelcards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l").exists(),
    reason="TSMC7 PDK missing"
)
def test_resolve_modelcard_tsmc_generates(tmp_path):
    """TSMC should generate modelcard on-the-fly and cache it."""
    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech = TECH_REGISTRY["TSMC7"]
    dev = tech.get_device("nmos_svt")
    cache = str(tmp_path / "cache")

    path = resolve_modelcard(dev, tech, 16e-9, cache_dir=cache)
    assert Path(path).exists()
    assert "nch_svt_mac_l16nm.l" in path

    # Second call should return cached (no regeneration)
    path2 = resolve_modelcard(dev, tech, 16e-9, cache_dir=cache)
    assert path == path2


@pytest.mark.skipif(
    not (ROOT / "modelcards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l").exists(),
    reason="TSMC7 PDK missing"
)
def test_resolve_modelcard_tsmc_invalid_l():
    """TSMC should raise RuntimeError for L outside all variant ranges."""
    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech = TECH_REGISTRY["TSMC7"]
    dev = tech.get_device("nmos_svt")
    with pytest.raises(RuntimeError, match="No length variant"):
        resolve_modelcard(dev, tech, 1e-9)  # 1nm is below all bins
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tech.py -v -k "resolve_modelcard"`
Expected: FAIL — `ImportError: cannot import name 'resolve_modelcard' from 'pycmg.tech'`

- [ ] **Step 3: Implement resolve_modelcard()**

In `pycmg/tech.py`, implement `resolve_modelcard()` that:
- For ASAP7 (device.modelcard set): returns modelcard path directly
- For TSMC (device.pdk_device set): checks cache, generates if needed using `generate_naive_tsmc_modelcard()` imported from `scripts/generate_naive_tsmc.py`
- Creates cache directory `build/modelcards/{tech.name}/` if needed
- Raises `ValueError` if neither strategy applies

Extract the `generate_naive_tsmc_modelcard()` function and its helper `_INSTANCE_PARAMS` set from `scripts/generate_naive_tsmc.py` into `pycmg/tech.py`. This avoids sys.path manipulation. The script can then import from `pycmg.tech` instead.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tech.py -v -k "resolve_modelcard"`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add pycmg/tech.py tests/test_tech.py
git commit -m "feat: add resolve_modelcard() with on-the-fly TSMC generation and caching"
```

---

## Task 5: Migrate `tests/conftest.py` to use `pycmg/tech.py`

Make existing 266 tests import the registry from the new module.

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -5`
Expected: 266 passed (or current count)

- [ ] **Step 2: Refactor conftest.py to import from pycmg.tech**

Rewrite `tests/conftest.py` to:
1. Import `TECH_REGISTRY`, `get_tech_config` from `pycmg.tech`
2. Reconstruct `TECHNOLOGIES`, `CORE_VT_VARIANTS`, `ALL_TECHNOLOGIES` dicts in the old format (dict-of-dicts) by mapping from the new `TechConfig`/`DeviceConfig` dataclasses
3. Preserve the `get_tech_modelcard()` function signature and behavior exactly
4. Keep `TECH_NAMES`, `CORE_VT_NAMES`, `ALL_TECH_NAMES` lists
5. Keep the pytest hook at the bottom

The old format uses flat dicts with keys like `"nmos_file"`, `"pmos_model"`, `"nmos_params"`. The new format uses `DeviceConfig` objects. The mapping layer in conftest.py bridges these formats.

**Critical:** The old format includes L and NFIN in inst_params. The conftest wrapper must add them back. Use the existing hardcoded values from the current conftest (ASAP7: L=7e-9, NFIN=1.0; TSMC NMOS: L=16e-9, NFIN=2.0; TSMC PMOS: L=20e-9, NFIN=2.0). These are test-specific defaults, NOT from `get_min_l()` — the tests were verified with these exact values.

Example mapping for the backward-compatible dict:
```python
# In conftest.py — bridge from new DeviceConfig to old dict format
_ASAP7_NMOS_PARAMS = {"L": 7e-9, "NFIN": 1.0, **TECH_REGISTRY["ASAP7"].get_device("nmos_rvt").inst_params}
_ASAP7_PMOS_PARAMS = {"L": 7e-9, "NFIN": 1.0, **TECH_REGISTRY["ASAP7"].get_device("pmos_rvt").inst_params}
_TSMC_NMOS_PARAMS = {"L": 16e-9, "NFIN": 2.0, **TECH_REGISTRY["TSMC7"].get_device("nmos_svt").inst_params}
_TSMC_PMOS_PARAMS = {"L": 20e-9, "NFIN": 2.0, **TECH_REGISTRY["TSMC7"].get_device("pmos_svt").inst_params}
```

This preserves the exact test parameters that the existing 266 tests were verified against.

- [ ] **Step 3: Run ALL existing tests to verify no regression**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: Same pass count as Step 1 — zero regressions

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "refactor: migrate conftest.py to import registry from pycmg.tech"
```

---

## Task 6: Create `pycmg/sweep.py` — voltage grid and threshold detection

Core sweep building blocks: `find_threshold()`, `build_voltage_grid()`, `build_nodes()`.

**Files:**
- Create: `pycmg/sweep.py`
- Create: `tests/test_sweep.py`

- [ ] **Step 1: Write failing tests for build_nodes**

Create `tests/test_sweep.py`:

```python
"""Tests for the sweep module."""
import pytest


def test_build_nodes_nmos():
    from pycmg.sweep import build_nodes
    nodes = build_nodes(0.4, 0.5, 0.0, 0.9, "nmos")
    assert nodes == {"g": 0.4, "d": 0.5, "s": 0.0, "e": 0.0}


def test_build_nodes_pmos():
    from pycmg.sweep import build_nodes
    nodes = build_nodes(0.4, 0.5, 0.0, 0.9, "pmos")
    assert nodes["s"] == 0.9
    assert abs(nodes["g"] - 0.5) < 1e-12   # Vdd - 0.4
    assert abs(nodes["d"] - 0.4) < 1e-12   # Vdd - 0.5
    assert abs(nodes["e"] - 0.9) < 1e-12   # Vdd - 0.0
```

- [ ] **Step 2: Implement build_nodes(), run test**

Create `pycmg/sweep.py` with `build_nodes()`. Run: `pytest tests/test_sweep.py -v -k "build_nodes"`

- [ ] **Step 3: Write failing tests for build_voltage_grid**

```python
import numpy as np

def test_build_voltage_grid_bounds():
    from pycmg.sweep import build_voltage_grid
    vg, vd = build_voltage_grid(0.9, 0.35, vg_points=50, vd_points=50)
    assert vg[0] >= 0.0
    assert vg[-1] <= 0.9
    assert vd[0] >= 0.0
    assert vd[-1] <= 0.9


def test_build_voltage_grid_density():
    from pycmg.sweep import build_voltage_grid
    vg, _ = build_voltage_grid(0.9, 0.35, vg_points=50, vd_points=10, dense_ratio=0.6)
    # Count points in dense region [0.35 - 0.135, 0.35 + 0.135] = [0.215, 0.485]
    dense_count = np.sum((vg >= 0.215) & (vg <= 0.485))
    sparse_count = len(vg) - dense_count
    assert dense_count > sparse_count, f"Dense {dense_count} should exceed sparse {sparse_count}"


def test_build_voltage_grid_no_duplicates():
    from pycmg.sweep import build_voltage_grid
    vg, vd = build_voltage_grid(0.75, 0.3, vg_points=50, vd_points=50)
    assert len(vg) == len(np.unique(vg))
    assert len(vd) == len(np.unique(vd))
```

- [ ] **Step 4: Implement build_voltage_grid(), run tests**

Run: `pytest tests/test_sweep.py -v -k "voltage_grid"`

- [ ] **Step 5: Write failing tests for find_threshold**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
ASAP7_MODELCARD = ROOT / "modelcards" / "ASAP7" / "7nm_TT_160803.pm"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_find_threshold_nmos():
    from pycmg import Model, Instance
    from pycmg.sweep import find_threshold
    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    inst = Instance(model, params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0})
    vth = find_threshold(inst, vdd=0.9, device_type="nmos")
    assert 0.1 < vth < 0.7, f"Vth={vth:.3f}V outside reasonable range"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_find_threshold_pmos():
    from pycmg import Model, Instance
    from pycmg.sweep import find_threshold
    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "pmos_rvt")
    inst = Instance(model, params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0})
    vth = find_threshold(inst, vdd=0.9, device_type="pmos")
    # Returns magnitude (positive)
    assert 0.1 < vth < 0.7, f"Vth={vth:.3f}V outside reasonable range"
```

- [ ] **Step 6: Implement find_threshold(), run tests**

Run: `pytest tests/test_sweep.py -v -k "threshold"`

- [ ] **Step 7: Commit**

```bash
git add pycmg/sweep.py tests/test_sweep.py
git commit -m "feat: add sweep building blocks (build_nodes, build_voltage_grid, find_threshold)"
```

---

## Task 7: `resolve_devices()` and `SweepConfig`/`SweepResult` dataclasses

**Files:**
- Modify: `pycmg/sweep.py`
- Modify: `tests/test_sweep.py`

- [ ] **Step 1: Write failing tests for resolve_devices and SweepConfig**

```python
def test_resolve_devices_all():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    devices = resolve_devices(None, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert "nmos_rvt" in devices
    assert len(devices) >= 8


def test_resolve_devices_filter():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"ASAP7": ["nmos_rvt", "pmos_rvt"]}
    devices = resolve_devices(filt, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert devices == ["nmos_rvt", "pmos_rvt"]


def test_resolve_devices_glob():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"ASAP7": ["nmos_*"]}
    devices = resolve_devices(filt, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert all(d.startswith("nmos_") for d in devices)
    assert len(devices) >= 4


def test_resolve_devices_missing_skipped():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"TSMC7": ["nmos_rvt"]}  # nmos_rvt doesn't exist in TSMC7
    devices = resolve_devices(filt, "TSMC7", TECH_REGISTRY["TSMC7"])
    assert devices == []  # Skipped with warning


def test_sweep_config_defaults():
    from pycmg.sweep import SweepConfig
    config = SweepConfig(techs=["ASAP7"])
    assert config.l_multipliers == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert config.nfins == [1.0, 2.0, 3.0]
    assert len(config.temperatures) == 5
    assert config.process_vars is None
```

- [ ] **Step 2: Implement resolve_devices, SweepConfig, SweepResult, run tests**

Run: `pytest tests/test_sweep.py -v -k "resolve_devices or sweep_config"`

- [ ] **Step 3: Commit**

```bash
git add pycmg/sweep.py tests/test_sweep.py
git commit -m "feat: add resolve_devices, SweepConfig, SweepResult"
```

---

## Task 8: `sweep_dc()` core loop

The main sweep engine.

**Files:**
- Modify: `pycmg/sweep.py`
- Modify: `tests/test_sweep.py`

- [ ] **Step 1: Write failing smoke test for sweep_dc**

```python
@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_smoke():
    """Minimal sweep: 1 tech, 1 device, 1 L multiplier, 1 NFIN, 1 temp, 5x5 grid."""
    from pycmg.sweep import SweepConfig, sweep_dc
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        l_multipliers=[1.0],
        nfins=[1.0],
        temperatures=[300.15],
        vg_points=5,
        vd_points=5,
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    assert len(result.columns) == 28  # No process vars
    assert len(result.data) == 25     # 5 Vg x 5 Vd
    # Check first row has correct tech/device
    assert result.data[0][0] == "ASAP7"
    assert result.data[0][1] == "nmos_rvt"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_pmos_voltages():
    """PMOS should have Vs=Vdd, Vg < Vdd."""
    from pycmg.sweep import SweepConfig, sweep_dc
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["pmos_rvt"]},
        l_multipliers=[1.0],
        nfins=[1.0],
        temperatures=[300.15],
        vg_points=3,
        vd_points=3,
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    # Vs column (index 8) should be Vdd=0.9 for PMOS
    vs_idx = result.columns.index("Vs")
    for row in result.data:
        assert abs(row[vs_idx] - 0.9) < 1e-12, f"PMOS Vs should be Vdd=0.9, got {row[vs_idx]}"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_with_process_vars():
    """Process variation should multiply config count."""
    from pycmg.sweep import SweepConfig, sweep_dc
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        l_multipliers=[1.0],
        nfins=[1.0],
        temperatures=[300.15],
        vg_points=3,
        vd_points=3,
        process_vars={"eot": [0.9e-9, 1.1e-9]},  # 2 values
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    assert "eot" in result.columns
    assert len(result.data) == 3 * 3 * 2  # 3Vg x 3Vd x 2 process combos
```

- [ ] **Step 2: Implement sweep_dc(), run tests**

Implement `sweep_dc()` following the core loop pseudocode from the spec. Key points:
- Resolve techs, expand process grid, compute lengths per-device
- `resolve_modelcard()` + `Model()` per L
- `Instance()` with `model_overrides` per process combo
- `find_threshold()` + `build_voltage_grid()` per (L, process, NFIN, temp)
- `build_nodes()` + `eval_dc()` per voltage point
- Collect rows into `SweepResult`

Run: `pytest tests/test_sweep.py -v -k "sweep_dc"`

- [ ] **Step 3: Commit**

```bash
git add pycmg/sweep.py tests/test_sweep.py
git commit -m "feat: add sweep_dc() core loop with process variation support"
```

---

## Task 9: `to_csv()` and `generate_dataset()`

Output and convenience wrapper.

**Files:**
- Modify: `pycmg/sweep.py`
- Modify: `pycmg/__init__.py`
- Modify: `tests/test_sweep.py`

- [ ] **Step 1: Write failing tests**

```python
def test_to_csv_split_by_tech(tmp_path):
    from pycmg.sweep import SweepResult, to_csv
    result = SweepResult(
        columns=["tech", "device", "L", "NFIN", "TFIN", "temp_K",
                 "Vg", "Vd", "Vs", "Ve", "Vth", "id"],
        data=[
            ["ASAP7", "nmos_rvt", 21e-9, 1.0, 6.5e-9, 300.15,
             0.45, 0.45, 0.0, 0.0, 0.35, 1.5e-5],
            ["TSMC7", "nmos_svt", 8e-9, 1.0, 6e-9, 300.15,
             0.375, 0.375, 0.0, 0.0, 0.3, 1.2e-5],
        ],
        metadata={},
    )
    paths = to_csv(result, str(tmp_path), split_by="tech")
    assert len(paths) == 2
    assert any("ASAP7_dc.csv" in p for p in paths)
    assert any("TSMC7_dc.csv" in p for p in paths)
    # Verify content
    import csv
    with open(paths[0]) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0] == "tech"
        row = next(reader)
        assert row[0] in ("ASAP7", "TSMC7")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_generate_dataset_smoke(tmp_path):
    from pycmg.sweep import generate_dataset
    paths = generate_dataset(
        osdi_path=str(OSDI_PATH),
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        output_dir=str(tmp_path),
        l_multipliers=[1.0],
        nfins=[1.0],
        temperatures=[300.15],
        vg_points=3,
        vd_points=3,
        verbose=0,
    )
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    assert Path(paths[0]).stat().st_size > 0
```

- [ ] **Step 2: Implement to_csv() and generate_dataset(), run tests**

Run: `pytest tests/test_sweep.py -v -k "to_csv or generate_dataset"`

- [ ] **Step 3: Update pycmg/__init__.py exports**

```python
from .model import Model, Instance
from .parser import parse_modelcard, parse_number_with_suffix
from .sweep import generate_dataset, SweepConfig, SweepResult

__all__ = [
    "Model", "Instance", "parse_modelcard", "parse_number_with_suffix",
    "generate_dataset", "SweepConfig", "SweepResult",
]
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All pass (old + new)

- [ ] **Step 5: Commit**

```bash
git add pycmg/sweep.py pycmg/__init__.py tests/test_sweep.py
git commit -m "feat: add to_csv, generate_dataset, update exports"
```

---

## Task 10: CLI script `scripts/generate_training_data.py`

**Files:**
- Create: `scripts/generate_training_data.py`

- [ ] **Step 1: Implement CLI script with argparse**

Create `scripts/generate_training_data.py` with:
- `--osdi` (required): path to OSDI binary
- `--tech` (default: `["all"]`): technology names
- `--devices`: device names (glob support)
- `--list-devices`: print devices and exit
- `--l-multipliers` (default: 1 2 3 4 5)
- `--nfins` (default: 1 2 3)
- `--temps` (in Celsius, default: -40 0 27 85 125)
- `--vg-points`, `--vd-points` (default: 50)
- `--ve-values` (default: 0.0)
- `--process-var` (repeatable, format: `NAME=V1,V2,V3`)
- `--dense-ratio` (default: 0.6)
- `--output-dir` (default: `./training_data`)
- `--split-by` (default: `tech`)
- `--verbose`/`--quiet`/`--silent`

Convert `--temps` from Celsius to Kelvin. Parse `--process-var` into `dict[str, list[float]]`.

- [ ] **Step 2: Test CLI manually**

Run: `python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi --tech ASAP7 --devices nmos_rvt --l-multipliers 1 --nfins 1 --temps 27 --vg-points 3 --vd-points 3 --output-dir /tmp/test_training_data`

Verify:
- Progress output printed
- CSV file created in `/tmp/test_training_data/ASAP7_dc.csv`
- CSV has 28 columns, 9 data rows

- [ ] **Step 3: Test --list-devices**

Run: `python scripts/generate_training_data.py --osdi build/osdi/bsimcmg.osdi --tech ASAP7 --list-devices`

Verify: prints device names and exits.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_training_data.py
git commit -m "feat: add CLI script for training data generation"
```

---

## Task 11: Rewrite README.md

**Files:**
- Rewrite: `README.md`

- [ ] **Step 1: Rewrite README following the spec outline**

Follow the structure from the spec:
1. **What is PyCMG?** — training data generator for NN compact models
2. **Quick Start** — build OSDI → generate dataset → load into PyTorch
3. **Installation** — prerequisites, build, env vars
4. **Generating Training Data** — CLI usage, Python API one-liner, composable pipeline, notebook workflow
5. **Supported Technologies** — Tier 1 + Tier 2 tables
6. **Output Format** — CSV schema, size estimates, non-uniform sampling, ML loading snippets (pandas, PyTorch, TF)
7. **Advanced Usage** — single-point eval, temperature, body bias, NFIN, custom grids, PMOS, process variation
8. **Verification** — NGSPICE strategy, tolerances, test commands
9. **Project Structure** — updated directory tree (fix stale paths)
10. **API Reference** — pycmg.sweep, pycmg.model, pycmg.parser, pycmg.tech
11. **License**

Fix all stale paths: `pycmg/ctypes_host.py` → split modules, `tech_model_cards/` → `modelcards/`, remove `cpp/` reference.

- [ ] **Step 2: Verify all code examples in README actually work**

Manually run the Quick Start example and at least one Python API example.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with data generation focus and updated examples"
```

---

## Task 12: Final integration test and cleanup

**Files:**
- All files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (old 266 + new sweep/tech tests)

- [ ] **Step 2: Run a realistic data generation**

Run:
```bash
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 \
    --devices nmos_rvt pmos_rvt \
    --l-multipliers 1 2 3 \
    --nfins 1 2 \
    --temps 27 85 \
    --vg-points 20 --vd-points 20 \
    --output-dir /tmp/final_test
```

Verify:
- Completes without errors
- CSV files created with expected row counts (2 dev × 3L × 2N × 2T × 400pt = 9,600 rows)
- Values are physically reasonable (Id > 0 for NMOS in saturation, Id < 0 for PMOS)

- [ ] **Step 3: Run with process variation**

Run:
```bash
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_rvt \
    --l-multipliers 1 --nfins 1 --temps 27 \
    --vg-points 5 --vd-points 5 \
    --process-var eot=0.9e-9,1.0e-9,1.1e-9 \
    --output-dir /tmp/process_test
```

Verify: CSV has `eot` column, 75 rows (5×5×3 process combos)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: training data generation - complete implementation"
```
