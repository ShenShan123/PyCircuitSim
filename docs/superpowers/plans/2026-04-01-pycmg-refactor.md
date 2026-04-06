# PyCMG Integration Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor NN_SPICE to use the updated PyCMG submodule APIs, eliminating duplicated tech configuration, hard-coded paths, and stale modelcard references.

**Architecture:** PyCMG's `TECH_REGISTRY` becomes the single source of truth for device structure (model names, VDD, TFIN, device registry). NN-specific config (ProcessParams, training hyperparams) stays in `nn_model/config.py` but references PyCMG's registry. Modelcard resolution uses pre-baked naive modelcards from the submodule (TSMC PDK files are gitignored), with `resolve_modelcard()` as fallback when full PDK files are available.

**Tech Stack:** Python 3.10, PyCMG (OSDI ctypes), PyTorch, conda `pycircuitsim` env

---

## Analysis: What Changed in PyCMG

### New Modules
- `pycmg/sensitivity.py` — OAT sensitivity analysis (`compute_sensitivity`, `SensitivityResult`)
- `pycmg/sweep.py` — Sweep engine (`SweepConfig`, `sweep_dc`, `generate_dataset`, `SweepResult`)

### Enhanced Modules
- `pycmg/tech.py` — Now has a production `TECH_REGISTRY` with `DeviceConfig`/`TechConfig`, `resolve_modelcard()`, `get_min_l()`, `get_geometry_combos()`. This was previously only test infrastructure.
- `pycmg/model.py` — New `Instance.set_params()`, `Instance.get_jacobian_matrix()`, `model_overrides` parameter
- `pycmg/parser.py` — New `parse_tsmc_pdk()`, `scan_pdk_geometry_combos()`, `ParsedModel` dataclass

### Backward-Compatible APIs (No Changes Needed)
- `Model(osdi_path, modelcard_path, model_name, model_card_name)` — same signature
- `Instance(model, params, temperature)` — same signature (new optional `model_overrides`)
- `Instance.eval_dc(nodes)` — same signature, same 17-key output dict
- `Instance.eval_tran(nodes, time, delta_t, prev_state)` — same signature

## Integration Points Requiring Refactoring

### 1. `nn_model/config.py` — Major duplication (~300 lines)
- Has its own `TechConfig`, `VariantConfig` dataclasses that duplicate PyCMG's `TechConfig`, `DeviceConfig`
- Hard-codes 21 device variants with model names, VDD, TFIN, L, modelcard paths — all available from PyCMG's `TECH_REGISTRY`
- Hard-codes OSDI path to standalone clone `/home/shenshan/pycmg-wrapper` instead of submodule
- **Keeps NN-specific data not in PyCMG:** `ProcessParams` (7 process params), `TrainConfig`, `OUTPUT_COLUMNS`, `INPUT_COLUMNS`

### 2. `nn_model/data/generate.py` — Path and modelcard issues
- `sys.path.insert(0, "/home/shenshan/pycmg-wrapper")` — hard-coded absolute path
- Uses `nn_model.config.TechConfig` for modelcard resolution — should use PyCMG's `resolve_modelcard()`
- Creates `Model()` per NFIN value — could reuse via `Instance.set_params()`

### 3. `pycircuitsim/models/mosfet_cmg.py` — Minor
- `sys.path.insert(0, str(PYCMG_PATH))` — fragile path manipulation
- Core API calls are backward-compatible, no functional changes needed

### 4. `pycircuitsim/parser.py` — Depends on nn_model/config
- LEVEL=73 parsing imports `TECH_CONFIGS` from `nn_model.config`
- Will need to follow config refactor

## Critical Constraints

1. **NN models are already trained** — ProcessParams values MUST NOT change. These are baked into trained checkpoints.
2. **ASAP7 VDD mismatch** — `nn_model/config.py` uses VDD=0.7V; PyCMG's `TECH_REGISTRY` uses VDD=0.9V. The NN was trained at 0.7V. Keep NN training VDD separate from PyCMG.
3. **No regressions** — All 67+ existing tests must pass after refactoring.
4. **TSMC PDK files are gitignored** — The submodule only has pre-baked naive modelcards under `modelcards/TSMC*/naive/`, NOT the full PDK `.l` files. `resolve_modelcard()` requires the full PDK and will fail without it. Modelcard resolution MUST fall back to pre-baked naive modelcards.
5. **Backward compatibility aliases** — 4 test files import `TechConfig` by name. Provide `TechConfig = NNTechConfig` alias to avoid updating all test files in this refactor.

---

## File Changes

| File | Action | Responsibility |
|------|--------|----------------|
| `nn_model/config.py` | **Major refactor** | Slim down: import from PyCMG TECH_REGISTRY, keep ProcessParams + training config |
| `nn_model/data/generate.py` | **Modify** | Fix sys.path, update type annotations, remove hard-coded paths |
| `pycircuitsim/config.py` | **Minor update** | Verify OSDI path points to submodule (already does) |
| `pycircuitsim/models/mosfet_cmg.py` | **Keep as-is** | sys.path setup stays (simulator may not import nn_model.config) |
| `pycircuitsim/parser.py` | **No change** | API compatible via backward-compat aliases |
| `tests/verify_nn_universal.py` | **Modify** | Remove `TechConfig` import (use alias), fix sys.path |
| `tests/verify_nn_universal_v2.py` | **Modify** | Remove pycmg-wrapper sys.path, fix `TechConfig` import |
| `tests/verify_nn_multi_tech.py` | **Modify** | Fix `TechConfig` import |
| `tests/verify_nn_leave_one_out.py` | **Modify** | Remove pycmg-wrapper sys.path |
| `tests/verify_bsimcmg_*.py` | **Verify** | Run to confirm no regression |
| `tests/verify_nn_tran.py` | **Verify** | Run to confirm no regression |

---

## Task 1: Fix sys.path and OSDI Path Configuration

**Files:**
- Modify: `nn_model/config.py:13-15` (PYCMG_DIR, OSDI_PATH)
- Modify: `nn_model/data/generate.py:22` (sys.path hard-code)
- Modify: `pycircuitsim/models/mosfet_cmg.py:22-25` (sys.path)

**Why:** Both `nn_model/config.py` and `generate.py` point to a standalone PyCMG clone at `/home/shenshan/pycmg-wrapper`. The project has a git submodule at `external_compact_models/PyCMG/`. All PyCMG references should use the submodule.

- [ ] **Step 1: Update `nn_model/config.py` PYCMG_DIR**

Change lines 13-18 from:
```python
PYCMG_DIR = Path("/home/shenshan/pycmg-wrapper")
OSDI_PATH = str(PYCMG_DIR / "build" / "osdi" / "bsimcmg.osdi")

# ASAP7 technology config
ASAP7_MODELCARD = str(PYCMG_DIR / "modelcards" / "ASAP7" / "7nm_TT_160803.pm")
```

To:
```python
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
OSDI_PATH = str(PYCMG_DIR / "build" / "osdi" / "bsimcmg.osdi")

# ASAP7 technology config
ASAP7_MODELCARD = str(PYCMG_DIR / "modelcards" / "ASAP7" / "7nm_TT_160803.pm")
```

Also update `TSMC_MODELCARDS` at line 26:
```python
TSMC_MODELCARDS = PYCMG_DIR / "modelcards"
```

- [ ] **Step 2: Remove hard-coded sys.path in `nn_model/data/generate.py`**

Remove line 22:
```python
sys.path.insert(0, "/home/shenshan/pycmg-wrapper")
```

The PyCMG submodule path is already added by `mosfet_cmg.py` or should be added here consistently:
```python
PYCMG_PATH = PROJECT_ROOT / "external_compact_models" / "PyCMG"
if str(PYCMG_PATH) not in sys.path:
    sys.path.insert(0, str(PYCMG_PATH))
```

- [ ] **Step 3: Centralize PyCMG sys.path setup**

Currently `mosfet_cmg.py:22-25` adds PyCMG to sys.path. Instead of repeating this in multiple files, add it once in `nn_model/config.py` (which is imported first by any NN or training code) and keep it in `mosfet_cmg.py` for the simulator path.

In `nn_model/config.py`, add near the top after PROJECT_ROOT:
```python
# Ensure PyCMG submodule is importable
_PYCMG_PYPATH = str(PROJECT_ROOT / "external_compact_models" / "PyCMG")
if _PYCMG_PYPATH not in sys.path:
    sys.path.insert(0, _PYCMG_PYPATH)
```

- [ ] **Step 4: Verify import works**

Run: `conda run -n pycircuitsim python -c "from nn_model.config import OSDI_PATH; print(OSDI_PATH)"`
Expected: Path containing `external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`

Run: `conda run -n pycircuitsim python -c "from nn_model.config import PYCMG_DIR; import pycmg; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add nn_model/config.py nn_model/data/generate.py pycircuitsim/models/mosfet_cmg.py
git commit -m "fix: use PyCMG submodule paths instead of hard-coded standalone clone"
```

---

## Task 2: Import PyCMG TECH_REGISTRY and Use resolve_modelcard()

**Files:**
- Modify: `nn_model/config.py` (major refactor of TechConfig/VariantConfig)

**Why:** `nn_model/config.py` has ~250 lines of hard-coded tech configuration (TechConfig, VariantConfig with model names, VDD, L, TFIN, modelcard paths for 5 techs x 21 variants). PyCMG's `TECH_REGISTRY` now provides all of this. The only NN-specific data is `ProcessParams` (7 values per device variant).

**Strategy:** Import device structure from PyCMG's `TECH_REGISTRY`, build a thin NN-specific wrapper that pairs PyCMG devices with their ProcessParams. Keep `ProcessParams`, `TrainConfig`, `OUTPUT_COLUMNS`, `INPUT_COLUMNS` unchanged.

- [ ] **Step 1: Add PyCMG tech imports**

Add at top of `nn_model/config.py` (after sys.path setup):
```python
from pycmg.tech import TECH_REGISTRY as _PYCMG_REGISTRY, resolve_modelcard, get_tech_config as _get_pycmg_tech
```

- [ ] **Step 2: Create NNVariantConfig (replaces VariantConfig)**

Replace the `VariantConfig` dataclass (~35 lines) with a slimmer version that references PyCMG for model names and modelcard resolution:

```python
@dataclass
class NNVariantConfig:
    """NN-specific variant config: pairs a PyCMG device with process params."""
    name: str                       # Variant name (e.g., 'rvt', 'svt')
    nmos_process: ProcessParams     # NN feature values for NMOS
    pmos_process: ProcessParams     # NN feature values for PMOS

    def get_process_params(self, device_type: str) -> ProcessParams:
        return self.nmos_process if device_type == "nmos" else self.pmos_process
```

- [ ] **Step 3: Create NNTechConfig (replaces TechConfig)**

Replace the ~65-line `TechConfig` dataclass with one that wraps PyCMG's TechConfig:

```python
@dataclass
class NNTechConfig:
    """NN training config wrapping PyCMG's tech registry."""
    pycmg_name: str                                # Key into TECH_REGISTRY (uppercase)
    vdd_train: float                               # VDD used for NN training (may differ from PyCMG)
    nfin_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 15, 20])
    temperature: float = DEFAULT_TEMPERATURE
    variants: Dict[str, NNVariantConfig] = field(default_factory=dict)
    default_variant: str = ""
    # Asymmetric L override (PyCMG uses get_min_l() per device, but NN was
    # trained with fixed L values — keep these for backward compatibility)
    L_nmos: Optional[float] = None
    L_pmos: Optional[float] = None

    @property
    def name(self) -> str:
        return self.pycmg_name

    @property
    def pycmg_tech(self) -> "pycmg.tech.TechConfig":
        return _PYCMG_REGISTRY[self.pycmg_name]

    @property
    def vdd(self) -> float:
        return self.vdd_train

    @property
    def tfin(self) -> float:
        return self.pycmg_tech.tfin

    @property
    def L(self) -> float:
        """Default L (uses NMOS L if asymmetric)."""
        if self.L_nmos is not None:
            return self.L_nmos
        if self.L_pmos is not None:
            return self.L_pmos
        # Fallback: auto-detect from PyCMG (ASAP7 default device)
        devices = self.pycmg_tech.list_devices()
        nmos_devs = [d for d in devices if d.startswith("nmos_")]
        if nmos_devs:
            dev = self.pycmg_tech.get_device(nmos_devs[0])
            return dev.get_min_l()
        return 7e-9

    def get_L(self, device_type: str) -> float:
        if device_type == "nmos" and self.L_nmos is not None:
            return self.L_nmos
        if device_type == "pmos" and self.L_pmos is not None:
            return self.L_pmos
        return self.L

    def get_model_name(self, device_type: str, variant: Optional[str] = None) -> str:
        """Get model name from PyCMG registry."""
        vname = variant or self.default_variant
        canon = f"{device_type}_{vname}"  # e.g., "nmos_svt"
        dev = self.pycmg_tech.get_device(canon)
        return dev.model_name

    def get_modelcard_path(self, device_type: str, variant: Optional[str] = None) -> str:
        """Resolve modelcard path.

        Priority:
        1. ASAP7: static modelcard from DeviceConfig.modelcard
        2. TSMC: pre-baked naive modelcard in submodule (PYCMG_DIR/modelcards/{TECH}/naive/)
        3. Fallback: resolve_modelcard() (requires full PDK file on disk)
        """
        vname = variant or self.default_variant
        canon = f"{device_type}_{vname}"
        dev = self.pycmg_tech.get_device(canon)
        L = self.get_L(device_type)

        # ASAP7: static modelcard
        if dev.modelcard is not None:
            return str(PYCMG_DIR / dev.modelcard)

        # TSMC: pre-baked naive modelcard (PDK files are gitignored in submodule)
        if dev.pdk_device is not None:
            L_nm = int(L * 1e9)
            naive_path = PYCMG_DIR / "modelcards" / self.pycmg_name / "naive" / f"{dev.pdk_device}_l{L_nm}nm.l"
            if naive_path.exists():
                return str(naive_path)

        # Fallback: resolve_modelcard (needs full PDK file)
        return resolve_modelcard(dev, self.pycmg_tech, L)

    def get_process_params(self, device_type: str, variant: Optional[str] = None) -> ProcessParams:
        vname = variant or self.default_variant
        if vname and vname in self.variants:
            return self.variants[vname].get_process_params(device_type)
        raise ValueError(f"No process params for tech={self.name}, device={device_type}, variant={vname}")

    def get_phig(self, device_type: str, variant: Optional[str] = None) -> float:
        return self.get_process_params(device_type, variant).phig
```

- [ ] **Step 4: Rebuild TECH_CONFIGS using NNTechConfig**

Replace the ~250 lines of ASAP7_CONFIG, TSMC5_CONFIG, etc. with slimmer definitions.
Each tech only needs: pycmg_name, vdd_train, L overrides, and ProcessParams per variant.

Example for ASAP7 (~15 lines vs ~30 lines before):
```python
ASAP7_CONFIG = NNTechConfig(
    pycmg_name="ASAP7",
    vdd_train=0.7,  # NN training VDD (PyCMG uses 0.9V)
    L_nmos=7e-9, L_pmos=7e-9,
    default_variant="rvt",
    variants={
        "rvt": NNVariantConfig("rvt",
            nmos_process=ProcessParams(phig=4.372, u0=0.0252, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.8108, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0)),
        # ... lvt, slvt, sram (keep ProcessParams values EXACTLY as-is)
    },
)
```

TSMC configs become even slimmer — modelcard paths are now resolved via PyCMG:
```python
TSMC5_CONFIG = NNTechConfig(
    pycmg_name="TSMC5",
    vdd_train=0.65,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.534, u0=0.0369, ...),
            pmos_process=ProcessParams(phig=4.56, u0=0.1288, ...)),
        # ... lvt, ulvt, elvt
    },
)
```

The modelcard path lines (`nmos_modelcard_path=str(_tsmc5_naive / ...)`) are ALL eliminated — `resolve_modelcard()` handles this.

- [ ] **Step 5: Update TECH_CONFIGS registry**

```python
TECH_CONFIGS: Dict[str, NNTechConfig] = {
    "asap7": ASAP7_CONFIG,
    "tsmc5": TSMC5_CONFIG,
    "tsmc7": TSMC7_CONFIG,
    "tsmc12": TSMC12_CONFIG,
    "tsmc16": TSMC16_CONFIG,
}
```

- [ ] **Step 6: Remove dead code and add backward-compat aliases**

Remove old `VariantConfig` class body (replaced by `NNVariantConfig`).
Remove old `TechConfig` class body (replaced by `NNTechConfig`).
Remove `_tsmc5_naive`, `_tsmc7_naive`, etc. path variables.
Remove `TSMC_MODELCARDS` variable.

Add backward compatibility aliases at module level (after TECH_CONFIGS):
```python
# Backward compatibility aliases — 4 test files import these by name.
# Remove once test files are updated to use NNTechConfig/NNVariantConfig.
TechConfig = NNTechConfig
VariantConfig = NNVariantConfig
```

- [ ] **Step 7: Verify config import**

Run: `conda run -n pycircuitsim python -c "from nn_model.config import TECH_CONFIGS; print(list(TECH_CONFIGS.keys()))"`
Expected: `['asap7', 'tsmc5', 'tsmc7', 'tsmc12', 'tsmc16']`

Run: `conda run -n pycircuitsim python -c "from nn_model.config import TECH_CONFIGS; t = TECH_CONFIGS['tsmc5']; print(t.get_model_name('nmos', 'svt'))"`
Expected: `nch_svt_mac`

Run: `conda run -n pycircuitsim python -c "from nn_model.config import TECH_CONFIGS; t = TECH_CONFIGS['tsmc5']; print(t.get_modelcard_path('nmos', 'svt'))"`
Expected: Path to cached/generated naive modelcard

- [ ] **Step 8: Commit**

```bash
git add nn_model/config.py
git commit -m "refactor: use PyCMG TECH_REGISTRY for device config, keep NN-specific ProcessParams"
```

---

## Task 3: Fix Test File Paths and Imports

**Files:**
- Modify: `tests/verify_nn_universal_v2.py:35` (remove pycmg-wrapper sys.path)
- Modify: `tests/verify_nn_leave_one_out.py:38` (remove pycmg-wrapper sys.path)

**Why:** Two test files hard-code `sys.path.insert(0, "/home/shenshan/pycmg-wrapper")`. After Task 1, `nn_model/config.py` handles PyCMG sys.path setup via the submodule. The `TechConfig` import works via the backward-compat alias added in Task 2, so no import changes needed.

- [ ] **Step 1: Fix `verify_nn_universal_v2.py`**

Remove line 35:
```python
sys.path.insert(0, "/home/shenshan/pycmg-wrapper")
```

- [ ] **Step 2: Fix `verify_nn_leave_one_out.py`**

Remove line 38:
```python
sys.path.insert(0, "/home/shenshan/pycmg-wrapper")
```

- [ ] **Step 3: Verify imports still work**

Run: `conda run -n pycircuitsim python -c "from tests.verify_nn_universal_v2 import *" 2>&1 || echo "Expected: may fail due to missing args, but should not fail on ImportError"`

Actually, just do a quick smoke test:
```bash
conda run -n pycircuitsim python -c "
import sys; sys.path.insert(0, '.')
from nn_model.config import TechConfig, TECH_CONFIGS
print(type(TECH_CONFIGS['asap7']))
print('TechConfig alias OK')
"
```
Expected: `NNTechConfig` type, `TechConfig alias OK`

- [ ] **Step 4: Commit**

```bash
git add tests/verify_nn_universal_v2.py tests/verify_nn_leave_one_out.py
git commit -m "fix: remove hard-coded pycmg-wrapper paths from test files"
```

---

## Task 4: Update Data Generation to Use New Config API

**Files:**
- Modify: `nn_model/data/generate.py`

**Why:** `generate.py` uses `tech.get_modelcard_path()`, `tech.get_model_name()`, and `tech.get_L()` which now delegate to PyCMG. The API names are the same (by design in Task 2), so changes are minimal. The `VariantConfig` type annotation should switch to `NNVariantConfig` (or use alias).

- [ ] **Step 1: Update imports and type annotations**

Change import to use new names (or keep old names via alias — both work):
```python
from nn_model.config import (
    OSDI_PATH, NNTechConfig, NNVariantConfig, ProcessParams, OUTPUT_COLUMNS,
    DATA_DIR, TECH_CONFIGS,
)
```

Update type annotations in function signatures:
```python
def create_pycmg_instance(tech: NNTechConfig, ...) -> Instance:
def generate_dataset(tech: NNTechConfig, ...) -> Dict[str, np.ndarray]:
```

- [ ] **Step 2: Fix `variant_cfg.get_model_name()` call**

At `generate.py:239`, `variant_cfg` is now `NNVariantConfig` which does NOT have `get_model_name()`. This is a print-only call but will crash at runtime.

Change line 239 from:
```python
print(f"  NFIN={nfin}: Creating PyCMG instance "
      f"(model={variant_cfg.get_model_name(device_type)})...")
```
To:
```python
print(f"  NFIN={nfin}: Creating PyCMG instance "
      f"(model={tech.get_model_name(device_type, variant_name)})...")
```

- [ ] **Step 4: Verify data generation**

Run: `conda run -n pycircuitsim python -m nn_model.data.generate --device nmos --tech asap7 --variants rvt 2>&1 | head -20`
Expected: Successful generation with correct model name and non-zero data points.

- [ ] **Step 5: Commit**

```bash
git add nn_model/data/generate.py
git commit -m "refactor: update data generation to use NNTechConfig API"
```

---

## Task 5: Verify Parser LEVEL=73 Compatibility

**Files:**
- Modify: `pycircuitsim/parser.py:581-621`

**Why:** The LEVEL=73 MOSFET parsing imports `TECH_CONFIGS` from `nn_model.config` and accesses `tech_cfg.variants[vt].get_process_params()`. Since `NNTechConfig` preserves the same access patterns, only the import names need updating.

- [ ] **Step 1: Verify backward compatibility**

The parser code at line 606-621 does:
```python
tech_cfg = TECH_CONFIGS[tech_key]
if nn_vt is not None:
    pp = tech_cfg.variants[vt_lower].get_process_params(device_key)
```

With `NNTechConfig`, `tech_cfg.variants` is `Dict[str, NNVariantConfig]` and `NNVariantConfig.get_process_params()` returns `ProcessParams`. The `.as_dict()` and `.phig` accesses are unchanged. **No code change needed** — the API is compatible.

- [ ] **Step 2: Run parser test**

Run: `conda run -n pycircuitsim python -c "
from pycircuitsim.parser import Parser
from pycircuitsim.config import BSIMCMG_OSDI_PATH
p = Parser(osdi_path=BSIMCMG_OSDI_PATH)
print('Parser import OK')
"`
Expected: `Parser import OK`

- [ ] **Step 3: Commit (if any changes were needed)**

No commit needed if no changes. Mark task complete.

---

## Task 6: Regression Testing — BSIM-CMG (LEVEL=72)

**Files:**
- Run: `tests/verify_bsimcmg_op.py`
- Run: `tests/verify_bsimcmg_dc.py`
- Run: `tests/verify_bsimcmg_tran.py`

**Why:** The BSIM-CMG path (`mosfet_cmg.py`) uses PyCMG's `Model`/`Instance`/`eval_dc` which are backward compatible. But PyCMG's internal code changed significantly, so we must verify no numerical regressions.

- [ ] **Step 1: Run OP verification**

Run: `conda run -n pycircuitsim python tests/verify_bsimcmg_op.py 2>&1 | tail -20`
Expected: All PASS with <0.02% error.

- [ ] **Step 2: Run DC sweep verification**

Run: `conda run -n pycircuitsim python tests/verify_bsimcmg_dc.py 2>&1 | tail -20`
Expected: All PASS with <0.1% NRMSE.

- [ ] **Step 3: Run transient verification**

Run: `conda run -n pycircuitsim python tests/verify_bsimcmg_tran.py 2>&1 | tail -20`
Expected: All PASS with <0.5% NRMSE.

- [ ] **Step 4: Run comprehensive transient**

Run: `conda run -n pycircuitsim python tests/verify_bsimcmg_tran_comprehensive.py 2>&1 | tail -30`
Expected: 21/21 configs PASS.

- [ ] **Step 5: Log results and commit any fixes**

If all pass, no commit needed. If fixes needed, commit with descriptive message.

---

## Task 7: Regression Testing — NN Model (LEVEL=73)

**Files:**
- Run: `tests/verify_nn_tran.py`
- Run: `tests/verify_multi_tech_tran.py`
- Run: `tests/verify_nn_universal_v2.py` (exercises TechConfig alias + config pipeline)
- Run: `tests/verify_nn_multi_tech.py` (exercises TechConfig alias + DC/VTC)

**Why:** These tests exercise the full NN pipeline including the refactored config. The universal/multi-tech tests also validate that the TechConfig backward-compat alias works.

- [ ] **Step 1: Run NN transient verification**

Run: `conda run -n pycircuitsim python tests/verify_nn_tran.py 2>&1 | tail -30`
Expected: 5/5 PASS with NRMSE < 15%.

- [ ] **Step 2: Run multi-tech transient**

Run: `conda run -n pycircuitsim python tests/verify_multi_tech_tran.py 2>&1 | tail -30`
Expected: All PASS.

- [ ] **Step 3: Run NN universal v2**

Run: `conda run -n pycircuitsim python tests/verify_nn_universal_v2.py 2>&1 | tail -30`
Expected: 19/21 PASS (known FAIL: ASAP7:SLVT, TSMC7:LVT on NMOS DC).

- [ ] **Step 4: Run NN multi-tech**

Run: `conda run -n pycircuitsim python tests/verify_nn_multi_tech.py 2>&1 | tail -30`
Expected: All PASS with NRMSE < 10%/15%.

- [ ] **Step 5: Log results and commit any fixes**

If all pass, no commit needed. If fixes needed, commit with descriptive message.

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Why:** After the refactoring, the documentation should reflect the new config structure and PyCMG integration.

- [ ] **Step 1: Update Architecture section**

Update the module structure to note that `nn_model/config.py` now imports from PyCMG's `TECH_REGISTRY`:
```
nn_model/
├── config.py                       # NNTechConfig wrapping PyCMG TECH_REGISTRY + ProcessParams
```

- [ ] **Step 2: Update NN Model Rules**

Add a note:
```
8. **PyCMG integration** — `nn_model/config.py` imports device structure from PyCMG's `TECH_REGISTRY`. ProcessParams (7 NN input features) are NN-specific and NOT from PyCMG. VDD may differ between NN training (e.g., ASAP7=0.7V) and PyCMG (0.9V).
```

- [ ] **Step 3: Update Quick Start section**

Update the NN Model usage section to note `resolve_modelcard()` handles modelcard paths automatically.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for PyCMG TECH_REGISTRY integration"
```

---

## Summary of Changes

| File | Action | Key Changes |
|------|--------|-------------|
| `nn_model/config.py` | **Major refactor** | -~170 lines: import PyCMG TECH_REGISTRY, replace TechConfig/VariantConfig, add aliases |
| `nn_model/data/generate.py` | **Minor update** | Fix sys.path, update type annotations, fix variant_cfg.get_model_name() |
| `tests/verify_nn_universal_v2.py` | **Fix path** | Remove pycmg-wrapper sys.path |
| `tests/verify_nn_leave_one_out.py` | **Fix path** | Remove pycmg-wrapper sys.path |
| `CLAUDE.md` | **Update docs** | Note PyCMG TECH_REGISTRY integration |

## What We Are NOT Changing (and Why)

1. **ProcessParams values** — Baked into trained NN checkpoints. Changing them invalidates all models.
2. **NMOS_CMG / PMOS_CMG class structure** — Works correctly. Deduplication is a separate effort.
3. **Data generation sweep logic** — NN-specific augmentation (zero-bias anchors, dense mid-supply) is not in PyCMG's sweep engine. Keep custom.
4. **Solver code** — No PyCMG API changes affect solver.py.
5. **mosfet_nn.py** — NN model wrapper is independent of PyCMG changes.
6. **mosfet_cmg.py sys.path** — Simulator path may be used without nn_model.config. Keep its own sys.path setup.
7. **Test file TechConfig imports** — Handled via backward-compat aliases. Full migration in a follow-up.
