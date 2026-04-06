# BSIMAR Accuracy Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix BSIMAR's poor test accuracy after denormalization by aligning normalization, loss, data filtering, and metrics with the paper's approach.

**Architecture:** Unified BSIMAR normalizer supporting both `zscore` and `signedlog` modes. Loss module supports simple losses (MAE) and composed losses (MAE+LDS). Data loader supports pre-training filtering. All BSIMAR-specific — nn_model's normalizer untouched.

**Tech Stack:** Python, PyTorch, NumPy, scikit-learn

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| CREATE | `BSIMAR/script/normalize.py` | Unified normalizer: z-score mode + signedlog mode |
| CREATE | `BSIMAR/script/data.py` | Data loading with small-value filtering |
| MODIFY | `BSIMAR/script/losses.py` | Add `MAELoss` (simple) + `LDSMAELoss` (composed) |
| MODIFY | `BSIMAR/script/metrics.py` | Support BSIMARNormalizer, higher MRE threshold |
| MODIFY | `BSIMAR/script/main.py` | Wire new normalizer, data loader, loss, CLI flags |

All paths relative to `external_compact_models/`.

---

### Task 1: Create unified BSIMAR normalizer

**Files:**
- Create: `external_compact_models/BSIMAR/script/normalize.py`

Single normalizer class with `mode` parameter:
- `mode="zscore"`: z-score for inputs, z-score for outputs (paper's approach)
- `mode="signedlog"`: min-max for inputs, signed_log+z-score for outputs (current nn_model approach)

**Stats dataclass stores all fields, using `Optional` for mode-specific ones:**
- Common: `mode`, `output_mean`, `output_std`
- zscore-only: `input_mean`, `input_std`
- signedlog-only: `input_min`, `input_max`, `output_log_floors`

- [ ] **Step 1:** Create `BSIMARNormStats` dataclass with all fields, `save()`/`load()`.
- [ ] **Step 2:** Create `BSIMARNormalizer(mode="zscore")` class:
  - `fit(inputs, geometry, outputs)` — branches on mode
  - `normalize_inputs(inputs, geometry)` — z-score or min-max
  - `normalize_outputs(outputs)` — z-score or signedlog+z-score
  - `denormalize_outputs(outputs_norm)` — inverse of above
  - `_build_combined_input(inputs, geometry)` — shared, same as nn_model

---

### Task 2: Create BSIMAR data loader with filtering

**Files:**
- Create: `external_compact_models/BSIMAR/script/data.py`

Paper filters samples where any target falls below a small-value threshold BEFORE normalization.

- [ ] **Step 1:** Create `filter_small_targets(outputs, column_names, thresholds)` — returns boolean mask. Per-group defaults: id/gm/gds/gmb > 1e-12, charges > 1e-19, caps > 1e-19.
- [ ] **Step 2:** Create `BSIMARDataset(Dataset)` — holds normalized tensors, returns `(x, y)`.
- [ ] **Step 3:** Create `load_and_split_bsimar(data_path, column_names, norm_mode, apply_filter, ...)`:
  - Loads .npz, optionally filters, shuffles/splits, fits `BSIMARNormalizer(mode=norm_mode)` on train, normalizes all splits.

---

### Task 3: Add MAELoss and LDSMAELoss to losses.py

**Files:**
- Modify: `external_compact_models/BSIMAR/script/losses.py`

Two new classes:
- `MAELoss(nn.Module)` — simple MAE: `mean(|pred - true|)`. Supports optional per-sample weights.
- `LDSMAELoss(nn.Module)` — composed: MAE weighted by LDS inverse-density weights. Same `forward(pred, true, weights)` signature.

Both share the same interface as `WeightedBNILoss` so existing `train_epoch_bni` / `validate_epoch_bni` work unchanged.

- [ ] **Step 1:** Add `MAELoss(nn.Module)` with `forward(y_pred, y_true, weights=None)`.
- [ ] **Step 2:** `LDSMAELoss` is just `MAELoss` used with pre-computed LDS weights (no separate class needed — `MAELoss` already accepts weights). Document this.

---

### Task 4: Update metrics for BSIMARNormalizer and higher MRE threshold

**Files:**
- Modify: `external_compact_models/BSIMAR/script/metrics.py`

- [ ] **Step 1:** Update `compute_physical_metrics()`:
  - Accept either `BSIMARNormalizer` or `Normalizer` (duck-typing on `denormalize_outputs`).
  - MRE threshold: `max(|y_t|) * 0.01` (1% of peak absolute value per target) instead of `floor * 100`.
  - Remove direct dependency on `normalizer.stats.output_log_floors`.

---

### Task 5: Update main.py to wire everything together

**Files:**
- Modify: `external_compact_models/BSIMAR/script/main.py`

**New CLI flags:**
- `--norm-mode {zscore,signedlog}` (default: `zscore`)
- `--no-filter` flag to skip data filtering
- `--loss {direct,mae,bni}` — `mae` is the new simple MAE; when combined with `--lds`, becomes composed MAE+LDS
- `--lds` flag — enables LDS reweighting (works with `mae` or `bni`)

**Data loading branch:**
- Always use `load_and_split_bsimar()` (it handles both norm modes via `norm_mode` param)
- Remove old `load_and_split()` import

**Loss routing:**
- `--loss direct`: existing `DirectLoss` + `train_epoch_direct`/`validate_epoch_direct`
- `--loss mae` (no --lds): `MAELoss` + `train_epoch_bni`/`validate_epoch_bni` (they handle the interface)
- `--loss mae --lds`: `MAELoss` + LDS weights computed + weighted TensorDataset + same training loops
- `--loss bni` (with/without --lds): existing `WeightedBNILoss` + same

- [ ] **Step 1:** Add CLI flags.
- [ ] **Step 2:** Replace data loading with `load_and_split_bsimar()`.
- [ ] **Step 3:** Wire loss selection with LDS flag.
- [ ] **Step 4:** Pass `BSIMARNormalizer` to metrics.
- [ ] **Step 5:** Update norm stats save path to use `BSIMARNormStats`.

---

### Task 6: Smoke test

- [ ] **Step 1:** Run zscore+mae mode: `python -m external_compact_models.BSIMAR.script.main --device-type nmos --universal --norm-mode zscore --loss mae --epochs 2`
- [ ] **Step 2:** Run signedlog+direct mode (backward compat): `python -m external_compact_models.BSIMAR.script.main --device-type nmos --universal --norm-mode signedlog --loss direct --epochs 2`
- [ ] **Step 3:** Run zscore+mae+lds mode: `python -m external_compact_models.BSIMAR.script.main --device-type nmos --universal --loss mae --lds --epochs 2`
- [ ] **Step 4:** Commit all changes.
