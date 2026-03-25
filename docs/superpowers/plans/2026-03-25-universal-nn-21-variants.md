# Universal NN Compact Model v2 — 21 Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a universal NN compact model covering 21 Vt variants across 5 process nodes, and evaluate cross-device cross-technology transferability via leave-one-out experiments.

**Architecture:** Reuse Phase 14 universal NN pipeline (13-dim input, DirectNet MLP) with expanded device coverage. New pycmg-wrapper provides 21 variants (was 13). Leave-one-out holds out 5 variants (1 per tech) for zero-shot testing.

**Tech Stack:** PyTorch 2.10.0 (CPU), pycmg-wrapper, NumPy, Matplotlib

---

### Task 1: Update config.py — New PyCMG paths + 21 variants ✅

**Files:**
- Modify: `nn_model/config.py`

- [x] Point PYCMG_DIR to `/home/shenshan/pycmg-wrapper`
- [x] Update ASAP7 L=7nm (was 30nm), add TFIN field to TechConfig
- [x] Add 8 new variants: TSMC5 ulvt/elvt, TSMC7 ulvt (real PMOS), TSMC12 ulvt/hvt/lnvt, TSMC16 ulvt/hvt/lnvt
- [x] Extract process params from all 42 modelcards

### Task 2: Update generate.py — New wrapper API ✅

**Files:**
- Modify: `nn_model/data/generate.py`

- [x] Update sys.path to pycmg-wrapper
- [x] Add TFIN and DEVTYPE to instance params

### Task 3: Generate universal training data

- [ ] Run: `python -m nn_model.data.generate --device both --universal`
- [ ] Verify: ~800K+ points per device type

### Task 4: Train universal NMOS + PMOS models

- [ ] NMOS: `python -u -m nn_model.train --device-type nmos --universal --mode direct13 --epochs 800 --hidden 384 --layers 6 --patience 150 --batch-size 2048`
- [ ] PMOS: same with `--device-type pmos`

### Task 5: Full model verification (63 tests)

**Files:**
- Create: `tests/verify_nn_universal_v2.py`

- [ ] NMOS DC sweep per variant (21 tests)
- [ ] PMOS DC sweep per variant (21 tests)
- [ ] Inverter VTC per variant (21 tests)
- [ ] Target: device DC < 10% NRMSE, inverter VTC < 15% NRMSE

### Task 6: Leave-one-out transferability experiment

**Files:**
- Create: `tests/verify_nn_leave_one_out.py`

Held-out variants:
- ASAP7: slvt, TSMC5: elvt, TSMC7: ulvt, TSMC12: hvt, TSMC16: lnvt

- [ ] Train on 16 variants, test zero-shot on 5 held-out
- [ ] Compare in-distribution vs zero-shot NRMSE
- [ ] Report transferability gap

### Task 7: Run verification + update CLAUDE.md

- [ ] Execute full verification
- [ ] Execute leave-one-out experiments
- [ ] Update CLAUDE.md Phase 15 section
- [ ] Commit all changes
