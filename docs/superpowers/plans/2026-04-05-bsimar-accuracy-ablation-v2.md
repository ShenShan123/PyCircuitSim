# BSIM-AR Accuracy Ablation Study v2 — Fast Iteration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix negative R² metrics and identify which training techniques improve BSIM-AR autoregressive accuracy. Use a small model for rapid iteration (~10-20s/epoch instead of 144s).

**Root Cause (from Exp 1 analysis):** The model trains with teacher forcing (ground-truth targets) but validates/tests with autoregressive inference (own predictions). The model never learns to handle its own errors → error cascades through 13 outputs → predictions degrade to near-mean → all R² negative. This is classic "exposure bias."

**Key Changes from v1:**
1. **Small model** (d=64, 2 layers) for ~10x faster iteration vs d=256/6L
2. **Fix R² metric** — add `R2_norm` in normalized (log) space alongside physical `R2`
3. **Shorter training** — 200 epochs, patience=30
4. **All ablation features already implemented** — no new model/train code needed, just CLI flags
5. **Single new code change**: `metrics.py` + `print_metrics` to add `R2_norm`

**Tech Stack:** PyTorch, CUDA, existing nn_model normalization pipeline, DirectLoss.

---

## File Structure

| File | Responsibility | Tasks |
|------|---------------|-------|
| `external_compact_models/BSIMAR/script/metrics.py` | Add R2_norm metric | 1 |
| `external_compact_models/BSIMAR/script/main.py` | No changes (all flags exist) | — |
| `external_compact_models/BSIMAR/results/ablation_v2_*.txt` | Experiment results | 2-6 |

---

### Task 1: Fix R² Metric — Add Normalized-Space R²

The current R² is computed in physical space after exponential denormalization. For outputs spanning 14 decades (1e-20 to 1e-6), this is numerically meaningless. Add `R2_norm` computed in normalized (signed-log + z-score) space — the space the model actually optimizes.

**Files:**
- Modify: `external_compact_models/BSIMAR/script/metrics.py`

- [ ] **Step 1: Add R2_norm to `compute_physical_metrics()`**

In `metrics.py`, inside the per-output loop (after the existing R² computation, ~line 56), add:

```python
        # R2 in normalized space (meaningful for log-transformed data)
        y_t_n = true_norm[:, i]
        y_p_n = pred_norm[:, i]
        ss_res_n = np.sum((y_t_n - y_p_n) ** 2)
        ss_tot_n = np.sum((y_t_n - y_t_n.mean()) ** 2)
        r2_norm = 1.0 - ss_res_n / ss_tot_n if ss_tot_n > 0 else 0.0
```

Add `"R2_norm": r2_norm` to the metrics dict for each output.

- [ ] **Step 2: Update `print_metrics()` to show R2_norm**

Update the header and row format to include `R2_norm` column. Also update the AVG line.

```python
def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{'Target':>8s} | {'NRMSE%':>8s} | {'MRE%':>8s} | {'R2':>8s} | {'R2_norm':>8s} | {'MAE_n':>8s}")
    print("-" * 62)
    for name in OUTPUT_COLUMN_ORDER:
        m = metrics[name]
        mre_str = f"{m['MRE(%)']:8.2f}" if not np.isnan(m["MRE(%)"]) else "     N/A"
        print(f"{name:>8s} | {m['NRMSE(%)']:8.3f} | {mre_str} | {m['R2']:8.4f} | {m['R2_norm']:8.4f} | {m['MAE_norm']:8.4f}")

    avg_nrmse = np.mean([m["NRMSE(%)"] for m in metrics.values()])
    valid_mre = [m["MRE(%)"] for m in metrics.values() if not np.isnan(m["MRE(%)"])]
    avg_mre = np.mean(valid_mre) if valid_mre else float("nan")
    avg_r2 = np.mean([m["R2"] for m in metrics.values()])
    avg_r2_norm = np.mean([m["R2_norm"] for m in metrics.values()])
    print("-" * 62)
    mre_avg_str = f"{avg_mre:8.2f}" if not np.isnan(avg_mre) else "     N/A"
    print(f"{'AVG':>8s} | {avg_nrmse:8.3f} | {mre_avg_str} | {avg_r2:8.4f} | {avg_r2_norm:8.4f} |")
```

- [ ] **Step 3: Verify the metric change**

```bash
conda run -n pycircuitsim python -c "
import numpy as np
from external_compact_models.BSIMAR.script.metrics import compute_physical_metrics, print_metrics
from nn_model.data.normalize import Normalizer
# Quick sanity: perfect predictions should give R2_norm = 1.0
norm = Normalizer.from_file('external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_norm.npz')
fake = np.random.randn(100, 13).astype(np.float32)
m = compute_physical_metrics(fake, fake, norm)
assert all(abs(m[k]['R2_norm'] - 1.0) < 1e-6 for k in m), 'R2_norm should be 1.0 for identical pred/true'
print('R2_norm sanity check PASSED')
"
```

- [ ] **Step 4: Commit**

```bash
git add external_compact_models/BSIMAR/script/metrics.py
git commit -m "fix: add R2_norm metric in normalized space for BSIM-AR"
```

---

### Task 2: Exp 1 — Small Model Baseline (Teacher Forcing Only)

Establish baseline with small model. Expected: same negative R² pattern as the large model (exposure bias still present), but much faster to confirm.

**No code changes. CLI only.**

- [ ] **Step 1: Train**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 200 --patience 30 \
    --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 256 --dropout 0.1 \
    --batch-size 2048 --lr 8e-4 \
    --exp-name exp1v2 2>&1 | tee results_exp1v2_nmos.log
```

- [ ] **Step 2: Save results and backup checkpoint**

```bash
# Copy metrics from log to results file
tail -25 results_exp1v2_nmos.log > external_compact_models/BSIMAR/results/ablation_v2_exp1.txt
cp external_compact_models/BSIMAR/checkpoints/exp1v2_nmos_best.pt \
   external_compact_models/BSIMAR/checkpoints/exp1v2_nmos_backup.pt
```

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/results/ablation_v2_exp1.txt
git commit -m "exp: BSIM-AR v2 Exp 1 — small model baseline (d=64, 2L)"
```

**Expected:** R² negative (physical), R2_norm also likely negative or near 0 (AR inference still breaks). Confirms the exposure bias is the problem, not model capacity.

---

### Task 3: Exp 2 — Scheduled Sampling (The Key Fix)

This is the most impactful change. Gradually introduce model's own predictions during training so it learns to self-correct.

**No code changes. CLI only.**

- [ ] **Step 1: Train**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 200 --patience 30 \
    --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 256 --dropout 0.1 \
    --batch-size 2048 --lr 8e-4 \
    --scheduled-sampling --ss-warmup 50 --ss-max-ratio 0.5 \
    --exp-name exp2v2 2>&1 | tee results_exp2v2_nmos.log
```

- [ ] **Step 2: Save results**

```bash
tail -25 results_exp2v2_nmos.log > external_compact_models/BSIMAR/results/ablation_v2_exp2.txt
```

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/results/ablation_v2_exp2.txt
git commit -m "exp: BSIM-AR v2 Exp 2 — scheduled sampling (warmup=50, max=0.5)"
```

**Expected:** Significant R2_norm improvement (>0). This directly attacks the exposure bias. Val loss should actually decrease over training.

---

### Task 4: Exp 3 — Scheduled Sampling + Output Reorder

Reorder outputs so easy targets (charges) come first, giving the AR model better context for harder targets (conductances, current).

**No code changes. CLI only.**

- [ ] **Step 1: Train**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 200 --patience 30 \
    --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 256 --dropout 0.1 \
    --batch-size 2048 --lr 8e-4 \
    --scheduled-sampling --ss-warmup 50 --ss-max-ratio 0.5 \
    --reorder \
    --exp-name exp3v2 2>&1 | tee results_exp3v2_nmos.log
```

- [ ] **Step 2: Save results and commit**

```bash
tail -25 results_exp3v2_nmos.log > external_compact_models/BSIMAR/results/ablation_v2_exp3.txt
git add external_compact_models/BSIMAR/results/ablation_v2_exp3.txt
git commit -m "exp: BSIM-AR v2 Exp 3 — scheduled sampling + output reorder"
```

**Expected:** Moderate improvement on top of Exp 2, especially for later targets (gm, gds, id).

---

### Task 5: Exp 4 — Full Kitchen Sink

All techniques combined: scheduled sampling + reorder + consistency loss + curriculum.

**No code changes. CLI only.**

- [ ] **Step 1: Train**

```bash
conda run -n pycircuitsim python -u -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --universal --cuda \
    --epochs 200 --patience 30 \
    --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 256 --dropout 0.1 \
    --batch-size 2048 --lr 8e-4 \
    --scheduled-sampling --ss-warmup 50 --ss-max-ratio 0.5 \
    --reorder \
    --consistency-weight 0.1 \
    --curriculum --curriculum-warmup 30 \
    --exp-name exp4v2 2>&1 | tee results_exp4v2_nmos.log
```

- [ ] **Step 2: Save results and commit**

```bash
tail -25 results_exp4v2_nmos.log > external_compact_models/BSIMAR/results/ablation_v2_exp4.txt
git add external_compact_models/BSIMAR/results/ablation_v2_exp4.txt
git commit -m "exp: BSIM-AR v2 Exp 4 — full kitchen sink (SS+reorder+consistency+curriculum)"
```

---

### Task 6: Ablation Summary Comparison

- [ ] **Step 1: Compile results into a comparison table**

Read all 4 result files and create a summary table at `external_compact_models/BSIMAR/results/ablation_v2_summary.txt`:

```
| Exp | Description                     | id NRMSE | gm NRMSE | avg NRMSE | avg R2 | avg R2_norm |
|-----|---------------------------------|----------|----------|-----------|--------|-------------|
| 1   | Baseline (TF only)              |   ?      |   ?      |   ?       |  ?     |   ?         |
| 2   | + Scheduled sampling            |   ?      |   ?      |   ?       |  ?     |   ?         |
| 3   | + Output reorder                |   ?      |   ?      |   ?       |  ?     |   ?         |
| 4   | + Consistency + Curriculum       |   ?      |   ?      |   ?       |  ?     |   ?         |
```

- [ ] **Step 2: Commit summary**

```bash
git add external_compact_models/BSIMAR/results/ablation_v2_summary.txt
git commit -m "exp: BSIM-AR v2 ablation summary — 4 experiments compared"
```

---

## Experiment Matrix

| Exp | Change | CLI Flags (on top of base) | Builds on |
|-----|--------|---------------------------|-----------|
| 1 | Small baseline (TF only) | `--d-model 64 --num-layers 2 --dim-feedforward 256` | — |
| 2 | + Scheduled sampling | + `--scheduled-sampling --ss-warmup 50 --ss-max-ratio 0.5` | Exp 1 |
| 3 | + Output reorder | + `--reorder` | Exp 2 |
| 4 | + Consistency + Curriculum | + `--consistency-weight 0.1 --curriculum --curriculum-warmup 30` | Exp 3 |

**Base args** (all experiments): `--device-type nmos --universal --cuda --epochs 200 --patience 30 --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 256 --dropout 0.1 --batch-size 2048 --lr 8e-4`

**Estimated time per experiment:** 15-30 min on GPU (vs 1-2 hours in v1). **Total:** ~1-2 hours.

## Verification

After all experiments, confirm:
1. **R2_norm > 0** for at least Exp 2-4 (proves scheduled sampling fixes exposure bias)
2. **Val loss decreases** over training for Exp 2-4 (proves the model is learning AR inference)
3. **NRMSE improves** monotonically from Exp 1 → Exp 4
4. Results in `external_compact_models/BSIMAR/results/ablation_v2_summary.txt`
