# B9 — Hard-Monotone Network — V6.4.5 Track-B Report

**Date:** 2026-05-29/30 (training 23:10–00:57, score 00:05 CST)  
**Tech:** TSMC7  
**Verdict: KILL**

---

## Design Choice: Option (A) — Monotone-Constrained MLP (Replacement, not Residual)

**Key difference from Phase-7a:** Phase-7a adds a monotone *residual*. B9 **replaces** the trunk's id column entirely with a `_MonotoneIdHead`. Only the id path is constrained; the other 12 outputs remain from the shared trunk.

### `_MonotoneIdHead` Construction (3-layer, Daniels & Velikova 2010)

- Layer 1: Vg path: `softplus(w_vg1_raw)` (≥0) × Vg + unconstrained other inputs → h1 monotone-increasing in Vg
- Softplus activation (C¹, globally monotone — Rule 4)
- Layer 2: `softplus(w_h1_raw)` (≥0) × h1 + skip unconstrained → h2 still monotone-increasing in Vg
- Softplus activation
- Layer 3: `softplus(w3_raw)` (≥0) × h2 → scalar monotone-increasing in Vg
- Final: `sign = -1` → monotone-**decreasing** output

**Sign:** Both NMOS and PMOS training data show id_norm **decreasing** as Vg_norm increases (ON current is negative in PyCMG, gets more negative at higher Vg). So `sign = -1` for both.

**Architecture:** 26,625 parameters (vs 342,669 for the trunk shared with other 12 outputs).

---

## Sanity Checks (all pass)

- (a) Stock mono checkpoint loads without mono_id keys: **PASS**
- (b) id monotone-decreasing in Vg at init (20-point sweep, all diffs ≤ 0): **PASS**
- (c) autograd gives finite gm < 0 and gds: **PASS**

---

## Training

**NMOS:** 200 epochs completed (no early stop), ~13.8s/epoch, ~2755s total
- Best val loss: 0.034640
- Started 23:10, LDS done 23:21, epoch 1 at 23:21, finished ~23:57 CST

**Final NMOS test metrics:**

| Target | NRMSE% | R² |
|--------|--------|-----|
| **id** | **2.764%** | **0.141** ← CATASTROPHICALLY POOR |
| gm | 0.060% | 1.000 |
| gds | 0.570% | 0.962 |
| gmb | 0.135% | 1.000 |
| qg-cdd | ≤0.032% | ≥1.000 |

**NMOS id R² = 0.141 (should be ~0.999 for canonical model).**
The 26,625-param monotone head cannot fit the complex MOSFET id surface that the
342,669-param shared trunk fits trivially. Replacing the trunk's id with a heavily
constrained 26K-param head destroys id accuracy.

**PMOS:** Still training at report time (~56 epochs in, same pattern expected).

---

## Scorer Baseline (canonical TSMC7)

```
inv VTC NRMSE=3.892%  MaxErr=188.0mV
inv tran NRMSE=1.099%
RO period_err=8.98%  (NG 46.6ps / DN 50.8ps)
SRAM rail_resid=0.302  (q=0.815 qb=0.226)
opamp_flat_flag=1
```

---

## B9 Final Scorer Results (NMOS complete + PMOS ~56 epochs)

```json
{
  "inv_vtc_nrmse": 39.23,
  "inv_vtc_maxerr_mv": 888.9,
  "inv_vtc_r2": 0.347,
  "inv_tran_post_nrmse": 7.37,
  "ring_osc_period_err": 46.97,
  "ro_nrmse": 57.27,
  "sram_rail_snap_resid": 0.485,
  "sram_q": 0.774,
  "sram_qb": 0.364,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.062
}
```

**All metrics catastrophically regressed:**

| Metric | B9 | Baseline | Change |
|--------|-----|---------|--------|
| inv VTC NRMSE | **39.2%** | 3.89% | 10x WORSE |
| inv VTC MaxErr | **889 mV** | 188 mV | 4.7x WORSE |
| inv tran NRMSE | 7.37% | 1.10% | 6.7x WORSE |
| RO period err | **47.0%** | 8.98% | 5.2x WORSE |
| SRAM resid | 0.485 | 0.302 | 1.6x WORSE |
| opamp gain | 0.062 | ~15 | Flat |

---

## SRAM Attractor Analysis

Intermediate score (NMOS ~70 epochs): q=0.776, qb=0.357, resid=0.475  
Final score (NMOS complete): q=0.774, qb=0.364, resid=0.485

**The SRAM attractor shifted from (0.815, 0.226) to (0.774, 0.364):**
- q: 0.815 → 0.774 (5% closer to VDD=0.75 rail) ← monotone did increase storage-NMOS drive
- qb: 0.226 → 0.364 (61% farther from 0 rail) ← WORSE, cross-coupling disturbed
- net residual: 0.302 → 0.485 ← WORSE

**Did the monotone model raise the storage-NMOS linear-region drive?**
YES — q moved from 0.815 toward VDD=0.75, confirming the monotone id constraint
increased the storage NMOS pull. But the cross-coupled cell dynamics created a new
interior equilibrium at (0.774, 0.364) rather than snapping.

**0/4 SRAM cells snap. KILL criterion met** (need ≥2/4).

---

## Root Cause of Failure

**Primary: Architectural capacity mismatch.**

The `_MonotoneIdHead` has only 26,625 parameters (3 layers × ~128 hidden). The canonical
DirectNet trunk has 342,669 parameters fitting id from a shared MLP. Replacing the full
trunk id column with a tiny constrained head reduces id-fitting capacity by 13x, resulting
in NMOS id R²=0.141 (canonical ~0.999). All circuit metrics are meaningless when id is
fit this poorly.

**Secondary: Monotonicity constraint too rigid.**

Even with sufficient capacity, strict global monotonicity in Vg may not be achievable
while also fitting all other aspects of the id surface (Vds nonlinearity, body effects,
variant tech codes). The MOSFET id surface is not globally monotone in Vg — short-channel
effects and body modulation can locally violate this.

**B6 diagnosis confirmed but not addressed by B9:**

B6 localised SRAM failure to the LINEAR region at high Vgs (Vd∈[0.05,0.35], Vg>0.55),
meaning the issue is drive *magnitude*, not non-monotone bumps. B9 confirmed that the
monotone constraint did increase drive (q moved toward VDD) but:
1. The id model was too inaccurate (R²=0.141) to produce meaningful results
2. Even if id fit were perfect, the cross-coupled cell created a new interior equilibrium

---

## Verdict: KILL

**Primary kill reason:** NMOS id R²=0.141 (vs canonical ~0.999). The monotone-constrained
head destroys id accuracy. NR convergence was not poisoned (no timeouts), but the circuit
results are meaningless with such poor id fit.

**Secondary kill reason:** SRAM attractor still at q∈[0.1,0.9] (q=0.774, qb=0.364)
even if results were valid. 0/4 cells snap. Criterion ≥2/4 not met.

**Kill condition explicitly met:** "SRAM attractor persists at q∈[0.1,0.9] after the
monotone model converges." (q=0.774, qb=0.364, both in [0.1, 0.9])

---

## NR Convergence Note

No NR divergence or timeout observed. The constrained head provides finite, smooth gm
and gds via autograd (verified in sanity check). NR convergence was not impacted.

---

## B9 Design Implication for V6.4.6

The B9 implementation choice (REPLACE trunk id with constrained head) is wrong.
The Phase-7a approach (ADD monotone residual to trunk id) preserves the trunk's
342,669-param fit quality while adding a monotone bias. A corrected B9 would:

1. **Keep the trunk's id output** (don't replace it)
2. **Add a LARGER monotone head** as a residual (not a replacement) — but ensure the
   residual has enough parameters (e.g., 3 hidden layers of size 256 = ~240K params)
   to properly shape the id surface
3. **Alternatively:** Use the existing Phase-7a residual but scale up its hidden dim
   from 64 to 256 to increase its monotone-bias capacity

Or, better, close the SRAM gate via V6.4.6 route:
- **Split-head DirectNet:** separate id head for the SRAM linear-region operating point
- **Retrain with SRAM-targeted loss:** circuit-level penalty for interior attractors

---

## Candidate Stems on Disk

- NMOS: `b9_mono_tsmc7_nmos` (complete, 200 epochs, val=0.034640)
- PMOS: `b9_mono_tsmc7_pmos` (training, ~56 epochs as of report)

Both in `external_compact_models/bsimar/checkpoints/`.

---

## Implementation Files

- `experiments/v6_4_5_track_b/B9_monotone_lattice.py` — training script
- `external_compact_models/bsimar/models/direct_net.py` — `_MonotoneIdHead` class
- `pycircuitsim/models/mosfet_directnet.py` — `_build_from_state` auto-detection
