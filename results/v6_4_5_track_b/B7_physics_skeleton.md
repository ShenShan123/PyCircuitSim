# B7 Physics-Anchored Skeleton + Residual — V6.4.5 Track-B Report

**Date:** 2026-05-29  
**Tech:** TSMC7  
**Verdict:** KILL

---

## Skeleton Design

### Architecture

`_PhysicsSkeletonId` (`bsimar/models/direct_net.py`): BSIM-style 3-region closed-form Id model
implemented as a differentiable `nn.Module`.

**Forward:** receives normalised inputs x (B,7); denormalises Vd,Vg,Vs,Vb,NFIN using registered
buffers (in_mean, in_std from the norm.npz fit). Computes Vgs=Vg-Vs, Vds=Vd-Vs, Vbs=Vb-Vs.

**Three-region formula:**
- Subthreshold (Vov < 0): `Id = I0·NFIN·exp(Vov/n·vt)·(1-exp(-|Vds|/vt))`  
- Linear (|Vds| < Vov): `Id = β·NFIN·(Vov·|Vds| - Vds²/2)`  
- Saturation (|Vds| ≥ Vov): `Id = 0.5·β·NFIN·Vov²·(1+λ·|Vds|)`

All region boundaries use smooth blends (softplus/sigmoid, C1 — Rule 4 compliant).

**Output normalisation:** skeleton output is re-normalised to asinh z-score space
(same space as the base head) so the skeleton + ε·base_head sum is dimensionally consistent.

**PyCMG sign convention:** nmos_sign=-1 for NMOS (id < 0 when ON), +1 for PMOS.

**Learnable parameters (9 per model):**
| Param | Init | Final (NMOS) | Physical meaning |
|-------|------|--------------|-----------------|
| Vth0 | 0.28 V | 0.549 V | Avg threshold over SVT/LVT/ULVT variants |
| n_factor | -2.3 (n=1.1) | 0.33 (n=2.39) | Subthreshold slope factor |
| beta_log | -8.6 (1.8e-4) | -9.01 (1.2e-4 A/V²/fin) | Mobility×Cox proxy |
| lambda_log | -1.2 (0.30) | -2.18 (0.11 V⁻¹) | Channel-length modulation |
| gamma | 0.05 | 0.003 | Body-effect (learned ~0 for multi-VT) |
| phi_log | -0.36 (0.7) | 1.35 (3.86 V) | Surface potential |
| log_I0 | -12.0 (6e-6) | -12.97 (2.3e-6 A/fin) | Subthreshold current scale |
| k_blend | 10.0 | 6.74 | Region-blend sharpness |
| log_vt | -3.65 (0.026) | -2.54 (0.079 V) | Effective thermal voltage |

The high Vth0=0.549V and vt=0.079V represent the average across all three TSMC7 VT variants
(SVT/LVT/ULVT with Vth ~0.3-0.5V range).

### DirectNet Integration

- `DirectNet.__init__` gains `physics_skeleton` flag + output-norm buffers (skeleton_{in_mean,in_std,nmos_sign,vdd_train,eps,id_asinh_scale,id_out_mean,id_out_std})`
- `forward`: if `skeleton is not None`: `id_col = skeleton(x) + skeleton_eps * base_head`  
  (x = continuous features only, NOT concatenated with tech embedding)
- Default path (physics_skeleton=False) is byte-identical to before this change
- `_build_from_state` in `mosfet_directnet.py`: auto-detects `skeleton.*` keys and reconstructs
  DirectNet with physics_skeleton=True

### Sanity checks (all pass)
- (a) Stock mono checkpoint loads without skeleton keys — PASS
- (b) Skeleton at init: id ≤ 0 for NMOS ON, near-zero at Vds=0, very small for reverse Vds — PASS
- (c) Skeleton model forward: finite 13-col output, autograd(id) gives finite gm/gds — PASS

---

## Training

**NMOS:** `CUDA_VISIBLE_DEVICES=3`, seed=42, medium size (256 hidden, 5 layers), eps=0.1,
200 epochs (ran to completion, no early stop), 10.5s/epoch, ~35 min.  
**Best val loss:** 0.001189 (vs NMOS training, not directly comparable to canonical)  
**Final NMOS test set metrics:**
| Target | NRMSE% | R² |
|--------|--------|-----|
| id | 0.029 | 0.9999 |
| gm | 0.063 | 1.0000 |
| gds | 0.448 | 0.9764 |
| gmb | 0.138 | 0.9999 |
| qg-cdd | ≤0.05 | ≥1.000 |

**PMOS:** Same settings, stem `b7_skel_tsmc7_pmos`. Training started at 12:21 CST, still running.

---

## Scorer Baseline (canonical, measured by THIS scorer)

```
=== TSMC7  nmos=tsmc7_dn_medium_nmos  pmos=tsmc7_dn_medium_pmos ===
  inv  VTC  NRMSE=3.892%  MaxErr=188.0mV
  inv  tran NRMSE=1.099%
  RO   period_err=8.98%  (NG 46.6ps / DN 50.8ps)
  SRAM rail_resid=0.302  (q=0.815 qb=0.226)
```

---

## B7 Scorer Results: NMOS + canonical PMOS (partial)

```
=== TSMC7  nmos=b7_skel_tsmc7_nmos  pmos=tsmc7_dn_medium_pmos ===
  inv  VTC  NRMSE=2.404%  MaxErr=82.2mV
  inv  tran NRMSE=1.076%
  RO   period_err=11.80%  (NG 46.6ps / DN 52.1ps)
  SRAM rail_resid=0.313  (q=0.814 qb=0.235)
```

**SRAM both-states probe (B7 NMOS + canonical PMOS, state-1 force_ic):**
q=0.814, qb=0.235, resid=0.313. The attractor barely moved (0.815→0.814).

---

## Analysis: Why the Skeleton Did Not Fix the SRAM Attractor

### Root cause: joint training with eps=0.1 allows the residual to dominate

The forward is: `id_total = id_skel_norm + 0.1 × id_base_norm`

During training, the NN residual (id_base) receives gradients to minimise
`||id_total - id_target||`. The loss gradient w.r.t. id_base is 0.1× the gradient w.r.t.
id_skel. After 200 epochs, the training converged to:
`id_base ≈ (id_target - id_skel_norm) / 0.1`

The residual learned the NEGATIVE of the skeleton's errors, making the combined model
functionally equivalent to the canonical baseline at in-distribution inputs:

| Point (L=16nm NFIN=2 ULVT T=300K) | B7 raw id | Canonical raw id |
|-------------------------------------|-----------|------------------|
| Vg=VDD, Vd=0.375V (saturation) | -1.11e-4 A | -1.19e-4 A |
| Vg=VDD, Vd=0.02V (small linear) | -1.51e-5 A | -1.50e-5 A |
| Vg=0 (off) | +1.14e-7 A* | -1.60e-7 A |

*B7 off-state has wrong sign raw (positive), but the Vds correction at step (d) clamps it to 0.

The SRAM false attractor at q≈0.815, qb≈0.226 is an IN-DISTRIBUTION failure — the NN
correctly learns the same attractor that is in the training data because the training samples
(multi-VT average) model this behavior.

### What the skeleton accomplished
- Better inverter VTC: MaxErr dropped from 188 mV → 82 mV (56% improvement)
- Subthreshold suppression improved (off-current nearly 0 after Vds correction)
- But RO regressed: 8.98% → 11.80% (skeleton perturbed the transition-edge timing)

### Why the SRAM attractor persists
The SRAM cross-coupled cell needs the storage NMOS (Vgs=0.524V, near threshold) to sink
enough current at the low node (qb→0) to overcome the PMOS pull-down. The B7 model
predicted essentially the same id as canonical at all these operating points, so the
equilibrium point barely moved.

---

## Verdict: KILL

**SRAM:** q=0.814, qb=0.235, resid=0.313 — no snap to rails. 0/4 cells snap. **FAIL** (need ≥2/4).

**Inverter VTC:** NRMSE 2.404% / MaxErr 82.2 mV vs scorer baseline NRMSE 3.892% / MaxErr 188 mV.
VTC IMPROVED significantly (MaxErr -105 mV). But this improvement comes at the cost of RO regression.

**RO:** 11.80% vs baseline 8.98% — **REGRESSED** past the 5% gate. **FAIL**.

**Kill reason:** SRAM still stalls at q ∈ [0.1, 0.9] after residual converged (q=0.814 = interior
attractor, not the rail). The skeleton approach failed because joint training with eps=0.1 allows
the NN residual to absorb the skeleton's errors, making the combined model equivalent to the
canonical baseline in-distribution. The SRAM false attractor IS in-distribution.

---

## Candidate Stems on Disk

- NMOS: `b7_skel_tsmc7_nmos` (complete, 200 epochs, val=0.001189)
- PMOS: `b7_skel_tsmc7_pmos` (training in progress, expected ~35 min from 12:21 CST)

Both stored in `external_compact_models/bsimar/checkpoints/`.

---

## What Would Fix This

1. **Two-stage training:** pre-fit the skeleton offline (freeze it), then train only the residual.
   This prevents the residual from cancelling the skeleton.
2. **Larger eps or frozen skeleton:** eps=1.0 would give the skeleton equal weight as the residual,
   or freeze skeleton after offline fit.
3. **Different data:** augment the SRAM operating region (Vgs=VT-VDD range, small Vds, 
   q/qb equilibrium points) in the training set so the model sees these points more often.
4. **SRAM-targeted loss term:** a physics constraint that penalises interior equilibria of the
   cross-coupled cell (requires circuit-level gradient — complex).

---

## Implementation Files

- `external_compact_models/bsimar/models/direct_net.py` — `_PhysicsSkeletonId` class + DirectNet changes
- `pycircuitsim/models/mosfet_directnet.py` — `_build_from_state` auto-detection of skeleton keys  
- `experiments/v6_4_5_track_b/B7_train.py` — training script
- `experiments/v6_4_5_track_b/B7_skeleton_directnet.patch` — the `direct_net.py` +
  `mosfet_directnet.py` skeleton implementation (preserved as a patch; the live
  edits were reverted after the kill).

---

## Independent re-score (full B7 N+P pair, after PMOS finished)

The agent's in-run scoring used B7-NMOS + canonical-PMOS (PMOS still training).
Re-scored with the completed `b7_skel_tsmc7_{nmos,pmos}` pair:

```
RO period_err = 11.89%  (NG 46.6 / DN 52.2 ps)   ← REGRESSED vs 8.98%
inv VTC NRMSE = 2.40%  MaxErr = 63.6 mV          ← improved vs 3.89% / 188 mV
inv tran NRMSE = 1.10%
SRAM rail_resid = 0.310  (q=0.814, qb=0.233)     ← unchanged interior attractor
```

Confirms the KILL: SRAM attractor unmoved, RO regressed. The skeleton helps the
inverter VTC but the subthreshold-exp suppressor targets the wrong region —
B6 localised the SRAM failure to the **linear region at high Vgs** (Vg>0.55,
Vd∈[0.05,0.35]), not subthreshold. A frozen-skeleton refit (the agent's
suggested fix to stop the residual cancelling the skeleton) was NOT pursued
because (a) the skeleton regressed RO and (b) its subthreshold focus cannot
address the linear-region under-drive that B6 identified as the SRAM root cause.

