# V6 follow-up — Inverter coverage + sample-class weighting + VTC model selection

**Date:** 2026-05-10
**Branch:** `feat/v6` (off `c8a783b` — post-refactor verified state)
**Scope:** DirectNet (LEVEL=73) and BSIMAR Transformer (LEVEL=74). ASAP7 excluded.
**Strategy:** Three composable levers, one retrain per lever, gated revert.
The shipping set is whichever subset survives the inverter acceptance gate.

> **Premise:** v6 sprint closed solver-side levers (Tier 1A NR clamp, Tier 1.5 env override, Tier 2). The remaining inverter accuracy gap is a **data + selection** problem, not solver or loss-architecture. All loss-side derivative-consistency knobs (ChargeConsistencyLoss N4, SlopeMatchLoss B2, SignConsistencyLoss, BoundaryLoss) have been rejected on principle (asinh chain rule) or A/B (no benefit). **Out of scope.** See §6.

---

## 1. Starting state (post-refactor, `c8a783b`)

DirectNet v4-re universal, all 16 cells verified by `verify_nn_dc_tran.py`.

| Suite | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|
| 1-dev DC NRMSE % (NMOS) | ~1.0 | **14.72** | 0.18 | 0.19 |
| Inverter VTC NRMSE % | borderline | **~19** | ~10 | ~10 |
| Inverter Tran NRMSE % | ~12 | ~9 | ~7 | ~8 |

BSIMAR v4-re Transformer numbers track DirectNet within ±1 pp on inverter VTC
and within ±2 pp on transient.

**Goal:** TSMC7 NMOS DC ≤ 8 %, all inverter VTC ≤ 12 %, all inverter
transients ≤ 15 %, TSMC5 inverter transient ≤ 10 %.

---

## 2. Failure-mode map (for context)

D1 diagnostic (`tests/diag_d1_tsmc7_nmos_errors.py`) and v5 session summary established:

1. **TSMC7 NMOS DC error concentrates in the saturation plateau** `Vgs ∈ [0.52, 0.73] V × Vds ∈ [0.40, 0.75] V` — the inverter trip-point cone. B1 hybrid sampler under-covers this band ~16× relative to the uniform `Id-Vgs` verifier weighting.
2. **Train→inverter gap is ~29×** (E3 evidence: 0.45 % training NRMSE → 14.74 % inverter NMOS DC). Loss-best and phys-best checkpoints select the wrong models for inverter purposes.
3. **Inverter transient holds via rule 19a** (rail-restoring quadratic extrapolation). Don't learn the rail in training — v5 overshoot overlay caused TSMC7/12/16 NR runaway. Inference correction is the right layer.

---

## 3. Levers in execution order

Each tier is one commit. If the verification gate fails, `git reset --hard
HEAD~1` and skip to the next tier.

### Tier 1 — Extend `inv_trip` overlay to TSMC7/12/16

**File:** `external_compact_models/PyCMG/pycmg/nn_generate.py:621-634`

Drop the `spec.tech_name == "tsmc5"` gate. Add a per-tech cone table that
centers on `Vgs ≈ Vds ≈ VDD/2` with a half-width of `0.10·VDD`, sampled at
~675 points/bin (matching the existing TSMC5 overlay density).

Current code:
```python
# nn_generate.py:621-634
if spec.enable_inv_trip and spec.tech_name == "tsmc5":
    ...
    for vg, vd, vbs in _inv_trip_points(vth_mag, spec.vdd, is_pmos):
```

Change to:
```python
if spec.enable_inv_trip and spec.tech_name in {"tsmc5", "tsmc7", "tsmc12", "tsmc16"}:
    ...
```

Plus: in `_inv_trip_points`, replace peak-gm Vth derivation with a **fixed cone centered at VDD/2** for non-TSMC5 techs. Rationale: D1 diagnostic showed the verifier metric is dominated by the saturation plateau at `Vgs ≈ Vds ≈ VDD/2`, not Vth. Keep Vth-centered overlay for TSMC5 (where it produced the 18× transient gain).

**Data regen:** rerun `external_compact_models/PyCMG/scripts/generate_nn_data.py --device both --universal` with `--enable-inv-trip` (current default). Expect ~7–10 % more rows per non-TSMC5 tech.

**Retrain:** DirectNet NMOS+PMOS only this tier. Skip Transformer until Tier 1 validated (saves ~80 min GPU).

```bash
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos --exclude-techs asap7 \
    --num-tech-codes 18 --epochs 800 --hidden 384 --layers 6 \
    --patience 150 --batch-size 2048 --cuda \
    --exp-name v6_t1_dn_universal_nmos
```
(Same for PMOS.)

**Verification gate (Tier 1):**
- `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16` against new `v6_t1_dn_*` checkpoints via `PYCIRCUITSIM_NN_CHECKPOINT_DN_*` env vars.
- **PASS criteria:** TSMC7 NMOS DC ≤ 11 % (≥3 pp improvement), no PASS-cell regression on TSMC12/16 (±0.5 pp tolerance), TSMC5 inverter transient ≤ 13 %.
- **REVERT if:** any TSMC12/16 inverter flips PASS→FAIL, or TSMC7 NMOS DC stays >12 %, or TSMC5 transient regresses past 13 %.

---

### Tier 2 — Wire `sample_class` into the trainer's LDS

**Files:**
- `external_compact_models/PyCMG/pycmg/nn_generate.py:698, 824` — column already exists
- `external_compact_models/bsimar/data/dataset.py:38-90` — loader needs to pass column through
- `external_compact_models/bsimar/losses/bni_mae.py:29-62` — multiply LDS weights by class weights

**Class weight table (initial):**

| sample_class | Code | Weight | Reason |
|---|---|---|---|
| `anchor` | 0 | 1.0 | base |
| `vds_zero` | 1 | 0.5 | rule 19 enforces Id(Vds=0)=0 — supervised label is structurally redundant |
| `subthresh` | 2 | 1.0 | base |
| `small_vds` | 3 | 1.0 | base |
| `grid` | 4 | 1.0 | base |
| `hot` | 5 | 1.5 | saturation quadrant, modest emphasis |
| `inv_trip` | 7 | 3.0 | **highest priority — direct verifier target** |
| `overshoot` | 8 | 0.0 | rule 19a learns this at inference; supervised labels here destabilized v5 TSMC7/12/16 |
| `vbs_lhs` | 9 | 1.0 | base |

**Implementation sketch:**

```python
# bsimar/losses/bni_mae.py — add a 5-line table lookup
SAMPLE_CLASS_WEIGHTS = {0: 1.0, 1: 0.5, 2: 1.0, 3: 1.0, 4: 1.0,
                        5: 1.5, 7: 3.0, 8: 0.0, 9: 1.0}

def apply_class_weights(per_sample_weights, sample_class):
    cls_w = torch.tensor([SAMPLE_CLASS_WEIGHTS[int(c)] for c in sample_class.tolist()],
                         device=per_sample_weights.device)
    return per_sample_weights * cls_w.unsqueeze(-1)
```

Backward compat: if `.npz` lacks `sample_class` (legacy v4 data), default all weights to 1.0 — no behavior change.

**Retrain:** DirectNet NMOS+PMOS on **Tier 1 dataset** (clean A/B vs Tier 1). Save under `v6_t2_dn_*`.

**Verification gate (Tier 2):**
- Same 16-cell verify.
- **PASS criteria:** ≥1 pp inverter VTC improvement on TSMC7 vs Tier 1 alone, no regression elsewhere.
- **REVERT if:** Tier 2 underperforms Tier 1 on any cell by >1 pp.

---

### Tier 3 — Inverter VTC as the model-selection metric

**Files:**
- `external_compact_models/bsimar/training/trainer.py` — replace
  `phys_best_metric` selection with a per-N-epoch mini-VTC eval.

**Mini-VTC harness:** every 25 epochs (configurable), DC-solve one inverter per tech (4 total) using the in-training checkpoint, compute mean inverter VTC NRMSE vs PyCMG. Cost: ~10 s × 32 evals over 800 epochs = ~5 min added to a 100-min run.

Lives in `bsimar/training/inverter_eval.py` (new file). It must:
1. Accept live model + norm stats (no checkpoint round-trip).
2. Build 4 inverters via existing `pycircuitsim` parser, tiny netlist string (no file I/O).
3. Sweep Vin 0→VDD at 21 points, compute Vout, compare to cached PyCMG reference VTC (precomputed once at trainer startup).
4. Return mean NRMSE_phys across 4 techs.

Selection rule: `model_best_inverter = argmin(mean inverter VTC NRMSE)`. Persist as `*_best.inv.pt` alongside `_best.pt` and `_best.phys.pt`. The simulator's `_resolve_nn_checkpoint` cascade picks `_best.inv.pt` when present.

**Retrain:** DirectNet NMOS+PMOS using Tier 1+2 dataset and class weights. Save under `v6_t3_dn_*`.

**Verification gate (Tier 3):**
- Same 16-cell verify, using `*_best.inv.pt`.
- **PASS criteria:** mean inverter VTC NRMSE strictly better than `*_best.pt` from Tier 2; no inverter cell regresses by >1 pp.
- **REVERT if:** inverter-best worse than loss-best on any cell, or mini-VTC harness adds >15 % wall-time.

---

### Tier 4 (conditional) — Promote winners to BSIMAR Transformer

Only if Tiers 1–3 ship measurable DirectNet inverter improvement. Apply same dataset (Tier 1) + class weights (Tier 2) + selector (Tier 3) to Transformer. ~3 h GPU per device.

Save under `v6_universal_{nmos,pmos}_*`. Same verification gate as DirectNet.

---

## 4. Acceptance gate (full plan)

Final shipping set must clear:

| Metric | Threshold | Source |
|---|---|---|
| TSMC7 NMOS DC NRMSE | ≤ 8 % | trim plan §B1 gate |
| Inverter VTC (all 4 TSMC) | ≤ 12 % | this plan |
| Inverter transient (all 4 TSMC) | ≤ 15 % | trim plan |
| Inverter transient TSMC5 | ≤ 10 % | tighten — currently 12 % |
| BSIM-CMG L1 byte-identical | exact | regression guard |
| 6 currently-PASSing inverter cells | no regression >0.5 pp | regression guard |

Tier 1 shipping alone is still a win — dataset improvement compounds. Tier 2 and Tier 3 are independent.

---

## 5. Diagnostic logging (always on, all tiers)

Add to `bsimar/training/trainer.py`:

1. Per-epoch: log mean `|autograd_gds|` vs mean `|gds_label|` over a fixed trip-point validation slice (Vgs, Vds ∈ trip cone, NMOS only). Ratio divergence >5× by epoch 200 → flag for §6.
2. Per-epoch: log mean `|autograd_cgg|` and `cgd-cdg` symmetry residual on same slice. Residual >10 % → flag.

These logs answer: **do we need a derivative-consistency loss?** Current evidence says no, but trip-cone data starvation may have hidden the answer. Post Tier 1 retrain, diagnostic will tell cleanly.

---

## 6. Out of scope (do NOT attempt in this plan)

- **Derivative-consistency loss on Id (∂Id/∂V vs gm/gds/gmb labels).** Inference uses `torch.autograd.grad(Id, V)` and discards predicted gm/gds/gmb columns — such a loss tightens consistency between two unused outputs. Prior attempts (SlopeMatchLoss B2) deleted unvalidated; ChargeConsistencyLoss N4 killed by asinh chain-rule mismatch. Revisit only if Tier 1 diagnostic (§5.1) shows >5× autograd-vs-label gap in trip cone.
- **Charge-consistency loss on Q (∂q/∂V vs c\*\* labels).** Same reasoning; asinh chain rule carries `cosh(asinh(q/s))` factor breaking label equivalence. If transient KCL drift becomes a problem post Tier 1, the right fix is a *targeted* Maxwell-symmetry penalty (`MSE(cgd_autograd, cdg_autograd)` — no labels, no normalization issues), not full ∂q/∂V supervised loss.
- **Per-tech checkpoints.** Breaks portability/storage; D1 evidence says gap is universal-vs-data, not universal-vs-per-tech.
- **Overshoot overlay reactivation.** Rule 19a handles `|Vds| > VDD_train` at inference. Supervised labels there caused TSMC7/12/16 NR runaway in v5.
- **AR-finetune phase / `forward_scheduled`.** Deleted 2026-05-03 trim. Cosine schedule sufficient.
- **SignConsistencyLoss / BoundaryLoss / id_gate.** All deleted, superseded by rule 19.

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tier 1 cone wrong for TSMC7 (Vth offset from VDD/2) | Medium | Cone defined at VDD/2, not Vth. D1 shows saturation plateau is under-sampled, not Vth band. |
| Tier 2 class weights de-emphasize critical region | Low-Med | Weights table is a 5-line revert; runtime knob. |
| Tier 3 mini-VTC eval too noisy | Medium | Mean of 4 inverters; if noisy, raise interval to 50 epochs and average over 3 evals. |
| Retrain breaks BSIM-CMG L1 numerics | Very low | NN training touches no BSIM-CMG path; explicit byte-identical guard. |
| Tier 1 dataset regen >1 h | Medium | Parallel with Tier 2 impl; `--workers 32`. |

---

## 8. Rollback plan

Each tier is one commit. Final branch state is whichever `git revert` / no-op combo leaves verification gate green. If all tiers fail, branch ends at `c8a783b` (post-refactor verified), plan archived as closed dead-end, postmortem added to `Future Work` in CLAUDE.md.

---

## 9. Postmortem template (fill on close)

- Tier 1 outcome: \_\_\_ (numbers)
- Tier 2 outcome: \_\_\_
- Tier 3 outcome: \_\_\_
- Tier 4 outcome (if run): \_\_\_
- Diagnostic §5.1 finding (autograd-gds vs label-gds ratio): \_\_\_
- Diagnostic §5.2 finding (cap symmetry residual): \_\_\_
- Decision on derivative-consistency loss follow-up: \_\_\_
- Final shipping set: \_\_\_
- CLAUDE.md updates needed: \_\_\_
