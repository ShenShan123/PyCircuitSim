# B6 — Adversarial Harvest-then-Retrain (Track B, Tier 2)

**Date:** 2026-05-29 · **Branch:** `feat/v6.4.5`
**Verdict: KILL (in-box interpolation failure; retrain blocked)**
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" (harvest/analysis); GPU=3 (retrain attempt)`

## Mechanism

Run the V6.4.4 canonical TSMC7 DirectNet on (a) the 5-stage ring-oscillator transient
(1.2 ns, 2 ps step) and (b) the SRAM 6T force_ic `.op` (both storage states).
Capture every NN operating point (vd, vg, vs, vb, NFIN, L in NN frame).
Evaluate PyCMG BSIM-CMG at the same points to compute residual |id_NN − id_CMG|.
Check whether the worst-residual points are inside or outside the training box.

## Step 1 — Harvest Collection

| Source | NMOS raw pts | PMOS raw pts |
|--------|-------------|-------------|
| Ring oscillator (5-stage, 1.2 ns) | 115,505 | 115,505 |
| SRAM force_ic state1 (q=VDD, qb=0) | 648 | 324 |
| SRAM force_ic state0 (q=0, qb=VDD) | 648 | 324 |
| **Total raw** | **116,801** | **116,153** |
| **Unique (after dedup)** | **14,587** | **14,467** |

Both states settled at the interior attractor: q≈0.8151, qb≈0.2264 (TSMC7),
consistent with B4 and B2 findings.

## Step 2 — Residual Computation

PyCMG (BSIM-CMG OSDI, TSMC7 ulvt, L=16nm NMOS / L=20nm PMOS, NFIN=2, T=300.15K)
evaluated at every unique operating point.

### NMOS residual statistics (n=14,587 valid)

| Metric | Value |
|--------|-------|
| mean \|Δid\| | 6.03 µA |
| p25 | 0.04 µA |
| p50 | 1.24 µA |
| p75 | 10.51 µA |
| p90 | 14.05 µA |
| p95 | 17.75 µA |
| max | 266.6 µA |

### PMOS residual statistics (n=14,467 valid)

| Metric | Value |
|--------|-------|
| mean \|Δid\| | 2.63 µA |
| p25 | 0.13 µA |
| p50 | 0.21 µA |
| p75 | 3.65 µA |
| p90 | 6.02 µA |
| p95 | 6.36 µA |
| max | 224.3 µA |

### NMOS residuals by Vd regime

| Vd regime | n pts | mean \|Δid\| | p90 \|Δid\| | max \|Δid\| |
|-----------|-------|-------------|------------|------------|
| Vd < −0.225 (out-of-box reverse) | 45 | 100.8 µA | 179.7 µA | 266.6 µA |
| Vd ∈ [−0.225, 0) (reverse_vds class) | 1,667 | 12.8 µA | 23.9 µA | 169.4 µA |
| Vd ∈ [0, 0.05] (near-zero Vds) | 3,006 | 7.8 µA | 13.7 µA | 18.9 µA |
| Vd ∈ (0.05, VDD] (normal on) | 8,427 | 4.3 µA | 15.2 µA | 68.4 µA |
| Vd > VDD (overshoot) | 1,442 | 1.7 µA | 5.1 µA | 113.6 µA |

**SRAM attractor region** (NMOS: Vd ∈ [0.05, 0.35], Vg > 0.55 — the operating point
of Mnl at the interior attractor q≈0.815, qb≈0.226):
n=1,295 pts, mean=15.93 µA, max=19.63 µA — all **fully in-box**.

## Step 3 — Training Box Analysis (Falsifier)

Training box from `tsmc7_nmos.npz` / `tsmc7_pmos.npz`:
- NMOS: Vd ∈ [−0.225, 1.5], Vg ∈ [0, 1.5], Vs = 0, Vb ∈ [−0.541, 0.75]
- PMOS: Vd ∈ [−1.5, 0.225], Vg ∈ [−1.5, 0], Vs = 0, Vb ∈ [−0.75, 0.541]

### Top-5% worst-residual points: in-box vs out-of-box

| Device | n valid | Top-5% n | In-box | Out-of-box | **VERDICT** |
|--------|---------|----------|--------|------------|-------------|
| NMOS | 14,587 | 729 | 672 (92.2%) | 57 (7.8%) | **IN-BOX** |
| PMOS | 14,467 | 723 | 618 (85.5%) | 105 (14.5%) | **IN-BOX** |
| Full dist | 14,587 | 14,587 | 90.6% | 9.4% | **IN-BOX** |

Out-of-box breakout by dimension:
- NMOS out-of-box: Vd axis (36), Vg axis (5), Vs axis (16), Vb (0)
- PMOS out-of-box: Vd axis (93), Vg axis (13), Vs axis (0), Vb (0)

The 7–14% out-of-box fraction is almost entirely at extreme reverse-Vds
(Vd < −0.225 V for NMOS, Vd > 0.225 V for PMOS) corresponding to NR
voltage ringing past the supply rail in the RO transient. These are covered
by the `overshoot` sample class boundary but not the `reverse_vds` corridor
inside [-0.225, 0).

**Falsifier verdict: IN-BOX.**
92% (NMOS) and 86% (PMOS) of the worst-residual operating points are
**inside the training distribution**. This is interpolation failure, not
extrapolation. Adding more in-distribution data cannot fix a model that
already has training data in the same region but fails to model it correctly.

### Root-cause analysis

The key failure modes identified:

1. **Reverse-Vds corridor (in-box):** Vd ∈ [−0.225, 0), mean error 12.8 µA.
   The `reverse_vds` sample class (class 10, ~7.5% of dataset) was added in
   V6.3 to teach this regime, but the model still has 10–24 µA errors here.
   This is the dominant RO error source (inverter output swings through this
   regime on every switching transition).

2. **Near-zero Vds (in-box, SRAM attractor):** Vd ∈ [0, 0.05], mean 7.8 µA.
   The SRAM attractor at (Vd≈0.15–0.20, Vg≈0.75) shows 15–20 µA residuals,
   all in-box. The NN predicts the wrong Id at low-Vds/high-Vgs even though
   the training set has points in this region (`small_vds` and `vds_zero`
   sample classes). This is the driving force for the SRAM interior attractor.

3. **Largest individual errors:** out-of-box at Vd < −0.225 V (max 267 µA),
   but these are sparse (45 NMOS points out of 14,587) and don't dominate
   the circuit-level error.

## Step 4 — Augmented Dataset (built, retrain blocked)

Augmented datasets built successfully:
- `tsmc7_nmos_b6.npz`: 2,078,136 + 20,000 = 2,098,136 rows (NMOS)
- `tsmc7_pmos_b6.npz`: 2,424,313 + 20,000 = 2,444,313 rows (PMOS)

**Retrain attempt blocked:** The trainer's LDS weight computation
(`compute_lds_weights_per_target` using `KBinsDiscretizer`) requires
iterating over 1.6M train-split rows × 13 output columns. For the 2M-row
augmented dataset, this step alone takes > 40 minutes (estimated from CPU
utilization and timing). With 3 seeds × 2 devices = 6 runs, total LDS
overhead would be > 4 hours before any gradient updates.

**Secondary kill reason:** Adding 20,000 rows to a 2,078,136-row dataset
(0.97% increase) in the same distribution as the existing data cannot
reasonably fix an in-box interpolation failure. The in-box falsifier makes
retrain completion unnecessary — the negative result is predictable.

**No b6 checkpoints were trained.** `b6_tsmc7_s{42,7,17}_{nmos,pmos}` do not
exist on disk.

## Baseline Metrics (V6.4.4 canonical TSMC7)

Scored by `eval_v6_4_5_candidate.py` (TSMC7, canonical checkpoints):

| Metric | V6.4.4 Baseline | B6 target |
|--------|----------------|-----------|
| RO period error | 8.98% | ≤ 5% |
| inv VTC NRMSE | 3.89% | ≤ 2.5% |
| inv tran post NRMSE | 1.10% | ≤ 5% |
| SRAM rail_snap_resid | 0.302 | < 0.05 |
| opamp_flat_flag | 1 (flat) | 0 |

Note: inv VTC NRMSE measured by this scorer as 3.89%; CLAUDE.md V6.4.4 baseline
of 2.37% was from a different evaluator (eval_v6_3_1_inverter.py). The scorer
measures both and both use NGSPICE BSIM-CMG ground truth.

## Promote / Kill Criteria

**PROMOTE** iff: RO ≤ 5% AND SRAM ≥ 2/4 snap AND inv VTC NRMSE ≤ 2.5%.
**KILL** iff: worst-residual points are in-box (interpolation failure).

**→ KILL.**

The in-box falsifier fires conclusively for both NMOS (92.2%) and PMOS (85.5%).
The worst-residual operating points for both circuit failures (RO reverse-Vds
regime and SRAM low-Vds/high-Vgs regime) are INSIDE the training box. This
means the NN already has training data in the failure region but fails to model
it correctly — a property of the network's capacity/architecture, not its
training distribution.

## Next steps (Tier-3 recommendations)

Closing either gate requires architectural changes, not data augmentation:

1. **B9 — Monotone lattice / physics-skeleton network:** Hard monotonicity
   constraints (Id monotone in Vgs, Vds) enforced via a lattice or
   monotone neural network architecture. Targets the SRAM low-Vds/high-Vgs
   failure where Id should be strictly monotone in both voltage axes.

2. **B7 — Physics skeleton (analytical Ids + NN correction):** Add an
   analytic long-channel MOSFET baseline (simple square-law or VT-based
   subthreshold + linear/saturation) and train the NN to predict only the
   residual. Constrains the output manifold topology.

3. **Track-B Tier-2 remaining — B8 data density:** Instead of B6's
   harvest-and-retrain, regenerate the `reverse_vds` and `small_vds` sample
   classes with 10× density in the high-error sub-regions and retrain on
   the original-size dataset (not augmented) — avoiding the LDS bottleneck.

## On-disk artifacts

- `results/v6_4_5_track_b/B6_harvest/harvest_cache.npz` — raw harvest (116k pts)
- `results/v6_4_5_track_b/B6_harvest/residuals_{nmos,pmos}.npz` — residuals + inputs
- `external_compact_models/bsimar/data/datasets/tsmc7_{nmos,pmos}_b6.npz` — augmented
  datasets (built but retrain blocked; kept for possible B8 reuse)
- `experiments/v6_4_5_track_b/B6_harvest_retrain.py` — harvest orchestrator
- `experiments/v6_4_5_track_b/B6_fast_augment.py` — fast augment + retrain script
- `experiments/v6_4_5_track_b/B6_augment_and_train.py` — earlier augment script
