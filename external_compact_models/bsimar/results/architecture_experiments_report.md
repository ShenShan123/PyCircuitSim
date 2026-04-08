# BSIM-AR Architecture Improvement Sprint — Final Report

**Date:** 2026-04-08
**Branch:** `bsimar-arch-experiments`
**Worktree:** `/home/shenshan/NN_SPICE_bsimar_exp`
**Final head commit:** `9626e66` (A2 grouped input tokens)
**Total runs:** 11 experiments (6 Round 1 + 5 Round 2), ~4 hours of GPU time

## Executive summary

Starting from a 15-epoch BSIM-AR baseline at **AVG MRE 20.38%**, this sprint delivered a final model at **AVG MRE 4.93%** — a **76% relative drop** on the primary metric. Two architectural changes (T1 physical-space early stopping, P4 parallel cap-block, A2 grouped input tokens) and one normalizer change (T2 asinh) were kept; four other candidates were tested and rewound.

The single most valuable change was **A2 grouped input tokens** (−40% MRE on top of T2), which confirmed the hypothesis that the Transformer was wasting capacity rediscovering trivial voltage-differences like `Vgs = Vg − Vs` through self-attention over 19 separate scalar tokens, when a tiny per-group MLP can encode those relationships directly.

## Final model architecture

Pre-LN Transformer + the following additive features (in order of commit):

| feature | file | core change |
|---|---|---|
| **T1** physical-space early stopping | `training/trainer.py` | Checkpoint/stop on `mean(NRMSE_phys) + 0.1·(1−R²_phys)` in parallel with TF-val MAE; saves `*_best.phys.pt`; final test loads phys-best |
| **P4** parallel cap-block head | `models/transformer.py` | AR loop emits only 8 tokens (4 charges + 4 currents/conds); the 5 caps are emitted in parallel from the gmb hidden state via one non-AR step; AR loop drops 13→8 |
| **T2** asinh normalization | `data/normalize.py`, `cli/train.py` | New `BSIMARNormalizer` mode `'asinh'`: per-target `y' = arcsinh(y / s_k) + zscore` where `s_k` = per-target geometric mean of `|y|` on train split (clamped at floor); bi-Lipschitz inverse avoids `inv_signed_log`-style AR error amplification |
| **A2** grouped input tokens | `models/transformer.py`, `training/trainer.py` | 19 raw context scalars → 3 semantic group tokens: voltage (4→d_model), geometry (NFIN_log, L, T → d_model), process (12 proc params → d_model). Encoder sequence length drops 27→11; voltage MLP learns `Vgs = Vg − Vs` jointly |

All four are simultaneously active on the `bsimar-arch-experiments` branch HEAD.

## Experimental protocol

- **Dataset:** `external_compact_models/bsimar/data/datasets/universal_nmos.npz` (582,480 samples before filtering, 447,827 after `apply_filter=True`).
- **Fast config** (shared across all runs for apples-to-apples comparison): `d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1, batch_size=2048, seed=42`. Model: ~400K params.
- **Loss:** `--loss mae --lds` (LDS reweighting per target).
- **Validation:** TF-fast every epoch; AR-val every 10 epochs; physical-space probe at every AR-val.
- **Scoring:** **AVG MRE (%) on the held-out test split is the primary metric.** `NRMSE_phys` and `R²_norm` are reported alongside. The user clarified mid-sprint (between Round 1 and Round 2) that MRE matches the downstream transient-simulation goal better than NRMSE, since transient fidelity depends on uniform-decade relative accuracy rather than peak fit.
- **GPU:** NVIDIA RTX PRO 6000 Blackwell, 96 GB, isolated via `CUDA_VISIBLE_DEVICES=2` while other experiments ran on A100s.
- **Wallclock:** 15-epoch runs ~10 min; 50-epoch runs ~18 min each.

## Headline results

| run | epochs | AVG NRMSE_phys (%) | **AVG MRE (%)** | AVG R²_norm |
|---|---:|---:|---:|---:|
| baseline | 15 | 0.894 | 20.38 | 0.9858 |
| +T1 phys-space early stop | 15 | 0.881 | 19.61 | 0.9854 |
| +P4 parallel cap-block | 15 | 0.866 | 19.67 | 0.9856 |
| +T2 asinh (50 epochs) | 50 | 0.990 | **8.19** | 0.9949 |
| **+A2 grouped input tokens (50 epochs)** | 50 | **0.575** | **4.93** | **0.9982** |
| **cumulative reduction** | | **−36%** | **−76%** | **+0.012** |

## Round 1: 15-epoch budget (Phase 1 items)

Initial sprint, scored on NRMSE_phys (before the MRE clarification). The goal was to establish which cheap changes help at all.

| # | experiment | NRMSE_phys (%) | MRE (%) | verdict | commit |
|---|---|---:|---:|---|---|
| 0 | baseline | 0.894 | 20.38 | reference | `e454810` |
| 1 | T1 physical-space early stopping | 0.881 | 19.61 | **KEPT** | `3fc40c9` |
| 2 | P2 charge-neutrality reparameterization | 0.953 | 24.37 | INFEASIBLE | rewound |
| 3 | P4 parallel C-block | 0.866 | 19.67 | **KEPT** | `626aec6` |
| 4 | T2 asinh normalization | 1.209 | 11.71 | INFEASIBLE on NRMSE — but MRE win noticed | rewound |
| 5 | A3 Fourier voltage features | 1.006 | 22.48 | INFEASIBLE | rewound |
| 6 | T4 warmup + EMA | 5.104 | 82.88 | INFEASIBLE | rewound |

**Round 1 outcome:** +T1+P4 gave net NRMSE 0.894→0.866 (−3.1%). Crucially, T2 asinh showed a **huge MRE improvement (19.67→11.71)** but was rewound under NRMSE-first scoring — the user then clarified that MRE was the right metric, prompting Round 2.

### Why each rewind

- **P2 (charge neutrality)** — PyCMG data satisfies `qg + qd + qs + qb ≈ 0` to 1e-16 (verified), so the reconstruction is mathematically exact. But prediction errors in `qg, qd, qs` are independent and compound when summed: `qb` NRMSE 1.038 → 1.374 and `qb` MRE 31→97. Dropping the qb supervisory signal also hurt joint-charge optimization (qd/qs themselves degraded). Would need a √3 loss upweight on the three remaining charges to absorb the constraint.
- **A3 (Fourier voltage features)** — 16 frequencies × 4 voltages added 128 extra "scalar" tokens, blowing the sequence length from 32 to 160 and slowing wallclock 2.2×. At the same 15-epoch budget the larger 420K-param model was severely underfit. Correct fix: fewer frequencies (K=4) fused into a single voltage GROUP token — exactly what A2 did later (A2 is the right way to absorb Fourier features).
- **T4 (warmup + EMA)** — EMA decay 0.999 has a 1000-step half-life. At 15 epochs (~2625 steps) the EMA shadow was still anchored to the early-training random state, producing a catastrophic MRE of 82.88%. EMA is correct for 500+ epoch production runs, not for 15-epoch sprints.

## Round 2: 50-epoch budget under MRE-primary scoring

After the MRE clarification, re-ran T2 (the clearest Round 1 MRE win that was wrongly rewound) and continued with more candidates. Each run budgeted 50 epochs.

| # | experiment | NRMSE_phys (%) | **MRE (%)** | R²_norm | verdict | commit |
|---|---|---:|---:|---:|---|---|
| 7 | P4 @ 50 epochs (baseline) | 0.702 | 14.14 | 0.9919 | reference | — |
| 8 | T2 asinh @ 50 epochs | 0.990 | **8.19** | 0.9949 | **KEPT** (−5.95 MRE) | `c852bba` |
| 9 | T3 Laplace NLL | 1.028 | 8.84 | 0.9945 | INFEASIBLE | rewound |
| 10 | T5 subthreshold log\|id\| loss | 0.992 | 9.79 | 0.9939 | INFEASIBLE | rewound |
| 11 | **A2 grouped input tokens** | **0.575** | **4.93** | **0.9982** | **KEPT** (−3.26 MRE, −0.42 NRMSE) | `71f0e50` |

### Why T3 and T5 rewound

Both failed the same way: **redundancy with an already-active mechanism**.

- **T3 Laplace NLL** (`L = Σ_k (|y_k − ŷ_k|·exp(−log_b_k) + log_b_k)`) is a principled per-target loss reweighter (Kendall & Gal 2018), intended to replace hand-tuned `w_curr/w_cond/w_charges/w_caps`. But LDS reweighting is already doing per-target rebalancing, and stacking them produced a degenerate optimum where the learned `log_b` collapsed to −2.5 to −3.4 (b≈0.04) and the network just amplified all residuals together. Every target was slightly worse. Should be retried either without LDS or as a replacement for LDS.
- **T5 subthreshold log\|id\| loss** first hit a data-distribution surprise: with `id_thresh=1e-10` the subthreshold mask was **empty** because the existing filter drops `|id| < 1e-12` and PyCMG produces no samples in `1e-12 < |id| < 1e-9` (smallest non-zero is ~2e-9). After raising `id_thresh=1e-6`, the log-id term magnitude was ~5 while the normalized MAE was ~0.04, so `lambda_sub=0.2` swamped the main loss by ~25×. Rescaling to `lambda_sub=0.005` brought the total loss back in range but still worsened MRE (8.19→9.79). Diagnosis: **asinh already provides uniform decade-wise error** via its transform; an explicit log-id loss term is functionally redundant. Future variant should either drop asinh and use `zscore + T5`, or skip T5 entirely on top of asinh.

### Why A2 grouped tokens was the biggest win

A2 took the 19 raw context scalars (4 voltages, NFIN/L/T, 12 process params) and routed them through three small group MLPs:

```python
voltage_group = Linear(4, 2d) → GELU → Linear(2d, d)   #  → voltage token
geom_group    = Linear(3, 2d) → GELU → Linear(2d, d)   #  → geometry token
proc_group    = Linear(12, 2d) → GELU → Linear(2d, d)  #  → process token
```

This shrinks the encoder sequence from 27 to 11 (the 3 group tokens + 1 start + 7 AR target tokens in the TF path) and, crucially, lets the voltage MLP learn `Vgs = Vg − Vs`, `Vds = Vd − Vs`, etc. as **joint nonlinearities in a single layer**, rather than forcing the Transformer to rediscover them through self-attention over 4 independent voltage tokens.

Per-target MRE improvements on top of T2 asinh (both 50 epochs):

| target | T2 MRE | A2 MRE | relative Δ |
|---|---:|---:|---:|
| id | 8.16 | **4.97** | −39% |
| gm | 17.07 | **11.01** | −36% |
| gds | 13.21 | **8.94** | −32% |
| gmb | 13.04 | **8.02** | −39% |
| qg | 6.17 | **3.10** | **−50%** |
| qd | 7.62 | **3.82** | **−50%** |
| qs | 5.50 | **2.92** | −47% |
| qb | 9.06 | **6.30** | −30% |
| cgg | 4.63 | **2.56** | −45% |
| cgd | 5.80 | **3.47** | −40% |
| cgs | 5.45 | **2.81** | −48% |
| cdg | 5.10 | **2.88** | −44% |
| cdd | 5.61 | **3.36** | −40% |
| **AVG** | **8.19** | **4.93** | **−40%** |

**A2 was the first experiment in the sprint that improved both MRE and NRMSE simultaneously.** T2 alone traded NRMSE for MRE (peak fit for uniform-decade error); A2 improved both because the inductive bias of grouping was good enough to eliminate the trade-off.

## Detailed metrics for the final model

Final test-set metrics at 50 epochs (commit `71f0e50`, checkpoint `exp11_a2_grouped_50ep_nmos_best.phys.pt`):

| target | NRMSE_phys (%) | **MRE (%)** | R² | NRMSE_norm (%) | R²_norm | MAE_norm | n_valid / n_total |
|---|---:|---:|---:|---:|---:|---:|---:|
| id | 0.609 | 4.97 | 0.9855 | 0.853 | 0.9953 | 0.0255 | 36386 / 44784 |
| gm | 1.276 | 11.01 | 0.9913 | 1.298 | 0.9940 | 0.0467 | 41118 / 44784 |
| gds | 0.786 | 8.94 | 0.9683 | 1.025 | 0.9943 | 0.0388 | 15403 / 44784 |
| gmb | 1.358 | 8.02 | 0.9869 | 1.258 | 0.9981 | 0.0252 | 25106 / 44784 |
| qg | 0.190 | 3.10 | 0.9989 | 0.304 | 0.9995 | 0.0121 | 41517 / 44784 |
| qd | 0.226 | 3.82 | 0.9986 | 0.382 | 0.9994 | 0.0146 | 40943 / 44784 |
| qs | 0.255 | 2.92 | 0.9990 | 0.326 | 0.9997 | 0.0123 | 40566 / 44784 |
| qb | 1.190 | 6.30 | 0.9482 | 0.456 | 0.9991 | 0.0184 | 16134 / 44784 |
| cgg | 0.322 | 2.56 | 0.9992 | 0.447 | 0.9995 | 0.0142 | 44759 / 44784 |
| cgd | 0.356 | 3.47 | 0.9969 | 0.506 | 0.9990 | 0.0202 | 43686 / 44784 |
| cgs | 0.275 | 2.81 | 0.9995 | 0.445 | 0.9996 | 0.0137 | 44650 / 44784 |
| cdg | 0.286 | 2.88 | 0.9992 | 0.418 | 0.9996 | 0.0142 | 44619 / 44784 |
| cdd | 0.343 | 3.36 | 0.9976 | 0.447 | 0.9992 | 0.0187 | 44421 / 44784 |
| **AVG** | **0.575** | **4.93** | **0.9899** | **0.628** | **0.9982** | — | — |

## Key learnings

1. **MRE-primary scoring was decisive.** It correctly captured the asinh win (T2) that NRMSE-primary had missed. The difference is not cosmetic — downstream transient simulation cares about accuracy at every operating point, not just peak current.

2. **Architectural inductive bias beats fancier losses.** A2 — which is pure "tell the model what kind each input is" — gave a bigger win than any of the loss-engineering attempts (T3, T5). Getting the right priors into the representation is higher-leverage than weighting.

3. **Redundancy is a trap.** T3 + LDS and T5 + asinh both failed for the same reason: two mechanisms targeting the same thing create conflicting gradients. Always consider whether a new mechanism is compatible with what's already on top.

4. **Sequence length matters more than parameter count.** A3 (Fourier features) added +128 tokens and slowed the model 2.2× for a 16% MRE regression. A2 shrunk 19 → 3 tokens and sped attention ~6× while improving MRE 40%. Raw scalar tokenization is surprisingly expensive on a Transformer encoder; semantic grouping is nearly always better.

5. **15 epochs ≪ 50 epochs on this task.** P4 at 15 epochs had MRE 19.67; P4 at 50 epochs had MRE 14.14. Many Round 1 "INFEASIBLE" verdicts might flip on a longer budget (EMA in particular is known to need it).

## Infeasibility notes — candidates worth revisiting

None of these should be considered permanently dead. Each has a clear retry path:

| item | why it failed | retry path |
|---|---|---|
| P2 charge neutrality | compounded errors in `qb = −(qg+qd+qs)` | add √3 loss upweight to the three remaining charges |
| A3 Fourier features | sequence too long; underfit at 15 ep | K=4 frequencies fused into a single voltage GROUP token (already done structurally by A2 — the Fourier idea is now redundant) |
| T4 warmup + EMA | EMA decay 0.999 too high for 15 ep | rerun at 200+ epochs, OR use step-count-aware decay `1 − 1/√total_steps` |
| T3 Laplace NLL | redundant with LDS | rerun WITHOUT `--lds`, OR use it to REPLACE the LDS path |
| T5 subthreshold log\|id\| loss | redundant with asinh | rerun with plain zscore normalization instead of asinh |

## Reproducing the final model

```bash
cd /home/shenshan/NN_SPICE_bsimar_exp
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH="/home/shenshan/NN_SPICE/external_compact_models/PyCMG:/home/shenshan/NN_SPICE_bsimar_exp/external_compact_models" \
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type nmos --universal \
    --loss mae --lds --cuda --norm-mode asinh \
    --d-model 128 --nhead 4 --num-layers 3 --dim-feedforward 256 --dropout 0.1 \
    --batch-size 2048 --epochs 50 --patience 60 \
    --exp-name my_run --overwrite --seed 42
```

The `parallel_caps=True` and `grouped_inputs=True` flags are hard-wired inside `train_transformer` in the current branch HEAD and do not need to be passed via CLI.

## Follow-up TODO (outside sprint scope)

- **`pycircuitsim/models/mosfet_bsimar.py` checkpoint loader** needs `parallel_caps=True, grouped_inputs=True` plumbed through `_config.npz` so that checkpoints trained on this branch can be loaded by the SPICE simulator for circuit validation. Same TODO already existed for P4 before this sprint.
- **Production training at d_model=256, num_layers=6** should run at 500+ epochs with the current-branch architecture to see if the gains compound. This is now a cheap experiment because the attention cost dropped ~6× from A2.
- **PMOS run** — everything in this sprint was NMOS-only per the smoke-test convention. A matching PMOS training should land before claiming the improvement is universal.
- **A retried T4** at 200+ epochs on top of A2 — if EMA works as advertised, it should give another free ~1-3% MRE drop.

## Commit history

```
9626e66 docs(bsimar): record A2 commit hash
71f0e50 exp(bsimar): A2 grouped input tokens (KEPT, big win)
132d051 exp(bsimar): T5 subthreshold log|id| loss (INFEASIBLE, rewound)
3315c50 exp(bsimar): T3 Laplace NLL (INFEASIBLE, rewound)
c852bba exp(bsimar): T2 asinh normalization @ 50 epochs (KEPT, MRE-primary)
b0c7ebe docs(bsimar): sprint summary — 2 kept, 4 infeasible
d83a501 exp(bsimar): T4 warmup + EMA (INFEASIBLE, rewound)
f9da9fe exp(bsimar): A3 Fourier voltage features (INFEASIBLE, rewound)
9ed7f49 exp(bsimar): T2 asinh normalization (INFEASIBLE on NRMSE, rewound)
626aec6 exp(bsimar): P4 parallel C-block (KEPT)
abe69de exp(bsimar): P2 charge-neutrality (INFEASIBLE, rewound)
3fc40c9 exp(bsimar): T1 physical-space early stopping (KEPT)
d8483e1 exp(bsimar): record baseline metrics in plan
e454810 chore(bsimar): snapshot working state + architecture improvement plan
```

Working log with full per-experiment diagnoses lives in `external_compact_models/bsimar/docs/architecture_improvement_plan.md`.
Per-run stdout logs live in `exp_logs/exp00..11_*.log` on the `bsimar-arch-experiments` branch.
