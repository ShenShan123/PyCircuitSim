# BSIM-AR-Full (LEVEL=76) — clean full-terminal scan

This autoregressive Transformer learns the same six independent OSDI terminal
surfaces as LEVEL=75. It emits `qg`, `qb`, `qd`, `i_d`, `i_g`, and `i_b`
autoregressively; normalization remains in the canonical full-terminal order.
Source current and charge and both full Jacobians are reconstructed at the
shared physical boundary. LEVEL=72 is the ground truth.

Clean means one uniform run per technology, polarity, and tier with
`--model transformer --swa-mode ema --seed 42` through the full-terminal-only
trainer.

Evidence pass: **V7.7.6**. Campaign manifest SHA-256 `297868d022a20f839516c417f8bf19e7c55eee51dd7844eea1860c1fad3f9b28` pins gate commit `ff4234b56ac2ea0f7f86603753b45bca6edcd9b8`, 600 jobs, and 280 checkpoint artifacts. Raw evidence: `results/v776_full_clean/`. Dataset/training source: `6be83348c1f5db6720d7504ed6dcea874a3a7418`; original source inventory SHA-256 `41f82c77a3ef2aa185f053d1a41d65ff7c4dcbe2ec446510fddf7870de107a64`. Evaluation runtime: `ff4234b56ac2ea0f7f86603753b45bca6edcd9b8`; evaluation inventory SHA-256 `d0d4ed4d52f08b08c2bcf143b940501fbacedf6b9269759f62f3db9f8907c243`. This is a separate evaluation of the saved weights; it does not claim source equivalence or retraining.

Gate definitions, strict OMP scoring, denominator rules, and comparability are
owned by [`methodology.md`](methodology.md).

## Status

LEVEL=76 is the supported autoregressive alternative. This scan reports its
evidence without changing the DirectNet-Full default.

The evidence is the clean pool of the scored (limiter-off) arm of the
[V7.7.6 evaluation](v776-evaluation.md), which also covers simple-v2, the
canary, the NN limiter and per-technology error.

## Headline — circuit gates by tier

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **19/20** | 5/5 | 4/5 | 5/5 | 5/5 | 0 | tsmc16-opamp |
| medium | **17/20** | 5/5 | 3/5 | 4/5 | 5/5 | 0 | tsmc5-opamp, tsmc12-opamp, tsmc16-sram_snm |
| large | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| xl | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |

## By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 1.30% | **PASS** 3.31% | **PASS** 3.60% | **PASS** 0.16% | **PASS** 1.88% |
| medium | **PASS** 1.06% | **PASS** 0.40% | **PASS** 1.33% | **PASS** 0.69% | **PASS** 1.23% |
| large | **PASS** 0.25% | **PASS** 0.28% | **PASS** 0.28% | **PASS** 0.52% | **PASS** 0.04% |
| xl | **PASS** 0.02% | **PASS** 0.04% | **PASS** 0.04% | **PASS** 0.40% | **PASS** 0.08% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 0.99% | **PASS** 1.08% | **PASS** 0.99% | **PASS** 0.10% | ERROR |
| medium | ERROR | **PASS** 0.21% | **PASS** 0.07% | ERROR 0.27% | **PASS** 0.41% |
| large | **PASS** 0.56% | **PASS** 0.02% | **PASS** 0.02% | **PASS** 0.19% | **PASS** 0.19% |
| xl | **PASS** 0.19% | **PASS** 0.19% | **PASS** 0.19% | **PASS** 0.24% | **PASS** 0.18% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.53% | **PASS** 1.65% | **PASS** 1.76% | **PASS** 2.42% | **PASS** 1.10% |
| medium | **PASS** 6.28% | **PASS** 1.61% | **PASS** 1.57% | **PASS** 0.84% | ERROR 1.18% |
| large | **PASS** 6.55% | **PASS** 1.64% | **PASS** 1.64% | **PASS** 0.70% | **PASS** 1.84% |
| xl | **PASS** 6.74% | **PASS** 1.56% | **PASS** 1.56% | **PASS** 0.70% | **PASS** 1.10% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 0.72% | **PASS** 0.05% | **PASS** 0.04% | **PASS** 0.13% | **PASS** 0.05% |
| medium | **PASS** 0.35% | **PASS** 0.08% | **PASS** 0.10% | **PASS** 0.44% | **PASS** 0.77% |
| large | **PASS** 0.10% | **PASS** 0.06% | **PASS** 0.06% | **PASS** 0.14% | **PASS** 0.11% |
| xl | **PASS** 0.04% | **PASS** 0.04% | **PASS** 0.04% | **PASS** 0.06% | **PASS** 0.20% |

## By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 4/4 | 3/4 | 4/4 | 4/4 | **15/16** |
| **TSMC6** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| **TSMC7** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| **TSMC12** | 4/4 | 3/4 | 4/4 | 4/4 | **15/16** |
| **TSMC16** | 4/4 | 3/4 | 3/4 | 4/4 | **14/16** |

## By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 | **19/20** |
| medium | 3/4 | 4/4 | 4/4 | 3/4 | 3/4 | **17/20** |
| large | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | **20/20** |
| xl | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | **20/20** |

## Device and AC suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.63 / 9.12 / 0.566 / 38.9 (25/26) | 1.40 / 5.26 / 0.946 / 18.8 | 1.54 / 4.69 / 0.946 / 18.8 | 0.81 / 2.64 / 0.989 / 22 | 1.40 / 4.26 / 0.985 / 46.1 | 128/129 |
| medium | 0.72 / 4.08 / 0.800 / 23.4 | 0.51 / 2.18 / 0.996 / 13.3 | 0.40 / 1.66 / 0.998 / 13.3 | 0.61 / 2.25 / 0.994 / 21.8 | 0.59 / 2.08 / 0.997 / 10.9 | 129/129 |
| large | 0.49 / 2.68 / 0.862 / 18.3 | 0.33 / 1.64 / 0.999 / 8.71 | 0.31 / 1.52 / 0.999 / 8.71 | 0.37 / 1.22 / 0.993 / 37.6 | 0.40 / 1.28 / 0.988 / 37 | 129/129 |
| xl | 0.45 / 2.50 / 0.859 / 18.5 | 0.17 / 0.91 / 1.000 / 4.03 | 0.18 / 0.95 / 1.000 / 4.03 | 0.24 / 0.81 / 0.997 / 22 | 0.20 / 0.66 / 0.994 / 28.2 | 129/129 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.57 / 3.90 / 0.982 / 376 | 1.25 / 4.66 / 0.989 / 261 | 1.24 / 4.39 / 0.990 / 254 | 1.34 / 4.75 / 0.991 / 284 | 1.14 / 3.98 / 0.992 / 208 | 100/100 |
| medium | 1.00 / 3.85 / 0.991 / 187 | 0.87 / 3.38 / 0.990 / 252 | 0.82 / 3.45 / 0.990 / 251 | 1.07 / 4.44 / 0.992 / 222 | 1.27 / 4.92 / 0.993 / 205 | 100/100 |
| large | 0.88 / 3.29 / 0.991 / 186 | 0.78 / 3.28 / 0.991 / 249 | 0.78 / 3.28 / 0.991 / 249 | 0.90 / 4.00 / 0.992 / 220 | 0.84 / 3.65 / 0.992 / 209 | 100/100 |
| xl | 0.79 / 3.17 / 0.991 / 186 | 0.75 / 3.23 / 0.991 / 251 | 0.75 / 3.23 / 0.991 / 251 | 0.78 / 3.71 / 0.987 / 320 | 0.81 / 3.61 / 0.992 / 212 | 100/100 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, valid refined reference and converged NN OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 22.994227199049917 dB | FAIL 12.2800026137878 dB | FAIL 11.43719650943067 dB | **PASS** 1.066422518178129 dB | FAIL 27.353846619140917 dB | **1/5** |
| medium | ERROR | **PASS** 1.7819081304634707 dB | **PASS** 0.3953912259779315 dB | **PASS** 1.5998585913664343 dB | **PASS** 0.14863465588352653 dB | **4/5** |
| large | FAIL 13.91291507777153 dB | **PASS** 0.25323569624814724 dB | **PASS** 0.25323569624814724 dB | **PASS** 0.9337481271208077 dB | **PASS** 0.6578404569599101 dB | **4/5** |
| xl | ERROR | **PASS** 0.06441607048673603 dB | **PASS** 0.06441607048673603 dB | **PASS** 0.42344663792481185 dB | **PASS** 0.9516343180167937 dB | **4/5** |

## Interpretation

The tables preserve every scientific failure and every declared parametric
configuration. Numeric aggregates omit only rows that produced no numeric
metric; those rows remain in the denominator as `ERROR`. Compare tiers within
this family; compare LEVEL=75 and LEVEL=76 only through their separately pinned
artifacts and identical gate matrix.

## Reproduction

Checkpoints are `tsmc{5,6,7,12,16}_tff_{small,medium,large,xl}_{nmos,pmos}`.
Each requires `_best.pt`, `_norm.npz`, `_config.npz`, and the checksum-bound
`_best.pt.complete` marker. The provenance above identifies the selected
campaign, its source commit, and its artifact hashes. See the repository
README for the complete generation, training, gate, coverage, and report
commands.
