# DirectNet-Full (LEVEL=75) — clean full-terminal scan

This default DirectNet family learns six independent OSDI terminal surfaces:
`i_d`, `i_g`, `i_b`, `qd`, `qg`, and `qb`. Source current and charge are
closed analytically, and the solver stamps the full current and charge
Jacobians. LEVEL=72 on the identical BSIM-CMG OSDI model is the reference.

Clean means one uniform run per technology, polarity, and tier with
`--swa-mode ema --seed 42` through the full-terminal-only trainer.

Evidence pass: **V7.7.6**. Campaign manifest SHA-256 `297868d022a20f839516c417f8bf19e7c55eee51dd7844eea1860c1fad3f9b28` pins gate commit `ff4234b56ac2ea0f7f86603753b45bca6edcd9b8`, 600 jobs, and 280 checkpoint artifacts. Raw evidence: `results/v776_full_clean/`. Dataset/training source: `6be83348c1f5db6720d7504ed6dcea874a3a7418`; original source inventory SHA-256 `41f82c77a3ef2aa185f053d1a41d65ff7c4dcbe2ec446510fddf7870de107a64`. Evaluation runtime: `ff4234b56ac2ea0f7f86603753b45bca6edcd9b8`; evaluation inventory SHA-256 `d0d4ed4d52f08b08c2bcf143b940501fbacedf6b9269759f62f3db9f8907c243`. This is a separate evaluation of the saved weights; it does not claim source equivalence or retraining.

Gate definitions, strict OMP scoring, denominator rules, and comparability are
owned by [`methodology.md`](methodology.md).

## Status

LEVEL=75 is the maintained default. This report still gates scientific claims
on the measured rows; default status does not turn failures into passes.

The evidence is the clean pool of the scored (limiter-off) arm of the
[V7.7.6 evaluation](v776-evaluation.md), which also covers simple-v2, the
canary, the NN limiter and per-technology error.

## Headline — circuit gates by tier

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **18/20** | 5/5 | 5/5 | 5/5 | 3/5 | 0 | tsmc6-switchcap, tsmc7-switchcap |
| medium | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| large | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| xl | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |

## By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 0.82% | **PASS** 0.09% | **PASS** 0.09% | **PASS** 1.14% | **PASS** 1.99% |
| medium | **PASS** 0.13% | **PASS** 0.47% | **PASS** 0.47% | **PASS** 0.17% | **PASS** 0.38% |
| large | **PASS** 0.00% | **PASS** 0.10% | **PASS** 0.93% | **PASS** 0.01% | **PASS** 0.08% |
| xl | **PASS** 0.40% | **PASS** 0.05% | **PASS** 0.05% | **PASS** 0.08% | **PASS** 0.25% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 7.50% | **PASS** 4.62% | **PASS** 4.62% | **PASS** 1.95% | **PASS** 0.20% |
| medium | **PASS** 0.69% | **PASS** 1.54% | **PASS** 1.54% | **PASS** 0.78% | **PASS** 1.73% |
| large | **PASS** 0.19% | **PASS** 3.43% | **PASS** 0.04% | **PASS** 0.10% | **PASS** 0.30% |
| xl | **PASS** 0.07% | **PASS** 7.51% | **PASS** 7.51% | **PASS** 0.19% | **PASS** 0.04% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.10% | **PASS** 1.54% | **PASS** 1.54% | **PASS** 3.71% | **PASS** 1.60% |
| medium | **PASS** 6.54% | **PASS** 1.74% | **PASS** 1.74% | **PASS** 1.14% | **PASS** 1.30% |
| large | **PASS** 7.08% | **PASS** 1.96% | **PASS** 1.79% | **PASS** 1.10% | **PASS** 2.32% |
| xl | **PASS** 6.70% | **PASS** 1.64% | **PASS** 1.64% | **PASS** 2.00% | **PASS** 3.85% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 0.15% | FAIL 0.27%† | FAIL 0.27%† | **PASS** 0.07% | **PASS** 0.08% |
| medium | **PASS** 0.02% | **PASS** 0.03% | **PASS** 0.03% | **PASS** 0.10% | **PASS** 0.07% |
| large | **PASS** 0.02% | **PASS** 0.03% | **PASS** 0.00% | **PASS** 0.01% | **PASS** 0.10% |
| xl | **PASS** 0.06% | **PASS** 0.10% | **PASS** 0.10% | **PASS** 0.01% | **PASS** 0.08% |

† failed on **hold droop**, the half of this gate the headline number does not show — the metric above is inside its threshold.

## By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| **TSMC6** | 4/4 | 4/4 | 4/4 | 3/4 | **15/16** |
| **TSMC7** | 4/4 | 4/4 | 4/4 | 3/4 | **15/16** |
| **TSMC12** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| **TSMC16** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |

TSMC6 and TSMC7 count as two campaign columns but remain one controlled
ground-truth repeat.

## By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 4/4 | 3/4 | 3/4 | 4/4 | 4/4 | **18/20** |
| medium | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | **20/20** |
| large | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | **20/20** |
| xl | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | **20/20** |

## Device and AC suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.46 / 7.77 / 0.094 / 47.4 (25/26) | 1.39 / 5.46 / 0.976 / 13.5 | 1.33 / 5.34 / 0.976 / 13.5 | 0.47 / 2.14 / 0.998 / 14.5 | 0.75 / 2.59 / 0.995 / 20.3 | 128/129 |
| medium | 0.47 / 2.56 / 0.900 / 16.3 | 0.57 / 2.13 / 0.995 / 10 | 0.52 / 1.92 / 0.995 / 10 | 0.34 / 1.22 / 0.981 / 50.3 | 0.36 / 0.88 / 0.994 / 24.4 | 129/129 |
| large | 0.53 / 2.72 / 0.857 / 18.6 | 0.36 / 1.38 / 0.998 / 4.73 | 0.37 / 1.40 / 0.998 / 15.3 | 0.91 / 2.21 / 0.617 / 269 (29/30) | 1.27 / 2.65 / 0.575 / 243 (25/26) | 127/129 |
| xl | 0.72 / 3.02 / 0.864 / 18.1 | 0.81 / 2.16 / 0.986 / 53.5 | 0.80 / 1.93 / 0.986 / 53.5 | 1.08 / 2.10 / 0.649 / 271 (29/30) | 1.57 / 3.68 / 0.676 / 201 (24/26) | 126/129 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.43 / 5.35 / 0.992 / 212 | 1.15 / 3.98 / 0.990 / 255 | 1.15 / 3.98 / 0.990 / 255 | 1.33 / 4.69 / 0.992 / 224 | 1.08 / 4.62 / 0.993 / 205 | 100/100 |
| medium | 0.84 / 3.36 / 0.991 / 185 | 0.81 / 3.28 / 0.991 / 248 | 0.81 / 3.28 / 0.991 / 248 | 0.74 / 3.71 / 0.992 / 219 | 0.76 / 3.69 / 0.993 / 209 | 100/100 |
| large | 0.84 / 3.23 / 0.991 / 185 | 0.83 / 3.44 / 0.991 / 246 | 0.82 / 3.26 / 0.991 / 247 | 0.73 / 3.73 / 0.992 / 220 | 0.90 / 3.92 / 0.985 / 304 | 100/100 |
| xl | 0.81 / 3.33 / 0.991 / 185 | 0.96 / 3.66 / 0.987 / 309 | 0.96 / 3.66 / 0.987 / 309 | 0.78 / 4.08 / 0.992 / 220 (18/20) | 0.74 / 3.60 / 0.993 / 210 | 98/100 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| xl | ✓ / ✓ | ✓ / ✗ gain 1.532903659042267 dB, mag 18.43735370877581 % | ✓ / ✗ gain 1.532903659042267 dB, mag 18.43735370877581 % | ✓ / ✓ | ✓ / ✓ | **8/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, valid refined reference and converged NN OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 35.88721989290879 dB | FAIL 20.52463875732854 dB | FAIL 20.52463875732854 dB | FAIL 29.633892841031216 dB | FAIL 4.981802113367884 dB | **0/5** |
| medium | FAIL 11.863560313084278 dB | FAIL 6.933160269989429 dB | FAIL 6.933160269989429 dB | FAIL 4.643728409274331 dB | FAIL 9.130428912890011 dB | **0/5** |
| large | FAIL 22.313381041297724 dB | FAIL 15.871229925437895 dB | **PASS** 2.7258360473760916 dB | FAIL 3.0989639478506703 dB | **PASS** 2.8957350278218428 dB | **2/5** |
| xl | FAIL 12.825476519917373 dB | FAIL 19.370854421110188 dB | FAIL 19.370854421110188 dB | **PASS** 2.201414951034792 dB | **PASS** 0.3479427676108955 dB | **2/5** |

## Interpretation

The tables preserve every scientific failure and every declared parametric
configuration. Numeric aggregates omit only rows that produced no numeric
metric; those rows remain in the denominator as `ERROR`. Tier comparisons are
within this family, and the TSMC6 column is the controlled TSMC7 repeat rather
than an independent ground truth.

Promotion requires a complete tier to satisfy the declared circuit, device,
and AC gates. Read scientific failures and explicit `ERROR` rows directly from
the tables rather than inferring them from model size.

## Reproduction

Checkpoints are `tsmc{5,6,7,12,16}_dnf_{small,medium,large,xl}_{nmos,pmos}`.
Each requires `_best.pt`, `_norm.npz`, and the checksum-bound
`_best.pt.complete` marker. The provenance above identifies the selected
campaign, its source commit, and its artifact hashes. See the repository
README for the complete generation, training, gate, coverage, and report
commands.
