# BSIM-AR (LEVEL=74) — the clean recipe

**What it is.** An autoregressive Transformer that emits the thirteen outputs
one token at a time, each conditioned on the ones already produced. It shares
DirectNet's data, normalization, loss, training and evaluation pipeline
entirely — the `neural_network` package — so the only variable against
[`DirectNet-L73-clean.md`](DirectNet-L73-clean.md) is the architecture.

**What it costs.** The autoregressive loop re-enters the network once per
output token, and inference is memory-bandwidth-bound on the weights, so cost
tracks *parameters × times the checkpoint is streamed*. That is why BSIM-AR
runs ~40× DirectNet rather than ~5×, and why DirectNet stays production
regardless of which family scores higher.

**What "clean" means.** One training run, no addendum:
`--apply-filter off --swa-mode ema --seed 42`. BSIM-AR's resolver-visible slots
`tsmc{X}_tf_{size}_*` **are** the clean checkpoints — the curriculum variants
live under their own stems and are covered in
[`BSIM-AR-L74-recipes.md`](BSIM-AR-L74-recipes.md).

**Measurement provenance.** The datasets and S/M/L/XL checkpoints are the
preserved V7.4.0 clean rebuild, on exactly the same data and recipe as
DirectNet. Every generated table below was remeasured in the complete V7.5.17
CPU-pinned pass after the coverage-audit contracts were enforced. Campaign
manifest SHA-256
`b3fd59028cd5ec6961f329ba5b1d9205c4d835dded27a81c9683cd7cef06195d`
pins gate commit `db1b2958e17c72c6b6506fe43efe34e17cd97859` and all 280
checkpoint artifacts. The V7.5.15 audit trail remains in
[`simple-circuits-recheck-2026-08-19.md`](simple-circuits-recheck-2026-08-19.md).

Gate definitions, strict-OMP scoring, comparability, and evidence rules:
[`methodology.md`](methodology.md).

| tier | shape | params | CPU cost, 1 thread |
|---|---|---|---|
| `small` | — | 0.67 M | — |
| `medium` | — | **1.94 M** | **61.5 ms/eval** |
| `large` | — | 5.02 M | — |
| `xl` | 384 × 8L, ff1536 | 14.81 M | — |

Tier names compare a family with itself, never across families: BSIM-AR
`small` is already larger than DirectNet `large`.

> **Denominators changed in V7.3.0.** TSMC6 folds into the headline, so complex
> totals are **/20**, device AC **/10**, opamp AC **/5**, where every earlier
> report said /16, /8 and /4. No total here is comparable to a pre-V7.3.0 total
> without rescaling.

TSMC6 remains the controlled repeat defined in `methodology.md` §7.

---

## 1. Headline — complex gates by tier

Strict: a cell passes only if it passes at OMP ∈ {1, 2, 4}.

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **9/20** | 4/5 | 0/5 | 1/5 | 4/5 | 0 | tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap |
| medium | **9/20** | 2/5 | 0/5 | 2/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm |
| large | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |
| xl | **11/20** | 2/5 | 0/5 | 4/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc7-sram_snm |

## 2. By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 3.56% | **PASS** 4.33% | FAIL 5.86% | **PASS** 2.36% | **PASS** 1.58% |
| medium | FAIL 5.42% | FAIL 6.32% | FAIL 6.63% | **PASS** 1.61% | **PASS** 1.51% |
| large | FAIL 6.95% | FAIL 7.20% | FAIL 7.20% | **PASS** 1.78% | **PASS** 1.63% |
| xl | FAIL 7.77% | FAIL 12.10% | FAIL 10.15% | **PASS** 1.99% | **PASS** 2.00% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL | FAIL | FAIL | FAIL | FAIL |
| medium | FAIL | FAIL | FAIL | FAIL | FAIL |
| large | FAIL | FAIL | FAIL | FAIL | FAIL |
| xl | FAIL | FAIL | FAIL | FAIL | FAIL |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.26% | FAIL 1.28%† | FAIL 1.69%† | FAIL 0.64%† | FAIL 1.44%† |
| medium | FAIL 4.92%† | FAIL 1.15%† | FAIL 1.16%† | **PASS** 1.38% | **PASS** 2.69% |
| large | **PASS** 5.83% | **PASS** 1.61% | **PASS** 1.61% | **PASS** 2.07% | **PASS** 1.49% |
| xl | **PASS** 6.33% | **PASS** 1.74% | FAIL 1.66%† | **PASS** 2.44% | **PASS** 2.03% |

† failed on **lobe positivity**, the half of this gate the headline number does not show — the metric above is inside its threshold.

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 0.84%† | **PASS** 2.39% | **PASS** 2.62% | **PASS** 4.15% | **PASS** 3.28% |
| medium | **PASS** 2.20% | **PASS** 2.57% | **PASS** 2.52% | **PASS** 4.11% | **PASS** 3.36% |
| large | **PASS** 2.37% | **PASS** 2.58% | **PASS** 2.58% | **PASS** 4.28% | **PASS** 3.48% |
| xl | **PASS** 2.50% | **PASS** 2.65% | **PASS** 2.61% | **PASS** 4.25% | **PASS** 3.39% |

† failed on **hold droop**, the half of this gate the headline number does not show — the metric above is inside its threshold.

## 3. By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 1/4 | 0/4 | 3/4 | 3/4 | **7/16** |
| **TSMC6** | 1/4 | 0/4 | 2/4 | 4/4 | **7/16** |
| **TSMC7** | 0/4 | 0/4 | 1/4 | 4/4 | **5/16** |
| **TSMC12** | 4/4 | 0/4 | 3/4 | 4/4 | **11/16** |
| **TSMC16** | 4/4 | 0/4 | 3/4 | 4/4 | **11/16** |

Rings and Miller opamp fixed points are the two weaknesses under the current
solver contract. `small` keeps four of five rings but loses TSMC5 switchcap on
hold droop; every larger tier keeps two rings and all five switchcaps. Every
tier is 0/5 on opamp because the selected NN fixed points do not converge at
the final physical homotopy step.

The old V7.4 report's **TSMC12 `xl` collapse is retracted**. All twelve raw
runs in that group carried two completion markers and were correctly collected
as `RACED`, but the report builder counted them as scientific failures. The
fresh isolated pass restores the TSMC12-`xl` SRAM, ring and switchcap passes;
only its opamp remains a real failure in that group.

The controlled repeat is still doing its job. TSMC6 and TSMC7 reproduce
**14/16** verdicts across the four tiers. The splits are the noisy `small` ring
cell (4.33 % PASS vs 5.86 % FAIL) and the `xl` SRAM lobe-positivity gate. Every
opamp now fails the honest convergence contract, and every switchcap verdict
agrees.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 2/4 | 2/4 | 1/4 | 2/4 | 2/4 | **9/20** |
| medium | 1/4 | 1/4 | 1/4 | 3/4 | 3/4 | **9/20** |
| large | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |
| xl | 2/4 | 2/4 | 1/4 | 3/4 | 3/4 | **11/20** |

The V7.5.17 strict curve is **9 → 9 → 12 → 11/20**. The V7.5.16
13 → 12 → 12 → 12 curve is superseded by this audited pass, and the still
older 18 → 17 → 15 → 13 curve combined nonconverged opamp fixed points with
race-corrupted TSMC12-`xl` cells. Capacity does not recover any Miller opamp;
`large` is the current circuit peak and `xl` loses one SRAM cell.

`large` has the best circuit score at 12/20. `medium` has the best parametric
DC result at 104/129, while `xl` has the best transient result at 92/100.
No tier dominates all three views.

## 5. Device-level suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 7.69 / 38.99 / -1.970 / 494 (20/26) | 9.78 / 38.45 / -3.760 / 484 (19/26) | 11.12 / 43.82 / -3.645 / 484 (15/21) | 4.32 / 14.04 / -2.254 / 198 (27/30) | 5.08 / 17.06 / -2.215 / 198 (22/26) | 103/129 |
| medium | 7.47 / 36.23 / -1.955 / 494 (20/26) | 9.62 / 37.23 / -3.640 / 484 (20/26) | 11.04 / 42.59 / -3.614 / 484 (15/21) | 4.62 / 14.01 / -2.287 / 199 (27/30) | 5.21 / 17.01 / -2.178 / 198 (22/26) | 104/129 |
| large | 7.69 / 36.39 / -1.874 / 494 (20/26) | 9.93 / 37.61 / -3.526 / 484 (19/26) | 11.16 / 42.84 / -3.526 / 484 (15/21) | 4.93 / 14.97 / -2.275 / 198 (25/30) | 5.32 / 16.55 / -2.190 / 198 (22/26) | 101/129 |
| xl | 7.64 / 36.13 / -1.827 / 494 (20/26) | 10.55 / 38.88 / -3.183 / 484 (19/26) | 11.81 / 44.27 / -3.278 / 484 (15/21) | 4.31 / 13.28 / -2.257 / 198 (27/30) | 5.02 / 15.80 / -2.171 / 198 (22/26) | 103/129 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.72 / 6.84 / 0.991 / 242 (18/20) | 1.40 / 7.79 / 0.990 / 253 (17/20) | 1.45 / 8.02 / 0.991 / 250 (18/20) | 1.32 / 7.42 / 0.992 / 220 (18/20) | 1.45 / 6.99 / 0.992 / 209 (18/20) | 89/100 |
| medium | 1.43 / 6.97 / 0.991 / 184 (19/20) | 1.41 / 7.60 / 0.991 / 251 (17/20) | 1.42 / 8.21 / 0.991 / 250 (18/20) | 1.31 / 7.87 / 0.992 / 220 (18/20) | 1.28 / 7.20 / 0.992 / 210 (19/20) | 91/100 |
| large | 1.45 / 6.81 / 0.991 / 185 (17/20) | 1.37 / 8.27 / 0.991 / 250 (18/20) | 1.37 / 8.27 / 0.991 / 250 (18/20) | 1.35 / 7.77 / 0.992 / 221 (16/20) | 1.34 / 7.28 / 0.992 / 210 (17/20) | 86/100 |
| xl | 1.39 / 7.24 / 0.991 / 184 (19/20) | 1.37 / 8.57 / 0.991 / 250 (19/20) | 1.37 / 8.56 / 0.991 / 250 (19/20) | 1.34 / 7.90 / 0.992 / 220 (17/20) | 1.33 / 7.39 / 0.992 / 210 (18/20) | 92/100 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✗ mag 14.51 % / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| medium | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| large | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| xl | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, valid refined reference and converged NN OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |
| medium | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |
| large | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |
| xl | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |

The expanded audit matrix changes the denominators: parametric DC is
**103 → 104 → 101 → 103/129**, while transient is
**89 → 91 → 86 → 92/100**. `medium` is the best device-DC tier and `large`
the worst; transient ranks them differently. Device AC is 0/10 at every tier
because none of its DC operating points satisfies the current convergence
prerequisite. The previously reported 9/10 · 10/10 · 10/10 · 10/10
response-shape counts were measured at nonconverged states and are retracted.

## 6. Controlled-repeat reproducibility

The TSMC6 controlled repeat retrains one recipe on **bit-identical rows** and
compares strict verdicts, which measures the whole pipeline's run-to-run
variance with the data held exactly fixed (`methodology.md` §7).

Under the current contract, BSIM-AR reproduces fourteen of sixteen compared
verdicts. The splits are `small` ring (4.33 % vs 5.86 %) and `xl` SRAM lobe
positivity. DirectNet is 15/16, with its sole split at `large` SRAM.

The old claim that this verdict count makes BSIM-AR the more reproducible
family is therefore retracted. The current convergence contract collapses
every Miller opamp to the same failure, so 14/16 versus 15/16 is driven by one
ring threshold and SRAM lobe positivity, not comparative fidelity on viable
opamp fixed points.

## 7. The AR prefix cache

`PYCIRCUITSIM_NN_AR_CACHE=1` keeps per-layer K/V across the autoregressive loop
so each token is encoded once, instead of re-encoding the whole growing prefix
at every step. The V7.5.17 rerun passed all 10 checks and measured
118.5 ms → 74.2 ms (**1.60×**) for the cached evaluation.

**It is default-off and stays off until a full re-gate clears it.** It is exact
in real arithmetic but not in float32, because `F.linear` is not row-stable on
CPU — so *no* incremental formulation can be bit-identical, and the deviation
(≤1.6 µV on solved nodes) has to be shown harmless rather than argued away.
`tests/perf/verify_ar_cache.py` guards the lever with 10 checks, since no accuracy
gate can reach the path while it is off.

## 8. What is open

| open | reading |
|---|---|
| **Capacity tradeoff** | 9 → 9 → 12 → 11/20. `large` is best; `xl` loses one SRAM lobe-positivity gate. |
| **Low-VDD rings** | `medium` through `xl` fail TSMC5/6/7; `small` passes TSMC5/6 but narrowly fails TSMC7. The historical corridor curriculum closes this class; recipes were not retrained in V7.4.0. |
| **Inference cost** | ~40× DirectNet per evaluation. Structural, not a tuning matter: it is the token count times the weight stream. |
| **Audited device matrix** | No tier closes all cells: DC spans 101–104/129 and transient 86–92/100. |
| **Miller opamp DC/AC** | DC is 0/5 and AC is 0/5 at every tier. Fix the physical operating-point convergence/basin before interpreting small-signal metrics. |

## 9. Reproduction

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}`,
each with `_norm.npz` and a **required** `_config.npz` sidecar.
Raw runs: `results/v7517_clean/`. The complete training, gate, coverage, and
report-build commands are in the
[README](../../README.md#run-the-complete-clean-checkpoint-matrix).
