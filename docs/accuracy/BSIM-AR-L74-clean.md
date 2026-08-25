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
DirectNet. Every generated table below was remeasured in the complete V7.5.16
CPU-pinned pass at gate commit `49f0426`, after the MNA residual and opamp-AC
bias contracts were corrected. The shared 280-artifact checkpoint archive has
manifest SHA-256
`8e4245f1ab563cd116a789cb02388e0f7b736186141694d3242ede2a7ed07868`.
The V7.5.15 audit trail remains in
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
| small | **13/20** | 4/5 | 0/5 | 5/5 | 4/5 | 0 | tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-switchcap |
| medium | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |
| large | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |
| xl | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |

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
| small | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| medium | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| large | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| xl | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.26% | **PASS** 1.67% | **PASS** 1.69% | **PASS** 1.46% | **PASS** 1.60% |
| medium | **PASS** 6.68% | **PASS** 1.74% | **PASS** 1.77% | **PASS** 1.38% | **PASS** 2.69% |
| large | **PASS** 5.83% | **PASS** 1.61% | **PASS** 1.61% | **PASS** 2.07% | **PASS** 1.49% |
| xl | **PASS** 6.33% | **PASS** 1.74% | **PASS** 1.66% | **PASS** 2.44% | **PASS** 2.03% |

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
| **TSMC5** | 1/4 | 0/4 | 4/4 | 3/4 | **8/16** |
| **TSMC6** | 1/4 | 0/4 | 4/4 | 4/4 | **9/16** |
| **TSMC7** | 0/4 | 0/4 | 4/4 | 4/4 | **8/16** |
| **TSMC12** | 4/4 | 0/4 | 4/4 | 4/4 | **12/16** |
| **TSMC16** | 4/4 | 0/4 | 4/4 | 4/4 | **12/16** |

Rings and Miller opamp fixed points are the two weaknesses under the current
solver contract. `small` keeps four of five rings but loses TSMC5 switchcap on
hold droop; every larger tier keeps two rings and all five switchcaps. Every
tier is 0/5 on opamp because the selected NN fixed points do not converge at
the final physical homotopy step.

The old V7.4 report's **TSMC12 `xl` collapse is retracted**. All twelve raw
runs in that group carried two completion markers and were correctly collected
as `RACED`, but the report builder counted them as scientific failures. The
fresh isolated pass restores the expected TSMC12 SRAM, ring and switchcap
passes; only its opamp remains a real failure.

The controlled repeat is still doing its job. TSMC6 and TSMC7 reproduce
**15/16** verdicts across the four tiers; the sole disagreement is the noisy
`small` ring cell (4.33 % PASS vs 5.86 % FAIL). Every opamp now fails the
honest convergence contract, and every SRAM and switchcap verdict agrees.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 2/4 | 3/4 | 2/4 | 3/4 | 3/4 | **13/20** |
| medium | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |
| large | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |
| xl | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |

The current strict curve is **13 → 12 → 12 → 12/20**. The old
18 → 17 → 15 → 13 curve combined nonconverged opamp fixed points with
race-corrupted TSMC12-`xl` cells and is retracted. Capacity does not recover
the low-VDD rings or any Miller opamp; beyond `medium`, the circuit score is
flat.

`small` is the clean selection: it has the best circuit score, 67/69
parametric-DC configs and 80/80 transient configs at 0.67 M parameters.
`medium` reaches the best device-DC result, 68/69, but loses one ring.
`large` and `xl` add parameters without improving the 12/20 complex score.

## 5. Device-level suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.98 / 7.87 / 0.961 / 37 | 2.33 / 8.60 / 0.885 / 29.8 (13/14) | 1.46 / 4.93 / 0.989 / 28 | 0.94 / 3.64 / 0.974 / 49.3 | 1.58 / 6.07 / 0.860 / 106 (13/14) | 67/69 |
| medium | 1.78 / 5.81 / 0.955 / 23.4 | 2.09 / 7.28 / 0.914 / 24.4 | 1.22 / 3.31 / 0.988 / 23.3 | 1.46 / 4.03 / 0.891 / 116 | 1.80 / 6.22 / 0.774 / 151 (13/14) | 68/69 |
| large | 2.12 / 6.58 / 0.959 / 30.1 | 2.51 / 7.77 / 0.855 / 32.5 (13/14) | 1.27 / 3.40 / 0.984 / 20.7 | 1.99 / 5.85 / 0.804 / 171 (16/18) | 1.97 / 5.53 / 0.809 / 151 (13/14) | 65/69 |
| xl | 1.91 / 6.00 / 0.954 / 33.2 | 3.22 / 9.57 / 0.808 / 38 (13/14) | 2.41 / 6.61 / 0.950 / 64.8 | 0.98 / 2.98 / 0.906 / 114 | 1.45 / 4.31 / 0.790 / 158 (13/14) | 67/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.98 / 8.24 / 0.991 / 242 | 1.48 / 8.02 / 0.990 / 253 | 1.51 / 8.31 / 0.991 / 250 | 1.51 / 8.44 / 0.992 / 220 | 1.63 / 8.22 / 0.992 / 209 | 80/80 |
| medium | 1.68 / 8.57 / 0.990 / 184 | 1.51 / 8.06 / 0.991 / 251 | 1.56 / 8.19 / 0.991 / 250 | 1.48 / 8.75 / 0.992 / 220 | 1.46 / 8.28 / 0.992 / 210 | 80/80 |
| large | 1.74 / 8.76 / 0.990 / 185 | 1.48 / 8.35 / 0.991 / 250 | 1.48 / 8.35 / 0.991 / 250 | 1.50 / 9.03 / 0.992 / 221 | 1.49 / 8.67 / 0.992 / 210 | 80/80 |
| xl | 1.66 / 8.85 / 0.990 / 184 | 1.47 / 8.31 / 0.991 / 250 | 1.46 / 8.30 / 0.991 / 250 | 1.50 / 8.93 / 0.992 / 220 | 1.50 / 8.66 / 0.992 / 210 | 80/80 |

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
| small | FAIL 54.03 dB | FAIL 49.84 dB | FAIL 49.59 dB | FAIL 31.75 dB | FAIL 36.13 dB | **0/5** |
| medium | FAIL 55.07 dB | FAIL 51.07 dB | FAIL 51.73 dB | FAIL 35.19 dB | FAIL 39.38 dB | **0/5** |
| large | FAIL 52.82 dB | FAIL 50.69 dB | FAIL 50.69 dB | FAIL 34.64 dB | FAIL 33.02 dB | **0/5** |
| xl | FAIL 54.81 dB | FAIL 49.92 dB | FAIL 50.13 dB | FAIL 34.31 dB | FAIL 32.48 dB | **0/5** |

Parametric DC is 67/69 · 68/69 · 65/69 · 67/69 from small→xl, while
transient is 80/80 at every tier. `medium` is the best device fit and `large`
the worst. Device AC is 0/10 at every tier because none of its DC operating
points satisfies the current convergence prerequisite. The previously
reported 9/10 · 10/10 · 10/10 · 10/10 response-shape counts were measured
at nonconverged states and are retracted.

## 6. Controlled-repeat reproducibility

The TSMC6 controlled repeat retrains one recipe on **bit-identical rows** and
compares strict verdicts, which measures the whole pipeline's run-to-run
variance with the data held exactly fixed (`methodology.md` §7).

Under the current contract, BSIM-AR reproduces fifteen of sixteen compared
verdicts. The single split is `small` ring (4.33 % vs 5.86 %), exactly the
threshold-sensitive class the methodology's noise floor predicts. DirectNet
is 16/16 after its old nonconverged opamp passes are removed; PFN's latest
controlled comparison remains ten of twelve from V7.3.0.

The old claim that this verdict count makes BSIM-AR the more reproducible
family is therefore retracted. The current convergence contract collapses
every Miller opamp to the same failure, so 15/16 versus 16/16 measures one
threshold-sensitive ring split, not comparative fidelity on viable opamp
fixed points.

## 7. The AR prefix cache

`PYCIRCUITSIM_NN_AR_CACHE=1` keeps per-layer K/V across the autoregressive loop
so each token is encoded once, instead of re-encoding the whole growing prefix
at every step. The V7.5.16 rerun passed all 10 checks and measured
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
| **Capacity plateau** | 13 → 12 → 12 → 12/20. Larger tiers do not recover a ring or opamp fixed point. |
| **Low-VDD rings** | The repeatable failure class after the universal opamp-convergence failures. The historical corridor curriculum closes them; recipes were not retrained in V7.4.0. |
| **Inference cost** | ~40× DirectNet per evaluation. Structural, not a tuning matter: it is the token count times the weight stream. |
| **TSMC7-NMOS device DC** | Grows with capacity. Unexplained. |
| **Miller opamp DC/AC** | DC is 0/5 and AC is 0/5 at every tier. Fix the physical operating-point convergence/basin before interpreting small-signal metrics. |

## 9. Reproduction

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}`,
each with `_norm.npz` and a **required** `_config.npz` sidecar.
Raw runs: `results/v7516_clean/`. The complete training, gate, coverage, and
report-build commands are in the
[README](../../README.md#run-the-complete-clean-checkpoint-matrix).
