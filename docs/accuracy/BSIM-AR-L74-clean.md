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
DirectNet. Every table below was remeasured in one complete CPU-pinned pass on
2026-08-19 at commit `24c181a`, after the solver began requiring convergence
at the final physical homotopy step and before AC linearization. The shared
checkpoint archive and exact old-versus-new comparison are recorded in
[`simple-circuits-recheck-2026-08-19.md`](simple-circuits-recheck-2026-08-19.md).

Gate definitions, the strict-OMP rule and the code ladder:
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
>
> V7.4.0 makes `methodology.md` §7's TSMC6 finding exhaustive rather than
> exemplary: of 1748 shared modelcard parameters exactly 11 differ in value,
> plus 74 TSMC6-only and 12 TSMC7-only keys, and **all 97 have zero occurrences
> anywhere in the BSIM-CMG Verilog-A** (against 333 of the 1737 identical keys
> that do appear). All 97 belong to TSMC's TMI layout-dependent-effect layer
> (LOD, ODX, isolated-CPODE); no core device-physics parameter differs. TSMC6
> stays a **controlled repeat** whose sole degree of freedom is training
> nondeterminism — see `DirectNet-L73-clean.md` for the full breakdown.

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
| small | **PASS** 6.27% | **PASS** 1.65% | **PASS** 1.69% | **PASS** 1.45% | **PASS** 1.44% |
| medium | **PASS** 6.67% | **PASS** 1.71% | **PASS** 1.75% | **PASS** 1.38% | **PASS** 2.69% |
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

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.98 | 2.33 (13/14) | 1.46 | 0.94 | 1.58 (13/14) | 67/69 |
| medium | 1.78 | 2.09 | 1.22 | 1.46 | 1.80 (13/14) | 68/69 |
| large | 2.12 | 2.51 (13/14) | 1.27 | 1.99 (16/18) | 1.97 (13/14) | 65/69 |
| xl | 1.91 | 3.22 (13/14) | 2.41 | 0.98 | 1.45 (13/14) | 67/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.96 | 1.46 | 1.50 | 1.51 | 1.62 | 80/80 |
| medium | 1.68 | 1.51 | 1.55 | 1.48 | 1.47 | 80/80 |
| large | 1.72 | 1.47 | 1.47 | 1.49 | 1.49 | 80/80 |
| xl | 1.66 | 1.47 | 1.46 | 1.50 | 1.50 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✗ mag 14.51 % / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| medium | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| large | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| xl | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 15.52 dB | FAIL 30.40 dB | FAIL 30.18 dB | FAIL 21.58 dB | FAIL 30.94 dB | **0/5** |
| medium | FAIL 12.42 dB | FAIL 29.81 dB | FAIL 29.24 dB | FAIL 30.26 dB | FAIL 32.14 dB | **0/5** |
| large | FAIL 17.05 dB | FAIL 30.98 dB | FAIL 30.98 dB | FAIL 27.81 dB | FAIL 33.11 dB | **0/5** |
| xl | FAIL 13.96 dB | FAIL 30.72 dB | FAIL 30.72 dB | FAIL 28.00 dB | FAIL 33.55 dB | **0/5** |

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
at every step. It is worth 1.2–1.6× in the solver and 4.3× on `no_grad` batch
evaluation.

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
| **Miller opamp DC/AC** | DC is 0/5 and AC is 0/5 at every tier. Fix the physical operating-point convergence/basin before interpreting small-signal metrics; the AC gate also has the bias-grid caveat in `DirectNet-L73-clean.md` §6. |

## 9. Reproducing

```bash
# train the clean control, all tiers, on GPU (same driver as DirectNet;
# MODEL=transformer selects LEVEL=74 and the `_tf_` stems)
MODEL=transformer RECIPES=clean TECHS='tsmc5 tsmc6 tsmc7 tsmc12 tsmc16' \
  SIZES='small medium large xl' GPUS='1 1 0' NSTREAMS=8 TRAIN_OMP=4 \
  bash scripts/recipe_train.sh

# BSIM-AR gates through the same drivers; the tag selects LEVEL=74
BSIMAR_CHECKPOINT_DIR=external_compact_models/neural_network/v740_archive/checkpoints \
NN_PY=$(command -v python) bash scripts/v710_regate.sh \
  _one tf medium TSMC7 verify_complex_opamp 4
BSIMAR_CHECKPOINT_DIR=external_compact_models/neural_network/v740_archive/checkpoints \
python scripts/v730_coverage.py --set clean --tag tf \
  --passes simple-recheck --require-complete
```

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}`,
each with `_norm.npz` and a **required** `_config.npz` sidecar.
Raw runs: `results/simple_recheck_24c181a/`.
