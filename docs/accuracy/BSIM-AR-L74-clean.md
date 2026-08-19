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

**Every number below is from the V7.4.0 rebuild.** Datasets, checkpoints and
gate verdicts were all regenerated from scratch on new hardware, on exactly the
same datasets and the same clean recipe DirectNet used, so the two clean
reports remain a like-for-like architecture comparison. Nothing is inherited
from an earlier pass.

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
| small | **18/20** | 4/5 | 5/5 | 5/5 | 4/5 | 0 | tsmc7-ring_osc, tsmc5-switchcap |
| medium | **17/20** | 2/5 | 5/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc |
| large | **15/20** | 2/5 | 3/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc6-opamp, tsmc7-opamp |
| xl | **13/20** | 1/5 | 4/5 | 4/5 | 4/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc12-ring_osc, tsmc12-opamp, tsmc12-sram_snm, tsmc12-switchcap |

## 2. By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 3.56% | **PASS** 4.33% | FAIL 5.86% | **PASS** 2.36% | **PASS** 1.58% |
| medium | FAIL 5.42% | FAIL 6.32% | FAIL 6.63% | **PASS** 1.61% | **PASS** 1.51% |
| large | FAIL 6.95% | FAIL 7.20% | FAIL 7.20% | **PASS** 1.78% | **PASS** 1.63% |
| xl | FAIL 7.77% | FAIL 12.10% | FAIL 10.15% | FAIL 1.99% | **PASS** 2.00% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 4.24% | **PASS** 4.56% | **PASS** 1.45% | **PASS** 4.92% | **PASS** 8.37% |
| medium | **PASS** 1.76% | **PASS** 4.51% | **PASS** 4.51% | **PASS** 5.75% | **PASS** 5.62% |
| large | **PASS** 2.79% | FAIL 100.00% | FAIL 100.00% | **PASS** 5.82% | **PASS** 6.39% |
| xl | **PASS** 4.41% | **PASS** 4.45% | **PASS** 4.63% | FAIL 5.26% | **PASS** 5.81% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.27% | **PASS** 1.65% | **PASS** 1.69% | **PASS** 1.46% | **PASS** 1.45% |
| medium | **PASS** 6.68% | **PASS** 1.72% | **PASS** 1.76% | **PASS** 1.38% | **PASS** 2.69% |
| large | **PASS** 5.83% | **PASS** 1.61% | **PASS** 1.61% | **PASS** 2.07% | **PASS** 1.49% |
| xl | **PASS** 6.33% | **PASS** 1.74% | **PASS** 1.66% | FAIL 2.44%† | **PASS** 2.02% |

† failed on **lobe positivity**, the half of this gate the headline number does not show — the metric above is inside its threshold.

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 0.84%† | **PASS** 2.39% | **PASS** 2.62% | **PASS** 4.15% | **PASS** 3.28% |
| medium | **PASS** 2.20% | **PASS** 2.57% | **PASS** 2.52% | **PASS** 4.11% | **PASS** 3.36% |
| large | **PASS** 2.37% | **PASS** 2.58% | **PASS** 2.58% | **PASS** 4.28% | **PASS** 3.48% |
| xl | **PASS** 2.50% | **PASS** 2.65% | **PASS** 2.61% | FAIL 4.25%† | **PASS** 3.39% |

† failed on **hold droop**, the half of this gate the headline number does not show — the metric above is inside its threshold.

## 3. By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 1/4 | 4/4 | 4/4 | 3/4 | **12/16** |
| **TSMC6** | 1/4 | 3/4 | 4/4 | 4/4 | **12/16** |
| **TSMC7** | 0/4 | 3/4 | 4/4 | 4/4 | **11/16** |
| **TSMC12** | 3/4 | 3/4 | 3/4 | 3/4 | **12/16** |
| **TSMC16** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |

The V7.4.0 rebuild **retracts the clean separation that V7.3.0 showed**. Rings
remain the dominant weakness, but they are no longer the whole failure set:
`small` loses TSMC5 switchcap on hold droop, `large` rails the duplicate
TSMC6/TSMC7 opamps, and `xl` loses all four TSMC12 cells. The TSMC12 collapse
is especially diagnostic because the other four technologies retain three or
four cells at the same tier; it is a checkpoint-specific failure, not a law of
the architecture.

The controlled repeat is still doing its job. TSMC6 and TSMC7 reproduce
**15/16** verdicts across the four tiers; the sole disagreement is the noisy
`small` ring cell (4.33 % PASS vs 5.86 % FAIL). Their large-tier opamps rail
together, and every SRAM and switchcap verdict agrees.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 3/4 | 4/4 | 3/4 | 4/4 | 4/4 | **18/20** |
| medium | 3/4 | 3/4 | 3/4 | 4/4 | 4/4 | **17/20** |
| large | 3/4 | 2/4 | 2/4 | 4/4 | 4/4 | **15/20** |
| xl | 3/4 | 3/4 | 3/4 | 0/4 | 4/4 | **13/20** |

> **RETRACTED in V7.4.0: "BSIM-AR is flat across capacity."** On the fresh
> checkpoints the strict curve is **18 → 17 → 15 → 13/20**. Capacity now
> hurts monotonically, the opposite of DirectNet's 11 → 12 → 14 → 15/20
> climb under the identical data and recipe. The architectural comparison is
> therefore sharper than a score: additional capacity helps the feed-forward
> MLP and hurts the autoregressive model on this training run.

`small` is the clean selection: it has the best circuit score, 67/69
parametric-DC configs, 80/80 transient configs and 9/10 device-AC cells at
0.67 M parameters. `medium` repairs the last device-AC cell and reaches 68/69
DC, but loses one complex cell. `xl` costs 7.6× medium's parameters and is the
worst circuit tier, with the TSMC12 four-cell collapse.

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
| small | 1.96 | 1.46 | 1.50 | 1.51 | 1.63 | 80/80 |
| medium | 1.68 | 1.51 | 1.55 | 1.48 | 1.47 | 80/80 |
| large | 1.72 | 1.48 | 1.48 | 1.50 | 1.49 | 80/80 |
| xl | 1.66 | 1.47 | 1.46 | 1.50 | 1.50 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✗ mag 14.51 % / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **9/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 0.57 dB | **PASS** 0.82 dB | FAIL 10.88 dB | FAIL 4.61 dB | FAIL 34.15 dB | **1/5** |
| medium | FAIL 2.70 dB | FAIL 3.65 dB | FAIL 4.87 dB | FAIL 4.98 dB | FAIL 9.02 dB | **0/5** |
| large | FAIL 2.40 dB | FAIL 31.25 dB | FAIL 31.25 dB | **PASS** 1.24 dB | FAIL 3.79 dB | **1/5** |
| xl | FAIL 0.34 dB | FAIL 6.69 dB | FAIL 3.79 dB | FAIL 1.67 dB | **PASS** 2.89 dB | **1/5** |

The device results do not rescue the larger tiers. Parametric DC is
67/69 · 68/69 · 65/69 · 67/69 from small→xl, while transient is 80/80 at
every tier. `medium` is the best device fit and `large` the worst; the circuit
curve keeps falling through `xl`. Device AC is the stable axis: 9/10 at
`small`, then 10/10 at every larger tier.

## 6. Reproducibility — the reason to prefer this family for fidelity work

The TSMC6 controlled repeat retrains one recipe on **bit-identical rows** and
compares strict verdicts, which measures the whole pipeline's run-to-run
variance with the data held exactly fixed (`methodology.md` §7).

**BSIM-AR remains the most stable family, but no longer a perfect one.** Across
the V7.4.0 repeat, fifteen of sixteen compared verdicts reproduce. The single
split is `small` ring (4.33 % vs 5.86 %), exactly the threshold-sensitive class
the methodology's noise floor predicts. DirectNet reproduces fourteen of
sixteen in the same rebuild, with both disagreements in the bimodal opamp
column; PFN's latest controlled comparison remains ten of twelve from V7.3.0.

That is a practical argument for BSIM-AR as the fidelity option **on top of**
its gate score: a number measured on a BSIM-AR checkpoint is more likely to
survive a retrain than the same number measured on the other two families.

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
| **Capacity decline** | 18 → 17 → 15 → 13/20. The old flat-capacity claim is retracted. |
| **Low-VDD rings** | Still the dominant failure class. The historical corridor curriculum closes them; recipes were not retrained in V7.4.0. |
| **TSMC12 `xl` collapse** | All four complex cells fail while the other techs retain 3–4/4. Checkpoint-specific and unexplained. |
| **Inference cost** | ~40× DirectNet per evaluation. Structural, not a tuning matter: it is the token count times the weight stream. |
| **TSMC7-NMOS device DC** | Grows with capacity. Unexplained. |
| **Opamp open-loop AC** | BSIM-AR passes this gate more often than either other family, but part of the denominator is unreachable by construction — see the caveat in `DirectNet-L73-clean.md` §6. |

## 9. Reproducing

```bash
# train the clean control, all tiers, on GPU (same driver as DirectNet;
# MODEL=transformer selects LEVEL=74 and the `_tf_` stems)
MODEL=transformer RECIPES=clean TECHS='tsmc5 tsmc6 tsmc7 tsmc12 tsmc16' \
  SIZES='small medium large xl' GPUS='1 1 0' NSTREAMS=8 TRAIN_OMP=4 \
  bash scripts/recipe_train.sh

# BSIM-AR gates through the same drivers; the tag selects LEVEL=74
NN_PY=$(command -v python) bash scripts/v710_regate.sh \
  _one tf medium TSMC7 verify_complex_opamp 4
python scripts/v730_coverage.py --set clean --tag tf --require-complete
```

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}`,
each with `_norm.npz` and a **required** `_config.npz` sidecar.
Raw runs: `results/v740_regate/`.
