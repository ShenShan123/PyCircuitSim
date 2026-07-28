# BSIM-AR (LEVEL=74) — the clean recipe

**What it is.** An autoregressive Transformer that emits the thirteen outputs
one token at a time, each conditioned on the ones already produced. It shares
DirectNet's data, normalization, loss, training and evaluation pipeline
entirely — the `bsimar` package — so the only variable against
[`DirectNet-L73-clean.md`](DirectNet-L73-clean.md) is the architecture.

**What it costs.** The autoregressive loop re-enters the network once per
output token, and inference is memory-bandwidth-bound on the weights, so cost
tracks *parameters × times the checkpoint is streamed*. That is why BSIM-AR
runs ~40× DirectNet rather than ~5×, and why DirectNet stays production
regardless of which family scores higher.

**What "clean" means.** One training run, no addendum:
`--apply-filter off --swa-mode ema --seed 42`. Unlike DirectNet, BSIM-AR's
production slots `tsmc{X}_tf_{size}_*` **are** the clean checkpoints — the
curriculum variants live under their own stems and are covered in
[`BSIM-AR-L74-recipes.md`](BSIM-AR-L74-recipes.md).

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

---

## 1. Headline — complex gates by tier

Strict: a cell passes only if it passes at OMP ∈ {1, 2, 4}.

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **17/20** | 2/5 | 5/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc |
| medium | **17/20** | 2/5 | 5/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc |
| large | **17/20** | 2/5 | 5/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc |
| xl | **17/20** | 2/5 | 5/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc |

## 2. By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 6.53% | FAIL 6.14% | FAIL 5.97% | **PASS** 1.92% | **PASS** 2.06% |
| medium | FAIL 5.55% | FAIL 6.66% | FAIL 7.41% | **PASS** 1.52% | **PASS** 1.59% |
| large | FAIL 7.38% | FAIL 11.61% | FAIL 8.63% | **PASS** 1.54% | **PASS** 1.92% |
| xl | FAIL 7.61% | FAIL 11.99% | FAIL 12.55% | **PASS** 1.98% | **PASS** 2.19% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 3.07% | **PASS** 4.49% | **PASS** 0.55% | **PASS** 8.53% | **PASS** 6.11% |
| medium | **PASS** 1.48% | **PASS** 5.29% | **PASS** 4.12% | **PASS** 4.78% | **PASS** 6.79% |
| large | **PASS** 3.00% | **PASS** 4.72% | **PASS** 5.39% | **PASS** 5.81% | **PASS** 5.74% |
| xl | **PASS** 2.73% | **PASS** 4.21% | **PASS** 4.26% | **PASS** 5.81% | **PASS** 5.87% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.79% | **PASS** 1.40% | **PASS** 1.75% | **PASS** 1.85% | **PASS** 5.78% |
| medium | **PASS** 6.14% | **PASS** 1.70% | **PASS** 1.47% | **PASS** 2.64% | **PASS** 1.45% |
| large | **PASS** 6.45% | **PASS** 2.32% | **PASS** 1.59% | **PASS** 1.36% | **PASS** 1.95% |
| xl | **PASS** 6.08% | **PASS** 1.95% | **PASS** 1.95% | **PASS** 1.69% | **PASS** 1.93% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 2.05% | **PASS** 2.72% | **PASS** 2.46% | **PASS** 4.50% | **PASS** 3.68% |
| medium | **PASS** 2.11% | **PASS** 2.60% | **PASS** 2.62% | **PASS** 4.22% | **PASS** 3.43% |
| large | **PASS** 2.46% | **PASS** 2.69% | **PASS** 2.64% | **PASS** 4.15% | **PASS** 3.33% |
| xl | **PASS** 2.57% | **PASS** 2.74% | **PASS** 2.73% | **PASS** 4.19% | **PASS** 3.36% |

## 3. By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 0/4 | 4/4 | 4/4 | 4/4 | **12/16** |
| **TSMC6** | 0/4 | 4/4 | 4/4 | 4/4 | **12/16** |
| **TSMC7** | 0/4 | 4/4 | 4/4 | 4/4 | **12/16** |
| **TSMC12** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |
| **TSMC16** | 4/4 | 4/4 | 4/4 | 4/4 | **16/16** |

BSIM-AR's failures are **entirely** the low-VDD rings, exactly as DirectNet's
concentrate there, and for the same reason: the corridor curriculum is the only
lever that has ever moved them, and clean has no corridor. TSMC6 fails with
TSMC7, as a duplicate must.

Where it differs sharply from DirectNet is the **opamp column**, which it holds
completely — every tech at every tier — rather than sporadically. On the other
two families opamps are the hard class; here they are free, and rings are the
only thing standing between clean BSIM-AR and a sweep.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 3/4 | 3/4 | 3/4 | 4/4 | 4/4 | **17/20** |
| medium | 3/4 | 3/4 | 3/4 | 4/4 | 4/4 | **17/20** |
| large | 3/4 | 3/4 | 3/4 | 4/4 | 4/4 | **17/20** |
| xl | 3/4 | 3/4 | 3/4 | 4/4 | 4/4 | **17/20** |

**BSIM-AR is flat, and V7.3.0 makes that a much stronger statement than it
was.** Until this campaign only `large` had been strict-swept; the other three
tiers rested on single runs. All four are now measured at OMP ∈ {1, 2, 4}, and
they are **identical** — the same score, the same open cells, zero flips, from
0.67 M parameters to 14.81 M. A 22× capacity range that changes nothing is not
a curve with a gentle slope; it is the absence of a capacity effect.

What is left open is the low-VDD ring column, and nothing else: the opamp,
SRAM and switchcap columns are **complete at every tier**. Holding the entire
opamp column at all four capacities is the sharpest contrast with the other two
families, where opamps are the hard class.

The pre-fix reading — "capacity peaks at medium" — was largely a `gds`
artifact; what the fix bought was that opamp column, not a change in shape.
The practical consequence is that the smallest tier clearing the bar is the
right one: `xl` costs 7.7× `medium`'s parameters and roughly eleven days of
training to tie it exactly.

## 5. Device-level suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.54 | 2.38 | 1.24 | 0.82 | 1.61 (13/14) | 68/69 |
| medium | 1.77 | 2.24 (13/14) | 1.46 | 1.34 (17/18) | 1.57 (13/14) | 66/69 |
| large | 1.80 | 2.93 (13/14) | 1.21 | 1.67 (16/18) | 1.58 (13/14) | 65/69 |
| xl | 1.94 | 3.16 | 2.92 | 1.08 | 1.07 | 69/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.54 | 1.48 | 1.47 | 1.53 | 1.60 | 80/80 |
| medium | 1.80 | 1.50 | 1.52 | 1.52 | 1.50 | 80/80 |
| large | 1.66 | 1.50 | 1.48 | 1.51 | 1.49 | 80/80 |
| xl | 1.62 | 1.48 | 1.48 | 1.51 | — | 58/58 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 0.39 dB | FAIL 9.82 dB | FAIL 11.33 dB | FAIL 10.27 dB | **PASS** 0.54 dB | **1/5** |
| medium | FAIL 4.63 dB | **PASS** 1.23 dB | FAIL 11.13 dB | FAIL 0.17 dB | **PASS** 2.62 dB | **2/5** |
| large | FAIL 2.54 dB | FAIL 4.38 dB | **PASS** 0.12 dB | FAIL 4.34 dB | **PASS** 0.33 dB | **2/5** |
| xl | FAIL 24.95 dB | FAIL 4.60 dB | FAIL 3.86 dB | FAIL 5.18 dB | **PASS** 0.97 dB | **1/5** |

One counter-intuitive row deserves attention: **BSIM-AR's TSMC7-NMOS parametric
DC error *grows* with capacity**. It is the worst device-DC surface in the
project, on the tech that also carries the hardest circuit cell, and it moves
the wrong way as the model gets bigger — the clearest single instance of extra
capacity buying a tighter fit to the training distribution and worse
generalization.

## 6. Reproducibility — the reason to prefer this family for fidelity work

The TSMC6 controlled repeat retrains one recipe on **bit-identical rows** and
compares strict verdicts, which measures the whole pipeline's run-to-run
variance with the data held exactly fixed (`methodology.md` §7).

**BSIM-AR is the stable family.** Across the repeat, all sixteen of its
compared verdicts reproduced, its rings never crossed the gate in either
direction, and its opamps never railed in either run. DirectNet reproduced
eleven of sixteen and PFN ten of twelve, both with bimodal opamps.

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
`tests/verify_ar_cache.py` guards the lever with 10 checks, since no accuracy
gate can reach the path while it is off.

## 8. What is open

| open | reading |
|---|---|
| **Low-VDD rings under every clean tier** | Deterministic. The corridor curriculum closes them — see the recipes report. |
| **Inference cost** | ~40× DirectNet per evaluation. Structural, not a tuning matter: it is the token count times the weight stream. |
| **TSMC7-NMOS device DC** | Grows with capacity. Unexplained. |
| **Opamp open-loop AC** | BSIM-AR passes this gate more often than either other family, but part of the denominator is unreachable by construction — see the caveat in `DirectNet-L73-clean.md` §6. |

## 9. Reproducing

```bash
# BSIM-AR runs through the same drivers; the tag selects LEVEL=74
bash scripts/v710_regate.sh _one tf medium TSMC7 verify_complex_opamp 4
python scripts/v730_coverage.py --set clean --tag tf
```

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}`,
each with `_norm.npz` and a **required** `_config.npz` sidecar.
Raw runs: `results/v730_regate/`, `results/v710_regate/`, `results/a3_regate/`.
