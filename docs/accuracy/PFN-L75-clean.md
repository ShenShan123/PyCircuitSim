# PFN (LEVEL=75) — the clean recipe

**What it is.** A faithful scaled-down port of TabPFN-v3's in-context
transformer, used as a compact model: the tech code enters as an eighth column
token and predictions are conditioned on a table of context rows rather than on
weights alone. It shares the `bsimar` data, normalization, loss, training and
evaluation pipeline with the other two families.

**Two deliberate deviations from TabPFN**, both forced by the solver:

1. **A frozen learned context.** A stratified K-row buffer is baked into the
   checkpoint and its attention keys/values are cached at inference. A compact
   model is called millions of times inside Newton-Raphson; re-attending to a
   live context each call is not affordable.
2. **A direct 13-output value head** instead of TabPFN's distributional head.
   Newton needs a smooth surface it can differentiate, and a
   bucketed/sampled output is neither smooth nor cheap to differentiate.

The `_config.npz` sidecar is **required** to rebuild the architecture — a
checkpoint without it cannot be loaded.

**Status: research.** PFN is not production and is not the fidelity option; it
is here because it is a genuinely different inductive bias on the same problem,
and because it is ~4× cheaper than BSIM-AR at ~15.6 ms/eval.

Gate definitions and the code ladder: [`methodology.md`](methodology.md).

> **Evidence boundary.** PFN was not rebuilt in V7.4.0; no PFN checkpoints are
> present in the new-hardware artifact set. All measurements in this file are
> the latest available PFN evidence, retained from V7.3.0.

| tier | shape | params |
|---|---|---|
| `small` | — | **0.69 M** |
| `medium` | — | 2.03 M |
| `large` | — | 4.65 M |
| `xl` | 192 × (4+4+9) blocks | 14.86 M |

The `xl` tier is new in V7.1.0 — the family had three scales until then. Its
14.86 M mirrors BSIM-AR `xl`'s 14.81 M and its in-context width equals that
model's `d_model`, so the top of the capacity axis is comparable across
families.

> **Denominators changed in V7.3.0.** TSMC6 folds into the headline: complex
> totals are **/20**, device AC **/10**, opamp AC **/5**. No total here is
> comparable to a pre-V7.3.0 total without rescaling.

---

## 1. Headline — complex gates by tier

Strict: a cell passes only if it passes at OMP ∈ {1, 2, 4}.

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **14/20** | 3/5 | 2/5 | 5/5 | 4/5 | 0 | tsmc5-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-switchcap |
| medium | **12/20** | 1/5 | 2/5 | 5/5 | 4/5 | 1 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc16-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc16-opamp, tsmc12-switchcap |
| large | **10/20** | 1/5 | 1/5 | 5/5 | 3/5 | 1 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc16-ring_osc, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc12-switchcap, tsmc16-switchcap |
| xl | **11/20** | 2/5 | 1/5 | 5/5 | 3/5 | 1 | tsmc5-ring_osc, tsmc7-ring_osc, tsmc12-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc16-opamp, tsmc12-switchcap, tsmc16-switchcap |

## 2. By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 8.15% | **PASS** 4.83% | FAIL 9.72% | **PASS** 3.89% | **PASS** 2.90% |
| medium | FAIL 8.87% | FAIL 7.21% | FAIL 9.85% | **PASS** 3.44% | ⚡FLIP 3.24% |
| large | FAIL 9.06% | FAIL 9.12% | FAIL 8.32% | **PASS** 2.52% | ⚡FLIP 2.86% |
| xl | FAIL 5.61% | **PASS** 0.08% | FAIL 10.53% | FAIL 5.21% | **PASS** 2.38% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | **PASS** 6.23% | **PASS** 6.53% |
| medium | FAIL 100.00% | FAIL 100.00% | **PASS** 5.45% | **PASS** 5.33% | FAIL 100.00% |
| large | **PASS** 6.73% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| xl | ⚡FLIP 23.07% | FAIL 100.00% | FAIL 100.00% | **PASS** 7.31% | FAIL 100.00% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 7.18% | **PASS** 1.59% | **PASS** 1.54% | **PASS** 1.55% | **PASS** 2.62% |
| medium | **PASS** 6.91% | **PASS** 1.77% | **PASS** 1.65% | **PASS** 3.16% | **PASS** 5.93% |
| large | **PASS** 6.12% | **PASS** 1.88% | **PASS** 1.54% | **PASS** 1.31% | **PASS** 1.85% |
| xl | **PASS** 6.71% | **PASS** 1.69% | **PASS** 1.45% | **PASS** 2.29% | **PASS** 1.66% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 2.14% | **PASS** 2.64% | **PASS** 3.07% | FAIL 5.32% | **PASS** 3.80% |
| medium | **PASS** 2.77% | **PASS** 3.74% | **PASS** 2.59% | FAIL 5.10% | **PASS** 4.19% |
| large | **PASS** 2.84% | **PASS** 2.83% | **PASS** 3.08% | FAIL 5.30% | FAIL 4.99%† |
| xl | **PASS** 2.82% | **PASS** 2.65% | **PASS** 3.83% | FAIL 5.82% | FAIL 5.09% |

† failed on **hold droop**, the half of this gate the headline number does not show — the metric above is inside its threshold.

## 3. By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 0/4 | 1/4 | 4/4 | 4/4 | **9/16** |
| **TSMC6** | 2/4 | 0/4 | 4/4 | 4/4 | **10/16** |
| **TSMC7** | 0/4 | 1/4 | 4/4 | 4/4 | **9/16** |
| **TSMC12** | 3/4 | 3/4 | 4/4 | 0/4 | **10/16** |
| **TSMC16** | 2/4 | 1/4 | 4/4 | 2/4 | **9/16** |

PFN's distinctive failure is **`switchcap` on TSMC12**, which it misses by a
tenth to a third of a percentage point — the closest open cells in the project.
It is a charge/off-state-surface miss, and it is the one place where PFN is
weaker than both other families rather than merely smaller.

TSMC12 is also the tech whose off-grid NFIN=10 case probes the sampling gap
between the PDK's 6 and 21 fin bins, which is where an in-context model's
geometry interpolation is most exposed: PFN interpolates by attending to
context rows, so a bin with no near neighbour has nothing to attend to.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 2/4 | 3/4 | 2/4 | 3/4 | 4/4 | **14/20** |
| medium | 2/4 | 2/4 | 3/4 | 3/4 | 2/4 | **12/20** |
| large | 3/4 | 2/4 | 2/4 | 2/4 | 1/4 | **10/20** |
| xl | 2/4 | 3/4 | 2/4 | 2/4 | 2/4 | **11/20** |

**PFN declines past `medium`, and the decline survived the `gds` fix.** Two
things are going on and they should not be conflated:

* The capacity story proper — the same "beyond the sweet spot, extra capacity
  buys a tighter fit and loses fixed points" law that both other families obey.
* **`large` is optimization-unstable**: its training runs recorded eight
  divergence-collapse events. Part of that tier's weakness is that its runs are
  not reliably converging, which is a trainability problem, not a capacity one.

`xl` should be read with a caveat of its own: six of its eight runs banked
before epoch 50 of 150, so it is not a fully-converged tier either.

## 5. Device-level suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.14 | 1.46 | 1.58 | 0.56 | 1.12 (13/14) | 68/69 |
| medium | 2.36 | 2.28 (13/14) | 1.62 | 1.95 (17/18) | 2.65 (13/14) | 66/69 |
| large | 2.65 | 1.93 | 1.52 | 1.04 | 1.10 | 69/69 |
| xl | 1.54 | 2.53 | 1.94 | 1.18 | 1.38 | 69/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.88 | 1.43 | 1.44 | 1.50 | 1.48 | 80/80 |
| medium | 1.71 | 1.53 | 1.48 | 1.50 | 1.49 | 80/80 |
| large | 2.23 | 1.43 | 1.49 | 1.50 | 1.51 | 80/80 |
| xl | 1.87 | 1.56 | 1.45 | 2.45 | 1.50 | 80/80 |

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
| small | FAIL 16.04 dB | FAIL 32.11 dB | FAIL 30.04 dB | FAIL 8.46 dB | FAIL 4.44 dB | **0/5** |
| medium | FAIL 15.61 dB | FAIL 31.53 dB | **PASS** 0.84 dB | FAIL 3.29 dB | FAIL 33.44 dB | **1/5** |
| large | FAIL 5.32 dB | FAIL 28.86 dB | FAIL 30.99 dB | FAIL 30.92 dB | FAIL 36.34 dB | **0/5** |
| xl | FAIL 12.37 dB | FAIL 32.86 dB | FAIL 61.23 dB | FAIL 65.95 dB | FAIL 34.39 dB | **0/5** |

Device-level, PFN is competitive with both other families — its weakness is
entirely at the circuit level. That is the same pattern the whole project
shows, and it is worth stating once more here because PFN is the family where
someone is most likely to reach for device metrics as evidence: they will look
fine, and they do not predict the gates.

## 6. Determinism

PFN was for a while the project's only flip-free family, which was read as a
property of the architecture. That reading is **retired**: after the V6.13.0
`gds` fix every family became flip-free, so the earlier uniqueness was a
statement about the other two families' bug exposure, not about PFN.

The V7.1.0 strict sweep then recorded a flip at two PFN tiers — so "every group
in every family is flip-free" is *also* too strong. The flip count in §1 is the
current measurement; treat any group with a flip as unbankable, per
`methodology.md` §3, regardless of what it scores single-run.

## 7. What is open

| open | reading |
|---|---|
| **`tsmc12-switchcap`** | Misses by 0.1–0.3 pp at every tier. The closest open cell in the project and the one most likely to fall to a targeted charge-surface lever. |
| **Low-VDD rings and opamps** | The same cells the other families fail, and the corridor curriculum is the known lever — see the recipes report, where it is run on PFN for the first time. |
| **`large` trainability** | Divergence-collapse events, not a capacity ceiling. Worth separating before drawing a capacity conclusion about this tier. |
| **`xl` convergence** | Most runs banked before a third of the schedule. |

## 8. Reproducing

```bash
# every driver takes MODEL=tabpfn; the gate driver's tag is pfn
bash scripts/v710_regate.sh _one pfn small TSMC12 verify_complex_switchcap 1
python scripts/v730_coverage.py --set clean --tag pfn
```

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_pfn_{small,medium,large,xl}_{nmos,pmos}`,
each with `_norm.npz` and a **required** `_config.npz`. Env pins
`PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`; hook
`PYCIRCUITSIM_NN_FORCE_LEVEL=75` retargets a LEVEL=73 deck at PFN, so the whole
gate infrastructure runs this family with zero deck changes.
