# DirectNet (LEVEL=73) — the clean recipe

**What it is.** A feed-forward MLP compact model. Seven inputs (Vgs, Vds, Vbs,
NFIN, L, T, and a tech code through `nn.Embedding`), thirteen outputs.
`gm`/`gds`/`gmb` are the **autograd Jacobian** of the predicted `id`, and the
small-signal capacitances are the `dQ/dV` autograd of the predicted terminal
charges — nothing is fitted twice.

**What "clean" means.** One training run, no addendum:
`--apply-filter off --swa-mode ema --seed 42`. It is the control every recipe
in [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md) is measured against.
DirectNet's production checkpoint is *not* clean — the `tsmc{X}_dn_large_*`
slots have carried the `crit30f` curriculum since V6.6.4 — so the clean `large`
row below reads the archived `v660clean_large` stems. On TSMC6, which never had
a curriculum applied, `large` is itself clean.

Gate definitions, the strict-OMP rule, the `gds`-fix code ladder and the
measured noise floor: [`methodology.md`](methodology.md). Read it before
comparing any two numbers here.

| tier | width × depth | params | CPU cost, 1 thread |
|---|---|---|---|
| `small` | 128 × 3 | ≈ 0.06 M | — |
| `medium` | 256 × 5 | ≈ 0.40 M | — |
| `large` | 384 × 6 | **≈ 0.92 M** | **1.5 ms/eval** |
| `xl` | 512 × 8 | ≈ 2.13 M | 3.4 ms/eval |

> **Denominators changed in V7.3.0.** TSMC6 is now folded into the headline, so
> complex totals are **/20**, device AC **/10** and opamp AC **/5**. Every
> earlier report scored /16, /8 and /4 over the four electrically distinct
> techs. **No total here is comparable to a pre-V7.3.0 total without
> rescaling.** TSMC6 remains TSMC7 relabelled (`methodology.md` §7); what
> changed is the denominator, not that finding.

---

## 1. Headline — complex gates by tier

Strict: a cell passes only if it passes at OMP ∈ {1, 2, 4}.

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **12/20** | 3/5 | 3/5 | 5/5 | 1/5 | 0 | tsmc5-ring_osc, tsmc7-ring_osc, tsmc6-opamp, tsmc12-opamp, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| medium | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |
| large | **16/20** | 3/5 | 3/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-opamp, tsmc16-opamp |
| xl | **14/20** | 2/5 | 2/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc12-opamp |

## 2. By testcase

The four circuits load different parts of the NN surface — the ring the
switching edge, the opamp the high-gain fixed point, the SRAM the bistable
latch, the switchcap the charge and off-state surface. A family can be
excellent on three and fail the fourth, so the per-testcase view is the one
that localizes a weakness.

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 8.06% | **PASS** 4.32% | FAIL 5.94% | **PASS** 1.95% | **PASS** 1.47% |
| medium | FAIL 5.89% | FAIL 9.37% | FAIL 10.86% | **PASS** 2.26% | **PASS** 2.22% |
| large | FAIL 11.47% | FAIL 9.04% | **PASS** 4.82% | **PASS** 2.17% | **PASS** 2.23% |
| xl | FAIL 12.98% | FAIL 15.05% | FAIL 12.78% | **PASS** 3.05% | **PASS** 2.78% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 0.88% | FAIL 11.15% | **PASS** 1.81% | FAIL 100.00% | **PASS** 6.60% |
| medium | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| large | **PASS** 5.08% | **PASS** 7.12% | FAIL 100.00% | **PASS** 6.26% | FAIL 100.00% |
| xl | FAIL 100.00% | FAIL 100.00% | **PASS** 4.20% | FAIL 100.00% | **PASS** 6.24% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 5.07% | **PASS** 1.68% | **PASS** 1.91% | **PASS** 1.51% | **PASS** 2.62% |
| medium | **PASS** 7.98% | **PASS** 2.10% | **PASS** 2.22% | **PASS** 1.21% | **PASS** 2.41% |
| large | **PASS** 6.04% | **PASS** 1.58% | **PASS** 2.50% | **PASS** 1.88% | **PASS** 1.74% |
| xl | **PASS** 5.90% | **PASS** 1.86% | **PASS** 1.92% | **PASS** 4.39% | **PASS** 2.22% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 1.72% | FAIL 2.27%† | FAIL 2.34%† | FAIL 4.09%† | FAIL 2.76%† |
| medium | **PASS** 1.54% | **PASS** 2.75% | **PASS** 2.81% | **PASS** 4.19% | **PASS** 3.22% |
| large | **PASS** 3.48% | **PASS** 2.52% | **PASS** 2.45% | **PASS** 4.14% | **PASS** 3.32% |
| xl | **PASS** 3.18% | **PASS** 2.65% | **PASS** 2.67% | **PASS** 4.19% | **PASS** 3.42% |

† failed on **hold droop**, the half of this gate the headline number does not show — the metric above is inside its threshold.

## 3. By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 0/4 | 2/4 | 4/4 | 4/4 | **10/16** |
| **TSMC6** | 1/4 | 1/4 | 4/4 | 3/4 | **9/16** |
| **TSMC7** | 1/4 | 2/4 | 4/4 | 3/4 | **10/16** |
| **TSMC12** | 4/4 | 1/4 | 4/4 | 3/4 | **12/16** |
| **TSMC16** | 4/4 | 2/4 | 4/4 | 3/4 | **13/16** |

The split that matters is **supply voltage, not vendor node**. TSMC5 and TSMC7
at 0.65–0.75 V behave as one class and TSMC12/16 at 0.80 V as another: the
low-VDD pair own nearly every ring failure, because their transfer curves are
steepest exactly where the NN under-drives. TSMC6 sits by construction on top
of TSMC7, and the two disagree only by training-run luck — which is precisely
what makes it useful (`methodology.md` §7).

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 3/4 | 2/4 | 2/4 | 2/4 | 3/4 | **12/20** |
| medium | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |
| large | 3/4 | 3/4 | 3/4 | 4/4 | 3/4 | **16/20** |
| xl | 2/4 | 2/4 | 3/4 | 3/4 | 4/4 | **14/20** |

**DirectNet peaks at `large`.** Beyond it the model fits the training
distribution roughly ten times tighter — validation loss ~2e-4 against
`medium`'s ~2e-3 — and generalizes *worse* off-nominal. Extra capacity buys a
tighter fit to the data and loses circuit fixed points. This is the one
capacity curve in the project that survived the `gds` fix unchanged.

## 5. Device-level suites

Parametric DC is **`gds`-invariant**, so its numbers are comparable across the
whole campaign history; transient and AC are not (`methodology.md` §6).

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.39 | 2.06 | 1.81 | 0.57 | 0.62 | 69/69 |
| medium | 1.48 | 2.21 | 1.46 | 0.56 | 0.58 | 69/69 |
| large | 2.60 | 2.25 | 1.31 | 1.72 (17/18) | 0.93 | 68/69 |
| xl | 2.91 | 3.18 (13/14) | 2.35 | 2.73 (16/18) | 1.52 | 66/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.99 | 1.57 | 1.53 | 2.07 | 1.62 | 80/80 |
| medium | 1.90 | 1.45 | 1.48 | 1.52 | 1.47 | 80/80 |
| large | 1.68 | 1.45 | 1.46 | 1.50 | 1.47 | 80/80 |
| xl | 1.66 | 1.46 | 1.45 | 1.52 | 1.48 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✓ / ✓ | ✓ / ✓ | ✗ gain 2.026 dB, mag 24.89 % / ✓ | ✓ / ✓ | ✓ / ✓ | **9/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✗ f3db 1.78 / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **9/10** |
| xl | ✗ f3db 2.51 / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **9/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 2.41 dB | FAIL 3.22 dB | FAIL 19.50 dB | FAIL 42.27 dB | **PASS** 1.78 dB | **1/5** |
| medium | FAIL 22.02 dB | FAIL 30.11 dB | FAIL 29.46 dB | FAIL 27.69 dB | FAIL 35.11 dB | **0/5** |
| large | FAIL 2.31 dB | FAIL 31.67 dB | FAIL 31.05 dB | FAIL 5.12 dB | FAIL 34.41 dB | **0/5** |
| xl | FAIL 14.72 dB | FAIL 30.35 dB | FAIL 31.48 dB | FAIL 208.73 dB | FAIL 3.73 dB | **0/5** |

Reading the DC column: **`medium` is the best device fit**, while `large` wins
the *circuit* gates. That inversion is the central result of this project in
one line — circuit gates measure the value surface at fixed points, not
pointwise device accuracy. Device fidelity has never been the bind.

## 6. What the AC gates actually diagnose

DC gain is excellent everywhere, so the autograd `gm`/`gds` feeding the AC
stamp are accurate, and the dominant cap-driven pole is faithful wherever the
cell is well fit. Three limits are real and specific:

1. The **Cgd-feedforward RHP zero** — high-frequency phase lag — is not
   reproduced. It is *reported* by `verify_nn_ac` as a diagnostic and
   deliberately not gated.
2. Some cells **under-predict the output capacitance** (f3db ratio 1.1–2.5) and
   miss the magnitude gate.
3. The opamp AC inherits the DC value-surface fragility: where the operating
   point lands in the good basin the NN nails GBW and phase margin, so the
   *dynamics* are right and only the DC-gain *level* misses.

These are value-surface and feedforward limits, **not** a charge-derivative
deficiency — that would corrupt gain *and* pole everywhere, and it does not.

> **The opamp open-loop AC row is a lower bound, not a result.**
> `verify_complex_opamp_ac` picks its bias by `argmax |dVout/dVin|` on a 2 mV
> grid, but a two-stage Miller opamp with 33–48 dB of gain has a transition
> only 3–14 mV wide. On three of the four original techs the **NGSPICE
> reference's own** operating point at the chosen bias falls outside the
> gate's 15–85 %·VDD validity window, and that window is then applied to the NN
> alone — so a model that faithfully reproduces the reference is scored
> `OP-MISBIAS`. This is a gate-construction defect and is deliberately **not**
> fixed here: changing an accuracy gate changes the accuracy record, which is a
> separate decision.

## 7. What is open

| open | reading |
|---|---|
| **Low-VDD rings under every clean tier** | Deterministic, not marginal. Closed only by the corridor curriculum — see the recipes report. |
| **`tsmc7-opamp` at `large`** | A tier-specific basin, not a family wall: DirectNet banks it at `small` and `xl`, and BSIM-AR at every size. |
| **`tsmc5-opamp` margin at production** | The thinnest margin in the matrix. Treat any recipe change as a threat to it. |
| **Opamp open-loop AC** | Partly unreachable by construction; see the caveat above. |

Every one of these is a **value-surface / fixed-point** property — open-loop
gain, oscillation period at a sharp edge — not a device-current or
charge-derivative fidelity gap. Device DC, inverter, switchcap, SRAM and AC
gain are all strong.

## 8. Reproducing

```bash
# gate one (tier, tech) cell, isolated and CPU-pinned
bash scripts/v710_regate.sh _one dn xl TSMC12 verify_complex_ring_osc 1

# the coverage map: what is measured, by which pass, and what is missing
python scripts/v730_coverage.py --set clean --tag dn

# rebuild this file from the evidence
python scripts/v710_regate_collect.py --root results/v730_regate
python scripts/v730_docs_build.py
```

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_dn_{small,medium,large,xl}_{nmos,pmos}`,
with the clean `large` control archived as `tsmc{X}_dn_v660clean_large_*`.
Raw runs: `results/v710_regate/`, `results/v730_regate/`, `results/a3_regate/`.
