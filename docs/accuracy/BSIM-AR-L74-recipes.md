# BSIM-AR (LEVEL=74) — recipe variants

A **recipe** is one identical training addendum applied to every
(tech × device × size) checkpoint (`methodology.md` §5). The control is the
clean model in [`BSIM-AR-L74-clean.md`](BSIM-AR-L74-clean.md); the recipe map
is shared with DirectNet, so the same flags mean the same thing in both
families — which is what makes the comparison in §4 possible at all.

> **Evidence boundary.** Recipe rows in this file are retained from V7.3.0.
> V7.4.0 rebuilt the clean BSIM-AR matrix on new hardware but did not retrain
> these curricula, so recipe-to-clean deltas below refer to the V7.3.0 clean
> controls. They must not be subtracted from the V7.4.0 clean table.

> **Read the noise floor before ranking anything by one cell.** `ring_osc`
> carries ±4 pp of run-to-run scatter across a 5 % gate and `opamp` is bimodal
> for two of the three families; `sram_snm` and `switchcap` reproduce to
> ≤0.3 pp (`methodology.md` §7). BSIM-AR happens to be the family where this
> bit least in the V7.3.0 recipe campaign — that repeat reproduced every
> verdict — but the recipes are still ranked on the same noisy cells. The
> V7.4.0 clean repeat is 15/16 (`BSIM-AR-L74-clean.md` §6).

> **Denominators.** Totals are **/20** — 4 circuits × 5 techs, TSMC6 included (`methodology.md` §2). Earlier reports scored /16 over four techs, so a /20 total here and a /16 total there can be the same measurement.

---

## 1. The kept recipes

All are 120-epoch fine-tunes at lr 3e-4, patience 40, warm-started from their
own tier's clean checkpoint, on the ring-only `corro` corridor dataset.

| recipe | tiers | class weights |
|---|---|---|
| **`corroft`** | `medium`, `large`, `xl` | `traj_corridor=3.0` |
| **`corro15`** | `medium`, `xl` | `traj_corridor=1.5` |
| `crit15m` | `large`, `xl` | `traj_corridor=1.5, inv_trip=3.0` |
| `crit30` | `large`, `xl` | `traj_corridor=3.0, inv_trip=2.0` |

Dropped by the V7.3.0 filter: `invtrip@large` and `csob@xl`, which both land on
exactly clean's failure set — they are not levers on this family, and keeping
them would pad the table with rows that say nothing. They are recorded in §6.

## 2. Gates by recipe

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| `corroft`@medium | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `corro15`@medium | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `corroft`@large | **19/20** | 5/5 | 4/5 | 5/5 | 5/5 | 0 | tsmc7-opamp |
| `crit15m`@large | **18/20** | 5/5 | 3/5 | 5/5 | 5/5 | 0 | tsmc6-opamp, tsmc7-opamp |
| `crit30`@large | **19/20** | 5/5 | 4/5 | 5/5 | 5/5 | 0 | tsmc7-opamp |
| `corroft`@xl | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `corro15`@xl | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `crit15m`@xl | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `crit30`@xl | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |

## 3. What each recipe changed against clean

| recipe | tier | cells gained vs clean | cells lost vs clean | net |
|---|---|---|---|---|
| `corroft` | medium | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | — | **+3** |
| `corro15` | medium | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | — | **+3** |
| `corroft` | large | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | tsmc7-opamp | **+2** |
| `crit15m` | large | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | tsmc6-opamp, tsmc7-opamp | **+1** |
| `crit30` | large | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | tsmc7-opamp | **+2** |
| `corroft` | xl | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | — | **+3** |
| `corro15` | xl | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | — | **+3** |
| `crit15m` | xl | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | — | **+3** |
| `crit30` | xl | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc | — | **+3** |

## 3a. Six groups sweep 20/20 — including the duplicate column

`corroft`@`medium`, `corro15`@`medium` and all four corridor recipes at `xl`
pass **every cell of the matrix**, TSMC6 included. These are the first full
sweeps recorded in this project, and the fifth column is what makes them worth
more than the old 16/16s: a sweep that includes a duplicate technology has
survived the run-to-run lottery on both copies of it, which the DirectNet
recipes did not (`DirectNet-L73-recipes.md` §4a — `crit15m@xl` passes
`tsmc7-opamp` and fails its `tsmc6` twin).

The one place BSIM-AR shows the same coin-flip is `crit15m`@`large`, which
fails `tsmc6-opamp` where `corroft`@`large` and `crit30`@`large` — same tier,
same data — do not. That is the exception that locates the rule: it is at
`large`, the tier §4 already identifies as this family's worst corridor tier.

## 4. The recipe laws, as they apply to BSIM-AR

**The corridor is the ring lever here too.** Every BSIM-AR recipe that closes a
low-VDD ring has `traj_corridor` in it, and clean fails those rings at every
tier. On this family the corridor takes `tsmc5-ring` from the 5.5–7.6 % band to
around 3.3 % — a move of roughly 4 pp on a 5 % gate, which is at the edge of
the noise floor for a single cell but consistent across eight of them.

**The `inv_trip` anchor is inert on BSIM-AR.** `corroft` and `crit30` agree to
under 0.5 % on every cell, and `invtrip` alone reproduces clean's failure set.
This is a genuine family difference: on DirectNet the same anchor composes with
the corridor and banks an opamp. Pre-fix the anchor looked actively *harmful*
here — `invtrip@large` railed `tsmc7-opamp` — but post-fix both pass, so that
reading was the `gds` floor railing the opamp, not the anchor.

**"The recipe decides which opamp basin you get" is retracted for BSIM-AR.**
Post-fix, the `large` curricula agree with one another and the `xl` curricula
agree with one another. The recipe discriminates only on rings in this family.
It still discriminates on opamps in DirectNet, at the same tier and on the same
data — so this is architectural, not a property of the recipes.

**Capacity and recipe interact in the opposite direction to DirectNet's.** The
corridor removes BSIM-AR's flatness by making `large` the *worst* corridor tier
rather than the best, while `medium` and `xl` both do better. DirectNet's
curriculum inverts its capacity story the other way, lifting `xl` above
`large`. Two families, one recipe map, opposite tier preferences: the
weight → basin map is not portable across architectures.

## 5. Per testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@medium | **PASS** 3.33% | **PASS** 2.09% | **PASS** 2.25% | **PASS** 2.13% | **PASS** 2.19% |
| `corro15`@medium | **PASS** 3.40% | **PASS** 2.09% | **PASS** 2.31% | **PASS** 2.16% | **PASS** 2.13% |
| `corroft`@large | **PASS** 3.88% | **PASS** 2.04% | **PASS** 2.31% | **PASS** 2.08% | **PASS** 2.53% |
| `crit15m`@large | **PASS** 3.93% | **PASS** 1.82% | **PASS** 2.34% | **PASS** 2.07% | **PASS** 2.52% |
| `crit30`@large | **PASS** 3.85% | **PASS** 2.04% | **PASS** 2.32% | **PASS** 2.08% | **PASS** 2.54% |
| `corroft`@xl | **PASS** 3.83% | **PASS** 2.27% | **PASS** 2.21% | **PASS** 2.57% | **PASS** 2.77% |
| `corro15`@xl | **PASS** 3.86% | **PASS** 2.12% | **PASS** 1.98% | **PASS** 2.53% | **PASS** 2.77% |
| `crit15m`@xl | **PASS** 3.87% | **PASS** 2.11% | **PASS** 1.91% | **PASS** 2.54% | **PASS** 2.76% |
| `crit30`@xl | **PASS** 3.88% | **PASS** 2.24% | **PASS** 2.24% | **PASS** 2.55% | **PASS** 2.75% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@medium | **PASS** 2.52% | **PASS** 7.61% | **PASS** 6.73% | **PASS** 5.32% | **PASS** 5.82% |
| `corro15`@medium | **PASS** 2.38% | **PASS** 7.59% | **PASS** 6.81% | **PASS** 5.29% | **PASS** 6.08% |
| `corroft`@large | **PASS** 2.89% | **PASS** 8.65% | FAIL 100.00% | **PASS** 5.69% | **PASS** 5.28% |
| `crit15m`@large | **PASS** 3.68% | FAIL 100.00% | FAIL 100.00% | **PASS** 5.63% | **PASS** 5.39% |
| `crit30`@large | **PASS** 3.14% | **PASS** 8.53% | FAIL 100.00% | **PASS** 5.64% | **PASS** 5.56% |
| `corroft`@xl | **PASS** 4.06% | **PASS** 7.59% | **PASS** 7.21% | **PASS** 5.91% | **PASS** 6.00% |
| `corro15`@xl | **PASS** 3.51% | **PASS** 8.00% | **PASS** 7.27% | **PASS** 5.98% | **PASS** 5.92% |
| `crit15m`@xl | **PASS** 3.76% | **PASS** 7.96% | **PASS** 7.06% | **PASS** 6.08% | **PASS** 5.81% |
| `crit30`@xl | **PASS** 4.17% | **PASS** 7.67% | **PASS** 7.07% | **PASS** 5.85% | **PASS** 5.83% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@medium | **PASS** 6.11% | **PASS** 1.72% | **PASS** 1.43% | **PASS** 3.47% | **PASS** 1.34% |
| `corro15`@medium | **PASS** 6.10% | **PASS** 1.72% | **PASS** 1.43% | **PASS** 3.57% | **PASS** 1.31% |
| `corroft`@large | **PASS** 6.60% | **PASS** 3.01% | **PASS** 2.25% | **PASS** 1.31% | **PASS** 1.66% |
| `crit15m`@large | **PASS** 6.54% | **PASS** 2.92% | **PASS** 2.16% | **PASS** 1.27% | **PASS** 1.58% |
| `crit30`@large | **PASS** 6.57% | **PASS** 3.04% | **PASS** 2.21% | **PASS** 1.25% | **PASS** 1.62% |
| `corroft`@xl | **PASS** 6.17% | **PASS** 1.64% | **PASS** 1.89% | **PASS** 1.81% | **PASS** 1.54% |
| `corro15`@xl | **PASS** 6.13% | **PASS** 1.78% | **PASS** 1.85% | **PASS** 1.77% | **PASS** 1.55% |
| `crit15m`@xl | **PASS** 6.11% | **PASS** 1.73% | **PASS** 1.90% | **PASS** 1.66% | **PASS** 1.83% |
| `crit30`@xl | **PASS** 6.17% | **PASS** 1.70% | **PASS** 1.84% | **PASS** 1.75% | **PASS** 1.78% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@medium | **PASS** 1.99% | **PASS** 2.33% | **PASS** 2.33% | **PASS** 4.15% | **PASS** 3.36% |
| `corro15`@medium | **PASS** 2.01% | **PASS** 2.32% | **PASS** 2.35% | **PASS** 4.12% | **PASS** 3.38% |
| `corroft`@large | **PASS** 1.69% | **PASS** 2.23% | **PASS** 2.27% | **PASS** 4.14% | **PASS** 3.36% |
| `crit15m`@large | **PASS** 1.74% | **PASS** 2.20% | **PASS** 2.26% | **PASS** 4.15% | **PASS** 3.37% |
| `crit30`@large | **PASS** 1.68% | **PASS** 2.22% | **PASS** 2.25% | **PASS** 4.14% | **PASS** 3.38% |
| `corroft`@xl | **PASS** 1.74% | **PASS** 2.24% | **PASS** 2.23% | **PASS** 4.25% | **PASS** 3.39% |
| `corro15`@xl | **PASS** 1.74% | **PASS** 2.24% | **PASS** 2.21% | **PASS** 4.25% | **PASS** 3.39% |
| `crit15m`@xl | **PASS** 1.74% | **PASS** 2.23% | **PASS** 2.22% | **PASS** 4.25% | **PASS** 3.38% |
| `crit30`@xl | **PASS** 1.75% | **PASS** 2.24% | **PASS** 2.24% | **PASS** 4.25% | **PASS** 3.37% |

## 6. Device fidelity and AC by recipe

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| `corroft`@medium | — | 1.38 | 1.00 | 1.15 (17/18) | 1.54 (13/14) | 53/55 |
| `corro15`@medium | 1.58 | 1.43 | 1.00 | 1.17 (17/18) | 1.56 (13/14) | 67/69 |
| `corroft`@large | 1.25 | 2.00 | 0.89 | 1.63 (17/18) | 1.52 (13/14) | 67/69 |
| `crit15m`@large | 1.37 | 1.99 | 0.84 | 1.57 (17/18) | 1.53 | 68/69 |
| `crit30`@large | 1.26 | 2.01 | 0.86 | 1.55 (17/18) | 1.57 (13/14) | 67/69 |
| `corroft`@xl | 1.36 | 2.82 | 3.22 | 1.12 | 0.94 | 69/69 |
| `corro15`@xl | 1.41 | 3.22 | 3.21 | 1.23 | 1.06 | 69/69 |
| `crit15m`@xl | 1.40 | 3.15 | 3.27 | 1.20 | 0.94 | 69/69 |
| `crit30`@xl | 1.34 | 2.97 | 3.24 | 1.26 | 1.00 | 69/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| `corroft`@medium | — | 1.48 | 1.53 | 1.50 | 1.49 | 64/64 |
| `corro15`@medium | 1.95 | 1.48 | 1.53 | 1.51 | 1.50 | 80/80 |
| `corroft`@large | 1.66 | 1.49 | 1.47 | 1.50 | 1.47 | 80/80 |
| `crit15m`@large | 1.67 | 1.48 | 1.47 | 1.51 | 1.48 | 80/80 |
| `crit30`@large | 1.67 | 1.48 | 1.46 | 1.51 | 1.48 | 80/80 |
| `corroft`@xl | 1.69 | 1.48 | 1.48 | 1.50 | 1.49 | 80/80 |
| `corro15`@xl | 1.68 | 1.48 | 1.47 | 1.50 | 1.49 | 80/80 |
| `crit15m`@xl | 1.68 | 1.48 | 1.47 | 1.50 | 1.49 | 80/80 |
| `crit30`@xl | 1.69 | 1.48 | 1.48 | 1.50 | 1.49 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| `corroft`@medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `corro15`@medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `corroft`@large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `crit15m`@large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `crit30`@large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `corroft`@xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `corro15`@xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `crit15m`@xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| `crit30`@xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| `corroft`@medium | FAIL 28.12 dB | FAIL 8.09 dB | **PASS** 2.78 dB | FAIL 3.14 dB | **PASS** 1.90 dB | **2/5** |
| `corro15`@medium | FAIL 10.63 dB | FAIL 7.96 dB | FAIL 3.40 dB | **PASS** 2.35 dB | **PASS** 2.70 dB | **2/5** |
| `corroft`@large | FAIL 2.69 dB | FAIL 7.69 dB | FAIL 31.09 dB | **PASS** 2.72 dB | **PASS** 2.52 dB | **2/5** |
| `crit15m`@large | FAIL 1.34 dB | FAIL 31.09 dB | FAIL 31.22 dB | **PASS** 2.90 dB | **PASS** 1.38 dB | **2/5** |
| `crit30`@large | FAIL 1.86 dB | FAIL 7.79 dB | FAIL 31.36 dB | **PASS** 2.73 dB | **PASS** 0.23 dB | **2/5** |
| `corroft`@xl | FAIL 5.18 dB | FAIL 8.62 dB | FAIL 8.26 dB | FAIL 6.24 dB | **PASS** 2.26 dB | **1/5** |
| `corro15`@xl | FAIL 0.60 dB | FAIL 8.18 dB | FAIL 8.30 dB | FAIL 6.67 dB | **PASS** 1.47 dB | **1/5** |
| `crit15m`@xl | FAIL 0.40 dB | FAIL 8.15 dB | FAIL 7.68 dB | FAIL 6.84 dB | **PASS** 0.92 dB | **1/5** |
| `crit30`@xl | FAIL 0.26 dB | FAIL 8.58 dB | FAIL 8.01 dB | FAIL 6.17 dB | **PASS** 0.80 dB | **1/5** |

## 7. Dead ends

| arm | verdict |
|---|---|
| **`invtrip`@`large`** | Inert — lands on clean's failure set. Dropped from the kept table. |
| **`csob`@`xl`** | Inert on the complex matrix — ties clean. Retained only as a device/charge reference. |
| **`xl` as a promotion target** | 7.7× `medium`'s parameters and ~11 days of training to tie it. The V6.8.1 reading that "AC collapses at xl" is separately **retracted**: post-fix, `xl` banks the TSMC16 opamp-AC cell, and the `tsmc7-opamp-AC` run that was recorded as never converging after ~6 h now completes. That pathology was the railed operating point, i.e. the `gds` bug. |

## 8. Recommendation

`corroft@medium` is the fidelity option: a full 20/20 sweep at 1.94 M
parameters, with comfortable margins and the family's reproducibility advantage
behind it. `large` is the worst corridor tier, and `xl` sweeps too but ties
`medium` at 7.7× the parameters — so `medium` wins on cost alone.

**It is the only checkpoint in the project that passes every cell**, and the
reason to trust that more than a bare score is §3a: it holds both copies of the
duplicated technology, so it is not one lucky basin on a bimodal cell.

It is **not** production, and the reason is cost alone: ~61.5 ms/eval against
DirectNet's 1.5 ms. Use it where fidelity is worth ~40× the runtime.
