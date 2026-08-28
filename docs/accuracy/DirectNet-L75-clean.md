# DirectNet-Full (LEVEL=75) — clean full-terminal scan

This experimental DirectNet learns six independent OSDI terminal surfaces:
`i_d`, `i_g`, `i_b`, `qd`, `qg`, and `qb`. Source current and charge are
closed analytically, and the solver stamps the full current and charge
Jacobians. LEVEL=72 on the identical BSIM-CMG OSDI model is the reference.

Clean means one uniform run per technology, polarity, and tier with
`--output-contract full-terminal --apply-filter off --swa-mode ema --seed 42`.
No reduced-head loss or correction path is enabled.

Evidence pass: **V7.6.1**. Campaign manifest SHA-256 `defd42a5c0a1f5e76f059e4306a3e16951a9960b6444878c40d883031d908f65` pins gate commit `bfa3630daa3951453088ff12784c4763daf2d53e`, 240 jobs, and 120 checkpoint artifacts. Raw evidence: `results/v761_directnet_full_clean/`.

Gate definitions, strict OMP scoring, denominator rules, and comparability are
owned by [`methodology.md`](methodology.md).

## Headline — complex gates by tier

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **0/20** | 0/5 | 0/5 | 0/5 | 0/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc12-ring_osc, tsmc16-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| medium | **2/20** | 0/5 | 2/5 | 0/5 | 0/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc12-ring_osc, tsmc16-ring_osc, tsmc5-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| large | **0/20** | 0/5 | 0/5 | 0/5 | 0/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc12-ring_osc, tsmc16-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| xl | **1/20** | 0/5 | 1/5 | 0/5 | 0/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc12-ring_osc, tsmc16-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |

Only three of the 80 strict complex cells pass: the medium TSMC6/TSMC7
opamps and the XL TSMC16 opamp. The tier curve is **0 → 2 → 0 → 1/20**;
larger models therefore show no monotonic qualification gain.

## By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR |
| medium | ERROR | ERROR | ERROR | ERROR | ERROR |
| large | ERROR | ERROR | ERROR | ERROR | ERROR |
| xl | ERROR | ERROR | ERROR | ERROR | ERROR |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR |
| medium | ERROR | **PASS** 0.10% | **PASS** 0.10% | ERROR | ERROR |
| large | ERROR | ERROR 0.03% | ERROR 0.03% | ERROR 0.17% | ERROR |
| xl | ERROR | ERROR | ERROR | ERROR 0.78% | **PASS** 1.35% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | ERROR 4.65% | ERROR 1.88% | ERROR 1.88% | ERROR 1.52% | ERROR 1.70% |
| medium | ERROR 4.86% | ERROR 1.62% | ERROR 1.62% | ERROR 0.40% | ERROR 0.85% |
| large | ERROR 6.92% | ERROR 0.16% | ERROR 0.16% | ERROR 0.16% | ERROR |
| xl | ERROR | ERROR | ERROR | ERROR 0.15% | ERROR 0.17% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR |
| medium | ERROR | ERROR | ERROR | ERROR | ERROR |
| large | ERROR | ERROR | ERROR | ERROR | ERROR |
| xl | ERROR | ERROR | ERROR | ERROR | ERROR |

Every ring-oscillator and switched-capacitor cell is an explicit `ERROR`.
Every SRAM cell also errors before all required lobes complete, even where a
completed lobe has a small diagnostic NRMSE. Most failures are certified-input
support violations during circuit iteration—for example, a PMOS source-relative
drain voltage of 0.65 V against a recorded upper bound of 0.52 V. These are
scientific/runtime qualification failures, not missing jobs, and remain in the
denominator without invented numeric metrics.

## By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 0/4 | 0/4 | 0/4 | 0/4 | **0/16** |
| **TSMC6** | 0/4 | 1/4 | 0/4 | 0/4 | **1/16** |
| **TSMC7** | 0/4 | 1/4 | 0/4 | 0/4 | **1/16** |
| **TSMC12** | 0/4 | 0/4 | 0/4 | 0/4 | **0/16** |
| **TSMC16** | 0/4 | 1/4 | 0/4 | 0/4 | **1/16** |

TSMC6 and TSMC7 are the controlled repeat and produce identical scores. They
count as two campaign columns but only one independent ground truth.

## By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | **0/20** |
| medium | 0/4 | 1/4 | 1/4 | 0/4 | 0/4 | **2/20** |
| large | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | **0/20** |
| xl | 0/4 | 0/4 | 0/4 | 0/4 | 1/4 | **1/20** |

Neither the complex curve nor the device-level curves improve monotonically
with parameter count. Capacity alone does not resolve the observed interface
and support-envelope failures.

## Device and AC suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 34.02 / 112.63 / -7.141 / 494 (11/26) | 40.88 / 120.01 / -9.468 / 709 (11/26) | 39.63 / 119.07 / -9.468 / 709 (9/21) | 39.56 / 102.56 / -9.755 / 881 (15/30) | 38.04 / 102.59 / -9.534 / 811 (13/26) | 59/129 |
| medium | 33.12 / 110.08 / -6.641 / 494 (11/26) | 40.36 / 116.35 / -9.420 / 708 (11/26) | 38.99 / 115.28 / -9.420 / 708 (9/21) | 39.74 / 102.16 / -9.697 / 887 (15/30) | 38.22 / 102.71 / -9.400 / 822 (13/26) | 59/129 |
| large | 33.09 / 109.69 / -6.385 / 494 (11/26) | 40.30 / 116.07 / -9.226 / 706 (11/26) | 38.92 / 114.97 / -9.226 / 706 (9/21) | 40.82 / 104.11 / -9.837 / 1.04e+03 (14/30) | 38.42 / 103.34 / -9.512 / 878 (13/26) | 58/129 |
| xl | 33.22 / 110.00 / -6.366 / 494 (11/26) | 40.50 / 116.61 / -9.235 / 677 (11/26) | 39.19 / 115.78 / -9.235 / 677 (9/21) | 40.38 / 102.76 / -9.873 / 829 (14/30) | 38.35 / 102.86 / -9.559 / 821 (13/26) | 58/129 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.22 / 2.64 / 0.995 / 253 (6/20) | 1.73 / 2.83 / 0.993 / 284 (6/20) | 1.73 / 2.83 / 0.993 / 284 (6/20) | 1.81 / 2.53 / 0.994 / 229 (6/20) | 1.22 / 1.93 / 0.998 / 151 (6/20) | 30/100 |
| medium | 0.43 / 1.09 / 1.000 / 60 (6/20) | 0.43 / 0.89 / 1.000 / 56.5 (6/20) | 0.43 / 0.89 / 1.000 / 56.5 (6/20) | 0.57 / 1.39 / 0.999 / 104 (6/20) | 0.43 / 0.96 / 1.000 / 57.4 (6/20) | 30/100 |
| large | 0.28 / 0.59 / 1.000 / 57.2 (6/20) | 0.09 / 0.32 / 1.000 / 7.08 (6/20) | 0.09 / 0.32 / 1.000 / 7.08 (6/20) | 0.33 / 0.78 / 1.000 / 49.9 (6/20) | 0.22 / 0.33 / 1.000 / 35.5 (6/20) | 30/100 |
| xl | 0.45 / 0.92 / 1.000 / 53.7 (6/20) | 0.17 / 0.33 / 1.000 / 29.9 (6/20) | 0.17 / 0.33 / 1.000 / 29.9 (6/20) | 0.27 / 0.39 / 1.000 / 20.9 (5/20) | 0.38 / 0.58 / 1.000 / 59.8 (6/20) | 29/100 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✓ / ✓ | ✗ mag 11.33 % / ✓ | ✗ mag 11.33 % / ✓ | ✓ / ✓ | ✓ / ✓ | **8/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, valid refined reference and converged NN OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR | **0/5** |
| medium | ERROR | **PASS** 1.14 dB | **PASS** 1.14 dB | ERROR | ERROR | **2/5** |
| large | ERROR | ERROR | ERROR | ERROR | ERROR | **0/5** |
| xl | ERROR | ERROR | ERROR | ERROR | **PASS** 2.48 dB | **1/5** |

Parametric DC passes **59 → 59 → 58 → 58/129** configurations. All 64 PMOS
configurations fail at every tier, while NMOS passes 59/65 at small and medium
and 58/65 at large and XL. This headline is dominated by an interface-contract
mismatch: LEVEL=75's scalar `calculate_current()` exposes signed solver-positive
drain-terminal current, while the legacy PMOS device gate expects a positive
PMOS magnitude. The table is retained as executed, but it is not a pure measure
of PMOS surface-fit error.

Parametric transient passes **30 → 30 → 30 → 29/100**. The numeric aggregates
come from configurations that returned metrics—principally the six VTC rows
per technology—while unsupported transient rows remain `ERROR`. Device CS-amp
AC improves from **8/10** at small to **10/10** at every larger tier, showing
useful local current/charge derivatives at the tested converged operating
points. Opamp open-loop AC follows the complex result at **0 → 2 → 0 → 1/5**.

## Qualification verdict

The tables preserve every scientific failure and every declared parametric
configuration. Numeric aggregates omit only rows that produced no numeric
metric; those rows remain in the denominator as `ERROR`. Tier comparisons are
within this family, and the TSMC6 column is the controlled TSMC7 repeat rather
than an independent ground truth.

DirectNet-Full is **not promotion-ready**. It demonstrates accurate in-domain
VTCs and strong device-level AC above the small tier, but fails the dynamic and
multi-device qualification needed to replace LEVEL=73. Promotion requires at
least a circuit-safe support/iteration policy and a corrected scalar PMOS
current contract, followed by a complete clean re-gate.

## Reproduction

Checkpoints are `tsmc{5,6,7,12,16}_dnf_{small,medium,large,xl}_{nmos,pmos}`.
Each requires `_best.pt`, `_norm.npz`, and the checksum-bound
`_best.pt.complete` marker. This report is the isolated 240-job DirectNet-Full
family pass in `results/v761_directnet_full_clean/`; the combined DirectNet /
BSIM-AR scoreboard is intentionally not regenerated while the independent
BSIM-AR campaign is still running. See the repository README for the complete
generation, training, gate, coverage, and report commands.
