# DirectNet-Full (LEVEL=75) — production qualification

This experimental DirectNet learns six independent OSDI terminal surfaces:
`i_d`, `i_g`, `i_b`, `qd`, `qg`, and `qb`. Source current and charge are
closed analytically, and the solver stamps the full current and charge
Jacobians. LEVEL=72 on the identical BSIM-CMG OSDI model is the reference.

Clean means one uniform run per technology, polarity, and tier with
`--output-contract full-terminal --apply-filter off --swa-mode ema --seed 42`.
No reduced-head loss or correction path is enabled.

Evidence pass: **V7.6.2**. Campaign manifest SHA-256 `d9c1a2910334d02ba91c24820de5f843d3346dd4dc06aa4d5077ae28f8ab8913` pins gate commit `5eabb6bf09107c674ad08d180631a5b6b2a5d909`, 240 jobs, and 120 checkpoint artifacts. Raw evidence: `results/v762_directnet_full_clean/`.

Gate definitions, strict OMP scoring, denominator rules, and comparability are
owned by [`methodology.md`](methodology.md).

## Production decision

**Do not promote LEVEL=75.** No tier satisfies the production circuit gates.
The best strict tier is small at 8/20; the production-sized `large` tier is
5/20 and passes 0/248 tracked AnalogGym decks. LEVEL=73 `large` therefore
remains the served DirectNet path.

## Headline — circuit gates by tier

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **8/20** | 5/5 | 0/5 | 3/5 | 0/5 | 0 | tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| medium | **5/20** | 5/5 | 0/5 | 0/5 | 0/5 | 0 | tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| large | **5/20** | 5/5 | 0/5 | 0/5 | 0/5 | 0 | tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| xl | **7/20** | 5/5 | 2/5 | 0/5 | 0/5 | 0 | tsmc5-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc5-switchcap, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |

Twenty-five of the 80 strict circuit cells pass: all 20 ring oscillators,
three small-tier SRAM cells, and the XL TSMC6/TSMC7 opamps. The tier curve is
**8 → 5 → 5 → 7/20**; larger models therefore show no monotonic
qualification gain.

## By testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 1.59% | **PASS** 0.02% | **PASS** 0.02% | **PASS** 1.68% | **PASS** 1.33% |
| medium | **PASS** 0.10% | **PASS** 0.18% | **PASS** 0.18% | **PASS** 0.00% | **PASS** 0.17% |
| large | **PASS** 0.28% | **PASS** 0.84% | **PASS** 0.84% | **PASS** 0.03% | **PASS** 0.20% |
| xl | **PASS** 0.17% | **PASS** 0.70% | **PASS** 0.70% | **PASS** 0.87% | **PASS** 0.72% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR |
| medium | ERROR | ERROR | ERROR | ERROR | ERROR |
| large | ERROR | ERROR | ERROR | ERROR | ERROR |
| xl | ERROR | **PASS** 0.96% | **PASS** 0.96% | ERROR | ERROR 0.29% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 5.58% | **PASS** 1.80% | **PASS** 1.80% | ERROR | ERROR 0.59% |
| medium | ERROR | ERROR 0.83% | ERROR 0.83% | ERROR 0.71% | ERROR 0.14% |
| large | ERROR 7.19% | ERROR | ERROR | ERROR 1.68% | ERROR 0.94% |
| xl | ERROR | ERROR 1.31% | ERROR 1.31% | ERROR | ERROR 0.60% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR |
| medium | ERROR | ERROR | ERROR | ERROR | ERROR |
| large | ERROR | ERROR | ERROR | ERROR | ERROR |
| xl | ERROR | ERROR | ERROR | ERROR | ERROR |

Every ring-oscillator cell passes with zero thread-count flips, while every
switched-capacitor cell fails. Only the small-tier TSMC5/TSMC6/TSMC7 SRAM cells
and the XL TSMC6/TSMC7 opamps pass their strict circuit gates. The other rows
are scientific/runtime qualification failures, not missing jobs, and remain
in the denominator without invented numeric metrics.

## By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 4/4 | 0/4 | 1/4 | 0/4 | **5/16** |
| **TSMC6** | 4/4 | 1/4 | 1/4 | 0/4 | **6/16** |
| **TSMC7** | 4/4 | 1/4 | 1/4 | 0/4 | **6/16** |
| **TSMC12** | 4/4 | 0/4 | 0/4 | 0/4 | **4/16** |
| **TSMC16** | 4/4 | 0/4 | 0/4 | 0/4 | **4/16** |

TSMC6 and TSMC7 are the controlled repeat and produce identical scores. They
count as two campaign columns but only one independent ground truth.

## By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 2/4 | 2/4 | 2/4 | 1/4 | 1/4 | **8/20** |
| medium | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 | **5/20** |
| large | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 | **5/20** |
| xl | 1/4 | 2/4 | 2/4 | 1/4 | 1/4 | **7/20** |

Neither the complex curve nor the device-level curves improve monotonically
with parameter count. Capacity alone does not resolve the observed interface
and support-envelope failures.

## Device and AC suites

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 5.12 / 30.52 / -1.968 / 494 (22/26) | 6.58 / 31.84 / -4.099 / 484 (22/26) | 7.83 / 38.12 / -4.099 / 484 (17/21) | 2.18 / 8.56 / -2.176 / 198 (29/30) | 2.79 / 10.79 / -2.221 / 198 (25/26) | 115/129 |
| medium | 4.51 / 28.73 / -1.981 / 494 (22/26) | 5.95 / 28.65 / -4.200 / 484 (22/26) | 7.17 / 34.71 / -4.200 / 484 (17/21) | 1.94 / 7.90 / -2.194 / 198 (29/30) | 2.29 / 9.17 / -2.147 / 198 (25/26) | 115/129 |
| large | 4.43 / 28.45 / -1.999 / 494 (22/26) | 6.09 / 29.17 / -4.085 / 484 (22/26) | 7.31 / 35.20 / -4.085 / 484 (17/21) | 2.58 / 9.16 / -2.160 / 198 (28/30) | 2.56 / 9.81 / -2.150 / 198 (25/26) | 114/129 |
| xl | 4.53 / 28.45 / -1.990 / 494 (22/26) | 6.16 / 29.10 / -4.257 / 484 (22/26) | 7.54 / 35.60 / -4.257 / 484 (17/21) | 3.00 / 9.86 / -2.254 / 280 (28/30) | 2.77 / 10.15 / -2.151 / 198 (25/26) | 114/129 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; passing/total configs in parentheses)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 1.50 / 5.26 / 0.988 / 281 (19/20) | 1.19 / 4.18 / 0.990 / 253 (19/20) | 1.19 / 4.18 / 0.990 / 253 (19/20) | 1.03 / 4.18 / 0.992 / 219 (19/20) | 1.53 / 4.55 / 0.975 / 470 (19/20) | 95/100 |
| medium | 0.93 / 3.58 / 0.991 / 185 (19/20) | 1.12 / 3.69 / 0.990 / 255 (19/20) | 1.12 / 3.69 / 0.990 / 255 (19/20) | 0.72 / 3.82 / 0.992 / 220 (19/20) | 0.72 / 3.45 / 0.992 / 210 (19/20) | 95/100 |
| large | 0.74 / 3.14 / 0.991 / 185 (19/20) | 0.85 / 3.30 / 0.991 / 249 (19/20) | 0.85 / 3.30 / 0.991 / 249 (19/20) | 0.73 / 4.10 / 0.992 / 220 (17/20) | 0.80 / 3.90 / 0.993 / 210 (17/20) | 91/100 |
| xl | 0.77 / 3.34 / 0.991 / 185 (19/20) | 0.88 / 3.62 / 0.991 / 249 (18/20) | 0.88 / 3.62 / 0.991 / 249 (18/20) | 0.71 / 3.89 / 0.992 / 219 (18/20) | 0.72 / 3.33 / 0.993 / 208 (17/20) | 90/100 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✓ / ✓ | ✗ mag 10.55 % / ✓ | ✗ mag 10.55 % / ✓ | ✓ / ✓ | ✓ / ✓ | **8/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| xl | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, valid refined reference and converged NN OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | ERROR | ERROR | ERROR | ERROR | ERROR | **0/5** |
| medium | ERROR | ERROR | ERROR | ERROR | ERROR | **0/5** |
| large | ERROR | ERROR | ERROR | ERROR | ERROR | **0/5** |
| xl | ERROR | FAIL 5.51 dB | FAIL 5.51 dB | ERROR | **PASS** 1.83 dB | **1/5** |

Parametric DC passes **115 → 115 → 114 → 114/129** configurations after the
PMOS scalar-current boundary correction. Parametric transient passes
**95 → 95 → 91 → 90/100**. Unsupported and failed configurations remain in
both denominators, while numeric aggregates use only rows that returned
commensurate metrics. Device CS-amp AC improves from **8/10** at small to
**10/10** at every larger tier. Opamp open-loop AC is
**0 → 0 → 0 → 1/5**.

## AnalogGym complex-circuit accuracy

The production-sized `large` checkpoint pair for each technology was evaluated over the tracked 255-row basket in `examples/complex_circuits/`. NGSPICE 45.2 used LEVEL=72 as ground truth; DirectNet-Full used LEVEL=75 on CPU with one OpenMP, MKL, and Torch thread. Seven whole-deck invalid examples remain quarantined, leaving **248 scored decks. DirectNet-Full passes 0/248.**

| tech | AC | dc_source | dc_temp | transient | total | Py failures | NG failures |
|---|:--:|:--:|:--:|:--:|:--:|--:|--:|
| TSMC5 | 0/27 | 0/6 | 0/9 | 0/7 | **0/49** | 36 | 0 |
| TSMC6 | 0/28 | 0/6 | 0/9 | 0/7 | **0/50** | 36 | 0 |
| TSMC7 | 0/28 | 0/6 | 0/9 | 0/7 | **0/50** | 36 | 0 |
| TSMC12 | 0/28 | 0/6 | 0/9 | 0/7 | **0/50** | 32 | 0 |
| TSMC16 | 0/28 | 0/6 | 0/9 | 0/6 | **0/49** | 34 | 3 |
| **all** | **0/139** | **0/30** | **0/45** | **0/34** | **0/248** | **174** | **3** |

| tech | comparable metric cells agreeing | missing Py values | quarantined metric cells |
|---|---:|---:|---:|
| TSMC5 | 8/63 | 239 | 20 |
| TSMC6 | 3/58 | 245 | 19 |
| TSMC7 | 3/58 | 245 | 19 |
| TSMC12 | 16/79 | 224 | 19 |
| TSMC16 | 11/68 | 202 | 30 |
| **all** | **41/326** | **1155** | **107** |

Voltage-state metrics use only rows that produced comparable operating-point or DC-sweep samples; incomplete rows remain failures in the deck denominator.

| technology | rows with state data | samples | MRE | R² | NRMSE | max abs error |
|---|---:|---:|---:|---:|---:|---:|
| TSMC5 | 9 | 4269 | 40.7% | -0.706134 | 43.9% | 0.9831 V |
| TSMC6 | 9 | 4451 | 41.37% | -0.353539 | 38.56% | 0.989809 V |
| TSMC7 | 9 | 4451 | 41.37% | -0.353539 | 38.56% | 0.989809 V |
| TSMC12 | 14 | 6002 | 37.34% | -0.234386 | 37.64% | 1.1389 V |
| TSMC16 | 11 | 6127 | 13.64% | 0.928219 | 8.729% | 0.58004 V |

Aggregate SHA-256 `0228f7db941b569efdf6e8c5a7ee0149629a93adc3dcddbc5da9af04ebcfc397` pins simulation commit `5eabb6bf09107c674ad08d180631a5b6b2a5d909`, the five checkpoint pairs, normalization and completion hashes, modelcards, the LEVEL=72 OSDI binary, NGSPICE, and per-deck fidelity policy. Detailed circuit evidence and quarantine ownership are in [`examples/complex_circuits/RESULTS_TSMC.md`](../../examples/complex_circuits/RESULTS_TSMC.md). The upstream AnalogGym source tree is absent, so this is a complete tracked-deck rerun, not a refreshed source-topology audit.

## Qualification verdict

The tables preserve every scientific failure and every declared parametric
configuration. Numeric aggregates omit only rows that produced no numeric
metric; those rows remain in the denominator as `ERROR`. Tier comparisons are
within this family, and the TSMC6 column is the controlled TSMC7 repeat rather
than an independent ground truth.

DirectNet-Full is **not promotion-ready**. The regenerated models repair the
PMOS scalar boundary, pass every ring-oscillator gate, and retain strong
device-level AC above the small tier, but still fail the SRAM,
switched-capacitor, and most opamp qualification needed to replace LEVEL=73.
Promotion requires closing the remaining circuit convergence, accuracy, and
certified-support failures, then passing future complete clean and AnalogGym
gates.

## Reproduction

Checkpoints are `tsmc{5,6,7,12,16}_dnf_{small,medium,large,xl}_{nmos,pmos}`.
Each requires `_best.pt`, `_norm.npz`, and the checksum-bound
`_best.pt.complete` marker. This report is the isolated 240-job DirectNet-Full
family pass in `results/v762_directnet_full_clean/`. AnalogGym evidence is in
`results/v762_directnet_full_analoggym_tsmc{5,6,7,12,16}/`, with the aggregate
at `results/v762_directnet_full_analoggym_aggregate.json`. The TSMC12
`front_end_25_6T_schematic/tb_dc` row exceeded the default 3,600-second wall
limit and completed from fresh scratch under a 7,200-second allowance; its
model pins, thread count, and fidelity policy were unchanged. The combined
DirectNet / BSIM-AR scoreboard is intentionally unchanged because V7.6.2
reran only DirectNet-Full. See the repository README for the complete
generation, training, gate, coverage, and report commands.
