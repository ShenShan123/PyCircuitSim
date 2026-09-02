# DirectNet LEVEL=73 retrain and AnalogGym accuracy

Measured 2026-08-19. Ground truth is NGSPICE 45.2 using the identical
BSIM-CMG OSDI LEVEL=72 model. The scored DirectNet inference path used CPU
only with one OpenMP, MKL, and Torch thread.

## Verdict

The retrained per-technology `large` DirectNet checkpoints remain accurate on
single devices and digital primitives, but are **not qualified for the
AnalogGym complex-circuit basket**. No scored AnalogGym deck fully agrees with
the reference: **0/248**, with seven invalid deck cells quarantined from the
255-row corpus. Across the voltage states that were comparable, aggregate
MRE is **35.41%**, R² is **-42.66**, NRMSE is **73.22%**, and maximum voltage
error is **12.696 V**.

This is a model result, not an L72 simulator regression. A same-topology L72
Fan amplifier smoke test remained 8/8 with a 0.77 µV maximum operating-point
delta. DirectNet's simple gates also remain strong, but the Miller opamp,
nonlinear AC operating points, and AnalogGym rows expose failure modes not
predicted by held-out device loss.

## Training run

Ten independent checkpoints were trained: NMOS and PMOS for TSMC5, TSMC6,
TSMC7, TSMC12, and TSMC16. The recipe was DirectNet `large` (907,565
parameters), filter off, per-step EMA 0.999, seed 42, 800 epochs, patience 150.
The jobs ran across GPUs 0–4, two streams per GPU.

```bash
GPUS="0 1 2 3 4" NSTREAMS=10 \
TECHS="tsmc5 tsmc6 tsmc7 tsmc12 tsmc16" \
SIZES="large" DEVS="nmos pmos" \
bash scripts/benchmark_train_sml.sh --force
```

| technology | NMOS best epoch / val | PMOS best epoch / val |
|---|---:|---:|
| TSMC5 | 771 / 0.000220 | 757 / 0.000217 |
| TSMC6 | 751 / 0.000246 | 729 / 0.000254 |
| TSMC7 | 751 / 0.000246 | 729 / 0.000254 |
| TSMC12 | 749 / 0.000230 | 727 / 0.000233 |
| TSMC16 | 736 / 0.000235 | 773 / 0.000215 |

Training started from commit
`9b88098f142494e95f84d602f50c87c2d7e4a89f`; the training implementation
and datasets did not change before evaluation.

## Checkpoint provenance

Every scored JSON row records the full checkpoint and normalization hashes,
the explicit resolver stem, completion marker, and evaluation commit. It also
records the per-design modelcard, OSDI, and NGSPICE executable hashes plus the
effective stride/refinement policy. The shared OSDI SHA-256 is
`f089f17d5d5b1178c48932ff699960dab3ab509c33b34c798421eccbbf14a78b`;
the NGSPICE 45.2 executable SHA-256 is
`3b931f4ecb53a9e2e650087be470b3decc49654ce2237a204c6cb6b4ab45d764`.

| checkpoint stem | checkpoint SHA-256 | normalization SHA-256 |
|---|---|---|
| `tsmc5_dn_large_nmos` | `669d1bdb939e2277768ee7dce316ec2e6be09977aaed9c5c5f4dc8d94ca59393` | `831d0c135c4847096df63d137dedab7652fe1ac7bce6de194ba1e95da940dc58` |
| `tsmc5_dn_large_pmos` | `f1e29e18572c5528492b3f4cc08296a0597fea99325fd8ac900877274a696cfe` | `2446276148c4aa61fb87826d3c781fc53022e546f8154d1e27125ece51ad9a71` |
| `tsmc6_dn_large_nmos` | `f1c538b46e264fbe00ceb23b1cea80e7be085cb6b16d056717fa30f2dfb29ec7` | `147f74ab3ab345b3bf2955df304b7138c422e93d44f8a08f38a7ac62f8116d50` |
| `tsmc6_dn_large_pmos` | `99bfe9ba243a2a3fefd7e8838323366cdc13533377aa398cfe678427c69b563a` | `c8c9be7722d22784f7bed9a16ec00e2957a3501048305d496b673fdbaeabd160` |
| `tsmc7_dn_large_nmos` | `10d9941ed6cf924ca78451732888570c8a2302d37eb6c1c828bbb0081f870d3d` | `147f74ab3ab345b3bf2955df304b7138c422e93d44f8a08f38a7ac62f8116d50` |
| `tsmc7_dn_large_pmos` | `d09b0dca7d8294f61041cd05e3e5b52ab190e7de4864b2679f48b3f59ad995c3` | `c8c9be7722d22784f7bed9a16ec00e2957a3501048305d496b673fdbaeabd160` |
| `tsmc12_dn_large_nmos` | `b6a093ae5fecfbabd742cad1fd080187c9396e690ebe06f87ff77bb7a8207f65` | `5894e8da0f6b6b148ca28182a8e7c44832f2489e4e98ec4f00895ae35d63b793` |
| `tsmc12_dn_large_pmos` | `6c58cc0cc4002dca20ce444a6cfe8bda37a0445db68be41a64dd848a05cedf5c` | `ad69aee9e02ae8bef47385ac5b85ce84a6f34284d40d4e6173878535765c13f5` |
| `tsmc16_dn_large_nmos` | `c691004a286e9e8780547177ebc3eed8f9c97b33d43f4fd1a93f429d2304db99` | `322ea73c01e2d1cab5a465dd36aad9052970ceeb02bfa6212a493a1809908bad` |
| `tsmc16_dn_large_pmos` | `4893235e92eb543dc4b6cfa8c30b0781f236ebdaebf2398d5fd968fb8e6ef563` | `38222063b12a52260c3b55ee4d11e953269b15ccd921345685191f02d6532b72` |

## Pre-AnalogGym gates

| gate | result | interpretation |
|---|---:|---|
| lifted-source NMOS DC | 15/15 | source-relative inference contract holds |
| single-device DC/transient | 20/20 | NMOS DC NRMSE 0.20% average; PMOS 0.03% |
| inverter VTC/transient | 10/10 | VTC NRMSE 1.10% average |
| parametric device DC | 68/69 | TSMC12 PMOS NFIN=10 failed: 13.25% NRMSE, 39.29% MRE |
| parametric inverter transient | 80/80 | all geometry/supply/load corners passed |
| device AC, strict | 0/10 | every DirectNet DC bias was nonconverged; close frequency responses are diagnostic only |
| ring oscillator period | 5/5 | period error 1.95–3.45%; phase-drift waveform NRMSE 53.85–72.39% |
| Miller opamp | 0/5 | DirectNet gain was zero on every technology; NRMSE about 70% |
| SRAM SNM | 5/5 | all technology gates passed |
| switched-cap cell | 5/5 | charge error 1.42–4.13% of VDD |

The initial AC script classified all ten numerically close responses as passes
despite nonconverged DC states. That result is retracted. The strict gate and
production AC path now require a converged fixed point.

## AnalogGym voltage-state accuracy

The complete campaign used commit
`d52a1d36f24d926a672c85ba7cc0cd80a79dfca3`, 51 rows per technology, and the
V7.5.9 curated basket. MRE uses the symmetric denominator. NRMSE uses each
technology campaign's NGSPICE voltage range; the aggregate row uses the
five-technology range.

| technology | rows with state data | samples | MRE | R² | NRMSE | max abs error |
|---|---:|---:|---:|---:|---:|---:|
| TSMC5 | 49 | 16,729 | 32.57% | -50.056 | 84.10% | 12.326 V |
| TSMC6 | 48 | 16,917 | 34.12% | -42.987 | 76.98% | 12.577 V |
| TSMC7 | 48 | 16,917 | 34.12% | -42.987 | 76.98% | 12.577 V |
| TSMC12 | 40 | 15,599 | 37.47% | -38.987 | 76.02% | 12.681 V |
| TSMC16 | 43 | 14,137 | 39.57% | -43.443 | 80.13% | 12.696 V |
| **all** | **228** | **80,299** | **35.41%** | **-42.655** | **73.22%** | **12.696 V** |

The earlier 54,045-sample aggregate is retracted: multi-plan line sweeps and
temperature-recovery rows contributed only their first/cold segment. The
corrected pass aggregates every saved comparable segment from one campaign.

Across metric cells, 110/1,016 comparable values agree at the existing gate,
465 NGSPICE values are missing on the PyCircuitSim side, and 107 cells are
quarantined as invalid comparisons. The authoritative family and circuit
breakdown is in
[`RESULTS_TSMC.md`](../../examples/complex_circuits/RESULTS_TSMC.md).

## Corpus limitation

The untracked upstream AnalogGym source design tree is absent on this machine,
so source-tree preflight correctly prevented regeneration. This campaign used
the existing tracked V7.5.9 generated decks and modelcards materialized from
the local ignored TSMC PDK cards. It is valid evidence for the compact-model
comparison on those decks, but it is not a refreshed source-topology audit.

Raw evidence remains on disk in
`results/analoggym-directnet-large-d52a1d3-tsmc{5,6,7,12,16}/`; training and
gate logs remain in `results/benchmark_sml/train_logs/` and
`results/directnet-retrain-large/gates/`.
