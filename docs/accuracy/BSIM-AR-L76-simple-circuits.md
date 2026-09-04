# BSIM-AR-Full (LEVEL=76) — TSMC5 simple-circuit recovery

> V7.7.0 policy note: LEVEL=76 remains the supported autoregressive
> full-terminal alternative after LEVEL=73/74 retirement. This dated TSMC5
> campaign is still not a five-technology qualification.

## Scope and verdict

This is a TSMC5 development campaign for the full-terminal BSIM-AR family.
Ground truth is NGSPICE with the identical LEVEL=72 BSIM-CMG
OSDI model. It is not a five-technology clean-matrix qualification and its
counts must not be compared with the `/20` clean scoreboards.

The strongest complete circuit result is shared by `small` and `large`: both
pass the strict ring oscillator, SRAM, and switched-capacitor cells while the
Miller opamp remains an explicit error. The selected `medium` teacher-forced
bundle instead passes both device-AC polarities, but loses switched-capacitor
hold droop. Capacity is not monotonic: `xl` also loses the switched-capacitor
and one inverter-VTC configuration.

The implementation gain is an opt-in LEVEL=76 autoregressive training mode.
It trains later outputs against predicted charge prefixes, matching deployed
inference. The default teacher-forced path and checkpoint shapes remain
unchanged. Exact gates, support checks, convergence rules, and denominators
were not altered.

## Selected checkpoint matrix

All four selected bundles are mirrored under the gitignored directory
`results/bsimar_full_simple_20260831/ar_recipe_tiers/selected/checkpoints`.
Every polarity has a checkpoint, normalization sidecar, architecture sidecar,
and checksum-valid completion marker.

| tier | parameters | selected training mode | reason |
|---|---:|---|---|
| small | 666,118 | autoregressive rollout | retains all three non-opamp complex passes |
| medium | 1,935,558 | teacher-forced | 2/2 device AC; rollout was 1/2 with no circuit gain |
| large | 5,012,230 | autoregressive rollout plus targeted fine-tune | retains all three non-opamp complex passes and restores the full inverter matrix |
| xl | 14,802,822 | autoregressive rollout | fully evaluated capacity endpoint |

Selected checkpoint SHA-256 values:

| tier | NMOS | PMOS |
|---|---|---|
| small | `0125fe94ce56715f0b3db49e393eb8765e66c8345c2afe04bf03b1587e9b54d5` | `7cdcd5d612ab5f046aab6fa72fc0446ebe67edb917ecc4dd755b317d5ebde857` |
| medium | `23eccac16ef0d7b4f0ef7f11a9c0fff59ed7be382ea19516029879ee90cee8c9` | `9877063cee3e659cba0056b85408a6ca5acc633ea1192e082c27684020b5e742` |
| large | `e6688eca34003094e8716a00ae4a0167341fd476ac68d4f5a659c77287b51ac2` | `360c3a77d20011d00ecb8a72909547fb597c024a3dd924917b794b269c578490` |
| xl | `9cc4ebd70eeb90a60f74eeceec9ea95a4d33fc91ec6a52947ef58040f1f7f0d8` | `eeeb5e8f0dc18b96ab13f82a50e24f773030ec08da9c5eeb8777575769f5dfd8` |

## Exact CPU gate results

Each complex row reports all four cells explicitly. `ERROR` is not converted
to a failure metric or silently removed: it means the simulator did not
produce a comparable numerical result. Ring and opamp verdicts are strict over
OMP thread counts 1, 2, and 4; SRAM and switched-capacitor are deterministic
OMP=1 cells.

| tier | device AC | DC | inverter | ring | opamp DC / AC | SRAM | switch-cap |
|---|---:|---:|---:|---|---|---|---|
| small | 1/2 | 26/26 | 20/20 | PASS, 2.21% | ERROR / ERROR | PASS, 5.20% | PASS, 1.57% |
| medium | **2/2** | 26/26 | 20/20 | PASS, 4.19% | ERROR / ERROR | PASS, 5.22% | FAIL, 1.80% charge error; droop gate |
| large | 1/2 | 26/26 | 20/20 | PASS, 4.41% | ERROR / ERROR | PASS, 4.98% | PASS, 0.29% |
| xl | 1/2 | 26/26 | 19/20 | PASS, 3.16% | ERROR / ERROR | PASS, 4.92% | FAIL, 0.96% charge error; droop gate |

The report generator's scorable-complex totals are 3/3 for `small`, 2/3 for
`medium`, and 3/3 for `large`; the explicit table above also preserves the
opamp `ERROR` cell. `xl` is 2/3 because switched-capacitor fails. No strict
ring verdict flipped across OMP settings in any tier.

## Parametric error and held-out model error

The first table is circuit-level TSMC5 evidence. Maximum voltage error is from
the inverter VTC/transient suite; numeric aggregates exclude rows that returned
`ERROR`, while the pass denominator retains them.

| tier | DC mean NRMSE / MRE / min R2 / max error | inverter mean NRMSE / MRE / min R2 / max error |
|---|---|---|
| small | 1.816% / 9.488% / 0.93036 / 39.91 uA | 2.571% / 6.594% / 0.97873 / 498.3 mV |
| medium | 2.140% / 9.863% / 0.93478 / 38.04 uA | 1.555% / 5.710% / 0.98196 / 352.8 mV |
| large | **1.427% / 6.111% / 0.95661 / 17.65 uA** | **1.340% / 4.479% / 0.99048 / 193.4 mV** |
| xl | 1.794% / 8.660% / 0.84101 / 60.11 uA | 1.387% / 5.115% / 0.99093 / 181.8 mV (19/20) |

Held-out six-surface physical metrics use the same combo split for every tier.
The table is `average NRMSE / average MRE / average R2` across `i_d`, `i_g`,
`i_b`, `qd`, `qg`, and `qb`.

| tier | NMOS | PMOS |
|---|---|---|
| small | 0.542% / 6.81% / 0.9824 | 0.719% / 8.86% / 0.9719 |
| medium | 0.676% / 7.15% / 0.9742 | 0.843% / 8.68% / 0.9642 |
| large | **0.515% / 4.91% / 0.9858** | **0.662% / 5.48% / 0.9772** |
| xl | 0.577% / 6.53% / 0.9800 | 0.780% / 6.79% / 0.9644 |

Validation loss and circuit rank disagree. In particular, rollout fine-tuning
improved Medium's held-out averages but moved NMOS device AC from PASS to FAIL
and did not recover switched-capacitor droop. The selected matrix therefore
uses exact circuit evidence rather than a uniform training-mode rule.

## Measured gain from the recovery work

The pre-rollout large control passed 26/26 DC, 19/20 inverter configurations,
0/2 device AC, SRAM, and switched-capacitor; its opamp escaped certified PMOS
support. The selected large checkpoint reaches 26/26 DC, 20/20 inverter,
1/2 device AC, strict ring, SRAM, and switched-capacitor. That is one recovered
inverter cell, one recovered device-AC polarity, and a fully measured strict
ring without losing a previously passing circuit.

The remaining Miller opamp is not an accuracy-threshold miss. Depending on
tier and candidate, it either exits certified support or fails physical DC
convergence near the transition. No tested capacity closes that fixed-point
basin, so opamp AC remains downstream-blocked.

## Data, recipe, and provenance

Training used the isolated TSMC5 full-terminal datasets with OSDI-evaluated
failed-NR/opamp-AC and temperature corridors:

| polarity | rows | dataset SHA-256 | trajectory rows |
|---|---:|---|---:|
| NMOS | 6,023,791 | `df029a00ddf59fda6444b9d9ec8871021b9ad754a232768c5838978825666d55` | 49,351 |
| PMOS | 6,004,236 | `c35b64de05b786530fa6c85e44bfac78c1760f55e0f6588102f5bf61973aa7a1` | 32,866 |

The NMOS overlay contract promoted 12,789 hot-corner rows that otherwise fell
entirely outside training. Training used combo splits, EMA 0.993, BF16 AMP,
batch size 7,168, seed 42, and class weight 3 for subthreshold, off-state,
zero/small-VDS, and trajectory rows. Scored inference was CPU-only with one
OpenMP, MKL, and Torch thread. Gate source commit is
`f13137f1fe7fd23948c56ee09f8bc013af6e6fdb`.

Complete collected reports and immutable manifests:

| candidate | manifest SHA-256 | report SHA-256 |
|---|---|---|
| small rollout | `554b71ba2ee9bed0b1ca5cf5e0ebfa2cb0a7d67c5ad18e3363c930ab14ad7a07` | `34e4c875a33bedc39f58ac7a082e47ed2348acbfc194b03461cfdaa11d8855cd` |
| medium teacher | `5087118b7c68c30728f3a0539184f9a79b8d78cbdb086288a05185c6430afd52` | `cb7be7fd8563de72afc660e978234b070c35bbd9aced52e6d7c63b4f0828d289` |
| large targeted rollout | `939e63b491ea1b3480bd05ef00be96b12f99ef72dcc6c5de5cf14dc07156952c` | `3d96c8f155ce9ebeb98873d9534a6b9d4861f7eb95c869922ea057b697ce820e` |
| xl rollout | `04a6204c2c8e66964c89358491f8e38b0ef497e9a198e62c45721c62a1a3958e` | `c1c1377cec545b874edc38839b7ad9b9ac00ae23e20f95ee3c08b24817f95175` |

## Rejected alternatives

- A first rollout-only Large fine-tune changed the opamp failure mode but
  gained no gate. Adding the exact hot/opamp/device-AC overlay was required to
  restore inverter 20/20 and PMOS device AC.
- Mixing checkpoint polarities is not compositional. The Large hybrid failed
  the opamp at the first sweep point; the Medium teacher-NMOS/rollout-PMOS
  hybrid worsened switched-capacitor droop to 1.598 mV, 246% of allowance.
- Support-aware stamping delayed the same support escape. A physical line
  search moved the failure earlier and cost roughly 14 minutes. Both runtime
  experiments were fully reverted.
- Teacher forcing remains useful as a checkpoint candidate: Medium teacher is
  2/2 on device AC. It is not globally superior, because its switched-capacitor
  droop is worse than rollout Medium and still fails.
