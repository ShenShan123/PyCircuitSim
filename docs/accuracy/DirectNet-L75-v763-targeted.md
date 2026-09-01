# DirectNet-Full (LEVEL=75) — V7.6.3 full-scale targeted evaluation

Date: 2026-08-29

This report supersedes the earlier `large`-only V7.6.3 result. It evaluates
all four DirectNet-Full capacities (`small`, `medium`, `large`, and `xl`) with
the accepted V7.6.3 pass-device data corridor and LEVEL=75-only 0.1 V Newton
trust region. NGSPICE LEVEL=72 on the identical BSIM-CMG OSDI model remains
ground truth.

The targeted gate campaign is complete for the 240 declared device, AC, and
strict circuit jobs: 12 jobs per tier/technology, with ring oscillator and
Miller DC swept at OMP 1/2/4. The remaining AnalogGym evaluation is also
complete: four tiers by five technologies by 51 tracked deck cells gives
1,020 physical rows, 992 scored rows, and 28 established quarantines. This is
still a targeted warm-start campaign, not a replacement for the V7.6.2
clean-from-scratch matrix. The upstream AnalogGym source tree is absent, so
the follow-up evaluates the tracked V7.5.9 corpus rather than a regenerated
source-topology audit.

## Outcome

| tier | strict complex /20 | parametric DC /129 | inverter /100 | device AC /10 | op-amp AC /5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `small` | 8 → **16** | 115 → **115** | 95 → **99** | 8 → **8** | 0 → **0** |
| `medium` | 5 → **20** | 115 → **115** | 95 → **99** | 10 → **10** | 0 → **3** |
| `large` | 5 → **20** | 114 → **114** | 91 → **100** | 10 → **10** | 0 → **3** |
| `xl` | 7 → **19** | 114 → **114** | 90 → **97** | 10 → **10** | 1 → **1** |

Arrows compare the V7.6.2 clean matrix with this V7.6.3 targeted pass. The
accepted alteration is therefore a strong circuit-convergence recovery, not a
general device-accuracy recovery. `medium` and `large` close all 20 strict
circuit cells. They tie on the unweighted declared gate total: `medium` gains
one DC configuration while `large` gains one inverter configuration.

## Strict circuit gates

Explicit `ERROR` cells remain in the `/20` denominator. The `flips` column
counts only PASS/FAIL thread flips; the XL TSMC5 cell instead changes from PASS
to an explicit support `ERROR`.

| tier | strict /20 | ring | Miller DC | SRAM SNM | switch-cap | flips | open cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `small` | **16/20** | 5/5 | 4/5 | 5/5 | 2/5 | 0 | TSMC5 Miller; TSMC6/7/16 switch-cap |
| `medium` | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `large` | **20/20** | 5/5 | 5/5 | 5/5 | 5/5 | 0 | — |
| `xl` | **19/20** | 5/5 | 4/5 | 5/5 | 5/5 | 0 | TSMC5 Miller |

The five remaining tier/cell failures are specific:

| tier / cell | outcome | observed cause |
| --- | --- | --- |
| small / TSMC5 Miller | `ERROR` at OMP 1/2/4 | DC sweep point 76 at 0.36 V did not converge |
| small / TSMC6 and TSMC7 switch-cap | `FAIL` | 1.852 mV hold-droop error versus 0.750 mV allowance; charge error is only 0.25% VDD |
| small / TSMC16 switch-cap | `FAIL` | 3.858 mV hold-droop error versus 0.800 mV allowance; charge error is only 0.82% VDD |
| XL / TSMC5 Miller | PASS at OMP 1/2, `ERROR` at OMP 4 | PMOS input reaches 0.797305 V, outside certified `[-1.30, 0.78]` support |

Every SRAM cell passes. All 20 ring-period cells pass, but period is not a
waveform-shape gate:

| tier | maximum period error | maximum waveform NRMSE | technology |
| --- | ---: | ---: | --- |
| `small` | 1.77% | 49.33% | TSMC12 |
| `medium` | 0.47% | 32.30% | TSMC6/7 |
| `large` | 1.06% | 51.96% | TSMC6/7 |
| `xl` | 0.83% | 29.94% | TSMC12 |

Thus 20/20 strict circuit closure does not mean waveform fidelity is solved.

## Device and inverter error

Cells below are `mean NRMSE % / mean MRE % / minimum R² / maximum absolute
error`; parentheses show passing configurations when the technology is not
perfect. Numeric aggregates exclude only configurations that returned no
commensurate metric; those configurations remain in the pass denominator.

### Parametric DC (`max error` in µA)

| tier | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
| --- | --- | --- | --- | --- | --- | ---: |
| `small` | 5.09 / 30.38 / -1.965 / 494 (22/26) | 6.47 / 31.40 / -4.096 / 484 (22/26) | 7.68 / 37.61 / -4.096 / 484 (17/21) | 2.24 / 8.64 / -2.206 / 198 (29/30) | 2.64 / 10.48 / -2.219 / 198 (25/26) | **115/129** |
| `medium` | 4.49 / 28.56 / -1.980 / 494 (22/26) | 6.03 / 28.88 / -4.158 / 484 (22/26) | 7.20 / 34.72 / -4.158 / 484 (17/21) | 1.93 / 7.95 / -2.196 / 198 (29/30) | 2.32 / 9.29 / -2.167 / 198 (25/26) | **115/129** |
| `large` | 4.39 / 28.26 / -1.992 / 494 (22/26) | 6.22 / 29.45 / -4.040 / 484 (22/26) | 7.34 / 35.15 / -4.040 / 484 (17/21) | 2.59 / 9.22 / -2.173 / 198 (28/30) | 2.66 / 10.09 / -2.153 / 198 (25/26) | **114/129** |
| `xl` | 4.50 / 28.44 / -1.991 / 494 (22/26) | 6.22 / 29.35 / -4.264 / 484 (22/26) | 7.51 / 35.49 / -4.264 / 484 (17/21) | 2.98 / 9.89 / -2.245 / 278 (28/30) | 2.80 / 10.26 / -2.158 / 198 (25/26) | **114/129** |

Fourteen DC failures recur at every size. They are the +125 °C and joint
geometry/temperature configurations for TSMC5/6/7, plus the TSMC12/16 PMOS
joint configurations. `large` additionally loses TSMC12 NMOS NFIN=10; `xl`
loses TSMC12 PMOS NFIN=10. Increasing capacity does not address the dominant
temperature/joint error.

### Parametric inverter (`max error` in mV)

| tier | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
| --- | --- | --- | --- | --- | --- | ---: |
| `small` | 1.29 / 4.56 / 0.992 / 179 (19/20) | 1.22 / 4.21 / 0.991 / 250 | 1.22 / 4.21 / 0.991 / 250 | 1.02 / 4.00 / 0.992 / 219 | 1.37 / 4.21 / 0.981 / 379 | **99/100** |
| `medium` | 0.88 / 3.47 / 0.991 / 185 (19/20) | 0.77 / 3.15 / 0.991 / 249 | 0.77 / 3.15 / 0.991 / 249 | 0.77 / 3.79 / 0.992 / 220 | 0.74 / 3.43 / 0.992 / 211 | **99/100** |
| `large` | 0.80 / 3.14 / 0.991 / 185 | 0.76 / 3.18 / 0.991 / 248 | 0.76 / 3.18 / 0.991 / 248 | 0.80 / 3.79 / 0.988 / 316 | 0.90 / 3.79 / 0.992 / 215 | **100/100** |
| `xl` | 0.82 / 3.45 / 0.991 / 185 (19/20) | 0.81 / 3.33 / 0.991 / 248 | 0.81 / 3.33 / 0.991 / 248 | 0.81 / 4.09 / 0.992 / 220 (18/20) | 0.77 / 3.50 / 0.993 / 210 | **97/100** |

The remaining errors are non-converged VTC points: TSMC5 low-VDD for small
and medium, TSMC5 PN=0.5 for XL, and TSMC12 PN=1.5/2.0 for XL. `large` is the
only tier with complete inverter closure.

## AC gates

| tier | device CS amplifier | op-amp open-loop |
| --- | ---: | ---: |
| `small` | **8/10** (TSMC6/7 NMOS mag NRMSE 11.62%) | **0/5** |
| `medium` | **10/10** | **3/5** |
| `large` | **10/10** | **3/5** |
| `xl` | **10/10** | **1/5** |

| tier | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
| --- | --- | --- | --- | --- | --- |
| `small` | `ERROR` | `ERROR` | `ERROR` | `ERROR` | FAIL 5.65 dB |
| `medium` | `ERROR` | **PASS** 1.25 dB | **PASS** 1.25 dB | `ERROR` | **PASS** 0.49 dB |
| `large` | **PASS** 2.97 dB | **PASS** 2.74 dB | **PASS** 2.74 dB | FAIL 6.15 dB | FAIL 6.84 dB |
| `xl` | **PASS** 1.24 dB | FAIL 4.20 dB | FAIL 4.20 dB | FAIL 6.55 dB | FAIL 8.07 dB |

GBW ratios and phase errors pass for every numeric op-amp AC result except the
small TSMC16 phase error (15.7°). The dominant numeric failure is DC gain;
small/medium `ERROR` cells fail earlier during bias-sweep convergence or
certified-support checks.

## AnalogGym S/M/L/XL evaluation

Every campaign used DirectNet-Full LEVEL=75 on CPU with one OpenMP, MKL, and
Torch thread. NGSPICE 45.2 used LEVEL=72 on the identical BSIM-CMG OSDI model.
All 20 campaign roots contain their complete 51-row physical inventory; there
were no timeouts, infrastructure tracebacks, or omitted failure rows. Seven
whole-deck invalid examples per tier remain quarantined, leaving `/248`.

| tier | deck pass | Py failures | metric cells agreeing | missing Py values |
| --- | ---: | ---: | ---: | ---: |
| `small` | **0/248** | 177 | 21/107 | 1,374 |
| `medium` | **0/248** | 172 | 31/130 | 1,351 |
| `large` | **0/248** | 180 | 23/101 | 1,380 |
| `xl` | **0/248** | 180 | 17/89 | 1,392 |

No analysis family closes at any size:

| tier | AC /139 | DC source /30 | DC temperature /45 | transient /34 |
| --- | ---: | ---: | ---: | ---: |
| `small` | 0/139 | 0/30 | 0/45 | 0/34 |
| `medium` | 0/139 | 0/30 | 0/45 | 0/34 |
| `large` | 0/139 | 0/30 | 0/45 | 0/34 |
| `xl` | 0/139 | 0/30 | 0/45 | 0/34 |

### Support diagnosis

The failure is dominated by a static training-domain mismatch before circuit
accuracy can be measured. Counts below are affected deck rows; a row may carry
both a geometry and voltage error.

| tier | length input 5 | NFIN input 4 | temperature input 6 | voltage inputs 0–3 |
| --- | ---: | ---: | ---: | ---: |
| `small` | 184 | 18 | 3 | 45 |
| `medium` | 184 | 18 | 2 | 40 |
| `large` | 184 | 18 | 3 | 48 |
| `xl` | 184 | 18 | 1 | 49 |

Exactly 140 PyCircuitSim failures per tier are geometry-support aborts. The
remaining execution failures are voltage-support aborts (37, 32, 38, and 39
from small through XL), plus two large nonconvergences and one XL NaN/Inf.
DC sweeps can persist partial rows, so support errors also account for most of
the missing values in rows whose top-level process completed.

The preserved data artifacts identify their generator release as V7.6.0. Their
certified length range ends at 107.754 nm for TSMC5 and 190.493 nm for the
other technologies, while tracked circuits request the terminal 135 nm and
240 nm PDK edges. The normalizers also start at `log2(NFIN)=1`, so the existing
project-wide NFIN=1 exclusion appears as 18 explicit unsupported rows per tier.
All four sizes share the same normalization artifacts, explaining why capacity
cannot change these counts.

### Comparable voltage-state error

These aggregates use only rows that returned commensurate operating-point or
DC-sweep states. The changing, very small row counts are shown explicitly;
they prevent treating a lower error as a whole-tier improvement.

| tier | technology | rows | samples | MRE | R² | NRMSE | max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | TSMC5 | 2 | 1,237 | 32.61% | 0.143 | 30.66% | 0.792 V |
| `small` | TSMC6 | 2 | 1,132 | 23.01% | 0.759 | 16.72% | 0.636 V |
| `small` | TSMC7 | 2 | 1,132 | 23.01% | 0.759 | 16.72% | 0.636 V |
| `small` | TSMC12 | 3 | 2,489 | 22.43% | 0.252 | 27.94% | 0.919 V |
| `small` | TSMC16 | 7 | 1,032 | 28.91% | 0.599 | 23.54% | 0.662 V |
| `medium` | TSMC5 | 2 | 994 | 25.08% | 0.920 | 9.83% | 0.219 V |
| `medium` | TSMC6 | 2 | 1,189 | 15.05% | 0.713 | 18.75% | 0.961 V |
| `medium` | TSMC7 | 2 | 1,189 | 15.05% | 0.713 | 18.75% | 0.961 V |
| `medium` | TSMC12 | 8 | 2,248 | 10.99% | 0.888 | 10.72% | 0.623 V |
| `medium` | TSMC16 | 5 | 686 | 13.89% | 0.967 | 7.37% | 0.359 V |
| `large` | TSMC5 | 2 | 606 | 39.96% | 0.129 | 36.67% | 0.757 V |
| `large` | TSMC6 | 2 | 1,303 | 14.66% | 0.883 | 11.80% | 0.347 V |
| `large` | TSMC7 | 2 | 1,303 | 14.66% | 0.883 | 11.80% | 0.347 V |
| `large` | TSMC12 | 3 | 2,204 | 10.32% | 0.885 | 10.80% | 0.606 V |
| `large` | TSMC16 | 2 | 860 | 36.70% | -0.789 | 50.99% | 0.802 V |
| `xl` | TSMC5 | 2 | 886 | 23.39% | 0.671 | 20.69% | 0.427 V |
| `xl` | TSMC6 | 2 | 883 | 17.82% | 0.869 | 12.56% | 0.364 V |
| `xl` | TSMC7 | 2 | 883 | 17.82% | 0.869 | 12.56% | 0.364 V |
| `xl` | TSMC12 | 3 | 1,155 | 6.24% | 0.977 | 5.57% | 0.490 V |
| `xl` | TSMC16 | 3 | 916 | 15.67% | 0.934 | 9.83% | 0.358 V |

`medium` has the best execution and metric coverage, but still passes no deck.
Compared with the V7.6.2 large campaign, V7.6.3 large moves Py failures from
174 to 180, comparable metrics from 41/326 to 23/101, and missing Py values
from 1,155 to 1,380. The V7.6.3 change is therefore a successful targeted
simple-circuit recovery, not an AnalogGym recovery.

## Analysis

1. **The previous alteration worked at its intended boundary.** The added
   pass-device corridor and 0.1 V trust region recover SRAM, switch-cap, ring,
   and Miller fixed points across medium/large, and raise every tier's inverter
   score. It does not cover the AnalogGym geometry vocabulary.
2. **Geometry support precedes model tuning.** Terminal channel-length edges
   affect 184 rows per tier and NFIN=1 affects another 18. Increasing capacity,
   adding Jacobian loss, or changing Newton globalization cannot make an
   evaluator run on a statically rejected device geometry.
3. **Pointwise validation is not selecting simulator fidelity.** Fine-tune
   validation loss generally falls as capacity increases, while strict
   circuits peak at medium/large and op-amp AC follows `0 → 3 → 3 → 1/5`.
   Current/charge values can be close while their autograd Jacobians move a
   high-gain operating point or AC gain.
4. **The dominant device gap is structured, not capacity-limited.** The same
   14 hot/joint DC configurations fail at every size with up to 50.04% NRMSE,
   250.74% MRE, and R² down to -4.264.
5. **Headline circuit gates hide residual error.** Ring period passes despite
   waveform NRMSE up to 51.96%; small switch-cap charge passes while hold
   droop fails. Next-round selection must include waveform, droop, and
   truth-relative derivative metrics rather than optimize pass counts alone.
6. **Physical support and trial-iterate support remain distinct.** After the
   static geometry gap closes, accepted LEVEL=72 physical states must determine
   data expansion, while limited Newton trial overshoots must test a separate
   globalization arm. Normalizer-only bound widening already produced runaway
   behavior and remains rejected.

## Review → proposal → evaluation → analysis

Use `medium` as the fast development baseline and retain `large` as the
production-sized confirmation control. Do not stack variants before their
individual effect is measured.

| review | proposal | evaluation | analysis / decision |
| --- | --- | --- | --- |
| Circuit scores improve with capacity only through `medium`/`large`. | Use a larger network as the recovery. | AnalogGym is 0/248 at S/M/L/XL; coverage peaks at medium and degrades afterward. | **Rejected.** Capacity does not change shared support artifacts and is non-monotonic numerically. |
| V7.6.3 closes medium/large strict circuits. | Reuse the pass corridor and 0.1 V trust region as the corpus fix. | Large AnalogGym has more Py failures and less metric coverage than V7.6.2. | **Retain locally, reject as a corpus-wide remedy.** |
| 184 rows per tier reject terminal lengths absent from the saved data. | Regenerate from the current terminal-upper-edge parser without widening bounds by hand. | The current parser/corridor contracts pass seven focused tests; saved datasets still report V7.6.0 and shorter maxima. | **Advance as the next isolated arm.** The tests establish enumeration feasibility, not model accuracy. |
| 18 rows per tier use NFIN=1, excluded from canonical training. | Clamp, extrapolate, or silently remove those rows. | The support guard rejects them consistently at every capacity. | **Rejected.** Either certify NFIN=1 separately against OSDI or retain explicit unsupported outcomes and a closed promotion gate. |

### Proposed next arms

| priority / variant | single hypothesis and change | focused evaluation | acceptance condition |
| --- | --- | --- | --- |
| P0 `G-terminal-L` | Missing terminal length knots cause the 184-row invariant gap. Regenerate the ten datasets with the current parser, retrain `medium`, and add a corpus-to-checkpoint geometry preflight. | Static S/M/L/XL corpus inventory, then all medium AnalogGym cells. | Zero input-5 support errors, V7.6.3 source-pinned dataset markers, and no regression from 20/20 strict, 10/10 device AC, and 99/100 inverter. |
| P1 `N1-certified` | The 18 NFIN=1 rows need a separate support decision. Audit each exact `(tech,VT,L,NFIN=1)` bin through OSDI/NGSPICE; train only if a reviewed stable artifact contract exists. | Exact geometry canaries before any circuit rerun. | A row can pass only with generated, finite, OSDI-backed support. Otherwise it remains explicit `UNSUPPORTED`, not a fabricated numeric pass. |
| P2 `V-physical` / `V-globalize` | Remaining voltage errors mix accepted physical states with nonlinear trial overshoot. Add trace labels first; expand data only for accepted states and test backtracking/pseudo-transient fallback only for trial states. | Geometry-cleared medium support canaries, then all medium AnalogGym cells. | No accepted-state support error, fewer or equal nonlinear failures, and an independently reported wall-time delta. |
| P3 `T-hot-joint` | The recurring 14 device-DC misses are sampling/selection error. Add explicit +125 °C × legal `(L,NFIN,VT)` strata and group-held-out selection. | All 129 DC configurations, then the 240-job medium campaign. | Recover at least 7/14 misses, lower each affected technology's NRMSE/MRE, and retain the selected circuit/AC/inverter baseline. |
| P4 `J-current` | After execution closes, op-amp gain is limited by current-Jacobian error. Add truth-relative full-terminal Sobolev loss/selection on forward and circuit corridors; exclude known-invalid reverse-`Vds` `gds` labels. | OSDI Jacobian holdout, device AC, op-amp AC, then AnalogGym AC. | At least 4/5 op-amp AC with zero `ERROR`, no device-AC loss, and increased AnalogGym comparable-metric agreement. |
| P5 `J-charge` | Residual droop/waveform/transient error is charge-Jacobian error after DC support closes. Add a separate OSDI transcapacitance loss/selector with analytic charge closure. | Switch-cap droop, phase-aligned ring waveform, then AnalogGym transient. | 5/5 switch-cap, maximum ring waveform NRMSE ≤20%, and no loss of ring-period or inverter closure. |

After each arm, compare it with the last accepted winner, reject it on any gate
regression, and preserve the raw ledger. Only then compose independently
successful arms. Confirm the winner with whole-campaign seeds (for example
7/42/123), never per-cell seed cherry-picking, and rerun both medium and large.
A final promotion still requires a complete clean campaign and refreshed
source-tree AnalogGym evidence.

## Production decision

**Do not promote LEVEL=75.** This targeted matrix closes the medium/large
simple-circuit gates, but device DC, op-amp AC, waveform fidelity, and support
remain open. The V7.6.3 tracked-deck AnalogGym result is 0/248 at every tier;
V7.6.2 remains the latest clean-from-scratch qualification baseline.

## Provenance and verification

- Evaluation source: HEAD `84f965bce12fd94a2e37e55d0ed2cefd5bab6ae9`
  plus tracked-diff SHA-256
  `4307417d7df3d0c283cabbe05aeba87390ac19f02c1b2b6ea917c125e891d63d`.
- V7.6.2 controls: manifest SHA-256
  `d9c1a2910334d02ba91c24820de5f843d3346dd4dc06aa4d5077ae28f8ab8913`.
- V7.6.3 data: 50% source guard for TSMC5/16 NMOS and 20% for every other
  device; sorted data-marker aggregate
  `5279a128613a726e8019af8b5836f09384dcf7dec8bdff91a4f2683930937bd5`.
- Checkpoints: `results/v763_directnet_full_all_scales_checkpoints/`, 40
  bundles and 120 DirectNet-Full model/normalization/completion artifacts;
  sorted aggregate
  `54f16e183cc6bde287b9b2c0a6893b30892d3bacb4928a805138274d5f996c7c`.
- Training logs: `results/v763_directnet_full_all_scales_training/`, aggregate
  `0df3356219e247dd509e7c3423502caaa5ac6f181d834b41fb4040f3deacced6`.
- Job list: `results/v763_directnet_full_all_scales_jobs.txt`, 240 jobs,
  SHA-256 `822504829488ecbfbcb856065b7e427c0c2ffc450de30fcc48185dbf92cf7282`.
- Raw evaluation: `results/v763_directnet_full_all_scales_eval/`; 240/240
  verdict logs, zero coverage gaps, zero infrastructure tracebacks. Sorted log
  aggregate
  `c37a39e3b8c5fea9b86419b701897e1d5c286c70a9d5e12a3e43cab542886163`;
  collected `data.json` SHA-256
  `4d16a90a3dd6119e1fe052808cc0a0eca6c8b7adf24da02b769007cef35daa92`.
- Campaign identity digest:
  `322e4e4bf2bde3460837ba416506af11bc3cdcc87e2ee1000174300cf21df9e1`.
- Focused verification: 46 full-terminal/dataset/coverage tests passed; the
  56 emitted messages were PyTorch deprecation warnings, not skipped tests.
- AnalogGym source snapshot: detached commit
  `f600853939cf3d98ceec8871f9f6ba76c5523cde`, tree
  `da225ffc7225f754e4a973561255c18223100d28`, built from the same HEAD above.
  The stored snapshot patch at
  `results/v763_directnet_full_analoggym/provenance/source-snapshot.patch` has
  SHA-256 `ab4f5396c8f0578b28636e42e3ef138fd4a6d33fc7a387d645381bf04594e794`.
- AnalogGym raw evaluation:
  `results/v763_directnet_full_analoggym/`; 20 complete campaign roots,
  1,020/1,020 physical rows, 992 scored rows, 28 quarantines, zero timeouts,
  and zero infrastructure tracebacks. The checksum-bound `aggregate.json` has
  identity SHA-256
  `8e475d523b53463ddf7f8820254ba6c6c739528592d8c13d860b64411ef1ca7a`;
  its row, campaign-provenance, and top-log digests are respectively
  `035f5d86ccc9078fb3603b89805693c8395bdd8d8c985210bf3784396506c796`,
  `068994cb526eefeb9d4bfca4c02c354a659b7226ae29b260ec04965b8fa4b809`,
  and `0f0892d432efb8cb7d21cd5bec52ec6f8b80b26766fa1e4850a0841d7b5d993a`.
- Follow-up verification: seven focused terminal-length, NFIN-aware variant,
  pass-corridor, and support-guard tests passed with no skips. An initial mixed
  nested/root-suite command collected no tests because it selected the wrong
  import root; the suites were rerun separately and only those reruns count as
  evidence.

The targeted dataset markers predate the current clean-manifest source-commit
contract, identify the generator as V7.6.0, and omit the terminal PDK length
edges. The accepted `large` completion markers also predate embedded dataset
provenance. The hashes above preserve the actual campaigns, but these
limitations are why the result is not relabeled as a clean qualification pass.

The V7.6.2 clean baseline and gate definitions remain in
[`DirectNet-L75-clean.md`](DirectNet-L75-clean.md).
