# V7.7.5 checkpoints — quick round 1 across L0–L4

Date: 2026-09-14

**Status: scoped diagnostic, not a clean qualification.** This round scores
the 80 saved V7.7.2-trained bundles that V7.7.5 ships with. It uses a small
nominal, OMP=1 subset of each template tier so the user can decide whether to
run a longer campaign. It does not replace
[`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) or
[`BSIM-AR-L76-clean.md`](BSIM-AR-L76-clean.md). It does not satisfy the
OMP 1/2/4 strictness, corner coverage, or complete-pool rules in
[`methodology.md`](methodology.md).

**Verified 2026-09-25.** An independent parser re-read all 480 raw verdict
logs. It reproduced every gate count, tier count, physical error and
per-technology value below, `cells.json`, both manifest digests and both
job-list digests. The 280 checkpoint hashes also match the r5 and
[round-3](v775-round3.md) manifests. These statements were corrected in place:

- The BSIM-AR medium ring runtime was an interim value that averaged the two
  cells finished at the time. Over five technologies it averaged 269 minutes,
  not 85 minutes, so the medium ring slowdown is about 187×, not 60×.
- Both maximum-voltage columns cover simple-v2 rows only, not gate traces.
- The flat-trace ranges for the differential pair, and the rows labelled
  `Mp_r` and `infra`, now match their logs. The provenance commit list is now
  complete.

## Scope

Every cell uses the nominal corner, one CPU thread, and NGSPICE BSIM-CMG
LEVEL=72 on the same OSDI model as its reference. The matrix is
2 families × 4 sizes × 5 technologies × 12 cases = **480 cells**.

| tier | what the deck supplies | cases in this round |
|---|---|---|
| L0 devices | every terminal voltage | `verify_nn_multi_tech_dc` (129 parametric Id–Vgs configurations) |
| L1 primitives | bias plus one passive load | `common_source_nn`, `source_follower`, `common_gate` |
| L2 stages | ideal gate rails | `verify_circuit_sram_snm` (gate), `current_mirror`, `diffpair_ideal` |
| L3 blocks | supply only | `verify_circuit_ring_osc`, `verify_circuit_opamp`, `verify_circuit_switchcap` (gates), `bias_tree_fanout_3t` |
| L4 systems | supply and reference | `ota_5t_buffer` (DC transfer, uncued settling transient, closed-loop AC) |

The five `verify_*` gates keep their methodology thresholds. The seven
simple-v2 cases are held-out diagnostics without thresholds, so they are
reported as physical errors. A row that did not reach a converged fixed point
is an `ERROR`: it stays in its denominator and is excluded from numeric
aggregates.

## DirectNet-Full (LEVEL=75) — complete

### Scored gates at OMP=1

| size | device DC configs | SRAM | ring | Miller DC | switch-cap | simple-v1 cells |
|---|---:|---:|---:|---:|---:|---:|
| small | 128/129 | 5/5 | 5/5 | 5/5 | 3/5 | 18/20 |
| medium | **129/129** | 5/5 | 5/5 | 5/5 | 5/5 | **20/20** |
| large | 127/129 | 5/5 | 5/5 | 5/5 | 5/5 | **20/20** |
| xl | 126/129 | 5/5 | 5/5 | 5/5 | 5/5 | **20/20** |

Worst scored metrics across all sizes and technologies were: SRAM worst-lobe
NRMSE 7.08% (limit 10%); ring period error 1.99% (limit 5%); Miller DC gain
error 7.51% (limit 10%); switch-cap charge error 0.27% of VDD (limit 5%).
For context only, the preserved V7.6.6 `large` report scored 115/129 on
device DC. Its checkpoints and evaluation runtime differ, so the two
denominators are not one campaign.

Failures:

- Device DC: small TSMC5 NMOS +125 °C (NRMSE 19.4%). At large and xl,
  TSMC12/16 NMOS `nfin_10` fails (17.2–19.7%), and xl adds TSMC16 PMOS
  `nfin_10` (12.7%). The miss has moved from the hot corner for small to
  high NFIN for the larger models.
- Switch-cap small TSMC6/TSMC7: charge error is 0.27% of VDD, but hold droop
  is 1.98 mV against a 0.75 mV allowance (264%). TSMC6 is the documented
  TSMC7 repeat, so this is one failure seen twice.

### Simple-v2 diagnostics by tier

| size | L1 rows converged | L2 rows converged | L3 rows converged | L4 rows converged |
|---|---:|---:|---:|---:|
| small | 40/40 | 45/45 | 20/20 | 6/15 |
| medium | 40/40 | 45/45 | 20/20 | **15/15** |
| large | 40/40 | 41/45 | 20/20 | 12/15 |
| xl | 40/40 | 45/45 | 20/20 | 12/15 |

L2 and L3 counts include the gate rows in each tier (SRAM in L2; ring, Miller
and switch-cap in L3). L4 has no gate rows.

Physical errors over converged rows, all sizes pooled (median / worst):

| tier | case | metric | median | worst |
|---|---|---|---:|---:|
| L1 | common_source_nn | gain error | 1.18% | 19.3% |
| L1 | common_source_nn | bandwidth error | 1.13% | 12.7% |
| L1 | source_follower | gain error | 0.26% | 2.56% |
| L1 | common_gate | gain error | 1.02% | 6.08% |
| L2 | current_mirror | mirror ratio error | 0.34% | 1.86% |
| L2 | current_mirror | output resistance error | 4.72% | 39.1% |
| L2 | current_mirror | worst ratio error across the Iref sweep | 3.99% | 131% |
| L2 | diffpair_ideal | differential gain error | 0.78% | 2.19% |
| L2 | diffpair_ideal | CMRR error | 2.9 dB | 27.4 dB |
| L3 | bias_tree_fanout_3t | bias node error | 0.39 mV | 2.6 mV |
| L3 | bias_tree_fanout_3t | supply current error | 0.16% | 1.5% |
| L4 | ota_5t_buffer | follow error | 1.5 mV | 4.5 mV |
| L4 | ota_5t_buffer | closed-loop gain error | 1.83% | 5.25% |
| L4 | ota_5t_buffer | bandwidth error | 1.2% | 14.4% |
| L4 | ota_5t_buffer | settling error | 0% | 0.17% |

Errors (all attributable to the candidate; the reference converged in every
row. The derived CMRR row's raw marker says `result_schema`; see
[Interpretation limits](#interpretation-limits)):

- `ota_5t_buffer` small: 9 of 15 rows. Eight are `CandidateSupportError` on
  TSMC5/6/7/12/16, where an evaluation voltage left the persisted normalization
  box by 5–83 mV. One is a GMIN-retry nonconvergence on TSMC5.
- `ota_5t_buffer` large TSMC5: DC, transient and AC operating points did not
  converge.
- `ota_5t_buffer` xl TSMC5: `Mp_r` evaluated at −1.35 V (transfer and
  closed-loop AC) and −1.3175 V (settling) against its −1.30 V support floor.
- `diffpair_ideal` large TSMC5: DC and AC operating points did not converge,
  so derived CMRR has no inputs.

Per-technology aggregates over converged rows, all cases and sizes pooled.
Medians are used because range-normalized NRMSE and R² diverge on nearly flat
AC magnitude traces even when the physical gain error is small. Maximum voltage
error covers simple-v2 DC, operating-point and transient traces only. Gate
traces are excluded and are scored by their gate metrics above. AC `max_err`
is a small-signal magnitude per 1 V stimulus (V/V); it reaches 0.30 V/V on
common-source rows and is excluded.

| technology | rows | converged | median NRMSE | rows >10% NRMSE | median MRE | median R² | max simple-v2 DC/transient voltage error |
|---|---:|---:|---:|---:|---:|---:|---:|
| TSMC5 | 200 | 187 | 0.20% | 6 | 0.57% | 0.9999 | 4.5 mV |
| TSMC6 | 200 | 198 | 0.51% | 8 | 1.25% | 0.9997 | 22.3 mV |
| TSMC7 | 180 | 178 | 0.47% | 8 | 1.14% | 0.9997 | 22.3 mV |
| TSMC12 | 216 | 215 | 0.16% | 5 | 0.31% | 1.0000 | 4.8 mV |
| TSMC16 | 200 | 199 | 0.22% | 6 | 0.40% | 0.9999 | 7.6 mV |

## BSIM-AR-Full (LEVEL=76) — complete

All 240 cells finished. For scale, BSIM-AR ring cells averaged 39 minutes at
small and 269 minutes at medium (longest 738, TSMC7). DirectNet's averaged 68
and 87 seconds. Runtime is measured from log creation to verdict on a shared
host, so it includes load.

### Scored gates at OMP=1

| size | device DC configs | SRAM | ring | Miller DC | switch-cap | simple-v1 cells |
|---|---:|---:|---:|---:|---:|---:|
| small | 128/129 | 5/5 | 5/5 | 4/5 | 5/5 | 19/20 |
| medium | **129/129** | 4/5 | 5/5 | 4/5 | 5/5 | 18/20 |
| large | **129/129** | 5/5 | 5/5 | 5/5 | 5/5 | **20/20** |
| xl | **129/129** | 5/5 | 5/5 | 5/5 | 5/5 | **20/20** |

Worst scored metrics were: SRAM worst-lobe NRMSE 6.74%; ring period error
3.60%; Miller DC gain error 1.08%; switch-cap charge error 0.77% of VDD, with
droop at most 74% of its allowance.

Failures and errors:

- Device DC small TSMC5: NMOS +125 °C, NRMSE 13.5%.
- SRAM medium TSMC16: the BSIM-AR operating point at NFIN=5 did not converge.
  NGSPICE converged at every corner; see the attribution defect below.
- Miller DC small TSMC16: the candidate DC sweep did not converge at its first
  point (0.29 V).
- Miller DC medium TSMC5: `Mn1` evaluated at 1.318 V against its 1.30 V
  support ceiling.

### Simple-v2 diagnostics by tier

| size | L1 rows converged | L2 rows converged | L3 rows converged | L4 rows converged |
|---|---:|---:|---:|---:|
| small | 40/40 | 37/45 | 19/20 | 6/15 |
| medium | 40/40 | 44/45 | 19/20 | 12/15 |
| large | 40/40 | **45/45** | **20/20** | **15/15** |
| xl | 40/40 | **45/45** | **20/20** | 13/15 |

L2 and L3 counts include the gate rows in each tier, as in the DirectNet table.
L4 has no gate rows.

Errors by tier:

- L2 small: TSMC12 `diffpair_ideal` did not converge (4 rows). TSMC16
  `current_mirror` rejected `Mp_ref` at 0.98 V against a 0.96 V ceiling in all
  four analyses.
- L4 small: 9 rows. TSMC6 did not converge at DC, transient or AC. TSMC12
  had two convergence failures and one support rejection. TSMC16 had two
  support rejections and one GMIN failure.
- L4 medium: TSMC12 transfer and TSMC16 transfer/settling rejected `Mp_r`
  13–40 mV below its −1.60 V floor.
- L4 xl TSMC5: transfer and closed-loop AC rejected `Mn_r` at −1.036 V
  (floor −0.975 V); settling converged. The other four xl technologies
  converged in all three analyses.

Physical errors over converged rows, all sizes pooled (median / worst):

| tier | case | metric | median | worst (cell) |
|---|---|---|---:|---|
| L1 | common_source_nn | gain error | 2.66% | 20.0% (small TSMC5) |
| L1 | common_source_nn | bandwidth error | 1.80% | 21.1% |
| L1 | source_follower | gain error | 0.24% | 1.13% |
| L1 | common_gate | gain error | 0.82% | 6.88% |
| L2 | current_mirror | mirror ratio error | 0.27% | 2.74% |
| L2 | current_mirror | output resistance error | 3.55% | 48.2% (small TSMC12) |
| L2 | current_mirror | worst ratio error across the Iref sweep | 2.21% | 31.2% (small TSMC7) |
| L2 | diffpair_ideal | differential gain error | 1.34% | 4.38% |
| L2 | diffpair_ideal | CMRR error | 2.2 dB | 11.3 dB |
| L3 | bias_tree_fanout_3t | bias node error | 1.47 mV | 4.1 mV |
| L3 | bias_tree_fanout_3t | supply current error | 0.14% | 1.09% |
| L4 | ota_5t_buffer | follow error | 1.1 mV | 7.6 mV |
| L4 | ota_5t_buffer | closed-loop gain error | 1.01% | 55.4% (small TSMC5; medium TSMC5 50.6%) |
| L4 | ota_5t_buffer | bandwidth error | 1.8% | 12.7% |
| L4 | ota_5t_buffer | settling error | 0% | 0.17% |

The worst BSIM-AR diagnostic errors all come from small or medium models.
Large converges every row; xl misses only TSMC5's OTA transfer and
closed-loop AC.

Per-technology aggregates over converged rows, pooled the same way as
DirectNet. The voltage column covers simple-v2 rows only. AC `max_err` reaches
1.05 V/V on small TSMC5 common-source rows and is excluded from that column.

| technology | rows | converged | median NRMSE | rows >10% NRMSE | median MRE | median R² | max simple-v2 DC/transient voltage error |
|---|---:|---:|---:|---:|---:|---:|---:|
| TSMC5 | 200 | 197 | 0.34% | 9 | 0.86% | 0.9998 | 17.3 mV |
| TSMC6 | 200 | 197 | 0.34% | 9 | 1.24% | 0.9999 | 23.9 mV |
| TSMC7 | 180 | 180 | 0.35% | 9 | 1.10% | 0.9998 | 23.9 mV |
| TSMC12 | 216 | 208 | 0.39% | 3 | 0.90% | 0.9998 | 6.4 mV |
| TSMC16 | 200 | 189 | 0.36% | 7 | 1.15% | 0.9998 | 8.7 mV |

## Round-1 comparison

| size | DirectNet device DC | BSIM-AR device DC | DirectNet simple-v1 | BSIM-AR simple-v1 | DirectNet L4 rows | BSIM-AR L4 rows |
|---|---:|---:|---:|---:|---:|---:|
| small | 128/129 | 128/129 | 18/20 | 19/20 | 6/15 | 6/15 |
| medium | 129/129 | 129/129 | 20/20 | 18/20 | 15/15 | 12/15 |
| large | 127/129 | 129/129 | 20/20 | 20/20 | 12/15 | 15/15 |
| xl | 126/129 | 129/129 | 20/20 | 20/20 | 12/15 | 13/15 |

At large and xl, BSIM-AR matches DirectNet's 20/20 simple-v1 passes and
passes every device-DC configuration, where DirectNet misses two or three
high-NFIN ones. It also converges more L4 rows (15 and 13 of 15, against 12
and 12). At medium, DirectNet is ahead: 20/20 and 15/15 against 18/20 and
12/15. At small the two are within one cell. BSIM-AR's worst diagnostic
errors sit at small and medium.

Runtime is the other difference. Averaged over five technologies, BSIM-AR ring
cells took 39 minutes at small and 269 minutes at medium, against DirectNet's
68 and 87 seconds (about 35× and 187×). Switch-cap took 27 and 75 minutes
against 43 and 48 seconds (about 37× and 93×). The medium ring mean is set by
TSMC7 at 738 minutes; the other four took 68–244 minutes.

## Interpretation limits

- **Flat-trace metrics.** Of 38 converged `differential_ac` rows, 18 exceed
  100% range-normalized NRMSE (116–2,439%) while their differential gain
  error is 0.3–4.4%. Common-mode gain is 7.6e-9 to 1.4e-7 V/V with an ideal tail, so its relative error
  and the CMRR dB error amplify numerical noise. Read the domain metrics.
- **Support errors checked against the reference trajectory.** LEVEL=75/76
  reject every evaluation outside the normalization box, and no NN-side
  voltage limiter precedes that check. A follow-up diagnostic ran only the
  NGSPICE reference for the DC and transient analyses that raised a support
  error in round 1:
  `ota_5t_buffer` transfer and settling, all four `current_mirror` analyses,
  and Miller `opamp` transfer. It checked each accepted terminal voltage
  against the box of all 40 checkpoint pairs (source `d708d4b`, evidence in
  `PyCircuitSim-v775-round2/results/v775_round2/l4_support/`). All 280 records,
  2,875,880 values, are inside support. Those 18 of round 1's 24
  `CandidateSupportError` rows therefore came from intermediate Newton
  iterates: a solver-globalization limit, not missing training data. The other
  6 are `ota_5t_buffer` closed-loop AC rows; they were not checked and remain
  unattributed.
- **Harness misclassification.** When a differential pair's candidate
  operating point does not converge, the derived CMRR row reports
  `result_schema`, the worker exits 2, and the dispatcher labels the cell
  `infra`. The rows are scientific `ERROR`s and are counted that way here.
  The same defect stopped the fifth staged V7.7.2 arm at 3,429 jobs. Both
  `diffpair_ideal` logs (DirectNet large TSMC5, BSIM-AR small TSMC12) still end
  in `rc=infra`, because the fixed marker did not exist yet.
- **SRAM error attribution.** `verify_circuit_sram_snm.py:471-472` sets both
  `reference_converged` and `candidate_converged` from "any NFIN corner
  errored". `GateResult` then infers `error_kind=reference`
  (`gate_result.py:106-108`) even when only the candidate failed. The BSIM-AR
  medium TSMC16 row is such a case: NGSPICE converged at NFIN 2, 5 and 10, and
  BSIM-AR's NFIN=5 operating point did not. The FAIL verdict is correct, the
  marker's attribution is not, and its metric covers only the surviving
  corners. It is counted here as a candidate `ERROR`.
- **OMP=1 only.** Ring and Miller verdicts lack the OMP 2/4 flip check;
  [round 2](v775-round2.md) ran it. TSMC6's reference repeats TSMC7's, but its
  checkpoints are a separate training run: several sizes give identical rows,
  others differ (methodology §7).

## Provenance

- Checkpoints: `PyCircuitSim-v771/results/v771_r2_checkpoints`, 80 bundles
  and 280 artifacts. Every hash matches the r5 evaluation manifest.
  `PyCircuitSim-v771/results` is a symlink to main's `results/`, so the two
  paths name one directory. The training and dataset source is
  `6be83348c1f5db6720d7504ed6dcea874a3a7418`.
- Evaluation source: clean detached worktree `PyCircuitSim-v775-quick` at
  `39b14f11b92dbb9efc4aa00244e5a75613aeb2a4`. That is main `d6ae11c` (V7.7.5)
  plus nine evaluation-branch commits: the staged-campaign scheduler,
  fixture isolation from campaign data, every applicable device-integrity
  corner, terminal admittance phase, subthreshold reference windows and the
  `reference_metric` error kind, DC-sweep endpoints, lifted-source canary
  sign, and hierarchy AC phase. The manifest records a `distinct-evaluation-arm`; it does
  not claim numerical-source equivalence. Those repairs were unmerged while
  this round ran and reached `main` in merge `b688412` (2026-09-21), after
  every cell here was scored.
- Manifests (SHA-256): clean pool
  `7108d9334457086efef90cd029d7fa84c908523aab2b60d5903dd45e530a4c98`
  (200 jobs, list `cf852d77…e9ef`), simple-v2 pool
  `c067ff314b4427e4d5d1a38bcd6a4892d6a7d14d255e327630d6d47c18383825`
  (280 jobs, list `abd3f5e0…9204`). Every verdict log carries its pool digest.
- OSDI `f089f17d…a78b`; NGSPICE 45.2 `3b931f4e…d764`.
- Geometry preflight on `v771_r2_data`: 463/463 PASS.
- Execution: 56 cells in flight were killed when the first launch exceeded the
  session's background-task memory limit. Their partial logs and artifacts are
  archived under `results/v775_quick_r1/attempts/killed_20260914T0845/`. The
  cells were rerun from scratch under capped `systemd --user` units with the
  same manifest digests. Two units walk the clean and simple-v2 lists forward,
  and a third walks the clean list in reverse. Every cell takes a `flock` and
  skips any completed verdict, so no cell runs twice. The runner also re-executes cells labelled `infra`;
  both `diffpair_ideal` reruns reproduced the same nonconvergence.
- Raw evidence (ignored by Git):
  `PyCircuitSim-v775-quick/results/v775_quick_r1/`, including `SUMMARY.md`,
  `cells.json`, launch scripts and the aggregator.
