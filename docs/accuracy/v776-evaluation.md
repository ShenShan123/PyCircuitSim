# V7.7.6 evaluation — full campaign, NN limiter off and on

Run: 2026-09-26 to 2026-09-30.

**Status: one complete campaign pass from one commit.** It scores the 80
V7.7.2-trained bundles (DirectNet-Full and BSIM-AR-Full × 4 sizes × 5
technologies × NMOS/PMOS) on every nominal cell the campaign generator emits,
in two arms that differ only in `PYCIRCUITSIM_NN_NR_LIMIT`. The `off` arm is
the scored V7.7.6 runtime; the `on` arm is the limiter re-gate (criterion 4 of
the [plan](../plans/2026-09-21-v776-nn-voltage-limiting.md)). Methodology §5's
single-campaign condition holds, and the `off` arm's clean pool is the evidence
pass of [`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) and
[`BSIM-AR-L76-clean.md`](BSIM-AR-L76-clean.md). Gate rules are owned by
[`methodology.md`](methodology.md).

The model × test circuit × technology tables are in
[`v776-case-matrix-off.md`](v776-case-matrix-off.md) (scored) and
[`v776-case-matrix-on.md`](v776-case-matrix-on.md) (limiter).

- 1,840/1,840 cells per arm, none `infra`. `off`: 1,602 PASS, 238 FAIL.
  `on`: 1,638 PASS, 202 FAIL.
- NGSPICE converged in all 7,232 rows per arm; every `ERROR` row is attributed
  to the candidate.
- 7,151 of 7,232 rows are bit-identical to V7.7.5 rounds 1–3, and every scored
  field matches in 7,214. The capacitor-state fix corrects four published
  V7.7.5 values.
- The limiter converts 71 of 128 support-rejection `ERROR`s to converged rows
  and changes no row that converged without it.
- Every scored DC sweep in both arms passes a post-hoc KCL re-stamp.
- BSIM-AR costs 67× DirectNet in cell-hours (2,067 against 31).

## Campaign

| item | value |
|---|---|
| runtime | `ff4234b` (V7.7.6 + trace archives), clean worktree, CPU, 1 OMP/MKL/Torch thread |
| checkpoints | `v771_r2_checkpoints`, dataset source `6be8334`, 80 bundles with completion markers |
| pools per arm | clean 600, simple-v2 1,200, canary 40 (generator output; jobs `68249aa6`, `c3c0ec8b`, `c21fba8a`) |
| manifests `off` | clean `297868d0`, simple-v2 `eba855e4`, canary `05ffc7b5` (`runtime_knobs` empty) |
| manifests `on` | clean `9dc4581f`, simple-v2 `312629cd`, canary `b81851f0` (`PYCIRCUITSIM_NN_NR_LIMIT=1`) |
| preflight | 1,174 tests passed; geometry coverage 463/463 |
| interruption | a host reboot on 2026-09-27 killed 68 in-flight cells (34 per arm); they reran from scratch on byte-identical manifests, and their partial logs are archived |

OMP 2/4 repeats cover only the ring and Miller gates (160 cells per arm).
Every other table uses the 1,680 OMP=1 cells.

## Qualification gates (`off`)

| gate | DN-S | DN-M | DN-L | DN-XL | AR-S | AR-M | AR-L | AR-XL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| device DC configs (/129) | 128 | 129 | 127 | 126 | 128 | 129 | 129 | 129 |
| simple-v1 cells at OMP=1 (/20) | 18 | 20 | 20 | 20 | 19 | 18 | 20 | 20 |
| ring + Miller strict at OMP 1/2/4 (/10) | 10 | 10 | 10 | 10 | 9 | 8 | 10 | 10 |
| device AC (/10) | 10 | 10 | 10 | 8 | 10 | 10 | 10 | 10 |
| Miller open-loop AC (/5) | 0 | 0 | 2 | 2 | 1 | 4 | 4 | 4 |
| inverter VTC/transient configs (/100) | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 |
| lifted-source canary configs (/30) | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |

DN = DirectNet-Full (LEVEL=75), AR = BSIM-AR-Full (LEVEL=76); S/M/L/XL = size.
Every verdict equals V7.7.5 rounds 1–3, whose reports (in Git history)
explain each miss: [round 1](https://github.com/ShenShan123/PyCircuitSim/blob/bde2c11/docs/accuracy/v775-quick-round1.md),
[round 2](https://github.com/ShenShan123/PyCircuitSim/blob/bde2c11/docs/accuracy/v775-round2.md), [round 3](https://github.com/ShenShan123/PyCircuitSim/blob/bde2c11/docs/accuracy/v775-round3.md). One cell depends on
the thread count: BSIM-AR medium TSMC12 Miller DC passes at 1 and 2 threads and
does not converge at 4. The clean reports score it strictly as `ERROR`, so
BSIM-AR medium is 17/20 there. Device and terminal integrity and the NN
off-current excess are bit-identical to round 3. So is flat/nested netlist
agreement, except the BSIM-AR medium TSMC16 rows described below.

With the limiter on, every gate verdict is unchanged. One qualification row
moves from `ERROR` to FAIL: BSIM-AR medium TSMC5 Miller AC converges with a
34.8 dB DC-gain error.

## Models × test circuits × technology

Median NRMSE (%) over converged rows · converged share, pooled over the five
technologies. The matrix files split each level by technology and test case.

| level | DN-S | DN-M | DN-L | DN-XL | AR-S | AR-M | AR-L | AR-XL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 devices | 0.74 · 100% | 0.17 · 99% | 0.08 · 100% | 0.12 · 99% | 0.83 · 97% | 0.38 · 99% | 0.21 · 99% | 0.12 · 100% |
| L1 primitives | 1.52 · 100% | 0.35 · 100% | 0.19 · 99% | 0.15 · 99% | 2.01 · 100% | 1.06 · 100% | 0.49 · 96% | 0.23 · 96% |
| L2 stages | 1.21 · 98% | 0.57 · 97% | 0.57 · 97% | 0.66 · 98% | 1.07 · 93% | 0.83 · 96% | 0.59 · 98% | 0.32 · 97% |
| L3 blocks | 1.44 · 85% | 0.37 · 88% | 0.25 · 82% | 0.37 · 90% | 0.96 · 85% | 0.57 · 89% | 0.36 · 91% | 0.24 · 85% |
| L4 systems | 6.88 · 68% | 2.28 · 92% | 1.00 · 82% | 2.01 · 93% | 3.00 · 53% | 1.98 · 93% | 0.75 · 88% | 0.43 · 80% |
| hierarchy | 1.14 · 100% | 0.37 · 100% | 0.39 · 100% | 0.38 · 100% | 1.77 · 100% | 2.10 · 87% | 0.59 · 100% | 0.39 · 100% |
| L4, limiter on | 5.82 · 70% | 1.71 · 95% | 1.00 · 82% | 1.52 · 100% | 4.94 · 62% | 1.93 · 100% | 0.86 · 100% | 0.51 · 97% |

- BSIM-AR error falls with size at every level from L0 to L3. DirectNet's
  stops improving after medium at L2 and after large at L3. Small models are
  roughly 6–10× worse than XL at L0–L1.
- Convergence, not accuracy, limits L3–L4. Of the 360 `ERROR` rows,
  `beta_multiplier` holds 76 and `sram6t_modes` 52; `ota_5t_buffer`,
  `multistage_buffer_12t` and `ldo_regulator` hold 28–29 each.
- The limiter changes no L0, L1 or hierarchy row, and moves L2 and L3
  convergence by at most 4 points. At L4 it lifts DirectNet XL and BSIM-AR
  medium and large to 100%, and BSIM-AR XL to 97%. The L4 medians of BSIM-AR
  small, large and XL rise because the rows recovered there carry larger
  errors; the others fall.

## Per-technology error (`off`, OMP=1)

| model | technology | rows | converged | ERROR | median NRMSE % | NRMSE > 10% | median MRE % | median R² | row max V error, median (mV) | max V error (mV) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DirectNet-Full | TSMC5 | 708 | 646 | 62 | 0.29 | 53 | 0.69 | 0.9998 | 8.7 | 668 |
| DirectNet-Full | TSMC6 | 708 | 682 | 26 | 0.57 | 58 | 1.23 | 0.9996 | 16.1 | 760 |
| DirectNet-Full | TSMC7 | 688 | 661 | 27 | 0.54 | 59 | 1.23 | 0.9996 | 16.2 | 777 |
| DirectNet-Full | TSMC12 | 724 | 694 | 30 | 0.24 | 39 | 0.52 | 0.9999 | 6.8 | 841 |
| DirectNet-Full | TSMC16 | 708 | 693 | 15 | 0.35 | 48 | 0.61 | 0.9998 | 7.8 | 871 |
| BSIM-AR-Full | TSMC5 | 708 | 662 | 46 | 0.53 | 68 | 1.08 | 0.9996 | 9.9 | 687 |
| BSIM-AR-Full | TSMC6 | 708 | 673 | 35 | 0.47 | 54 | 1.20 | 0.9997 | 12.2 | 788 |
| BSIM-AR-Full | TSMC7 | 688 | 664 | 24 | 0.47 | 60 | 1.12 | 0.9997 | 11.0 | 786 |
| BSIM-AR-Full | TSMC12 | 724 | 685 | 39 | 0.47 | 52 | 1.07 | 0.9997 | 11.7 | 829 |
| BSIM-AR-Full | TSMC16 | 708 | 657 | 51 | 0.53 | 54 | 1.27 | 0.9995 | 20.0 | 872 |

Voltage error covers every DC, operating-point and transient voltage trace:
simple-v2 cases, the four simple-v1 gates, inverter transients and
`nn_subckt`. It excludes the NAND2/NOR2 stack nodes `v(nint)`/`v(pint)` in
every analysis, because NGSPICE's `uic` start-up step drives them past the
rails ([round 2](https://github.com/ShenShan123/PyCircuitSim/blob/bde2c11/docs/accuracy/v775-round2.md)). Every per-technology maximum comes from a
free-running ring oscillator: `ring_osc_supply`, or the `ring_osc` gate for
BSIM-AR TSMC6. A small period error (`ring_osc_supply`: median 0.27%, worst
3.6% over 40 rows) accumulates into phase drift across the window, so the
pointwise error measures accumulated phase, not voltage levels. The
next-largest are one-step (5 mV) VTC trip shifts in `nn_subckt` DC (up to
775 mV), described in [round 3](https://github.com/ShenShan123/PyCircuitSim/blob/bde2c11/docs/accuracy/v775-round3.md).

## Change from V7.7.5

Pairing every row by cell, case, analysis and corner, and comparing every
field including nested payloads, 7,151 of 7,232 rows are bit-identical to
V7.7.5 rounds 1–3.

| cause | rows | effect |
|---|---:|---|
| capacitor-state fix (`ee8fef4`) | 15 | scored fields change; see below |
| capacitor-state fix, unscored payload only | 63 | `ERROR` rows whose `unconverged_diagnostic` (the readable-trace rerun no score uses) changed; 52 are `sram6t_modes` |
| error attribution fixed in `b688412` | 3 | `result_schema`/`reference` labels become `candidate`; messages unchanged; two V7.7.5 `infra` cells now exit 1 |

All 78 fix-driven rows are in circuits with a capacitor. Reruns with the fix
disabled reproduce V7.7.5 exactly: all eight DirectNet small TSMC5
`sram6t_modes` rows in every field, and the steering row below. Four of the 15
scored changes correct published V7.7.5 numbers:

- DirectNet small `diffpair_active_load` steering (TSMC6/7 PMOS, TSMC12 NMOS),
  round 2: `v(out)` NRMSE was 4.56%, 4.56% and 2.07%; it is 0.04%, 0.04% and
  0.01%. On TSMC6 the rerun without the fix gives the V7.7.5 row NRMSE
  exactly (4.557%); with it, 0.246%. The analysis is an operating point, so
  the round-3 KCL audit, which checked saved DC sweeps only, could not see
  it.
- BSIM-AR medium TSMC16 `nn_subckt` AC, round 3: V7.7.5 reported a converged
  row at 43% NRMSE. Its operating point came from the leaked state. It is now
  a candidate `ERROR`, like the DC row round 3 reclassified.

Of the other 11, one DirectNet small TSMC12 `diffpair_active_load` AC row now
converges. The `nn_subckt` DC row that round 3 reclassified as a false
convergence is now an ordinary non-convergence. Nine rows stay `ERROR` with a
different message (support rejection ↔ non-convergence).

## NN limiter re-gate

| plan criterion | result |
|---|---|
| 1. support rejections converge or fail for a physical reason | partly: of 128 support-rejection rows, 71 converge, 1 stays a support rejection, 56 become non-convergence |
| 2. converged cells bit-identical when the limiter does not engage | met: every row that converges with the limiter off is bit-identical with it on; 7,088 of 7,232 rows are bit-identical in every field |
| 3. latch-basin and solver-numerics contracts pass on and off | met in the preflight suite (1,174 tests) |
| 4. re-gate shows the effect on convergence | this section |

- **Converged: 71 rows, plus one derived CMRR row whose input now converges;
  36 cells FAIL → PASS.** They include `diffpair_active_load` (13 rows),
  `ldo_regulator` (12), `cascode_stack` (9), `multistage_buffer_12t` (9) and
  `ota_5t_buffer` (9). Their median NRMSE is 0.91%; 11 of 71 with a trace
  metric exceed 10%, led by BSIM-AR TSMC5 LDO line regulation (131–501%). A
  recovered row is a physical fixed point, not necessarily an accurate one.
- **Still a support rejection: 1 row.** It is DirectNet small TSMC16 NOR2
  transient, where an accepted time point is 5 µV past the box edge. The
  limiter never clamps an accepted state.
- **Non-convergence: 56 rows.** In 17, the final iterate is still clamped. For
  example, BSIM-AR medium TSMC5 `beta_multiplier` stops at `nb − vdd` =
  0.91 V against a 0.78 V bound on `Mp_mir`. There the NN circuit's Newton
  attractor lies outside the box, while round 2 found every NGSPICE solution
  inside it. The other 39 end without a clamp and fail like any hard NN solve.
  `beta_multiplier` (11 rows), `multistage_buffer_12t` (12), `ota_5t_buffer`
  (12) and NOR2 (10) dominate.
- **Other changes: 15 rows, unscored.** None of the 204 rows that fail to
  converge with the limiter off changes status, error or metric. In 15 of
  them the `_unconv` diagnostic rerun, which the support check stopped
  without the limiter, now returns a readable trace that no score uses.
- **Cost.** Cells with no changed row ran slightly faster in the `on` arm:
  the on/off wall-time median is 0.95 for BSIM-AR (5–95%: 0.68–1.08) and 0.83
  for DirectNet, so differences within about ±30% are host noise. BSIM-AR
  cells whose scored rows changed take a median 3.2× longer. The worst,
  BSIM-AR medium TSMC5 `beta_multiplier`, took 8.7 h against 75 s and still
  ends in `ERROR`. BSIM-AR total cell-hours rise 6% (2,067 → 2,190).

A default flip is not proposed here. The case for it is 72 more converged
rows and no changed converged row. The case against is the unbounded cost of a
clamped solve that cannot converge, and no evidence yet that 17 clamped
non-convergences are closer to the reference than the support rejections they
replace.

## KCL audit

Every saved candidate DC sweep was re-stamped point by point on a fresh parse
of its own deck. Its MNA residual was compared with the DC solver's acceptance
threshold, max(1 µA, 100·RELTOL·|current RHS|). The method is round 3's, and
it now also finds the device-DC and inverter-VTC decks, so every sweep has a
deck.

| arm | scored sweeps (with a capacitor) | not fixed points | worst residual / threshold | `_unconv` sweeps flagged |
|---|---:|---:|---:|---:|
| `off` | 3,930 (480) | 0 | 0.45 | 12 of 122 |
| `on` | 3,957 (495) | 0 | 0.45 | 16 of 154 |

Every scored DC sweep is a true fixed point. The flagged sweeps are `_unconv`
reruns of `ERROR` rows, which solve without `require_convergence` only to keep
a readable trace; no score uses them. Operating points behind AC and transient
analyses are not saved and remain unaudited.

## Runtime

| family | `off` cell-hours | `on` cell-hours | longest cell |
|---|---:|---:|---|
| DirectNet-Full | 30.7 | 27.0 | 1.1 h, XL TSMC12 `ring_osc_supply` |
| BSIM-AR-Full | 2,066.7 | 2,189.6 | 57.0 h, XL TSMC16 `ring_osc_supply` |

Wall time runs from verdict-log creation to verdict. Both arms ran together on
a shared 76-core host with 76 parallel workers (84 after the reboot). Cells over
1 h that started after the reboot took a median 1.77× their V7.7.5 time.

## Artifacts

Under `PyCircuitSim-v776-eval/results/v776_eval/` (not in Git):

- `{off,on}/{clean,simple_v2,canary}/`: verdict logs, manifests and per-cell
  artifacts, including every compared candidate/reference trace pair.
- `waveforms/{off,on}/<level>/<case>/<tech>/`: 699 figures per arm, each
  overlaying NGSPICE and the four sizes of both families
  (`scripts/campaign_waveform_plots.py`).
- `views/`, `compare_rows.py`, `tools/campaign_aggregates.py`,
  `kcl_audit/`, `capfix_repro/`: the round summarizers rerun unchanged on this
  campaign, the row pairing, the aggregates, the audit, and the reruns with
  the capacitor fix disabled.
- `interrupted_20260927/`: partial logs of the 68 cells the reboot killed.

The clean-pool verdict logs and collected `data.json` behind the clean reports
are in `results/v776_full_clean/` of the main checkout. The V7.7.5 raw evidence
that the "Change from V7.7.5" section compared against was purged on
2026-10-01; `capfix_repro/` keeps the reruns that attribute the changes.
