# AnalogGym across TSMC FinFET nodes

This is a strict NGSPICE/PyCMG audit of every generated design. Topologies are checked against their source, dimensions are constrained to each modelcard, and every available DC, AC, temperature, PSRR, line/load-regulation and transient gate is included. A result is fully passing only when every gate passes and the simulator reports no analysis error. `!` marks a partial result with one or more simulator/evaluator errors. Per-tree details: `designs_<tech>/RESULTS.md`.

Since the migration into PyCircuitSim (`examples/complex_circuits/`, V7.5.0) this file
carries **two** axes. The next section scores **PyCircuitSim against NGSPICE**
on the identical BSIM-CMG (LEVEL=72) OSDI model. Everything after it is the
original NGSPICE/PyCMG design audit, unchanged — that audit is the reference the
comparison is measured against, not a second opinion about it.


## The curated core basket (V7.5.9)

Two curations, one direction. V7.5.6 took the corpus from **38 designs / 159
scored decks per tech to 18 / 75**; V7.5.9 takes it to **12 / 51** — 60 design
instances and 255 scored decks across the five techs, down from 190 and 795
originally.

**Denominators, in one place.** `/159`, `/679`, `/795`, 38/38, 17/17, 13/13 are
the original corpus. `/75`, `/375`, 18/18 are V7.5.6. `/51`, `/255`, 12/12 are
V7.5.9. Sections below this line are dated and keep the denominator they were
measured with; none of them is comparable to another without rescaling.

### What V7.5.9 removed, and how it was chosen

The V7.5.6 cut was argued from **structure** — topology class, device count,
byte-identical netlists. V7.5.9 ran the whole basket through both simulators at
HEAD first (`v759_baseline_*`, 375 decks) and cut on **measured
discrimination**: how many decks a design disagrees with NGSPICE on, against
what it costs to find that out. The two rankings disagree, which is the point.

| removed | decks agreeing | cost (5 techs) | why |
|---|:--:|--:|---|
| `ptat_2` | 5/5 | 22 s | never disagreed anywhere; 6 MOS, and every one of its (Vt, L, NFIN) cells is carried by another design |
| `front_end_11_6T_schematic` | 5/5 | 25 s | never disagreed anywhere; **six devices of one model card** (`nulvt_l60_f10`) on 4 nodes — the least discriminating circuit in the corpus. `front_end_25_6T` is the same 6T class and disagrees on 4 of 5 techs |
| `SMCNR_SE_2st_AMP` | 10/10 | 63 s | never disagreed anywhere; an 8-MOS amplifier filed under sensors, whose `tb_ac` metric set is `amplifier/tb_gain`'s eight with `phase_in_rad` spelled `ph_rad`. Five 24–37-MOS amplifiers carry that bench |
| `Alfio_RAFFC_Pin_3` | 27/30 | 303 s | the shallowest amplifier residuals in the basket (≤3.6 %); 24 MOS, structurally the same three-stage Miller cell as `Fan_SMC`, which is additionally the sole NFIN=7 / L=67 carrier and holds the corpus's only AC miss |
| `Qu_LEC_Pin_3` | 25/30 | **3145 s (28 %)** | see below |
| `Basic_LDO` | 35/39 | 1316 s (12 %) | 24 MOS; its two miss classes (`overshoot`/`undershoot`) are held by both surviving LDOs, and V7.5.5 characterized all three as reference-tolerance artifacts |

`Qu_LEC` is the interesting one. V7.5.6 kept it as "the hardest known
transient" and it was: 5/11 with a 31 % `sr_fall` residual, then the only deck
that deep. It is no longer. On this pass `Leung_NMCNR` reads **0–1/11 with
29–30 % slew residuals on four of five techs at a fifth of the cost**, and
`Song_DACFC` reads 3/11 at 12–15 %. `Qu_LEC` was buying 28 % of the campaign
to say something two cheaper decks now say louder.

**The pruning is measurement-preserving, and that is measured, not asserted.**
The same campaign was re-run over the pruned tree (`v759_basket_*`): all
**255 surviving decks are identical to their pre-prune run — same verdict,
same miss set, and every miss's relative error equal to six decimal places.**

| | designs/tech | decks/tech | 5-tech cost | speed-up |
|---|:--:|:--:|--:|--:|
| V7.5.6 basket, at V7.5.9 HEAD | 18 | 75 | 3.35 CPU-h | — |
| V7.5.9 basket | 12 | 51 | **1.77 CPU-h** | 1.89× |
| V7.5.9 basket, tsmc6 skipped | 12 | 51 | **1.39 CPU-h** | **2.41×** |

What the basket provably retains, checked against the same 375-deck pass:

* **All four analysis families** (AC 28 decks/tech, dc_temp 9, dc_source 6,
  transient 8) and 17 of the 18 `(category, deck)` metric classes — the one
  lost is `sensing_front_end/tb_ac`, which measures the same eight quantities
  as `amplifier/tb_gain` under one renamed key.
* **Every metric name that has ever disagreed with NGSPICE — 34 of 34 — is
  still measured by a surviving deck.**
* **The complete NFIN vocabulary** {1,2,3,4,5,6,7,8,10,12} and **all six Vt
  flavors** (nsvt/nlvt/nulvt/psvt/plvt/pulvt). Channel length drops to
  **31 of 33 L bins**; the two lost (110, 128 nm) are single-instance bins and
  the widest resulting gap is 120→135 nm.
* Every design that disagrees with NGSPICE anywhere. Nothing removed was a
  known-miss carrier.

### The tech axis: run four, not five

`designs_tsmc6` stays on disk and is **not** run by the L72 bench. Under
LEVEL=72 it is an exact simulation duplicate of `designs_tsmc7` — identical
netlists, modelcards differing only in TMI layout-effect keys the BSIM-CMG
Verilog-A never reads. Re-measured at V7.5.9 HEAD: **75/75 decks identical in
verdict AND in every miss's relative error to four decimal places.** It costs a
fifth of the campaign to learn nothing.

It is kept as a directory because the NN families train **separately** on it,
which makes it the training-run-variance control on the NN accuracy axis
(`methodology.md` §7) — a different axis, where it is not a duplicate. Score
`--tech tsmc5,tsmc7,tsmc12,tsmc16` and quote tsmc6 as the repeat.

### Historical: what V7.5.6 removed

Why prune: the corpus was built to audit *AnalogGym designs*, and it did that
job (38/38 on every tech). As an *accuracy benchmark for compact models* it
was heavily degenerate — 17 amplifiers of one topology class sharing one
6-bench structure, seven 2-to-6-transistor sensor stacks sharing one DC
bench, and two literal duplicates. Measured on the V7.5.4-corrected per-deck
times from `pycircuitsim_bench_results/`, the full corpus costs **5.32 CPU-h**
and the basket **3.35 CPU-h — a 37 % saving for a 53 % deck cut**, the gap
being deliberate: the pruning kept the expensive hard decks and dropped the
cheap saturated ones.

What was removed, and the evidence for each:

| removed | n | evidence |
|---|:--:|---|
| `ptat_6` | 1 | **byte-identical** to `ptat_2` — netlist and bench differ only in the subckt name |
| `three_output_vref` | 1 | its MOS core is byte-identical to `dual_output_subthreshold_vref`; the third output is that core plus two ideal 1e18 Ω resistors onto a node the audit itself records as *"intentionally high impedance and not qualified for load drive"*. The qualified core is the one kept |
| 6 sensor stacks | 6 | `PTAT_SENSOR` (2 MOS), `front_end_31_3T` (3), `PTAT_65_classic1`, `PTAT_CLASSIC`, `ptat_3`, `front_end_42_2_2015_REF` (4 each) — single DC bench, no unique L or NFIN bin, all subsumed by the 6-and-8-MOS sensors kept |
| 2 LDOs | 2 | `ldo_simple` (6 MOS) and `ldo_folded_cascode` (10 MOS, L ⊂ `ldo_1`'s) |
| 10 amplifiers | 10 | all 17 are three-stage Miller-class `Pin_3` designs on one 6-bench structure; the 7 kept span 24→37 MOS and five distinct numerical stress modes |

V7.5.6 kept 7 amplifiers, retained all 18 metric classes and 29 of the 31
metric names that had ever missed, the complete NFIN vocabulary, and 33 of 45
L bins. Post-prune health check (`--stride 1`, one deck per family, tsmc5):
`tb_gain` 8/8, `ldo_1/tb_load` 11/11, `ptat_1/tb_dc` 13/13,
`dual_output_subthreshold_vref/tb_dc` 12/12 — every verdict identical to its
pre-prune value.

### The 12 designs in the basket, each for a named reason

| category | design | MOS | why it is here |
|---|---|--:|---|
| amplifier | `Fan_SMC_Pin_3` | 24 | sole source of NFIN=7 and L=67; carries the corpus's only AC miss (`tb_cmrr`, reference-side) |
| amplifier | `Qu2017_AZC_Pin_3` | 25 | the NGSPICE Newton-basin case; the only deck that exercises the `tb_tran_altns` alternate-seed fallback |
| amplifier | `Leung_NMCNR_Pin_3` | 24 | **the deepest transient residual in the basket** — 0–1/11 at 29–30 % slew on four techs, 11/11 on tsmc5 |
| amplifier | `Peng_IAC_Pin_3` | 34 | 27 nodes; the AC-pathological design that forced the `pm_true` wrap-aware rework |
| amplifier | `Song_DACFC_Pin_3` | 37 | largest amplifier; the open multi-OP basin case, and the only design whose basin split reaches the **AC** benches (tsmc12 `tb_gain` 0/1) |
| ldo | `ldo_1` | 9 | the `tb_load` anchor; sole carrier of the two shortest L bins (6, 8 nm) |
| ldo | `ldo_2` | 20 | the hardest LDO — carries every LDO miss class (`lr`/`lr_pp`, `lnrmax`, load-step excursions) |
| sensing_front_end | `ptat_1` | 6 | the V7.5.1 subthreshold-floor regression anchor (5/13 → 13/13) |
| sensing_front_end | `ptat_4` | 8 | widest sensor geometry (8 cells, 3 Vt flavors); `min_slope` miss carrier |
| sensing_front_end | `front_end_25_6T_schematic` | 6 | the V7.5.4 deep-subthreshold case (m=360/1728 nulvt cores); misses on 4 of 5 techs |
| voltage_reference | `dual_output_subthreshold_vref` | 7 | the only design carrying **all six** Vt flavors; sole `voltage_reference` metric class |
| charge_pump | `chargepump` | 41 | the only switching circuit; the 10 ps current-reversal spike that drove the whole refine-controller line of work |


## The gap between PyCircuitSim and NGSPICE (V7.5.9, all five techs)

**This is the current number. Everything in the sections after it is older.**
Both simulators run the identical BSIM-CMG (LEVEL=72) OSDI model; both are
scored by one measurement engine, so a reported difference cannot be a
difference in `.meas` semantics. Transients run refine-on (the scored mode
since V7.5.5); stride policy unchanged (amplifier `tb_dc` @25, `tb_tran` @4,
charge pump @20 refine+trap, everything else full grid). Evidence:
`pycircuitsim_bench_results/v759_basket_tsmc{5,6,7,12,16}/`, untracked, on
disk.

| tech | AC | dc_source | dc_temp | transient | total |
|---|:--:|:--:|:--:|:--:|:--:|
| TSMC5 | 27/28 | 5/6 | 7/9 | 4/8 | **43/51** |
| TSMC6 | 28/28 | 3/6 | 6/9 | 2/8 | 39/51 |
| TSMC7 | 28/28 | 3/6 | 6/9 | 2/8 | 39/51 |
| TSMC12 | 25/28 | 5/6 | 7/9 | 2/8 | 39/51 |
| TSMC16 | 28/28 | 6/6 | 8/9 | 1/8 | **43/51** |
| **all** | **136/140** | **22/30** | **34/45** | **11/40** | **203/255 (79.6 %)** |

A deck counts only when every metric both simulators produced agrees inside
the 2 % gate and none is missing. TSMC6 and TSMC7 are verdict-identical and
miss-magnitude-identical on all 51 decks, as they must be.

Three numbers that qualify the table:

* **Operating-point agreement, which the metric columns can hide** (V7.5.4's
  lesson): over 253 decks the worst node error is **median 4.9 µV, p90
  0.26 mV**. The single 0.495 V outlier is `Song_DACFC` on tsmc6/7/12/16 — a
  genuine multi-OP basin difference, flagged below, not a convergence failure.
* **Engine control: 255/255 decks clean.** Our reader of NGSPICE's own sweep
  reproduces the `.meas` values NGSPICE printed from that same run — so the
  disagreements below are simulator differences, not measurement differences.
* **Cost: 1.77 CPU-h for the five techs, 13× NGSPICE's 0.13 CPU-h.** Two
  thirds of that is the transient family (66.6 %), and `ldo_2/tb_tran` alone
  is a quarter of the whole pass.

### Where the 52 disagreements are

**AC is essentially exact: 136/140.** The four misses are two known cases.
`Fan_SMC/tb_cmrr` (tsmc5, 2.3 %) is the **reference side** — a −CMRR residual
behind a 108 dB loop at 9.7 dB/mV OP sensitivity, where NGSPICE's number is a
default-tolerance early stop from the deck's own `.nodeset`; probed over six
seeds it lands on our value for four of them including no seed at all, and its
own tolerance ladder converges there at reltol ≤ 3e-4. The other three are
`Song_DACFC` on tsmc12 (`tb_gain` 0/1 with 7 metrics unmeasured, `tb_psrrn`,
`tb_psrrp`) — the same multi-OP basin that shows up in that design's `op_delta`,
now reaching the AC benches. **This is the one genuinely open item in the
table and the first thing to look at next.**

**Transient is the weak family: 11/40**, and every miss is a slew or
edge-timing metric (`sr_fall`/`sr_rise`/`t_fall`/`t_rise` and their
`_edge` variants). Depth, per design, worst over techs:

| design | verdicts | worst residual | reading |
|---|---|--:|---|
| `Leung_NMCNR` | 11/11 tsmc5, 0–1/11 elsewhere | 30.0 % | slew-limited class-A; the deepest and most tech-dependent residual in the corpus |
| `Song_DACFC` | 8/11 tsmc5, 3/11 elsewhere | 15.4 % + `v_pre` 83 % | the `v_pre` figure is the multi-OP basin, not a slew error |
| `Qu2017_AZC` | 8/11, 5/11 tsmc16 | 12.7 % | |
| `Fan_SMC` | 8/11, 5/11 tsmc5 | 6.6 % | |
| `Peng_IAC` | 8/11, 5/11 tsmc16 | 6.6 % | |
| `chargepump` | 5/6 tsmc6/7, 3/6 tsmc12, 4/6 tsmc16, **6/6 tsmc5** | `up_imin` 45.6 % | the 10 ps reversal spike; the tsmc5 tuning does not transfer to the other nodes |
| `ldo_1` / `ldo_2` | 3/5 and 2–3/5 | `overshoot` 41 % | load-step excursions; V7.5.5 probed NGSPICE's own reltol ladder moving these ±40 % and bracketing our values |

**dc_temp 34/45**, and all 11 misses are the same two metrics:
`min_slope_25_75c` and `max_step_frac_25_75c`. V7.5.4 characterized this pair
as a **reference-noise statistic, not a solver gap** — a minimum over 100
adjacent steps of a 0.5 °C staircase whose steps are ~225 µV, so one bad
reference sample sets the whole number; at reltol=1e-5 NGSPICE's own
`min_slope` goes *negative*. Read the median slope instead (it agrees to
0.5 %), and do not chase it in the solver.

**dc_source 22/30**: `lr`/`lr_pp` on `ldo_2` (39–98 %) and `ldo_1` (2.0 %), and
`lnrmax`/`lnr_ppmax` on `ldo_2` (3.4 %) — all the peak-to-peak-of-a-flat-curve
cancellation class, where `lr_pp` is a sub-mV difference of two ~0.48 V
endpoints that agree to 2e-5 each.

**Summary of what is actually open:** `Song_DACFC`'s operating-point basin
(now visible in AC), the transient slew family, and the charge pump's spike
away from tsmc5. Everything else in the table is a characterized
reference-side or cancellation artifact with its own diagnosis above.


## PyCircuitSim versus NGSPICE (V7.5.1–V7.5.3 pilot — historical)

**Status: pilot, not campaign — but the pilot now PASSES.** Seven decks of one
technology (TSMC5) have been run through both simulators; after the V7.5.1
solver work every deck's metrics agree. The 795-deck full corpus has **not**
been run; the per-family verdicts below are single-deck evidence. Nothing here
is a scoreboard number yet.

Harness: `pycircuitsim_bench/` — `translate.py` (deck to PyCircuitSim netlist),
`measure.py` (the `.meas` engine), `run_compare.py` (drives both simulators,
writes per-deck JSON). Re-audited over the V7.5.6 basket: **all 410 deck
files translate with zero failures**, over 9,530 emitted device cards, and
all 2,280 `.meas` cards in all five trees parse. (Pre-V7.5.6 full corpus:
880 decks, 19,405 device cards, 5,145 `.meas` cards.)

The engine scores **both** simulators through one code path, so a reported
difference cannot be a difference in measurement semantics. The control for that
claim is the `engine` column: the engine's reading of NGSPICE's own sweep versus
the `.meas` values NGSPICE itself printed, from the same run.

**V7.5.1 (2026-08-11).** The three causes below were run to ground and fixed in
the simulator (docs/CHANGELOG.md §V7.5.1 for the full defect list): the PyCMG
internal-node solve floored sub-nA currents to exact zero (tolerance now
NGSPICE's 1e-12 A, with a float-safe voltage-delta acceptance and a
poisoned-warm-state reset); LEVEL=72 now stamps the **full 4-terminal Newton
companion** from the condensed OSDI Jacobian, so body-junction and gate-leakage
conductances participate in NR exactly as in NGSPICE (the channel-only
gm/gds/gmb Jacobian is what locked the hot amplifier family into limit
cycles); SPICE-style damped limiting (fetlim/limvds/pnjlim), a source-referenced
eval frame, an honest wide gmin homotopy with automatic fallback, and a
transient retry that actually subdivides the interval.

**V7.5.2 (2026-08-11) closed the two follow-ups V7.5.1 left open**
(docs/CHANGELOG.md §V7.5.2): AC now stamps the **full 4-terminal
`Y = G4 + jωC4`** for L72 (the AC rows below tightened by 1–3 orders of
magnitude), and transients gained **opt-in LTE-driven output refinement**
(`PYCIRCUITSIM_BENCH_TRAN_REFINE=1`): every committed march piece is emitted
into the waveform, PULSE corners are breakpoints with a small BE restart, and
each piece is LTE-checked with rollback — so `.meas` finally sees the same
events NGSPICE's saved adaptive timepoints carry.

**V7.5.3 (2026-08-12) closed the pilot 7/7** (CHANGELOG §V7.5.3):
per-device **charge-state LTE** inside the refine mode (NGSPICE's CKTterr
shape with a device-scale CHGTOL) resolves the charge pump's reversal spike,
and the measurement/comparison layer was made NGSPICE-exact — sample-exact
`.meas` windows, NGSPICE's own sweep-grid termination rule (Kelvin-domain
accumulation for temperature sweeps), grid-matched extremum comparison for
strided sweeps, absolute floors for near-zero cancellation metrics, and the
alternate-nodeset / NGSPICE-failure recovery paths the reference flow always
had. Current pilot table (stride 1 unless noted):

| family | deck | metrics agreeing | worst node error | engine control | py / ng seconds |
|---|---|:--:|--:|:--:|--:|
| amplifier AC | `tb_gain` | **8/8** (dcgain 1.5e-07, GBW 1.0e-06) | 2.2 uV op | 4/4 | 2.7 / 0.3 |
| ldo dc source | `tb_load` | **11/11** (lr 1.8e-03 †) | ~148 uV | 11/11 | **5.2** / 0.3 |
| ldo AC | `tb_loop_max` | **8/8** (GBW 7.4e-06, was 1.4e-03) | 0.27 uV | 4/4 | 9.9 / 0.3 |
| amplifier transient | `tb_tran` (stride 4) | **11/11**, 1101/1101 steps | 3.0 uV | 11/11 | 42 / 1.1 |
| sensor dc temp | `ptat_1/tb_dc` | **13/13**, 0 mono violations | ~15 uV | 9/9 | **9.6** / 0.1 |
| amplifier dc temp | `tb_dc` (stride 25) | **15/15**, monolithic — recovery no longer needed | 63 uV | 15/15 | 67 / 2.1 |
| charge pump transient | `tb_tran` (stride 20, refine+trap) | **6/6**, up_imin at 1.8 % | -- * | 6/6 | 670 / 31 |

\* The charge pump's `op_delta` is the pre-transient operating point of a
switching circuit whose output node has no DC path — both simulators float it
differently and the transient rails it immediately; the six scored metrics are
the verdict. The sixth metric (`up_imin`) is a ±4 µA, ~10 ps current-reversal
spike. Under V7.5.1's fixed grids its amplitude was qualitatively wrong in
every configuration (BDF-2 damped it to **+3.2 µA — sign flipped**; fixed
2 ps trapezoid over-rang it 2× to −8.4 µA; NGSPICE's LTE-adaptive trapezoid
reads −4.031 µA). V7.5.2 refinement captured it at −3.84 µA (4.7 %);
V7.5.3's per-device charge-state LTE closes it: **−3.97 µA, 1.49 % at
stride 100 (94 s after the V7.5.3 PyCMG opvar memoization; 417 s before)
and 1.84 % at stride 20 (670 s pre-memoization)** — inside the 2 % gate
at both strides, the other five metrics at 4e-6…1.2e-3. The charge test is
NGSPICE's CKTterr shape with two measured deviations (CHANGELOG §V7.5.3
item 1): CHGTOL=1e-18 C — the stock 1e-14 floor sits 100× above a FinFET
terminal charge and the test never fires — and the true trapezoid error
(h³/12)|q'''| rather than CKTterr's looser factor·|DD3| (stock NGSPICE
resolves the spike through ITL4 iteration-count timestep control instead;
reading `cktterr.c` settled this).

† `lr`/`lr_pp` were excluded from evidence until V7.5.3: the 1.1 % engine-
control gap was ONE SAMPLE — NGSPICE's `.meas` windows select samples by
exact double comparison and its accumulated grid overshoots `to=0.055` by
six ulp, so NGSPICE drops the final (minimum) sample while the engine
interpolated an edge onto it. With sample-exact windows the engine control
reads 1.6e-06 and the py-vs-ng gap is 0.18 % — a real number again. The
residual is cancellation amplification (lr_pp is a 3.3e-4 difference of
two 0.477 V endpoints that agree to 1.9e-05 each), quoted as such.

The former table (for the record of what the simulator work fixed): tb_tran
2/6 at 119/221 steps, ptat 5/13 with 76 mV worst node and 67 monotonicity
violations, amplifier tb_dc 0/15 with a 9.75 V diverged first point carried
down the sweep, charge pump dead at the first timestep at every dt, tb_load at
678 s (the proper Jacobian is also a 130x speedup), and worst-node errors of
~100 uV on the passing decks that are now single-digit uV.

**Amplifier `tb_tran` category sweep** (17 designs, stride 4, refine;
single-deck evidence per design — visibility, not a scoreboard):

| config | designs fully agreeing | movement |
|---|:--:|---|
| fixed grid (flags off, V7.5.2) | 4/17 | pilot regression `Alfio` stays 11/11 |
| + output refinement (V7.5.2) | 7/17 | `Peng_ACBC/IAC/TCFC` 8→11, `Leung_DFCFC2` 5→8 |
| + charge-state LTE + altns fallback (V7.5.3) | **7/17**, net-positive shift | `Fan_SMC`/`Leung_DFCFC1`/`Yan_AZ` 5→8, `Qu2017_AZC` 0/0→**11/11**; `Leung_DFCFC2` 8→5 and `Peng_IAC` 11→8 back |

The two V7.5.3 step-backs are marginal gate-crossings, not regressions of
substance: `Leung_DFCFC2 sr_rise` moved 1.88 % → 2.45 % and `Peng_IAC
sr_fall` 1.93 % → 2.17 % across the 2 % gate — slew metrics that sat just
inside under V7.5.2's accepted-step pattern sit just outside under
charge-LTE's, while every flags-off miss those designs had is *improved*
by refine (DFCFC2 7.3 % → 4.6 %). Every remaining miss in the category is
a slew/edge-timing metric on a never-validated design; the refine cost is
1.2–3× (one pathological deck, `Qu_LEC`, pays ~30× for its 1 ns edge —
**5/11 at 944 s** after the opvar memoization, the identical verdict
V7.5.2 read at 1872 s: its six misses are genuine slew-edge residuals,
sr_fall 31 %, stable across every step-control policy tried).
`Qu2017_AZC` was diagnosed in V7.5.3 as an **NGSPICE Newton-basin
artifact**: NGSPICE fails every transient-op homotopy from the deck's
primary `.nodeset` (which sits 0.6 mV from the true OP; four of six probed
seeds INCLUDING no seed at all converge), while PyCircuitSim solves both
seeds with metrics agreeing to 1.7e-7. The corpus ships an alternate-seed
twin (`tb_tran_altns.cir`) and the reference runner falls back to it; the
bench now does the same and records which seed won (4/85 primary decks
across the five techs need it: tsmc5 Qu2017_AZC, tsmc6+tsmc7 Yan_AZ,
tsmc16 Leung_DFCFC2).

## First full-tech campaign (V7.5.3, tsmc5, 159 decks)

Run by `pycircuitsim_bench/campaign.py` (pilot stride policy: amplifier
`tb_dc` @25, `tb_tran` @4, charge pump @20 refine+trap, everything else
full grid; the transient family runs refine). Decks **fully agreeing** —
every metric both simulators measured inside the 2 % gate, no metric
missing, both engines produced data:

| family | decks fully agreeing | the misses, each diagnosed |
|---|:--:|---|
| AC (9 deck kinds) | **88/89** | `Fan_SMC/tb_cmrr` 0/1 — NGSPICE-side (see below) |
| dc_source | **14/15** | `ldo_2/tb_load` 9/11 — `lr`/`lr_pp` on a ~110 µV-flat replica-regulated curve (py holds it 16× flatter; a genuine small residual amplified by a peak-to-peak-of-flat-curve metric) |
| dc_temp | **28/31** + 1 timeout → **29/32 measurable** in V7.5.4 | 3 sensors miss only `min_slope_25_75c` (2.1–11 % — a per-step derivative at the µV node-agreement floor); `front_end_25_6T` exceeded the 1 h budget here (~11 s/point) and **is fixed in V7.5.4** — 281/281 points converged in 3.2 s, now 11/13, missing only that same `min_slope_25_75c`/`max_step_frac_25_75c` derivative pair |
| transient | **11/23** | 10 amplifiers on slew/edge metrics (above; incl. the two marginal crossings and `Qu_LEC` 5/11 — genuine 1 ns-edge residuals); `ldo_1` 3/5 and `ldo_2` 1/5 (load-step excursions — ldo_2's differ 11–57 %, a genuine open item on the corpus's delicate local-loop design); charge pump **6/6** |

The `Fan_SMC/tb_cmrr` miss is the REFERENCE side: its `cmrrdc` is a −CMRR
residual behind a 108 dB loop at 9.7 dB/mV operating-point sensitivity, and
NGSPICE's number is a default-tolerance early stop seeded by the deck's own
`.nodeset` — probed over six seeds, NGSPICE lands on PyCircuitSim's
−36.0255 dB for four of them *including no seed at all*, and its own
tolerance ladder converges there (reltol ≤ 3e-4); PyCircuitSim's answer is
seed-independent to 0.002 dB, and its AC engine reproduces NGSPICE's own
value to 2 µdB when linearised about NGSPICE's OP. One cell of 85; the deck
is untouched and the metric carries this caveat.

**All five techs, AC + DC families** (the same driver, per-tech runs;
tsmc5 additionally ran the transient family above):

| tech | AC | dc_source | dc_temp |
|---|:--:|:--:|:--:|
| TSMC5 | 88/89 | 14/15 | 28/31 (+1 timeout) |
| TSMC6 | **89/89** | 12/15 | 26/32 |
| TSMC7 | **89/89** | 12/15 | 26/32 |
| TSMC12 | **89/89** | 13/15 | 30/32 |
| TSMC16 | **89/89** | 14/15 | **31/32** |

**650 of 679 scored decks fully agree (95.7 %)**; AC is 444/445 with the
single miss being the NGSPICE-side Fan_SMC cmrrdc above. TSMC6 and TSMC7
produce **identical verdicts and identical miss relative-errors to four
decimals across all 136 decks** — the relabelled-tech control
(methodology.md §7) holding at campaign scale. Every one of the 29 misses
falls into four already-classified families: `min_slope_25_75c` (18 —
a per-0.5 °C derivative at the µV node-agreement floor),
`lr`/`lr_pp`/`lnr*` (10 — peak-to-peak-of-flat-curve cancellation, worst
on `ldo_2` everywhere), the `front_end_25_6T` cold-end cluster (its deck
completes on tsmc12/16 and passes outright on tsmc16), and the tsmc5
timeout. No new failure class appeared beyond tsmc5.

Open findings the campaign surfaced (the point of running it):

* **`front_end_25_6T` cold-end robustness — CLOSED in V7.5.4.** It was
  indeed the successor to the V7.5.1 subthreshold-floor class, and the same
  defect: PyCMG's internal-node solve accepted on an **absolute** 1e-12 A
  residual evaluated before any Newton step, so this deck's
  deep-subthreshold nulvt cores (~1e-13 A per unit device, m=360/1728) had
  their internal nodes rubber-stamped. The test is now scaled to the
  device's own terminal current (CHANGELOG §V7.5.4). Was: ~11 s/point,
  3/7 probe points non-converged, 8 mV cold error, >1 h budget overrun.
  Now: **281/281 points converged in 3.2 s, worst node 0.28 mV, 11/13** —
  the two remaining misses are the `min_slope_25_75c` /
  `max_step_frac_25_75c` derivative pair (the metric class below), and the
  seed bisection that diagnosed it now returns the same answer from the
  deck's nodesets and from NGSPICE's own solution. Answer-preserving on
  every other deck: all 15 previously scored tsmc5 `sensing_front_end` /
  `voltage_reference` decks are verdict- and op_delta-identical.
  **Cross-tech, and the metric set had been hiding the size of it:** on
  tsmc16 all 14 sensing decks stay verdict-identical while this deck's
  `op_delta` falls **49.29 mV → 0.249 mV** (it was scoring 13/13 while
  sitting 49 mV off NGSPICE's operating point — the `.meas` cards simply do
  not probe the node that was wrong); on tsmc12, 13 of 14 unchanged and
  this deck goes **8/13 → 11/13** with `op_delta` 9.97 mV → 0.303 mV. This
  is the case for `op_delta` being a first-class harness output rather than
  trusting `_last_solve_converged` or the metric columns alone.
* **`Basic_LDO/tb_tran` refine cost — STILL OPEN, and re-diagnosed in
  V7.5.4.** Two corrections to the V7.5.3 record. (a) The verdict under
  refine is **4/5, not 5/5**: overshoot reads 1.11 mV against NGSPICE's
  3.04 mV (rel 0.634). (b) The march does not park at ~1 ns — measured over
  0–6 µs (gear2, dt=20 ns) it runs 100 pieces at the full 20 ns grid before
  the load step, 991 pieces across 2.0–2.5 µs, then **12 868 pieces at
  median 47 ps** for the remaining 3.5 µs with **zero** NR failures. The
  mechanism is a **dead zone in the growth law**, not a noise-limited
  estimator: `min(2, max(1, 0.9·r^(-1/3)))` exceeds 1 only when r < 0.9³ =
  0.729, so any accepted ratio in [0.729, 1) freezes dt exactly. The V7.5.3
  fix path was implemented and **reverted**: it holds the charge pump at 6/6
  (both strides) but costs 3.3× more (630 s → 2067 s, 19 897 → 78 298
  pieces) and flips no verdict, though it does improve overshoot 8.5×
  (rel 0.634 → 0.074) — which also means refine at HEAD **under-resolves
  this deck ~4×** on that metric. Moving the safety factor inside the
  exponent shrinks the dead zone to [0.9, 1) but is worth only ~6 % in dt.
  **CLOSED in V7.5.5** — controller rebuilt on ngspice dctran semantics
  with a legacy corner guard (see the V7.5.5 section below and CHANGELOG
  §V7.5.5.1; the ~4×-under-resolving claim was itself calibrated against a
  reference-unstable overshoot and is softened there).

* **`min_slope_25_75c` — CHARACTERIZED in V7.5.4 as a reference-noise
  statistic, not a solver gap.** It is a *minimum over 100 adjacent steps*
  of a 0.5 °C staircase whose steps are ~225 µV, so a single bad sample
  sets the whole number. On `front_end_25_6T` NGSPICE's own curve moves
  only **83.8 µV** across 31.5→32.0 °C where its neighbours move
  234/226/279 µV — a one-sample ~100 µV wobble in the reference's own DC
  solution, which drops its `min_slope` to 1.68e-4 against our smooth
  4.27e-4. The `Fan_SMC/tb_cmrr` treatment (re-run the reference tighter)
  does **not** rescue it: at `reltol=1e-5` NGSPICE's `min_slope` goes
  *negative* (−4.08e-4 — its curve becomes locally non-monotone), so the
  quantity is not tolerance-stable on the reference side at all. The
  **median** slope, the same physical property robustly estimated, agrees
  to **0.5 %** across all three runs (4.485e-4 default, 4.461e-4 at
  reltol=1e-5, ours ~4.4e-4); `mono_violations` is 0 on both sides and the
  281-point operating point agrees to 0.28 mV worst over 1124 node
  comparisons. Our curve is additionally smoothed by per-point continuation
  seeding (NGSPICE continues too, but converges each point only to its own
  default tolerance). **Quote this pair with the caveat and read the median
  slope instead; do not chase it in the solver, and do not "fix" it by
  tightening the reference.** This is the campaign's largest miss family
  (18 decks).

Representative agreements where the two simulators do match:

| deck | metric | PyCircuitSim | NGSPICE | rel. error |
|---|---|--:|--:|--:|
| `tb_gain` | dcgain (dB) | 75.8749 | 75.8719 | 3.9e-05 |
| `tb_gain` | GBW (Hz) | 541384 | 541380 | 6.9e-06 |
| `tb_gain` | pm_true (deg) | 55.3411 | 55.3411 | 3.1e-07 |
| `tb_load` | vout_max (V) | 0.476883 | 0.476874 | 1.9e-05 |
| `tb_load` | ivdd_max (A) | -0.055017 | -0.055017 | 7.9e-09 |
| `tb_loop_max` | GBW (Hz) | 1.04792e7 | 1.04647e7 | 1.4e-03 |
| `tb_tran` | v_high (V) | 0.228067 | 0.228067 | 4.8e-07 |
| `ptat_1/tb_dc` | vout50 (V) | 0.0716231 | 0.0716246 | 2.1e-05 |

The AC path is essentially exact: linearised about NGSPICE's *own* operating
point it returns dcgain within 0.004 dB, GBW ratio 1.0000 and phase within
0.0001 deg. **Every failure in the former table was the DC operating point**;
the three recorded causes are all FIXED in V7.5.1 (each unwound into more than
one defect — the full list with measurements is docs/CHANGELOG.md §V7.5.1):

1. **Subthreshold current floor — FIXED.** The `id = 0.0 A`-below-~60 mV cliff
   was the PyCMG internal-node solve accepting its warm-started entry state
   whenever the true current sat below its 1e-9 A absolute residual tolerance.
   The tolerance is now NGSPICE's own 1e-12 A, backed by a voltage-delta
   acceptance that only fires after a Newton step was actually taken (which is
   what keeps sub-nA currents resolving without re-opening the floor) and a
   reset of the poisoned internal state a diverged solve used to leave behind.
   `ptat_1/tb_dc` went 5/13 → 13/13 with zero monotonicity violations.
2. **Newton start at 125 C — FIXED, and it was the model's Jacobian, not just
   the start.** The channel-only gm/gds/gmb opvars are blind to the
   body-junction conductances that carry the drain current at high temperature
   (measured: id = +1.8 mA against gds = 4.3e-13 S), so no amount of homotopy
   converged — NR cycled around a KCL violation its Jacobian could not see.
   With the full 4-terminal OSDI-Jacobian stamp the 125 C cold start converges
   in 1.8 s of plain NR. Where the fork at 125.0 C genuinely has two Newton
   roots, the outward-from-25 C recovery (now actually triggerable — its gate
   read a key the op-delta never carried) reconciles both simulators onto one
   branch: 15/15 with a 2.6 uV worst node.
3. **Missing per-terminal limiting — FIXED, two layers deeper than expected.**
   SPICE-style damped limiting (fetlim/limvds/pnjlim, evaluation at the
   limited bias, linearization about it) now exists in `mosfet_cmg.py`; the
   OSDI eval frame is source-referenced because the internal solve is not
   shift-robust (identical pairs evaluate at s=-8.5 V and diverge at
   s=-16.1 V); and the transient retry ladder now cuts the LOCAL timestep and
   marches the interval instead of stiffening the cap companions against a
   fixed target time — the "minimum dt" the charge pump died at was never a
   smaller step at all. The amplifier transient runs 1101/1101 steps at 11/11.

Measurement caveats that belong with the numbers above:

* The amplifier `tb_dc` row still runs at `--stride 25` (67 of NGSPICE's 1650
  points); the MAX/MIN/AVG keys agree anyway because the fork recovery puts
  both simulators on the same branch and the sweep tracks NGSPICE to 63 µV
  per shared point. `tb_tran` runs at stride 4 (20 ns on a 5 ns deck); its
  slew metrics agree at that grid.
* `ok` (per-point solver flag) is reported, not obeyed — though after V7.5.1
  the flag and the values finally coincide on every pilot deck (ptat's 14
  flagged points and the LDO sweep's 101 are all flag-converged now). The
  harness still defaults to NGSPICE's RELTOL 1e-3 / VNTOL 1e-6 rather than
  PyCircuitSim's 10x-tighter 1e-4 / 1e-7 (a deck's own `.options` wins).
* A converged flag is not sufficient evidence anywhere in this comparison: the
  historical diverged `tb_dc` sweep produced a full set of plausible-looking
  numbers (a `tc` of 0.0139 against NGSPICE's 0.0044). Node voltages, not
  metrics, distinguish a converged sweep from a diverged one, which is why
  every row carries a worst-node-error column — and why V7.5.1 also fixed the
  solver's own flag semantics (an intermediate gmin level could mark a
  diverged final level converged).

**Extrapolated campaign cost** — from measured per-deck rates, not measured
end to end: 795 scored decks (445 AC, 160 dc_temp, 75 dc_source, 115
transient). The V7.5.1 rates change the picture materially: dc_source decks
dropped 678 s → 5.2 s and dc_temp 1250 s → 67 s at stride 25 (~3 s/point), so
the DC families now project to single-digit CPU-hours; amplifier transients at
their own dt remain the dominant, least-constrained term. On the pilot
evidence the three former blocker causes are gone; what remains untested at
campaign scale is untested, not distrusted.


## Transient family, all five techs (V7.5.5)

The V7.5.5 refine controller (dctran-exact open water + a legacy corner
guard, CHANGELOG §V7.5.5.1) closed the LDO-cost blocker, so the transient
family now has campaign coverage on every tech, in both modes. Same driver,
same stride policy (amplifier/ldo `tb_tran` @4, charge pump @20 refine+trap).

| tech | flags-off | refine-on |
|---|:--:|:--:|
| TSMC5 | — (V7.5.3 ran refine only) | **10/23** |
| TSMC6 | 1/23 | **8/23** |
| TSMC7 | 1/23 | **8/23** |
| TSMC12 | 6/23 | **8/23** |
| TSMC16 | 2/23 | **9/23** |

Refine is net-positive on every tech, and **TSMC6 ≡ TSMC7 verdict-identical
across all 23 decks in both modes** — the relabelled-tech control extends to
the transient family. Per-deck notes that belong with these numbers:

* **tsmc5 vs the V7.5.3 rows, per deck:** `ldo_2` 1/5 → 3/5 at 3 006 s →
  214 s; `Basic_LDO` at its honest (V7.5.4-corrected) 4/5, 563 s → 340 s;
  both V7.5.3 marginal slew regressions fixed (`Leung_DFCFC2` 5/11 → 11/11,
  `Peng_IAC` 8/11 → 11/11); four decks picked up NEW 2.2–2.7 % slew
  crossings (`Fan_SMC`, `Peng_ACBC`, `Peng_TCFC`, `Yan_AZ`) — shallow-margin
  scatter at the 2 % gate where the V7.5.3 misses ran 4.6–7.3 % deep.
  Amplifiers fully agreeing: 7/17 in both passes, different composition,
  at 3–7× less cost per deck (family ~4 090 s → ~1 155 s).
* **`ldo_2`'s two residual misses and `Basic_LDO`'s overshoot are
  reference-tolerance artifacts** (probed in V7.5.5): NGSPICE's own reltol
  ladder moves each by ±40 % or more, brackets our values, and its default
  run is demonstrably unsettled (tb_tran pre-step average 1.4 mV above its
  own DC; tb_load's 110 µV `lr_pp` collapses onto ours when the SAME deck
  is swept in reverse). The loop itself agrees 8/8 on both AC benches.
* **`Song_DACFC` solves to an operating point 0.30–0.50 V from NGSPICE's on
  tsmc6/7/12/16** while agreeing to 9.6e-05 V on tsmc5 — a basin/multi-OP
  difference on a never-validated design (`Sau_CFCC` shows a 0.07 V version
  on tsmc6 only). Flagged, not scored as a solver defect.
* **tsmc12 charge pump under refine: 3/6** — the first cp number on any
  tech but tsmc5 (the gate deck has always been tsmc5-pinned). Coverage,
  not regression.

Evidence: `pycircuitsim_bench_results/v755_campaign_tsmc{6,7,12,16}_tran/`
(flags-off) and `v755_campaign_tsmc{5,6,7,12,16}_tran_refine/` (refine-on),
untracked, on disk.

## Audit validation

Re-run over the V7.5.9 basket (`verify_tsmc_sizing.py`, 0 problems):

| check | result |
|---|---:|
| generated designs simulated | 60/60 |
| source/qualified-core topology checks | 60/60 |
| generated MOS instances checked | 1,205 |
| sizing vectors inside modelcard envelopes | 604/604 |
| referenced local PyCMG model aliases valid | 456/456 |

Both earlier corpora audited clean at the same rate: V7.5.6's basket 90/90
designs, 1,695 MOS, 779/779 vectors, 612/612 aliases; the original full corpus
190/190 designs, 3,240 MOS, 1,417/1,417 vectors, 1,155/1,155 aliases.

Topology checks cover MOS/passive connectivity, channel type, amplifier mirror
ratios, the retained charge-pump hierarchy, the permitted `ldo_1` compensation
network, and the subthreshold reference core. Geometry checks cover L, NFIN,
multiplicity, local model definition, and every model used by a generated MOS.

## Corrections made in this audit

* NGSPICE analysis failures are now fatal even when NGSPICE exits with status
  zero; partial AC/DC/transient runs can no longer be reported as successes.
* `ldo_1` received a per-node series-RC compensation network and re-sized bias,
  driver, and pass devices. It now passes all 9 LDO gates on all five nodes.
* The LDO bias-source polarity now follows the actual NMOS/PMOS diode topology;
  the previous common polarity drove `ldo_1`/`ldo_2` bias nodes below ground.
* LDO verdicts now include line/load regulation, both-load PSRR, and load-step
  excursions; the transient excursion measurement now uses the actual pre-step
  output and cannot report a negative overshoot.
* Amplifier verdicts now include PSRR-, temperature coefficient, and both slew
  directions. Wide temperature sweeps retry as two continuations from 25 C,
  and slew crossings are restricted to the commanded input-edge windows.
* `three_output_vref` now uses the qualified dual-output core and derives a
  third low-load output at half `vref2`; all 15 derived outputs meet their
  voltage and temperature gates. The third output is intentionally high
  impedance and is not qualified for load drive.
* `ldo_2`'s error amp again regulates the exposed replica node (the source
  bench ties `vinp` to `vfb`); the delivered output keeps its own local loop
  whose setpoint derives from the replica. Closing the error amp around the
  output instead double-drove the output with two different setpoints and the
  transient walked off the DC operating point. Every gate still measures the
  voltage delivered to the load, and the load step integrates with Gear.
* LDO line-regulation sweeps run as two DC continuations from the nominal
  rail outward and are recombined; a monolithic sweep started Newton cold at
  the worst-headroom corner and jumped onto non-regulating branches that no
  reachable operating point has.
* Stability margins are now the true wrap-aware margin (`pm_true`): the AC
  phase is unwrapped across the dumped sweep and the margin is the unwrapped
  distance from the instability point at the 0 dB crossover. Lead-compensated
  loops whose feedforward zeros recover phase legitimately report margins
  above 90 degrees; a loop whose unwrapped phase has actually fallen through
  -180 before crossover now reports a negative margin and fails. This
  replaced the earlier magnitude-of-principal-value reading, which could not
  tell those two cases apart: it exposed one amplifier whose AC solution was
  numerically pathological (`Peng_IAC`, re-tuned) and one LDO loop that was
  genuinely unstable at minimum load behind a +54-degree wrapped reading
  (`Basic_LDO`, re-compensated). GBW has a single definition everywhere: the
  0 dB crossover frequency of the measured sweep.
* The per-tree summary artifacts (`run_all.json`, `summary.csv`) are now
  produced by the same measurement code, metric definitions, and convergence
  ladder as the per-design verdicts, so a tree can no longer ship two
  contradicting sets of numbers for the same design.
* Sensor verdicts now check monotonicity over every solved point of the full
  temperature sweep and require a smooth response inside the 25-75 C
  sensitivity window (a minimum local slope and no single step contributing
  most of the rise), so a discontinuous operating-branch jump can no longer
  masquerade as sensitivity. Sweeps that need per-point pseudo-transient
  rescues are flagged as errors. Seven staircase sensors were re-seeded onto
  a single operating branch across the five nodes under these gates.
* `SMCNR_SE_2st_AMP`, the amplifier shipped inside AnalogGym's sensing front
  end category (previously skipped by the sensor flow), is now generated and
  gated on all five nodes with its own AC bench (gain, GBW, true PM, power).
  The two BJT-based references remain out of scope: `bandgap_vref` needs an
  NPN and `subthreshold_vref` a PNP, and the PyCMG compact-model library is
  FinFET-only.
* `Basic_LDO` carries a per-device Vt override (ULVT pass transistor only):
  the SVT pass device could not hold the max-load line sweep at low rail, and
  a whole-design ULVT swap mis-biased the error amp. `ldo_folded_cascode` on
  the 0.65/0.75 V rails was re-seeded with a matched mirror/cascode/sink
  bias plan that keeps the folded branch saturated across the +/-10 % line
  window; the shipped random-search geometry had its PMOS mirror in deep
  triode and lost the regulating branch a few tens of mV above nominal.
* The amplifier slew bench seeds its transient at the pulse baseline rather
  than the common mode (the pre-edge settle was itself a slew event), with a
  common-mode-seeded fallback deck for designs whose transient operating
  point only solves from there. Slew-limited class-A amplifiers
  (`Leung_NMCNR`) got their free bias current raised inside the topology's
  fixed mirror ratios instead of failing the slew gate.

The audit is nominal/TT compact-model verification across the stated
temperature sweeps, not statistical mismatch or full foundry-corner signoff.


## Fully-passing designs per tech

| category | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|--:|--:|--:|--:|--:|
| amplifier | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| ldo | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| sensing_front_end | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| voltage_reference | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| charge_pump | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| **total** | **12/12** | **12/12** | **12/12** | **12/12** | **12/12** |


## Gates passed, per design and tech


### amplifier

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| Fan_SMC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Leung_NMCNR_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Peng_IAC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Qu2017_AZC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Song_DACFC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |


### ldo

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| ldo_1 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| ldo_2 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |


### sensing_front_end

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| front_end_25_6T_schematic | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_1 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |


### voltage_reference

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| dual_output_subthreshold_vref | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |


### charge_pump

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| chargepump | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |


## Amplifier medians per tech

| tech | VDD | gain (dB) | GBW | power | PM (deg) |
|---|--:|--:|--:|--:|--:|
| TSMC5 | 0.65 V | 105.6 | 561 kHz | 15.3 uW | 69.8 |
| TSMC6 | 0.75 V | 117.1 | 744 kHz | 20.6 uW | 68.2 |
| TSMC7 | 0.75 V | 117.1 | 744 kHz | 20.6 uW | 68.2 |
| TSMC12 | 0.80 V | 112.7 | 633 kHz | 21.8 uW | 55.8 |
| TSMC16 | 0.80 V | 114.1 | 1.17 MHz | 21.8 uW | 59.3 |

These are medians over the **5 basket amplifiers** (V7.5.6: 7, originally 17).
Only TSMC12 moved from the V7.5.6 table — GBW 856 → 633 kHz and PM 82.9 →
55.8° — because dropping two of seven designs moves which one sits at the
median. **No design's own numbers changed**, on any tech; the V7.5.9 campaign
re-measured all 255 surviving decks and reproduced every pre-prune verdict and
every miss magnitude to six decimal places.


## Remaining partial designs

Every generated design now passes every gate on every node with no simulator analysis errors. The table below lists any design that regresses in a future re-run.


| tech | category/design | failed gates | analysis errors |
|---|---|---|--:|
| -- | all designs | -- | 0 |
