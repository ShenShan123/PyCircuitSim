# AnalogGym across TSMC FinFET nodes

This is a strict NGSPICE/PyCMG audit of every generated design. Topologies are checked against their source (or, for the derived three-output reference, its qualified two-output core), dimensions are constrained to each modelcard, and every available DC, AC, temperature, PSRR, line/load-regulation and transient gate is included. A result is fully passing only when every gate passes and the simulator reports no analysis error. `!` marks a partial result with one or more simulator/evaluator errors. Per-tree details: `designs_<tech>/RESULTS.md`.

Since the migration into PyCircuitSim (`examples/analoggym/`, V7.5.0) this file
carries **two** axes. The next section scores **PyCircuitSim against NGSPICE**
on the identical BSIM-CMG (LEVEL=72) OSDI model. Everything after it is the
original NGSPICE/PyCMG design audit, unchanged — that audit is the reference the
comparison is measured against, not a second opinion about it.


## PyCircuitSim versus NGSPICE

**Status: pilot, not campaign — but the pilot now PASSES.** Seven decks of one
technology (TSMC5) have been run through both simulators; after the V7.5.1
solver work every deck's metrics agree. The 795-deck full corpus has **not**
been run; the per-family verdicts below are single-deck evidence. Nothing here
is a scoreboard number yet.

Harness: `pycircuitsim_bench/` — `translate.py` (deck to PyCircuitSim netlist;
all 880 decks translate, audited over 19,405 emitted device cards),
`measure.py` (the `.meas` engine; all 5,145 cards in all five trees parse),
`run_compare.py` (drives both simulators, writes per-deck JSON).

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
1.2–3× (one pathological deck, `Qu_LEC`, pays ~30×+ for a 1 ns edge —
V7.5.2 refine read 5/11 at 1872 s; under charge-LTE it exceeds the
campaign's 1 h deck budget, re-run pending).
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
| dc_temp | **28/31** + 1 timeout | 3 sensors miss only `min_slope_25_75c` (2.1–11 % — a per-step derivative at the µV node-agreement floor); `front_end_25_6T` exceeds the 1 h budget (~11 s/point, open solver finding below) |
| transient | **11/22** + 1 over budget | 10 amplifiers on slew/edge metrics (above; incl. the two marginal crossings), `Qu_LEC` over the 1 h budget under charge-LTE; `ldo_1` 3/5 and `ldo_2` 1/5 (load-step excursions — ldo_2's differ 11–57 %, a genuine open item on the corpus's delicate local-loop design); charge pump **6/6** |

The `Fan_SMC/tb_cmrr` miss is the REFERENCE side: its `cmrrdc` is a −CMRR
residual behind a 108 dB loop at 9.7 dB/mV operating-point sensitivity, and
NGSPICE's number is a default-tolerance early stop seeded by the deck's own
`.nodeset` — probed over six seeds, NGSPICE lands on PyCircuitSim's
−36.0255 dB for four of them *including no seed at all*, and its own
tolerance ladder converges there (reltol ≤ 3e-4); PyCircuitSim's answer is
seed-independent to 0.002 dB, and its AC engine reproduces NGSPICE's own
value to 2 µdB when linearised about NGSPICE's OP. One cell of 85; the deck
is untouched and the metric carries this caveat.

Open findings the campaign surfaced (the point of running it):

* **`front_end_25_6T` cold-end robustness** — ~11 s/point at full stride
  with 3/7 stride-50 probe points flagged non-converged (values kept), and
  a genuine 2–8 mV disagreement at the cold end of the sweep. The successor
  to the closed V7.5.1 subthreshold-floor class: a cold sensor OP the
  solver only holds approximately. Open.
* **`Basic_LDO/tb_tran` refine cost** — 2684 s against 28 s flags-off
  (96×), for a **5/5** verdict that captures the 3 mV load-step overshoot
  flags-off reads as exactly zero (and the gain is march accuracy, not
  output density: +0.61 mV of the recovered undershoot vs +0.28 mV).
  Diagnosed (CHANGELOG §V7.5.3 item 10): 67 % of the wall was a PyCMG
  descriptor-scan tax on every L72 transient — fixed bit-identically in
  V7.5.3 (~3×) — and the rest is a stable ~1 ns equilibrium in the
  refine step controller (its shrink/grow laws assume the LTE ratio
  falls as h³; measured h^1.05 because the estimator is noise-limited
  at these step sizes). Controller fix path recorded; needs a
  charge-pump re-gate before it ships.

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


## Audit validation

| check | result |
|---|---:|
| generated designs simulated | 190/190 |
| source/qualified-core topology checks | 190/190 |
| generated MOS instances checked | 3,240 |
| sizing vectors inside modelcard envelopes | 1,417/1,417 |
| referenced local PyCMG model aliases valid | 1,155/1,155 |

Topology checks cover MOS/passive connectivity, channel type, amplifier mirror
ratios, the retained charge-pump hierarchy, the permitted `ldo_1` compensation
network, and the derived reference core. Geometry checks cover L, NFIN,
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
| amplifier | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 |
| ldo | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| sensing_front_end | 13/13 | 13/13 | 13/13 | 13/13 | 13/13 |
| voltage_reference | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| charge_pump | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| **total** | **38/38** | **38/38** | **38/38** | **38/38** | **38/38** |


## Gates passed, per design and tech


### amplifier

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| Alfio_RAFFC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Fan_SMC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| HoiLee_AFFC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Leung_DFCFC1_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Leung_DFCFC2_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Leung_NMCF_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Leung_NMCNR_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Peng_ACBC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Peng_IAC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Peng_TCFC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Qu2017_AZC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Qu_LEC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Ramos_PFC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Sau_CFCC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Song_DACFC_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Tan_CLIA_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Yan_AZ_Pin_3 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |


### ldo

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| Basic_LDO | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| ldo_1 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| ldo_2 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| ldo_folded_cascode | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| ldo_simple | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |


### sensing_front_end

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| PTAT_65_classic1 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| PTAT_CLASSIC | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| PTAT_SENSOR | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| SMCNR_SE_2st_AMP | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| front_end_11_6T_schematic | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| front_end_25_6T_schematic | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| front_end_31_3T_schematic | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| front_end_42_2_2015_REF_schematic | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_1 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_2 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_3 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| ptat_6 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |


### voltage_reference

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| dual_output_subthreshold_vref | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| three_output_vref | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |


### charge_pump

| design | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|:--:|:--:|:--:|:--:|:--:|
| chargepump | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |


## Amplifier medians per tech

| tech | VDD | gain (dB) | GBW | power | PM (deg) |
|---|--:|--:|--:|--:|--:|
| TSMC5 | 0.65 V | 93.7 | 541 kHz | 15.4 uW | 56.9 |
| TSMC6 | 0.75 V | 117.1 | 674 kHz | 20.9 uW | 68.0 |
| TSMC7 | 0.75 V | 117.1 | 674 kHz | 20.9 uW | 68.0 |
| TSMC12 | 0.80 V | 113.6 | 601 kHz | 13.1 uW | 57.5 |
| TSMC16 | 0.80 V | 113.2 | 618 kHz | 21.8 uW | 61.0 |


## Remaining partial designs

Every generated design now passes every gate on every node with no simulator analysis errors. The table below lists any design that regresses in a future re-run.


| tech | category/design | failed gates | analysis errors |
|---|---|---|--:|
| -- | all designs | -- | 0 |
