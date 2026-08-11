# AnalogGym across TSMC FinFET nodes

This is a strict NGSPICE/PyCMG audit of every generated design. Topologies are checked against their source (or, for the derived three-output reference, its qualified two-output core), dimensions are constrained to each modelcard, and every available DC, AC, temperature, PSRR, line/load-regulation and transient gate is included. A result is fully passing only when every gate passes and the simulator reports no analysis error. `!` marks a partial result with one or more simulator/evaluator errors. Per-tree details: `designs_<tech>/RESULTS.md`.

Since the migration into PyCircuitSim (`examples/analoggym/`, V7.5.0) this file
carries **two** axes. The next section scores **PyCircuitSim against NGSPICE**
on the identical BSIM-CMG (LEVEL=72) OSDI model. Everything after it is the
original NGSPICE/PyCMG design audit, unchanged — that audit is the reference the
comparison is measured against, not a second opinion about it.


## PyCircuitSim versus NGSPICE

**Status: pilot, not campaign.** Seven decks of one technology (TSMC5) have been
run through both simulators. The 795-deck full corpus has **not** been run; the
per-family verdicts below are single-deck evidence and the cost figures are
extrapolations from measured rates, labelled as such. Nothing here is a
scoreboard number yet.

Harness: `pycircuitsim_bench/` — `translate.py` (deck to PyCircuitSim netlist;
all 880 decks translate, audited over 19,405 emitted device cards),
`measure.py` (the `.meas` engine; all 5,145 cards in all five trees parse),
`run_compare.py` (drives both simulators, writes per-deck JSON).

The engine scores **both** simulators through one code path, so a reported
difference cannot be a difference in measurement semantics. The control for that
claim is the `engine` column: the engine's reading of NGSPICE's own sweep versus
the `.meas` values NGSPICE itself printed, from the same run.

| family | deck | metrics agreeing | worst node error | engine control | py / ng seconds |
|---|---|:--:|--:|:--:|--:|
| amplifier AC | `tb_gain` | **8/8** | 110 uV | 4/4 | 2.2 / 0.3 |
| ldo dc source | `tb_load` | **11/11** | 148 uV | 11/11 * | 678 / 0.4 |
| ldo AC | `tb_loop_max` | **8/8** | 91 uV | 4/4 | 8.1 / 0.3 |
| amplifier transient | `tb_tran` | 2/6, 119 of 221 steps | 111 uV | 11/11 | 105 / 1.1 |
| sensor dc temp | `ptat_1/tb_dc` | 5/13 | 76 mV | 9/9 | 74 / 0.1 |
| amplifier dc temp | `tb_dc` | 0/15 | 9.75 V | 15/15 | 1250 / 0.7 |
| charge pump transient | `tb_tran` | dies at first step | 161 mV | 6/6 | -- / 30.7 |

\* `lr` and `lr_pp` are the one exception to the engine control: they agree with
NGSPICE's own `.meas` only to 1.1e-02, where every other metric on that deck
agrees to 1e-08 or better. PyCircuitSim's `lr` (0.0137316) is in fact *closer* to
NGSPICE's own `.meas` (0.0137973) than the engine's reading of NGSPICE's sweep
is (0.0139486). The 1.6 % `lr` gap is therefore mostly a load-regulation
measurement-definition difference, not a simulator difference, and those two
keys should not be quoted as accuracy evidence until reconciled.

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
0.0001 deg. **Every failure above is the DC operating point**, and the causes are
three distinct problems, not one:

1. **Subthreshold current floor — blocks `sensing_front_end` + `voltage_reference`
   (75 decks).** At NGSPICE's own operating point PyCircuitSim's LEVEL=72 returns
   `id = 0.0 A` *exactly* at Vgs = 55.3 mV, and 2.16 nA at Vgs = 61.9 mV. A
   weak-inversion stack whose devices all read zero current has no operating
   point to find. This is visible in the `ptat_1` row as a split verdict: above
   ~50 C the same deck agrees to 2e-05 (`vout50` 2.1e-05, `vout100` 8.2e-05),
   while `vout25` is 59 % off and the sweep reports 67 monotonicity violations
   against NGSPICE's zero. The floor is m-independent (m=1 and m=4 both return
   hard zero, and above 0.08 V they agree to 0.00 % and scale exactly 4x), so it
   is not the instance multiplier — it is an OSDI/PyCMG evaluation gap below
   ~60 mV Vgs, and it is not fixable in the harness.
2. **Newton start, not the model — amplifier `tb_dc` (85 decks).** That bench
   sweeps 125 C to -40 C and PyCircuitSim diverges at the *first* point: zero of
   67 points converged, with failures beginning at exactly 125.0 C and walking
   down 122.5, 120.0, 117.5, 115.0 as the continuation carries the garbage. The
   identical solve at 25 C is sound (node error 114 uV, `vout6` 0.193383 against
   NGSPICE's 0.193383), and the same subcircuit's `tb_gain` deck agrees to
   3.9e-05. `compare_with_recovery` ports `finalize.py`'s outward-from-25 C
   recovery for this; it is implemented but its end-to-end numbers are **not yet
   measured**, so 0/15 stands as this family's recorded result.
3. **Missing per-terminal limiting — charge pump dead (5 decks), transients
   partial (~110 decks).** The `dv_limit` trust region bounds the Newton *step*,
   not the terminal voltage, so capped iterations still walk a gate to -2.96 V
   and a drain to +3.94 V on a 0.65 V rail and OSDI rejects the point. The charge
   pump dies at the first timestep at every dt tried (2 ps, 20 ps, 200 ps, 1 ns).
   Transients are not uniformly broken, though: the amplifier `tb_tran` deck
   commits 119 of 221 steps and its settled levels match to 4.8e-07 before NR
   exhausts on the falling slew edge. The fix is SPICE-style `fetlim`/`limvds`
   inside `models/mosfet_cmg.py`, which does not exist — and it must be damped
   limiting, not a hard clamp, which would zero the derivative and stall NR.

Measurement caveats that belong with the numbers above:

* The amplifier `tb_dc` row was run at `--stride 25`, so PyCircuitSim measured
  67 points against NGSPICE's 1650 — the grids do **not** match and its
  MAX/MIN/AVG keys are not a like-for-like comparison. The failure itself is
  independently established by `flag_ok = 0`, not by those metrics. Full-stride
  would be ~740 h for the family, which is why the recovery path matters.
* `tb_tran`'s slew metrics are 21 % off because `--stride 20` asks for a 100 ns
  timestep on a deck written for 5 ns. That is the stride, not the simulator.
* `ok` (per-point solver flag) is reported, not obeyed. On the LDO sweep
  PyCircuitSim matched NGSPICE to 2e-05 while the convergence flag rejected all
  101 points under PyCircuitSim's native tolerances, so the operating-point
  delta is the verdict. The harness therefore defaults to NGSPICE's RELTOL 1e-3
  / VNTOL 1e-6 rather than PyCircuitSim's 10x-tighter 1e-4 / 1e-7 (a deck's own
  `.options` still wins), which took that sweep to 101/101 flag-converged with
  identical values.
* A converged flag is not sufficient evidence anywhere in this comparison: the
  diverged `tb_dc` sweep produces a full set of plausible-looking numbers
  (a `tc` of 0.0139 against NGSPICE's 0.0044). Node voltages, not metrics,
  distinguish a converged sweep from a diverged one, which is why every row
  carries a worst-node-error column.

**Extrapolated campaign cost** — from measured per-deck rates, not measured
end to end: 795 scored decks (445 AC, 160 dc_temp, 75 dc_source, 115 transient)
at roughly 110 CPU-hours, parallelisable per deck. Amplifier transients at their
own dt (~90 h) dominate that and are the least-constrained estimate; amplifier
dc_temp is ~740 h as it stands, projected ~2 h if the recovery path restores the
25 C pass's 3.1 s/point rate. On present evidence about 520 of 795 decks would
produce trustworthy metrics, with the three causes above accounting for the rest.


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
