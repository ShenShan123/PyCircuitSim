# V7.7.5 checkpoints — targeted round 2

Run: 2026-09-15 to 2026-09-18; published 2026-09-21.

**Status: scoped diagnostic, not a clean qualification.** Round 2 follows
[round 1](v775-quick-round1.md) on the same 80 V7.7.2-trained bundles, using
the scope the user selected after that round. It keeps the nominal corner,
does not rescore round-1 cells, and does not replace
[`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) or
[`BSIM-AR-L76-clean.md`](BSIM-AR-L76-clean.md). Gate and denominator rules are
owned by [`methodology.md`](methodology.md).

**Verified 2026-09-25.** An independent parser re-read all 1,080 raw verdict
logs and the support-diagnostic outputs. It reproduced the verdict totals,
every convergence table, the OMP strictness counts, the per-technology tables,
all four support-pass record and value counts, and every manifest and
job-list digest. These statements were corrected in place: support-rejection
coverage (35 of 100 rows were never checked), the NAND2 spike range, the LDO
and Miller slope values, the sign of the inverter-energy error on
TSMC12/16, four BSIM-AR error labels, the SRAM failure mode, the hold-margin
median, the runtime section, and the NRMSE comparison (two technologies, not
three).

Round 2 is complete: 1,080/1,080 cells, 913 scientific PASS and 167 FAIL, with
no cell labelled `infra`. Every error row in both families is attributed to the
candidate model, and the NGSPICE reference converged in every row.

## Scope

| part | cells | question |
|---|---:|---|
| 23 remaining simple-v2 cases, nominal, OMP=1 | 920 | circuit behavior beyond the round-1 subset, L1–L4 |
| Ring oscillator and Miller DC gates at OMP 2 and 4 | 160 | strictness: with round 1's OMP=1, a pass needs all three thread counts |
| Reference-support diagnostic | 440 runs | do round-1 and round-2 support rejections come from the physical trajectory? |

The two harness defects found in round 1 were fixed before any round-2 cell
ran (below).

## Harness fixes (`d708d4b`)

- **Derived rows inherit the upstream error kind.** A missing CMRR/PSRR row now
  takes the kind of the input analysis that failed. An unconverged candidate
  stays a scientific `ERROR` with exit 1, not a `result_schema` row that
  dispatched as infrastructure. Converged inputs that still lack a finite gain
  remain `result_schema`.
- **SRAM corner errors name the failing engine.** `verify_circuit_sram_snm.py`
  tags each corner error with its stage and derives `reference_converged` and
  `error_kind` from that tag.

Regression tests for both fail on `39b14f1` and pass on `d708d4b`. The full
suite passed 1,160 tests with none skipped. In round 2, an `opamp_rejection`
derived row with missing inputs is recorded as `candidate`, and no cell has
received an `infra` verdict.

## Support rejections checked against the reference trajectory

LEVEL=75/76 reject every evaluation outside the persisted normalization box,
and no NN-side voltage limiter precedes that check. For every case that raised
a support error in round 1, the diagnostic ran only the NGSPICE reference and
checked each accepted terminal voltage against the box of all 40 checkpoint
pairs. The cases were `ota_5t_buffer` transfer and settling, all four
`current_mirror` analyses, and Miller `opamp` transfer.

All 120 runs completed. All 280 records, 2,875,880 values, are inside support.
The 18 round-1 `CandidateSupportError`s in these DC and transient analyses
therefore came from intermediate Newton iterates, a solver-globalization
limit, not from missing training data. The other 6, `ota_5t_buffer`
closed-loop AC rows, were not checked and remain unattributed.

A second pass covered the round-2 cases that raised support errors and have a
DC or transient analysis: `cascode_stack`, `nand2`, `nor2`,
`diffpair_active_load`, `multistage_buffer_12t` and `unity_gain_buffer`. It
ran 240 reference-only runs across all 40 checkpoint pairs. Of 520 records
covering 12,523,560 values, every DC sweep, compliance sweep, steering sweep,
and settling or transfer run is inside support.

The only outside points are 2–8 terminal values across 2,035 samples in NAND2
and NOR2 transients,
on the devices beside the internal node that the reference start-up drives past
the rails. Their terminal values equal that spike: NAND2 `Mn_a` Vgs = Vbs =
−0.985 V, NOR2 `Mp_b` Vds 1.903 V and Vbs 1.915 V, and NOR2 `Mp_a` Vds −1.46 V on
TSMC5. Round-2 support rejections in these cases are therefore also Newton trial
states. `opamp_rejection` has only AC analyses and was not checked.

BSIM-AR later raised support errors on `ldo_regulator` (`Mn_fb`), so a third
pass of 40 runs covered its line-regulation sweep and load-step transient for
all 40 checkpoint pairs. All 80 records, 4,019,960 values, are inside support.

A fourth pass covered `beta_multiplier`, where BSIM-AR rejected `Mp_mir`: its
operating point and both supply ramps are inside support for all 40 pairs
(120 records, 424,200 values). Across the four passes the diagnostic ran 440
reference-only runs. Of round 2's 100 support-rejection rows, it covers the 65
in DC, operating-point and transient analyses of the checked cases. It does
not cover 33 AC rows (their operating points were never checked) or two
BSIM-AR `self_biased_cascode` compliance rows, a case no pass included. Every
checked rejection is a Newton trial state rather than missing training data;
the unchecked 35 are unattributed.

## Reference start-up artifact

With `tran ... uic`, NGSPICE's first stored step drives some initialized nodes
past the rails. On TSMC6, NOR2 `v(pint)` is −1.165 V from `.ic` 0.75 V and NAND2
`v(nint)` is 0.985 V from `.ic` 0 V; on TSMC5, inverter-chain `v(n1)` is 0.803 V
against a 0.65 V supply. The resulting `max_err` on the internal nodes, 0.52–0.99 V
for NAND2 `v(nint)` and 1.05–1.92 V for NOR2 `v(pint)`, comes from the reference
spike and is not model error.
The per-technology maximum voltage error below excludes those two nodes. The
inverter chain's ~17% amplitude error at every TSMC5 size is consistent with
the same spike but was not proven.

## DirectNet-Full (LEVEL=75)

### OMP 1/2/4 strictness — 40/40, no flips

| gate | small | medium | large | xl |
|---|---:|---:|---:|---:|
| ring oscillator, period error | 5/5 | 5/5 | 5/5 | 5/5 |
| Miller DC, gain error | 5/5 | 5/5 | 5/5 | 5/5 |

Every scored metric is identical to two decimals at 1, 2 and 4 threads, except
medium TSMC12 Miller gain error (0.78/0.79/0.78%).

### Simple-v2 convergence by tier (23 cases)

| size | L1 rows | L2 rows | L3 rows | L4 rows |
|---|---:|---:|---:|---:|
| small | 25/25 | 86/90 | 110/135 | 35/45 |
| medium | 25/25 | 83/90 | 115/135 | 40/45 |
| large | 24/25 | 86/90 | 107/135 | 37/45 |
| xl | 24/25 | 87/90 | 119/135 | 44/45 |

Of 1,180 rows, 1,047 converged. The 133 error rows are all attributed to the
candidate: 96 nonconvergence, 36 support rejections, and 1 derived row without
inputs. The reference converged in every row.

| tier | case | small | medium | large | xl | errors |
|---|---|---:|---:|---:|---:|---|
| L1 | diode_load | 25/25 | 25/25 | 24/25 | 24/25 | sweep start (0 V) nonconvergence |
| L2 | cascode_stack | 15/15 | 12/15 | 15/15 | 15/15 | support |
| L2 | diffpair_active | 20/20 | 20/20 | 20/20 | 20/20 | — |
| L2 | inverter_chain | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L2 | inverter_energy | 10/10 | 10/10 | 10/10 | 10/10 | — |
| L2 | nand2 | 14/15 | 14/15 | 13/15 | 15/15 | support, convergence |
| L2 | nor2 | 12/15 | 12/15 | 13/15 | 12/15 | support, DC sweep points |
| L2 | transmission_gate_dc | 10/10 | 10/10 | 10/10 | 10/10 | — |
| L3 | beta_multiplier | **6/15** | **7/15** | **4/15** | **8/15** | OP and supply-ramp nonconvergence |
| L3 | bias_tree_fanout_5t / 9t / 17t | 15/15 | 15/15 | 15/15 | 15/15 | — |
| L3 | diffpair_active_load | 13/20 | 18/20 | 16/20 | 15/20 | support, AC operating point |
| L3 | opamp_rejection | 20/20 | 20/20 | 16/20 | 20/20 | support, derived |
| L3 | ring_osc_supply | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L3 | self_biased_cascode | 2/5 | 5/5 | 4/5 | 5/5 | operating point |
| L3 | self_biased_cascode_pmos | 3/5 | 5/5 | 5/5 | 5/5 | operating point |
| L3 | sram6t_modes | 36/40 | 30/40 | 32/40 | 36/40 | write-margin DC sweep points near 0.2–0.3 V |
| L3 | switchcap_multicycle | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L3 | transmission_gate_hold | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L4 | ldo_regulator | 16/20 | 20/20 | 16/20 | 20/20 | DC/GMIN/AC operating point |
| L4 | multistage_buffer_12t | **5/10** | 7/10 | 7/10 | 9/10 | support, operating point |
| L4 | unity_gain_buffer | 14/15 | 13/15 | 14/15 | 15/15 | support, operating point |

The self-biased cells converge least: `beta_multiplier` has a degenerate
zero-current state that the supply ramp must leave. Capacity helps the L4
buffers (12T buffer 5/10 at small, 9/10 at xl) but not `beta_multiplier` or
`sram6t_modes`.

### Physical errors over converged rows (median / worst, all sizes)

| tier | case | metric | median | worst |
|---|---|---|---:|---:|
| L1 | diode_load | diode drop error | 0.38 mV | 7.5 mV |
| L1 | diode_load | line sensitivity error | 0.88% | 14.4% |
| L2 | cascode_stack | NMOS / PMOS gain error | 0.88% / 1.58% | 10.1% / 26.7% |
| L2 | cascode_stack | output resistance error | 11.2% | 114% |
| L2 | diffpair_active | differential gain error | 2.47% | 10.4% |
| L2 | diffpair_active | CMRR error | 0.50 dB | 2.8 dB |
| L2 | inverter_chain | delay / rise-fall error | 1.04% / 0.82% | 2.5% / 4.3% |
| L2 | inverter_energy | delay error | 0.47% | 2.2% |
| L2 | inverter_energy | switching energy error | **18.3%** | **27.1%** |
| L2 | nand2 | delay / rise-fall error | 0.75% / 1.07% | 3.1% / 3.8% |
| L2 | nor2 | delay / rise-fall error | 0.57% / 0.37% | 2.0% / 2.8% |
| L2 | transmission_gate_dc | on-resistance error | 2.79% | 8.2% |
| L3 | beta_multiplier | bias current error | 0.65% | 6.8% |
| L3 | beta_multiplier | start-up supply error | 10 mV | 40 mV |
| L3 | bias_tree_fanout (5/9/17T) | bias node error | 0.39 mV | 2.6 mV |
| L3 | bias_tree_fanout (5/9/17T) | supply current error | 0.2% | 2.1% |
| L3 | diffpair_active_load | gain / bandwidth error | 1.27% / 0.18% | 7.6% / 1.4% |
| L3 | opamp_rejection | differential gain error | **26.5%** | **163%** |
| L3 | opamp_rejection | CMRR / PSRR error | 2.4 / 3.8 dB | 10.3 / 16.5 dB |
| L3 | ring_osc_supply | period / supply current error | 0.20% / 0.34% | 2.0% / 17.5% |
| L3 | self_biased_cascode (NMOS / PMOS) | output resistance error | 31% / 43% | 83% / 506% |
| L3 | sram6t_modes | hold margin / write trip error | 0.14 / 5.0 mV | 1.9 / 9.8 mV |
| L3 | sram6t_modes | read disturb error | 25 mV | 48 mV |
| L3 | switchcap_multicycle | cycle drift / final sample error | 0.48 / 0.29 mV | 2.2 / 1.5 mV |
| L3 | transmission_gate_hold | droop error | 0.04 mV | 1.5 mV |
| L4 | ldo_regulator | output voltage / load droop error | 1.4 / 0.78 mV | 15 / 5.6 mV |
| L4 | ldo_regulator | PSRR error | 12.6 dB | 28.4 dB |
| L4 | ldo_regulator | output impedance error | 28.6% | 125% |
| L4 | multistage_buffer_12t | gain / bandwidth / settling error | 0.80% / 1.06% / 0.39% | 4.1% / 5.1% / 9.9% |
| L4 | unity_gain_buffer | follow error | 1.0 mV | 4.9 mV |
| L4 | unity_gain_buffer | closed-loop gain / settling error | 2.4% / 0.17% | 21.5% / 20% |

Three results need care:

- **LDO line regulation.** The relative error (median 260%) divides by a nearly
  flat reference slope. NGSPICE gives 0.0043–0.0105 V/V; DirectNet gives
  −0.018 to +0.257 V/V. Seven of 18 converged rows have the opposite sign, and
  ten are 1.4–32× steeper. That is a real loss of line rejection in the model,
  even though the regulated output voltage is within 15 mV.
- **Miller differential gain.** The reference is 29–47 V/V at this bias and
  DirectNet 9.2–86.9 V/V, too low in 12 of 19 rows and too high in the rest.
  The error has no consistent sign. Its size is consistent with the open-loop
  AC weakness recorded in the V7.6.6 report.
- **Inverter switching energy.** On TSMC5 the model is 25.8–27.1% high at every
  size (about 1.03e-14 J against 8.12e-15 J), on TSMC6/7 it is 18–21% high,
  and on TSMC12/16 it is 2.6–11.6% low. Capacity does not change it, so the cause
  is systematic, not fit quality. The reference start-up spike carries about
  2e-17 J and does not explain it; the cause is unattributed.

Output-resistance errors (cascodes, current mirrors) and LDO output impedance
are small-signal slopes of nearly flat curves and amplify fit noise; the gain
and bias quantities in the same rows are the better accuracy readout.

### Per-technology aggregates (round-2 simple-v2 rows)

| technology | rows | converged | median NRMSE | rows >10% NRMSE | median MRE | median R² | row max voltage error, median / p95 | max excl. NAND/NOR stack nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TSMC5 | 236 | 190 | 0.78% | 33 | 0.72% | 0.9815 | 5.7 mV / 836 mV | 669 mV |
| TSMC6 | 236 | 213 | 1.10% | 36 | 0.96% | 0.9827 | 4.7 mV / 929 mV | 760 mV |
| TSMC7 | 236 | 212 | 1.39% | 36 | 0.97% | 0.9843 | 5.7 mV / 975 mV | 777 mV |
| TSMC12 | 236 | 209 | 0.58% | 23 | 0.61% | 0.9969 | 4.0 mV / 517 mV | 841 mV |
| TSMC16 | 236 | 223 | 0.59% | 31 | 0.71% | 0.9921 | 4.6 mV / 606 mV | 871 mV |

The typical row's worst voltage error is 4–6 mV. Of 794 converged DC,
operating-point and transient rows, 78 exceed 0.1 V after the NAND/NOR stack
nodes are removed, and almost all are edge-timing or threshold shifts that
maximum absolute error magnifies:

- ring oscillator `v(n5)` without phase alignment (19), where period error is at
  most 2%;
- the inverter-chain start-up spike (12);
- steep NAND/NOR/inverter DC transfer edges (19);
- `beta_multiplier` start-up at a supply a few mV away (12);
- SRAM write-margin nodes flipping at a slightly different sweep point (12).

Four rows are `multistage_buffer_12t` settling, where the settling error itself
reaches 9.9%.

## BSIM-AR-Full (LEVEL=76)

All 540 cells completed. Runtime dominated the round: the last five cells to
finish took 28–53 h each (xl `ring_osc_supply` on TSMC6/7/12/16, xl
`beta_multiplier` on TSMC5). The same two cases took 6.8–22 h on the other xl
technologies, and medium TSMC7 `ring_osc_supply` took 37.6 h but finished earlier. Four of those five still
passed; see the runtime note below.

### Simple-v2 convergence by tier

| size | L1 rows | L2 rows | L3 rows | L4 rows |
|---|---:|---:|---:|---:|
| small | **25/25** | 82/90 | 112/135 | 26/45 |
| medium | **25/25** | 81/90 | 119/135 | 44/45 |
| large | 22/25 | 85/90 | 121/135 | 38/45 |
| xl | 22/25 | 83/90 | 112/135 | 35/45 |

Of 1,180 rows, 1,032 converged. All 148 error rows are attributed to the
candidate, and the reference converged in every row.

| tier | case | small | medium | large | xl | errors |
|---|---|---:|---:|---:|---:|---|
| L1 | diode_load | 25/25 | 25/25 | 22/25 | 22/25 | DC sweep points below 0.2 V |
| L2 | cascode_stack | 12/15 | 12/15 | 15/15 | 15/15 | support |
| L2 | diffpair_active | 20/20 | 20/20 | 20/20 | 20/20 | — |
| L2 | inverter_chain | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L2 | inverter_energy | 10/10 | 10/10 | 10/10 | 10/10 | — |
| L2 | nand2 | 11/15 | 10/15 | 11/15 | 12/15 | support (7), nonconvergence (9) |
| L2 | nor2 | 14/15 | 14/15 | 14/15 | 11/15 | support (4), nonconvergence (3) |
| L2 | transmission_gate_dc | 10/10 | 10/10 | 10/10 | 10/10 | — |
| L3 | beta_multiplier | **5/15** | **6/15** | **5/15** | **3/15** | OP, supply-ramp start point, support |
| L3 | bias_tree_fanout_5t / 9t / 17t | 15/15 | 15/15 | 15/15 | 15/15 | — |
| L3 | diffpair_active_load | 18/20 | 20/20 | 20/20 | 20/20 | support |
| L3 | opamp_rejection | 20/20 | 20/20 | 20/20 | 20/20 | — |
| L3 | ring_osc_supply | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L3 | self_biased_cascode | 2/5 | 4/5 | 5/5 | 3/5 | compliance nonconvergence, support |
| L3 | self_biased_cascode_pmos | 5/5 | 5/5 | 5/5 | 4/5 | operating point |
| L3 | sram6t_modes | 32/40 | 34/40 | 36/40 | 32/40 | write-margin DC sweep points |
| L3 | switchcap_multicycle | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L3 | transmission_gate_hold | 5/5 | 5/5 | 5/5 | 5/5 | — |
| L4 | ldo_regulator | **8/20** | 20/20 | 16/20 | 16/20 | support (`Mn_fb`, `Mn_ref`, `Mp_l`), GMIN |
| L4 | multistage_buffer_12t | **4/10** | 9/10 | 7/10 | **4/10** | support, operating point |
| L4 | unity_gain_buffer | 14/15 | 15/15 | 15/15 | 15/15 | support |

`beta_multiplier` is the worst case in both families: BSIM-AR converges 19 of
60 rows against DirectNet's 25 of 60. Capacity does not fix it, and its xl
cell is the round's only scientific FAIL among the five slowest.

### Physical errors over converged rows (median / worst, all sizes)

| tier | case | metric | median | worst |
|---|---|---|---:|---:|
| L1 | diode_load | diode drop error | 1.8 mV | 6.2 mV |
| L1 | diode_load | line sensitivity error | 2.05% | 13.3% |
| L2 | cascode_stack | NMOS / PMOS gain error | 1.98% / 1.44% | 18.7% / 24.7% |
| L2 | cascode_stack | output resistance error | 10.5% | 560% |
| L2 | diffpair_active | differential gain error | 3.73% | 15.5% |
| L2 | diffpair_active | CMRR error | 0.76 dB | 4.8 dB |
| L2 | inverter_chain | delay / rise-fall error | 1.00% / 0.85% | 2.2% / 3.7% |
| L2 | inverter_energy | switching energy error | **18.3%** | **32.9%** |
| L2 | nand2 / nor2 | delay error | 0.59% / 0.53% | 3.0% / 2.2% |
| L2 | transmission_gate_dc | on-resistance error | 2.75% | 8.3% |
| L3 | beta_multiplier | bias current error | 3.46% | 16.1% |
| L3 | bias_tree_fanout (5/9/17T) | bias node error | 1.47 mV | 4.1 mV |
| L3 | diffpair_active_load | gain error | 1.65% | 18.2% |
| L3 | opamp_rejection | differential gain error | **19.7%** | **1,930%** |
| L3 | opamp_rejection | CMRR / PSRR error | 0.91 / 1.69 dB | 21.7 / 24.7 dB |
| L3 | ring_osc_supply | period / supply current error | 0.41% / 0.41% | 3.6% / 3.8% |
| L3 | self_biased_cascode (NMOS / PMOS) | output resistance error | 12% / 14% | 28% / 387% |
| L3 | sram6t_modes | read disturb / write trip error | 24 / 4.7 mV | 47 / 5.0 mV |
| L3 | switchcap_multicycle | cycle drift error | 0.98 mV | 8.0 mV |
| L4 | ldo_regulator | output voltage error | 0.84 mV | 9.3 mV |
| L4 | ldo_regulator | PSRR error | 3.95 dB | 29 dB |
| L4 | multistage_buffer_12t | gain / settling error | 0.09% / 0.20% | 2.2% / 44.3% |
| L4 | unity_gain_buffer | follow error | 0.66 mV | 7.6 mV |
| L4 | unity_gain_buffer | closed-loop gain / settling error | 1.18% / 0.44% | 41.7% / 15.7% |

The same three caveats as DirectNet apply, with BSIM-AR's own values: inverter
switching energy is 18–33% high on TSMC5/6/7 and 2–6% low on TSMC12/16,
the Miller differential gain is unreliable at this bias (median 19.7%, one
medium TSMC12 row at 1,930%), and output-resistance and line-regulation errors
are slopes of nearly flat curves.

### Per-technology aggregates

| technology | rows | converged | median NRMSE | rows >10% NRMSE | median MRE | median R² | row max voltage error, median / p95 | max excl. NAND/NOR stack nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TSMC5 | 236 | 201 | 1.10% | 41 | 1.09% | 0.9754 | 5.7 mV / 836 mV | 688 mV |
| TSMC6 | 236 | 206 | 0.91% | 32 | 0.87% | 0.9898 | 5.5 mV / 985 mV | 779 mV |
| TSMC7 | 236 | 214 | 0.87% | 39 | 0.89% | 0.9898 | 4.5 mV / 985 mV | 786 mV |
| TSMC12 | 236 | 209 | 0.78% | 33 | 1.20% | 0.9795 | 6.3 mV / 517 mV | 828 mV |
| TSMC16 | 236 | 202 | 0.69% | 33 | 0.97% | 0.9838 | 6.1 mV / 647 mV | 872 mV |

### Runtime

BSIM-AR's cost is the practical finding of round 2. Cost tracks convergence
difficulty, not circuit size: `ring_osc_supply` xl took 12 h on TSMC5 but
28–53 h on the other four technologies, and `beta_multiplier` ranged from
38 s (medium TSMC5, stopped by support rejections) to 38 h (xl TSMC5) on the
same template. Per cell against DirectNet on identical cases, BSIM-AR
`ring_osc_supply` ran 16–613× slower, `switchcap_multicycle` 33–173×, and the
OMP-2/4 ring gate 8–643×. Over all 540 paired cells the ratio has a median
of 39× and spans 0.6–2,516×; BSIM-AR is faster only on a few `ldo_regulator`
cells that stop early. The last five cells held round 2 open for 33.6 h
after the other 1,075 had finished. The round took 3.4 days. Wall times are
measured from log creation to verdict on a shared host, so they include load.

## Round-2 comparison

| measure | DirectNet-Full (LEVEL=75) | BSIM-AR-Full (LEVEL=76) |
|---|---:|---:|
| simple-v2 rows converged | **1,047/1,180** | 1,032/1,180 |
| error rows (all candidate) | 133 | 148 |
| OMP 1/2/4 strict cells | **40/40** | 37/40 (1 flip) |
| median NRMSE by technology | 0.58–1.39% | 0.69–1.10% |
| median MRE by technology | **0.61–0.97%** | 0.87–1.20% |
| median R² by technology | **0.9815–0.9969** | 0.9754–0.9898 |
| runtime per cell on identical cases | **1×** | median 39× (0.6–2,516×) |

The convergence totals are close, but the two families fail on different
circuits. BSIM-AR converges more rows on `diffpair_active_load` (78/80 against
62/80), `opamp_rejection` (80/80 against 76/80), `unity_gain_buffer` (59/60
against 56/60) and `nor2` (53/60 against 49/60). DirectNet converges more on
`nand2` (56/60 against 44/60), `beta_multiplier` (25/60 against 19/60),
`ldo_regulator` (72/80 against 60/80) and `diode_load` (98/100 against 94/100).
`sram6t_modes` is identical at 134/160, and both converge every
`ring_osc_supply` and bias-tree row.

Neither family is uniformly better on accuracy: DirectNet holds a small edge in
median MRE and R², BSIM-AR in median NRMSE on two technologies (TSMC6, TSMC7).
Both carry the same technology-dependent inverter-energy error and the same
unreliable Miller differential gain. The decisive difference is cost: BSIM-AR
needs a median 39× the runtime per cell for no consistent accuracy gain, and
its worst cells took 28–53 h each.
On this evidence DirectNet remains the practical default, which matches the
V7.7.0 policy already recorded in the accuracy index.

## Provenance

- Checkpoints: the 80 bundles and 280 artifacts in
  `PyCircuitSim-v771/results/v771_r2_checkpoints`, all hashes matching round 1.
  Training and dataset source `6be83348c1f5db6720d7504ed6dcea874a3a7418`.
- Evaluation source: branch `eval/v775-round2` at
  `d708d4bbe66a4b9ff9e5ca41650dd190c24581a2` in the clean worktree
  `PyCircuitSim-v775-round2`. That commit is round 1's `39b14f1` plus the two
  harness fixes, all of which reached `main` in merge `b688412` (2026-09-21),
  after every cell here was scored. The manifests record a
  `distinct-evaluation-arm`. The
  evaluation-source inventory hash `c15516725166…` equals round 1's, because the
  fixes change only test-harness files.
- Manifests (SHA-256): simple-v2 pool
  `570a2090c200d3748bdd0d6d3a70309fae8f4731bb9ac4644328e2fb71a75b43` (920
  jobs, list `a6a1d8f2…83a7`), OMP-2/4 pool
  `34f4cfc4f21beaea6fc4887e6b02e12af9e494a9551cce4c602cdb0f11af78d3` (160
  jobs, list `72621ecb…a949`). Every verdict log carries its pool digest. OSDI
  `f089f17d…a78b`; NGSPICE 45.2 `3b931f4e…d764`.
- Execution: capped `systemd --user` units, one CPU thread per OMP=1 cell. The
  OMP-2/4 cells use 2 or 4 threads.
- Raw evidence (ignored by Git): `PyCircuitSim-v775-round2/results/v775_round2/`,
  including `SUMMARY.md`, `cells.json`, the launch scripts, the aggregator, and
  the support diagnostic under `l4_support/`.
