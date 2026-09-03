# V7.6.7 coverage report — device integrity, self-bias, and feedback

Ground truth is NGSPICE on the identical BSIM-CMG LEVEL=72 OSDI model, from
the same `circuit_templates/` source the candidate renders. Gate definitions
and evidence rules are in [`methodology.md`](methodology.md).

## Scope and status

This report characterizes the **new diagnostics added in V7.6.7**. It is not a
campaign and not a score:

- One checkpoint population — the served DirectNet (LEVEL=73) `large` clean
  checkpoints — on **TSMC5 and TSMC12 only**, at the `nominal` corner.
- No threshold is frozen, so nothing here passes or fails. The numbers exist
  to establish that the new instruments discriminate, and to record what they
  measured the first time they were pointed at a shipping model.
- The simple-v1 `/20` denominator, its four cases and its thresholds are
  unchanged. No published score moves because of anything in this document.

The motivating contradiction is in the
[V7.6.7 plan](../plans/2026-09-02-v767-evaluation-coverage.md): L75 `large`
scores 20/20 strict simple-v1 while solving 0/248 AnalogGym decks, and V7.6.6
retired that corpus as an executable gate.

## 1. Single-device integrity

`verify_device_integrity.py`, 36/36 sweeps characterized, 36/36 converged.

### 1.1 The origin crossing — the sharpest result

The `linear` suite sweeps Id–Vds through `Vds = 0` and fits the slope twice:
once over the conducting half, once over a window straddling the origin.
Reporting one number would conflate "the triode resistance is wrong" with
"the origin crossing is wrong". They differ by two orders of magnitude.

| tech | device | Ron forward (ref → test) | forward error | Ron across origin (ref → test) | origin error |
|---|---|---|---|---:|---:|
| TSMC5 | NMOS | 1762 Ω → 2088 Ω | 18.49% | 1405 Ω → 3868 Ω | **175.25%** |
| TSMC5 | PMOS | 3132 Ω → 3084 Ω | 1.52% | 2475 Ω → 5637 Ω | **127.73%** |
| TSMC12 | NMOS | 1806 Ω → 1765 Ω | 2.27% | 1408 Ω → 3209 Ω | **127.88%** |
| TSMC12 | PMOS | 2352 Ω → 2431 Ω | 3.35% | 1994 Ω → 4521 Ω | **126.72%** |

The conducting half is tracked to a few percent. The slope *through* `Vds = 0`
is roughly 2.3–2.8× too high on every technology and polarity measured. The
model's Id–Vds curve is flat where it should be steepest.

Nothing in the previous suite could see this. The scored device sweep is
Id–Vgs at a fixed `Vds = 0.5·VDD`; the only reverse-bias configuration
anywhere is a single `Vds = −0.25·VDD` point; and the same sweeps that produce
a 126% origin error report a 9.1–12.7% linear-axis current NRMSE.

`Id(Vds=0)` itself is correct — the candidate returns ~3e-21 A where the
reference sits on its own ~1e-12 A numerical floor. The defect is the
derivative at the origin, not an injected current.

### 1.2 Subthreshold decades

`max_decade_error` is the largest error in `log10|Id|` across the sweep;
`lin-NRMSE` is what a linear-axis metric reports for the identical curve.

| tech | device | span | max decade error | Ioff error | lin-NRMSE |
|---|---|---:|---:|---:|---:|
| TSMC5 | NMOS | 5.22 dec | **9.27 dec** | 100.0% | 7.47% |
| TSMC5 | PMOS | 5.48 dec | 2.82 dec | 0.8% | 0.30% |
| TSMC12 | NMOS | 5.34 dec | 0.43 dec | 35.9% | 0.29% |
| TSMC12 | PMOS | 6.30 dec | **2.93 dec** | 0.6% | 0.29% |

TSMC12 PMOS is the clearest case for the metric: the linear-axis NRMSE is
0.29% — indistinguishable from a good model — while the curve is nearly three
decades out somewhere in subthreshold. TSMC5 NMOS is worse than its own
dynamic range: a 9.27-decade error over a 5.22-decade span means the candidate
leaves the reference's range entirely.

Where the subthreshold slope is reported as `n/a`, the candidate's current in
the reference-defined window is pinned at the floor, so no slope can be fitted
to it. That is itself the finding, not a missing measurement. The one fitted
value, TSMC12 NMOS, is 54.5% out.

The window is derived from the **reference only**. Letting the candidate
influence it would let a broken model select the range that flatters it.

### 1.3 Derivatives against ground truth

Both engines are differentiated with the identical central-difference stencil
on the identical grid, so this measures surface-derivative fidelity rather
than analytic-versus-finite-difference stencil error.

| tech | `gm` | `gds` | `gmb` |
|---|---:|---:|---:|
| TSMC5 NMOS | 5.82% | 3.36% | **123.23%** (R² −15.50) |
| TSMC5 PMOS | 0.17% | 5.42% | 20.94% (R² 0.52) |
| TSMC12 NMOS | 0.10% | 5.31% | 1.50% |
| TSMC12 PMOS | 0.14% | 4.97% | **30.71%** (R² −0.03) |

`gm` is excellent, `gds` is uniformly ~5%, and `gmb` is the weak channel —
negative R² on two of four cells means the predicted body-effect derivative
carries less information than predicting its own mean. Sign agreement is 100%
everywhere, so no cell is at risk of the negative-`gds` instability
`AGENTS.md` floors the stamp against.

### 1.4 Output characteristics

Saturation `gds` from an Id–Vds family at four gate biases:

| tech | device | Vgs=0.45 | 0.60 | 0.80 | 1.00 ×VDD |
|---|---|---:|---:|---:|---:|
| TSMC5 | NMOS | 34.60% | 21.23% | 10.02% | 20.57% |
| TSMC5 | PMOS | 11.98% | 2.21% | 6.50% | 4.62% |
| TSMC12 | NMOS | 1.41% | 0.28% | 0.11% | 0.29% |
| TSMC12 | PMOS | 3.29% | 2.65% | 2.45% | 0.29% |

Error rises sharply toward weak gate drive, and TSMC5 — the low-supply
technology that fails every published ring cell at every tier — is an order of
magnitude worse than TSMC12 across the family. The saturation knee position
agrees to one sweep step in every TSMC12 cell.

## 2. Circuit ladder

Characterization of the new and expanded cases on TSMC12, `nominal`, served
DirectNet `large`. Convergence is reported separately from error throughout: a
DC row that did not reach a physical fixed point is an `ERROR` that keeps its
denominator slot and is never averaged into an accuracy number.

### 2.1 The two-element reproducer

The most consequential result is the smallest circuit. A diode-connected NMOS
fed **through a resistor from the rail** does not satisfy the convergence
contract; the same device fed by an **ideal current source** does; and
LEVEL=72 converges on both through the same solver, at the same tolerance, on
the same deck.

| circuit | LEVEL=72 | DirectNet L73 `large` |
|---|---|---|
| diode NMOS + 200 kΩ from supply | converged, `nr` = 0.39347 V | **not converged**, `nr` = 0.39290 V |
| diode NMOS + ideal 5 µA source | converged, `nr` = 0.43680 V | converged, `nr` = 0.43645 V |

The value is right to 0.6 mV. What fails is the KCL residual gate, and it
fails at every bias impedance from 20 kΩ to 500 kΩ, so it is not a
conditioning artifact of one resistor choice.

This matters because **every diode-connected device in the previous catalog is
fed by an ideal current source** — `current_mirror` and `cascode_stack` both
impose the current and leave only the voltage to solve. The load-line
intersection, where the model must determine both, was untested. It is now
`L1_primitives/diode_load.spice.tmpl`, the bottom rung of the self-bias
ladder.

### 2.2 Full ladder result

One consolidated pass over the new and expanded cases: **12 of 24 analyses
characterized, 11 of 24 candidate solves reached a physical fixed point.**
Rows marked `unconv` are `ERROR`s whose numbers come from the diagnostic
re-run; no scoring path reads them.

| tier | case / analysis | state | measured |
|---|---|---|---|
| L1 | `diode_load` supply_ramp | unconv | diode drop 0.60 mV, worst 1.40 mV |
| L2 | `current_mirror` nmos / pmos | ok | output resistance 0.13% / 1.21% |
| L2 | `current_mirror` nmos_iref / pmos_iref | ok | ratio error 0.032% / 0.027% median, 0.89% / 0.85% worst |
| L3 | `switchcap_multicycle` accumulate | ok | drift 10.7 mV vs 0.56 mV reference; final sample 7.9 mV out |
| L3 | `ring_osc_supply` oscillation | ok | period 2.14%, supply current 0.89% |
| L3 | `sram6t_modes` hold / read / write | ok | 4.87% / 9.50% / 90.01% NRMSE |
| L3 | `sram6t_modes` write_margin | unconv | — |
| L3 | `opamp_rejection` ×3 + derived | error | AC operating point never converged |
| L3 | `beta_multiplier` supply_ramp | unconv | bias current 1.04%, bias nodes 0.42 mV, start-up at 0.53 V vs 0.50 V |
| L3 | `self_biased_cascode` compliance | unconv | output resistance 34.68%, bias nodes 14.0 mV |
| L3 | `mos_ratio_reference` supply_ramp | unconv | Vref 0.88 mV, line sensitivity 0.71% |
| L4 | `unity_gain_buffer` transfer | unconv | reference follows to 4.4 mV; candidate 527 mV out |
| L4 | `unity_gain_buffer` settling | partial | committed 168 µV before failing |
| L4 | `ota_5t_buffer` transfer | unconv | reference follows to 12.6 mV; candidate 528 mV out |
| L4 | **`ota_5t_buffer` settling** | **ok** | **5.83% NRMSE, 4.4 mV max, settling within 0.17%** |
| L4 | `ldo_regulator` line_regulation | unconv | reference 0.765 %/V; candidate flat, 544 mV out |
| L4 | **`ldo_regulator` load_step** | **ok** | **droop 22.9 mV vs 20.6 mV (2.3 mV apart), recovery 1.1 mV** |

Two facts fall out that the previous suite could not have produced.

**The failure is the DC-sweep path on internally biased topologies, not
closed-loop behaviour.** Every self-biased DC sweep fails to certify, yet the
transients on the *same* circuits converge and are accurate: `ota_5t_buffer`
settles to 4.4 mV and `ldo_regulator` tracks a load step to within 2.3 mV of
the reference droop, both starting from a computed DC operating point with no
`.ic` anywhere in the deck. "Complex circuits fail" was never a precise
enough statement to act on; this one is.

**The unconverged values are mostly close.** `beta_multiplier` bias nodes are
0.42 mV out and its start-up supply 30 mV out; `mos_ratio_reference` is 0.88 mV
out with 0.71% line-sensitivity error. What fails is certification, not
accuracy — which is exactly why convergence and error have to be reported as
two numbers.

The exceptions are real failures rather than convergence artifacts:
`self_biased_cascode` output resistance is 34.7% out, and `sram6t_modes`
`write` is 90% NRMSE.

### 2.3 Expanded existing cases

- `switchcap_multicycle`: over ten clock periods the candidate drifts 10.7 mV
  where the reference drifts 0.56 mV — an 18× amplification, ending 7.9 mV
  from the reference. The scored three-cycle switched-capacitor cell **passes**
  on TSMC12. A per-cycle error inside threshold still accumulates, and a
  three-cycle window cannot distinguish the two.
- `ring_osc_supply`: period error 2.14%, reproducing the published TSMC12
  `large` ring number exactly, and adding a dynamic supply current agreeing to
  0.89%.
- `current_mirror` reference-current range: median ratio error 0.03%, worst
  0.89% from deep subthreshold to several times the nominal bias — the mirror
  itself is not the weak link.
- `opamp_rejection`: all three AC sweeps and the derived row are `ERROR`s
  because the candidate operating point never converged. This reproduces the
  published `OP-NOT-CONVERGED` opamp AC result rather than adding to it, and
  the derived row names its missing inputs explicitly instead of vanishing.

## 3. Instrument caveats

- Two technologies, one corner, one checkpoint population. Nothing here
  supports a family-level claim.
- `unity_gain_buffer` was mis-wired on first construction: feedback to the
  wrong input railed the output in **both** engines. The LEVEL=72 reference
  follow error was 558 mV before the fix and 4.4 mV after. A reference that
  does not do its job is an infrastructure failure, not a model result; the
  pre-fix numbers are not evidence and are not reported.
- `diffpair_ideal` does not converge on TSMC12. Its template is a byte-
  identical rename from before this work, so that failure is pre-existing and
  is not attributable to the V7.6.7 changes.
- The subthreshold slope needs at least four reference points spanning half a
  decade inside its window. Cells that do not meet that report `n/a` rather
  than a fitted number.

## 4. Reproduction

```bash
conda run -n pycircuitsim python \
  tests/single_devices/verify_device_integrity.py \
  --tech TSMC5,TSMC12 --device nmos,pmos

conda run -n pycircuitsim python \
  tests/simple_circuits/verify_circuit_topologies.py \
  --case diode_load,beta_multiplier,self_biased_cascode,mos_ratio_reference,\
unity_gain_buffer,ota_5t_buffer,ldo_regulator --tech TSMC12
```

Raw evidence is local and gitignored under
`results/tests/simple_circuits/device-integrity/` and
`results/tests/simple_circuits/simple-v2/`.
