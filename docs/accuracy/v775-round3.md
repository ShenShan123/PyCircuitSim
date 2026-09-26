# V7.7.5 checkpoints — round 3: the never-run campaign suites

Run: 2026-09-25.

**Status: scoped diagnostic, not a clean qualification.** Round 3 finishes
coverage of the campaign job generator (`scripts/v710_regate_jobs.py`) for the
80 V7.7.2-trained bundles scored in [round 1](v775-quick-round1.md) and
[round 2](v775-round2.md). It runs the seven campaign suites neither round
ran. It does not rescore earlier cells. The three rounds used different
evaluation commits, so methodology §5 forbids combining them into a clean
denominator. Gate rules are owned by [`methodology.md`](methodology.md).

Round 3 is complete: 280/280 cells, 237 PASS and 43 FAIL, none labelled `infra`.
The NGSPICE reference converged in every row. One row that the harness counted
as converged is not a physical fixed point. This report reclassifies it as an
`ERROR`; see the [solver defect](#solver-defect-leaked-capacitor-state-after-the-pseudo-transient-fallback)
below.

## Coverage: what rounds 1–2 left out

Every `circuit_templates/` topology reaches the campaign through a catalog case
or a suite. Rounds 1 and 2 ran all 34 catalog cases (4 simple-v1 gates and 30
simple-v2 cases) and one suite, `nn_multi_tech_dc`. They did not run these:

| suite | pool | templates exercised | role |
|---|---|---|---|
| `verify_nn_ac` | clean | `L1_primitives/common_source` | qualification: device AC, /10 per size |
| `verify_circuit_opamp_ac` | clean | `L3_blocks/opamp_miller` | qualification: Miller open-loop AC, /5 per size |
| `verify_nn_multi_tech_tran` | clean | `L2_stages/inverter` | qualification: 20 VTC/transient configurations per technology |
| `verify_nn_lifted_source_dc` | canary | `L0_devices/mosfet` | runtime source-frame canary, 6 configurations per checkpoint group |
| `verify_device_integrity` | clean | `L0_devices/mosfet` | diagnostic: output, subthreshold, linear, `gm`/`gds`/`gmb` |
| `verify_terminal_integrity` | clean | `L0_devices/mosfet` | diagnostic: four terminal currents, KCL, 4×4 transcapacitance |
| `verify_nn_subckt` | clean | `subcircuits/inverter_buffer_{flat,hierarchical}` | diagnostic: flat vs nested NN buffer, DC/transient/AC |

That is 6 suites × 40 checkpoint groups (240 clean cells) plus the 40-cell
canary pool, at OMP=1 as the generator defines them. After round 3, every job
kind the generator emits has run on these bundles at nominal.

Templates still outside every campaign pool:

- The ring 3-, 7- and 9-stage templates, reached only by `verify_circuit_sweep.py`.
  Adding them as catalog cases changes the simple-v2 denominator and is an open
  V7.7.3 decision.
- Non-nominal simple-v2 corners.
- `controls/rc_lowpass` and the passive or LEVEL=72-only `subcircuits/` fixtures,
  which contain no NN device.

## Qualification gates

| gate | DirectNet small | medium | large | xl | BSIM-AR small | medium | large | xl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| device AC (/10) | 10/10 | 10/10 | 10/10 | 8/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Miller open-loop AC (/5) | 0/5 | 0/5 | 2/5 | 2/5 | 1/5 | 4/5 | 4/5 | 4/5 |
| inverter VTC and transient configs (/100) | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 |
| lifted-source canary configs (/30) | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |

- **Device AC.** DirectNet misses only xl TSMC6/TSMC7 PMOS: a 1.53 dB gain error
  against a 1.5 dB limit and 18.4% magnitude NRMSE. The two checkpoints give
  identical rows. Over all 40 rows, the f3dB ratio stays within 0.89–1.12.
  BSIM-AR passes 40/40, with a worst gain error of 0.74 dB.
- **Miller open-loop AC.** DirectNet passes 4 of 20. Every failure misses the
  3 dB DC-gain limit (median error 14.3 dB over the 16 failures, worst
  35.9 dB); two also miss phase margin or GBW. At the NGSPICE-refined input
  bias, the NN output sits elsewhere on its transfer curve (for example
  0.085 V against 0.210 V on medium TSMC5), so the gain is measured at a
  different point. This is the
  V7.6.6 weakness (2/5 at large) seen at every size. BSIM-AR passes 13 of 20,
  with a median DC-gain error of 0.94 dB. Its misses are small TSMC5/6/7/16 and
  large TSMC5, plus two candidate `ERROR`s on TSMC5: `Mp3` outside support at
  medium (0.854 V against 0.78 V) and a DC operating point that did not converge
  at xl.
- **Inverter parametric transient.** DirectNet xl TSMC12 cannot characterize
  two VTC configurations (VDD 0.70 and 0.90 V): a DC sweep point did not
  converge. Both are candidate `ERROR`s. Every other row passes. The worst
  converged NRMSE is 5.4% (DirectNet, 10 ps input slew) and 6.7% (BSIM-AR,
  small TSMC5 VTC at 0.55 V).
- **Lifted-source canary.** All 240 rows pass, 120 per family: source-relative
  inference holds for NMOS and PMOS at every lift. The worst NRMSE is 1.7%
  (DirectNet) and 2.4% (BSIM-AR), against a 10% limit.

## Diagnostics

### Device integrity: linear-domain currents are accurate; off-state current is not

The table gives median / worst per size over 5 technologies × NMOS/PMOS. The
Idsat and saturation-gds rows also pool four gate biases, so they cover 40
rows per size.

| metric | DirectNet small | medium | large | xl | BSIM-AR small | medium | large | xl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Idsat error, % | 2.7 / 14.7 | 0.26 / 2.3 | 0.11 / 2.6 | 0.13 / 2.4 | 3.8 / 41 | 1.2 / 18 | 0.72 / 6.4 | 0.31 / 3.4 |
| Ron error, % | 5.5 / 9.7 | 0.45 / 2.4 | 0.40 / 3.8 | 1.3 / 5.3 | 5.2 / 14 | 2.3 / 3.3 | 0.62 / 5.3 | 0.40 / 1.4 |
| gm MRE, % | 3.1 / 7.1 | 0.75 / 1.3 | 0.26 / 1.2 | 0.65 / 1.9 | 5.4 / 7.9 | 1.5 / 4.8 | 1.1 / 1.3 | 0.70 / 1.4 |
| gds MRE, % | 17 / 27 | 7.5 / 9.8 | 4.6 / 21 | 7.3 / 28 | 11 / 37 | 6.1 / 10 | 2.0 / 5.5 | 2.2 / 3.6 |
| gmb MRE, % | 18 / 53 | 3.8 / 7.0 | 1.9 / 11 | 2.8 / 41 | 16 / 66 | 8.3 / 21 | 4.3 / 15 | 1.6 / 3.3 |
| saturation gds error, % | 18 / 230 | 5.9 / 23 | 4.0 / 49 | 5.1 / 42 | 12 / 82 | 7.4 / 46 | 3.4 / 13 | 2.1 / 22 |

Above threshold, both families track the reference. BSIM-AR improves with
size in every metric except gds MRE, which is flat from large to xl (2.0% and
2.2%). Output conductance and body transconductance are the weakest
linear-domain quantities.

Below threshold they do not track it. In every scoreable subthreshold row, the
NN off-current at Vgs = 0 is **higher** than the reference, never lower:

| family | scoreable rows | median log₁₀(Ioff_NN / Ioff_ref) | ≥ 1 decade high | unscoreable |
|---|---:|---:|---:|---:|
| DirectNet-Full | 33/40 | 2.0 | 28 | 7 (medium 4, large 1, xl 2) |
| BSIM-AR-Full | 22/40 | 2.1 | 22 | 18 (all 10 small rows, medium 5, large 2, xl 1) |

An unscoreable row is one where the NN current spans less than half a decade
across the reference subthreshold window, so no swing can be fit. The NN
current is nearly flat there, which is worse than a shifted curve. Among
scoreable rows the median swing error is 41–92% by size. In the same rows the
linear-domain NRMSE is typically below 1%. A linear-scale current metric cannot
see this error. These rows are the only reason device-integrity cells exit 1.

### Terminal integrity

All 720 rows characterize. KCL closes to within 1.4e-19 A, because source current
and charge are derived analytically. Terminal-current NRMSE has a median of
0.02–0.36% by size and a worst case of 2.7% (BSIM-AR small TSMC6/7 PMOS body
sweep). The largest 4×4 transcapacitance error has a median of 0.19–0.91% of
the largest reference element; the worst is 8.6% (DirectNet xl TSMC6/7, NMOS
linear).

### Flat vs hierarchical NN netlists

Flat and nested buffers agree in every analysis to within 1.5e-14 V, so
subcircuit expansion is representation-exact for both families. Against
NGSPICE:

- **DC** (buffer transfer, 5 mV steps). Median NRMSE is 0.2–7.6% by size. The
  7.6–7.9% rows are a one-step (5 mV) shift of the buffer trip point on a
  vertical edge. One flipped sample among 151–161 points gives about 1/√n ≈ 8%
  NRMSE and a 0.73–0.77 V point error (checked on three traces). BSIM-AR medium
  TSMC16 (70.4% as reported) is not a model result. It is the solver defect
  below and counts as an `ERROR`, so BSIM-AR medium converges 4/5.
- **Transient.** Median NRMSE is 0.38–0.53% by size, worst 0.83%.
- **AC with the input at 0 V.** Both inverters sit at the rails, so the response
  is off-state gain and gate-drain feedthrough. The reference `v(mid)` is
  7.9e-7 to 2.7e-5 V/V at 1 kHz (TSMC6/7 are the high end). Range-normalized NRMSE (up to 101%) and phase (180°) are
  not usable here, so the saved candidate decks were rerun and compared point
  by point:

  | family | 1 kHz `v(mid)` NN / ref | inverting sign lost | 100 GHz `v(mid)` NN / ref |
  |---|---:|---:|---:|
  | DirectNet-Full | 8–4,141× (median 111×) | 10/20 | 0.90–1.05 at small/medium; 1.42–2.82 at large/xl |
  | BSIM-AR-Full | 16–4,193× (median 189×) | 11/20 | 1.93–2.84 in 19/20; 0.91 at small TSMC5 |

  The low-frequency excess is consistent with the off-current excess above: an
  off NMOS with about 100× too much current has about 100× too much subthreshold
  transconductance. The sign loss means the NN's net transconductance at this
  bias has the wrong sign, which an error at floor-level current can produce. The high-frequency feedthrough excess was measured at
  this bias only and is not attributed. Only one of these 40 AC reruns used the
  fallback involved in the defect below.

## Solver defect: leaked capacitor state after the pseudo-transient fallback

For NN circuits, `_solve_dc_with_retry` tries the fast path, then GMIN
stepping, then `_pseudo_transient_dc`. The last stage runs a `TransientSolver`
(dt = 1 ps) on the live circuit object and strips its artificial `_pseudo_*`
capacitors. It does not reset the circuit's real capacitors.
`Capacitor.stamp_conductance()` and `stamp_rhs()` always stamp the companion
`_g_eq`/`_i_eq`, so every later DC solve on that circuit sees each capacitor as
a conductance 2C/dt plus a current source. That includes the fallback's own
polishing solve and every remaining point of a DC sweep. Newton then converges
honestly, but on a different circuit, and `require_convergence` accepts it.
LEVEL=72 circuits never use this fallback.

Reproduction: BSIM-AR medium TSMC16 `nn_subckt`, flat buffer DC.

- The operating point at Vin = 0 needs the fallback. Afterward `Cload` (5 fF)
  carries `g_eq` = 0.01 S and `i_eq` = 28.7 µA.
- For Vin ≥ 0.40 V the second stage stays at 8–11 mV instead of rising to
  0.80 V.
- Re-stamped on a fresh parse, those points leave 80 µA of KCL residual at
  `out`, 52× the solver's own acceptance threshold (82 of 161 points).
- The same point solved from a fresh circuit converges to 0.798 V.

### Post-hoc KCL audit of rounds 1–3

The defect is silent: a sweep that went through the fallback looks converged.
Saved candidate DC sweeps were therefore re-stamped on a fresh parse of their
own decks, point by point. Each point's MNA residual was compared with the DC
solver's acceptance threshold, max(1 µA, 100·RELTOL·|current RHS|). Only
circuits with a capacitor can carry the defect. The counts below include
capacitor-free sweeps, which serve as controls.

| round | scored sweeps audited (with capacitor) | not fixed points | diagnostic-only sweeps flagged |
|---|---:|---:|---:|
| 1 | 539 (66) | 0 | 3 of 5 |
| 2 | 763 (336) | 0 | 60 of 119 |
| 3 | 1,360 (80) | 2, one row | — |

- Rounds 1–2: every scored DC sweep is a true fixed point, at no more than
  0.45× the threshold, so neither report's DC numbers change. The flagged
  sweeps are all `_unconv` reruns: the harness re-solves an already-`ERROR` row
  without `require_convergence` only to keep a readable trace, and no score
  uses them.
- Not in the counts: round 1's 1,032 `nn_multi_tech_dc` sweeps and round 3's
  238 inverter VTC sweeps (the two `ERROR` VTC rows saved none). The audit
  script looked for `candidate_<stem>.sp` or `<stem>.sp` and these suites save
  `nn_*.sp`, so it skipped them. Every one of those decks was checked
  afterwards and none contains a capacitor, so none can carry the defect.
- Round 3: the only affected scored row is the BSIM-AR medium TSMC16
  `nn_subckt` DC row (its flat and hierarchical sweeps). It is reclassified here
  as an `ERROR` of kind `solver_false_convergence` and excluded from every
  aggregate below. Its cell verdict (exit 0; the suite is diagnostic) is
  unchanged.
- Not auditable after the fact: operating points that feed AC and transient
  analyses, which are not saved. A fallback there can bias the operating point
  silently.

The fix belongs in `_pseudo_transient_dc`: restore every capacitor's companion
state after its transient stage, with a regression test that a DC sweep after a
fallback matches a fresh-circuit solve. It changes numerics, so it needs its own
re-gate and is not made here. It shipped in
[V7.7.6](../CHANGELOG.md#v776--capacitor-state-fix-and-opt-in-nn-limiting).

## Per-technology aggregates

These are over converged rows with trace metrics, all suites and sizes pooled.
Maximum voltage error covers `nn_multi_tech_tran` and `nn_subckt` DC/transient
rows only; current, capacitance and AC rows are excluded. The 0.73–0.77 V
entries are the one-step buffer trip shifts described above. The false-converged
BSIM-AR row is excluded as an `ERROR`.

| technology | DirectNet rows | converged | median NRMSE | >10% | median MRE | median R² | max V error | BSIM-AR rows | converged | median NRMSE | >10% | median MRE | median R² | max V error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TSMC5 | 272 | 269 | 0.27% | 14 | 0.79% | 0.9999 | 212 mV | 272 | 264 | 0.46% | 18 | 1.18% | 0.9997 | 376 mV |
| TSMC6 | 272 | 271 | 0.50% | 14 | 1.41% | 0.9998 | 731 mV | 272 | 270 | 0.41% | 13 | 1.35% | 0.9998 | 730 mV |
| TSMC7 | 272 | 271 | 0.47% | 15 | 1.40% | 0.9998 | 731 mV | 272 | 270 | 0.43% | 12 | 1.20% | 0.9998 | 725 mV |
| TSMC12 | 272 | 269 | 0.19% | 11 | 0.57% | 1.0000 | 775 mV | 272 | 268 | 0.44% | 16 | 1.15% | 0.9998 | 774 mV |
| TSMC16 | 272 | 271 | 0.35% | 11 | 0.81% | 0.9999 | 304 mV | 272 | 267 | 0.58% | 15 | 1.76% | 0.9996 | 212 mV |

## Runtime

Wall time runs from verdict-log creation to verdict on a shared host. Values
are minutes, averaged over five technologies.

| suite | DirectNet small | medium | large | xl | BSIM-AR small | medium | large | xl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nn_subckt` | 1.2 | 1.4 | 2.3 | 5.1 | 36 | 75 | 126 | 265 |
| `nn_multi_tech_tran` | 1.1 | 1.2 | 1.9 | 4.1 | 28 | 58 | 92 | 194 |
| `terminal_integrity` | 0.5 | 0.4 | 0.3 | 0.4 | 1.8 | 2.9 | 5.1 | 12 |
| `device_integrity` | 0.6 | 0.4 | 0.3 | 0.4 | 1.7 | 2.6 | 4.5 | 10 |
| `circuit_opamp_ac` | 0.4 | 0.2 | 0.1 | 0.2 | 0.6 | 0.8 | 1.4 | 12 |
| `nn_ac` | 0.4 | 0.3 | 0.1 | 0.2 | 0.2 | 0.3 | 0.4 | 1.0 |
| `nn_lifted_source_dc` | 0.2 | 0.2 | 0.2 | 0.2 | 0.8 | 1.2 | 3.1 | 7.0 |

The round took 5 h 50 min, with 60 workers on the clean pool. The BSIM-AR xl
hierarchy and inverter cells set its length.

## Provenance

- Checkpoints: the 80 bundles and 280 artifacts in
  `PyCircuitSim-v771/results/v771_r2_checkpoints` (a symlink to main's
  `results/`). All 280 hashes match the round-1 and r5 manifests. Training and
  dataset source is `6be83348c1f5db6720d7504ed6dcea874a3a7418`.
- Evaluation source: clean detached worktree `PyCircuitSim-v775-round3` at main
  `b24bfca671dd7849afb836b7ae0767230a17bac3`. That is round 2's `d708d4b` plus
  documentation and one removed dead alias, so no numerical path changed. The
  alias removal still changes the evaluation-source inventory hash, to
  `24019219…5f26`. The manifests record a `distinct-evaluation-arm`.
- Preconditions: 1,160 tests passed with none skipped; geometry preflight on
  `v771_r2_data` passed 463/463.
- Manifests (SHA-256): clean pool
  `8e11928baf453ce92472675cb97609089d8ee686e41c52ad4ae4b3bcdeeb8d15` (240 jobs),
  canary pool `41e3689921f1ec41ba4fe14b5b21096ef24fe14f3966caf0a1d32bee8ba14f17`
  (40 jobs). Every verdict log and GateResult row carries its pool digest.
  OSDI `f089f17d…a78b`; NGSPICE 45.2 `3b931f4e…d764`.
- Execution: capped `systemd --user` units, one CPU thread per cell. A forward
  walker (PAR 40) and a reverse walker (PAR 20) shared the clean list, with a
  canary unit at PAR 10. Every cell ran exactly once (211 + 29 + 40 runs). The
  45 lock-contention skips are why both clean units exit with xargs code 123.
- Raw evidence (ignored by Git): `PyCircuitSim-v775-round3/results/v775_round3/`.
  It holds `SUMMARY.md` and `cells.json` from `summarize_r3.py`, which parses raw
  logs only and fails closed on any missing verdict or provenance mismatch. It
  also holds the launch script, job lists, `pytest_full.log` and `geometry.log`.
