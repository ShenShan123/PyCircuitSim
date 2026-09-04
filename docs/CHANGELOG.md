# PyCircuitSim changelog

This is the compact release and decision ledger. Commands belong in
[README.md](../README.md), durable implementation contracts in
[AGENTS.md](../AGENTS.md), compact-model scoreboards in
[docs/accuracy/](accuracy/). Detailed per-commit chronology and superseded prose
remain in Git history.

## Reading historical scores

- V7.3 added the deliberate TSMC6 repeat to compact-model headlines: strict
  circuits changed from /16 to /20, device AC from /8 to /10, and op-amp AC
  from /4 to /5.
- AnalogGym denominators changed as invalid or redundant decks were measured,
  quarantined, or pruned. Compare totals only when the basket is identical.
- Current generated reports, not this ledger, own detailed score tables.

## V7.7 — full-terminal-only NN stack

### V7.7.0 — retire reduced compact-model families (2026-09-04)

DirectNet-Full LEVEL=75 is now the default NN family and BSIM-AR-Full LEVEL=76
is the supported autoregressive alternative. LEVEL=73/74 are rejected rather
than silently remapped: keeping their level numbers while changing terminal
physics would make an old deck solve a different model without saying so.

Removed the reduced NN runtime base and both adapters, the classic
`gm/gds/gmb` drain-source solver stamp, reduced transient/AC capacitance
reconstruction, reduced batching/fused-Jacobian paths, and their performance
probes. Every supported MOSFET now supplies complete 4x4 current and charge
stamps or fails loudly.

Dataset generation and training now have one six-surface contract:
`i_d,i_g,i_b,qd,qg,qb`. Removed output-contract selection, the 13-head schema,
reduced filtering/subsets, derivative-head losses, monotone/EKV checkpoint
compatibility, and reduced recipe scripts. Campaign generation, collection,
coverage, and report tooling now accept only `dnf`/`tff` families.

Purged 280 ignored reduced checkpoint files (about 990 MiB) from the default
checkpoint directory. Preserved all `results/v766_full_*` evidence and linked
complete full-terminal artifacts into the default local locations. All 40 DNF
bundles are complete; only polarity-paired complete TFF tiers were linked, so
an incomplete or mixed-tier default cannot masquerade as ready. These ignored
artifact changes are local and are not represented by the Git commit.

This maintenance decision supersedes the earlier recommendation not to use
LEVEL=75 generally, but does not rewrite its evidence: the latest report is
20/20 on the declared simple-circuit matrix, 115/129 on parametric device DC,
and 2/5 on Miller AC. LEVEL=76 still lacks a complete five-technology clean
matrix. No new accuracy promotion is claimed. The collected unit suite passes
560 tests with two CPU pin-memory warnings.

## V7.6 — full-terminal families and closure

### V7.6.10 — metric oracles, hierarchy breadth, and stale-test purge (2026-09-04)

Audited V7.6.9's NN compact-model harness for questions that had declarations
but no independent behavioral witness. The executable report is
[`v7610-harness-audit.md`](accuracy/v7610-harness-audit.md). No diagnostic was
promoted, no threshold moved, and the frozen `simple-v1` `/20` denominator is
unchanged.

All 80 catalog analysis layouts across the 47 live metric profiles now receive
a known-trace identity check and a targeted mutation check through
`compare_traces`, with exact PMOS-compliance and CMRR/PSRR dB oracles. This
found and fixed the PMOS self-biased-cascode compliance region being measured
with the NMOS sweep orientation, and removed two unused generic profile names.
The NN flat/nested buffer gate now executes DC, transient, and AC and the
campaign collector requires all three rows. TSMC12 LEVEL=73 flat/hierarchical
differences were 0 V, 0 V, and 1.06e-22 V respectively.

Added a resistor-fed PMOS-only generated-cascode-bias topology. NGSPICE and the
PyCircuitSim LEVEL=72 control converge; the served LEVEL=73 checkpoint remains
an explicit nonconvergence with a recoverable, non-scoring diagnostic. The
3/5/9/17-device generated-bias fanout ladder remains NMOS-only.

Merged the duplicated DirectNet-Full and BSIM-AR-Full contract modules into one
parametrized family suite, renamed the remaining release-stamped root tests by
their durable responsibilities, and deleted the closed one-off LEVEL=72
ring/opamp diagnostic. Standalone gates now share fail-closed comma-selection
parsing, and package/README version identity is collected. Verification:
567 tests passed, 4 `simple-v1` + 30 `simple-v2` catalog cases, 4,911 static
render/parity cells, and 600/600 clean campaign jobs.

### V7.6.9 — harness coverage audit: untested features and engine agreement (2026-09-04)

Asked what the harness does not test at all, rather than whether the catalog is
consistent. The executable review is
[`v769-harness-audit.md`](accuracy/v769-harness-audit.md). No diagnostic was
promoted, no threshold moved, and the frozen `simple-v1` `/20` denominator is
unchanged.

**Coverage restored.** Three simulator-free gate suites — the catalog contract,
the 4,854-cell render/parity canary, and the 600-job campaign tooling — were
reachable only by running each script by hand and never entered
`pytest -q tests`, the run `tests/README.md` calls authoritative. They are now
collected. Five V7.5 core gates had been deleted while `tests/common/core_gates.py`
still advertised them, leaving `Inductor`, `TransientSolver(integration_method=)`
and in-place `set_temperature` with no test anywhere; transient branch currents
and the current-source sign convention had none directly. All five questions
return as hermetic contracts, the RL check against `jwL/(R + jwL)` rather than
against PyCircuitSim itself. Collected suite: 256 to 357 tests.

**Engine agreement is now checked, not assumed — and one reading was wrong.**
Candidate/reference parity compares two decks rendered from one template; it
cannot see a card that NGSPICE honours and PyCircuitSim drops. Every template
and every rendered deck is now held against the parser's real support surface.

`Parser._parse_value` read `m` as mega where SPICE — and therefore the NGSPICE
reference — reads milli, and refused `meg`, `mil` and trailing unit text:
`Rload out 0 1m` was 1 MOhm to the candidate and 1 mOhm to the reference on a
byte-identical deck, and no parity check could see it. The parser now follows
SPICE, against a scale-factor table **measured on NGSPICE 45.2** rather than
declared. `Parser._eval_expr` carried a second copy of the same bug and could
not evaluate `10n` or `1e-3` in a `{...}` subcircuit parameter at all. Nothing
in the repository used the suffix, so no result moves.

Directives the parser drops are no longer dropped in silence:
`Parser.PHYSICAL_DIRECTIVES` names the ones NGSPICE acts on that change the
circuit, and each is warned once per deck. Presentation-only cards stay quiet,
and every deck still parses. `.op` is recorded as honoured through the
no-analysis fallback rather than silenced.

`AGENTS.md` claimed `.options cshunt`/`rshunt` were applied; the only
implementation was in the AnalogGym bench translator deleted in V7.6.6. The
contract now states that, keeps the measured V7.5.10 lesson for any
reimplementation, and the parser warns when it drops the card.

**Enrichment.** Four-terminal currents and the 4x4 transcapacitance matrix now
accept the same fourteen-corner matrix `verify_device_integrity` already swept,
opt-in through `--corner` and defaulting to `nominal`, so the charge surface the
LEVEL=75/76 families exist to provide is no longer measured at one temperature
and one geometry. `cascode_ac` was the only AC profile with an empty metric
contract; it now scores each polarity's gain and headlines the worse of the two.

**Repairs.** Ten gate entry points ignored their argument vector — `--tech` was
silently dropped and `--help` launched NGSPICE on six of them — and now reject
an unknown flag. Six instructions pointed `NGSPICE_BIN` at a bundled
`tools/ngspice-45.2` that does not exist in this checkout. Two stale module
paths and one empty directory were removed.

**One regression, caught and fixed.** Reordering `UNIT_SUFFIXES` broke
`_eval_expr`'s dict lookup and took `verify_subckt` from 11/11 to 7/8. The
collected unit suite stayed green throughout; only re-running the
simulator-backed gates found it. Both the lookup and the exponent handling are
now under contract, and the gate is back at 11/11.

**Kept, with the reason recorded.** `verify_nn_inverter`/`verify_nn_dc` are
configuration subsets of the parametric gates and call the same suite bodies,
but apply the tight qualification thresholds where the parametric gates apply
loose stress thresholds; that is one gate per question. The two full-terminal
family contract modules duplicate four questions across families and are
flagged for a parametrized merge rather than merged unilaterally.

### V7.6.8 — fail-closed circuit evidence and missing-model coverage (2026-09-03)

Kept the published simple-v1 `/20` score unchanged and froze SHA-256 hashes for
all 40 rendered candidate/reference decks (four cases, five technologies, two
adapters). The collected unit suite and catalog checks now fail if a later
renderer change moves any of those bytes.

**Evidence harness.** A partial or unconverged solve can no longer be emitted
as a characterized diagnostic. Traces require finite, monotonic, complete axes;
one declared DC/transient increment is allowed only for simulator endpoint
roundoff. Metric profiles declare required finite outputs, result rows record
the selected LEVEL=73–76 family, checkpoint pins, campaign digest, thread
settings, execution state, and error origin, and campaign collection rejects a
missing, duplicate, or unexpected catalog marker. Physical parity now includes
passive/source values, source waveforms, temperature, initial-condition values,
analysis cards, VT, L, and NFIN. PyCircuitSim LEVEL=72 is available as a third,
opt-in control adapter so solver-owned failures remain inconclusive.

**Catalog repair.** Moved the passive RC deck to `controls/`; moved the coupled
inverter, transmission-gate DC, transmission-gate hold, and forced-input SRAM
half-cell to tiers matching the crutches they actually remove. Collapsed the
duplicate `mos_ratio_reference` topology into a high-impedance analysis of
`diode_load`. Corners now include alternate/asymmetric VT, independent N/P
length, and high-NFIN cases, and are filtered per analysis/device role so a
no-op corner cannot create a denominator row. Named device roles support
independent L/NFIN/VT and distinct baked OSDI aliases.

**New diagnostics.** Added physical four-terminal current/KCL sweeps and a
four-excitation 4x4 transcapacitance matrix; NN floating-bulk common-source AC;
inverter leakage, delay, and switching energy; both SRAM states and write
directions; L4 closed-loop AC/PSRR/output-impedance analyses; an NMOS/PMOS
active-mirror-loaded differential stage; a 3/5/9/17-device generated-bias
fanout ladder; a 12-MOS cold-start feedback proxy; and flat-versus-nested NN
subcircuit execution for all four NN families. Device-integrity, terminal, and
hierarchy suites are now part of campaign generation and coverage.

**Instrument corrections found during smoke qualification.** Sequential
analysis substitutions had frozen generated AC sources at zero; overrides are
now registered before recursive expansion. NGSPICE `.op` scales and one-step
DC/transient endpoint differences are canonicalized without accepting arbitrary
truncation. The first active-load DC sweep, fanout transient, and wide 12-MOS
DC transfer were withdrawn because LEVEL=72 controls showed they were
solver-owned or reference-invalid; fixed-bias OP or already-qualified
transient/AC questions replaced them. TSMC12 LEVEL=73 smokes produced complete
reference/control rows for terminal integrity, active load, the 17-device
ladder, inverter energy, and NN hierarchy. These are diagnostics, not a new
published score or threshold campaign.

**Post-change harness audit.** The executable review in
[`v768-template-harness-audit.md`](accuracy/v768-template-harness-audit.md)
fixed a derived-row CLI crash, an unmeasurable common-source bandwidth,
PMOS subthreshold ordering, corner/role geometry enumeration, incomplete-axis
acceptance, physical-parity gaps, stale diagnostics, and fail-open campaign and
parametric-sweep exits. `mos_ratio_reference` was merged into
`diode_load/load_high`; active-load and 3/5/9/17-device scale topology now lives
entirely in explicit L3 templates, as do the 3/5/7/9-stage ring variants. The
legacy device-AC, opamp-AC, and NN parametric suites now emit complete,
provenance-bound result rows; historical regex-only logs cannot satisfy
coverage. The frozen `simple-v1` cells remain byte-identical and no diagnostic
was promoted.

**AC gate-definition correction.** Device and opamp AC comparison now uses one
physical bias located by the LEVEL=72 reference, rather than independently
moving each model to its own peak-gain point. Both adapters are parity-checked
at that bias, raw DC/AC axes are validated before interpolation, and each row
reports MRE, R², NRMSE, and maximum error in addition to its AC figures of
merit. The historical device-AC `/10` and opamp-AC `/5` results used the old
per-engine-bias definition and are not comparable to the current gates; they
remain historical records pending a new five-technology campaign. A pinned
TSMC12 LEVEL=75 smoke passed both device polarities and produced an 8.91 dB
opamp gain miss under the corrected shared-bias experiment.

**Legacy sweep closure.** The circuit-parametric driver now executes every
declared cell even when its baseline misses, rejects duplicate technology
selections, and validates complete DC/transient axes for opamp, ring,
switched-capacitor, and SRAM runs. Partial transients, nonoscillation, and
unmeasurable reference SNM remain explicit `ERROR` rows and never enter numeric
aggregates.

**Subcircuit seam decision (2026-09-04).** Kept the nine hierarchy fixtures in
their orthogonal `circuit_templates/subcircuits/` directory rather than
distributing them by compact-model difficulty. The flat decks are independent
flattening oracles, not redundant accuracy cases. The standalone harness now
rejects unconverged DC prerequisites and incomplete candidate/reference traces,
uses the same complete AC-axis constructor as production orchestration and the
catalog harness, and verifies the claimed nested L/NFIN propagation. DEC/OCT
cards now match NGSPICE on integral and fractional bands. Sub-decade DEC and
all OCT sweeps advance by their native points-per-band ratios, including the
next point just above the upper bound when NGSPICE's default frequency tolerance
admits it; DEC spans of at least one decade distribute
`floor(points*decades)+1` samples across both bounds. Its 11 parser, linear,
LEVEL=72, and NGSPICE checks remain green.

### V7.6.7 — evaluation coverage: device integrity, self-bias, and feedback (2026-09-02)

Motivated by a contradiction the reports already carried: DirectNet-Full L75
`large` scores 20/20 strict simple-v1, 10/10 device AC and 100/100 inverter
configurations while solving 0/248 AnalogGym decks, and V7.6.6 then retired
that corpus as an executable gate. The plan and its diagnosis are in
[`docs/plans/2026-09-02-v767-evaluation-coverage.md`](plans/2026-09-02-v767-evaluation-coverage.md).
Everything added here is a diagnostic; the simple-v1 `/20` denominator, its
four cases, and its thresholds are unchanged, and the frozen cells still
render byte-identically.

**Template tree.** Renamed `examples/` to `circuit_templates/` and reordered it
by what a circuit demands of a compact model rather than by application:
`L0_devices`, `L1_primitives`, `L2_stages`, `L3_blocks`, `L4_systems`, plus the
orthogonal `subcircuits/` parser fixtures. Each catalog case declares its
`tier`, and `template_deck()` resolves a bare template name to its tier,
rejecting a name that no tier owns or that two tiers both claim.

**Single-device integrity.** Added `verify_device_integrity.py` and
`tests/common/device_integrity.py`: output characteristics (`gds`),
subthreshold decades (`Ioff`, subthreshold slope), the triode region (`Ron`,
origin symmetry), and `gm`/`gds`/`gmb` differentiated from both engines with
the identical stencil. Both engines are measured as `id = -i(Vds)`, one
definition rather than a per-engine correction. Previously the only scored
device sweep was Id–Vgs on a linear axis at a single `Vds = 0.5*VDD`.

**Expanded existing coverage.** Mirror ratio across a reference-current range;
opamp CMRR and PSRR from a differential/common-mode/supply AC triple; CMRR
reported for both differential pairs; SRAM write-margin trip point; a
ten-period switched-capacitor accumulation; ring period plus dynamic supply
current. Source specs became tokens so one opamp topology serves both the
frozen DC transfer and the new rejection experiments.

**New held-out cases.** Tier A removes the ideal bias: `diode_load`,
`beta_multiplier`, `self_biased_cascode`, `mos_ratio_reference`. Tier B closes
a negative-feedback loop: `unity_gain_buffer`, `ota_5t_buffer`,
`ldo_regulator` — the last has seven coupled MOSFETs with all bias internal.
The original above-ten-device claim was incorrect and is retracted by V7.6.8.
Every L4 case runs at
least one transient without `uic`, and the catalog check enforces it; before
this, every transient in the catalog was handed its initial state through
`.ic`.

**Reporting.** Convergence is reported beside error, never folded into it. A
DC row that does not converge stays an `ERROR` and keeps its denominator slot,
but now carries an `unconverged_diagnostic` payload from one repeat run
without the convergence requirement, which no scoring path reads. Signals
whose reference has no dynamic range are held out of the case aggregate and
the exclusion is recorded, so a sub-millivolt error on a reported bias rail no
longer produces a six-figure NRMSE.

**Contract checks added,** each verified to fail when violated: a case
declaring no device kinds must have no body token in its template; a derived
metric must name a known ratio; an L4 system must declare a transient and not
every one of them may use `uic`.

**Measured on the served DirectNet L73 `large` checkpoints.** Reported in
[the coverage report](accuracy/device-and-feedback-coverage-v767.md). The
sharpest result is a two-element reproducer: a diode-connected NMOS fed
through a resistor from the rail does not satisfy the convergence contract,
while the same device fed by an ideal current source does, and LEVEL=72
converges on both through the same solver. Every diode-connected device in the
previous catalog was fed by an ideal current source.

**Campaign wiring needed no changes.** `v710_regate_jobs.py` and
`v730_coverage.py` derive their suites from the catalog, so all 22 simple-v2
cases enumerate automatically.


### V7.6.6 — clean repository and full-terminal requalification (2026-09-02)

- Retired the AnalogGym `examples/complex_circuits/` corpus and its dedicated
  tests, scripts, campaign adapters, and current-document links. Historical
  measurements remain in this ledger and Git history but are no longer an
  executable compact-model gate.
- Replaced 48 separately maintained `.sp`/`.cir` example decks with 29 strict
  `.spice.tmpl` sources, including flat/hierarchical subcircuit fixtures.
  Candidate NN and LEVEL=72 reference decks now render
  from one topology while exposing technology, VT, independent P/N geometry,
  PVT, body bias, slew, load, bias, timing, and analysis tokens.
- Removed `verify_complex_*` and `PYCIRCUITSIM_COMPLEX_RESULTS` compatibility
  aliases. Campaign enumeration, collection, coverage, and report generation
  now use canonical `verify_circuit_*` suite IDs and fail loud on stale names.
- Moved persistent test decks and simulation artifacts from `tests/` to
  `results/tests/`, changed every default output root, and added a catalog guard
  against materialized decks or result files returning to source directories.
  Root pytest discovery now excludes archived worktrees under `results/`.
- Added `circuit_templates/README.md` and `tests/README.md` as the template and test-tree
  contracts.
- Removed unused internal buffers, telemetry, normalization wrappers, imports,
  locals, historical collectors, retired AnalogGym-only tests, and orphaned
  scratch reports. Public APIs, ctypes ABI fields, and checkpoint-compatible
  optional model structures remain intact.
- Hardened full-terminal provenance: dataset generation treats every
  nonignored untracked file as dirty, the campaign manifest requires the whole
  worktree to be clean, and both LEVEL=75 and LEVEL=76 bundles must carry the
  checksum-bound source-dataset identity.

#### Versioned simple-topology diagnostics

- Versioned the existing ring/opamp/SRAM-SNM/switched-capacitor qualification
  matrix as `simple-v1` without changing its `/20` denominator. These simple
  circuits continue to live under `circuit_templates/`.
- Added 12 held-out `simple-v2` topology pairs spanning source-driven stages,
  mirrors/cascodes, open logic chains and stacks, transmission gates,
  differential pairs, and full 6T SRAM modes across DC, transient and AC.
  Every candidate/reference pair is strictly rendered and topology-checked
  before simulation against the identical LEVEL=72 OSDI reference.
- Added engine-neutral multi-signal traces, signed-current/domain metrics,
  structured gate-result markers, LEVEL=72 accepted-trajectory support checks,
  reference-repeat diagnostics, temperature/body/supply/fin-ratio/joint
  corners, and catalog-driven geometry coverage.
- Added a separate `simple_v2` campaign pool, version-selectable coverage and
  diagnostic report collection. The historical clean pool stays 480 jobs;
  the new diagnostics do not enter qualification totals until a new frozen
  score version satisfies the documented promotion rule.

### Post-V7.6.4 — LEVEL=76 simple-circuit recovery (2026-09-01)

- Made the parser apply the global Celsius `.temp` card to LEVEL=72–76
  devices independent of card order. Existing devices rebind through their
  cache-clearing `set_temperature` contract; later devices receive the
  selected Kelvin value at construction. Decimal conversion is canonicalized
  so `-25 C` remains inside a checkpoint whose lower support edge is exactly
  `248.15 K`.
- On the fixed TSMC5 LEVEL=76 large DC denominator, the unchanged 20-epoch
  checkpoint improved from 22/26 to 25/26 with no `ERROR` rows. PMOS `+125 C`
  and both joint length/NFIN/temperature corners changed from fail to pass;
  NMOS `+125 C` remains a model failure at 21.86% NRMSE.
- Extended the value-only subthreshold loss to the full-terminal `i_d` column
  and LEVEL=76 training. It remains compatible with BF16 autocast because it
  does not build a derivative or double-backward graph; full-terminal
  Sobolev losses remain rejected because the six-surface data has no
  derivative labels.
- Added an opt-in training-overlay split contract for circuit-derived sample
  classes. It promotes whole technology/VT/L/NFIN/temperature strata into the
  training partition, reports the movement, rejects unknown classes and
  random splitting, and leaves default combo splits unchanged. This caught a
  targeted L=16 nm, 398.15 K hot-NMOS overlay whose 12,789 rows otherwise
  landed entirely in validation.
- Added opt-in LEVEL=76 autoregressive fine-tuning so the current tail and
  later charge heads train against predicted charge prefixes, matching the
  deployed rollout instead of ground-truth teacher-forcing prefixes. Existing
  training remains unchanged by default, and both bundle sidecars record the
  selected mode.
- Retrained and CPU-gated a fresh TSMC5 S/M/L/XL matrix. All tiers pass 26/26
  DC configurations; inverter VTC/transient scores are 20/20, 20/20, 20/20,
  and 19/20. Strict non-opamp complex passes are 3, 2, 3, and 2 of three
  scorable cells, with the Miller opamp remaining an explicit error at every
  tier. Medium's teacher checkpoint is retained because it passes 2/2 device
  AC versus rollout's 1/2 without losing another circuit pass.
- Selected Small, targeted Large, and XL rollout bundles plus the Medium
  teacher bundle into one checksum-valid S/M/L/XL artifact root. Large has the
  best aggregate DC/transient error among the two three-circuit tiers; the
  full tables and provenance are in
  [the LEVEL=76 simple-circuit report](accuracy/BSIM-AR-L76-simple-circuits.md).
- Rejected polarity hybrids, support-aware stamping, and a physical line
  search. The Medium hybrid worsened switched-capacitor droop to 1.598 mV
  against a 0.650 mV allowance, while the runtime experiments delayed or
  moved the same opamp failure without producing a new pass.

### Post-V7.6.3 — V7.6.4 closure loop and cleanup (2026-08-29 to 2026-08-31)

- Regenerated terminal-upper-edge full-terminal data and eliminated the
  dominant length-support error without admitting unstable NFIN=1 points.
  Support-aware limiting, matched-row appends, and exact OSDI current-J
  fine-tuning still left the fixed TSMC5 development basket at 0/15 or lost
  the Song basin; no runtime or checkpoint was promoted.
- Ran a bounded 15-cycle basin-entry investigation. Normalization anchoring,
  all-geometry Hermite training, pass-current correction, reference-seeded
  unrolled loss, a 0.1 V production cap, cold residual/step distillation, and
  affine/quadratic C2 local adapters all failed a predeclared device or
  production-circuit gate. The last quadratic arm passed replay, PMOS, closure,
  support, and reproducibility checks but reached 1.022918x held-out NMOS
  normalized MAE against the 1.02x limit.
- The strongest static candidates still solved production LDO 0/28. The
  cold-loss treatment best reached residual/step ratios 1.29005/0.931470
  against required 0.50/0.50. No later circuit or /248 campaign was opened,
  no candidate was published, and LEVEL=75 remains 0/248.
- Removed experiment-only trainers, harvesters, fitters, private tests,
  solver-factory and unpublished adapter hooks, plus rejected V7.6.4–V7.7.4
  result/checkpoint payloads. Current V7.6.2/V7.6.3 qualification evidence and
  the active five-technology DirectNet large bundles remain.
- Retained fail-closed single-deck/campaign agreement checks: finite values,
  complete sweeps, converged DC states, no truncation, successful NGSPICE, and
  the fixed denominator are all required.
- The condensed hypotheses, measurements, stop rules, and future boundary are
  in [the closure plan](plans/2026-08-29-v764-complex-circuit-closure-loop.md).

### V7.6.3 — targeted LEVEL=75 recovery (2026-08-29)

- Added terminal rail endpoints, technology/polarity-specific pass-device
  guards, and a LEVEL=75-only 0.1 V DC/transient Newton-step cap. Fine-tuning
  used manifest-pinned V7.6.2 controls.
- On the targeted CPU gates, large improved inverter VTC/transient from
  91/100 to 100/100, op-amp AC from 0/5 to 3/5, and strict simple circuits
  from 5/20 to 20/20; device DC stayed 114/129 and device CS AC 10/10.
- The four-tier targeted matrix scored strict circuits 16/20, 20/20, 20/20,
  and 19/20 from small through XL. It was warm-started development evidence,
  not a clean qualification.
- All four tiers remained 0/248 on tracked AnalogGym. Per tier, 184 rows
  requested absent terminal lengths and 18 requested excluded NFIN=1. Larger
  capacity did not solve the corpus failure.
- Rejected a device-boundary limiter, an old PMOS warm start that cut device
  DC to 58/129, normalization-only support widening, and an incomplete
  LEVEL=75 detector. Details are in
  [DirectNet-L75-v763-targeted.md](accuracy/DirectNet-L75-v763-targeted.md).

### V7.6.2 — clean DirectNet-Full qualification (2026-08-28)

- Corrected the PMOS scalar-current comparison sign without changing the
  solver-positive full terminal stamp. Added rail and terminal-length samples,
  isolated data/checkpoint roots, and checksum-bound dataset-to-checkpoint
  provenance.
- Regenerated ten datasets, trained all 40 clean bundles, and completed one
  CPU-pinned 240-job pass without coverage, artifact, thread, or
  infrastructure gaps. Strict circuits scored 8/20, 5/20, 5/20, and 7/20
  from small through XL.
- Re-ran all 255 tracked AnalogGym cells with large. Seven invalid decks stayed
  quarantined; the scored verdict was 0/248, with 41/326 comparable metric
  cells agreeing, 1,155 missing PyCircuitSim values, 174 Py failures, and
  three NGSPICE failures.
- LEVEL=75 was rejected for promotion; LEVEL=73 large remains served. The
  source design tree was absent, so this is a complete tracked-deck rerun, not
  a refreshed topology audit. See
  [DirectNet-L75-clean.md](accuracy/DirectNet-L75-clean.md).

### V7.6.1 — full-terminal BSIM-AR and campaign integrity (2026-08-27)

- Added experimental LEVEL=76 FAMILY=bsimar-full with six independent
  terminal surfaces and analytical current/charge closure. Isolated tff
  artifacts require checksum-valid model, normalization, configuration, and
  completion sidecars.
- Extended parser, solver discovery, CPU gates, manifests, report generation,
  and AnalogGym provenance to LEVEL=76 while leaving batched/fused evaluation
  disabled.
- Made full-terminal generation preserve audited rejection coordinates and
  reason counts while failing on dropped bins or unknown exceptions.
- The initial DirectNet-Full 240-job pass scored 0/20, 2/20, 0/20, and 1/20.
  Its later AnalogGym denominator was corrected from a misleading 0/218 to
  0/248: rows with missing required metrics remain scored failures.

### V7.6.0 — attributed boundaries and LEVEL=75 introduction (2026-08-27)

- Fixed NN instance multipliers across current, conductance, charge, and
  capacitance paths. TSMC5 LDO maximum node error fell 12.3250 V to 0.306632 V,
  but the deck remained 0/3.
- Exact reduced OSDI passed LDO but failed a high-temperature Fan sweep at
  12/15 with an 817.3 V maximum state error, establishing that a full-terminal
  interface was required.
- Introduced experimental LEVEL=75 FAMILY=directnet-full: three independent
  currents and charges, analytical source closure, full 4x4 Jacobians, lazy
  charge derivatives, separate dnf datasets/checkpoints, and fail-closed input
  support.
- Generation probes were diagnostic; no checkpoint or circuit result was
  promoted. See
  [DirectNet-L75-V760-recovery.md](accuracy/DirectNet-L75-V760-recovery.md).

## V7.5 — AnalogGym migration and evidence repair

### V7.5.17 — audited clean matrix and PFN retirement (2026-08-26)

- Retired PFN/TabPFN and its old LEVEL=75 path before the new full-terminal
  family reused that level in V7.6.
- Closed the simple-circuit coverage audit: convergence, signed current,
  temperature, body bias, reverse VDS, joint corners, exact ratio cells, and
  339/339 requested geometry coverage became binding.
- Canonical generation now fails on dropped bins/rejected rows and binds
  command, source, commit, requested geometry, artifacts, and completion
  markers. Campaign logs bind immutable source, job, PDK, OSDI, NGSPICE, and
  checkpoint provenance.
- One CPU-pinned 480-job clean pass completed with no infrastructure errors.
  Strict circuits from small through XL were DirectNet 5/7/9/10 and BSIM-AR
  9/9/12/11 out of 20. Device and op-amp AC stayed 0/10 and 0/5 because the
  required NN operating points did not converge.
- The 6–14 hour BSIM-AR XL AC tail was CPU work, not a hang: repeated failed
  DC operating points dominated. The floating-point-perturbing AR prefix
  cache remained default-off.

### V7.5.13–V7.5.16 — layout, retraining, and honest gates (2026-08-19 to 2026-08-25)

- Moved PDK ownership to PDKs/, separated the BSIM-CMG evaluator from the
  neural-network stack, and removed closed campaign debris.
- Retrained ten DirectNet large bundles. Focused device/inverter/ring/SRAM/
  switch-cap gates largely passed, but Miller op-amp and corrected device AC
  did not. The pinned LEVEL=73 AnalogGym verdict was 0/248 with 35.41% MRE,
  -42.66 R², 73.22% NRMSE, and 12.696 V maximum error over 80,299 samples.
- Retracted an initial 10/10 device-AC claim because all DC linearization
  states were unconverged, and retracted a 54,045-sample voltage aggregate
  that omitted later sweep/recovery segments.
- Repaired race-corrupted logs, checkpoint-root selection, missing/invalid
  completion handling, and report completeness. Raw logs override stale JSON;
  explicit ERROR rows remain in denominators; tracebacks are infrastructure
  failures.
- Corrected MNA residual probes to recover ideal-source branch currents and
  corrected high-gain AC gates to refine each simulator's physical bias.
  A nonexistent-interpreter 480-job run was quarantined; drivers now fail
  before dispatch.

### V7.5.8–V7.5.12 — measured corpus and transient closure (2026-08-13 to 2026-08-18)

- Consolidated one circuit library/tool tree, then measured and reduced the
  corpus from 375 to 255 rows while preserving every surviving verdict and all
  discriminating metrics. The measured runtime cut was 47%; the earlier
  structural 55% estimate was retracted.
- Fixed repository-root campaign imports, nodeset clamp/release semantics,
  PULSE breakpoint coalescing, flattened-node cshunt/rshunt application, and
  matched transient sampling.
- The final LEVEL=72 tracked basket reached 242/248: AC 139/139, source DC
  30/30, temperature DC 45/45, and transient 28/34. Seven invalid decks are
  explicit NOT_COMPARABLE outcomes, never passes.
- Retracted broad Leung/Peng instability claims after discovering mismatched
  transient resolution. The remaining six transient misses stayed scored.
- Dead ends: global stride 1 cost 15x and worsened charge-pump agreement;
  control-mode cshunt removal did not undo parse-time shunts; lowered decks
  could not reconstruct the absent source design tree.

### V7.5.0–V7.5.7 — migration foundations (2026-08-10 to 2026-08-13)

- Imported/translated the AnalogGym corpus and added shared DC, AC, transient,
  temperature, measurement, provenance, and campaign infrastructure.
- Fixed full-terminal LEVEL=72 stamps, source-relative OSDI evaluation,
  convergence/GMIN/limiting behavior, transient history/retries, PULSE
  handling, substepping, and parser parity.
- Rejected local equation substitutes, partial capacitance rows, unmatched AC
  GMIN, output-only interpolation, indiscriminate BDF-2, and presentation-only
  timestep changes because they did not solve the same problem as NGSPICE.

## V7.4 — clean rebuild and repository consolidation

- Vendored PyCMG, clarified generated/private artifact boundaries, and pruned
  stale plans, results, and scripts.
- Rebuilt 40 DirectNet and 40 BSIM-AR clean checkpoints across five scopes and
  four sizes. DirectNet large served at 14/20; XL reached 15/20 at 2.3x cost.
  BSIM-AR declined 18→17→15→13 from small through XL.
- CPU/GPU binding gates matched 12/16; CUDA stayed opt-in. The clean rebuild
  did not reproduce V7.3 recipe peaks.

## V7.3 — normalized evidence

- Regenerated reports from one committed, complete campaign; separated clean
  controls from recipe addenda and centralized gate/OMP/denominator rules.
- Historical recipe peaks were DirectNet 19/20 and BSIM-AR 20/20. They are not
  clean-rebuild or current production verdicts.

## V7.2–V7.0 — performance and corrected accuracy

- Added topology-versioned caches, batched/fused NN tails, AR caching, CUDA
  selection, thread control, and charge-Jacobian avoidance in DC/OP.
- Shipped bit-identical optimizations after focused gates; floating-point-
  perturbing paths stayed opt-in.
- Measured and rejected TF32, torch.compile, and bfloat16 DirectNet inference.
- Restored TSMC6 as an independently trained repeat and retracted comparisons
  that mixed pre- and post-gds-fix code states.

## V6 — foundational milestones

### V6.12–V6.13 — hierarchy and silent-green audit

- Added subcircuit flattening, validation, model/include hoisting, and
  flat-versus-hierarchical gates.
- Fixed 22 validation/parser/solver/model issues and replaced abs(gds) with a
  sign-preserving floor. Every checkpoint was re-gated.

### V6.8–V6.11 — model families and TSMC6

- Reintroduced BSIM-AR LEVEL=74, added resolver/force-level support, and filled
  its XL matrix.
- Trained TSMC6 as the deliberate TSMC7-ground-truth repeat.
- Studied the former PFN/TabPFN LEVEL=75 path; its cost/accuracy result kept it
  research-only before retirement in V7.5.17.

### V6.6–V6.7 — recipe and universal studies

- Completed curriculum and universal DirectNet studies. Historical recipe
  peaks reached 14/16–15/16 depending on the then-current denominator;
  universal models remained explicit-pin only because per-tech models were
  more reliable.
- Removed incomparable or generated campaign artifacts between rounds rather
  than mixing them into later claims.

### V6.4–V6.5 — production baseline, AC, and circuit training

- Established per-technology DirectNet data/checkpoints and the serialized
  14/16 baseline, including 8/8 hard-IC SRAM and the lifted-source canary.
- Added AC analysis and derivative-sensitive NN AC gates. Fixed a switch-cap
  harness clock bug and retracted the affected model diagnosis.
- Differentiable DC training produced the first historical 16/16 result, but
  later evidence normalization distinguishes that recipe peak from clean
  production checkpoints.

## Earlier history

Before V6, the project progressed from the pure-Python simulator through
BSIM-CMG/PyCMG integration, SPICE-compatible parsing, DC/transient solvers,
initial neural compact models, and the first NGSPICE comparison harnesses.
Those releases predate the current artifact and evidence contracts; use Git
history for their chronology.
