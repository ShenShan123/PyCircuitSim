# PyCircuitSim changelog

This is the compact project-evolution ledger. It records what changed, the
release verdict, retractions, and failed approaches worth not repeating.
Commands belong in [`../README.md`](../README.md), durable implementation
rules in [`../AGENTS.md`](../AGENTS.md), compact-model evidence in
[`accuracy/`](accuracy/), and the AnalogGym tables in
[`../examples/complex_circuits/RESULTS_TSMC.md`](../examples/complex_circuits/RESULTS_TSMC.md).

The pre-compaction long-form narrative remains available in Git history.

## Reading historical scores

- V7.3.0 folded the deliberate TSMC6 repeat into compact-model headlines:
  complex `/16 → /20`, device AC `/8 → /10`, and opamp AC `/4 → /5`.
- AnalogGym denominators changed as invalid or redundant decks were
  quarantined/pruned. Compare totals only when an entry says the basket is the
  same.
- Current detailed scoreboards are generated evidence, not maintained here.

## V7.5 — AnalogGym migration

### V7.5.12 — transient diagnostics and corpus integrity (2026-08-18)

- Replaced the unrelated unconstrained `op` used by transient diagnostics with
  NGSPICE's time-zero transient state. Charge-pump startup deltas fell from
  94.6–160.9 mV to 0.21–3.04 µV; the worst of all 40 transient rows was
  56.6 µV.
- Reduced 25 automatically emitted alternate-seed amplifier helpers to the one
  independently validated TSMC5 `Qu2017_AZC` fallback.
- Re-ran 255/255 rows without analysis errors or non-finite metrics. The scored
  electrical verdict remained 242/248 (97.6%): AC 139/139, source DC 30/30,
  temperature DC 45/45, transient 28/34, plus seven invalid deck cells outside
  the denominator.
- Added direct measurement-engine controls: 1,265/1,265 comparisons on 248
  monolithic rows and 301/301 on the saved segments behind seven recovered
  amplifier DC rows.
- Corrected evidence provenance: all 289 saved rawfile paths resolve to their
  final baskets; the report now separates fresh results from historical
  topology evidence.
- Dead end: faithful regeneration was attempted, but the untracked upstream
  `examples/complex_circuits/designs/` source tree is absent. Lowered decks
  cannot reconstruct it. Regeneration now fails before mutation when the
  source tree or required inputs are incomplete.

Open after this release: six scored transient quantities at 2.1–4.4%; the
source-topology audit cannot be refreshed until the upstream design tree is
restored.

### V7.5.11 — matched transient resolution and invalid-example quarantine (2026-08-14)

- Found that campaign transient `stride` changed PyCircuitSim's integration
  step while leaving NGSPICE unchanged. Removing amplifier/LDO transient
  stride policies closed 11 of 20 misses and moved the headline from 215/255
  to 242/248 (97.6%).
- Retracted the V7.5.10 broad claims that `Leung_NMCNR` was generally an
  unstable-equilibrium mismatch, that `Peng_IAC` fell below its reproducibility
  floor, and that no simulator action remained. Most affected cells agree at
  the deck timestep; only the measured TSMC16 Leung case remains invalid.
- Added `NOT_COMPARABLE` and quarantined seven deck cells whose reference
  quantities were tolerance-unstable or non-commensurate. They remain visible
  and never count as passes.
- Dead end: forcing stride 1 on the charge pump cost 15× and worsened agreement
  (typically 4/6 instead of 5–6/6). Its validated stride-20 exception remains.
- Left six transient misses scored: three `Qu2017_AZC`, two `Song_DACFC`, and
  one TSMC12 charge-pump maximum.

### V7.5.10 — nodesets, breakpoints, and node shunts (2026-08-13)

- Implemented SPICE-compatible `.nodeset` clamp-then-release semantics. This
  recovered the intended `Song_DACFC` operating-point basin and reduced hard
  amplifier startup cost.
- Coalesced and matched PULSE breakpoints using a `CKTminBreak`-style tolerance,
  eliminating one-ulp corner misses and restoring charge-pump agreement.
- Applied `.options cshunt`/`rshunt` to every flattened node; 85 corpus decks
  had declared shunts that the translator previously ignored.
- On the then-current 255-row basket, the result improved from 203/255 to
  215/255 while runtime fell from 1.77 to 1.15 CPU-hours.
- Retracted the V7.5.3 claim that charge-pump controller tuning did not transfer;
  missed floating-point breakpoints were the cause.
- Later correction: the release's Leung/Peng characterizations were based on
  mismatched transient resolution and were narrowed by V7.5.11.
- Dead ends: NGSPICE control-mode `option cshunt=0` does not remove parse-time
  shunts; finer strides/method swaps did not explain Leung under the mismatched
  campaign; input-edge resolution did not explain the Song residual.

### V7.5.9 — measured corpus minimization (2026-08-13)

- Fixed the V7.5.8 campaign regression by pinning `AG_TREE` before lazy tool
  imports; all campaign rows had otherwise failed from the repository root.
- Measured the full 375-row basket, then reduced each technology from 18
  designs/75 decks to 12 designs/51 decks. The reduced 255-row campaign
  preserved every surviving verdict and all 34 metric names that had ever
  disagreed, while cutting runtime from 3.35 to 1.77 CPU-hours.
- Measurement overruled structural intuition: retained `Leung_NMCNR` and
  `Song_DACFC` because they exposed unique failures; removed expensive
  `Qu_LEC` because cheaper decks covered its failure class.
- Simplified `examples/` and `tests/` so each gate uses the shared deck library
  and answers one question.
- Dead end: deck-level pruning could not preserve design initialization and
  helper relationships reliably; pruning stayed design-aware.

### V7.5.8 — one circuit library and one taxonomy (2026-08-13)

- Moved circuit sources into `examples/{single_devices,simple_circuits,
  complex_circuits}` and gates into matching test groups.
- Removed embedded/private test netlists; gates now render paired `.sp`/`.cir`
  sources through shared infrastructure.
- Consolidated five duplicated AnalogGym technology tool trees into one
  package and repaired references, generated-result ignores, and sizing audits.
- Re-measured runtime instead of carrying the planning estimate: the structural
  cut saved about 13%, not the previously stated 55%.
- Follow-up: the shared-tool refactor broke campaign imports outside a design
  tree; V7.5.9 fixed the missing `AG_TREE` initialization.

### V7.5.7 — scale-based examples and broken-deck discovery (2026-08-13)

- Introduced the scale-based example hierarchy and surfaced five decks whose
  includes/model names had silently broken during earlier moves.
- Repaired gate paths and clarified `.sp` PyCircuitSim sources versus `.cir`
  NGSPICE templates.
- Left two pre-existing items open: opamp accuracy residuals and generated-tree
  Git-ignore hygiene. Later V7.5.x entries address both areas.

### V7.5.6 — structural corpus curation (2026-08-12)

- Removed structurally redundant AnalogGym designs and established explicit
  category/metric denominators.
- Preserved distinct topology, sizing, threshold, and analysis coverage.
- This was a structural hypothesis only; V7.5.9 replaced it with measured
  discrimination and rescued/cut several contrary cases.

### V7.5.5 — transient refine-controller rebuild (2026-08-12)

- Reworked refine stepping around SPICE `dctran` semantics: accepted-step
  history, real retry substeps, breakpoint restarts, and corner guards.
- Closed the V7.5.4 controller open list without changing flags-off transient
  behavior.
- Rejected fixes that changed only output `tmax`, forced backward Euler, or
  promoted BDF-2 indiscriminately; they added cost or moved errors without
  addressing the controller defect.

### V7.5.4 — internal-solve current floor (2026-08-12)

- Corrected the BSIM-CMG internal-node current floor dimensionally and removed
  a convergence failure source.
- Measured and rejected a proposed refine-controller scaling fix: its units and
  behavior did not match the actual residual mechanism.
- Established that several `min_slope` mismatches were dominated by reference
  noise; V7.5.11 later formalized this class as invalid examples.

### V7.5.3 — fair comparison harness and campaign start (2026-08-12)

- Closed the seven-deck pilot and began the full campaign.
- Added charge-based LTE refinement, fallback handling, shared sampling grids,
  measurement normalization, and cached reference work.
- Recorded the first charge-pump stride trade-off; V7.5.10 retracted the early
  non-transfer diagnosis, while V7.5.11 retained the measured stride-20
  exception.
- Left the refine-step controller as the principal open solver issue.

### V7.5.2 — full AC stamp and LTE output refinement (2026-08-11)

- Completed the four-terminal LEVEL=72 AC current/capacitance stamp and removed
  an unmatched external AC GMIN.
- Added optional LTE substepping while preserving requested output points.
- Closed both V7.5.1 follow-ups.
- Dead ends: partial capacitance rows, external regularization, and output-only
  interpolation could make plots smoother but did not solve the same AC or
  transient problem as NGSPICE.

### V7.5.1 — SPICE robustness for real analog decks (2026-08-11)

- Fixed eleven migration defects spanning full terminal currents/charges,
  source-relative OSDI evaluation, voltage limiting, GMIN stepping, convergence
  acceptance, transient history, PULSE handling, measurements, and parser
  compatibility.
- Converted initial failure classes from analysis crashes into comparable
  numerical results and isolated the remaining AC/transient follow-ups.
- Rejected local equation substitutions and measurement-specific patches;
  fixes were required to improve the shared simulator path.

### V7.5.0 — initial in-repository AnalogGym migration (2026-08-10)

- Imported and translated 190 analog designs across TSMC technologies, with
  reusable generation, comparison, measurement, and campaign tooling.
- Added support needed by the corpus, including aliases, sources/options,
  temperature/DC/AC/transient testbench translation, and sizing audits.
- The first broad run exposed solver robustness, parser parity, and measurement
  comparability as separate failure classes; V7.5.1–V7.5.12 resolve or
  quarantine them.
- Kept the migration at LEVEL=72. NN compact-model evidence remained in the
  established device and compact circuit gates.

## V7.4 — clean rebuild and repository consolidation

### V7.4.1 — vendoring and housekeeping (2026-08-10)

- Vendored PyCMG into the repository, pruned stale plans/results/scripts, and
  made generated/raw PDK boundaries explicit.
- Compacted project documentation while keeping accuracy reports generated from
  evidence.

### V7.4.0 — clean checkpoint rebuild and GPU re-gate (2026-07-30 to 2026-08-06)

- Rebuilt all 40 clean DirectNet and all 40 clean BSIM-AR checkpoints across
  five scopes, four sizes, and two polarities on new hardware.
- DirectNet clean verdict: `large` served production at 14/20 strict with zero
  flips; `xl` reached 15/20 at 2.3× cost for one additional cell.
- BSIM-AR clean verdict: `small` led at 18/20 with zero flips; larger capacity
  declined 18→17→15→13.
- Closed the GPU fidelity axis: the executed/binding gates completed, and the
  strict basket matched CPU exactly (12/16). CUDA stayed opt-in because CPU
  flags-off remains the compatibility contract.
- Confirmed TSMC6 and TSMC7 LEVEL=72 equivalence exhaustively while their
  independently trained NN checkpoints preserved the intended noise-floor
  control.
- Dead end: a complete clean rebuild did not reproduce V7.3 corridor-recipe
  peaks; those remain historical rather than current checkpoint claims.

## V7.3 — evidence normalization

### V7.3.0 — one code state, explicit denominators (2026-07-27 to 2026-07-29)

- Regenerated family reports from one pinned campaign pass with committed-SHA
  and completeness guards.
- Split each family into clean controls and recipe addenda; centralized gate
  definitions, OMP discipline, the gds code-state ladder, TSMC6 interpretation,
  and measured noise floor in `docs/accuracy/methodology.md`.
- Folded TSMC6 into current `/20`, `/10`, and `/5` denominators and archived
  pre-gds-fix claims.
- Historical recipe peaks were DirectNet 19/20 and BSIM-AR 20/20; they are not
  clean-rebuild verdicts.

## V7.2 — performance paths

### V7.2.0 — GPU-oriented large SRAM transient (2026-07-27 to 2026-07-28)

- Added topology-versioned caches, batched/fused NN tails, AR state caching,
  and CUDA selection behind fidelity-aware controls.
- Default-on CPU optimizations passed the prescribed gate bundle; perturbing
  paths remained default-off.
- Added exact batched-tail and latch-basin checks to catch order-dependent
  nonlinear solution changes.

## V7.1 — accuracy pivots and TSMC6 control

### V7.1.0 (2026-07-25)

- Restored TSMC6 as a controlled repeat after its premature retirement in
  V6.13.0.
- Re-measured device/AC behavior on the corrected gds code state, added PFN xl,
  and separated value-surface gains from circuit-gate gains.
- Retracted conclusions derived from incomparable pre-fix versus post-fix
  checkpoint gates.

## V7.0 — NN performance campaign

### V7.0.0–V7.0.4 (2026-07-25)

- Avoided NN charge Jacobians in DC/OP, cached invariant work, reduced framework
  overhead, and added controlled thread selection.
- Kept changes that were bit-identical or fully re-gated; isolated changes that
  altered floating-point order behind flags.
- Measured and rejected TF32, `torch.compile`, and bfloat16 as production
  DirectNet paths on the tested hardware.

## V6.13 — audit and gds correction

### V6.13.1 — systematic audit wave 1 (2026-07-24)

- Fixed 22 gate-neutral issues in validation, parser, solver, and model plumbing
  found by a systematic audit.

### V6.13.0 — gds sign fix and re-gate (2026-07-24)

- Replaced `abs(gds)` with a sign-preserving floor and guarded non-finite device
  outputs. Re-gated every checkpoint because the solver surface changed.
- Temporarily retired TSMC6 as duplicated technology evidence; V7.1 restored it
  for its actual purpose as an independently trained repeat.

## V6.12 — hierarchy and silent-green fixes

### V6.12.1 (2026-07-24)

- Merged the P0 silent-green fixes and split accuracy evidence by model family.

### V6.12.0 — `.subckt`/`.ends` support (2026-07-18)

- Added parse-time subcircuit flattening, model/include hoisting, global ground,
  hierarchical names, recursion/port validation, and flat-versus-hierarchical
  equivalence gates.

## V6.11 — TSMC6 NN family

### V6.11.0 (2026-07-14 to 2026-07-17)

- Trained and gated TSMC6 NMOS/PMOS checkpoints for all families and scales as
  an independent-run repeat over TSMC7-equivalent LEVEL=72 data.

## V6.10 — PFN / TabPFN

### V6.10.0 (2026-07-11 to 2026-07-14)

- Added LEVEL=75, a scaled TabPFN-v3 port with a frozen learned context,
  cached context KV, and a smooth direct 13-output head for NR derivatives.
- Required the configuration sidecar to reconstruct the architecture and kept
  PFN a research path after its cost/accuracy evaluation.

## V6.9 — TSMC6 and PDK audit

### V6.9.0 (2026-07-12)

- Onboarded CLN6 as the deliberate TSMC7 relabel and audited all TSMC modelcard
  parsers, variants, and generated naive-card paths.

## V6.8 — BSIM-AR return

### V6.8.1 — xl fill (2026-07-11 to 2026-07-23)

- Completed missing BSIM-AR xl checkpoints and evidence rows.

### V6.8.0 — LEVEL=74 recipe campaign (2026-07-06 to 2026-07-07)

- Reintroduced the autoregressive Transformer on the unified pipeline, added
  resolver/force-level support, and measured its fidelity/CPU-cost trade-off.

## V6.7 — universal DirectNet study

### V6.7.1 — campaign cleanup (2026-07-05)

- Removed generated campaign debris while retaining scripts, evidence, and
  recoverable checkpoint interfaces.

### V6.7.0 (2026-07-04 to 2026-07-05)

- Trained 18-code universal DirectNet checkpoints and studied TSMC5 transfer.
  The best historical universal recipe reached 10/12 strict with zero flips.
- Kept universal models environment-pin-only after per-tech models remained the
  more reliable automatic-resolution path.

## V6.6 — curriculum recipe campaign

### V6.6.7 (2026-07-03)

- `csobcrit` and `crit30a1` both reached 13/16; neither advanced the 15/16 hunt.

### V6.6.6 (2026-07-03)

- XL curriculum tied production at 14/16 and triggered a full gate-
  infrastructure audit.

### V6.6.5 (2026-07-03)

- Completed the 13-recipe × four-size matrix.

### V6.6.4 (2026-07-02)

- Promoted `crit30f` after its complete production gate pass.

### V6.6.3 (2026-07-02)

- Full recipe re-test moved the leader from `crit15` to `crit30` at 14/16
  strict.

### V6.6.2 (2026-07-02)

- `crit15` broke the 13/16 wall by one cell.

### V6.6.1 (2026-07-01)

- Ran the first uniform recipe comparison sweep.

### V6.6.0 (2026-06-29)

- Reset the experiment matrix and removed incomparable recipe artifacts.

## V6.5 — complex gates and AC

### V6.5.9 — differentiable DC training (2026-06-29)

- A differentiable-DC-solver training term closed the TSMC7 opamp and produced
  the first 16/16 historical complex-gate result.

### Test-infrastructure correctness sprint (2026-06-28)

- Fixed 11 gate bugs that could mislabel, omit, or compare the wrong evidence.
  Historical verdicts were regenerated rather than patched by hand.

### V6.5.8 (2026-06-28)

- Rejected the EKV high-output-resistance backbone after it drove the TSMC7
  opamp to the wrong rail.

### V6.5.7 (2026-06-27)

- Corrected the V6.5.6 opamp verdict after independent review exposed an
  interpretation error.

### V6.5.6 (2026-06-26)

- Used independent diagnostic routing and tested a KCL-residual training lever;
  the lever did not justify promotion.

### V6.5.5 (2026-06-24 to 2026-06-25)

- A diagnosis-routed corridor retrain reached 15/16.

### V6.5.4 (2026-06-23 to 2026-06-24)

- A fresh full retrain plus best configuration per technology reached 14/16.

### V6.5.3 (2026-06-23)

- Found that the switch-capacitor gap was a harness clock bug, not a compact-
  model error; corrected the test and retracted the affected diagnosis.

### V6.5.2 (2026-06-22)

- Tested charge-derivative levers and refuted the claim that the switch-
  capacitor miss belonged to the solver.

### V6.5.1 (2026-06-22)

- Added the XL capacity tier; killed the µA-band loss lever after it failed to
  improve circuit gates.

### V6.5 — NN AC accuracy (2026-06-22)

- Added derivative-sensitive NN AC gates and connected learned charge
  derivatives to the AC solver.

### AC analysis (2026-06-21)

- Added small-signal frequency-domain analysis around the DC operating point.

## V6.4 — DirectNet production baseline

### V6.4.9 (2026-06-21)

- Benchmarked DirectNet small/medium/large capacity and selected the production
  size from measured circuit accuracy and inference cost.

### V6.4.8+ (2026-06-20)

- Added parametric sweep infrastructure; killed a broad TSMC7 retrain that did
  not improve the targeted gates.

### V6.4.8 (2026-06-17 to 2026-06-20)

- Ran the value-surface accuracy campaign and moved the conditional complex-
  circuit result from 14 to 15/16.

### V6.4.7 (2026-06-10 to 2026-06-16)

- Established the serialized 14/16 production baseline and 8/8 hard-IC SRAM
  initialization evidence.
- Fixed source-frame handling for NMOS as well as PMOS and added the lifted-
  source canary.

### V6.4.6 (2026-06-01 to 2026-06-02)

- Diagnosis-only iteration; no behavioral change shipped.

### V6.4.5 (2026-05-29)

- Track-A experiment produced no promotable change.

### V6.4.4 (2026-05-28)

- Added inference-only mixing of per-technology DirectNet checkpoints.

### V6.1–V6.3.2 (2026-05-12 to 2026-05-15)

- Established per-technology DirectNet data, checkpoints, resolver behavior,
  normalization, and device/circuit gates.

## Earlier history

Before V6.0 the project progressed through the original pure-Python simulator,
BSIM-CMG/PyCMG integration, basic HSPICE-compatible parsing, DC/transient
solvers, initial neural compact models, package refactors, and the first
NGSPICE comparison harnesses. Those early releases predate the current
checkpoint families and evidence methodology; consult Git history when their
implementation chronology is needed.
