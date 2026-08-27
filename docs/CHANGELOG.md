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

## V7.6 — full-terminal DirectNet recovery

### V7.6.1 — full-terminal BSIM-AR and clean campaign contract (2026-08-27)

- Added the explicit experimental `LEVEL=76 FAMILY=bsimar-full` family. It
  predicts the same six independent terminal surfaces as LEVEL=75, in the
  autoregressive order `qg,qb,qd,i_d,i_g,i_b`, then reuses the shared
  KCL/translation-invariant full current and charge boundary.
- Generalized the Transformer AR split without changing the legacy LEVEL=74
  default, parameter names, or state-dict shapes. Full-terminal normalization
  remains in canonical `i_d,i_g,i_b,qd,qg,qb` order; its configuration records
  the output contract, target order, and six-target AR dimension.
- Isolated BSIM-AR-Full checkpoints under `tff` stems. Runtime loading requires
  checksum agreement for the model, normalization, configuration, and JSON
  completion marker; missing explicit pins fail rather than falling back.
- Extended parser, solver discovery, CPU gate dispatch, immutable campaign
  manifests, coverage/report generation, and AnalogGym translation/provenance
  to LEVEL=76. The new family remains outside batched/fused evaluation until
  independent parity gates justify enabling those paths.
- Full-terminal generation now distinguishes audited safety exclusions from
  diagnostic incompleteness: NaN/Inf surfaces, terminal currents above 1 A,
  and failed internal-node solves may be excluded with their coordinates and
  reason counts preserved, while dropped bins and unknown exceptions remain
  fatal. Generator subprocesses are pinned to one BLAS thread so the ten-job
  canonical pass cannot exhaust the process/thread limit.

### V7.6.0 — attributed boundary errors and experimental LEVEL=75 (2026-08-27)

- Fixed NN MOSFET instance multipliers across current, conductance, charge, and
  capacitance paths. On the TSMC5 LDO diagnostic this reduced maximum node
  error from 12.3250 V to 0.306632 V, but the deck remained 0/3; multiplier
  handling was a real defect, not the complete AnalogGym cause.
- Added matched full-OSDI, exact-reduced-OSDI, raw-DirectNet, production-trace,
  and same-state evaluator boundaries. Exact reduced OSDI passed the LDO but
  failed the high-temperature Fan sweep at 12/15 with an 817.3 V maximum state
  error, establishing the full-terminal interface as a prerequisite. Removing
  production corrections lost convergence and did not recover the failing
  gate, so that change was rejected.
- Introduced the explicit experimental `LEVEL=75 FAMILY=directnet-full`
  family. It learns three solver-positive terminal currents and three charges,
  closes source values analytically, reconstructs full 4×4 current and charge
  Jacobians, and uses the existing full-terminal solver stamp. DC skips charge
  derivatives; AC/transient request them lazily.
- Added separate six-surface dataset/training contracts, checksum-bound model,
  normalization and completion artifacts, `dnf` checkpoint resolution,
  LEVEL=75 gate retargeting, and AnalogGym provenance/campaign support. The
  runtime fails outside recorded input bounds instead of applying LEVEL=73
  heuristic continuation.
- A real TSMC5 full-terminal generation probe exposed 132 PMOS rows where the
  `vds_zero` class exceeded its declared nominal envelope. Making that class
  honor the full-terminal envelope moved the matched run to 5,008,380 rows,
  780/780 bins, and zero rejects; the reduced generator remains unchanged.
- Full-terminal generation and training now use isolated `dnf` dataset and
  checkpoint stems by default, so the six-surface family cannot overwrite or
  masquerade as a LEVEL=73 artifact. Scored campaign resume also rejects raw
  evaluator and correction-trace diagnostic rows.
- No LEVEL=75 checkpoint or circuit result is promoted. The generation probes
  were dirty-source diagnostics, and multi-seed truth-surface, circuit,
  AnalogGym, and performance gates remain outstanding. Detailed evidence is
  in [`accuracy/DirectNet-L75-V760-recovery.md`](accuracy/DirectNet-L75-V760-recovery.md).

## V7.5 — AnalogGym migration

### V7.5.17 — PFN retirement and audited simple-circuit coverage (2026-08-26)

- Retired PFN/TabPFN and LEVEL=75 from the parser, solver, model/training
  packages, campaign tooling, tests, and active documentation. DirectNet
  LEVEL=73 and BSIM-AR LEVEL=74 remain the two NN families; historical PFN
  release notes remain below as history.
- Closed the simple-circuit coverage audit: DC/VTC gates reject unconverged
  solves, preserve signed current, and retain the complete expected matrix.
  Added scored temperature, body-bias, reverse-VDS, joint-corner, and exact
  0.5/1.5/2.0 N/P-ratio cells. The variant/temperature-aware joint geometry
  guard passes all **339/339** current requested device geometries.
- Made canonical dataset generation fail on any dropped bin or rejected row,
  pin intra-bin L spacing to 1.35, generate the transmission-gate corridor,
  and record requested/actual manifests, command/commit/source hashes, and
  checksum-bound completion markers. Training rejects diagnostic, stale,
  dirty-source, or incomplete data. Capped/fine-tune samples are stratified
  and the default model split holds out complete
  technology/VT/L/NFIN/temperature groups.
- Bound every clean re-gate log to one immutable manifest of the source commit,
  job list, NGSPICE/OSDI/PDKs, checkpoints, and sidecars. Collection and report
  generation fail on mixed or missing provenance.
- Completed one manifest-bound, CPU-pinned 480-job clean re-gate at gate commit
  `db1b295`, with 480 completion markers and no infrastructure errors or
  tracebacks. Strict complex scores are DirectNet **5/7/9/10** and BSIM-AR
  **9/9/12/11** out of 20 from S→XL. The expanded parametric matrix scores
  DirectNet DC **105/105/105/102 of 129** and transient **88/87/91/94 of 100**;
  BSIM-AR DC **103/104/101/103 of 129** and transient **89/91/86/92 of 100**.
  Device AC and opamp AC remain **0/10** and **0/5** at every tier because the
  required NN operating points do not converge.
- Investigated the 6–14 hour BSIM-AR XL AC long tail. The jobs were CPU-bound,
  not hung: the AC gate scans 66–82 fresh DC biases per technology, and hard
  points exhaust GMIN retry before a 200-step pseudo-transient fallback. Stock
  autoregressive XL evaluation measured 204.4 ms for the DC path and 380.9 ms
  for capacitances versus 29.0/54.4 ms at `small`; one failed point issued 513
  Transformer forwards. The 180-frequency linear AC solve reuses one
  linearization and is not the bottleneck. The non-bit-identical prefix cache
  remained off for the scored campaign.

### V7.5.16 — corrected clean evidence and BSIM-AR evaluation (2026-08-25)

- Corrected the MNA residual probe to recover ideal-source branch currents
  and scale only current-valued rows, then cached its topology-stable fit.
  Corrected opamp AC to refine each simulator's physical bias independently,
  validate the reference, and require a converged NN operating point. Family
  banners and accuracy CLIs now fail closed on invalid selections.
- Re-gated the preserved V7.4 DirectNet and BSIM-AR checkpoints in one isolated,
  CPU-pinned 480-job pass at gate commit `49f0426`. Strict complex scores are
  DirectNet **8/11/12/12** and BSIM-AR **13/12/12/12** out of 20 from S→XL.
  Device AC and opamp AC remain 0/10 and 0/5 at every tier because their NN
  operating points do not converge under the physical contract.
- Stopped the PFN rebuild before any checkpoint completed. Its partial files
  are excluded from V7.5.16 evidence, and the clean PFN report remains the
  checksum-pinned V7.3.0 record.
- Hardened clean training, re-gating, coverage, and report generation around
  isolated checkpoint roots, required sidecars/completion markers, exact
  family scope, argument validation, and interpreter preflight. Tracebacks are
  infrastructure failures; explicit parametric `ERROR` rows stay in their
  denominators while metric aggregates ignore nonnumeric rows.
- The opt-in BSIM-AR prefix cache passed 10/10 parity checks and measured
  118.5 ms → 74.2 ms (**1.60×**). It remains default-off pending a complete
  floating-point-perturbing accuracy re-gate.
- Dead end: a 480-job attempt launched with a nonexistent Python interpreter
  produced only tracebacks. That evidence was quarantined and excluded; the
  driver now rejects the interpreter before dispatch.

### V7.5.15 — clean simple-circuit recheck (2026-08-20)

- Re-ran all 480 CPU-pinned DirectNet and BSIM-AR clean S/M/L/XL suite runs
  from the retained V7.4 checkpoint matrix before continuing AnalogGym work.
  Current strict complex scores are DirectNet 8/11/12/12 and BSIM-AR
  13/12/12/12 out of 20 from S→XL. Parametric DC and transient reproduce;
  device AC is 0/10 and opamp AC 0/5 at every family/tier because AC now
  requires a converged physical operating point.
- Retracted old capacity trends based on nonconverged Miller fixed points.
  Also retracted the V7.4 BSIM-AR TSMC12-XL collapse: all 12 underlying logs
  were race-corrupted, and three of its four complex cells pass when isolated.
- Fixed the campaign path to honor checkpoint archives, generate one complete
  five-technology clean pool, retain TSMC6 in collection, and reject raced,
  timed-out, killed and missing-checkpoint entries from coverage/report
  denominators. Per-log locks now prevent concurrent dispatchers from mixing
  output, contention cannot report false completion, and invalid completion
  markers remain retryable. Raw logs override stale same-pass JSON, coverage
  can fail on gaps, and report completeness requires the parsed metrics used
  by every table. Repinned generated clean reports to one complete current-code
  pass and documented exact old-versus-new provenance in
  [`accuracy/simple-circuits-recheck-2026-08-19.md`](accuracy/simple-circuits-recheck-2026-08-19.md).

### V7.5.14 — DirectNet retrain and AnalogGym qualification (2026-08-19)

- Retrained ten per-technology DirectNet `large` checkpoints (NMOS and PMOS
  for TSMC5/6/7/12/16) with the clean seed-42, filter-off, EMA recipe. All ten
  completed 800 epochs; best validation losses were 0.000215–0.000254.
- The focused CPU gates passed lifted-source DC 15/15, single-device 20/20,
  inverter 10/10, parametric inverter transient 80/80, ring period 5/5, SRAM
  5/5, and switched-cap 5/5. Parametric device DC was 68/69; Miller opamp was
  0/5. Device AC is 0/10 under the corrected converged-operating-point gate.
- Ran one code- and checkpoint-pinned 255-row LEVEL=73 AnalogGym campaign
  against NGSPICE LEVEL=72. No scored deck fully agreed: 0/248 after seven
  invalid deck cells were quarantined. Aggregate comparable voltage error was
  35.41% MRE, -42.66 R², 73.22% NRMSE, and 12.696 V maximum error over
  80,299 samples.
- Added explicit DirectNet technology/VT translation, checkpoint completion
  and hash provenance, ground-truth modelcard/OSDI/NGSPICE hashes, code-state
  campaign manifests, exact transient-policy resume checks, per-technology
  voltage-error aggregation, and local modelcard materialization. Campaign
  children and persisted failed/missing rows now propagate failure to the
  driver; forced reruns invalidate stale rows and training completion markers.
- Fixed NN temperature sweeps to rebuild geometry inputs and invalidate model
  history; production-large checkpoint discovery; AC linearization about
  nonconverged DC states; and failure summaries/timing that previously hid the
  simulator side or reported a long failed attempt as zero seconds.
- Retracted the initial 10/10 device-AC diagnostic: its response shapes were
  close, but all ten DC states were nonconverged. The strict result is 0/10.
  Also retracted the first 54,045-sample AnalogGym voltage aggregate: it used
  only the first/cold segment of multi-plan and recovery sweeps. The corrected
  one-pass aggregate covers all 80,299 saved comparable samples.
  The missing upstream AnalogGym source tree also prevented regeneration, so
  this release scores the tracked V7.5.9 generated corpus and does not claim a
  refreshed source-topology audit.

Open after this release: DirectNet does not preserve AnalogGym operating
points, all five charge-pump transients fail at the first output step, and the
Miller opamp remains railed. These are model/solver research items, not gates
that were weakened to obtain a release pass.

### V7.5.13 — type-based compact-model layout and cleanup (2026-08-19)

- Moved technology inputs to root `PDKs/`: tracked ASAP7 cards and local,
  ignored TSMC cards now have one owner independent of the model evaluator.
- Renamed the two external stacks by role: `bsim_cmg/` owns the physics/OSDI
  evaluator and generated-card cache; `neural_network/` owns DirectNet,
  BSIM-AR, PFN, datasets, checkpoints, training, and evaluation. Python imports
  moved from `bsimar.*` to `neural_network.*`; the public `pycmg` API and
  simulator BSIM-AR family names remain unchanged.
- Retired the closed V7.2 GPU probes and V7.4 campaign babysitters, removed
  generated runner decks with machine-specific paths, and dropped two
  unreferenced campaign diagnostics. Current generation, training, re-gate,
  accuracy-document, AnalogGym, diagnostic, and performance-gate tooling stays;
  the retained V7.1 re-gate now retries a recorded no-checkpoint cell after its
  checkpoints appear.
- Fast verification passed shell syntax, Python compilation, the package
  import/path smoke, two focused PyCMG PDK-resolution tests, BSIM-CMG NMOS and
  PMOS operating points versus NGSPICE (2/2), the AnalogGym migration
  regressions (9/9), and the subcircuit gate (11/11).

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
