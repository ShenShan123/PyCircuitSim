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

### V7.7.6 — capacitor-state fix and opt-in NN limiting

Released 2026-09-25. Two LEVEL=75/76 solver changes, one fixed and one opt-in:

- **Fix.** `_pseudo_transient_dc`, the last NN DC fallback, now restores
  every real capacitor's companion and integration state after its transient
  stage. The leaked `_g_eq`/`_i_eq` made the polishing solve and every later
  DC solve on that circuit see each capacitor as a 2C/dt conductance plus a
  current source (round 3's BSIM-AR medium TSMC16 buffer). The fix changes
  numerics only where the fallback fires on a circuit with a capacitor. A
  regression test compares a DC point solved after the fallback with a
  fresh-circuit solve; without the fix it reads 0.067 V against 0.4 V.
- **Opt-in limiter.** With `PYCIRCUITSIM_NN_NR_LIMIT=1`, an NN evaluation
  whose source-relative Vds/Vgs/Vbs leaves the persisted normalization box
  runs at the nearest in-box bias. The companion linearizes there, and the
  iteration counts as limited, so it is never accepted. `_check_support`
  still rejects a physical answer outside the box, and a non-convergence
  message names any input that was still clamped. The limiter is off by
  default, and it is bit-identical whenever it does not engage. The
  latch-basin contract now covers the knob: on a tightly certified latch,
  the hard-`.ic` solve dies on the support check without it and keeps both
  stored states with it. On two real round-2 support-rejection cells
  (DirectNet small TSMC12 NAND2 and TSMC5 active-load pair), all four
  rejected rows converged and the other rows were unchanged.
- **Provenance.** Campaign manifests record every set numerics knob
  (`runtime_knobs`), so an arm run with the limiter gets its own digest.

The collected suite goes from 1,160 to 1,171 tests.

Planning record, opened 2026-09-21 after round 2. LEVEL=75/76 raise
`CandidateSupportError` on any evaluation outside the persisted normalization
box, and no NN-side limiter precedes that check, so an intermediate Newton
iterate can end a solve whose
physical answer is inside the box. Across both rounds, 440 reference-only
diagnostic runs found every accepted NGSPICE trajectory inside support for
every case that raised such an error in a DC or transient analysis, except
BSIM-AR `self_biased_cascode`. They cover 65 of round 2's 100 support-rejection
rows; the 33 AC rows and 2 cascode rows are unattributed. The
[plan](plans/2026-09-21-v776-nn-voltage-limiting.md) records the evidence, the
contract constraints, and the gating: the limiter perturbs floating-point
results, so it ships disabled until a full accuracy re-gate clears it, and it
must first pass the latch-basin contract.

The evaluation-arm harness repairs reached `main` in merge `b688412`: the ten
commits of `eval/v775-round2`, which are the harness both rounds actually ran
on. They retain DC-sweep endpoints within roundoff tolerance, compare signed
source currents in the lifted-source canary, widen unidentifiable subthreshold
reference windows and keep those rows as diagnostic `ERROR`s, emit terminal and
hierarchy AC phase, cover every applicable device-integrity corner, and
attribute candidate-only failures in derived CMRR/PSRR and SRAM rows. Each
carries its regression test; the collected suite goes from 1,101 to 1,160.
Endpoints, phase and window selection change numerical results, so no score
measured before this merge transfers to the merged runtime: both rounds stay
pinned to their own arms, and the V7.7.6 re-gate needs a separately
provenanced one.

Repository cleanup. Published both V7.7.5 round reports under `docs/accuracy/`
with the index and its preservation hash. Removed the uncalled
`_DirectNetFullBase` compatibility alias, whose only occurrence in any worktree
was its own definition; that changes the tracked source inventory hash although
no numerical path moved. Closed three stale statuses: V7.7.1 is superseded by
V7.7.2, V7.7.2 finished training on 2026-09-11 but never completed a scored
campaign, and the 2026-09-12 evaluation schedule closed when its fifth arm
stopped at 3,429 jobs. Extended `.gitignore` to model bundles and tensor
artifacts anywhere in the tree, so checkpoints and simulation results stay on
disk and out of the patch. Local cleanup removed 21 `__pycache__` directories
(160 files), the Ruff and pytest caches (168 KiB), an empty `docs/issue_report/`,
and two worktree registrations whose directories no longer exist. Datasets,
checkpoints, every campaign worktree, and all of `results/` are untouched; that
deletion is local and is not part of the Git patch.

Verification: 1,160 project tests passed, zero skipped, with the five existing
CPU pin-memory warnings. Focused Ruff checks (`F401`, `F811`, `F821`, `F841`),
the preserved-report checksums, and all 188 local Markdown links passed. No
re-gate, training, or PyCMG suite was run for this cleanup, and no accuracy
result is claimed.

### 2026-09-25 — V7.7.5 checkpoint round 3 and round-report verification

Round 3 ([report](accuracy/v775-round3.md)) ran the seven campaign suites that
neither earlier round covered, on the same 80 bundles: device AC, Miller
open-loop AC, parametric inverter transient, the lifted-source canary, device
and terminal integrity, and flat/nested NN netlists. That is 280 nominal OMP=1
cells from a clean worktree at `b24bfca`: 237 PASS, 43 FAIL, none `infra`.

- DirectNet-Full passes device AC 38/40, Miller open-loop AC 4/20, inverter
  configurations 398/400, and the canary 120/120.
- BSIM-AR-Full passes device AC 40/40, Miller open-loop AC 13/20, inverter
  configurations 400/400, and the canary 120/120.
- In every scoreable subthreshold row, both families put the off-current above
  the reference, never below; the median excess is about two decades. 25 of 80
  rows are too flat to fit a swing at all. Linear-domain NRMSE in the same rows
  is mostly below 1%.

New solver defect, open: `_pseudo_transient_dc`, the last NN DC fallback,
leaves capacitor companion state (`_g_eq`/`_i_eq`) on the live circuit. Later
DC solves on that circuit then converge on a circuit with the capacitors
replaced by conductances. The reproducer is BSIM-AR medium TSMC16
`nn_subckt` DC: its output stays at 8–11 mV instead of 0.80 V, with an 80 µA KCL
residual on a fresh parse. A post-hoc KCL re-stamp of the saved candidate DC
sweeps in rounds 1–3 found no other affected scored row; the 1,270 device and
VTC sweeps it skipped contain no capacitor. Round 3 reclassifies that one row
as `ERROR`. Operating points behind AC and transient analyses are not saved
and could not be audited. The fix, restoring capacitor state after the
fallback, changes numerics and needs its own test and re-gate; it shipped in
V7.7.6.

Verification of rounds 1–2: independent parsers reproduced every table in both
reports from the raw logs. Prose errors are corrected in place, and these
earlier statements are retracted:
- The BSIM-AR medium ring averaged 269 minutes, not 85, so the medium ring
  slowdown is about 187×. "34–90×" does not hold: round 1's ring cells ran 35×
  (small) and 187× (medium) slower, and its switch-cap cells 37× and 93×.
  Round-2 per-cell ratios have a median of 39× and span 0.6–2,516×.
- The last five round-2 cells took 28–53 hours each, not 28–52, and held the
  round open for 33.6 hours, not four days.
- Of round 1's 24 support rejections, the diagnostic attributes 18; the six
  `ota_5t_buffer` closed-loop AC rows were never checked.
- The support diagnostic does not attribute every rejection: 35 of round 2's
  100 rows were never checked.
- Inverter switching energy is 18–33% high on TSMC5/6/7 and 2–12% low on
  TSMC12/16, not uniformly high.
- The LDO line-regulation and Miller differential-gain errors have mixed sign;
  they are not a uniform "3–30× loss" or "37–87 V/V".
- NAND/NOR start-up node errors span 0.52–1.92 V.
- BSIM-AR beats DirectNet on median NRMSE on two technologies, not three.
- Four BSIM-AR error labels were wrong.
The V7.7.6 plan's evidence paragraph now states the support-diagnostic
coverage.

Verification: 1,160 tests passed before launch, geometry preflight 463/463,
and the preserved-report checksums verify. No numerical source changed.

### 2026-09-15 — V7.7.5 checkpoint targeted round 2

After round 1, the user chose a targeted round 2 and asked for both harness
defects to be fixed first ([report](accuracy/v775-round2.md)). It covers the 23
remaining simple-v2 cases at nominal (920 cells) and the ring and Miller gates
at OMP 2/4 (160 cells), from branch `eval/v775-round2` at `d708d4b`.

Harness fixes, with regression tests red on `39b14f1` and a full suite of
1,160 passed with none skipped:

- A derived CMRR/PSRR row inherits the error kind of the failed input analysis,
  so an unconverged candidate exits 1 instead of dispatching as infrastructure.
- The SRAM gate names the engine that failed a corner, instead of labelling
  candidate-only failures as reference errors.

In round 2 every error row is attributed to the candidate, and no cell has an
`infra` verdict.

A reference-only support diagnostic ran 440 runs over every case that raised a
`CandidateSupportError` in a DC or transient analysis, except BSIM-AR
`self_biased_cascode`, for all 40 checkpoint pairs. Every accepted NGSPICE
point is inside the normalization box, except NAND2/NOR2 transient samples
equal to a start-up spike. The 65 checked support rejections of round 2's 100
are therefore Newton trial states, a solver-globalization limit, not missing
training data. The spike comes from NGSPICE's first `uic` step, which drives
initialized series-stack and inverter-chain nodes past the rails (NOR2
`v(pint)` −1.165 V from 0.75 V). It produces 0.52–1.92 V voltage errors that
are not model error.

DirectNet-Full: ring and Miller pass OMP 1/2/4 in 40/40 cells with no flips.
Simple-v2 L1–L4 rows converge 24–25/25, 83–87/90, 107–119/135 and 35–44/45 by
tier; `beta_multiplier` converges in only 4–8 of 15 rows per size. Unattributed
systematic errors remain: TSMC5 inverter switching energy is 25.8–27.1% high
at every size, and the LDO line-regulation slope has the wrong sign in 7 of 18
converged rows and is 1.4–32× steeper in 10.
BSIM-AR-Full completed all 540 cells and converged 1,032 of 1,180 simple-v2
rows, with all 148 error rows attributed to the candidate. It converges fewer
rows than DirectNet at L3/L4 (`beta_multiplier` 19/60, `ldo_regulator` 60/80,
`multistage_buffer_12t` 24/40) and shows the same systematic inverter-energy
and Miller-gain errors. Its cost is the practical finding: a median 39×
DirectNet per cell on identical cases, with the last five cells taking
28–53 h each and holding the round open 33.6 h after the other 1,075 cells had
finished.

### 2026-09-14 — V7.7.5 checkpoint quick round 1

This round scored the 80 V7.7.2-trained bundles shipped with V7.7.5 on a
nominal, OMP=1 subset of the L0–L4 tiers: 12 cases × 2 families × 4 sizes ×
5 technologies = 480 cells ([report](accuracy/v775-quick-round1.md)). The
evaluation runtime is main `d6ae11c` plus the evaluation-branch harness
repairs (`39b14f1`), which were unmerged while the round ran and reached main
in merge `b688412`. Its manifests record a distinct evaluation arm,
not source equivalence. The round supports a continue/stop decision and is not
a clean qualification.

DirectNet-Full completed all 240 cells. Medium, large and xl pass 20/20
simple-v1 cells at OMP=1. Small passes 18/20; the misses are switch-cap droop
on the TSMC6/TSMC7 repeat. Parametric device DC is 128, 129, 127 and 126/129
by size; the large and xl misses are high-NFIN configurations on TSMC12/16.
L1–L3 diagnostics converge in 416/420 rows. The L4 5T OTA buffer converges in
45/60 rows, mostly blocked by support-box rejections.

BSIM-AR-Full completed all 240 cells. Device DC is 128/129 at small and
129/129 at every larger size. Simple-v1 cells pass 19/20, 18/20, 20/20 and
20/20 by size. The misses are Miller DC on small TSMC16 and medium TSMC5 and
SRAM on medium TSMC16, all candidate convergence or support failures. L4
converges 6, 12, 15 and 13 of 15 rows by size. Its ring cells ran about 35×
(small) and 187× (medium) slower than DirectNet's, and its switch-cap cells
37× and 93×.

Two harness defects were found and left unfixed. A nonconverged differential
pair's derived CMRR row turns the cell into `infra`, and the SRAM gate labels
candidate-only corner failures as `reference`. The report recounts both as
candidate `ERROR`s. The first launch exceeded the session's background-task
memory limit; its 56 interrupted cells are archived and were rerun.

Verification: the report-preservation check verified all pinned checksums, 47
dataset/campaign contract tests passed with no skips, and local Markdown links
resolve. Outside Markdown, only the accuracy-index preservation hash changed.

### 2026-09-12 — V7.7.2 model evaluation scheduled

The completed 80-model training inventory is retained unchanged. A separate
audited-runtime arm now schedules every gate entry point, catalog corner and
parametric dimension from devices through L4, with persistent supervision.
Its manifest explicitly distinguishes the original training source from the
evaluation source; default exact-source checks remain unchanged. Parametric
checkpoint auditing now honors `BSIMAR_CHECKPOINT_DIR` and labels LEVEL=76
correctly. No accuracy promotion or new package release is claimed.
See the [evaluation schedule](plans/2026-09-12-v772-model-evaluation.md).

The first preflight stopped before NN scoring because exported campaign data
paths overrode two pytest geometry fixtures. The scheduler now removes those
paths only for fixture-based contracts; real gates retain explicit dataset
and checkpoint roots. The failed bootstrap is archived separately and the
evaluation restarts from a fresh source pin. The PyCMG reference suite passed
315 tests without skips in that bootstrap.

The first scored arm stopped at 20 completed jobs with four terminal-integrity
infrastructure failures. Its capacitance path computed a full matrix but did
not emit the `phase_maxerr_deg` required by the audited AC schema, so all ten
capacitance rows per affected job were rejected. The second arm records the
maximum wrapped phase error across the complex 4×4 terminal admittance matrix.
Known 0°, 30°, and 180° perturbations reproduce the bug and verify the repair.
Capacitance values, simulator equations, thresholds, and the 19,266-job
inventory are unchanged; the second arm starts a complete separate pass.

The second arm completed 171 jobs before a TSMC12 hot/low-VDD PMOS device
diagnostic could not fit subthreshold slope. Deck parity and both converged
111-point traces were verified. The reference spanned 4.246 decades, but the
preferred window retained only 0.216; its fallback ran only for an empty
range. The third arm applies the existing reference-only fallback whenever
the initial window lacks the required four points or 0.5 decades. Already
identifiable windows and both fit requirements remain unchanged. Tests cover
NMOS/PMOS exact exponentials, preservation of valid windows, and continued
rejection of flat references and malformed metrics. Every earlier arm is
preserved separately; no model weights or simulator equations changed.

The all-corner NGSPICE device screen also found 12 TSMC5/6/7 hot-corner
subthreshold references whose current is maximal at zero gate bias. Those
complete, converged sweeps cannot identify the declared rising-window slope.
They now produce diagnostic `ERROR` rows of kind `reference_metric`, with
convergence retained and no scored metrics, instead of halting the campaign
as infrastructure failures. Flat and decreasing references are covered for
both polarities. Missing/malformed metrics, failed NGSPICE runs, partial
traces, and qualification rows cannot use this exception.

The third arm completed all 1,196 device-stage jobs. Review found the
lifted-source canary's apparent nonconvergence was missing DC endpoints:
repeated 5 mV additions crossed 0.8 V by roundoff and the strict loop stopped
one point early, in both sweep directions. A passive RC control reproduced
160 candidate versus 161 NGSPICE points. The fourth arm tolerates bounded
endpoint roundoff and stamps the requested final value, while preserving
interior step values and avoiding an added point for non-divisible spans.
The third arm's 1,310 completed jobs and 16 interrupted jobs are preserved.
The separate AR-cache diagnostic also measured a 2.31e-4 relative Jacobian
deviation on TSMC7 small NMOS against its 1e-4 tolerance; cache-off execution
remains the scored contract, and that diagnostic failure is retained.

The endpoint repair exposed a canary polarity bug: the PMOS scalar
`calculate_current()` already uses the project's comparison sign, but the
canary negated it again, yielding approximately 200% MRE. The candidate now
reads signed MNA `i(Vd)`, the same observable used by NGSPICE, with matching
NMOS/PMOS orientation. A boundary test checks both devices and both current
directions; it cannot hide a wrong sign through an absolute value. Device
equations and scalar API conventions are unchanged.

The fourth arm completed 3,398 jobs, including devices and L1, then stopped
at two hierarchy-AC schema errors. The flat/nested summary selected only four
aggregate fields and discarded the already-computed `phase_maxerr_deg`.
The fifth arm retains the worst phase error from both representations. Tests
exercise the complete DC, transient, and AC summary paths and a known 30°
phase rotation. The other production metric-validator call sites were checked:
the shared trace comparator and repaired terminal-admittance path emit phase;
the collector only validates or derives it from recorded per-signal values.
The simulation code, checkpoint weights, and 19,266-job inventory are unchanged.

### V7.7.5 — repository cleanup

Removed ten uncalled definitions: the DC last-solution accessor and its
write-only copy, the unused technology-distinctness helper and private parser,
the scalar normalizer derivative API and its three hooks, the benchmark VT
shortcut, the unused opamp deck rewriter, and the retired recipe-delta builder.
The accuracy report builder now accepts only its live clean-report path;
recipe-only dispatch, its empty registry, and boolean plumbing are gone. Existing
report/provenance tests retain their assertions against the simpler interface.
No device equations, solver convergence rules, gate inventories, or thresholds
changed.

Removed the four LEVEL=73/74 clean/recipe reports, superseded V7.5.15 recheck,
pre-fix claims register, completed V7.2 GPU plan, and retired-family V7.4.2
capacity plan. Their last tracked contents remain at `64ecd33`; recover any
one with `git show 64ecd33:<path>`. Current LEVEL=75/76 reports, harness audits,
and unfinished campaign plans remain. The V7.7.3 evaluation-arm plan no longer
claims ownership of package release metadata. Report checksum preservation
still applies after updating the accuracy index.

Historical retractions remain binding: pre-`gds`-fix AC/opamp scores and OMP
flip rankings are invalid; capacity/basin rankings from removed families do
not qualify current models; TSMC6 repeats TSMC7 rather than adding independent
technology evidence. The V7.4 capacity decline was traced to unsampled
intra-bin geometry, not proof that greater capacity hurts autoregression or
that exposure bias caused the decline. Within-bin geometry coverage remains
a live dataset gate.

Local cleanup removed 8,320 disposable simulation/test output and cache files
(358.53 MiB allocated), including obsolete modelcard/NGSPICE scratch directories.
Datasets, checkpoints, active release worktrees, campaign provenance, private
PDKs, and cited historical diagnostics are preserved. Ignored artifact deletion
is local and is not part of the Git patch. V7.7.5 is a maintenance release;
no new training or accuracy promotion is claimed.

Verification: all 1,101 project tests passed, zero skipped; five existing CPU
pin-memory warnings. Report checksum preservation, focused Ruff checks
(`F401`, `F811`, `F821`, `F841`), changed-Python syntax, and all 100 local
Markdown links passed. Full numerical re-gating, new training, and a separate
PyCMG suite were not run for this cleanup.

Documentation follow-up: cross-linked all seven READMEs and agent guidance
around the root workflow, package APIs, test/template contracts, and evidence
owners. Replaced stale PyCMG/ASAP7 examples, corrected the terminal `id` API
and current-sign explanation, removed copied obsolete inventories, and kept
generated output in the documented artifact locations. Dated reports now
label their retired-family context; campaign plans preserve their frozen IDs
without rolling back package metadata. The methodology now matches the
implemented shared NGSPICE-refined opamp AC bias. Historical score tables
and numerical code are unchanged; only the accuracy-index preservation hash
changed outside Markdown.

Follow-up verification: 51 metadata/report/campaign contracts and 48 PyCMG
API/sweep tests passed, zero skipped; one expected missing-device warning.
The documented ASAP7 API and 25-row CSV examples ran successfully. Local
Markdown targets/anchors, README shell syntax, and preserved-report checksums
were checked. Full numerical re-gating, training, and the full NGSPICE-backed
PyCMG suite were not run for this documentation follow-up.

Review follow-up: removed the unused generator temperature-parser import,
`OsdiModel.set_param()`, `ProcessParams.as_dict()`, `NNTechConfig.default_variant`
and its registry assignments, and the Transformer's write-only `raw_input_dim`.
Clean reports now use tiers directly instead of identity mappings, an empty
override registry, and duplicate label/key wrappers. Removed the obsolete
`pybind11` requirement; PyCMG uses ctypes. The version remains V7.7.5.

The edited compact-model files and requirements change the tracked source
inventory hash even though numerical paths and checkpoint tensors are
unchanged. Existing source-equivalence guards remain strict; frozen campaign
worktrees and original artifact provenance are preserved.

Review follow-up verification: all 1,101 project tests and 48 focused PyCMG
API/sweep tests passed, zero skipped; five CPU pin-memory warnings and one
expected missing-device warning. Focused Ruff checks (`F401`, `F811`, `F821`,
`F841`), changed-Python syntax, preserved-report checksums, and version checks
passed. All report sections, completeness decisions, and tier selection match
the previous builder across complete, mixed, partial, and empty synthetic
matrices for both families. Full numerical re-gating, training, and the full
NGSPICE-backed PyCMG suite were not run for this follow-up.

### V7.7.3 — root NN workflow

Unified `main.py` exposes `simulate`, `data`, `train`, and `evaluate` as
independent commands, with `flow` executing the three NN stages in order.
Direct netlist invocation remains supported; `all` aliases `flow`. Runs share an isolated artifact root, use the invoking interpreter,
support explicit physical GPU selection, and preview without dependencies or
writes. Existing generator/trainer defaults, completion/provenance checks,
and NGSPICE comparison gates remain authoritative. Existing datasets and
checkpoint bundles are protected against replacement.

The campaign job generator and geometry guard now accept validated subsets;
their default release inventories are preserved. Root evaluation selects
OMP=1 cells, separates clean/canary/held-out evidence, collects provenance-bound
reports, and distinguishes failed gates from incomplete execution. README
documents the root interface and repository layout; package metadata stays 7.7.3.
This is a workflow release with no newly trained model or accuracy promotion.

Dataset and training parsers now share dependency-free option definitions
with the root entry point. Every backend setting is available through stage
arguments, including sampling/temperature/voltage controls, output variants,
loss weights, auxiliary subthreshold loss, EMA/SWA, initialization, precision,
and autoregressive strategies. Named recipes preserve distinct checkpoint
stems through training and evaluation. Device suites and circuit catalog names
can be listed and selected; their owning pools and nominal circuit geometry
checks follow that selection. Legacy campaign defaults remain unchanged.

The argument audit also exposed the generator's existing overshoot/body-bias
overlay counts and progress verbosity. Both experimental overlays stay disabled
by default. Regression tests exercise actual root-to-backend argument round trips
and generation-function dispatch, including flow restrictions.

Verification: 1,101 project tests and 48 focused PyCMG API/sweep tests passed,
0 skipped. Five CPU pin-memory warnings and one deliberate missing-device
selection warning remain. The 258 workflow contracts cover 87% of the launcher,
99% of the shared parsers, and 92% of the job selector. A dependency-free
full-matrix dry run enumerated ten datasets, 80 training jobs, and three
evaluation pools without creating the run directory. Full data generation,
training, and numerical re-gating were not run for this CLI change.

### V7.7.3 — corrected-solver arm and open harness items (planned)

Opened 2026-09-06 after the V7.7.2 harness audit closed on `main`. The audit
follow-up corrected transient numerics, so `main` is no longer the numerical
source of the in-flight V7.7.2 campaign; V7.7.3 is the separately provenanced
arm that will score the corrected runtime on the V7.7.2 checkpoints and carry
the items the audit left open. The
[plan](plans/2026-09-06-v773-corrected-solver-arm.md) lists them with their
blockers. Opening commit: the [audit](accuracy/v772-harness-audit.md) was
compacted into one record of findings, fixes and witnesses, and
`run_simulation` gained an in-process witness for `.tran`, `.dc`, `.ac` and
the bare operating point (`simulation.py` collected-suite coverage
11 % → 74 %). Verification: 842 tests passed, 0 skipped, 0
expected failures. No accuracy claim; the release stays V7.7.0.

### V7.7.2 — complete four-terminal model refresh (training complete 2026-09-11)

Prepared a fresh ten-dataset, 80-model S/M/L/XL campaign using the latest
tested full-terminal generator and harness fixes. The existing dependency
runner now isolates release state and can wait for the active V7.7.1 training
to finish before claiming its GPUs. Clean and simple-v2 evidence remain
separate, and report generation explicitly registers the V7.7.2 campaign.
The [schedule](plans/2026-09-05-v772-full-retraining.md) records the complete
scope and release conditions. No completed training or accuracy promotion is
claimed by this preparation entry.

Preparation verification: 624 project tests and 315 PyCMG tests passed with
no skips. Three expected warnings cover CPU pin-memory and a missing-device
selection test. The rendered inventory contains 600 clean and 1,200 simple-v2
jobs. Fresh training and numerical evaluation remain scheduled work.

User-directed consolidation retains the four completed bundles and three
live XL jobs from the V7.7.1 queue without restarting them. Stopped duplicate
generation and separate pilot/release schedules. One V7.7.2 supervisor now
waits for all 80 successful training jobs, validates the bundles, and starts
full evaluation automatically. Original artifact provenance stays intact.
An explicit cross-commit manifest option accepts existing models only after
exact numerical-source inventory comparison; model/runtime/template changes
are rejected and default provenance remains strict. Reports name the
training and evaluation commits separately.

Consolidation verification: 635 project tests passed, no skips. Regression
tests exercise the training-to-evaluation barrier and reject changed numerical
source inputs. Two expected CPU pin-memory warnings remain.

Harness audit closure (2026-09-05, `main` only; the in-flight release
worktree is untouched). The [audit](accuracy/v772-harness-audit.md) records
each finding and its closure. Two of its proposed fixes were wrong as written:
the geometry guard takes no `--tech` and read the package dataset directory,
not the campaign root, so its 463/463 had been measured on the wrong grid; the
lifted-source canary takes `--techs` and emitted no result markers. The guard
now honours `--data-dir`/`BSIMAR_DATA_DIR` and runs once as an evaluate-stage
preflight (463/463 PASS against `results/v771_r2_data` by hand); the canary
accepts `--tech`, emits structured rows, and is dispatched as its own 40-job
`canary` pool so the clean denominator is unchanged. `v730_coverage.py` now
derives its device-suite list from the job generator instead of keeping a
fourth copy. Contract modules added: three-registry technology agreement with
the inverter divergence pinned; hermetic solver numerics against closed forms
(tolerances, both GMIN ladders, limiter and oscillation acceptance, the
integration ladder, breakpoints, `.nodeset` branch selection, both latch
basins under the reference and `refine_output` marches, the LEVEL=72 window
and the LEVEL=75/76 support box); calibrated mutation oracles for 23 metric
profiles; every AC analysis now requires `phase_maxerr_deg` (the collector
derives it for pre-V7.7.2 rows); `slew_slow` and `load_heavy` stimulus
corners; a byte-level freeze of all 760 nominal simple-v2 renders; `main.py`
and training-reproducibility witnesses; and a self-enumerating `--help`/
unknown-flag check over all 29 gates. Seven hand-rolled comma selectors were
replaced by `parse_csv_choices` (`--tech "TSMC5,"` used to exit 0), the
fixed-matrix gates parse inside `main(argv)`, `core_gates.py` was folded into
its one caller, and the empty `tests/diag/` package was removed. AGENTS.md no
longer claims ±5 V/±10 V outer clamps that no code implemented, and names the
latch-basin contract that replaces the deleted GPU gate.

One numerical defect was found and deliberately not fixed in this arm: after
the first backward-Euler transient step, `Capacitor.update_voltage` leaves the
stored current at zero, so the first trapezoidal step drops it (the LEVEL=75/76
charge history carries it correctly). The closed-form RC contract is a strict
expected failure until a fresh re-gated arm lands the fix. Verification: 785
project tests passed, 2 expected failures, 0 skipped, 5 CPU-only Torch
warnings; `verify_data_geometry_coverage` 463/463 on the campaign datasets.

V7.7.2 audit follow-up (2026-09-05, this checkout only): fixed the deferred
BE capacitor-current history defect, actual transient stop-time clipping, and
variable-step BDF-2 coefficients in passive and full-terminal charge stamps.
Startup integration now advances after the first accepted piece, and tiny
positive spans no longer round to zero intervals.
Added direct rejection/rollback and stiffness-promotion witnesses. The canary
now rejects stale/truncated sweeps, preserves current signs, covers NMOS and
PMOS, and reports diagnostic execution failures; the collector requires all
six rows per checkpoint group. AC phase now appears in the human report.
Retained LEVEL=72 suites now fail on mixed PASS/ERROR results and reject an
unconverged transient initialization; parametric errors update tech status.
The [audit](accuracy/v772-harness-audit.md#numerical-defects-found-and-fixed)
owns reproductions, verification, and remaining qualification limits. Version
scope remains V7.7.2; active training and release worktrees are untouched.
Corrected numerical source requires a separate complete evaluation before
accuracy promotion; existing campaign provenance is preserved.
Follow-up verification: 834 tests passed, zero skips or expected failures;
four ASAP7 LEVEL=72 transient comparisons passed against NGSPICE; all six
TSMC12 NMOS/PMOS reference-canary curves were complete. Five existing
CPU pin-memory warnings remain.

Second audit follow-up (2026-09-06, `main` only): checked the corrected
checkout against its own audit document, re-measured statement coverage
(collected suite 64 %, union with the four LEVEL=72 gates 69 %; `solver.py`
80 % against 36 % at the audited baseline), and added witnesses for three
AGENTS.md contracts no pass had enumerated: no GMIN in the AC stamp (a
100 GΩ / 1 fF node would read 0.909 instead of 1.000), LTE refinement opt-in,
and the per-tech local variant vocabulary at inference (the existing resolver
test used TSMC5, whose local and universal codes coincide). The stale
"still not covered" text was rewritten. Verification: 838 tests passed, 0
skipped, 0 expected failures; the four LEVEL=72 gates and the geometry guard
(463/463) passed. No campaign was run; both worktrees untouched.

Training closed on 2026-09-11: all 80 bundles carry completion markers in
`results/v771_r2_checkpoints`. The campaign's automatic clean and simple-v2
pools never completed — five staged evaluation arms stopped on the harness
defects recorded in the 2026-09-12 entry above — so V7.7.2 names a trained
model inventory, not a scored campaign. The saved bundles have been measured
only by the two V7.7.5 diagnostic rounds, which are explicitly not a clean
qualification.

### V7.7.1 — regeneration and retraining (superseded by V7.7.2)

Prepared an isolated ten-dataset, 80-bundle full-terminal refresh and a
persistent dependency runner for the complete clean and simple-v2 harness
pools. The [execution schedule](plans/2026-09-04-v771-full-retraining.md)
records scope, compute allocation, provenance, and release conditions.
No new accuracy or promotion is claimed until the complete campaign is scored.

Fresh-worktree preflight exposed the re-gate driver's stale bundled NGSPICE
default and a simulator-free lock test that inherited that external dependency.
The driver now uses the same system default as the shared harness; the lock
test supplies its own executable stub.

The first actual canonical regeneration found a missed six-surface migration:
`_subvt_off_points()` read `id` from an evaluator that now emits `i_d`, so
enabling the documented subthreshold overlay crashed every technology/polarity
with `KeyError` before any dataset completed. The sampler now uses the full
terminal drain-current name. A regression crosses the real evaluator adapter
for NMOS/PMOS, including valid, high-leakage, nonfinite, and failed-solve probes.
The failed kickoff logs remain preserved under `results/v771_campaign/`.

The first completed DirectNet-small TSMC5 pilot passed 25/26 parametric DC,
20/20 inverter configurations, and 2/2 device AC. The +125 C NMOS failure
reproduced in direct NN evaluation (19.47% NRMSE); runtime temperature was
398.15 K and the direct current agreed with the circuit solver within
3.4e-19 A on their shared sweep interval. PyCMG versus NGSPICE was 0.0164%
NRMSE. This attributes that early failure to the learned surface, with no
solver correction justified. Off-state current and body-derivative diagnostics
remain weak; these results are not the final campaign or a promotion.

The mixed A100/RTX host exposed another execution bug: CUDA's default
FASTEST_FIRST device order mapped numeric `CUDA_VISIBLE_DEVICES=0` to a busy
RTX while the scheduler had checked physical A100 0 with `nvidia-smi`.
Stopped the affected epoch, preserved its data, complete DirectNet pair and
partial Transformer logs, and changed allocation to physical GPU UUIDs.
The replacement epoch uses isolated `v771_r2_data`/`v771_r2_checkpoints` paths
and fresh source provenance; the numerical model and training recipe are
unchanged.

The Transformer-small pilot exposed a diagnostic classification bug. Both
subthreshold curves converged, but varied by only 0.200/0.048 current decades
in the reference-defined window, below the existing 0.5-decade slope
identifiability requirement. The resulting unavailable slopes were reported
as infrastructure crashes, and the CLI subtracted those rows from convergence.
They now remain candidate `ERROR` rows with empty scoring metrics and exit 1;
other measured quantities are retained only under `uncharacterized_diagnostic`.
Missing, malformed, and unexpectedly non-finite metric payloads still fail as
infrastructure. The printed convergence count uses the actual solver flags.

Isolated replay: DirectNet remains 18/18 characterized and converged;
Transformer is 16/18 characterized and 18/18 converged. All characterized
metrics are unchanged, and both complete row sets pass the collector contract.
The unit suite passes 612 tests. A wider 90-job TSMC5 pilot checks both model
families and every catalog topology before freezing another source epoch.

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

Review follow-up: standalone NN gates select only the requested full-terminal
family. Geometry coverage reads the canonical `dnf` dataset names. Report
generation can explicitly select the 600-job V7.7.0 clean campaign and rejects
missing metrics or mismatched provenance before changing reports. The AR-cache
gate now compares all six charge/current Jacobians; regression mutations cover
each current gradient in AR3 and AR6. These are harness corrections, with no
new accuracy promotion. Verification: 585 unit tests and 10/10 AR-cache checks
passed; the unit suite emitted two CPU pin-memory warnings and no skips.

Post-merge review fix, same day: `tests/common/nn_gate.py` resolved BOTH
families regardless of the selected one. That was inert while the second arm's
fallback stems named retired artifacts; renaming them onto the live
`{tech}_{dnf,tff}_{size}_{dev}` production slots made a run pinned to one
family silently also run, score, and pay ~40x inference for an unpinned
checkpoint of the other — outside the run's own checkpoint provenance.
`get_available_checkpoints()` now resolves only the family named by
`PYCIRCUITSIM_NN_FORCE_LEVEL`, and the banner names it, matching every other
harness module. Also retired the inert `MOSFET_CMG.evaluator_boundary`
attribute (its only reader was deleted with the reduced stamp, so setting it
would have quietly done nothing) and bumped the dataset generator release tag
to V7.7.0, since the generator's `inv_trip` overlay is no longer skipped when
the unused `find_threshold` probe raises. Collected suite: 562 tests.

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
