# PyCircuitSim changelog

This is the compact release and decision ledger. Commands belong in
[README.md](../README.md), durable implementation contracts in
[AGENTS.md](../AGENTS.md), compact-model scoreboards in
[docs/accuracy/](accuracy/), and AnalogGym tables in
[RESULTS_TSMC.md](../examples/complex_circuits/RESULTS_TSMC.md). Detailed
per-commit chronology and superseded prose remain in Git history.

## Reading historical scores

- V7.3 added the deliberate TSMC6 repeat to compact-model headlines: strict
  circuits changed from /16 to /20, device AC from /8 to /10, and op-amp AC
  from /4 to /5.
- AnalogGym denominators changed as invalid or redundant decks were measured,
  quarantined, or pruned. Compare totals only when the basket is identical.
- Current generated reports, not this ledger, own detailed score tables.

## V7.6 — full-terminal families and closure

### Post-V7.6.4 — versioned simple-topology diagnostics (2026-09-02)

- Versioned the existing ring/opamp/SRAM-SNM/switched-capacitor qualification
  matrix as `simple-v1` without changing its `/20` denominator. Historical
  `verify_complex_*` campaign IDs remain aliases only; these are simple
  circuits and continue to live under `examples/simple_circuits/`.
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
