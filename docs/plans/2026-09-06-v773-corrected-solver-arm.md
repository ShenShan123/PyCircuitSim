# V7.7.3 corrected-solver arm and open harness items

Status: planned, opened 2026-09-06 after the
[V7.7.2 harness audit](../accuracy/v772-harness-audit.md) closed on `main`.
This is the V7.7.3 evaluation-arm identifier, not the current package release;
the repository [README](../../README.md) owns release metadata.
V7.7.2 is still training in its frozen worktrees and is not touched by this
arm. No accuracy result or promotion is claimed here.

## Why this arm exists

The V7.7.2 audit follow-up corrected transient numerics on `main`: the
accepted capacitor current after a backward-Euler piece, variable-step BDF-2
coefficients, stop-time clipping, and startup integration. Those changes make
`main` a different numerical source from the frozen V7.7.2 campaign, so
V7.7.2 evidence cannot describe them and the source-equivalence check rejects
treating both as one arm. V7.7.3 is the separately provenanced arm that scores
the corrected runtime and carries the harness items the audit left open.

## Prerequisite: separate training-source identity from evaluation-runtime identity

The V7.7.2 checkpoints are trained from PyCMG datasets; the transient solver
plays no part in training. `scripts/v710_regate_manifest.py` currently hashes
compact-model, generator, runtime, template, PDK and environment inputs into
one numerical-source identity and rejects a runtime change when existing
models are reused. V7.7.3 needs that identity split in two: a training-source
hash that must match the V7.7.2 checkpoints exactly, and an
evaluation-runtime hash recorded separately in every manifest and report.
`test_campaign_source_equivalence.py` must keep rejecting a mixed arm; it
gains the case where the training source matches and the runtime does not,
which V7.7.3 reports must name as such.

## Open items carried in

| # | Item | Blocked on | Approach |
|---|---|---|---|
| 1 | Score the corrected solver | V7.7.2 training complete and bundles validated; the prerequisite above | Run the 600 clean and 1,200 simple-v2 pools from `main` into `results/v773_*` with their own manifest naming the V7.7.2 training source and the V7.7.3 runtime. Never combine with V7.7.2 logs. |
| 2 | Dispatch the `canary` pool | same | 40 jobs, 240 required rows (two polarities × three source shifts per checkpoint group), from the audited source into its own output root. Old NMOS-only logs are incomplete and are not reused. |
| 3 | Freeze `simple-v2` thresholds | a LEVEL=72 three-repeat stability matrix | Gather `--reference-repeats 3` at every nominal cell, then freeze thresholds and units per the [simple-v2 contract](../accuracy/simple-circuits-v2-topologies.md). Only after that may any simple-v2 row become qualification evidence. |
| 4 | Give the ring stage-count axis a catalog home | denominator decision | Add `ring_osc_{3,7,9}stage` as `simple-v2` diagnostic cases on the templates the harness `ring_n_stages` path already renders; regenerate the render freeze; then retire `circuit_sweep.py`, `verify_circuit_sweep.py`, its canaries and their five contract tests. This changes the simple-v2 denominator: compare only within V7.7.3. |
| 5 | PMOS bias-fanout ladder | new templates | Mirror `bias_tree_fanout_{3,5,9,17}t` for PMOS (reference from ground, PMOS mirrors from VDD), prove NGSPICE render parity, reuse the `bias_fanout_op` profile and its oracle. Changes the denominator likewise. |
| 6 | `run_simulation` coverage | none | Done in this arm's opening commit: an in-process witness dispatches `.tran`, `.dc`, `.ac` and the bare `.op` to their writers, so the tracer sees the dispatcher the `main.py` subprocess test hid. |

## Not in scope

- The twelve manual gates stay manual; `tests/README.md` records which pool
  runs what.
- No edit, checkout or test run inside `PyCircuitSim-v771` or
  `PyCircuitSim-v772` while their processes exist.

## Completion conditions

Items 1–3 have complete campaign evidence under `results/v773_*`, reports in
`docs/accuracy/` that name both source hashes, and a CHANGELOG entry that
states what moved against V7.7.2 under identical denominators. Items 4–5 land
with regenerated frozen renders and a rescaled denominator note. Package
workflow and maintenance releases do not imply completion of this evaluation.
