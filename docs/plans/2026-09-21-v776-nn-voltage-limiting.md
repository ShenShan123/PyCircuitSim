# V7.7.6 — NN-side voltage limiting for LEVEL=75/76

Opened 2026-09-21, after the [round-2 evaluation](https://github.com/ShenShan123/PyCircuitSim/blob/bde2c11/docs/accuracy/v775-round2.md).
No accuracy claim and no retraining: this is a solver-contract change whose
gating is defined below.

**Status: implemented in V7.7.6, off by default.** The knob is
`PYCIRCUITSIM_NN_NR_LIMIT=1`; the [changelog](../CHANGELOG.md#v776--capacitor-state-fix-and-opt-in-nn-limiting)
records the design and its contract tests. Success criterion 4 is the
V7.7.6 evaluation, which runs the full campaign once with the knob off (the
scored runtime) and once with it on. Its [result](../accuracy/v776-evaluation.md#nn-limiter-re-gate):
criteria 2–3 met, criterion 1 partly (71 of 128 support rejections converge),
and no default flip proposed.

## Problem

`mosfet_directnet_full.py` calls `_check_support()` on every evaluation and
raises when any input leaves the persisted normalization box. Nothing limits
the evaluation voltages before that check. LEVEL=72 has an equivalent bound
(`_NR_LIM_WINDOW` in `mosfet_cmg.py`), which both shapes NR steps and bounds
the internal-node solve; LEVEL=75/76 have only the hard rejection.

A Newton trial state can therefore end a solve whose physical answer is inside
the box.

## Evidence

A reference-only diagnostic ran 440 times across rounds 1 and 2. It covered
`ota_5t_buffer`, `current_mirror`, Miller `opamp`, `cascode_stack`, `nand2`,
`nor2`, `diffpair_active_load`, `multistage_buffer_12t`, `unity_gain_buffer`,
`ldo_regulator` and `beta_multiplier`. That is every case that raised a
`CandidateSupportError` in a DC or transient analysis except BSIM-AR
`self_biased_cascode`. It ran only the NGSPICE LEVEL=72
reference and checked each accepted terminal voltage against the box of all 40
checkpoint pairs.

Every accepted point is inside support, except NAND2/NOR2 transient samples
equal to the documented NGSPICE `uic` start-up spike. Raw evidence is under
`PyCircuitSim-v775-round2/results/v775_round2/l4_support/`.

Round 2 recorded 36 support-rejection rows for DirectNet and 64 for BSIM-AR.
The diagnostic checked the analyses behind 65 of those 100. The other 35 are 33
AC rows, whose operating points were never checked, and the two
`self_biased_cascode` rows; they remain unattributed. The rejections are
concentrated in `ldo_regulator`, `beta_multiplier`,
`multistage_buffer_12t` and the logic stacks. The related basin symptom is the
round's only OMP flip: BSIM-AR medium TSMC12 Miller DC passes at 1 and 2
threads and fails to converge at 4.

## Scope

- Apply voltage limiting to LEVEL=75/76 evaluation voltages, or damp the NR
  step, so trial states stay inside the certified box.
- Keep the physical MNA voltages uncorrupted, as the existing contract
  requires: a limited iteration is not converged, and an unlimited follow-up
  iteration must confirm convergence.
- Leave the hard `_check_support()` rejection in place for genuinely
  out-of-domain physical solutions. This change must not convert a real
  coverage hole into a silent extrapolation.

## Constraints and gating

- The limiter perturbs floating-point results, so it ships **disabled by
  default** until a full accuracy re-gate clears it (performance discipline in
  `AGENTS.md`).
- It can alter a nonlinear solution basin, so it must first pass the
  latch-basin contract in `tests/test_solver_numerics_contracts.py`, with the
  new knob added to that test's march parametrization.
- Both stored states of the bistable cell must survive the new knob.
- Re-gating must use a separately provenanced arm; round-2 numbers cannot be
  reused for a changed runtime.
- The runtime baseline is `main` at merge `b688412`, which carries the
  evaluation-arm harness repairs both rounds ran on.

## Success criteria

1. Support-rejection `ERROR` rows either converge or remain errors for a
   documented physical reason, not for a trial-state excursion.
2. Cells that already converge produce bit-identical results when the limiter
   never engages.
3. The latch-basin and solver-numerics contracts pass with the knob on and off.
4. A re-gate of the affected cases shows the change's effect on convergence
   counts before any default flip is proposed.

## Open question

Whether the same mechanism explains the BSIM-AR convergence-cost tail: the
last five round-2 cells took 28–53 h each, and cost tracked convergence
difficulty rather than circuit size. If step damping reduces retry storms, the
runtime gap to DirectNet (a median 39× per round-2 cell on identical cases)
may narrow; that is a hypothesis, not a claim.
