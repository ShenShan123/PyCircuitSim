---
title: V7.6.4 Complex-Circuit Closure Loop
type: investigation
date: 2026-08-29
closed: 2026-08-31
status: closed
---

# V7.6.4 complex-circuit closure loop

## Verdict

The loop did not produce a promotable LEVEL=75 checkpoint or runtime adapter.
DirectNet-Full remains **0/248** on the fixed tracked AnalogGym denominator,
and LEVEL=73 large remains the served DirectNet family.

The investigation isolated a nonlinear basin-entry failure that is not fixed
by terminal-length support, lower pointwise/Jacobian error, a tighter Newton
cap, reference-seeded circuit loss, cold-state distillation, or a
compact-input-only local adapter. Every candidate failed a predeclared device
or production-circuit gate before a broader campaign was opened.

## Fixed contract

- Ground truth: NGSPICE using the identical LEVEL=72 BSIM-CMG OSDI model.
- Target: LEVEL=75 DirectNet-Full on the tracked 248-deck scored denominator;
  unsupported, failed, and non-comparable rows remain visible.
- Primary gate: complete deck agreement. Device error, comparable-cell
  agreement, convergence, and support coverage are diagnostics or prerequisite
  gates, never substitutes for a deck pass.
- Scored execution: CPU with one OpenMP, MKL, and Torch thread.
- Correctness: full terminal KCL/charge closure, finite stamps, unchanged
  support sidecars, exact provenance, and no denominator changes.
- Search discipline: one explanatory variable per arm, no post-result
  coefficient/radius/epoch search, and stop at the first sealed-gate failure.

## Baseline

| evidence | result | role |
|---|---:|---|
| LEVEL=72 matched-engine AnalogGym | 242/248 | reference harness ceiling |
| V7.6.2 production-sized LEVEL=75 large | 0/248; 174 Py failures; 41/326 comparable metric cells agree | clean control |
| V7.6.3 targeted medium | 20/20 strict simple circuits, but 0/248 AnalogGym | development evidence only |
| fixed TSMC5 DC development basket | 0/15 | bounded feedback basket |
| strict ldo_1/tb_line_max | parent red, 0/28 | fastest production-basin gate |

Terminal-upper-edge regeneration removed the dominant static length-support
error, but completed rows still had large physical state errors and NFIN=1
remained deliberately unsupported. Support closure was necessary, not
sufficient.

## Experiment ledger

| cycle | single hypothesis | decisive result | decision |
|---:|---|---|---|
| 1 | Preserve the parent normalization while appending matched terminal-length rows. | Song returned 0/8 metrics after a support escape; PMOS replay also regressed. | Reject; normalization refitting was a real confound but not the closure mechanism. |
| 2 | All-geometry, geometry-disjoint Hermite current-J supervision. | Held-out current-J improved only 1.22%/1.19% versus 25% required; no epoch passed value maxima. | Reject before circuits. |
| 3 | Correct the multiplicity-dominant LDO PMOS pass-current row. | The full row failed device gates; a predeclared constrained step passed them but production LDO solved 0/28. | Reject; static eligibility did not move the basin. |
| 4 | Train on differentiable LEVEL=72-solved LDO curves. | Reference-seeded Vout MAE fell 70.312 to 23.558 mV and device gates passed; production LDO still solved 0/28. | Reject; solved-curve learning did not teach basin entry. |
| 5 | Restore the native 0.1 V LEVEL=75 Newton cap in the AnalogGym adapter. | 27/28 points returned, but 0/28 converged, 0/3 metrics passed, worst node error was 0.973587 V, and runtime rose 1.49 to 617.77 s. | Reject; changed the failure mode without closure. |
| 6–8 | Distill exact LEVEL=72 residuals and capped Newton steps at deterministic in-support production cold states. | Hardened trace/label artifacts reproduced exactly. Treatment best residual/step ratios were 1.29005/0.931470 versus 0.50/0.50 required, and every epoch failed worst-case value gates. | Reject; publish no checkpoint and keep all production decks sealed. |
| 9 | Fit an exact compact-input local current/Jacobian adapter. | The LEVEL=72 oracle reproduced residual and step but had support-margin ratio 6.51408 versus 0.25. | Reject before fitting. |
| 10 | Use exact LEVEL=72 current values with the frozen parent Jacobian. | Residual ratio was effectively zero, but step/margin ratios were 1.00504/1.0. | Reject before fitting. |
| 11 | Use fixed flat-top C2 local regions under closed-support safety. | Overlapping centers made the Hermite systems rank-deficient; current/J errors exceeded the exact-label contract. | Reject before publication. |
| 12 | Use deterministic non-overlapping C2 trust regions. | Next-state coverage was only 423/756 device rows. | Reject without enlarging the registered radius. |
| 13 | Use a fixed 0.25 V product-Wendland C2 affine basis. | Cold reconstruction and closure passed, but PMOS drain-current normalized MAE reached 1.02320x replay and 1.05803x validation versus 1.02x allowed. | Reject before publication. |
| 14 | Optimize only the affine Hermite null space against replay. | Every replay-intersecting scope was full column rank, leaving no usable null space. | Reject before implementation. |
| 15 | Add quadratic off-center freedom to the fixed Wendland basis. | Replay, PMOS, closure, support, and exact-repeat checks passed; held-out NMOS normalized MAE reached 1.022918x versus 1.02x allowed. | Reject before publication or circuit execution. |

No cycle after the clean baseline ran /248. ldo_2, LDO load, Song, and the
held-out /15 basket stayed sealed whenever their prerequisite failed.

## Durable conclusions

- Full-terminal length coverage and a 0.1 V LEVEL=75 iteration cap are valid
  runtime/data contracts, but neither closes AnalogGym.
- Pointwise device improvement, derivative improvement, and even a
  reference-seeded circuit solution do not establish production basin entry.
- The failing LDO state is coordinated across devices and free-node KCL rows;
  an MM8-only or single-output correction is not a defensible production fix.
- Exact LEVEL=72 current/Jacobian substitution can reproduce the desired
  Newton step while conflicting with a previously chosen inner-margin gate.
  Closed-support membership is the physical safety rule; changing that rule
  does not rescue the rejected adapter representations.
- Worst-case replay and point-disjoint gates are binding. Mean error
  improvements cannot excuse a tail regression.
- A future arm must use a new, prospectively registered mechanism and
  independent development/held-out circuits. Retuning these rejected losses,
  radii, kernels, epochs, or coefficients is not authorized evidence.

## Retained product changes

The closure experiments did not promote model weights. The independently
qualified V7.6.2/V7.6.3 work retained:

- checksum-bound dataset-to-checkpoint provenance;
- terminal upper-length and pass-device corridor coverage;
- the DirectNet-Full PMOS scalar-current comparison sign without changing its
  full terminal stamp;
- the LEVEL=75 0.1 V DC/transient iteration cap; and
- fail-closed AnalogGym agreement, finite-sweep, convergence, truncation, and
  denominator checks.

Experiment-only solver factories, local-adapter loading, private trainers,
harvesters, fitters, tests, and rejected result/checkpoint payloads are not
part of the product.

## Artifact closure

Tracked reports and Git history own the measurements above. Repository-local
retention is limited to the active five-technology large DirectNet bundles
and the current V7.6.2/V7.6.3 qualification evidence. V7.6.4–V7.7.4 matched
data, cold traces, labels, overlays, unpublished adapters, and intermediate
work directories are stale after this closure and may be purged.

Any new complex-circuit recovery effort must start with a new plan, declare a
distinct mechanism and gate order before generating artifacts, and use
LEVEL=72 OSDI as the only ground truth.
