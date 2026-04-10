# BSIMAR Cross-Technology Transfer Roadmap - 2026-04-09

This note merges the current BSIMAR implementation review with a second
agent's proposal list, deduplicates overlapping ideas, and turns them
into a practical experiment roadmap for improving cross-technology
transferability.

The immediate motivation is that the newest experiments show weak
cross-technology accuracy, especially under zero-shot transfer.

## Goal

Improve BSIMAR's zero-shot and low-shot transfer across technology nodes
and variants without giving up the current in-distribution gains of the
v3 recipe.

## Merged Idea List

### M1. Protocol and feature integrity

Audit end-to-end train/inference feature parity, especially the geometry
feature path and `L`, and replace pooled random validation with
leave-one-tech or leave-one-variant evaluation plus macro-average and
worst-tech checkpointing.

Why:

- If the feature path is inconsistent between training and inference,
  transfer numbers can be artificially degraded.
- The current pooled random split mainly optimizes interpolation, not
  true cross-technology generalization.

### M2. Group-aware optimization

Use technology-balanced or variant-balanced sampling, or a group-robust
objective such as group-DRO, instead of letting dense technologies
dominate the loss.

Why:

- Cross-technology transfer usually fails first on underrepresented
  nodes, variants, or bias regimes.
- Pooled averages can hide worst-group regressions.

### M3. Continuous technology conditioning

Merge process-parameter jitter with a richer conditioning path, such as
multiple process tokens or FiLM / conditional LayerNorm driven by
process parameters.

Why:

- The current model compresses all process information into a single
  process token, which may be too lossy for transfer.
- Process-parameter jitter encourages learning a smooth technology
  manifold instead of memorizing discrete technology clusters.

### M4. Constrained `Vov` / `Vth` conditioning

Retry `Vth_hat`, but with supervised anchors or hard bounds so the
learned threshold proxy cannot drift arbitrarily.

Why:

- This directly targets the missing threshold structure that is likely
  to break transfer across technology nodes.
- The earlier unconstrained `Vth_hat` attempt failed because it drifted,
  not because threshold-aware conditioning is fundamentally wrong.

### M5. Multi-tech pretraining plus lightweight adaptation

Restore a real pretraining path, then adapt to new technologies with
lightweight modules such as adapters, LoRA, or few-shot finetuning.

Why:

- The README still positions BSIMAR as a pretraining plus finetuning
  framework, but the current code path is mostly a supervised universal
  recipe with AR finetuning.
- This is a natural fit for cross-technology transfer, where shared
  device physics exists but node-specific details remain.

### M6. Physics-informed residual head

Add an EKV or Pao-Sah style analytic core for `id`, `gm`, and `gds`,
and restrict BSIMAR to learning bounded residual corrections.

Why:

- This offloads the hardest, most technology-sensitive current law
  structure to an analytic prior.
- The neural network can then focus on residual technology- and
  geometry-specific effects.

### M7. Controlled capacity sweep

Keep model scaling in the list, but treat it as a later calibration
step, not the primary fix.

Why:

- More capacity may help if the current model is underfitting the
  multi-technology surface.
- But scaling alone usually does not fix the wrong inductive bias.

### M8. Upstream smoothness regularization

Test intermediate-layer spectral normalization or Jacobian-style
regularization rather than constraining only the final output heads.

Why:

- Head-only spectral normalization already looked ineffective.
- If transfer brittleness comes from sharp upstream features, smoothing
  the encoder is more principled.

### M9. Structured specialization

Test output-group or regime-based experts, but only with strong router
regularization and multi-seed checks.

Why:

- This may help if failures cluster by physical regime or target group.
- It is also easy to overfit seen technologies, so it belongs later in
  the roadmap.

### M10. Physics-structured dependency mask

Keep a physical dependency attention mask as a low-priority experiment.

Why:

- It may reduce non-physical shortcuts.
- But the current architecture already uses causal masking, so this is
  less urgent than transfer-aware evaluation, conditioning, and physics
  priors.

## Experimental Roadmap

### Stage 0 - Baseline the right problem

Before changing the model, make sure the benchmark measures the right
failure mode.

Tasks:

1. Verify train/inference feature parity, especially `L`,
   geometry, and process parameters.
2. Replace pooled random validation with leave-one-tech and
   leave-one-variant evaluation.
3. Report:
   - macro-average zero-shot NRMSE / MRE / R2
   - worst-tech zero-shot NRMSE / MRE / R2
   - transfer gap = zero-shot - in-distribution
4. Keep one frozen baseline checkpoint and one frozen split definition
   for all later comparisons.

Success gate:

- We have a trusted cross-technology baseline that is reproducible.

### Stage 1 - Fix the training objective before changing the architecture

Start with the cheapest and most structural fixes.

Experiments:

1. `M2` group-aware optimization:
   - balanced technology / variant sampling
   - macro-average or worst-tech checkpoint selection
   - optionally group-DRO style reweighting
2. `M3` process-parameter jitter:
   - small, physically plausible perturbations
   - ideally correlated or normalized/log-space jitter rather than
     naive independent Gaussian noise

Success gate:

- Worst-tech and macro-average zero-shot metrics improve without
  destabilizing in-distribution accuracy.

### Stage 2 - Improve technology conditioning

Once the protocol is fixed, improve how the model represents
technology-dependent information.

Experiments:

1. Replace the single process token with multiple process tokens.
2. Try FiLM or conditional LayerNorm driven by process parameters.
3. Combine richer conditioning with the same process-jitter setup from
   Stage 1.

Success gate:

- Held-out-technology current and conductance metrics improve,
  especially on the worst technology nodes.

### Stage 3 - Retry threshold-aware features correctly

Revisit `Vth_hat`, but in a controlled way.

Experiments:

1. Add a bounded `Vth_hat(geom, process)` head.
2. Anchor it with weak supervision or a physical prior.
3. Feed `Vov = Vgs - Vth_hat` and optionally a normalized drain-drive
   feature.
4. Compare against the current `Vg`-proxy approach using the fixed
   transfer benchmark only.

Success gate:

- `id`, `gm`, `gds`, and `gmb` improve on held-out technologies.
- If capacitances improve but currents regress again, reject and move on.

### Stage 4 - Add an explicit transfer-learning path

If Stages 1 to 3 help but do not close the gap enough, add a real
pretraining plus adaptation workflow.

Experiments:

1. Multi-technology pretraining with a self-supervised or
   teacher-forced objective.
2. Lightweight adaptation on new technologies using:
   - adapters
   - LoRA
   - few-shot finetuning

Success gate:

- Better zero-shot or low-shot transfer than the supervised-only
  universal baseline.

### Stage 5 - Add stronger physics priors

Introduce the most ambitious but most principled architecture change.

Experiments:

1. Add an EKV-style residual head for `id`, `gm`, and `gds`.
2. Keep charge and capacitance heads neural in the first version.
3. Start on NMOS first before expanding the scope.

Success gate:

- Clear zero-shot gains on current and conductance outputs, especially
  on worst-tech slices.

### Stage 6 - Later-stage architecture ablations

Only spend effort here after the earlier stages are exhausted.

Experiments:

1. `M7` controlled capacity sweep
2. `M8` upstream smoothness regularization
3. `M9` structured specialization / experts
4. `M10` physics-structured dependency masks

Success gate:

- A new idea must beat the strengthened baseline on macro-average and
  worst-tech transfer metrics, not just pooled averages.

## Recommended Execution Order

### Batch A - Highest ROI

1. Stage 0 baseline and feature audit
2. Stage 1 group-aware optimization plus process jitter
3. Stage 2 richer technology conditioning

### Batch B - Most targeted cross-tech fix

4. Stage 3 constrained `Vth_hat` / `Vov` conditioning

### Batch C - Stronger transfer mechanisms

5. Stage 4 multi-tech pretraining plus lightweight adaptation
6. Stage 5 EKV residual head

### Batch D - Broader architecture exploration

7. Stage 6 later-stage ablations

## Decision Rule

Promote an experiment only if it improves at least one of:

- macro-average zero-shot transfer gap
- worst-tech zero-shot NRMSE
- worst-tech zero-shot MRE
- worst-tech zero-shot R2

without causing a meaningful regression in:

- pooled in-distribution accuracy
- transfer stability across seeds
- training stability or wall-clock to an unacceptable level

## Short Priority Summary

If only a few things can be done soon, the best order is:

1. Fix the benchmark and feature path.
2. Add group-aware training and process jitter.
3. Strengthen technology conditioning.
4. Retry constrained `Vth_hat`.
5. Add pretraining plus lightweight adaptation.
6. Add the EKV residual head.

The remaining ideas are worth exploring later, but they should not come
before the protocol, conditioning, and physics-prior fixes above.
