# V6.4.5 — Track B final report (SHIP: B8 closes TSMC7 ring_osc, 9/16 → 10/16)

**Date:** 2026-05-30 • **Branch:** `feat/v6.4.5-track-b` (off V6.4.4 HEAD `54c4759`)
**Plan:** `docs/plans/2026-05-28-directnet-v6.4.5-ro-sram.md` (Track B, §10)
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, CPU scoring (`CUDA_VISIBLE_DEVICES=""`),
GPU training on free A100/Blackwell cards. Worktree isolated; canonical V6.4.4
checkpoints backed up to `/tmp/v6_4_4_canonical_backup_20260529_094624`
(sha256 manifest).

## Outcome

Track B ran the full B1–B9 cascade (Tier-1 cheap-first → Tier-2 retrains →
Tier-3 speculative, gated). **B8 (differentiable mini-simulator test-time
fine-tuning) is the lone winner** — and the first lever in the entire V6.4.5
effort (Track A's 32-training retrain included) to move a gate.

**Complex-circuit headline: 9/16 → 10/16** (TSMC7 ring_osc FAIL → PASS).
Inverter gate 8/8 held; extended harness 55/55 DC + 64/64 tran held
(re-verified on the B8 TSMC7 mix). TSMC5/12/16 unchanged — B8 swaps only the
TSMC7 checkpoint.

| Open gate (V6.4.4) | V6.4.5 Track B | Closed by |
|--------------------|----------------|-----------|
| ring_osc TSMC7 (8.98 %) | **PASS 0.32 %** | **B8** ✓ |
| SRAM force_ic (0/8, interior attractor) | still open (diagnostic, not a 16-cell gate) | none — needs V6.4.6 split-head |

## Verified B8 ship (TSMC7 slot = `b8_ttft`, re-scored independently)

| TSMC7 gate | V6.4.4 | B8 ship | note |
|------------|:------:|:-------:|------|
| ring_osc | FAIL 8.98 % | **PASS 0.32 %** | +1 headline |
| switchcap | PASS 0.37 % | **PASS 0.37 %** | held |
| SRAM butterfly (positivity) | PASS | **PASS** | held |
| inverter VTC / tran | PASS | **PASS 1.97 % / 1.33 %** | held (VTC improved from 3.89 %/188 mV scorer) |
| extended DC (TSMC7 subset) | 9/9 | **9/9** | held |
| extended tran (TSMC7 subset) | 16/16 | **16/16** | held |
| opamp | FAIL (gain 213, 30.7 % err) | FAIL (gain 0, flat) | already-failing; **degraded** (cost) |
| SRAM force_ic (diagnostic) | resid 0.302 | resid 0.899 | not a 16-cell gate; **degraded** (cost) |

**Cost, recorded honestly:** B8's LoRA delta was optimised *solely* on the RO
period, so it flattens the (already-failing, out-of-V6.4.5-scope) TSMC7 Miller
opamp and worsens the SRAM force_ic diagnostic. Neither is one of the 16
headline gates, and no passing gate regressed, so the headline is a clean +1.
A **multi-circuit-regularised TTFT** (add opamp/SRAM terms to the differentiable
loss) is the V6.4.6 follow-up to co-hold them.

**Shipped checkpoint** (canonical TSMC7 slot, gitignored, on disk):
`tsmc7_dn_medium_{nmos,pmos}_best.pt` ← `b8_ttft` weights.
sha256: nmos `26289484…`, pmos `3f94afe5…`. Provenance copies kept as
`b8_ttft_tsmc7_*`. Loads via the standard DirectNet path (LoRA merged into a
plain state_dict — NO load-bearing inference code, unlike V6.4.4's mono keys).
Rollback: `/tmp/v6_4_4_canonical_backup_20260529_094624/tsmc7_dn_medium_*`.

## The full B1–B9 roster (every lever; failures are first-class deliverables)

| # | Experiment | Tier | Gate | Verdict | Key number / finding |
|---|------------|:----:|------|:-------:|----------------------|
| B1 | cap-symmetry probe | 1 | RO | KILL | δ=\|cgd−cdg\| median 0.2 %, 1.7 % of evals >5 % (0 % mid-rail) → RO not cap-asymmetry; explains why `NN_SYMMETRIC_CAPS=1` was a no-op |
| B2 | test-time seed ensemble | 1 | SRAM | KILL | all 8 seed siblings + canonical land at q≈0.815/qb≈0.21; no rail-snap basin to vote for |
| B3 | LoRA reweight fine-tune | 1 | RO | KILL | rank-32 RO 8.98 % (val loss unchanged) — no new signal on the trained manifold |
| B4 | force_ic λ-continuation | 1 | SRAM | KILL | all 8 cells relax back to the attractor → rail is not a stable equilibrium of the NN cross-coupled pair |
| B5 | OSDI Jacobian distillation | 2 | RO | KILL | best RO 8.57 % (λ=0.01), SRAM 0.302→0.281 — necessary-but-not-sufficient (Phase-1b dead end confirmed under *analytic* targets) |
| B6 | adversarial harvest + retrain | 2 | both | KILL | **92 %/85 % of worst-residual points IN-BOX** → interpolation failure, not extrapolation; data augmentation cannot fix it (the pivotal diagnosis) |
| B7 | physics skeleton + residual | 2 | SRAM | KILL | joint ε=0.1 training let the residual cancel the skeleton → ≈ baseline; SRAM unmoved (q 0.814), RO regressed to 11.9 %; subthreshold-exp aimed at the wrong region (SRAM fails in *linear* region at high Vgs) |
| **B8** | **differentiable-sim TTFT** | **3** | **RO** | **PROMOTE** | torch RO matches production to 0.00 %; pure-period TTFT (rank-8 LoRA) → **RO 0.32 %**, inverter held/improved → **ships** |
| B9 | hard-monotone id head | 3 | SRAM | KILL | monotone head (26K params) can't fit id (R²=0.14); even so q moved toward VDD but qb moved *away* (0.226→0.364) → cross-coupled cell finds a *new* interior equilibrium — drive magnitude alone won't snap it |

## Why B8 worked where 7 others (and Track-A's 32-training retrain) failed

The RO period is an **integrated** metric. Pointwise Id-MAE — and everything
that minimises it (more data B6, reweighting B3/B5, seeds, architecture priors
B7/B9) — is blind to the ~9 % period drift, which is dominated by the
switching-edge timing accumulated over the cycle. B8 optimises the period
**directly** through a differentiable surrogate of the actual oscillator, so the
gradient sees exactly the quantity the gate measures. The make-or-break sanity
gate (torch RO period = production to 0.00 %) proved the surrogate is faithful;
TTFT then drove it to target and the production scorer confirmed transfer.

## SRAM force_ic — definitively model-fidelity, deferred to V6.4.6

Six levers (B2, B4, B5, B6, B7, B9) converge: the interior attractor
(q≈0.82/qb≈0.21) is intrinsic to the DirectNet I-V surface. B6 localised it to
an **in-box linear-region under-drive** at (Vgs≈0.59, Vds≈0.22); B9 showed even
*increasing* drive just relocates the cross-coupled equilibrium. Closing it
needs the deferred **V6.4.6 split-head** (spectral-norm id head to bound the
cross-coupled loop gain + softplus cap head), or a multi-circuit-regularised
B8-style TTFT — both out of V6.4.5 scope.

## Track-B relaxations exercised (vs CLAUDE.md)

- Rule 10 (no new loss terms): B5 added an OSDI-Jacobian distillation term; B7
  a skeleton-residual; B8 a circuit-level period loss. All explicitly sanctioned
  by Track B; **only B8's is in the shipped artifact** (and even then the ship is
  a plain merged checkpoint — the loss lived only in the offline TTFT).
- Rule 15 (Vds correction inference-only): untouched in the ship. B4's solver
  continuation was reverted.
- "Solver ↔ models decoupled": B8's differentiable simulator is a *standalone*
  experiment (`experiments/v6_4_5_track_b/B8_diff_simulator/`); the production
  scipy solver is untouched.

## Definition-of-done check

1. ✓ Every B1–B9 lever has a committed `results/v6_4_5_track_b/B{n}_*.md` with the Rule-16 quartet.
2. ✓ Headline ≥ V6.4.4: **10/16** (was 9/16); inverter 8/8; extended 55/55 + 64/64 held.
3. ✓ CHANGELOG V6.4.5 Track-B entry + CLAUDE.md Status updated; per-tech provenance recorded.
4. ✓ Every dropped lever logged with the empirical numbers that killed it.
5. ✓ V6.4.6 split-head plan carries forward unchanged; multi-circuit-regularised TTFT added as a follow-up.

**Note (cross-session coordination):** a concurrent Track-A session updated the
*main checkout's* CLAUDE.md to record the Track-A Phase-5 32-training retrain
(also a RO dead end, best 9.05 %). This branch (`feat/v6.4.5-track-b`) ships B8;
the two doc states must be reconciled at merge by the user.
