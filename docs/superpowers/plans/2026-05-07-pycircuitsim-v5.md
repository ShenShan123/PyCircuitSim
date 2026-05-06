# Plan — PyCircuitSim **V5**: solver fixes, V5 dataset, and Jacobian-consistency loss A/B

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`.
> Each step is checkbox-tracked (`- [ ]`).

**Date:** 2026-05-07
**Branch target:** new branch off `main` (suggested `feat/pycircuitsim-v5`)
**Status:** PLAN — DO NOT EXECUTE without explicit user approval.
**Severity:** Medium — multi-phase plan. Phase A is solver-only and
behaviour-preserving for the shipping checkpoints; Phase B regenerates
the dataset under a new V5 lineage; Phase C is a controlled small-arch
A/B between the original MAE+LDS loss and a new Jacobian-consistency loss.

> **Naming convention — PyCircuitSim V5.** This is the V5 cut of the
> project. Everything new in this plan gets a `v5_` prefix to keep it
> separate from the V4 / v4-re lineage:
>
> | Artefact class | V5 name |
> |---|---|
> | Branch | `feat/pycircuitsim-v5` |
> | Datasets | `external_compact_models/bsimar/data/datasets/universal_v5_{nmos,pmos}.npz` |
> | DirectNet checkpoints | `v5_dn_s_{nmos,pmos}_{mae,jac}_*` |
> | BSIMAR checkpoints | `v5_tf_s_{nmos,pmos}_{mae,jac}_*` |
> | Reports | `results/v5_*.md` |
> | Plan files | `docs/superpowers/plans/2026-05-07-pycircuitsim-v5.md` (this file) |
>
> **ASAP7 is excluded everywhere.** Training: `--exclude-techs asap7`.
> Data generation: drop ASAP7 from the tech list. Verification: never
> include ASAP7 in `--tech` flags. The current vocabulary is the 18
> TSMC variants; this is preserved in V5 (no vocab changes in scope).

---

## 0. One-paragraph summary

`results/sml_report_2026_05_06.md` shows three categories of remaining
failure for the V4 production NN compact-model stack: solver-class
(4 VTC OVERFLOWs and a TSMC16 BSIMAR mid-transient NR_FAIL), data
coverage (TSMC5 inverter-transient 17–20 % NRMSE floor, TSMC7 NMOS DC
3.27 % residual), and loss decoupling (`gds_supervised ≠ ∂id/∂Vds`).
Three parallel review agents triaged each category. The user-approved
sequencing is:
**(A)** fix the solver first and re-run the existing V4 production
checkpoints to confirm the convergence-class failures fall away;
**(B)** regenerate the dataset as V5 with trip-point and Vbs-jitter
overlays;
**(C)** train *small*-arch DirectNet and BSIMAR on the V5 data under
two loss variants — MAE+LDS (control) vs MAE+LDS+Jacobian-consistency
(treatment) — and produce a comparison report in `results/`.
Each phase has its own gate. The accuracy axis is **both NRMSE % and
MRE %**, reported per-tech. Reports must explicitly attribute each
accuracy delta to the responsible phase (solver / data sampling / loss).

---

## 1. Acceptance gates and metrics

### 1.1 Metrics (every report)

For every NN-vs-BSIM-CMG comparison cell:

* **NRMSE %** — `100 · sqrt(mean((nn − ref)²)) / sqrt(mean(ref²))`.
  Already implemented in `tests/common/nn.py::nrmse`.
* **MRE %** — `100 · mean(|nn − ref| / max(|ref|, ε))` with
  `ε = 1e-12 A` for currents and `ε = 1e-18 C` for charges to avoid
  divide-by-zero in deep cutoff. Already in `tests/common/nn.py::mre`.

Reports include both for every cell, side by side. The 15 % threshold
applies to NRMSE; MRE is informational and surfaces sub-threshold /
small-Id mismatches that NRMSE under-weights.

### 1.2 Per-step accuracy attribution (mandatory in every report)

Every report produced by this plan **must include a per-step delta
table** that attributes accuracy improvement to the responsible
phase. The structure is:

| Pain cell | Baseline (V4 prod, pre-V5) | After Phase A (solver) | After Phase B (V5 data) | After Phase C MAE arm | After Phase C JAC arm | Δ from solver | Δ from data | Δ from loss |
|---|---|---|---|---|---|---|---|---|
| TSMC5 inv-tran NRMSE % | … | … | … | … | … | post − pre | post − pre | post − pre |
| TSMC5 inv-tran MRE % | … | … | … | … | … | … | … | … |
| TSMC7 NMOS DC NRMSE % | … | … | … | … | … | … | … | … |
| TSMC7 NMOS DC MRE % | … | … | … | … | … | … | … | … |
| TSMC16 BSIMAR inv-tran NRMSE % | … | … | … | … | … | … | … | … |
| TSMC16 BSIMAR inv-tran MRE % | … | … | … | … | … | … | … | … |
| Inverter VTC pass-rate (out of 8) | … | … | … | … | … | … | … | … |

Phase A's report fills the first two delta columns once (solver delta
against baseline; data + loss columns are blank). Phase C's report
fills all three columns by re-running the prior-phase artefacts on the
post-Phase-C simulator so all numbers are apples-to-apples (same
solver, swapping only data and/or loss). This makes "what got us this
improvement" explicit.

### 1.3 Phase gates

* **Phase A (solver, V4 checkpoints):** with the V4 production
  checkpoints loaded, `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16`
  must show **VTC pass-rate ≥ 6/8** (no OVERFLOW rows) and **0 ERROR
  rows** in inverter transient. No regression on NMOS pulse 8/8 PASS or
  single-device DC 20/20 PASS. BSIM-CMG suites unchanged.
* **Phase B (V5 dataset):** the new `universal_v5_{nmos,pmos}.npz` files
  exist, load via the unchanged `load_and_split_bsimar` API, and contain
  `sample_class` tags that include the new `inv_trip` and `overshoot`
  classes. Total row counts within ±20 % of V4 B1 dataset.
* **Phase C (small-arch loss A/B):** four checkpoints trained per loss
  variant (`v5_dn_s_*_mae`, `v5_dn_s_*_jac`, `v5_tf_s_*_mae`,
  `v5_tf_s_*_jac`, per polarity = 8 checkpoints total). Side-by-side
  comparison report written to `results/v5_jac_loss_ab_<date>.md`,
  attributing accuracy delta to **(i)** Phase A solver, **(ii)** Phase B
  data overlay (compare V4-prod-on-Phase-A-solver vs MAE arm),
  **(iii)** Phase C JAC loss (compare MAE arm vs JAC arm). Decision
  criterion: the Jacobian-consistency variant must beat the MAE-only
  variant on **at least two** of the four sprint pain cells (TSMC5
  inv-tran, TSMC7 NMOS DC, TSMC16 BSIMAR inv-tran, inverter VTC
  pass-rate) on **both** NRMSE and MRE, without regressing anywhere by
  more than +1 pp. If yes, recommend Jacobian-consistency for full
  M-scale retrain in a follow-up. If no, document the dead-end and
  revert.

### 1.4 ASAP7 policy

ASAP7 is **never** in the dataset, the training set, or the verification
matrix in V5. The 18-code TSMC vocabulary is preserved. Any future
ASAP7 readmission is a separate vocab-extension project and not in
scope here. In `verify_nn_dc_tran.py`, `--tech` always reads
`TSMC5,TSMC7,TSMC12,TSMC16`.

---

## 2. Reviewers' compressed input

* **Data agent.** Bundle B (inverter-trip-point overlay + NR-overshoot
  densification + Vbs LHS jitter + drop the AND-gate row filter) is
  the production data recommendation. ~9 eng-h + ~4 GPU-h to retrain on
  the same architecture. Expected: TSMC5 inv-tran 17 → 8–10 %, TSMC7
  NMOS DC 3.3 → 1.5 %, VTC overflow 4 → 0–1 cells.
* **Model/training agent.** Highest-leverage single proposal across all
  three agents is the **Jacobian-consistency loss** (∂id/∂V → gds,
  ∂qg/∂V → cgg, etc.) at `λ_jac ≈ 0.1`. The supervised gds and the
  autograd gds the simulator's NR Jacobian consumes are decoupled
  today; even a perfect MAE-only retrain leaves that decoupling.
* **Solver/inference agent.** Three solver fixes total ~8 eng-h, no
  retrain: tanh+sech² rail-restoring extrapolation (replaces rule-19a
  quadratic that runs away under NR overshoot), NN-aware GMIN stepping
  default-on for LEVEL=73/74, and a mid-transient dt-halve fallback
  for NR exhaustion. Together expected to take §10.2 from
  `4/8 VTC PASS, 6/8 inv-tran PASS, 1 ERROR` to roughly
  `7/8 VTC PASS, 6/8 inv-tran PASS, 0 ERROR`.

The user-approved sequencing puts solver fixes first, regenerates the
dataset in parallel, then runs the loss A/B at small architecture for
fast iteration.

---

## 3. Phase A — Solver fixes, then re-run V4 checkpoints

**Branch:** `feat/pycircuitsim-v5`. **Effort:** ~1 eng-day. **GPU:** none.
**Risk:** very low — every change activates only inside paths that are
currently diverging or overflowing.

### 3.1 A1 — Tanh-saturated rail-restoring extrapolation

* Target: `pycircuitsim/models/mosfet_directnet.py` (`_apply_vds_correction`,
  the rail-restoring branch around lines 449–465). BSIMAR inherits via
  `_MOSFETBSIMARBase`.
* Replace the additive quadratic
  `id_extra = ½·g_max·overshoot²/x_ref`
  with
  `id_extra  = g_max·x_ref·tanh(overshoot/x_ref)`
  and the matching gds with
  `gds_extra = g_max·sech²(overshoot/x_ref)`.
* The two functions match the current quadratic to second order at
  `overshoot=0`, so converged in-distribution operating points are
  unchanged to NR tolerance. Past the boundary they asymptote to a
  hard cap of `g_max · x_ref` (≈ 0.2 mA at VDD = 0.4 V), preventing the
  `1e150` runaway seen in TSMC5 BSIMAR-M VTC.
* Update CLAUDE.md rule 19 step (a) wording to the new tanh/sech² form.

- [ ] A1.1 Implement tanh/sech² in `_apply_vds_correction`.
- [ ] A1.2 Update CLAUDE.md rule 19 step (a) wording.

### 3.2 A2 — NN-aware GMIN stepping default-on

* Target: `pycircuitsim/simulation.py`. Add a one-line helper
  `_circuit_has_nn(circuit)` that returns True if any device is
  LEVEL ≥ 73. When True, pass `use_gmin_stepping=True` with the
  existing schedule `[1e-6, 1e-8, 1e-10, 1e-12]` to every `DCSolver`.
* BSIM-CMG (LEVEL=72) keeps `use_gmin_stepping=False` so the
  `verify_bsimcmg_*` suites stay byte-identical.

- [ ] A2.1 Add `_circuit_has_nn(circuit)` helper.
- [ ] A2.2 Wire it into all four `DCSolver(...)` call sites in `simulation.py`.

### 3.3 A3 — Mid-transient dt-halve fallback

* Target: `pycircuitsim/solver.py`, the per-step NR loop (~L1490–1623).
* On `max_iterations` exhaustion, gated on `_is_nn_circuit`: catch,
  restore `voltages = voltages_prev` and integration history
  (`q_prev`, `q_prev2`, integration-method state), halve `dt`, retry.
  Up to 4 successive halvings (16× sub-resolution) before re-raising.
* Orthogonal to LTE sub-stepping (which handles integration error, not
  NR failure). Always-on for NN circuits; no opt-in flag.
* Log every dt-halve event so a Phase A regression run that needs more
  than 1 halving on TSMC12/TSMC16 (currently passing cleanly) escalates
  as a model-fit issue, not a solver-only issue.

- [ ] A3.1 Implement dt-halve fallback gated on LEVEL ≥ 73.
- [ ] A3.2 Apply the same partial-result fallback to
      `run_pycircuitsim_nn_inverter_tran` (SML §10.4 follow-up #2,
      ~30 LOC) so the remaining ERROR row in the report becomes a
      numeric FAIL.

### 3.4 A4 — Re-run existing V4 production checkpoints, write Phase A report

* No retrain. Use the production `v4_universal_*` and `v4_dn_universal_*`
  checkpoints already on disk. Cascade resolver unchanged.
* Run the full matrix:
  ```bash
  conda run -n pycircuitsim python tests/verify_nn_dc_tran.py \
      --tech TSMC5,TSMC7,TSMC12,TSMC16
  ```
* Write **`results/v5_phase_a_solver_<date>.md`** with:
  * Pre-fix vs post-fix `failure_class` distribution.
  * Per-tech NRMSE % **and MRE %** tables for: single-device DC
    (NMOS+PMOS), NMOS pulse, inverter VTC, inverter transient.
  * **Per-step delta table** (per §1.2): solver column populated;
    data and loss columns left blank for now.
  * BSIM-CMG suite delta vs pre-sprint baseline (must be 0).
  * Sign-off table against §1.3 Phase A gate.
* If the gate is met, commit and tag `v5-phase-a-solver`.

- [ ] A4.1 Full regression run with V4 checkpoints.
- [ ] A4.2 Write `results/v5_phase_a_solver_<date>.md` (NRMSE + MRE +
      §1.2 delta table with solver column populated).
- [ ] A4.3 Tag `v5-phase-a-solver` once the §1.3 gate is met.

---

## 4. Phase B — V5 dataset regeneration

**Branch:** same `feat/pycircuitsim-v5`. **Effort:** ~0.5 eng-day code +
~30 min generator runtime. Can run in parallel with Phase A coding.
**Risk:** low. Uses existing sampler primitives.
**Gate (§1.3 Phase B):** `universal_v5_{nmos,pmos}.npz` exist, load
cleanly, contain the new `sample_class` tags.

### 4.1 B1 — Inverter trip-point overlay (`inv_trip`)

* Target: `external_compact_models/PyCMG/pycmg/nn_generate.py`.
* For each `(tech, variant)`: extract `Vth_n` and `Vth_p` from the
  modelcard via the existing `extract_process_params` helper, then
  seed a 25 × 9 × 3 (Vg × Vd × Vbs) grid in box
  `[Vth − 0.10, Vth + 0.15] × [0.30·VDD, 0.70·VDD] × {0, ±0.25·VDD}`.
* Tag with `sample_class="inv_trip"`. Budget: ~675 points × 3 T ×
  ~12 (L, NFIN) ≈ 24 k rows / variant / polarity.

### 4.2 B2 — NR-overshoot densification + lower box factor

* Same file. Lower the `grid` class box factor from 2.0 → 1.5
  (saves ~25 % rows). Reroute ~800 rows / bin to a separate
  `overshoot` class: 20 × 20 grid in
  `(Vgs, Vds) ∈ [VDD, 1.6·VDD]² × Vbs=0`. Tag with
  `sample_class="overshoot"`.
* Smooth-join with the Phase A tanh ramp is C∞ regardless of the
  `VDD_train` value the loader extracts.

### 4.3 B3 — Vbs LHS jitter

* Same file. Add 600 LHS samples / bin holding (Vgs, Vds) on the
  existing `grid` 30×30 lattice and jittering Vbs ~ U(−0.5·VDD,
  +0.5·VDD) once per (Vg, Vd). Targets the TSMC7 NMOS DC L-vs-S
  inversion in SML §3.3 (consistent with on-grid Vbs overfit).

### 4.4 B4 — Drop the global AND-gate row filter

* Target: `external_compact_models/bsimar/data/dataset.py`,
  `filter_small_targets`. Replace the current "all 13 outputs above
  threshold" AND with an Id-only gate (`|id| > 1e-15 A`). The per-target
  asinh floor (`OUTPUT_LOG_FLOORS`) handles small charges/caps during
  normalisation.

### 4.5 B5 — Generate datasets, exclude ASAP7

* Update `external_compact_models/PyCMG/scripts/generate_nn_data.py`
  to accept `--version v5` and write to
  `external_compact_models/bsimar/data/datasets/universal_v5_{nmos,pmos}.npz`.
* Drop ASAP7 from the tech list at generation time. The current
  generator already supports per-tech filtering — wire it through.
* Run:
  ```bash
  conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
      --device both --universal --version v5 --exclude-techs asap7
  ```
* Verify on disk:
  ```
  external_compact_models/bsimar/data/datasets/universal_v5_nmos.npz
  external_compact_models/bsimar/data/datasets/universal_v5_pmos.npz
  ```
* Print row-count summary by `sample_class` and per-tech distribution
  for the report.

- [ ] B1.1 Implement `inv_trip` sample class in `nn_generate.py`.
- [ ] B2.1 Lower grid box factor + add `overshoot` class.
- [ ] B3.1 Add LHS Vbs jitter.
- [ ] B4.1 Replace `filter_small_targets` AND with Id-only gate.
- [ ] B5.1 Add `--version` flag to `generate_nn_data.py`; wire ASAP7 exclusion.
- [ ] B5.2 Generate `universal_v5_{nmos,pmos}.npz`.
- [ ] B5.3 Save row-count + sample-class summary to
      `results/v5_dataset_summary_<date>.md`.

---

## 5. Phase C — Small-arch DirectNet + BSIMAR loss A/B (V5)

**Branch:** same `feat/pycircuitsim-v5`. **Effort:** ~1 eng-day code +
~6–8 GPU-h training. **Risk:** medium. The Jacobian-consistency term
adds ~30–40 % wall-clock per epoch; running at small-arch keeps each
training run to ~30 min, so the whole 8-checkpoint A/B fits in a single
GPU window.
**Gate (§1.3 Phase C):** decision criterion in §1.3.

### 5.1 Architecture choice — small (S-scale per SML §2)

| Model | S-scale config | Rationale |
|---|---|---|
| DirectNet | hidden=192, layers=4 (~159 k params) | 300-epoch run fits in ~20 min on a Blackwell GPU; fast A/B iteration. |
| BSIMAR Transformer | d_model=128, nhead=4, layers=4, ff=512 (~868 k params) | 100-epoch run fits in ~30 min; large enough to actually have a Jacobian to constrain. |

We deliberately do **not** run M or L scale here — the SML report
already showed M is the price/performance sweet-spot, and the goal of
Phase C is to isolate the **loss-function** effect, not retrain the
production stack. If the Jacobian-consistency loss wins at S, the
follow-up plan retrains M (and possibly L with warm-start) under the
winner.

### 5.2 C0 — Pre-flight: autograd-vs-FD Jacobian diagnostic

* New `tests/diag_nn_jacobian_consistency.py`. For a grid of (Vgs, Vds,
  Vbs, NFIN, L) inside and just outside the training range, compare
  `torch.autograd.grad(Id, V)` against a 5-point central-FD reference.
  Flag any cell where `|FD − autograd| > 0.1·|FD|`.
* Run on the V4 production checkpoints (before training V5). The
  diagnostic confirms or refutes the model agent's structural-decoupling
  hypothesis: if autograd is already self-consistent everywhere, the
  Jacobian-consistency loss has nothing to fix and Phase C is a
  null result.

- [ ] C0.1 Build `tests/diag_nn_jacobian_consistency.py`.
- [ ] C0.2 Run on V4 checkpoints; record per-region Δ.

### 5.3 C1 — Jacobian-consistency loss term (treatment arm)

* Target: `external_compact_models/bsimar/losses/bni_mae.py`. Add an
  optional Jacobian term to `MAELoss`:
  ```
  L_jac = λ_jac · (
      MAE(autograd(id, Vgs), gm_target ) +
      MAE(autograd(id, Vds), gds_target) +
      MAE(autograd(id, Vbs), gmb_target) +
      MAE(autograd(qg, Vgs), cgg_target) +
      MAE(autograd(qg, Vds), cgd_target) +
      MAE(autograd(qg, Vbs), cgs_target) +
      MAE(autograd(qd, Vgs), cdg_target) +
      MAE(autograd(qd, Vds), cdd_target)
  )
  ```
* Hard-wired `λ_jac = 0.1` for the A/B run; revisit if C0 suggests a
  different scale.
* Wired through a CLI flag `--jacobian-consistency` (default off, so
  the control arm is the existing zero-change codepath).
* Inputs `Vgs/Vds/Vbs` need `requires_grad_(True)` per batch with
  retain-graph.

- [ ] C1.1 Implement the Jacobian term in `MAELoss`.
- [ ] C1.2 Add `--jacobian-consistency` CLI flag in `bsimar.cli.train`.

### 5.4 C2 — Train 8 small-arch checkpoints (4 control, 4 treatment)

* Datasets: V5 from Phase B
  (`universal_v5_{nmos,pmos}.npz`).
* ASAP7 excluded everywhere: `--exclude-techs asap7 --num-tech-codes 18`.
* Schedule (matches SML §2 S-scale):
  * DirectNet: 300 epochs, patience 60, batch 2048, lr 8e-4, cosine.
  * BSIMAR: 100 epochs, patience 25, batch 1024, lr 8e-4, cosine.

```bash
# Control arm — MAE+LDS only (the existing v4-re hard-wired recipe).
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type nmos \
    --dataset universal_v5_nmos.npz \
    --hidden 192 --layers 4 --epochs 300 --patience 60 --batch-size 2048 \
    --exclude-techs asap7 --num-tech-codes 18 --cuda \
    --exp-name v5_dn_s_nmos_mae

conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --device-type pmos \
    --dataset universal_v5_pmos.npz \
    --hidden 192 --layers 4 --epochs 300 --patience 60 --batch-size 2048 \
    --exclude-techs asap7 --num-tech-codes 18 --cuda \
    --exp-name v5_dn_s_pmos_mae

conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type nmos \
    --dataset universal_v5_nmos.npz \
    --d-model 128 --nhead 4 --layers 4 --ff 512 \
    --epochs 100 --patience 25 --batch-size 1024 \
    --exclude-techs asap7 --num-tech-codes 18 --cuda \
    --exp-name v5_tf_s_nmos_mae

conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model transformer --device-type pmos \
    --dataset universal_v5_pmos.npz \
    --d-model 128 --nhead 4 --layers 4 --ff 512 \
    --epochs 100 --patience 25 --batch-size 1024 \
    --exclude-techs asap7 --num-tech-codes 18 --cuda \
    --exp-name v5_tf_s_pmos_mae

# Treatment arm — same flags + --jacobian-consistency.
# Replace --exp-name suffix _mae with _jac in each line above.
```

Naming convention follows §0:

```
v5_dn_s_{nmos,pmos}_{mae,jac}_best.pt          (+ _norm.npz)
v5_tf_s_{nmos,pmos}_{mae,jac}_best.pt          (+ _best.phys.pt + _best.ar.pt
                                                 + _norm.npz + _config.npz)
```

The `_resolve_nn_checkpoint` cascade is **not** modified. We invoke
each variant explicitly in the verification script via the
`PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE` env var (already supported, see
`mosfet_directnet.py::_resolve_nn_checkpoint`).

- [ ] C2.1 Train `v5_dn_s_nmos_mae`.
- [ ] C2.2 Train `v5_dn_s_pmos_mae`.
- [ ] C2.3 Train `v5_tf_s_nmos_mae`.
- [ ] C2.4 Train `v5_tf_s_pmos_mae`.
- [ ] C2.5 Train `v5_dn_s_nmos_jac` (`--jacobian-consistency`).
- [ ] C2.6 Train `v5_dn_s_pmos_jac` (`--jacobian-consistency`).
- [ ] C2.7 Train `v5_tf_s_nmos_jac` (`--jacobian-consistency`).
- [ ] C2.8 Train `v5_tf_s_pmos_jac` (`--jacobian-consistency`).

### 5.5 C3 — Evaluation, comparison, and per-step attribution report

For each of the 8 checkpoints, run **on the post-Phase-A simulator**
(so any solver-class wins are baked in and the comparison isolates the
loss effect):

```bash
conda run -n pycircuitsim env PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE=v5_dn_s_nmos_mae \
    python tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16
# (and seven more, one per checkpoint variant)
```

Aggregate the matrices into the consolidated report:

**`results/v5_jac_loss_ab_<date>.md`** — required content:

1. **Per-step accuracy attribution table (§1.2)** — fully populated.
   Five anchor columns:
   * `V4-baseline`: V4 production checkpoints + V4 (pre-Phase-A) solver.
     Source: SML report §10.2 + a fresh re-run for MRE numbers.
   * `+solver`: V4 production checkpoints + Phase A solver. Source:
     `results/v5_phase_a_solver_<date>.md`.
   * `+data (MAE arm)`: V5 small-arch MAE-only checkpoints + Phase A
     solver. Isolates Phase B contribution.
   * `+loss (JAC arm)`: V5 small-arch JAC checkpoints + Phase A solver.
     Isolates Phase C contribution on top of Phase B.
   * Three delta columns: solver-Δ, data-Δ, loss-Δ. Each is post − pre
     for that phase, so the three columns sum to the total V4→JAC
     improvement.
2. **Per-tech NRMSE % and MRE % tables** for the four pain cells
   (TSMC5 inv-tran, TSMC7 NMOS DC, TSMC16 BSIMAR inv-tran, inverter
   VTC pass-rate), MAE column vs JAC column for each model.
3. **Per-output (id / gm / gds / gmb / charges / caps) NRMSE delta** on
   the V5 held-out test split, to surface **where** the Jacobian term
   moves the needle (likely gds in saturation, cgg/cgd at trip point).
4. **Wall-clock comparison** (training time per epoch, total epochs).
5. **Phys-NRMSE on the validation set** (BSIMAR `_best.phys.pt`).
6. **C0 FD-vs-autograd diagnostic** results, pre-train and post-train
   on each of the 8 checkpoints. Quantifies whether the JAC loss
   actually closes the autograd-vs-FD gap.
7. **Decision against §1.3 Phase C gate.** If the JAC variant wins per
   the decision criterion, recommend a follow-up M-scale retrain.

- [ ] C3.1 Run `verify_nn_dc_tran.py` for each of the 8 checkpoints,
      record both NRMSE and MRE.
- [ ] C3.2 Run the C0 FD-vs-autograd diagnostic on all 8 checkpoints
      (post-train autograd consistency).
- [ ] C3.3 Re-run V4 production checkpoints on the post-Phase-A
      simulator to fill the `V4-baseline` and `+solver` columns of
      the per-step delta table.
- [ ] C3.4 Write `results/v5_jac_loss_ab_<date>.md` with all 7 sections
      above; the per-step attribution table (§1.2) is the headline.
- [ ] C3.5 Update `MEMORY.md` with the V5 outcome (one-line index
      entry pointing at the report).
- [ ] C3.6 Tag `v5-phase-c-loss-ab`.

---

## 6. Out-of-scope / deferred

* **M-scale and L-scale V5 retrains.** Run only as a follow-up if
  the Phase C A/B says JAC wins. The S-scale A/B isolates the loss
  effect cheaply; production retrain is a separate plan.
* **Boundary-aware id head** (`id = id_raw · tanh(Vds/VT)` baked into
  the head). Structural alternative to most of rule 19; filed as a
  follow-up if Phase A + Phase C still leave VTC OVERFLOWs in
  adversarial circuits.
* **Charge-conservation training penalty** (model-P2 in the agent
  reports). Stacks with the Jacobian term but is a different proposal.
  Hold for a follow-up after the C3 report comes in — if the JAC term
  alone closes the gap, the conservation penalty is unnecessary
  complexity; if not, run it as a third arm.
* **Per-tech + 2D-region sample reweight** (model-P3). Same logic:
  hold for follow-up.
* **Per-tech asinh-scale fit, transient-trajectory replay class,
  multiple process tokens, KV-cache encoder, hybrid model-blending.**
  All filed as separate work; not load-bearing for the V5 sprint.
* **L-scale 200-epoch retrain** (SML §3.2 schedule-starvation finding).
  Filed; revisit only after V5 ships and the loss question is settled.
* **Inverter-VTC auto-bisect Vin step** (solver-P3) and **adaptive gds
  floor** (solver-P5). Compounding wins to pick up only if Phase A's
  A1+A2+A3 don't clear the §1.3 Phase A gate.
* **ASAP7 readmission.** Out of scope. Will require a vocab extension
  (codes 18–21), retrain everything, and is an entirely separate
  project.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A1 tanh ramp regresses NMOS pulse (currently 8/8 PASS) | A4 explicitly re-runs NMOS pulse; the function is gated to the OOD regime, so converged in-distribution operating points cannot trigger it. |
| A2 GMIN-on slows BSIM-CMG suites | Detection helper gates on LEVEL ≥ 73; BSIM-CMG (LEVEL=72) keeps the original codepath. |
| A3 dt-halve fallback masks a real model bug by silently sub-stepping | Log every halve event during the regression run; >1 halving on a TSMC12/TSMC16 inverter-tran cell escalates as a model-fit issue, not a solver issue. |
| Phase B `inv_trip` overlay introduces non-iid samples that confuse the loss | Tag with `sample_class` and verify per-class loss curves are well-behaved during C2. If `inv_trip` rows stall the loss, drop their LDS weight or reduce overlay budget. |
| Phase C JAC term doubles wall-clock and the win is marginal | C0 pre-flight diagnostic gates whether the term has anything to fix. If V4 autograd is already self-consistent, abort Phase C and document the null result. |
| Small-arch result doesn't generalise to M/L | Expected and stated up front: Phase C is an A/B at S, not a production retrain. Decision criterion explicitly says "recommend M-scale follow-up", not "ship S to production". |
| MRE numerator divides by tiny refs in deep cutoff | `tests/common/nn.py::mre` already uses an `ε` floor; verify `ε` is appropriate for currents (1e-12 A) and charges (1e-18 C) before the Phase A report. |
| Per-step attribution conflates concurrent improvements (e.g. solver fix that only helps because new data also lands) | The §1.2 table is constructed by anchoring each step on a single-variable change against the prior anchor (V4-baseline → +solver → +data → +loss); the four anchor runs are all evaluated on the same final simulator (post-Phase-A) so only the *checkpoint* changes between data-Δ and loss-Δ. The solver-Δ is the only one with a different simulator (pre- vs post-Phase-A on the same V4 checkpoint), and that's by definition the solver's contribution. |

---

## 8. File-touch summary (for review)

| File | Phase | Change |
|---|---|---|
| `pycircuitsim/models/mosfet_directnet.py` | A1 | Replace rule-19a quadratic with tanh/sech². |
| `pycircuitsim/simulation.py` | A2 | Add `_circuit_has_nn`; gate `use_gmin_stepping` on it. |
| `pycircuitsim/solver.py` | A3 | dt-halve fallback in transient NR loop, gated on LEVEL ≥ 73. |
| `tests/verify_nn_dc_tran.py` | A3 | Apply partial-result fallback to inverter-tran runner (SML §10.4 #2). |
| `CLAUDE.md` | A1 | Update rule 19 step (a) to tanh/sech² wording. |
| `external_compact_models/PyCMG/pycmg/nn_generate.py` | B1–B3 | New sample classes `inv_trip`, `overshoot`; lower grid box factor; LHS Vbs jitter. |
| `external_compact_models/PyCMG/scripts/generate_nn_data.py` | B5 | `--version v5` flag; wire ASAP7 exclusion. |
| `external_compact_models/bsimar/data/dataset.py` | B4 | Replace AND-gate row filter with Id-only gate. |
| `external_compact_models/bsimar/losses/bni_mae.py` | C1 | Optional Jacobian-consistency term, default off. |
| `external_compact_models/bsimar/cli/train.py` | C1 | `--jacobian-consistency` CLI flag. |
| `tests/diag_nn_jacobian_consistency.py` | C0 | New diagnostic. |
| `results/v5_phase_a_solver_<date>.md` | A4 | Phase A report (NRMSE + MRE + solver-Δ row). |
| `results/v5_dataset_summary_<date>.md` | B5 | Dataset summary. |
| `results/v5_jac_loss_ab_<date>.md` | C3 | Loss A/B comparison report (NRMSE + MRE + full per-step attribution table). |
| `MEMORY.md` | C3 | One-line V5 outcome entry. |

---

## 9. Source agents

This plan synthesises three parallel review agents launched 2026-05-07
(data/sampling, model/training, solver/inference) and follows the
user-approved sequencing: solver-fix-then-verify first, V5 dataset in
parallel, small-arch A/B comparison of the loss-function change last,
with NRMSE % **and** MRE % reported per-tech everywhere, ASAP7 excluded
throughout, and **per-step accuracy attribution mandatory in every
report** (solver / data sampling / loss).
