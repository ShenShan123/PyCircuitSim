# V6 — DirectNet inverter accuracy plan

**Date:** 2026-05-09
**Branch:** `feat/v6` (off `refactor/nn-simple`)
**Scope:** DirectNet (LEVEL=73). BSIMAR Transformer deferred. ASAP7 excluded.
**Strategy:** Try each lever in sequence with a separate commit. If the lever
is useless or harms an already-PASSing inverter cell, restore the previous
commit (`git reset --hard HEAD~1`). Final shipping set is whatever survives.

## Starting state — V4-prod DirectNet on `refactor/nn-simple`

| Suite | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|
| 1-dev DC NRMSE % | 0.98 | **3.22** | 0.18 | 0.19 |
| Inverter VTC | **NR_FAIL** | **NR_FAIL** | 9.56 PASS | 9.42 PASS |
| Inverter Tran | **16.90 FAIL** | 9.68 PASS | 3.98 PASS | 9.06 PASS |

Goal: lift inverter VTC from 2/4 PASS and inverter transient from 3/4 PASS.

## Levers (in execution order)

### Tier 1A — Trust-region NR clamp  *(merge from `feat/v5-prime` `a3719c9`)*
Cap per-iteration `|ΔV| ≤ max_vs_voltage` for NN circuits (LEVEL ≥ 73) in both
`DCSolver` and `TransientSolver`. Replaces the rejected per-tech λ-floor lever
because the failure mode is NR runaway, not Jacobian shape. BSIM-CMG path
unchanged.

- **Verification (fast):** L1 BSIM-CMG (`verify_bsimcmg_op.py + _dc.py + _tran.py`)
  must remain byte-identical. NN inverter VTC at TSMC12 must remain at 9.56 %
  ± 0.05 pp.
- **Verification (full):** `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16`
  after Tier 1B is layered on top. No PASS-cell regression on the 6 currently
  passing inverter cells.
- **Revert if:** any TSMC12/16 inverter cell regresses or BSIM-CMG L1 drifts.

### Tier 1.5 — verify driver env-var override  *(merge subset of `dd7816a`)*
Patch only `tests/verify_nn_dc_tran.py` so it honors
`PYCIRCUITSIM_NN_CHECKPOINT_{DN,TF}_{NMOS,PMOS}`. Required for the Tier 2
A/B probe to compare new-norm vs old-norm checkpoints without renaming files.
The accuracy report and V5' DN checkpoints from `dd7816a` are NOT merged.

- **Verification:** spot-run a default verify call on TSMC12 — checkpoint
  resolution must still pick `v4_re_dn_universal_*` when no env var is set.

### Tier 1B — Restore GMIN homotopy level 1e-10
Phase A trimmed the GMIN retry schedule from 4 levels to `[1e-8, 1e-12]` for
wall-time, not accuracy. Restore one bridging level 1e-10. Only fires on the
slow-path retry, so PASS cells cannot regress in correctness — only wall-time.

- **Verification:** `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16`.
  Acceptance: ≥ 1 NR_FAIL → PASS conversion on TSMC5 or TSMC7 inverter VTC,
  AND wall-time regression < 2× on the slow path, AND no PASS-cell regression.
- **Revert if:** no NR_FAIL converts AND wall-time grows > 2×, OR any
  PASS-cell regresses.

### Tier 2 — DirectNet output asinh-zscore  *(retrain)*
Flip `_ADAPTERS["direct"].norm_mode` from `"zscore"` to `"asinh"` in
`external_compact_models/bsimar/training/trainer.py`. The chain rule is
already correct in `AsinhNormalizer.denormalize_derivative` (the Transformer
uses it).

Two-stage retrain to keep wall-time honest:

1. **E2-small probe** — 4-output head, 57 K params, ~4 min on one A100.
   `python -m bsimar.cli.train --model direct --size small --loss-preset e2 \
     --device-type nmos --exclude-techs asap7 --num-tech-codes 18 --cuda \
     --exp-name v6_dn_small_e2_asinh --overwrite`
   Repeat for `--device-type pmos`. Run `verify_nn_dc.py --tech TSMC12` with
   the env-var override pointing at the asinh checkpoints.
2. **Production retrain** (only if probe non-regressing): B0-large on the
   asinh recipe.
   `python -m bsimar.cli.train --model direct --size large --device-type {n,p}mos \
     --exclude-techs asap7 --num-tech-codes 18 --cuda \
     --exp-name v6_dn_large_asinh --overwrite`
   Then `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16` with env-var
   override.
- **Acceptance:** TSMC7 single-device DC ≤ 2.0 % (currently 3.22 %); no
  regression > 0.3 pp on TSMC5/12/16; no new NR_FAIL. ≥ 1 inverter-VTC
  cell converts NR_FAIL → PASS.
- **Revert if:** probe shows TSMC12 inverter VTC regresses, or
  production retrain regresses any of the 6 currently-PASS inverter cells.

### Tier 3 (optional) — B0-large capacity probe on the existing zscore recipe
Independent of Tier 2. The trim plan §Follow-ups names this as the next
untaken data point on the post-refactor pipeline.
`python -m bsimar.cli.train --model direct --size large --device-type {n,p}mos \
  --exclude-techs asap7 --num-tech-codes 18 --cuda \
  --exp-name v6_dn_large_zscore --overwrite`
Run only if Tier 1 + Tier 2 leaves residual TSMC7 VTC issues.

## Explicitly NOT in V6
- Per-tech λ gds floor (rejected — wrong lever for NR_FAIL, λ values handwavy).
- Trip-point importance-weighted resampling (rejected — same effective gradient
  shift as V5 `inv_trip` overlay, which regressed TSMC12/16 inv-tran).
- V5' dataset code (commit `2d52972`) and PyCMG submodule bump.
- V5' DN checkpoints (`v5p_dn_m_*`).
- AR finetune, JAC consistency loss, charge-consistency penalty, sign loss,
  boundary loss, id-gate, slope-match — all documented dead-ends.

## Per-step protocol
1. `git commit -am "..."` before any code change for the step.
2. Apply the change.
3. Commit with message `feat(v6): <step> — <name>`.
4. Run the step's verification.
5. If acceptance fails, `git reset --hard HEAD~1` and document the failure
   in this plan's "Outcomes" section below. Move on.
6. If acceptance passes, leave the commit, update Outcomes, move on.

## Outcomes

(Filled in as steps complete.)

| Step | Commit | Verify | Verdict |
|---|---|---|---|
| Plan | (this commit) | n/a | n/a |
| Tier 1A | | | |
| Tier 1.5 | | | |
| Tier 1B | | | |
| Tier 2 probe | | | |
| Tier 2 prod | | | |
| Tier 3 (opt) | | | |
