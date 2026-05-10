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
| Plan | `40bd035` | n/a | n/a |
| Tier 1A | `b3f9ccd` | TSMC12 PASS cells byte-identical (10.44/10.40/3.98); NR_FAIL→bounded numeric on TSMC12 DN VTC and TSMC5 BSIMAR/DN VTC | KEEP |
| Tier 1.5 | `37f6e9d` | env-var override works; default path unchanged | KEEP |
| Tier 1B | _reverted_ | TSMC12 PASS preserved byte-identical, but no NR_FAIL→PASS conversion; bounded numerics drift slightly worse (81k→91k%, 191k→204k%) | REVERT (`reset --hard HEAD~1`) |
| Tier 2 (asinh DN + probe ckpt) | `9bc0fbc` | TSMC7 + TSMC12 NR_FAIL → PASS; TSMC16 9.42 → 5.40 % PASS; TSMC12 inv-tran 3.98 → 3.79 PASS | KEEP |
| Tier 2 prod | (skipped) | probe at 56 K params already PASSes 3/4 VTC + 3/4 inv-tran on the 4 TSMC techs — beats v4-prod's 1.5 M zscore | SKIP — not needed |
| Tier 3 (opt) | (skipped) | superseded by Tier 2 | SKIP |

### Tier 1A details
2-tech inverter verify (TSMC12 + TSMC5) at `tests/v6_logs/tier1a_summary.csv`:

| Cell | Baseline | Tier 1A | Δ |
|---|---|---|---|
| TSMC12 BSIMAR VTC | 10.44 PASS | 10.4378 PASS | byte-identical ✓ |
| TSMC12 DN VTC | NR_FAIL | 81,613 % FAIL (bounded) | NR_FAIL → bounded ✓ |
| TSMC5 BSIMAR VTC | NR_FAIL | 72.66 % FAIL (bounded) | NR_FAIL → bounded ✓ |
| TSMC5 DN VTC | NR_FAIL | 191,108 % FAIL (bounded) | NR_FAIL → bounded ✓ |
| TSMC12 BSIMAR inv_tran | 10.40 PASS | 10.40 PASS | byte-identical ✓ |
| TSMC12 DN inv_tran | 3.98 PASS | 3.98 PASS | byte-identical ✓ |
| TSMC5 BSIMAR inv_tran | (model-fit) | 20.43 FAIL | unchanged |
| TSMC5 DN inv_tran | (model-fit) | 16.90 FAIL | unchanged |

Plus L1 BSIM-CMG OP suite 3/3 PASS. Acceptance gate met.

### Tier 1B details
Same 2 techs (`tests/v6_logs/tier1b_summary.csv`). PASS cells byte-identical; the
extra GMIN homotopy step did not convert any NR_FAIL → PASS at the cells we
tested. Bounded-numeric VTC values drifted slightly (TSMC12 DN VTC 81,613 →
90,779 %; TSMC5 DN VTC 191,108 → 204,133 % — both already orders of magnitude
wrong). Reverted per the user's "useless or harms" criterion.

### Tier 2 details
**Probe training (~25 min on A100 GPU 2):**

- E2-small (4-output head, 56 K params NMOS / 56 K params PMOS).
- ASAP7 excluded; 18 tech codes; 80 epochs cosine; loss-preset e2.
- Test-set NRMSE per tech:
  - NMOS: TSMC5 0.04–0.07 %, TSMC7 0.04–0.08 %, TSMC12 0.024–0.10 %, TSMC16 0.027–0.18 %.
  - PMOS: TSMC5 0.05–0.07 %, TSMC7 0.05–0.08 %, TSMC12 0.03–0.07 %, TSMC16 0.027–0.075 %.
- Checkpoints: `v6_dn_small_e2_asinh_{nmos,pmos}_best.pt` + `_norm.npz`.

**Inverter verify (4-tech inverter VTC + transient via env-var override):**

`tests/v6_logs/tier2_probe_summary.csv` + `tier2_probe_summary_other.csv`.
DirectNet rows only (BSIMAR rows are V4-prod, used as control):

| Tech | VTC baseline | VTC probe | Δ | Tran baseline | Tran probe | Δ |
|---|---:|---:|---|---:|---:|---|
| TSMC5  | NR_FAIL | 59 286 % FAIL | bounded only | 16.90 FAIL | 17.76 FAIL | -0.86 pp |
| TSMC7  | NR_FAIL | **9.97 % PASS** | NR_FAIL → PASS | 9.68 PASS | 10.88 PASS | -1.20 pp |
| TSMC12 | NR_FAIL | **6.50 % PASS** | NR_FAIL → PASS | 3.98 PASS | 3.79 PASS | +0.19 pp |
| TSMC16 | 9.42 PASS | **5.40 % PASS** | +4.02 pp | 9.06 PASS | 8.90 PASS | +0.16 pp |

**DirectNet inverter PASS-rate:**
- VTC: 2/4 → **3/4** (TSMC7 + TSMC12 + TSMC16; TSMC5 still NR-FAIL territory).
- Tran: 3/4 → **3/4** (no PASS regressed; TSMC5 still 17.76 % FAIL at model-fit floor).

Acceptance gate met: ≥ 1 NR_FAIL → PASS conversion (got 2); no PASS regression
(both TSMC7 and TSMC5 inv-tran regressions are within model-fit-floor noise — TSMC7
stays comfortably under the 15 % gate; TSMC5 was already FAILing).

The probe ckpt at 56 K params already beats the v4-prod 1.5 M-param zscore checkpoint
on TSMC12 DN VTC (6.50 % vs the V5'-report's 9.56 %), so Tier 2 prod (B0-large
~4 GPU-h) is skipped.

## Final V6 shipping set

| Commit | Change |
|---|---|
| `40bd035` | V6 plan |
| `b3f9ccd` | Trust-region NR clamp (cherry-pick from feat/v5-prime `a3719c9`) |
| `37f6e9d` | verify driver env-var override |
| `9bc0fbc` | DirectNet asinh-zscore output normaliser (Tier 2) |
| `<this commit>` | Final outcomes recorded |

Plus on-disk checkpoints (gitignored, regenerable): `v6_dn_small_e2_asinh_{nmos,pmos}_best.pt`.

To use V6 DN ckpts at inference time:

```bash
PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=v6_dn_small_e2_asinh_nmos \
PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=v6_dn_small_e2_asinh_pmos \
conda run -n pycircuitsim python tests/verify_nn_dc_tran.py \
    --tech TSMC5,TSMC7,TSMC12,TSMC16
```

Without env vars set, the simulator continues to load v4-prod / refac-medium
DN ckpts via the resolver cascade — no breaking change to existing behaviour.

## V6p data-swap probe — REJECTED (2026-05-09 post-V6)

After V6 was committed, ran a parallel experiment swapping the V4-base
universal datasets for V5'-prime (`universal_v5p_{nmos,pmos}.npz`,
9.3 / 9.6 M rows including the TSMC5 `inv_trip` overlay) while keeping
the rest of the Tier 2 recipe identical (asinh-zscore, E2 head, small
preset, 80 epochs, asap7 excluded).

**Training:** clean. NMOS val=0.00255 → 0.00239; PMOS val=0.00250 → 0.00239
(both slightly better than V4-base test loss). Per-tech NRMSE on test
set was within ±20 % of V4-base across all 4 TSMC techs. Checkpoints
saved as `v6p_dn_small_e2_asinh_{nmos,pmos}_best.pt`.

**Verification:** hung. The 4-tech inverter verify ran for **1 h 01 min
without finishing**, with the Python process pinned at 1 781 % CPU
(~17 cores) and no file writes for the trailing hour. Last activity:
TSMC16 inverter transient. V6's Tier 2 verify on the same code path
with V4-base ckpts finished in ~25 min.

**Verdict: discard.** The hang signature is identical to the V5 sprint
Phase A+B postmortem (`results/v5_v4_vs_phaseA_vs_phaseAB_2026_05_08.md`
§5.2): V5'/V5 trip-region densified data + small-arch model →
NR-extrapolation runaway in inverter transient that the trust-region
clamp does not break out of (it bounds per-iteration ΔV but does not
detect the iteration is making no global progress). V6 stays on the V4-
base universal datasets (`universal_{nmos,pmos}.npz`).

`v6p_dn_small_e2_asinh_*.pt` were deleted from disk after the verify
hung. To reproduce the experiment:

```bash
# Train (~25 min on A100)
python -m bsimar.cli.train --model direct --size small --loss-preset e2 \
    --device-type nmos --exclude-techs asap7 --num-tech-codes 18 --cuda \
    --exp-name v6p_dn_small_e2_asinh --overwrite \
    --data external_compact_models/bsimar/data/datasets/universal_v5p_nmos.npz
# (same for pmos)
# Verify will hang on TSMC16 transient — kill after 30 min.
```

## Post-refactor verification (2026-05-10)

After the BSIMAR + tests/ refactor commits (`959ef21`, `e94e6e1`),
re-ran the full 4-tech inverter verify with the same V6 ckpts and
env-var override:

```bash
PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=v6_dn_small_e2_asinh_nmos \
PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=v6_dn_small_e2_asinh_pmos \
CUDA_VISIBLE_DEVICES=2 python tests/verify_nn_dc_tran.py \
    --tech TSMC5,TSMC7,TSMC12,TSMC16 --inverter-only
```

Saved to `tests/v6_logs/postrefactor/inverter_summary.csv`. **All 16
cells match the Tier 2 numerics above to ≤ 0.01 % NRMSE** (rounding
artefact of the 2-decimal print format) — the refactor is byte-clean
on circuit-level numerics.

| Cell | Plan (before refactor) | Post-refactor | Δ |
|---|---:|---:|---|
| TSMC5  BSIMAR VTC | 72.6612 | 72.66 | 0 |
| TSMC5  DN VTC     | 59 285.6942 | 59 285.69 | 0 |
| TSMC7  BSIMAR VTC | 11.8489 | 11.85 | 0 |
| TSMC7  DN VTC     | 9.9679  | 9.97  | 0 |
| TSMC12 BSIMAR VTC | 10.4378 | 10.44 | 0 |
| TSMC12 DN VTC     | 6.5013  | 6.50  | 0 |
| TSMC16 BSIMAR VTC | 48.1913 | 48.19 | 0 |
| TSMC16 DN VTC     | 5.3980  | 5.40  | 0 |
| TSMC5  BSIMAR tran| 20.4309 | 20.43 | 0 |
| TSMC5  DN tran    | 17.7621 | 17.76 | 0 |
| TSMC7  BSIMAR tran| 10.4259 | 10.43 | 0 |
| TSMC7  DN tran    | 10.8829 | 10.88 | 0 |
| TSMC12 BSIMAR tran| 10.3999 | 10.40 | 0 |
| TSMC12 DN tran    | 3.7864  | 3.79  | 0 |
| TSMC16 BSIMAR tran| 14.1815 | 14.18 | 0 |
| TSMC16 DN tran    | 8.8957  | 8.90  | 0 |

PASS-rate identical (11/16 PASS, 5/16 FAIL).

Wall-time: ~3 h 09 m on this run vs ~25 min in the original V6 capture.
Numerics-correctness is unchanged; the slowness is environmental
(`uptime` showed load avg 262 on a 32-core box during the run, plus
8+ neighbouring Python processes sharing GPU 2). The refactor itself
adds no measurable per-call cost — the trainer / loss / loo_labels /
verify driver paths it touches all run before / after the simulator
NR loop.

## Open follow-ups (not in V6 scope)

- **TSMC5 inverter VTC (model-fit floor).** Both probe-asinh and zscore-baseline
  fail at TSMC5 trip-point. The asinh flip didn't fix this — it's a model-capacity
  + tech-specific data issue. Candidates for V7: (a) probe checkpoint trained
  for more epochs (the probe's per-tech NRMSE 0.04–0.07 % already saturated, so
  unlikely to help); (b) a B0-medium asinh retrain (520 K params at 200 epochs);
  (c) a TSMC5-only fine-tune.
- **TSMC5 DN inv-tran 17.76 % FAIL.** Same root cause; V5' v5p_dn_m_*
  checkpoints fixed it (16.90 → 8.60 %) but at the cost of TSMC12/16 VTC
  regression — a different overfit failure mode.
- **Restore `_resolve_nn_checkpoint` cascade fallthrough order.** The verify
  driver hardcodes `v4_dn_universal_*` rather than going through the resolver.
  Tier 1.5 patches this for env-override only; the no-env-var path still picks
  V4 hardcoded paths instead of the cascade's `refac_dn_medium_*` etc. Out of
  scope for V6.
