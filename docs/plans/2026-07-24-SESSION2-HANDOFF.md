# V6.13.0 campaign + audit fix waves — SESSION 2 HANDOFF

**Written 2026-07-24 23:31 mid-flight; campaign section updated 2026-07-25 02:15
when the gates finished.** This is the resume point. Read this file first, then
`docs/plans/2026-07-24-a3-regate-status.md` (campaign background) and
`docs/plans/2026-07-24-audit-fix-waves.md` (bug-fix triage).

**STATUS 2026-07-25 02:40 — V6.13.0 AND WAVE 1 ARE PUSHED TO `main`**
(`a96112a..9fd0301`). The campaign is finished (§1) and the docs are committed.
What remains is `audit-fixes-wave2`, which is committed on its branch and
**unmerged pending a re-gate** (§6), plus the deferred items in §7.

Verified on merged `main` before pushing: `verify_bsimcmg_op` 3/3,
`verify_bsimcmg_dc` 2/2, `verify_bsimcmg_tran` 1/1, `verify_ac` 2/2,
`verify_subckt` 11/11.

---

## 1. CAMPAIGN — ✅ COMPLETE (2026-07-25 01:00:38)

`ALL POOLS DONE` in `results/a3_regate/_resume_main.log`. Matrix **96/96** cells
(28/28 groups, 448 cells total), all **45/45** OMP strict runs, `nn_ac_tf` 8/8.
Nothing is running any more; the snapshot at `/data2/shenshan/a3_gate_snap` and
the scratch at `/data2/shenshan/a3_gate_scratch` can be deleted once the results
are committed.

Every V6.13.0 number is measured at **`d2ea720`** (frozen snapshot). Do not
re-run a campaign cell from the live repo.

### Two collection bugs hit and fixed during the wrap-up
- `scripts/a3_regate_collect.py` had `--out` defaulting to `None`, so a bare run
  printed a fresh report while leaving a **stale REPORT.md** on disk — which I
  read once as if it were the new result. Now defaults to `<root>/REPORT.md`
  (matching its own docstring and `a3_regate_omp_collect.py`). *Reading a report
  file after running its generator is not proof the generator wrote it.*
- The orchestrator's OMP sidecar fold globbed `tf_*.log`, so cells whose log did
  not previously exist (the `tf_corro15_xl` TSMC12/16 pairs) kept an unfolded
  `.log.omp<n>`. Folded by hand; all 80 logs are now complete, 0 sidecars left.
  `a3_regate_omp_collect.py` reads sidecars directly, so no result was lost.

## 2. COLLECTORS

```bash
python scripts/a3_regate_collect.py       # single-run matrix -> results/a3_regate/REPORT.md
python scripts/a3_regate_omp_collect.py   # STRICT/OMP       -> results/a3_regate/OMP_REPORT.md
python scripts/a3_regate_uni_collect.py   # universal (already final)
```
Both new scripts are **untracked** and must go into the V6.13.0 commit, along
with `scripts/a3_omp_one.sh`. `a3_regate_omp_collect.py` reads un-folded
sidecars, so it works mid-run.

---

## 3. FINAL RESULTS

Strict = all of OMP∈{1,2,4} pass. **Zero FLIPs in all 10 swept groups.**

| strict 16/16 | strict 15/16 | strict 14/16 | below |
|---|---|---|---|
| `dn_crit15m_xl`, `tf_corro15_xl`, `tf_corroft_medium` | `dn_clean_large` (production), `dn_corroft_xl` | `dn_crit10_xl`, `tf_clean_large` | `dn_v660clean_large` 13, `dn_clean_xl` 12, `pfn_clean_small` 11 |

Single-run matrix, **387/448 cells across 28 groups** (`results/a3_regate/REPORT.md`).
Six groups at 16/16: `dn/crit15m/xl`, `tf/corro15/{medium,xl}`,
`tf/corroft/{medium,xl}`, `tf/crit15m/xl`, `tf/crit30/xl` — that is seven counting
DirectNet's. Notable deltas: `dn/clean/small` **+3**, `dn/clean/xl` +2,
`dn/crit15m/xl` +2, `tf/clean/small` +2, `tf/crit30/xl` +2; one regression,
`dn/csob/large` **−1**.

**Every corridor recipe at `xl` now sweeps the matrix** (corroft, crit15m,
crit30, corro15 all 16/16; pre-fix 15/15/14/15), and all four sit at 15/16 at
`large` missing only `tsmc7-opamp`. The corridor's effect is uniform, not
recipe-specific — the opposite of the pre-fix reading.

BSIM-AR clean is **14/16 at every tier** (was 12/14/13/13) failing only
tsmc5-ring + tsmc7-ring, deterministically (7.38 % / 8.63 % identical at all
three thread counts). PFN 11/11/9 (was 11/10/8). Universal DN net **+3**, all 3
pre-existing FLIPs gone.

Suites: device AC DN **8/8**, PFN **8/8**, TF 8/8; `nn_dc_tran` 24/24;
lifted-source 12/12; parametric tran 64/64; parametric DC 54/55 with the one
failure **bit-identical** pre/post (DC exactly invariant at scale).
`complex_opamp_ac` 0/4 (OP un-rails; TSMC5 closest, 3.31 dB / PM 18.9°). L72
controls pass; `verify_multi_tech_dc` 43 PASS + 1 pre-existing ERROR, still
exits 0 (audit B3).

**Structural finding:** every cell the fix gained — across DirectNet's 4 sizes,
PFN's 3 and BSIM-AR's 4 clean tiers — is an **opamp**. Not one ring, SRAM or
switchcap cell moved anywhere.

## 4. DOCUMENTATION — DONE (uncommitted on `main`)

Modified and complete: `docs/accuracy/{DirectNet-L73,BSIM-AR-L74,PFN-L75,README}.md`,
`CLAUDE.md`, `README.md`, `docs/2026-07-21-systematic-audit.md` (disposition
header), `docs/plans/2026-07-24-a3-regate-status.md`.
New untracked: `docs/plans/2026-07-24-audit-fix-waves.md`.

Sections written: DN §1, §3.1b (new), §5 note, §6.1b (new), §6.2 retraction,
§8 note, §9.1b (new), §11 (rewritten), §12.1, §12.2 (rewritten), §12.4;
TF §1 (rewritten), §3.1 (new), §5.1 (new), §9 (rewritten); PFN §0, §1, §5, §7, §10.

Four claims **retracted** across all docs — verify they stay retracted:
1. "tsmc7-opamp is the universal ceiling, reachable only by the V6.5.9 T3 solver
   fine-tune" — false; two families sweep 16/16 with ordinary data recipes.
2. "PFN is the only flip-free family" — overtaken; all families are flip-free.
3. "BSIM-AR beats DirectNet by one cell" — both 16/16; they differ by 40× speed.
4. BSIM-AR "capacity peaks at medium" and "strict best is medium, not large" —
   the first was mostly the bug; the second rested on a FLIP that no longer exists.

Also fixed: `CLAUDE.md`/`README.md` carried pre-TSMC6-retire L72 suite counts
(81/53/45/86 → **67/44/37/72**).

### Still to write
- **`docs/CHANGELOG.md`** — two entries, drafted in full, ready to paste. Copied
  out of the session scratchpad into **`results/a3_regate/_handoff/`** (gitignored
  but persistent):
  - `changelog_v6130_draft.md` — V6.13.0 campaign. Has one `<!-- PENDING -->`
    marker for the BSIM-AR corridor recipes; **grep for it before committing**.
  - `changelog_v6131_draft.md` — V6.13.1 wave 1, complete.
  - Same directory also holds `findings.json` (all 43 verified audit findings
    with evidence, prescribed fix, risk class and test — the input to both fix
    waves) and `wave{1,2}_review.json` (the adversarial diff reviews).
  - Both entries are also reconstructible from this file plus the two branch
    commit messages, which are deliberately verbose.
- ✅ All four accuracy reports reconciled against the FINAL data (2026-07-25 02:2x):
  TF §1, §4.1 (new), §6.1 (new), Appendix A pre-fix marker; DN Part II pre-fix
  marker; DN alternates table corrected — **`csob@large`'s complex-gate rationale
  and `crit10@xl` are both WITHDRAWN**, they now fail the very cell they were
  documented to cover; PFN §5 cross-references refreshed.
- ✅ TF §1 updated with the final numbers (six 16/16 groups, the uniform xl
  corridor result). DN §11 / TF §9 / accuracy README were written assuming
  `tf_corro15_xl` would land at 16/16 — **it did, strict, zero flips**, so those
  sections are correct as written. Worth one last read-through anyway.
- Memories: `v6130-a3-fix-regate-campaign` needs the final numbers;
  `nn-gds-sign-bug-open` is already updated to RESOLVED;
  `v680-bsimar-transformer-15of16-strict` and
  `v659-t3-solver-lands-opamp-16of16` still assert tsmc7-opamp is unreachable
  and must be corrected.

---

## 5. BUG FIXES — committed on two branches, NOT merged

`/data2/shenshan/pcs-fixes` is a git worktree holding both branches. `tools/`
and `checkpoints/` in it are local symlinks; never `git add -A` there.

| branch | commit | content |
|---|---|---|
| `audit-fixes` | `698b101` | **wave 1** — 22 gate-neutral findings, 36 files |
| `audit-fixes-wave2` | `f118ce3` | **wave 2** — 18 gate-affecting findings, UNVALIDATED |

Both commit messages are long and carry the full per-finding evidence. Read them
rather than re-deriving.

Verified on both trees (the only gates that could be spared while the campaign
ran): `verify_bsimcmg_op` 3/3, `verify_bsimcmg_dc` 2/2, `verify_ac` 2/2,
`verify_subckt` **11/11**, with L72 numbers byte-identical to baseline
(`max|dV| = 0.000e+00`, inverter NRMSE 0.187 %, buffer 0.638 % / 0.861 %).

### Merge order (important)
1. Commit **V6.13.0** on `main` first — docs + the 3 new scripts. The campaign's
   numbers must sit at a commit whose harness is the one that produced them.
2. Then merge `audit-fixes`. It moves `verify_subckt` 8/8 → **11/11** (Level 0
   adds 3 tests for the V6.12.0 loud errors, which had none). Five doc sites
   still say 8: `CLAUDE.md:139`, `CLAUDE.md:306`, `README.md:718`,
   `README.md:790`, `README.md:827`. Update them **in the merge commit**.
3. `audit-fixes-wave2` stays unmerged until it is re-gated (§6).

### Three things future-me will want to know
- **C6t must NOT be "fixed".** Replacing `abs(g_ds)` in `mosfet_cmg.py` with the
  Rule 4 clamp diverges the plain L72 inverter (`verify_bsimcmg_op` 3/3 → crash,
  NR to d ≈ 9e8 V). The negative candidates BSIM-CMG emits are **9.3e-04 S** —
  physically sized with a wrong sign — so reflection recovers the magnitude,
  while `|id|/50 V` lands orders of magnitude low and opens the drain diagonal.
  Rule 4 is right for the NN families and does not transfer to OSDI. This is
  written into the code as a "do not re-apply" comment; leave it there.
- **C2 dropped.** The softplus-clamp NaN cannot fire (no autograd graph) and the
  audit's prescribed `F.softplus` rewrite is *not* bit-identical (23/401 samples
  differ in the last fp32 bit).
- **B5k and C6p** were already closed by the TSMC6 retire.

---

## 6. WHAT WAVE 2 OBLIGES

Wave 2 changes solver numerics and has **never been through a complex re-gate**.
Before merging:
- Run the full campaign shape (28 groups is overkill; production DirectNet stems
  + the device/parametric/L72 suites is the minimum) against `audit-fixes-wave2`.
- Ordering constraints already encoded in the code and the commit: B2 landed
  before B1; B2's threshold is **not** flipped — `_RESID_ABS_FLOOR` stays at
  1e-6 A with `PYCIRCUITSIM_RESID_DEBUG=1` printing the new/legacy pair and
  `PYCIRCUITSIM_RESID_FLOOR` allowing a sweep. Calibrate the floor from a
  passing gate's distribution before letting it bind.
- B1's NR-budget half (final source step gets the remaining iterations instead
  of `50//20 = 2`) is deliberately NOT implemented. Land it separately if the
  re-gate shows a jump in "DC fast-path did not converge; retrying with GMIN
  stepping" lines.
- Each wave-2 agent recorded a **prediction** of which gates should move and
  which way; they are in the workflow output and are the thing to check the
  re-gate against.
- **Known dead branch:** the DC oscillation-average block needs
  `len(voltage_history) >= 3` but the per-source-step budget is 2 iterations, so
  it is unreachable on the default path. C6d's DC half cannot be exercised
  without `use_source_stepping=False`. Do not read "zero samples" as "L72 never
  oscillates".

---

## 7. DEFERRED, WITH REASONS

- `scripts/gate_matrix_iso.sh` — the 12th B3 dispatcher plus all of **B5f**. Left
  byte-identical because it is *driving* the campaign. Apply once the pools end.
  (B5f's stale-`.cell_` arm did not affect V6.13.0: every `.cell_*` in
  `results/a3_regate/` was checked to post-date the gds fix commit.)
- `tests/verify_nn_ac.py` prints the DirectNet banner and a `DN=` column even
  under `FORCE_LEVEL=74/75`, so `nn_ac_tf.log` / `nn_ac_pfn.log` read as
  DirectNet results. Cosmetic, misleading, unfixed.
- **C6n retrain.** LEVEL=74 selects on teacher-forced loss while deployment is
  free-running AR (audit measured `gds` 33 % worse under AR). Wave 2 adds opt-in
  `--val-mode ar`, which changes nothing for existing weights. Realizing it means
  retraining the BSIM-AR family — a campaign of its own, and it should run
  against the post-wave-2 baseline so the effects are not confounded.
- No other retraining is pending. All 36 checkpoint sets on disk (28 per-tech +
  8 universal) already existed; the gds fix is inference-side, so V6.13.0 is
  purely re-evaluation.

---

## 8. GOTCHAS RE-CONFIRMED THIS SESSION

- Never drive these gates through Agent/workflow subagents — the agent's Bash
  call ends and orphans them. Launch as detached `setsid nohup` background bash.
  (Workflows were used all day for *analysis and code edits*, which is fine.)
- Concurrent appends to one OMP log interleave; `a3_omp_one.sh` writes a
  per-run sidecar and the orchestrator folds them in order.
- Cluster load was ~1100–1400 from other users all session (~160 of 192 cores
  busy); our jobs held ~44 cores.
