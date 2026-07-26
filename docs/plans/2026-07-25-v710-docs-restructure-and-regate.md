# V7.1.0 — accuracy-doc restructure, pre-fix re-gate, TSMC6 restore, PFN xl

**Status: IN FLIGHT (started 2026-07-25 19:36).** Docs and code are committed
(`cb508f7`, `73b7a00`, PyCMG `06a20b7`); three compute campaigns are running.
This file is the resume point — update it on every change (workflow rule).

---

## 1. What this version is

Four threads, in the order they were started:

| # | thread | state |
|---|---|---|
| A | **Restructure `docs/accuracy/`** into tech / scale / recipe pivots + family reports + methodology + archive | ✅ committed `cb508f7` |
| B | **Re-measure every accuracy number still standing on pre-gds-fix code** | 🔄 running (`results/v710_regate/`) |
| C | **Restore TSMC6** as a deliberate duplicate-of-TSMC7 repeat, all 3 families × 4 scales | 🔄 data regenerating, training queued |
| D | **Give PFN an `xl` tier** so all three families have 4 scales | 🔄 8 checkpoints training |

## 2. Thread B — the re-gate

**Why.** V6.13.0 re-gated the complex 16-cell matrix for all 28 on-disk
checkpoint groups, but ran the *device* suites (`verify_nn_ac`,
`verify_complex_opamp_ac`, `verify_nn_multi_tech_{dc,tran}`) only for each
family's **resolver-default** stem. Every per-size and per-recipe AC number in
`docs/accuracy/` was therefore still a pre-fix measurement — and AC is the axis
the gds fix moved most (device AC 8/10 → 10/10 on the audit arm, 4/12 → 8/8 on
the default stem). It also strict-swept OMP for only 10 of the 28 groups.

**Driver.** `scripts/v710_regate.sh` (one job = tag / variant / tech / suite /
omp, resumable, isolated, CPU-pinned), job lists from
`scripts/v710_regate_jobs.py`, collected by `scripts/v710_regate_collect.py`
into `results/v710_regate/REPORT.md` + `data.json`.

**Control.** `scripts/v710_regate_control.py` compares every complex cell
measured in both campaigns. At the last check: **150/150 agree** — HEAD (V7.0.x
perf work + audit wave 1) reproduces the V6.13.0 (`d2ea720`) verdicts, so the
two campaigns' numbers are interchangeable. Re-run it at the end and record the
final count.

It has already earned its keep: it caught **one** disagreement,
`dn/corroft/xl` TSMC12 `sram_snm`, which turned out to be
`KeyError('tsmc6')` — a job that ran inside the few-minute window when the
parent repo had TSMC6 back in its registry but the PyCMG submodule did not.
Log deleted, cell re-run, PASS, control clean. **The lesson is not "additive
edits are safe" but "a two-repo edit has a window, and only a control finds
what fell into it."**

**Results so far** (all in `docs/accuracy/by-scale.md` §4–§5):

* **"AC peaks at SMALL" is retracted.** DirectNet device CS-amp AC is
  7/8 · 8/8 · 8/8 · 7/8 across small→xl. Pre-fix: 5/12 · 4/12 · 4/12 · 4/12.
* The two surviving misses are the documented classes — `small` TSMC7-NMOS on
  *gain* (2.03 dB), `xl` TSMC5-NMOS on *pole placement* (f3db ratio 2.51).
* **The production curriculum improved the charge surface**: `v660clean@large`
  fails TSMC5-NMOS AC (f3db 1.78) where `crit30f` in the same slot is 8/8.
* **"Opamp open-loop AC is 0/4 everywhere for every family" is false** —
  DirectNet `small`, BSIM-AR `small`/`medium` bank TSMC16; BSIM-AR `large`
  banks TSMC7.
* **A gate-construction defect** (see §5).

**Still running when this file was written:** the DirectNet pool (~353/480), the
BSIM-AR device pool, the PFN pools, and the BSIM-AR strict-OMP pool (4/96 — the
multi-hour AR opamp/ring cells; it will not finish soon and its coverage should
be reported as partial).

## 3. Thread C — TSMC6

**The finding is unchanged**: TSMC6 is TSMC7 relabelled under BSIM-CMG
(`docs/accuracy/methodology.md` §7). It is carried again because a bit-identical
duplicate is the only instrument that measures this pipeline's run-to-run
variance with the data held fixed — and the first repeat already showed a
68.2 % vs 2.0 % SRAM disagreement collapsing to 5.2 % vs 6.2 % after the gds
fix, i.e. most of the "training lottery" was the wrong Jacobian.

* Registry restored on both sides (parent `73b7a00`, PyCMG `06a20b7`).
  `assert_tech_is_distinct()` kept; `tsmc6`↔`tsmc7` is the sole entry in
  `ACKNOWLEDGED_DUPLICATE_TECHS` → warns, does not raise.
* **Scoring rule: TSMC6 is a /4 column of its own, never inside the /16.**
* Datasets regenerating from the kept vendor PDK
  (`modelcards/TSMC6/cln6_1d8_sp_v1d0_2p2.l`).
* `scripts/tsmc6_restore_campaign.sh` waits for the datasets, then trains
  DirectNet → BSIM-AR → PFN, 4 sizes × 2 devices each (24 checkpoints, clean
  recipe). **Then they still need gating** — the 4-cell complex matrix + device
  suites per family/size. Not yet scripted; `MODEL=… SIZE=… TECHS=tsmc6 bash
  scripts/gate_matrix_iso.sh` is the shape.

## 4. Thread D — PFN xl

`("tabpfn", "xl")` preset added: embed 192, n_inducing 64, dist/agg 4/4, ICL 9
blocks, 12 heads → **14,856,877 params**, mirroring BSIM-AR xl's 14.81 M with
ICL width 384 = its `d_model`. lr 3e-4 (large's 4e-4 produced 5 of the 8
divergence collapses and this stack is deeper), 150 epochs, `--amp`.

8 checkpoints training (4 techs × N/P). Measured ~10 min/epoch for a solo job;
with 8 sharing 3 GPUs on a box at loadavg ~1840, expect **~2–3 days**. Then they
need the 16-cell gate + device suites, and `by-scale.md` §2's PFN row updates
from 3 tiers to 4.

## 5. Finding: the opamp open-loop AC gate has a bias-resolution defect

Not fixed here — changing an accuracy gate changes the accuracy record, which
should be a deliberate decision, not a side effect of a docs pass.

`verify_complex_opamp_ac` linearizes both sides about their own peak-gain bias,
found as `argmax |dVout/dVin|` over a DC sweep with a **2 mV step**
(`OpAmpParams.step`). A two-stage Miller opamp with 33–48 dB of gain has a
transition only ~3–14 mV wide, so that grid samples the transition with a
handful of points and the "peak-gain" sample lands off-centre. Measured
reference (NGSPICE, BSIM-CMG) output at the bias the harness picks:

| tech | reference Vout | % of VDD | inside the gate's 15–85 % window? |
|---|---|---|---|
| TSMC5 | 0.021 V | 3.2 % | no |
| TSMC7 | 0.584 V | 77.9 % | yes |
| TSMC12 | 0.722 V | 90.2 % | no |
| TSMC16 | 0.685 V | 85.6 % | no (marginally) |

`op_valid` is then applied **to the NN only**, so on three of four techs a model
that faithfully reproduces the reference OP is scored
`FAIL [OP-MISBIAS: NN opamp output railed]`. Clearest case: BSIM-AR `small` on
TSMC5 — NN OP 0.042 V vs the reference's 0.021 V, DC-gain err 0.39 dB, GBW 1.19,
PM err 14.3°, i.e. inside all three *gated* criteria — scored FAIL.

**Proposed fix (needs a decision + a re-gate):** refine the sweep near the
transition (local bisection or a 10× finer grid) so the peak-gain sample is the
actual steepest point, and judge `op_valid` against the *reference's* OP rather
than an absolute band. Until then treat the opamp-AC row as a lower bound and
TSMC5's cell as unreachable by construction.

## 6. Resume checklist

1. `python scripts/v710_regate_collect.py && python scripts/v710_regate_control.py`
   — refresh `results/v710_regate/REPORT.md`, confirm the control is still 0
   disagreements, then re-run the doc injectors (the generators live in the
   session scratchpad; regenerate or fold them into `scripts/` if they are
   needed again).
2. Check `scripts/tsmc6_restore_campaign.sh` progress; when the 24 checkpoints
   carry `.complete`, gate them and fill `by-tech.md` §5's V7.1.0 tables.
3. Check the PFN-xl logs for the divergence signature the `large` tier showed
   (train loss exploding 0.02 → 0.77 mid-run); a diverged run early-stops and
   banks its pre-divergence EMA best, which is still tier-representative.
4. Gate PFN-xl, fill `by-scale.md` §2's PFN xl cell and the family report's
   tier table.
5. `docs/CHANGELOG.md` V7.1.0 entry — draft in the session scratchpad.
6. Decide on the opamp-AC gate fix (§5).

## 7. Gotchas re-confirmed this session

* Never drive these gates through Agent/workflow subagents — launch detached
  `setsid nohup` background bash.
* Two dispatchers must never share a job: they collide on the log file. The
  collector now returns `RACED` for a log with two completion markers, and
  extra dispatchers were given **disjoint** job files.
* The box ran at loadavg 1500–1900 from other users all session; gate cells took
  8–30 min each instead of 20–50 s. Plan wall-clock from measured in-flight job
  age (`ps -o etimes`), not from an idle-box estimate.
* **Editing code mid-campaign cost one cell.** The TSMC6 restore spans two
  repos (parent + PyCMG submodule); a job that started between the two commits
  died with `KeyError('tsmc6')` and was scored FAIL. Additive registry edits are
  otherwise safe here because every job passes `--tech` explicitly — but the
  cross-repo window is not, and the only reason it was noticed is the
  HEAD-vs-`d2ea720` control. Prefer the V6.13.0 discipline: run long campaigns
  from a frozen snapshot of the tree.
