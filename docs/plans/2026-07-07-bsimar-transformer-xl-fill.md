# V6.8.1 — BSIM-AR Transformer (LEVEL=74) XL-tier fill

**Goal.** Fill the one untrained tier of the V6.8.0 BSIM-AR Transformer campaign:
the **XL** preset (384×8L×ff1536, **14.81M params**, 5.5× DirectNet-xl, 3× tf-large).
The V6.8.0 scale study covered small/medium/large (clean) and gated recipes at
medium+large; XL was coded (`SIZE_PRESETS[("transformer","xl")]`) but never
trained. Train the full Phase-B recipe mirror at XL, gate vs NGSPICE, update the
report.

**Status: LIVE** (2026-07-07).

## Why XL is worth a run

- Transformer capacity curve **peaks at MEDIUM** (clean complex 12→14→13 S/M/L)
  — one tier earlier than DN (peaks at large). So on raw count XL is expected to
  regress, BUT the DN XL study ([[v665-xl-retest-strict-omp]]) found basins
  **shuffle** at xl: corroft/crit10/crit15m@xl banked tsmc16-opamp and tied
  production 14/16 strict. Open question: does the transformer at XL shuffle
  basins / bank a different cell / change the strict count vs corroft@medium
  (15/16 strict, the standing best)?
- AC peaks at small for the transformer (7/8→4/8→4/8); DN AC collapsed to 0 at
  xl. Does tf-XL AC hold or collapse? (csob recipe probes the cap axis.)

## Recipe set (user-chosen: full Phase-B mirror, 6 recipes)

Mirrors the V6.8.0 large study exactly, at XL:

| recipe   | from        | args                                                        |
|----------|-------------|-------------------------------------------------------------|
| clean    | scratch     | (control; production slot `tsmc{X}_tf_xl_{dev}`)            |
| csob     | scratch     | `--charge-sobolev` (AC/cap axis; MATH SDPA, no AMP)         |
| corroft  | clean xl    | `traj_corridor=3.0`, 120ep lr3e-4 (the ring lever)          |
| crit30   | clean xl    | `traj_corridor=3.0,inv_trip=2.0` (DN production recipe)     |
| crit15m  | clean xl    | `traj_corridor=1.5,inv_trip=3.0` (DN xl winner)             |
| corro15  | clean xl    | `traj_corridor=1.5` (gentle hedge; keeps tsmc7-opamp basin) |

48 checkpoints (6 × 4 techs × 2 devs).

## Execution waves (considerate pacing: NSTREAMS 4-6, PAR 16-24, GPUs 0-2 shared)

1. **Wave 1 — clean xl** (8, from scratch). Curriculum dependency for 4 recipes.
2. **Wave 2 — csob xl** (8, from scratch, independent). Can overlap Wave 1 tail.
3. **Wave 3 — corridor curricula** (corroft/crit30/crit15m/corro15 = 32,
   fine-tune from clean xl). Blocked on Wave 1.
4. **Gate** — `MODEL=transformer SIZE=xl gate_matrix_iso.sh` 16-cell matrix per
   recipe (OMP=1); `recipe_eval.sh` device DC/tran + AC per tech.
5. **OMP strict** — `recipe_multirun_gate.sh` OMP∈{1,2,4} on the XL winner's
   fragile cells (rings/opamps).
6. **Report** — add an XL section to `docs/V6.8.0-bsimar-transformer-report.md`
   (retitle scope), CHANGELOG V6.8.1, memory.

## Gating (unchanged from V6.8.0)

Ground truth = NGSPICE BSIM-CMG (LEVEL=72), CPU-pinned. TF pins
`PYCIRCUITSIM_NN_CHECKPOINT_TF_{NMOS,PMOS}` + `PYCIRCUITSIM_NN_FORCE_LEVEL=74`
retarget LEVEL=73 decks to BSIM-AR. `GATE_SCRATCH` overridden to this session's
scratchpad (the script default points at a stale session dir).

## Execution log

- 2026-07-07: plan created. XL transformer confirmed builds @14.81M params
  (14,808,205). No `tf_xl` ckpts on disk (genuinely untrained). Disk 6.0T free.
  GPUs 0-2 at 100% from other users (hey/luojx/xuzh) but ~14-18GB free each; tf
  xl jobs use ~3.5GB GPU each → 2/GPU fits. Launched Wave 1 (clean xl, 8 jobs,
  NSTREAMS=6, TRAIN_OMP=4, nice -12). Timing: epoch 1 ~6 min wall (incl. CUDA
  init + LDS-weight compute + loader warmup); per-job CPU capped ~108% under the
  shared-box contention → steady-state epochs ~5-6 min under contention.
  patience=80 should early-stop well before 300ep. Background waiter (baqz84cac)
  fires on all 8 `.complete` markers or a proc-stall.

## ⚠️ RESUME STATE (2026-07-08 00:29 — server shutdown imminent, training WILL be killed)

**Wave 1 (clean xl) was INTERRUPTED mid-training — NOT complete.** Snapshot:

| job | status @00:29 | epoch | val NRMSE |
|---|---|---|---|
| tsmc5_tf_xl_nmos  | running, partial `_best.pt`, **no `.complete`** | 48 | 0.00359 (still `*best*`) |
| tsmc5_tf_xl_pmos  | running, partial `_best.pt`, **no `.complete`** | 49 | 0.00319 |
| tsmc7_tf_xl_nmos  | running, partial `_best.pt`, **no `.complete`** | 55 | 0.01073 |
| tsmc7_tf_xl_pmos  | running, partial `_best.pt`, **no `.complete`** | 45 | 0.01011 |
| tsmc12_tf_xl_nmos | running, partial `_best.pt`, **no `.complete`** | 39 | 0.00413 |
| tsmc12_tf_xl_pmos | running, partial `_best.pt`, **no `.complete`** | 39 | 0.00418 |
| tsmc16_tf_xl_nmos | **NOT STARTED** (queued behind NSTREAMS=6, no `_best.pt`) | — | — |
| tsmc16_tf_xl_pmos | **NOT STARTED** (no `_best.pt`) | — | — |

~5 min/epoch under contention; jobs were ~epoch 40-55 of 300 (patience 80), val
still improving each epoch → **none converged**. Waves 2 (csob) & 3 (curricula)
never launched. No gating done. No `.complete` markers were written for ANY job.

### 🔑 RESUME COMMAND (do this first on the fresh server)

The 6 partial `tsmc*_tf_xl_*_best.pt` files exist WITHOUT `.complete` markers.
`recipe_train.sh` **SKIPs** a bare `_best.pt` (warns but treats as done) — a
silent-green trap: a naive re-run would keep the un-converged epoch-~50 weights
for 6 techs and only train the 2 tsmc16. So **force a clean retrain of all 8**:

```bash
cd /data2/shenshan/PyCircuitSim
# nuke the 6 partial (un-converged, no .complete) clean-xl snapshots so the
# retrain is honest-uniform (matches the large-tier 300ep/early-stop contract):
rm -f external_compact_models/bsimar/checkpoints/tsmc*_tf_xl_{nmos,pmos}_best.pt \
      external_compact_models/bsimar/checkpoints/tsmc*_tf_xl_{nmos,pmos}_norm.npz \
      external_compact_models/bsimar/checkpoints/tsmc*_tf_xl_{nmos,pmos}_config.npz
# Wave 1 relaunch (clean xl, 8 ckpts):
MODEL=transformer SIZES=xl RECIPES=clean TECHS="tsmc5 tsmc7 tsmc12 tsmc16" \
  DEVS="nmos pmos" GPUS="0 1 2" NSTREAMS=6 TRAIN_OMP=4 \
  nice -n 12 bash scripts/recipe_train.sh > /tmp/wave1_clean_xl.log 2>&1 &
```

(Alternatively pass `--force` to `recipe_train.sh` instead of `rm`; the `rm` is
cleaner. Keep NSTREAMS≤6 / TRAIN_OMP=4 for considerate pacing — box is shared.)

### Then (unchanged from the plan above):
2. **Wave 2** csob xl (from scratch) + **Wave 3** corridor curricula
   (corroft/crit30/crit15m/corro15 — fine-tune, `--init-from tsmc{X}_tf_xl_{dev}`,
   so they REQUIRE the clean-xl `.complete` from Wave 1 first).
   `MODEL=transformer SIZES=xl RECIPES="csob corroft crit30 crit15m corro15" ... bash scripts/recipe_train.sh`
3. **Gate**: `MODEL=transformer SIZE=xl RECIPES="clean corroft crit30 crit15m corro15 csob" GATE_SCRATCH=<this-session-scratch>/gate_iso OMP=1 PAR=20 bash scripts/gate_matrix_iso.sh`
   + `MODEL=transformer SIZES=xl RECIPES="..." PAR=16 bash scripts/recipe_eval.sh` (device DC/tran + AC).
4. **OMP strict** on the winner: `MODEL=transformer bash scripts/recipe_multirun_gate.sh <recipe> xl <TECH> <suite>`.
5. **Report**: XL section into `docs/V6.8.0-bsimar-transformer-report.md`, CHANGELOG V6.8.1, memory.

Background waiter baqz84cac and the detached dispatcher die with the server —
nothing to clean up; just start from the RESUME COMMAND.

## ⚙️ RELAUNCH (2026-07-09 — fresh server, GPUs 0-2 free, box idle)

**Two lessons corrected the plan; run is LIVE at full fidelity.**

1. **csob OOM (memory profiles are NOT uniform).** `--charge-sobolev` (2nd-order
   autograd) uses **~10 GB GPU/job at xl** vs **~3.4 GB** for plain jobs. The
   first relaunch mixed clean+csob in ONE round-robin dispatcher → it co-located
   2 csob (20 GB) + 2 plain (6.8 GB) = 26.8 GB on a 24 GB GPU → `CUDA out of
   memory`. **Fix: homogeneous memory-profile waves, never co-locate heavy csob
   past capacity.** Driver (persistent, 2026-07-10):
   `results/recipe_bench/xl_campaign/xl_train_driver.sh`.
     - Wave A  clean    (8 plain, NSTREAMS=12, ~3/GPU=10 GB)   [curriculum base]
     - Wave C  curricula(32 plain, NSTREAMS=12, ~4/GPU=14 GB)  [needs clean .complete]
     - Wave B  csob     (8 heavy, NSTREAMS=3,  1/GPU=~10-14 GB)[independent, LAST]
       — NSTREAMS=3 not 6: XL is GPU-bound so 1 job/GPU already = 100% util & the
       SAME wall-clock as 2/GPU (time-slicing adds no throughput), while 1/GPU
       removes the OOM headroom risk of an 8-job static-assignment pile-up.
     - Order A→C→B: central strict-count result lands before the csob AC probe;
       `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, TRAIN_OMP=4.

2. **NO early-stop — the plan's "patience=80 early-stops before 300ep" is FALSE.**
   Existing `tf_large` logs ran the FULL 300 epochs (Done in 20363-34281 s,
   68-114 s/epoch). Measured xl: **~3.3 min/epoch (plain, 3/GPU, GPUs 100% util
   = GPU-bound at 3× params)**, csob heavier (~5 min/epoch). Full 48-ckpt mirror
   ≈ **~3 days GPU** (A ~16 h, C ~20 h, B ~33 h).
     - **User decision (2026-07-09): FULL 300/120 fidelity** — faithful mirror of
       the medium/large contract for a valid basin-shuffle comparison; run
       detached, `.complete`-gated resume on any interruption.

### 🔑 RESUME COMMAND (2026-07-10 PERSISTENT relaunch — supersedes 2026-07-09)

**Why this supersedes 2026-07-09:** the `nohup`'d scratchpad driver did NOT
survive the session teardown — the harness wiped the session scratchpad (driver
script + wave logs gone) AND the teardown SIGKILL reached the driver's process
group (`nohup` ignores SIGHUP only). The v1 run produced 0 usable ckpts (8 clean
best.pt killed at epoch 13, no `.complete`; csob logs show `Killed`=OOM).

**Two fixes make the relaunch teardown-proof:**
- Driver + all logs live under **`results/recipe_bench/xl_campaign/`** (persistent
  `/data2`), never the session scratchpad.
- Launch with **`setsid`** → the driver gets its own session (verified PPID=1,
  reparented to init), immune to the parent process-group teardown kill.

`.complete` markers gate re-runs (a completed job SKIPs). To (re)launch/resume:

```bash
cd /data2/shenshan/PyCircuitSim
CAMP=results/recipe_bench/xl_campaign
setsid nohup bash "$CAMP/xl_train_driver.sh" > "$CAMP/xl_driver.stdout" 2>&1 &
# Idempotent: recipe_train.sh SKIPs any stem whose _best.pt exists; completed
# jobs write _best.pt.complete. A rerun retrains only what is missing. If a
# partial (bare _best.pt, no .complete) blocks a retrain, rm it or pass --force.
# Wave A --force-retries once if <8 clean .complete; aborts (xl_train.FAILED) if
# the clean base can't complete (curricula need init-from it).
```

Live state / markers (all under `$CAMP`):
- `xl_driver.log`   — timestamped wave milestones (`say` lines).
- `waveA_clean.log`, `waveC_curricula.log`, `waveB_csob.log` — dispatcher logs.
- `xl_train.DONE` (48/48) or `xl_train.FAILED` — terminal markers.

Progress check: `.complete` counts per stem —
`ls external_compact_models/bsimar/checkpoints/tsmc*_tf_{xl,csob_xl,corroft_xl,crit30_xl,crit15m_xl,corro15_xl}_{nmos,pmos}_best.pt.complete | wc -l` (target 48).
Health: `pgrep -af xl_train_driver.sh` alive + `nvidia-smi` GPUs busy.

### ⚠️ Wave-transition instability (2026-07-13)

- **Wave A clean: 8/8 ✓** (completed 00:13, Jul 13). Curricula auto-started.
- **Wave C attempt 1 got 0/32 — transient SIGKILL event at 02:40.** All 3 in-flight
  corroft jobs were `Killed` (SIGKILL, no traceback = OOM/kill event) at epoch
  10-20, which tore down the whole `recipe_train.sh` xargs batch → attempt 1
  returned 0/32. Correlated with co-tenant **wangyk's Xyce fleet launch** (dozens
  of ~2.5 GB procs); our XL jobs are only ~1.7 GB RSS so we were collateral, not
  the cause. System RAM was fine at inspection (375 GB avail); `cron.service`
  cgroup (where the cron-launched trainer lives) is `memory.max=max` (unlimited),
  so it wasn't a per-cgroup cap. journalctl -k not accessible → exact killer
  unconfirmed, but SIGKILL signature + timing ⇒ system memory/kill event.
- **Driver auto-recovered:** attempt 2 (started 02:40) is HEALTHY — all 3 jobs
  banking fresh `best.pt` (R state, 100% GPU, growing CPU).
- **FIX (safe, relaunch-only):** raised `ensure_wave` attempt cap **3 → 30** +
  60 s backoff, so recurring transient kills can't exhaust the cap and abandon
  the campaign at <48 (`.complete` markers keep every re-dispatch monotonic —
  completed stems SKIP). The *running* driver already sourced the old function;
  the edit takes effect on the next relaunch. **Safety net:** if the running
  driver exits <48 (old cap hit), the session waiter detects STALL → relaunch the
  edited driver → it resumes from `.complete`.
- **If kills RECUR frequently** (jobs never survive the ~8 h to first `.complete`):
  deeper fix needed — identify the killer (who sends SIGKILL), and/or drop Wave
  C `NSTREAMS_ALL` 3→2, and/or coordinate co-tenancy. Not warranted yet (single
  event, attempt 2 stable).

### Then (gate + report — unchanged):
3. **Gate**: `MODEL=transformer SIZE=xl RECIPES="clean corroft crit30 crit15m corro15 csob" GATE_SCRATCH=/data2/shenshan/PyCircuitSim/results/recipe_bench/xl_campaign/gate_iso_scratch OMP=1 PAR=20 bash scripts/gate_matrix_iso.sh` + `recipe_eval.sh` (device DC/tran + AC). NOTE: `gate_matrix_iso.sh`'s GATE_SCRATCH default points at a STALE session dir — ALWAYS override it (persistent path above).
4. **OMP strict** on the winner: `MODEL=transformer bash scripts/recipe_multirun_gate.sh <recipe> xl <TECH> <suite>`.
5. **Report**: XL section into `docs/V6.8.0-bsimar-transformer-report.md`, CHANGELOG V6.8.1, memory.
