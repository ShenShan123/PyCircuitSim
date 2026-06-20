# Enrich the DirectNet complex-circuit test harness (parametric sweeps) + TSMC7 retrain

## Context

The four DirectNet (LEVEL=73) complex-circuit verifiers
(`tests/verify_complex_{opamp,ring_osc,switchcap,sram_snm}.py`) today run **one
fixed operating point per tech** — hardcoded geometry (L=16n/20n, NFIN=2), one
checkpoint VT per tech, one stimulus (sram already sweeps NFIN corners). The
inverter suite has a rich parametric harness (`tests/common/nn_sweep.py` +
`verify_nn_multi_tech_{dc,tran}.py`) that sweeps geometry/VT/VDD/stimulus,
baseline-gated, with Rule-16 metrics.

**Goal:** bring that parametric-sweep capability to the four complex circuits —
sweeping **technology, VT variants (incl. asymmetric NMOS/PMOS VT), geometry
(L / NFIN / P-N fin ratio), VDD, and per-circuit input stimuli** — mirroring the
inverter harness. Ground truth stays **NGSPICE BSIM-CMG (LEVEL=72)** always.
**Additionally, retrain the TSMC7 DirectNet model** so the swept geometry/stimulus
space is within-distribution (otherwise most TSMC7 sweep points fall outside the
specialized production checkpoint's training envelope and FAIL the hard gate as
model-extrapolation, not as a meaningful characterization).

### Decisions (confirmed with the user)
1. **Structure:** new shared `tests/common/complex_sweep.py` + new thin per-circuit
   driver scripts. The existing four `verify_complex_*.py` stay **untouched** as the
   authoritative single-point ship gate (campaign closed at 15/16).
2. **Gate semantics:** **hard gates** — every swept config is a real PASS/FAIL at
   the circuit's domain tolerance. The new drivers carry their own exit code.
3. **Tech scope:** implement correct for all 4 TSMC nodes; **verify on
   TSMC7/TSMC16** (only nodes with checkpoints present here). TSMC5/TSMC12 emit
   fail-loud ERROR rows until their checkpoints are restored.
4. **Breadth:** **thorough/exhaustive** — full per-tech L list, all usable VT
   variants + asymmetric N/P pairs, 3+ stimulus values per dimension.
5. **TSMC7 retrain:** retrain TSMC7 DirectNet (NMOS+PMOS) with broad
   geometry/stimulus coverage so sweep configs are in-distribution, and
   **overwrite the shipping `tsmc7_dn_medium`** so it becomes the new default for
   BOTH the sweep and the single-point gate. **Re-baseline** the TSMC7 ship gate
   afterward — the opamp PASS must be re-confirmed or documented as a regression
   (accepted risk to the 15/16 close).

### Verified feasibility
- **VT sweep works:** one per-tech checkpoint covers ALL its VT variants via the
  `nn.Embedding` tech-code; NMOS & PMOS resolve **independently** per `.model`
  line. **Asymmetric N/P VT is valid.**
- **Usable VT per tech** = `ALL_TECHS[t].vt_pairs` ∩ `bsimar.config.LOCAL_VARIANT_CODES[t]`:
  TSMC5 `{lvt,ulvt,elvt}`, **TSMC7 `{ulvt}` only** (svt/lvt removed from ground
  truth: garbage / PDIBL2_i<0), TSMC12 `{svt,lvt,hvt,ulvt,lnvt}`, TSMC16
  `{svt,lvt,hvt,ulvt}`. ⇒ The asymmetric-VT feature has a real witness only on
  TSMC16 here; TSMC7's VT space stays single even after retrain (ground-truth
  bound, not a model bound) — see Risk R-VT.
- **Silent-UNKNOWN pitfall:** an out-of-vocab (tech,VT) maps to LOCAL_UNKNOWN with
  NO warning (warning only in universal scope). Builders must enumerate from the
  `usable_vts()` intersection; `bench_variant()` must raise when
  `(tech,vt) ∉ LOCAL_VARIANT_CODES` — check the **DN vocab**, not just
  `get_vt_pair` (which only knows base.py).
- **Geometry works:** off-default L/VT modelcard resolution already used by the
  inverter DC sweep; `bake_inst_params` bakes arbitrary per-device {L,NFIN}.
- **Opamp S2 win** ("continuation-first") lives in `pycircuitsim/simulation.py`
  `run_dc_sweep` (NN-gated). Untouched — preserved for free.

### Resolved decision (TSMC7 retrain)
**Overwrite + re-baseline.** Train a broad-coverage TSMC7 checkpoint and overwrite
the shipping `tsmc7_dn_medium` (new default for sweep + single-point gate). The
shipping TSMC7 is the *specialized* `v6_4_7_pivcor_w2_s7_tsmc7`; the broad retrain
may not reproduce its opamp 8.63% PASS — after retrain, re-run the single-point
gate and either confirm the opamp still PASSES or **document the regression in
CHANGELOG/MEMORY** (a dead-end record if it regresses; the broad coverage is the
deliberate trade). TSMC16 (unchanged checkpoint) keeps its exact ship-gate
numbers.

---

## Architecture

All new code is **additive to shared infra** + **new files**. The four
`verify_complex_*.py` are not edited.

| File | Change |
|------|--------|
| `tests/common/complex.py` | **Additive, content-preserving for existing callers:** fix L cache-key bug; extend `BenchTech` (nmos_vt/pmos_vt/nfin_p + properties); add `usable_vts()` + `bench_variant()`; add per-circuit stimulus dataclasses; add **parametric programmatic netlist builders** (NGSPICE + DirectNet) for opamp/ring_osc/switchcap; add shared measurement helpers. |
| `tests/common/complex_sweep.py` | **NEW** — mirror `nn_sweep.py`: `ComplexSweepConfig`, baseline + parametric builders per circuit, single-test orchestrators, baseline-gated driver, summary table/CSV/bar plot, hard-gate thresholds, 3-state exit. |
| `tests/verify_complex_{opamp,ring_osc,switchcap,sram_snm}_sweep.py` | **NEW** — thin drivers (like `verify_nn_multi_tech_*.py`). |
| `tests/verify_complex_*.py` (existing 4) | **UNTOUCHED.** Helper duplication into shared infra is the accepted cost. |

> **Equivalence guarantee (corrected per review C1):** the complex.py edits change
> only the cache *key* and baked *filename*, never the baked *content* or the
> resolved checkpoint, for the baseline. Prove this with a **SHA-256
> content-equality check** of the baked `.lib` (old vs new, filename-normalized)
> per BENCH entry — NOT merely a headline-number match (numbers match even when
> decks differ, so they're a false negative for content equality).

---

## Implementation steps

### Step 1 — Fix `get_baked_modelcard` L cache-key bug (`complex.py`)
- Key `(bt.name, bt.vt, nfin)` → `(bt.name, bt.effective_nmos_vt,
  bt.effective_pmos_vt, bt.l_nmos, bt.l_pmos, nfin)`; update annotation.
- Baked filename embeds L + both VTs (avoid collisions in a shared `work_dir`).
- Resolve NMOS card from `get_vt_pair(bt.effective_nmos_vt)`, PMOS card from
  `get_vt_pair(bt.effective_pmos_vt)` (per-device). Baseline (equal VTs, L=16/20)
  → SHA-256-identical content.

### Step 2 — `BenchTech` parametric + `bench_variant()` / `usable_vts()` (`complex.py`)
- Add (defaults preserve all callers): `nmos_vt:str=""`, `pmos_vt:str=""`,
  `nfin_p:int=0`; properties `effective_nmos_vt`(→`nmos_vt or vt`),
  `effective_pmos_vt`, `effective_nfin_p`(→`nfin_p or nfin`).
- `_resolve_bench_tech` sets `nmos_vt=pmos_vt=ckpt_vt` (no behaviour change).
- `usable_vts(tech) -> set[str]` = `{v.vt_name for v in ALL_TECHS[tech].vt_pairs}`
  ∩ `LOCAL_VARIANT_CODES[tech.lower()]` keys (lazy import).
- `bench_variant(base, **overrides)` (review C2/M2): accept `nmos_vt`/`pmos_vt`
  **independently**; set `nmos_model = get_vt_pair(nmos_vt).nmos_model` and
  `pmos_model = get_vt_pair(pmos_vt).pmos_model` (pull each side from its own
  VtPair — the single most error-prone line). **Raise ValueError** when
  `(tech.lower(), v.lower()) ∉ LOCAL_VARIANT_CODES[scope]` (check DN vocab, not
  just `get_vt_pair`).

### Step 3 — Per-circuit stimulus dataclasses (`complex.py`)
Frozen; every field defaults to today's value so a bare `()` reproduces the single
point. `OpAmpParams` (vcm/vbn/vbp fracs, cc=20, cl=50, span=0.15, step=0.002 —
**assert these reproduce the script's lo/hi/step exactly**, review m2),
`RingOscParams` (n_stages=5, cload=0.5, tstep/tstop/settle + a **tstop formula**,
M3), `SwitchCapParams`, `SramParams` (wl_frac, nfins, storage_states, dc_step).

### Step 4 — Parametric programmatic builders + measurement helpers (`complex.py`)
Add parametric NGSPICE + DirectNet builders per circuit (sram already this shape):
`{ngspice,directnet}_{opamp,ringosc,switchcap}(...)`. Each emits per-device
`L/NFIN`, two `.model` lines from `effective_nmos_vt`/`effective_pmos_vt`, VDD/bias
from `bt.vdd`×fracs, stimulus from the dataclass. **Keep the two PULSE syntaxes
distinct per side** (NGSPICE `PULSE(...)` vs PyCircuitSim space-separated, review
m4). Move pure measurement helpers (`gain_trip`, `period_from_wave`,
`snm_from_lobes`, `at`, `bias`) into `complex.py` for sharing.

### Step 5 — `tests/common/complex_sweep.py` (NEW; mirror `nn_sweep.py`)
- `ComplexSweepConfig{bt, tech_key, circuit, stim, sweep_type, config_name, swept}`.
- `make_<c>_baseline` / `build_<c>_parametric` / `run_single_<c>` /
  `run_complex_multi_tech` (baseline-gated copy of `run_nn_multi_tech`) /
  summary table+CSV+bar plot carrying BOTH waveform NRMSE and the domain metric.
- **3-state exit code (review C3):** `0` = every attempted config PASSED and ≥1
  config actually ran; `1` = any FAIL; `2` = a requested tech was all-ERROR
  (could-not-characterize). **Bake/OSDI-Fatal/absent-checkpoint → ERROR, never
  FAIL** (review C5). CI invokes `--tech TSMC7,TSMC16` so absent techs never green
  the build.
- **Checkpoint pin (review m1):** at sweep start, sha256-verify
  `bsimar/checkpoints/` against a recorded manifest; fail loud on mismatch (the
  trip/opamp gain ~20× amplifies weight drift).

### Step 6 — Sweep dimensions (thorough; outer loop = tech; one-dim-at-a-time)
**Shared (all 4, via `bench_variant`):**
- **VT symmetric:** each `v ∈ usable_vts(tech) − {baseline}` (nmos_vt=pmos_vt=v).
- **VT asymmetric N/P:** all ordered `(vn,vp)`, vn≠vp, from `usable_vts` (emit only
  when |usable|≥2: TSMC16→12, TSMC12→20, TSMC5→6; TSMC7→0).
- **L:** `l_nmos ∈ l_values−{16n}`; `l_pmos ∈ l_values−{20n}`; symmetric
  `l_nmos=l_pmos ∈ l_values`. One-dim-at-a-time (never cross L×VT×NFIN — that
  keeps every point inside the per-(L,NFIN=2,defaultVT) PDIBL2 pruning that
  base.py already validated, review C5).
- **NFIN symmetric:** `{3,5,10}` (skip 2).
- **P/N ratio:** `nfin_p = 3` **only** (`≤ nfin+1`, the modelcard NFIN-group rule;
  review C5 — drop the invalid nfin_p=4).
- **VDD:** `±0.05, ±0.1 V`.

**Per-circuit stimulus (3+ values):**
- **opamp:** vcm_frac {0.45,0.50,0.60,0.65}; cc {10,40} fF; cl {20,100} fF; span {0.10,0.20} V.
- **ring_osc:** n_stages {3,7,9}; cload {0.25,1.0,2.0} fF (**tstop sized from the
  measured NGSPICE period**, M3).
- **switchcap:** vin_frac {0.3,0.4,0.8}; csample {50,200} fF; clk_per {2,8} ns; clk_slew {0.05,0.2} ns.
- **sram:** nfins (2,3,5,10); wl_frac {0.9,1.0} (**force_ic gated only at
  wl_frac=1.0/OFF-hold; wl_frac<1.0 rows are butterfly-only**, review M6); both
  storage states.

### Step 7 — Hard-gate thresholds (per circuit)
NN vs NGSPICE compared at the **same** config (apples-to-apples).
- **opamp:** `gain_err ≤ 10%`. **If the NGSPICE reference gain `ng_gain < 5 V/V`
  (out-of-region bias under VDD/VT sweep), classify ERROR/out-of-region, not
  PASS/FAIL** (review M4 — the ratio gate is uninformative near zero). Trip-shift
  reported.
- **ring_osc:** `period_err ≤ 5%`; NaN period (window too short) → ERROR, not FAIL.
- **switchcap:** `charge_err ≤ 5%·VDD` **and** droop within
  `max(10%·|ng_droop|, 0.1%·VDD)`.
- **sram:** all lobes positive **and** force_ic ok (wl=OFF only) **and** SNM gate.
  **SNM threshold derived from measured baselines (review M5):** run the four-tech
  baseline first, set the bound at ~2× the worst baseline SNM-err (NOT a round
  25%); if baselines are noisy, keep SRAM's existing gate (positivity + force_ic)
  and report SNM-err without gating.

### Step 8 — Thin driver scripts (NEW, per circuit)
`verify_complex_<c>_sweep.py` — argparse `--tech`/`--dimension`, call
`run_complex_multi_tech`, write summary under
`tests/verify_complex_results/<c>/sweep/`, honor the 3-state exit code.

---

## Phase A — TSMC7 retrain (run before/parallel to the sweep verification)

Mirror the CLAUDE.md per-tech pipeline. Train on GPU; **gate/eval on CPU**
(`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).

1. **Data:** regenerate TSMC7 with full bin coverage:
   `generate_nn_data.py --device both --tech tsmc7 --enable-inv-trip --n-workers 8`
   (per-bin OSDI-convergence drop keeps only valid (variant,L,NFIN) bins, Rule 9 —
   so svt/lvt survive only where ground truth is valid; this is why TSMC7 VT stays
   effectively single, R-VT).
2. **Train:** `python -m bsimar.cli.train --model direct --size medium
   --device-type {nmos,pmos} --tech-scope tsmc7 --cuda --overwrite` (overwrites
   `tsmc7_dn_medium`). **Back up the current `tsmc7_dn_medium*` first** so the
   prior specialized checkpoint is recoverable if the retrain regresses. **Size =
   medium**, not large (V6.4.8 S1 proved large COLLAPSES the opamp / regresses RO
   — capacity is not the bind).
3. **Validate + re-baseline:** lifted-source canary
   (`verify_nn_lifted_source_dc.py`), inverter gate, then the new TSMC7 sweep.
   **Re-run the untouched `verify_complex_*.py --tech TSMC7`** and record the new
   single-point numbers: opamp must still PASS, or document the regression in
   `docs/CHANGELOG.md` + MEMORY (dead-end record). TSMC16 numbers must be
   unchanged (its checkpoint is untouched).

---

## Verification (TSMC7/TSMC16; always CPU-pin: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`)

1. **Baseline content-equivalence (critical guard, C1):** SHA-256 of every baked
   `.lib` (pre- vs post- complex.py edits) equal — this holds for ALL techs since
   the NGSPICE ground truth is independent of the NN. **TSMC16** single-point
   numbers must match S2 EXACTLY (opamp 4.92%, ring 3.99%, sc 2.01%, sram
   all-positive+force_ic) — its checkpoint is untouched. **TSMC7** is intentionally
   **re-baselined** by Phase A (record the new numbers; opamp must PASS or be
   documented as regressed) — do NOT expect the old 8.63%.
2. **Opamp equivalence canary (C4):** assert the programmatic opamp baseline deck
   is **line-set-identical** (whitespace/comment/include-path normalized) to the
   template-rewrite deck, with **identical `.dc` lo/hi rounding (`round(...,3)`) and
   point count** — the opamp basin is path-dependent, so numeric match alone is
   unsound. Ring_osc second canary (preserve `.ic` seed ordering).
3. **Sweep smoke (TSMC16 — only runnable node with a real VT space):** run all four
   `_sweep.py --tech TSMC16`; confirm baseline row matches step 1; VT-sym
   {lvt,hvt,ulvt}; 12 asym pairs; L/NFIN/nfin_p/VDD/stimulus rows; CSV+PNG written.
4. **3-state exit (C3):** `--tech TSMC5` (absent ckpt) → all-ERROR → exit **2**;
   a real FAIL → exit **1**; clean sweep → exit **0**.
5. **VT-empty:** `--tech TSMC7` → VT/asym dims = 0 configs; no silent-UNKNOWN.
6. **Opamp determinism:** opamp sweep at OMP ∈ {1,2,4}; baseline gain_err in the
   documented tiny scatter.
7. **TSMC7 retrain validation:** in-distribution TSMC7 sweep PASS-rate rises vs the
   pre-retrain checkpoint; lifted-source + inverter gates still pass.

Run **per tech, per circuit, in the background** (exhaustive × transient, no
batched NN forward). ring_osc n_stages=9 / cload=2 is the runtime worst case (M3).

---

## Risks
- **R-VT (review M1):** asymmetric-VT — the marquee new feature — has a single
  runnable witness here (TSMC16). TSMC7's VT space is ground-truth-bound to
  `{ulvt}` even after retrain; TSMC12/TSMC5 need restored checkpoints. State this
  in the harness output so coverage isn't overclaimed.
- **Runtime:** exhaustive × transient. Mitigate: background per-tech runs, short
  windows, `--dimension` subsets; ring_osc tstop formula bounds the worst corner.
- **Opamp multistability:** CPU-pin; some VDD/VT/L points FAIL the gain gate
  legitimately (or hit out-of-region ERROR, M4). S2 win preserved (untouched).
- **TSMC7 retrain vs opamp specialization (accepted):** overwriting the
  specialized checkpoint with a broad retrain may regress the opamp ship gate. Back
  up the old checkpoint; re-baseline and document. The broad coverage is the
  deliberate trade for in-distribution sweeps.
- **Checkpoint drift (m1):** sha256-pin `bsimar/checkpoints/`.
- **L cache-key bug** is a prerequisite — land Step 1 before any L run.
