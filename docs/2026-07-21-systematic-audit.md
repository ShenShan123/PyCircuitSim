# Systematic audit — 2026-07-21

Five-area parallel audit (solver, parser/models, data generation, NN architecture,
test infrastructure). Every finding below was traced to specific lines; findings
marked **[verified]** were additionally reproduced by execution and independently
re-confirmed by the coordinating session.

Baseline at audit time: `verify_bsimcmg_op` 3/3, `verify_bsimcmg_dc` 2/2,
`verify_bsimcmg_tran` 1/1, `verify_subckt` 8/8 — all green.

> **Live campaign during audit.** `scripts/recipe_train.sh` (PID 866478) had been
> running 5 days on the V6.8.1 BSIM-AR XL fill, writing `tsmc16_tf_csob_xl`.
> Nothing in `checkpoints/` or the base/`_corro_` datasets was touched.

---

## P0 — Results-corrupting

### A1. NGSPICE return code never checked; stale CSV silently reused as ground truth **[verified]**
`tests/common/base.py:254`

`run_ngspice_subprocess` executes NGSPICE and validates only that the log lacks
`"Fatal:"` and that the CSV exists — both read from a persistent,
deterministically-named work dir that is never cleaned. `res.returncode` is
referenced only at `:272`, inside an error *message*, never gated on.

Reproduced with a dead binary:

```
NGSPICE_BIN=/bin/false python tests/verify_subckt.py     -> 8/8 PASS  (NRMSE 0.638% / 0.861%)
NGSPICE_BIN=/bin/false python tests/verify_bsimcmg_dc.py -> ALL 2 tests PASSED
```

The NRMSE values are byte-identical to a legitimate run, proving cached-CSV reuse.

Reaches **every** NGSPICE-gated suite: L72 DC/tran, the V6.12.0 subckt gate, all
four complex ship gates, all NN gates. `/usr/local/ngspice-45.2` does not exist on
this host, so all matrices already depend on `tools/ngspice-45.2`; deleting it
turns the whole matrix green on stale data.

The correct pattern exists twice in-repo — `verify_bsimcmg_op.py:128` and
`PyCMG/tests/helpers.py:174` — which is why `verify_bsimcmg_op.py` is the sole
immune suite. **Fix:** check `returncode` and unlink the CSV before invoking.

### A2. `spsolve` returns NaN instead of raising; transient reports it as converged **[verified]**
`pycircuitsim/solver.py:45-49`, `:1623-1652`

`scipy.sparse.linalg.spsolve` emits only a `MatrixRankWarning` and returns NaN on a
singular matrix — it never raises `LinAlgError`. Every `except np.linalg.LinAlgError`
guard on the sparse path is therefore dead code, and `_solve_linear`'s docstring
(`:503`) promises an exception it cannot deliver.

NaN then defeats the convergence test: `dv >= threshold` is False for NaN, so
`all_converged` stays True and NaN propagates into the waveform, CSV and plot.

The Phase-6b residual OR-gate would catch it (`_mna_residual_inf` returns `inf` for
non-finite products) but is `_has_nn_device`-gated at `:1639` — **so LEVEL=72
BSIM-CMG, the ground truth every NN family is trained and gated against, is
systematically less protected than the NN paths it validates.**

Composes with C2 below into a fully silent wrong answer.

### A3. NN `gds` is missing the sign negation `gm`/`gmb` receive **[verified]**
`pycircuitsim/models/mosfet_nn.py:352-358`

```python
gm_phys  = -self._denorm_deriv("id", in_col=1, ...)   # negated
gds_phys =  self._denorm_deriv("id", in_col=0, ...)   # NOT negated
gmb_phys = -self._denorm_deriv("id", in_col=3, ...)   # negated
```

This is a **documented internal contradiction**. `losses/bni_mae.py:100-113` settles
the convention empirically and pre-emptively refutes the exact rule the simulator
follows:

> `stored gds = -d(id_stored)/dVd` … a UNIFORM negation of all three channels … for
> BOTH device types. **This is NOT the 930c274 "gds is the diagonal so no flip"
> rule: that comment is wrong for this stored convention** (it would compare
> ∂id/∂Vd against +gds, doubling the gds residual).

The loss implements uniform negation (`bni_mae.py:226`); inference does not.

Evidence:
- Raw `tsmc5_nmos.npz` (2.0M rows, conducting): `id` 81% negative, `gds` 92% positive.
- Autograd `∂id/∂Vd` vs `−gds_head`: rel err **0.12**; vs `+gds_head`: **2.08**.
  A relative error of exactly 2.0 is the arithmetic signature of a pure sign flip.
- Control: `gm`, where the code *does* negate, tracks at 3.3% — so the fit is fine,
  the sign is wrong.

Because the raw value is negative, `_floor_gds = max(gds, max(|id|·0.5, 1e-12))`
always wins: the learned output conductance is discarded at **98–100%** of
conducting bias points and replaced by a hard-wired 2 V Early voltage.

DC is insensitive (gds cancels at the NR fixed point — why ten campaigns missed it).
AC is not: `ACSolver._precompute_smallsignal` (`solver.py:2574-2581`) consumes it
directly as small-signal `r_o`.

#### A3-measured — the sign fix and the floor are COUPLED

Four-arm experiment, `tests/verify_nn_ac.py`, all 5 techs × 2 devices, CPU-pinned,
out-of-tree monkeypatch (floor varied via the existing `PYCIRCUITSIM_GDS_FLOOR_K`
knob, no code change):

| arm | gds sign | floor k | AC gates |
|---|---|---|---|
| A (shipped) | buggy | 0.5 | **8/10** |
| B | **fixed** | 0.5 | **8/10** — *bit-identical to A, every digit* |
| C | **fixed** | 0.0 | **10/10** |
| D (control) | buggy | 0.0 | **5/10** |

**Neither change is a fix on its own.** The sign fix alone is invisible; relaxing
the floor alone makes things *worse* than shipped.

**Why B is bit-identical — verified arithmetically.** `_floor_gds` binds whenever
`|gds| < k·|id|`, i.e. whenever Early voltage `V_A = |id|/|gds| > 1/k = 2 V`. A
FinFET in saturation essentially always satisfies that, so **both** the buggy
(negative) and fixed (positive) candidates lose to the floor at every amplifying
operating point. At the tsmc5 CS-amp OP: raw `−4.851e-06`, negated `+4.851e-06`,
floor `1.259e-05` → both arms return `1.259e-05`, delta exactly zero. This is the
mechanism behind the 98–100% override rate.

**Arm D shows what the floor is actually protecting against:** removing it while
the sign is still wrong stamps a genuinely negative output conductance — the Rule 4
failure mode. **The floor is load-bearing today precisely because it masks the sign
bug.**

Arm A→C per-case: 9 of 10 improve, `f3db_ratio` moves to exactly 1.000 in 6 of 10,
and both baseline failures clear with margin (tsmc12_pmos magNRMSE 12.40→3.59%;
tsmc16_pmos 14.38→5.94%). Only tsmc6_nmos degrades (2.59→3.88%), still passing.

**Opamp AC un-rails.** `verify_complex_opamp_ac.py` still FAILS in every arm, but
A→C moves TSMC16 from 0/3 to 2/3 criteria met (GBW ratio 16.6→0.927, PM error
40.3°→0.647°) and TSMC5 from 0/3 to 1/3 — and in both techs the NN DC operating
point goes from **railed to un-railed** (TSMC16 `vout=0.000, vo1i=0.000` →
`vout=0.496, vo1i=0.418`). This is a candidate contributor to the long-standing
opamp railing in `v657-vout-existence-retrain-kill` / `v658-ekv-core-breaks-opamp-rail`.

**DC confirmed invariant.** `verify_nn_dc_tran.py --tech TSMC5 --inverter-only`:
VTC NRMSE **1.04% in all three arms, to the printed digit** — the "gds cancels at
the NR fixed point" claim is now empirical, not just argued. Inverter transient is
*not* invariant (0.79%→0.97%, still passing), because a transient visits bias points
off the amplifying OP where the floor does not bind in both arms.

**This falsifies a pre-registered prediction in the code.** `mosfet_nn.py:52-57`
states the floor "is Jacobian-only and cancels at the fixed point … a diagnostic
knob, NOT a shipping accuracy lever", predicting floor-k does not move converged
opamp gain. Correct for DC; **wrong for AC**, where k=0 (sign fixed) moved the gate
count and un-railed two opamp OPs.

**Not ship-ready.** Only the device AC suite (10 gates), one DC/tran gate on one
tech, and opamp AC on 2 of 5 techs were run — no ring-osc, SRAM, switchcap, or
parametric sweeps, which `v647-s10` and `v648-s1` both record as fragile to exactly
this class of Jacobian change. `k=0.0` is also a blunt instrument: the raw autograd
gds is still genuinely negative at ~3–8% of conducting points even with the correct
sign, and those points need *some* guard. A k-scan to find where the improvement
saturates has not been done.

---

## P1 — Silent-green and convergence integrity

### B1. `final_converged` latches across source-stepping steps **[verified]**
`pycircuitsim/solver.py:650` (init), `:964` (set), `:1072` (consumed)

Initialized once *outside* both the GMIN and source-stepping loops, only ever
assigned True, never reset. It means "some (gmin, source-step) pair converged", not
"the final full-supply solve converged".

Not opt-in: source stepping is default-on for every DC path that solves (7 call
sites). The default budget is `50 // 20 = 2` NR iterations per source step, and step 0
runs at 5% supply where devices are off and converges in 2 iterations essentially
always — arming the latch on nearly every solve.

Traced live on `examples/bsimcmg_inverter_dc.sp` at stock defaults:

```
Vin=0.35  steps converged 6/20  LAST source step converged=False  reported True
Vin=0.50  steps converged 2/20  LAST source step converged=False  reported True
```

Consequence: the GMIN-retry and pseudo-transient recovery ladder in
`_solve_dc_with_retry` is close to unreachable on the default path. Three gates
consume this flag (`verify_nn_dc_tran.py:1443`, `complex.py:474`,
`complex_ac.py:155`) — all via `getattr(..., True)`, defaulting to the
*reassuring* branch if the attribute is missing.

Returned voltages were correct in every observed case (continuation still lands the
right fixed point), so this is a low-risk fix — but it arms recovery paths that have
never run, so it needs its own commit and a full gate re-run.

### B2. The "TRUE convergence check" residual can never reach zero **[verified]**
`pycircuitsim/solver.py:1128-1131` (and `:725-732`, `:1531-1537`, `:1920-1923`)

`current_iterate` is built as `np.zeros(matrix_size)` with only node slots filled,
leaving V-source branch-current slots at zero. Node rows carry `±1` in those columns,
so the residual floors at the supply current.

Demonstrated on an exactly-solvable circuit (V1=1.0V, R1=100Ω):

```
solution : {'1': 1.0, '0': 0.0}   <- EXACT
residual : 1.000000e-02           == I_supply exactly
```

Independently reproduced on `V1=1V / R=1k`: residual at the exact solution is
`0.001`, not `0`. The in-code comment — *"branch-current slots left at 0 — they only
scale the residual, not the descent test"* — is factually wrong.

Second defect on the adjacent line: `rhs_scale = max|rhs|` is dominated by V-source
*value* rows and is therefore in **volts**, while the residual it gates is in **amps**.

Consequence: for SRAM the threshold is ~6–8 mA against nA–µA retention leakage, so
`resid_ok` at `verify_complex_sram_snm.py:266` and `complex_sweep.py:471` is always
True — the check is a no-op, leaving exactly the "stale flag + rail-proximity, no KCL
gate" state that `solver.py:985-987` claims V6.4.6 closed.

### B3. Dispatchers exit 0 regardless of sub-job outcome **[verified]**
`scripts/gate_matrix_iso.sh:95`, `benchmark_run_tests.sh:87`, `recipe_eval.sh:99`, +8 more

Workers end in `exit 0` so `xargs`'s 123 never fires, and a trailing
`echo "...COMPLETE"` overwrites `$?`. Only `train_per_tech_8cells.sh` sets `-e`.
Reproduced: `bash scripts/gate_matrix_iso.sh && echo OK` prints OK at 0/16.

### B4. Empty results tree publishes a report and exits 0 **[verified]**
`scripts/benchmark_collect.py:25,456,558`

`SIZES=[...] or _SIZE_ORDER[:3]` fabricates tier names; cells count only if their log
exists, so a never-run cell leaves both numerator and denominator. Run on an empty
tree it wrote REPORT.md, printed "0/0", exit 0 — while hardcoded prose still asserts
"6/16→9/16→12/16". Live now: `results/benchmark_sml/` holds only `gen_logs/`.

### B5. Other confirmed silent-green paths

| # | Location | Defect |
|---|---|---|
| B5a | `train_per_tech_8cells.sh:31-35` | no `--exp-name` + `--overwrite` writes into **production** slots, and omits the clean-recipe flags. CLAUDE.md:226 advertises it as a convenience sweep |
| B5b | `complex_sweep.py:575-604` | sha256 pin is gitignored, self-seeding, warn-only; no manifest exists on disk; `strict=False` everywhere |
| B5c | `verify_complex_opamp.py:151` | no minimum-gain guard — `ng_gain=0.30, dn_gain=0.31` PASSES at 3.3%, certifying an opamp with no gain. The sweep harness has `OPAMP_MIN_GAIN=5.0`; the ship gate doesn't |
| B5d | `verify_nn_dc_tran.py:445-450` | a PINNED-but-absent checkpoint becomes a printed SKIP with no result row, bypassing the V6.6.6 parser raise (model never instantiated) |
| B5e | `verify_nn_dc_tran.py:1986` | gate PRINTS a checkpoint it did not score — `get_available_checkpoints()` is an existence sentinel returning `tsmc5_dn_medium_nmos` for every tech while the parser resolves per-tech `large` |
| B5f | `gate_matrix_iso.sh:111-115` | numerator greps all-time SUMMARY, denominator is this run → "8/1"; `.cell_*` never invalidated, so a SIGKILLed worker's prior PASS folds back in |
| B5g | `verify_subckt.py` | the three advertised V6.12.0 loud errors (unknown subckt, port-count, recursion) have **zero** tests — delete all three raises and it still reports 8/8 |
| B5h | `diag_l72_complex_control.py:151,221` | NaN takes the reassuring branch: a ring that never oscillates prints "L72 matches NGSPICE". Per MEMORY.md these verdicts routed whole campaigns |
| B5i | `verify_nn_dc_tran.py:3141-3162` | `--idvds-diagnostic` is a green run with **zero** NGSPICE comparison; `:3154` sets `passed=True` on an empty mask |
| B5j | `diag_l72_switchcap_control.py:116` | prints ">> the solver is faithful" unconditionally |
| B5k | `gate_grid.py:15`, `recipe_retest_collect.py:59` | `NCELL`/`NGATES` hardcoded 20 after TSMC6 joined; live run gives clean 13/**20**, crit30f 14/**20** vs documented /16 |
| B5l | all complex gates | no expected-tech-count assertion; `--tech TSMC5,TSMC7X` reports 1/1 → exit 0 |
| B5m | `benchmark_run_tests.sh:55`, `recipe_eval.sh:64` | `===BENCH_DONE no-ckpt===` is a permanent poison pill — resume check precedes the checkpoint check |
| B5n | `verify_nn_lifted_source_dc.py:219` | `--techs TSMC4` → "0/0 configs PASSED", exit 0 |

### B6. Thread pinning — consistent, with one robustness hole **[verified]**
CLAUDE.md's V6.6.6 claim is real; the pin sits in the shared import
(`complex.py:43-48`, `complex_ac.py:39-44`) so all complex/AC gates inherit it.

Two gaps:
- **`verify_nn_dc_tran.py` is not pinned at all** — no `set_num_threads`, no env var
  in its docstring. This is the file gating the high-gain VTC trip, i.e. exactly the
  ±1% OMP-flip surface. Its green is not reproducible.
- An **empty** `PYCIRCUITSIM_TORCH_THREADS` raises ValueError, swallowed by
  `except (ImportError, ValueError)` → **96 threads**. (`EMPTY -> 96`, `UNSET -> 1`.)
  That is the multithreaded-GEMM condition the V6.6.4 memory root-causes as the
  opamp/SRAM OMP coin-flip, reachable via `export VAR=$UNSET_VAR`.

---

## P2 — Correctness defects

### C1. `+` continuation lines silently discarded on every line type except `.model` **[verified]**
`pycircuitsim/parser.py:421-451`

`continued_line` is primed only for `.model`; all other lines are appended
immediately, so a following `+` fragment accumulates into an empty buffer, is flushed
as a space-prefixed orphan, and `parse_line` drops it with no error.

Reproduced end-to-end through `main.py`:

```spice
X1 in mid rdiv
+ rval=9k
  -> V(mid) = 0.500000    (correct answer with rval=9k is 0.100 V)
  -> exit code 0, no warning
```

Four silent-wrong outcomes: lost X-instance params, lost `.ic` assignments, lost
MOSFET geometry, and **lost `AC=` stimulus — a full Bode sweep driven by a zero
source.** No shipped deck uses continuations, so gates are unaffected; but `+` is
standard HSPICE.

### C2. NaN autograd Jacobian from the softplus voltage clamp **[verified]**
`pycircuitsim/models/mosfet_nn.py:285-291`

`torch.where` evaluates both branches, so `log1p(exp(bx))` overflows in fp32 and
`0·inf = NaN` poisons the backward pass. The **forward value stays correct** — only
the gradient is NaN, which is what makes it silent.

Thresholds from shipped norm stats: NaN above Vd 6.44 V (tsmc5), 6.15 V (tsmc7),
7.92 V (tsmc12/16). A Newton excursion past that returns NaN conductances stamped
straight into the MNA — and per A2, `spsolve` then returns NaN silently.
Inherited by LEVEL=73/74/75. **Fix:** `F.softplus(bx, beta=1)/beta`.

### C3. Unit suffix `m` means mega, not milli **[verified]**
`pycircuitsim/parser.py:333-334`

```
1m   -> 1000000.0      (SPICE/NGSPICE: 0.001)   — 10^9 discrepancy
1meg -> ValueError                              — the correct SPICE spelling is rejected
1mil -> ValueError
```

Since NGSPICE is ground truth for every gate, a deck pair using `m` diverges silently
on the PyCircuitSim side. Contradicts the "basic HSPICE netlist compatibility" core
principle.

### C4. Capacitor trapezoidal history never seeded after the BE first step **[verified]**
`pycircuitsim/models/passive.py:727-729`

`update_voltage` guards the `_i_prev` recursion with `_use_trapezoidal`, but step 1
runs BE with that flag False, so `_i_prev` stays at its constructor `0.0`. Step 2
then starts trapezoidal with `I_prev = 0` instead of the true BE current.

MOSFET charges do *not* have this bug (`terminal_currents` computed every step); the
linear `Capacitor` is the odd one out. On an RC step: max error **47.07 mV** as
shipped vs **4.25 mV** with a correct handoff — an 11× degradation that shifts the
whole trajectory, not a startup blip.

Latent for standard decks (DC-seeded, PULSE `td>0` ⇒ `dv/dt≈0` at step 1, so
`I_prev=0` is accidentally correct), which is why the inverter gate still reads 0.19%.
**Live for any deck not in steady state at t=0** — i.e. `uic` and `.ic` decks, the
documented V6.12 switched-cap use case.

### C5. `.dc` sweep drops the endpoint **[verified — but masked]**
`pycircuitsim/simulation.py:405-415`

Float accumulation in `while current_value <= stop`. Verified: `.dc Vin 0 1.0 0.01`
yields **100 points ending at 0.99** where NGSPICE gives 101 ending at 1.0. Every
`.dc` line in `examples/` is affected.

**Impact is masked.** Both DC harnesses interpolate onto the NGSPICE grid *and* clip
with `min()` on the stop bound (`bsimcmg_dc.py:440-449`, `nn_sweep.py:99-107`), so the
extra NGSPICE sample is dropped rather than compared against a clamped extrapolation.
**No reported NRMSE in the project is misaligned.** What is lost is coverage: the
final step of every sweep is never validated — for the inverter VTC that is the upper
rail, where V(out) should be hard at ground.

### C6. Other confirmed defects

| # | Location | Defect |
|---|---|---|
| C6a | `solver.py:2081-2118` | final transient sample solved at the wrong time when `t_stop/dt` is fractional — recorded time is clamped, `sub_time` is not. Integer-ratio decks are clean, so b8d77f5's matrix could not have caught it |
| C6b | `solver.py:2117-2226` | dt-halving retry advances simulated time by the *full* step: `sub_time` computed outside the retry loop, never rewound. Error is permanent and compounds per halving |
| C6c | `solver.py:834-847` | DC convergence tested on the **damped** delta (`dv = damping·|Newton step|`); transient does it correctly. At damping 0.1 this is a silent 10× relaxation of RELTOL |
| C6d | `solver.py:938-961`, `:1718-1727` | oscillation-averaged acceptance has **no residual gate for LEVEL=72** — both short-circuit to True, promoting a non-converged NR to converged on the ground-truth model |
| C6e | `solver.py:2410-2415` | AC linearizes about a possibly non-converged OP and never checks; writes a full CSV and Bode plot with no diagnostic |
| C6f | `solver.py:1781-1794` | BDF-2 uses **uniform-step** coefficients under variable dt (reachable via C6b's halving path without opt-in) |
| C6g | `simulation.py:798-807` | AC `dec`/`oct` point count off by one vs NGSPICE (`N·log10(f2/f1)+1`). **CLI-only** — gates build their own correct grid via `verify_ac.py:57-62` |
| C6h | `parser.py:1300` | duplicate `X` instance names silently **merge** the two instances' internal nodes (NGSPICE rejects duplicates) |
| C6i | `parser.py:1371` | ground globality is case-sensitive — lowercase `gnd` inside a subckt becomes a floating node tied only to GMIN |
| C6j | `parser.py:1123-1126` | duplicate `.model` names silently overwrite **retroactively** (pre-pass resolution), changing polarity and LEVEL for devices written above |
| C6k | `parser.py:541`, `simulation.py:325-327` | unknown/typo'd directives silently ignored; `.tarn 1p 10p` degrades to an operating point with exit 0. `.temp` silently ignored |
| C6l | `parser.py:115-116` | env pin whose stem ends in the opposite polarity is silently ignored → falls through to production `large`. Not currently reachable from `scripts/` (all 30 sites audited) |
| C6m | `parser.py:303-309` | UNKNOWN-tech-code warning only covers the universal scope, so per-tech scopes silently use the UNKNOWN embedding slot (`TECH=tsmc5 VT=hvt` → `tech_code=4`) |
| C6n | `trainer.py:157-166` | LEVEL=74 selects checkpoints on **teacher-forced** loss while deployment is free-running; measured `gds` **33% worse** under AR |
| C6o | `loo_labels.py:183-189` | label sidecar cache validated by **row count only** — no content hash. Rule 1 invites regeneration, so this sits on the expected workflow |
| C6p | `loo_labels.py:50,157` | tsmc6/tsmc7 fingerprints collide **108/108**, so the miss-guard cannot fire and tsmc6 rows silently code as tsmc7 |
| C6q | `config.py:107-110` | `tech_scope_vocab_size("universal")` returns 18 while `NUM_TOTAL_CODES` is 25 — a universal run over tsmc6 (22-24) or ASAP7 (18-21) indexes `nn.Embedding(18)` out of range |
| C6r | `tabpfn.py:514-522` | `_ctx_cache` not invalidated by `load_state_dict`/EMA in-place writes; measured drift 4.0e-2. Safe today by ordering luck only |
| C6s | `solver.py:314/334` | gmb Jacobian/RHS inconsistency below the 1e-12 cutoff (negligible magnitude, but an asymmetry in the routine Rule 3 governs) |
| C6t | `mosfet_cmg.py:230` | `abs(g_ds)` violates Rule 4 textually; measured inert today (PyCMG gds already positive) |

---

## P3 — Methodology

### D1. TSMC6 is not an independent technology — it is TSMC7 relabelled **[verified]**

Chain of evidence, each step independently confirmed:

1. **Datasets bit-identical.** `tsmc6_{nmos,pmos}.npz` vs `tsmc7_*`: `inputs`,
   `geometry`, `outputs`, `sample_class` all `array_equal=True` (1,816,830 NMOS /
   2,187,292 PMOS rows). Only `meta_tech_name` differs.
2. **Not a caching bug.** Raw PDKs genuinely differ (4,090,118 B vs 4,136,044 B,
   different md5); resolved modelcards differ by 160 lines.
3. **The differences are invisible to the model.** Every differing key —
   `tmi_ver_lod`, `tmi_ver_isocpode`, `sfxmin`, `samax_c`, `wodx5akvth0` — has
   **zero occurrences** in the BSIM-CMG Verilog-A source, while genuine parameters
   (`toxp` 12, `phig` 2) are present. They are TSMC TMI-proprietary extensions the
   open BSIM-CMG does not implement.
4. **Confirmed end-to-end.** Two LEVEL=72 Id-Vgs sweeps, `TECH=tsmc6` vs `tsmc7`,
   identical geometry — every drain current matches to the last printed digit
   (3.096745e-09 … 5.195952e-04).

**Invalidates:** V6.9.0's "TSMC6 = 6th tech, L72 9/9 DC + 14/14 tran"; the V6.11.0
campaign's TSMC6 training (22 checkpoints, ~12 h re-gate); and specifically the
recorded claim that *"BSIM-AR is the only family to bank the tsmc6 opamp (9.83%)
while tsmc7-opamp is the universal ceiling"* — same input data, opposite outcomes is
the documented NR-basin coin-flip, **not** a fidelity difference between families.
The "6 techs / 16-gate matrix" denominators carry a duplicate.

This is a documentation and interpretation failure, not a code bug — the pipeline did
what it was told. Correct action: record TSMC6 ≡ TSMC7-under-BSIM-CMG and stop
counting it as an independent data point.

### D2. Validation split cannot measure what the gates measure **[verified]**
`external_compact_models/bsimar/data/dataset.py:198-204`

A uniform random row permutation over a **dense per-bin lattice** — no grouping or
stratification by bin, geometry, temperature, variant, or `sample_class`. Every
(variant, L, NFIN, T) bin appears in train, val *and* test.

Measured within one bin (tsmc5 NMOS, VDD=0.65):
- grid pitch 44.8 mV, jitter σ 32.5 mV
- **median val→train nearest-neighbour distance 28.4 mV** — below the pitch
- **10.75%** of val rows have a train neighbour within 10 mV
- plus 0.18% exact duplicate rows straddling the split (overlapping deterministic
  overlays sharing endpoints)

So val/test NRMSE measures *jitter-scale interpolation inside already-seen bins* and
is structurally incapable of measuring generalization across L, NFIN, T or variant.

**Compounded by D3.** This is a plausible mechanism for the project's recurring
"device fidelity excellent, circuit gates fail" pattern, and for ~10 campaigns of
recipe search that moved gate counts without ever moving the underlying measurement.

**Fix direction:** `GroupKFold` on bin id, or minimally a held-out-L report alongside
the existing row split.

### D3. Benchmark L=16nm is off the training grid for tsmc5/6/7 **[verified]**
`tests/common/complex.py:82`

The comment asserts L=16nm is pinned "so the model interpolates rather than
extrapolates". It is absent from the NMOS training grid for three of five techs:

| tech | NMOS training L (nm) | bench L=16nm |
|---|---|---|
| tsmc5 | 6, 20, 36, 54, 86 | **absent** |
| tsmc6/7 | 8, 11, 20, 36, 72, 120 | **absent** |
| tsmc12/16 | **16**, 20, 36, 72, 120 | present |

The three off-grid techs are exactly those owning every persistent gate failure in
MEMORY.md (tsmc5 ring, tsmc5 switchcap, tsmc7 opamp), while the two on-grid techs
pass. Correlational, not causal — but the cheapest untested hypothesis available, and
D2 guarantees the error it would produce is never measured.

### D4. Other data-pipeline findings

| # | Location | Defect |
|---|---|---|
| D4a | `nn_config.py:170-204` | non-integer NFIN sample points (12.001, 20.888, 24.888) leak from PDK bin edges — ~17-20% of a 5-6 point axis spent on non-manufacturable devices. CLAUDE.md's own `NFIN=10` example is off-grid for every tech |
| D4b | `nn_generate.py:861,520` | `find_threshold` runs a full peak-gm sweep per bin, then discards the result — pure waste on every regeneration |
| D4c | `nn_generate.py:377-379` | Gaussian jitter clipped rather than resampled, piling probability mass on the box faces at exactly the Vds=0/cutoff boundary taught separately as a class |
| D4d | `generate_nn_data.py:272` | per-tech generation reuses `seed=42` with `counter` restarting at 0, so tsmc5 bin *i* and tsmc12 bin *i* draw identical jitter. Harmless while files stay separate; becomes correlated noise under `uni_concat_npz.py` |

### D5. Verified clean (recorded so it is not re-audited)

- **`normalize.py`** — flagged as highest-risk, came back clean. Round-trip max
  relative error 6.31e-06 (z-score) / 3.80e-06 (asinh) over 200k real rows;
  train-only normalizer fit (no leakage); odd `arcsinh` round-trips PMOS negatives
  exactly; zero-variance guards deliberate; `_norm.npz` keys consistent between
  training and inference (the V6.8 qg-as-id fix is in place).
- **Units consistent end-to-end** — id in A, charges in C, caps in F, L in m, T in K.
  No A/µA or m/nm mismatch.
- **DirectNet is C∞, not merely C1** — `nn.SiLU` throughout; zero
  ReLU/BatchNorm/Dropout in the module tree. Autograd vs float64 FD over 2400 checks:
  median rel err 7.0e-10, **0 exceeding 1e-3**. Activations are exonerated. The real
  discontinuity is downstream in `_apply_vds_correction` at Vds≈−0.012 V, where the
  wrong-sign clamp hard-zeros `id`/`gm`/`gmb` (8.1e5× spike in the second difference).
- **EMA saves genuinely hold EMA weights** (`trainer.py:503`), buffers included.
- **Rule 16 honored** — `unknown_code_id` derived from `num_tech_codes` in all three
  families and all trainer entry points; no hardcoded 17 on any per-tech path.
- **Rule 12 enforced** simulator-side regardless of predicted `qs`.
- **Rule 3 VCCS stamps complete** — all four entries for `g_m`/`g_mb` including
  ground-terminal reductions; `gds` floored with `max`, never `abs`; AC stamp mirrors
  DC faithfully and its 3×3 rows/columns each sum to zero.
- **Autograd hygiene** — `create_graph=False` throughout, `autograd.grad` never
  touches `.grad`; no cross-timestep graph or gradient accumulation.
- **b8d77f5 is correct** — no off-by-one introduced (501 unique points ending exactly
  at 5e-9). The defect it left behind (C6a) is in the `ceil` branch it deliberately
  preserved.
- **`.subckt` core flattening is correct** — node prefixing, nested X-in-X, param
  chaining, `{expr}` arithmetic, `.ic` rewriting, PULSE passthrough, port-count
  validation, `.ends` mismatch, recursion guard all verified working. Mid-line `*`
  correctly parses as multiplication. The defects found (C6h, C6i) are in the
  surrounding envelope, which is why the 8/8 gate and these findings are consistent.
- **SRAM butterfly is genuinely NGSPICE-backed and gated** — best-constructed of the
  four complex gates; CLAUDE.md's "+NGSPICE-NRMSE tracking" claim is accurate.
- **`verify_ac.py`'s analytic RC is ANDed with NGSPICE**, not substituted; the
  `vp()`-radians trap is structurally avoided via `wrdata` real/imag.

---

## P4 — Documentation defects

| # | Claim | Reality |
|---|---|---|
| E1 | CLAUDE.md "Load-bearing code": `_MonotoneVgResidual` must stay because "on-disk checkpoints carry `mono.*` state_dict keys and fail to load without it" | **0 of 262 checkpoints carry `mono.*`; 0 carry `core.*`** — V6.7.1 pruned them all. Keep the code (`recipe_train.sh` can regenerate), but the justification is false and a false reason is worse than none |
| E2 | `solver.py:60-86` docstring asserts the residual rejects "a stalled iterate with tiny Δv yet a large residual" | It cannot — see B2 |
| E3 | `solver.py:725-728` comment: branch slots "only scale the residual, not the descent test" | Factually wrong — see B2 |
| E4 | `solver.py:209-256` docstrings say "LEVEL=73 DirectNet" / "every DirectNet MOSFET" | Dispatches on `_nn_mosfet_types()`, which includes BSIMAR since V6.8 and PFN since V6.9 |
| E5 | `solver.py:1-28` module docstring | Describes only Backward Euler and two solvers; there are three of each |
| E6 | `parser.py:79-84` `_resolve_nn_checkpoint` docstring | Documents "MODEL_PATH > v4 universal > per-tech > bare"; actual order is env-pin > MODEL_PATH > per-tech preempt > universal > bare |
| E7 | `parser.py:41-46`, `:13`, `:734` | Suffix list omits t/g/m/f and the mega-vs-milli hazard; docstrings still advertise `W=<w>`, which is parsed and forwarded to no constructor |
| E8 | `tests/references/ngspice_inverter_tran.cir:2` | Claims "Used by tests/verify_bsimcmg_tran.py" — the harness generates its deck inline; the file `.include`s a `combined_baked.lib` that exists nowhere |
| E9 | All four complex-gate docstrings + CLAUDE.md say 16 gates | `BENCH_TECHS` is now 5 techs = **20** |
| E10 | `solver.py:977-1009` | ~30-line dead-end narrative inside `_solve_newton`, contra CLAUDE.md's own "dead-ends live in CHANGELOG" convention; references `results/v6_4_6/phase1_force_ic_recovery.md`, since pruned |

---

## Dead code

- **The entire universal fallback cascade in `parser.py:186-283`** — all 16 non-per-tech
  stems it probes have **zero** files on disk, making `_select()` and its
  `phys_best_metric`/`_best.phys.pt` arbitration unreachable. ~90 lines. Removing it
  turns "no checkpoint" into an explicit failure instead of a resolve-to-nonexistent-path.
- `direct_net.py:39-101` `_MonotoneVgResidual`, `:104-285` `_EKVCore` and
  `mosfet_directnet.py:55-72` auto-detect — unreachable (E1), but `recipe_train.sh`
  can regenerate them. **Keep.**
- `cli/train.py:46-63` `LOSS_PRESETS` e1/e2/e3 — never exercised.
- `mosfet_nn.py:210-212` E2 subset branch — all 262 norm files declare 13 columns.
- `normalize.py:267-285` `ZScoreNormalizer` — `_NORM_MODE = "asinh"` hardcoded, no CLI
  flag; 262/262 files are asinh.
- `trainer.py:449-453` `swa_mode="swa"` — no script passes it.
- `transformer.py:131` `token_type_emb` row 11 — never indexed.
- `diag_nn_jacobian_consistency.py:277` — uses universal vocab instead of
  `local_variant_code`, so it raises on every production checkpoint. Correctly not
  wired into any gate; simply dead.
- Unused locals: `parser.py:411,432,444` (`in_model`), `:753` (`model`), `:757` (`W`);
  `passive.py:371` (`t_in_period`); `bni_mae.py:347` (`retain`);
  `tabpfn.py:607` (`device` param); `solver.py:384` (`tolerance`), `:391`
  (`damping_factor`, overwritten at `:596`), `:1348` (`nr_tolerance`);
  `simulation.py:145` (`skip_header`, passed at `:483` expecting effect).

## Performance

`_mosfet_types()` / `_pmos_types()` / `_nn_mosfet_types()` (`solver.py:117-201`)
rebuild their type tuples and re-run four `try: import` blocks on **every call**:

```
_mosfet_types() build       : 112.5 us/call
_is_mosfet()                :  37.6 us/call  (vs ~2 us cached)
two dispatch loops, 10 comps: 147.7 us per NR iteration
```

`_has_nn_device` is called 3× per DC iteration and 4× per transient iteration, each
time importing and scanning all components. Module-level memoization is a one-line
change and should be worth double-digit percentages on the CPU-pinned gate matrix.

---

## Suggested fix order

1. **A1** `base.py:254` — check `returncode`, unlink CSV before running. One function;
   closes the project-wide critical.
2. **A2** `solver.py:45-49` — detect singular/non-finite `spsolve` return and raise.
   Turns three silent-green paths loud at once; every surrounding
   `except LinAlgError` handler is already written and waiting.
3. **B2** `solver.py:1128-1131` — populate branch-current slots; scale the threshold on
   current rows only. Restores the SRAM residual gate.
4. **B1** `solver.py:650/964` — reset `final_converged` per source step. Own commit +
   full gate re-run, since it arms recovery paths that have never run.
5. **B6** reject empty `PYCIRCUITSIM_TORCH_THREADS`; add the pin to `verify_nn_dc_tran.py`.
6. **C2** `mosfet_nn.py:285-291` — `F.softplus(bx, beta=1)/beta`.
7. **A3** gds negation — **must ship together with a floor change; neither works
   alone** (sign alone: bit-identical; floor alone: 8/10 → 5/10). Measured
   sign+k=0: device AC 8/10 → 10/10, opamp OP un-railed, DC VTC exactly unchanged.
   Before shipping: scan k rather than using 0.0 (some guard is still needed —
   raw gds is genuinely negative at ~3–8% of conducting points), then run the full
   gate matrix including ring-osc/SRAM/switchcap, which are historically fragile to
   this class of change.
8. **B3/B4** `set -euo pipefail` + real status aggregation; make the collector fail on
   an empty tree.
9. **B5c** give `verify_complex_opamp.py` the `OPAMP_MIN_GAIN` guard the sweep harness
   already has.
10. **D1/D2/D3** methodology — record TSMC6 ≡ TSMC7; add a grouped/held-out-L split;
    test the L=16nm hypothesis.

## Open experiments

- **k-scan for `_GDS_FLOOR_K`** with the sign fixed — find where the AC improvement
  saturates while still guarding the ~3–8% of conducting points where the correctly-
  signed gds is genuinely negative. Then full gate matrix. (A3-measured settles the
  direction; it does not settle the value.)
- **L=16nm on-grid for tsmc5/6/7** (D3) — one-line geometry change, regenerate,
  re-gate.
- **Held-out-bin validation** (D2) — does any recipe ranking survive a grouped split?
- **Does the un-railing generalize?** (A3-measured) — opamp OP went railed →
  un-railed on both techs tested. If that holds across techs it reframes the opamp
  railing history as partly a gds-floor artifact rather than a value-surface limit.

## Incidental

`external_compact_models/bsimar/config.py:33` does `sys.path.insert(0, PYCMG_DIR)`
unconditionally, so `PyCMG/tests/` **shadows the repo's own `tests/` package** for
any script importing `bsimar` before `tests`. Verified: after `import bsimar.config`,
`import tests` resolves to `PyCMG/tests/__init__.py`. `sys.path.append` avoids it.
