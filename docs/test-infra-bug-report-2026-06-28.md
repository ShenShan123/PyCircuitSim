# Test-Infrastructure Bug Report — 2026-06-28

**Scope:** `tests/` (the verification + diagnostic harness), with focus on the
shared infra `tests/common/{base,nn,nn_sweep,complex,complex_sweep,complex_ac,
bsimcmg_dc,bsimcmg_tran}.py` and the `verify_*` gates that consume it.

**Mandate checked:** the infra should cover *different circuit topologies × geometry
combos × VTH types × technology nodes*, across *different analyses* (`.op/.dc/.tran/
.ac`), always gated against NGSPICE BSIM-CMG (LEVEL=72) ground truth.

**Method:** full read of the shared infra; three parallel audit passes over the
`verify_*` scripts; then **direct verification** of every headline finding (ran the
canary, probed the parser's checkpoint resolver, inspected on-disk checkpoints,
grepped flag set/read sites). Nothing below is speculative — each carries
reproduced evidence.

> Per request, **no code was changed**. This is the report only. Fixes are proposed
> but not applied.

---

## 0. Verdict

Coverage *breadth* is good (see §1). But there are **two correctness defects that
make a green run untrustworthy** and several that let a real failure score PASS:

| ID | Sev | One-line |
|----|-----|----------|
| **B1** | **CRITICAL** | Multi-tech device-DC / PMOS-DC / NMOS-transient gates evaluate the **wrong tech's checkpoint** (tsmc5 for *all* techs) in a bare run — proven empirically. |
| **B2** | **HIGH** | The sweep↔ship-gate equivalence **canary is RED right now** (8/8 ring+switchcap failures): sweep `.tran` builders dropped `uic`. |
| **B3** | **HIGH** | SRAM single-point gate scores a tech **PASS when every NFIN corner errors** (`all([]) == True`). |
| **B4** | **HIGH (latent)** | A diverged inverter transient can score **PASS** on its matching prefix — the `_nr_partial` "fail-loud" flag is set but never read. |
| B5 | MED | SRAM verdict ignores NGSPICE ground truth (SNM + force_ic computed, never gated). |
| B6 | MED | `run_dc_tests`/`run_tran_tests` don't skip ASAP7 (out of scope); default `--tech` includes it → garbage rows + nonzero exit. |
| B7 | MED | SRAM sweep "baseline" ≠ ship-gate definition, and SRAM is the one circuit the canary doesn't cover. |
| B8 | MED | Canary's NGSPICE-side check is a no-op; it compares against hand-copied replicas, not the real ship-gate deck code. |
| B9 | LOW/MED | `partial` (truncated transient) flag not gated in switchcap/ring single-point gates. |
| B10 | LOW | All four single-point ship gates `return 0` unconditionally — exit code never reflects a failure. |
| B11 | LOW | Misc: duplicate `l`-sweep points; checkpoint-drift non-gating by default; nn_ac docstring drift; off-bin L/NFIN hard-gated despite "expected" framing. |

---

## 1. Coverage assessment (does the infra meet the mandate?)

| Axis | Covered | Notes |
|------|---------|-------|
| **Topologies** | inverter, common-source amp, 2-stage Miller opamp, 5-stage ring osc, 6T SRAM, switched-cap cell, passive RC | Good. |
| **Geometry** | L (per-tech values), NFIN (sym {3,5,10}), P/N fin ratio | Good — `complex_sweep._shared_variants`, `nn_sweep.build_dc_parametric`. |
| **VTH** | `vt_sym`, `vt_asym` (independent N/P) | Good, but **structurally thin on TSMC7** (only 1 usable VT after PDIBL2 pruning → asym-VT dim has zero witnesses; the driver does print a coverage note). |
| **Tech nodes** | TSMC5/7/12/16 (NN), ASAP7 (ground-truth only) | As intended (ASAP7 out of NN scope). |
| **Analyses** | `.op`, `.dc`, `.tran`, `.ac` | All present. AC excludes ring/SRAM by design (no stable amplifying OP). |

**The catch:** B1 means the *device-level* `.dc`/`.tran` multi-tech coverage is
**illusory in a bare run** — 3 of 4 techs silently exercise the tsmc5 model. The
matrix looks full; it is not actually testing the per-tech models it claims to.

---

## 2. CRITICAL

### B1 — Device DC / PMOS-DC / NMOS-transient gates pin ONE checkpoint (tsmc5) for every tech
**Files:** `tests/verify_nn_dc_tran.py`
- runners append `MODEL_PATH` unconditionally: `run_pycircuitsim_nn_nmos_dc` L643-644, `run_pycircuitsim_nn_pmos_dc` ~L947, `run_pycircuitsim_nn_nmos_tran` ~L782 — they **lack the `_cascade_handles_stem` guard** that the inverter runners have (L1098-1103, L1356-1361).
- callers pass the single NMOS/PMOS alias for *all* techs: `run_dc_tests` L1935-1943 (`checkpoints["directnet_v4"]`), `run_pmos_dc_tests` L2092-2100, `run_tran_tests` L2482-2488.
- the alias resolves to the **first existing fallback** (L462-476): `directnet_v4_nmos → tsmc5_dn_medium_nmos_best.pt` (v4/refac stems are deleted on disk).

**Mechanism:** an explicit `MODEL_PATH` bypasses the parser's per-tech preempt
cascade, and the scope is then read from the *filename* (`tsmc5_*`), not from
`TECH=`.

**Empirical proof** (parser resolver log, this session):
```
A) .model NMOS (LEVEL=73 TECH=tsmc12 VT=svt MODEL_PATH=.../tsmc5_dn_medium_nmos_best.pt)
   [NN-resolver] L73 Mn1 TECH=tsmc12 VT=svt -> tsmc5_dn_medium_nmos_best.pt (scope=tsmc5, tech_code=4)
B) .model NMOS (LEVEL=73 TECH=tsmc12 VT=svt)            # inverter-suite path
   [NN-resolver] L73 Mn1 TECH=tsmc12 VT=svt -> tsmc12_dn_medium_nmos_best.pt (scope=tsmc12, tech_code=0)
```
In (A) TSMC12/svt is served by the **tsmc5** net at `tech_code=4`, which is tsmc5's
**UNKNOWN** embedding slot (TSMC5 vocab = variants+1 = 5 → codes 0-4, 4 = UNKNOWN).
So the model is doubly wrong: wrong tech weights **and** the unknown-tech code.

**Impact:** a bare `python tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,
TSMC16` (the canonical CLAUDE.md invocation) reports DC/tran numbers for TSMC7/12/16
that never touched their own checkpoints. The published "DC 55/55, tran 64/64"
baseline is only valid when run single-tech with the per-tech
`PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}` env override (as the benchmark scripts
do). The gate can either spuriously FAIL or *coincidentally PASS* (tsmc5/tsmc7 are
similar steep low-VDD techs), hiding that the real model was never gated.

**Fix:** route the three device runners through `_cascade_handles_stem` exactly like
the inverter runners — pass `model_path=None` (omit `MODEL_PATH`) for
`tsmc{5,7,12,16}_dn_*` / `refac_dn_*` stems so the parser resolves per-tech.

---

## 3. HIGH

### B2 — Sweep↔ship-gate equivalence canary is currently RED (`uic` dropped)
**Files:** `tests/common/complex.py` L739 (`directnet_ringosc`), L780
(`directnet_switchcap`); guard `tests/verify_complex_sweep_canaries.py`.

The sweep builders emit `.tran {tstep} {tstop}` with **no `uic`**, while the
ship-gate templates (`examples/complex/ring_osc_5stage_directnet.sp:37`,
`switchcap_unitcell_directnet.sp:33`) carry `.tran … uic`.

**Reproduced** (ran `verify_complex_sweep_canaries.py`, exit 1):
```
[C2] TSMC5 ring  FAIL   prog-only:{'.tran 2p 1.2n'}  tmpl-only:{'.tran 2p 1.2n uic'}
[--] TSMC5 sc    FAIL   prog-only:{'.tran 5p 12n'}   tmpl-only:{'.tran 5p 12n uic'}
… (same for TSMC7/12/16)
RESULT: 8 CANARY FAILURE(S)
```
**Impact:** the guard whose sole job is "sweep baseline == authoritative ship-gate
deck" has been failing on every tech. It's functionally benign *today* (the sweep's
`run_directnet_transient` pins `.ic` nodes regardless of the parsed `uic` flag), but
a red canary means the guarantee is unmet and nobody is running it green — exactly
the blind spot it was built to prevent.
**Fix:** append `" uic"` to both sweep `.tran` builder lines (or make the templates
match) so the canary is green and the decks are byte-faithful.

### B3 — SRAM single-point gate scores PASS when every corner errors
**File:** `tests/verify_complex_sram_snm.py` L305-306, L364.
```python
all_positive = all(r.get("positive", False)
                   for r in corner_rows if "error" not in r)   # all([]) == True
...
n_pass += int(r["all_positive"])
```
Every errored NFIN corner is appended with an `"error"` key (L267, L276) and then
filtered out. If *all* corners error (DirectNet DC-sweep raised, or NGSPICE failed),
the generator is empty → `all([]) == True` → the tech is counted **PASS** with zero
ground-truth comparison performed. Partial errors silently drop the failed corners.
This is the canonical "exception swallowed into a PASS," and it propagates:
`scripts/benchmark_collect.py` reads the printed `all-positive: yes`.
**Fix:** `all_positive = bool(corner_rows) and all("error" not in r and r["positive"]
for r in corner_rows)`.

### B4 — Diverged inverter transient can score PASS (`_nr_partial` set but never read)
**File:** `tests/verify_nn_dc_tran.py` L1457 sets `out["_nr_partial"] = True` on a
mid-transient NR failure (with comment "so the test reports a numeric FAIL … instead
of an ERROR"). Grep confirms it is **never read anywhere in `tests/`**.
`run_inverter_tran_tests` (L2326+) and `compare_inverter_tran_waveforms` truncate the
comparison to `min(ref[-1], test[-1])`, so a transient that diverges *after* a
matching prefix (both railed before the first switching edge) yields `nrmse ≈ 0 →
PASS`. Same class lives in `tests/common/nn_sweep.py` `run_single_nn_inv` L367 (only
a `len < 3` guard, which the `last_step ≥ 2` recovery path never trips).
**Impact:** latent today (production checkpoints converge fully) but it will mask a
future regression that causes mid-transient divergence — i.e. it defeats the very
"fail loud" intent the flag was added for.
**Fix:** in both orchestrators, treat `_nr_partial` (or `test["time"][-1] < tstop −
eps`) as an automatic FAIL.

---

## 4. MEDIUM

### B5 — SRAM verdict never compares against ground truth
**File:** `tests/verify_complex_sram_snm.py`. The only quantity folded into the gate
is `positive = dn_min >= -1e-3` (DirectNet butterfly ≥ −1 mV). `snm_err` vs NGSPICE
(L288) and the `force_ic` probe (L300) are printed but **never** enter
`all_positive`/`n_pass`. A wildly inaccurate-but-non-negative lobe passes. This
contradicts CLAUDE.md, which calls force_ic "the authoritative force_ic single-point
ship gate," yet its result has no effect on the verdict. (Stale prose, too: the
`force_ic_probe` docstring still describes the old wl=ON "0/8 basin" outcome while
the netlist now defaults to wl=OFF.)
**Fix:** decide intent — either AND `snm_err ≤ tol` and/or `force_ic` into the pass,
or reconcile the CLAUDE.md "authoritative" claim.

### B6 — `run_dc_tests` / `run_tran_tests` don't skip out-of-vocab ASAP7
**File:** `tests/verify_nn_dc_tran.py`. The ASAP7 `tech_code_in_vocab` skip exists in
`run_pmos_dc_tests` (L2024), `run_inverter_vtc_tests` (L2175), inverter-tran (L2278),
and the diagnostics (L2743, L3017) — but **not** in `run_dc_tests` (L1829) or
`run_tran_tests` (L2408). The default `--tech` is `",".join(TECH_ORDER)` with
`TECH_ORDER = ["ASAP7","ASAP7_30nm","TSMC5",…]` (L241, L3102). So a bare invocation
runs ASAP7 NMOS DC+tran (out of scope, Rule 14), producing garbage/ERROR rows and a
nonzero exit.
**Fix:** add the same `tech_code_in_vocab(...)` skip to both, or drop ASAP7 from the
default tech list.

### B7 — SRAM sweep "baseline" ≠ ship-gate definition (and is uncovered by the canary)
**File:** `tests/common/complex_sweep.py` `run_single_sram` L509:
`passed = positive and fic_ok` (force_ic AND'd at `wl_frac==1.0`, checking only
NFIN=2). The single-point ship gate (B5) gates on positivity *only*, across NFIN
={2,5,10}. So the sweep baseline can FAIL where the "authoritative" ship gate passes
(a force_ic regression sinks the whole SRAM sweep via baseline-gating). SRAM is also
the one circuit the equivalence canary does **not** cover, so this divergence is
unguarded.
**Fix:** make the two definitions deliberately identical and add an SRAM canary.

### B8 — The canary is partly hollow
**File:** `tests/verify_complex_sweep_canaries.py`.
- NGSPICE-side check (L136-145) only asserts the body is "non-degenerate" (≥6 lines +
  has `.include`); it **never compares** the sweep NGSPICE deck against the
  single-point `run_ngspice_*` body, so ground-truth-side drift goes uncaught.
- The DirectNet check compares the sweep builders against **private hand-copied
  replicas** (`opamp/ring/sc_template_rewrite`, L51-84), not the real ship-gate deck
  functions (`verify_complex_opamp.run_directnet_opamp`, etc.). If the ship-gate
  rewrite logic drifts (note: it uses a regex, the replicas use `.replace`), the
  canary won't notice.
**Fix:** import and diff against the actual single-point deck-producing functions for
both sides.

---

## 5. LOW

### B9 — `partial` flag not gated in switchcap/ring single-point gates
`verify_complex_switchcap.py` L156 `passed = charge_ok and droop_ok` and
`verify_complex_ring_osc.py` L135 `passed = np.isfinite(per_err) and per_err <= …`
both **store** `partial` in the result dict but never gate on it. A transient that
diverges *after* the measurement window leaves the sample/hold reads on a clamped
`np.interp` tail and can still land in tolerance → PASS. Lower severity than B4
because a truncated ring usually loses crossings → NaN → FAIL.
**Fix:** `passed = … and not partial`.

### B10 — Single-point ship gates never set a failing exit code
All four `verify_complex_{opamp,ring_osc,sram_snm,switchcap}.py` `main()` end in
`return 0` unconditionally (opamp L188, ring L186, sram L366, switchcap L213).
Harmless today (consumers parse stdout PASS/FAIL), but a latent trap for any future
exit-code-based CI caller — the "ship gates" always exit success.
**Fix:** `return 0 if n_pass == n_total else 1`.

### B11 — Misc
- `complex_sweep._shared_variants` `l` dimension (L154-168) emits **duplicate
  geometry** points (`ln_{L}` at `L == l_pmos` == `lsym_{L}`, etc.) — wasted runs.
- Checkpoint-drift detection (`verify_checkpoint_pin`) only warns unless
  `--pin-strict` is passed; the opamp/SRAM amplify weight drift ~20×, so a silent
  swap is detected-but-tolerated by default.
- `verify_nn_ac.py` docstring (L22-25) says the bias is the "peak |dVout/dVin|"
  point, but the code (`find_ng_bias`/`find_nn_bias`) uses the mid-rail
  `argmin(|Vout − VDD/2|)`.
- `nn_sweep.build_dc_parametric` off-bin L/NFIN extrapolation points are documented
  as "expected, not a fault" yet are hard-gated identically to in-bin points — a
  model regression and an expected extrapolation miss are indistinguishable.

---

## 6. Files audited and found clean
`tests/common/base.py`, `tests/common/nn.py` (metrics), `tests/common/bsimcmg_dc.py`,
`tests/common/bsimcmg_tran.py`, `tests/verify_ac.py` + `tests/common/complex_ac.py`
(AC metric primitives — unity-gain / GBW / phase-margin / −3 dB corner / complex-ratio
phase / NaN guards all correct), `tests/verify_nn_lifted_source_dc.py`. The four
`complex.py` NGSPICE↔DirectNet deck *pairs* are the same experiment (VDD/rail/clock/
bias/`.ic`/geometry/caps/windows match; the historical 0.80 V clock-over-drive bug is
genuinely fixed by the `render_directnet_netlist` regex; baked cache key now includes
L). The opamp single-point gate is clean and is correctly surfacing its known-open
tsmc7 case.

---

## 7. Suggested fix order
1. **B1** (CRITICAL — restores honest per-tech device coverage).
2. **B3, B4** (silent false-pass paths).
3. **B2** (turn the canary green; it's a 2-line builder fix).
4. **B5–B8** (gate-definition correctness; close the canary holes).
5. **B9–B11** (hardening / cleanup).
