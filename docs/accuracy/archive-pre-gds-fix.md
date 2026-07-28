# Archive — frozen pre-gds-fix data tables (2026-07-05 … 2026-07-23)

> ⚠ **Every table in this file was measured with the `gds` sign bug present**
> (`methodology.md` §6). They are kept verbatim for provenance, and because most
> of the checkpoints they describe **no longer exist on disk** — the V6.6.x
> recipe matrix (208 checkpoints) and the V6.6.6 xl curriculum wave (72) were
> archived to
> `/data2/shenshan/v66x_v670_retired_ckpts_2026-07-05.tar.gz` and
> `/data2/shenshan/v6.5.9_production_specials.tar.gz`, so these numbers cannot
> be re-measured without restoring those archives.
>
> **How to read them.** Per §6's measured invariance:
>
> | axis | pre-fix table still valid? |
> |---|---|
> | device **DC** (Id-Vgs NRMSE/MRE/R²) | **yes** — DC is exactly invariant (bit-identical) |
> | **ring**, **SRAM**, **switchcap** cells | **yes** — not one such cell moved in the re-gate |
> | **opamp** cells | **no** — every gained cell in the re-gate was an opamp |
> | any **total** containing an opamp column | **no** — systematically pessimistic |
> | **AC** (device CS-amp, opamp open-loop) | **no** — the axis the fix moved most |
> | **transient** | partially — mean NRMSE improved 1.876 % → 1.512 % |
> | OMP **FLIP** classifications | **no** — every re-measured group is now flip-free |
>
> Current numbers live in `README.md` (scoreboard), `by-tech.md`, `by-scale.md`,
> `by-recipe.md`. The raw post-fix re-gates are `results/a3_regate/REPORT.md`,
> `results/a3_regate/OMP_REPORT.md`, `results/a3_regate_uni/REPORT.md` and
> `results/v710_regate/REPORT.md`.

---

## Register of claims retracted by the re-gates

| retracted claim | where it came from | what replaced it |
|---|---|---|
| "**tsmc7-opamp** is the universal ceiling cell for all three families — no capacity, tier, scope, seed or data recipe has ever reached it; only the V6.5.9 T3 differentiable-DC-solver fine-tune did." | DN §6.2, TF §4/§6, cross-family tables, CLAUDE.md | BSIM-AR passes it at **every** size and recipe (0.55–7.3 %); DirectNet passes it at `small`, `xl` and strictly at `crit15m@xl`. The gds floor was holding a railed OP. |
| "**PFN is the only flip-free family.**" | PFN §1/§10 | True when measured, overtaken: after the fix *every* family is flip-free on all 18 re-gated per-tech groups and all 8 universal stems. The shared cause was the wrong-signed Jacobian entry. |
| "**BSIM-AR beats DirectNet by one cell** (tsmc16-opamp)." | cross-family tables | Both families reach 16/16 strict. The matrix no longer separates them; inference cost (≈40×) and device-suite breadth do. |
| "BSIM-AR **capacity peaks at medium** (12→14→13)." | TF §3 | Clean BSIM-AR is 14/16 at *every* tier post-fix, failing the same two rings. The curve was largely a gds artifact. |
| "The **strict best is medium, not large**" (BSIM-AR). | TF §5 | It rested on an OMP FLIP that no longer exists. |
| "The **three-basin simultaneous hold (5+12+16) is the open 15/16 target**." | DN §6.2 | `crit15m@xl` holds all **four** opamp basins with no recipe change. |
| "`csob@large` is the documented complex-gate alternate." | DN alternates table | **Withdrawn** — post-fix it is 11/16 (the campaign's only regression) and now fails tsmc16-opamp, which production banks. Still the device/AC alternate. |
| "`crit10@xl` covers tsmc16-opamp." | DN alternates table | **Withdrawn** — post-fix it fails that exact cell (gain err 100 %). Superseded by `crit15m@xl`. |
| "**AC pass-rate peaks at SMALL**" — a dQ/dV pole property that wants the opposite capacity to DC fixed points. | DN §8 (V6.6.5), TF §7, PFN §7 | **Retracted.** DirectNet device CS-amp AC is 7/8 · 8/8 · 8/8 · 7/8 across small→xl in the V7.1.0 re-gate: saturated at every capacity. The pre-fix 5/12 · 4/12 · 4/12 · 4/12 had both the level and the shape wrong. `by-scale.md` §5. |
| "The opamp **open-loop AC** gate is 0/4 at every tier for every family." | DN §8/§12.1, TF §6, PFN §7 | **Falsified.** DirectNet `small` and BSIM-AR `small`/`medium` bank TSMC16; BSIM-AR `large` banks TSMC7. And part of the remaining denominator is unreachable by construction — `by-scale.md` §5's bias-resolution defect. |

---

# Part 0 — TSMC6, the V6.11.0 pre-fix run

The "before" half of the TSMC6 controlled repeat, recovered from commit
`a96112a`. Superseded by the V7.1.0 re-training and re-gate in
[`by-tech.md`](by-tech.md) §5, which is post-fix and covers all four scales for
all three families; these are kept because they are the only record of the
first run.

**DirectNet** — complex 4-cell, and device/inverter (pre-fix):

| size | complex | ring period_err% | opamp gain_err% | sram lobeNRMSE% | switchcap chg_err% |
|---|---|---|---|---|---|
| small | 1/4 | 5.94 ✗ | 10.33 ✗ | 3.66 ✓ | 2.34 ✗ (droop) |
| medium | 2/4 | 10.86 ✗ | rails ✗ | 3.31 ✓ | 2.81 ✓ |
| **large** | **3/4** | **4.82 ✓** | rails ✗ | 3.61 ✓ | 2.45 ✓ |
| xl | 2/4 | 14.31 ✗ | rails ✗ | 2.93 ✓ | 2.67 ✓ |

| size | NMOS DC nrmse/mre% | PMOS DC nrmse/mre% | inv VTC% | inv tran% | dev-AC |
|---|---|---|---|---|---|
| small | 3.88 / 17.47 | 0.91 / 5.39 | 2.62 | 1.20 | 2/3 |
| medium | 6.84 / 17.49 | 0.06 / 0.46 | 2.48 | 1.19 | 1/3 |
| large | 2.02 / 5.70 | 0.04 / 0.61 | 1.79 | 0.98 | 2/3 |
| xl | 6.69 / 17.31 | 0.04 / 0.62 | 1.19 | 0.78 | 1/3 |

**BSIM-AR** — complex 4-cell, and device/inverter (pre-fix):

| size | complex | ring period_err% | opamp gain_err% | sram lobeNRMSE% | switchcap chg_err% |
|---|---|---|---|---|---|
| small | 2/4 | 5.97 ✗ | 13.61 ✗ | 2.10 ✓ | 2.46 ✓ |
| **medium** | **3/4** | 7.41 ✗ | **9.83 ✓** | 2.19 ✓ | 2.62 ✓ |
| large | 2/4 | 11.19 ✗ | 12.78 ✗ | 3.22 ✓ | 2.64 ✓ |
| xl | 2/4 | 12.55 ✗ | 10.13 ✗ | 2.21 ✓ | 2.73 ✓ |

| size | NMOS DC nrmse/mre% | PMOS DC nrmse/mre% | inv VTC% | inv tran% |
|---|---|---|---|---|
| small | 3.37 / 11.32 | 0.56 / 2.29 | 1.48 | 1.21 |
| medium | 4.07 / 11.71 | 0.16 / 0.85 | 2.97 | 1.15 |
| large | 4.77 / 12.54 | 0.10 / 0.55 | 2.35 | 1.18 |
| xl | 6.41 / 16.79 | 0.06 / 0.15 | 1.78 | 1.00 |

**PFN** — complex 4-cell, and device/inverter (pre-fix; the V6.11.0 run had no
xl tier, which V7.1.0's `("tabpfn","xl")` preset closes):

| size | complex | ring period_err% | opamp gain_err% | sram lobeNRMSE% | switchcap chg_err% |
|---|---|---|---|---|---|
| small | 2/4 | 8.22 ✗ | rails ✗ | 2.29 ✓ | 2.94 ✓ |
| medium | 2/4 | 9.93 ✗ | rails ✗ | 1.75 ✓ | 2.94 ✓ |
| large | 2/4 | 12.38 ✗ | rails ✗ | 4.92 ✓ | 2.85 ✓ |
| xl | *(V7.1.0, training)* | | | | |

| size | NMOS DC nrmse/mre% | PMOS DC nrmse/mre% | inv VTC% | inv tran% | dev-AC |
|---|---|---|---|---|---|
| small | 3.57 / 10.24 | 0.04 / 0.40 | 1.07 | 1.15 | 2/3 |
| medium | 3.80 / 8.01 | 0.05 / 0.59 | 1.06 | 0.97 | 1/3 |
| large | 4.75 / 11.22 | 0.03 / 0.64 | 1.06 | 1.17 | 2/3 |

PFN's inverter VTC (~1.06 %) is the tightest of the three families on this data.

---

# Part A — DirectNet recipe/size data tables (frozen 2026-07-05)

Collector-generated tables, carried verbatim from the retired
`docs/V6.6.6-accuracy-report.md` (which had itself merged them from
`results/recipe_bench/ACCURACY_REPORT.md`, now a regeneration stub). Content
untouched.

- **Appendix A** — isolated re-test, `large` tier (23 recipes, V6.6.3 methodology).
- **Appendix B** — isolated re-test, `xl` tier (22 recipes).
- **Appendix C** — the V6.6.5 13-recipe × 4-size matrix, incl. device + AC suites.

## Appendix A — large-tier isolated re-test (23 recipes)

Every on-disk uniform recipe at the `large` tier, re-gated on the authoritative `verify_complex_*` gates (CPU-pinned, isolated dirs, NGSPICE BSIM-CMG ground truth). Accuracy-first per plan §5: continuous metrics lead, X/16 is derived. `clean` = production control. Conclusions + production recommendation: `by-recipe.md` and `DirectNet-L73-accuracy.md`.

### Derived summary — pass counts

| Recipe | single-run OMP=1 | strict all-OMP (opamp/ring ∈ {1,2,4}) |
|---|---|---|
| clean | 13/16 | 12/16 |
| invtripft | 12/16 | 12/16 |
| invtrip | 11/16 | 10/16 |
| cor | 11/16 | 10/16 |
| corft | 9/16 | 9/16 |
| corrft | 12/16 | 10/16 |
| corroft | 13/16 | 13/16 |
| corro15 | 13/16 | 12/16 |
| crit10 | 14/16 | 13/16 |
| crit15 | 14/16 | 13/16 |
| crit15m | 12/16 | 12/16 |
| crit15h | 13/16 | 13/16 |
| crit20 | 13/16 | 13/16 |
| crit30 | 14/16 | 14/16 |
| csob | 12/16 | 12/16 |
| cs7 | 11/16 | 11/16 |
| csobekv | 10/16 | 9/16 |
| ekv | 10/16 | 9/16 |
| sob | 5/16 | 5/16 |
| s7 | 11/16 | 11/16 |
| s17 | 11/16 | 11/16 |
| s123 | 10/16 | 10/16 |
| crit30a1 | 13/16 | 13/16 |
| crit30f | 14/16 | 14/16 |
| csobcrit | 13/16 | 13/16 |

### ring_osc — period_err % (gate ≤5)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 12.66 FAIL | 12.46 FAIL | 13.49 FAIL | 5.15 FAIL | 4.73 PASS | 4.61 PASS | 4.04 PASS | 4.03 PASS | 4.04 PASS | 4.04 PASS | 4.04 PASS | 4.04 PASS | 4.04 PASS | 4.04 PASS | 10.31 FAIL | 11.57 FAIL | 5.80 FAIL | 6.10 FAIL | 16.83 FAIL | 12.64 FAIL | 9.64 FAIL | 11.83 FAIL | 4.04 PASS | 4.04 PASS | 4.03 PASS |
| tsmc7 | 4.82 PASS | 7.32 FAIL | 10.40 FAIL | 2.88 PASS | 2.87 PASS | 2.87 PASS | 2.42 PASS | 2.41 PASS | 2.39 PASS | 2.41 PASS | 2.64 PASS | 2.40 PASS | 2.42 PASS | 2.40 PASS | 5.09 FAIL | 6.65 FAIL | 4.51 PASS | 6.70 FAIL | 10.13 FAIL | 7.15 FAIL | 8.69 FAIL | 7.15 FAIL | 2.42 PASS | 2.40 PASS | 2.44 PASS |
| tsmc12 | 4.04 PASS | 4.05 PASS | 4.02 PASS | 3.85 PASS | 3.84 PASS | 3.84 PASS | 3.77 PASS | 3.76 PASS | 3.50 PASS | 3.16 PASS | 3.42 PASS | 3.49 PASS | 2.68 PASS | 2.68 PASS | 2.12 PASS | 2.15 PASS | 2.16 PASS | 2.24 PASS | 5.68 FAIL | 4.19 PASS | 2.83 PASS | 3.38 PASS | 3.77 PASS | 2.68 PASS | 2.67 PASS |
| tsmc16 | 2.59 PASS | 3.40 PASS | 4.01 PASS | 4.30 PASS | 4.00 PASS | 4.00 PASS | 3.46 PASS | 2.77 PASS | 2.76 PASS | 2.77 PASS | 2.77 PASS | 2.77 PASS | 2.78 PASS | 2.90 PASS | 3.24 PASS | 2.25 PASS | 2.19 PASS | 2.31 PASS | 3.61 PASS | 2.43 PASS | 3.40 PASS | 2.41 PASS | 3.46 PASS | 2.90 PASS | 2.75 PASS |

### opamp — gain_err % (gate ≤10)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 2.10 PASS | 0.67 PASS | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 2.24 PASS | 99.99 FAIL | 0.81 PASS | 0.77 PASS | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 0.21 PASS | 100.00 FAIL | 0.43 PASS | 100.00 FAIL | 1.58 PASS | 101.73 FAIL | 0.15 PASS | 100.00 FAIL | 100.00 FAIL | 99.99 FAIL | 0.21 PASS | 100.00 FAIL |
| tsmc7 | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 28.35 FAIL | 99.99 FAIL | 91.52 FAIL | 99.99 FAIL | 99.97 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 161.90 FAIL | 99.99 FAIL | 121.20 FAIL | 126.77 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL |
| tsmc12 | 6.25 PASS | 0.35 PASS | 5.71 PASS | 4.02 PASS | 16.10 FAIL | 4.33 PASS | 100.00 FAIL | 100.00 FAIL | 6.30 PASS | 6.32 PASS | 100.00 FAIL | 6.51 PASS | 6.27 PASS | 6.47 PASS | 5.82 PASS | 100.00 FAIL | 100.00 FAIL | 7.01 PASS | 99.99 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 6.25 PASS | 5.80 PASS |
| tsmc16 | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | — FAIL | 79.15 FAIL | 77.19 FAIL | 7.34 PASS | 100.00 FAIL | 100.00 FAIL | 7.13 PASS | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 1.28 PASS | 100.00 FAIL | 5.41 PASS | 100.00 FAIL | 99.99 FAIL | 100.00 FAIL | 6.16 PASS | 100.00 FAIL | 7.34 PASS | 100.00 FAIL | 100.00 FAIL |

### sram_snm — max lobe-NRMSE % (gate ≤10 + positivity)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 6.04 PASS | 6.13 PASS | 6.70 PASS | 5.83 PASS | 6.36 PASS | 6.31 PASS | 6.27 PASS | 6.24 PASS | 6.20 PASS | 6.22 PASS | 6.23 PASS | 6.23 PASS | 6.26 PASS | 6.31 PASS | 6.31 PASS | 6.51 PASS | 7.35 PASS | 6.65 PASS | 6.04 PASS | 6.50 PASS | 6.69 PASS | 6.39 PASS | 6.27 PASS | 6.31 PASS | 6.35 PASS |
| tsmc7 | 3.61 PASS | 2.85 PASS | 2.77 PASS | 4.87 PASS | 5.72 PASS | 8.36 PASS | 4.22 PASS | 3.13 PASS | 3.67 PASS | 3.68 PASS | 3.25 PASS | 4.17 PASS | 3.33 PASS | 4.04 PASS | 2.67 PASS | 2.47 PASS | 2.98 PASS | 1.99 PASS | 1.84 PASS | 2.02 PASS | 4.76 PASS | 3.73 PASS | 4.22 PASS | 3.68 PASS | 3.08 PASS |
| tsmc12 | 1.88 PASS | 1.94 PASS | 1.52 PASS | 62.04 FAIL | 105.21 FAIL | 11.60 FAIL | 1.96 PASS | 2.00 PASS | 1.96 PASS | 1.95 PASS | 1.94 PASS | 1.97 PASS | 1.90 PASS | 2.02 PASS | 1.86 PASS | 1.32 PASS | 29.69 FAIL | 27.92 FAIL | 1.39 PASS | 1.96 PASS | 1.35 PASS | 1.51 PASS | 1.96 PASS | 1.96 PASS | 1.85 PASS |
| tsmc16 | 1.74 PASS | 1.76 PASS | 2.87 PASS | 2.23 PASS | 11.64 FAIL | 1.87 PASS | 1.72 PASS | 1.73 PASS | 1.78 PASS | 1.76 PASS | 1.78 PASS | 1.79 PASS | 1.77 PASS | 1.73 PASS | 3.06 PASS | 1.40 PASS | 16.77 FAIL | 16.56 FAIL | 4.08 PASS | 2.39 PASS | 1.72 PASS | 1.12 PASS | 1.72 PASS | 1.73 PASS | 3.02 PASS |

### switchcap — charge_err %VDD / droop %allowance (gates ≤5 / ≤100)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 3.48/6 PASS | 3.47/3 PASS | 3.31/19 PASS | 0.89/0 PASS | 0.89/17 PASS | 0.89/13 PASS | 2.02/14 PASS | 2.03/16 PASS | 2.04/16 PASS | 2.04/16 PASS | 2.03/17 PASS | 2.04/16 PASS | 2.01/15 PASS | 2.05/13 PASS | 2.92/8 PASS | 3.23/0 PASS | 2.36/16 PASS | 2.32/1 PASS | 9.19/46 FAIL | 3.56/0 PASS | 2.53/9 PASS | 3.40/0 PASS | 2.02/14 PASS | 2.06/14 PASS | 1.70/13 PASS |
| tsmc7 | 2.45/11 PASS | 2.45/11 PASS | 2.59/6 PASS | 1.77/70 PASS | 1.77/210 FAIL | 1.76/194 FAIL | 2.19/6 PASS | 2.19/12 PASS | 2.18/12 PASS | 2.19/10 PASS | 2.18/10 PASS | 2.21/12 PASS | 2.19/9 PASS | 2.18/12 PASS | 2.42/13 PASS | 2.44/22 PASS | 2.29/4 PASS | 2.29/38 PASS | 4.01/109 FAIL | 2.42/0 PASS | 2.50/2 PASS | 2.35/4 PASS | 2.19/6 PASS | 2.17/11 PASS | 2.21/14 PASS |
| tsmc12 | 4.14/13 PASS | 4.15/10 PASS | 4.14/8 PASS | 2.54/3 PASS | 2.54/78 PASS | 2.54/25 PASS | 4.17/40 PASS | 4.17/31 PASS | 4.16/12 PASS | 4.17/22 PASS | 4.18/28 PASS | 4.17/36 PASS | 4.17/27 PASS | 4.17/21 PASS | 4.08/0 PASS | 4.08/0 PASS | 4.19/9 PASS | 4.05/0 PASS | 5.17/1 FAIL | 4.12/13 PASS | 4.11/5 PASS | 4.14/10 PASS | 4.17/40 PASS | 4.17/21 PASS | 4.11/4 PASS |
| tsmc16 | 3.32/8 PASS | 3.33/7 PASS | 3.27/1 PASS | 2.01/0 PASS | 2.01/58 PASS | 2.01/44 PASS | 3.32/31 PASS | 3.33/35 PASS | 3.35/41 PASS | 3.33/34 PASS | 3.34/30 PASS | 3.33/32 PASS | 3.33/31 PASS | 3.32/32 PASS | 3.35/3 PASS | 3.30/1 PASS | 3.17/16 PASS | 3.25/9 PASS | 7.26/2 FAIL | 3.30/5 PASS | 3.31/9 PASS | 3.29/14 PASS | 3.32/31 PASS | 3.32/32 PASS | 3.38/18 PASS |

### Waveform / locus NRMSE % (all circuits, lower = better)

#### ring_osc

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 63.39 | 64.02 | 62.05 | 72.41 | 73.69 | 74.01 | 74.39 | 74.39 | 74.40 | 74.39 | 74.40 | 74.40 | 74.40 | 74.40 | 63.98 | 65.05 | 69.26 | 67.47 | 57.06 | 64.19 | 62.59 | 64.92 | 74.39 | 74.40 | 74.39 |
| tsmc7 | 59.08 | 64.50 | 56.36 | 73.70 | 73.70 | 73.72 | 74.23 | 74.18 | 74.11 | 74.19 | 73.40 | 74.17 | 74.22 | 74.16 | 59.35 | 64.79 | 60.50 | 63.89 | 56.06 | 63.40 | 56.74 | 64.71 | 74.23 | 74.17 | 74.27 |
| tsmc12 | 67.41 | 67.41 | 71.05 | 70.63 | 70.62 | 70.59 | 70.94 | 67.36 | 66.56 | 62.42 | 64.26 | 66.48 | 60.85 | 60.81 | 54.15 | 54.48 | 54.69 | 54.19 | 71.17 | 70.05 | 63.18 | 61.28 | 70.94 | 62.28 | 60.75 |
| tsmc16 | 57.77 | 60.45 | 64.68 | 69.27 | 69.19 | 69.20 | 65.14 | 58.66 | 58.59 | 58.70 | 58.70 | 58.72 | 58.76 | 59.03 | 58.13 | 52.56 | 51.76 | 52.24 | 59.27 | 54.19 | 61.03 | 54.35 | 65.14 | 59.03 | 58.44 |

#### opamp

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 36.19 | 40.54 | 70.58 | 70.58 | 70.58 | 0.77 | 70.58 | 42.95 | 43.72 | 70.58 | 70.58 | 70.58 | 70.58 | 27.84 | 70.58 | 2.02 | 70.58 | 0.58 | 70.17 | 6.12 | 70.58 | 70.58 | 70.58 | 27.85 | 70.58 |
| tsmc7 | 70.08 | 70.08 | 70.08 | 68.21 | 70.07 | 70.08 | 70.07 | 70.07 | 70.07 | 70.07 | 70.07 | 70.07 | 70.07 | 70.07 | 70.08 | 70.08 | 70.07 | 70.08 | 69.64 | 70.08 | 69.66 | 69.65 | 70.07 | 70.07 | 70.07 |
| tsmc12 | 1.01 | 1.20 | 0.99 | 19.34 | 36.29 | 44.34 | 70.48 | 70.48 | 1.06 | 1.08 | 70.48 | 1.16 | 1.05 | 1.24 | 0.84 | 70.48 | 70.48 | 1.79 | 70.48 | 70.48 | 70.48 | 70.48 | 70.48 | 1.08 | 0.88 |
| tsmc16 | 70.43 | 70.43 | 70.43 | — | 70.45 | 70.48 | 1.77 | 70.43 | 70.43 | 1.69 | 70.43 | 70.43 | 70.43 | 70.43 | 2.30 | 70.43 | 0.79 | 70.43 | 70.43 | 70.43 | 1.19 | 70.43 | 1.77 | 70.43 | 70.43 |

#### sram_snm

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 6.04 | 6.13 | 6.70 | 5.83 | 6.36 | 6.31 | 6.27 | 6.24 | 6.20 | 6.22 | 6.23 | 6.23 | 6.26 | 6.31 | 6.31 | 6.51 | 7.35 | 6.65 | 6.04 | 6.50 | 6.69 | 6.39 | 6.27 | 6.31 | 6.35 |
| tsmc7 | 3.61 | 2.85 | 2.77 | 4.87 | 5.72 | 8.36 | 4.22 | 3.13 | 3.67 | 3.68 | 3.25 | 4.17 | 3.33 | 4.04 | 2.67 | 2.47 | 2.98 | 1.99 | 1.84 | 2.02 | 4.76 | 3.73 | 4.22 | 3.68 | 3.08 |
| tsmc12 | 1.88 | 1.94 | 1.52 | 62.04 | 105.21 | 11.60 | 1.96 | 2.00 | 1.96 | 1.95 | 1.94 | 1.97 | 1.90 | 2.02 | 1.86 | 1.32 | 29.69 | 27.92 | 1.39 | 1.96 | 1.35 | 1.51 | 1.96 | 1.96 | 1.85 |
| tsmc16 | 1.74 | 1.76 | 2.87 | 2.23 | 11.64 | 1.87 | 1.72 | 1.73 | 1.78 | 1.76 | 1.78 | 1.79 | 1.77 | 1.73 | 3.06 | 1.40 | 16.77 | 16.56 | 4.08 | 2.39 | 1.72 | 1.12 | 1.72 | 1.73 | 3.02 |

#### switchcap

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 6.72 | 6.75 | 6.26 | 2.48 | 2.44 | 2.46 | 5.10 | 5.15 | 5.21 | 5.17 | 5.15 | 5.16 | 5.12 | 5.19 | 5.30 | 5.94 | 5.33 | 4.95 | 8.36 | 6.47 | 5.59 | 6.35 | 5.10 | 5.19 | 3.83 |
| tsmc7 | 3.31 | 3.31 | 3.74 | 2.25 | 2.20 | 2.20 | 2.87 | 2.85 | 2.81 | 2.86 | 2.84 | 2.88 | 2.85 | 2.84 | 3.25 | 3.26 | 3.05 | 3.19 | 5.05 | 3.30 | 3.60 | 3.15 | 2.87 | 2.80 | 2.86 |
| tsmc12 | 5.44 | 5.42 | 5.43 | 3.63 | 3.56 | 3.51 | 5.38 | 5.39 | 5.37 | 5.37 | 5.37 | 5.36 | 5.36 | 5.40 | 5.38 | 5.37 | 5.39 | 5.32 | 5.24 | 5.38 | 5.41 | 5.47 | 5.38 | 5.41 | 5.36 |
| tsmc16 | 5.15 | 5.14 | 4.99 | 2.70 | 3.44 | 3.38 | 5.02 | 5.03 | 5.05 | 5.03 | 5.04 | 5.03 | 5.02 | 5.03 | 5.19 | 5.11 | 4.88 | 4.97 | 6.72 | 5.10 | 5.12 | 5.09 | 5.02 | 5.03 | 5.12 |

### Opamp open-loop AC — dc_gain_err dB / GBW ratio / PM err ° (gate ≤3dB, [0.6,1.67], ≤15°)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 114.0/—/— FAIL* | 125.2/—/— FAIL* | 9.6/22.60/0 FAIL* | 10.9/41.10/29 FAIL* | 5.2/6.60/33 FAIL* | 3.0/1.00/14 FAIL* | 151.2/—/— FAIL* | 50.3/—/— FAIL* | 175.4/—/— FAIL* | 138.3/—/— FAIL* | 10.0/30.70/12 FAIL* | 16.9/2.82/54 FAIL* | 7.7/25.30/1 FAIL* | 82.8/—/— FAIL | 6.5/15.00/21 FAIL* | 4.2/1.09/18 FAIL* | 36.6/—/— FAIL* | 3.6/1.20/17 FAIL* | 16.5/0.51/57 FAIL* | 2.9/0.85/15 FAIL* | 11.0/34.00/20 FAIL* | 11.8/32.80/21 FAIL* | 151.2/—/— FAIL* | 61.6/—/— FAIL* | 22.5/2.67/61 FAIL* |
| tsmc7 | 45.9/0.15/118 FAIL* | 27.3/1.99/29 FAIL* | 35.9/0.83/66 FAIL* | 34.7/0.46/69 FAIL* | 37.4/0.69/71 FAIL* | 235.8/—/— FAIL* | 20.9/4.83/22 FAIL* | 43.6/0.27/100 FAIL* | 37.0/0.67/70 FAIL* | 38.4/0.59/76 FAIL* | 25.1/2.89/12 FAIL* | 28.8/1.99/32 FAIL* | 29.0/1.91/34 FAIL* | 37.4/0.65/72 FAIL* | 31.1/1.40/46 FAIL* | 40.1/0.46/83 FAIL* | 23.5/3.34/2 FAIL* | 24.6/3.05/6 FAIL* | 217.8/—/— FAIL* | 102.7/—/— FAIL* | 33.8/0.37/69 FAIL* | 65.3/—/— FAIL* | 20.9/4.83/22 FAIL* | 52.9/—/— FAIL* | 22.4/4.24/11 FAIL* |
| tsmc12 | 5.1/0.97/1 FAIL | 8.0/0.99/2 FAIL | 8.8/1.02/2 FAIL | 22.4/7.70/21 FAIL* | 279.7/—/— FAIL* | 40.6/0.20/91 FAIL* | 4.5/20.80/45 FAIL* | 3.5/18.80/48 FAIL* | 6.0/0.96/2 FAIL | 7.2/0.97/2 FAIL | 4.3/21.70/50 FAIL* | 65.1/—/— FAIL* | 5.0/0.95/2 FAIL | 9.8/0.98/2 FAIL | 5.4/1.00/1 FAIL | 56.1/—/— FAIL* | 106.3/—/— FAIL* | 282.0/—/— FAIL* | 56.5/—/— FAIL* | 4.2/20.90/48 FAIL* | 4.0/21.70/52 FAIL* | 94.3/—/— FAIL* | 4.5/20.80/45 FAIL* | 7.9/0.98/2 FAIL | 6.9/1.00/1 FAIL |
| tsmc16 | 8.7/14.90/32 FAIL* | 8.2/19.30/46 FAIL* | 8.1/20.20/49 FAIL* | 36.8/0.23/64 FAIL* | 233.0/—/— FAIL | 8.9/27.00/85 FAIL* | 42.2/0.70/76 FAIL* | 40.6/0.74/70 FAIL* | 23.7/10.10/15 FAIL* | 32.5/1.86/50 FAIL* | 261.7/—/— FAIL* | 8.7/18.00/48 FAIL* | 77.1/—/— FAIL* | 8.3/16.60/40 FAIL* | 2.1/0.94/5 PASS | 78.7/—/— FAIL* | 2.4/0.91/4 FAIL* | 8.5/23.00/66 FAIL* | 32.9/4.29/43 FAIL* | 37.3/0.21/80 FAIL* | 6.6/1.06/1 FAIL | 27.4/0.75/51 FAIL* | 42.2/0.70/76 FAIL* | 8.3/16.60/40 FAIL* | 32.8/1.24/53 FAIL* |

`*` = OP-MISBIAS (NN opamp output railed at the linearization point).

### OMP∈{1,2,4} determinism (opamp + ring) — detPASS / detFAIL / FLIP

FLIP = multistable coin-flip (unbankable, §9 discipline #3). Cell shows class + per-OMP headline err%.

#### opamp

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | FLIP 2.1/100.0/0.7 | detPASS 0.7/1.3/0.7 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | FLIP 2.2/100.0/100.0 | detFAIL 100.0/100.0/100.0 | FLIP 0.8/100.0/100.0 | FLIP 0.8/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | FLIP 100.0/0.2/0.2 | detFAIL 100.0/100.0/100.0 | detPASS 0.2/0.2/0.7 | detFAIL 100.0/100.0/100.0 | detPASS 0.4/0.4/0.4 | detFAIL 100.0/100.0/100.0 | FLIP 1.6/100.0/100.0 | detFAIL 101.7/101.6/100.0 | detPASS 0.1/0.1/0.1 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 0.2/0.2/0.2 | detFAIL 100.0/100.0/100.0 |
| tsmc7 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/14.4/19.3 | detFAIL 28.4/99.9/99.9 | detFAIL 100.0/99.8/99.8 | detFAIL 91.5/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 161.9/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 121.2/29.2/29.2 | detFAIL 126.8/117.9/119.9 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 |
| tsmc12 | detPASS 6.2/6.2/6.2 | detPASS 0.3/6.4/6.4 | FLIP 5.7/100.0/0.8 | FLIP 4.0/74.9/74.9 | detFAIL 16.1/103.8/103.8 | FLIP 4.3/35.0/4.9 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 6.3/4.0/4.0 | detPASS 6.3/6.3/6.3 | detFAIL 100.0/100.0/100.0 | detPASS 6.5/6.5/6.5 | detPASS 6.3/6.3/6.3 | detPASS 6.5/6.5/6.5 | detPASS 5.8/5.8/5.8 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 7.0/7.0/7.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | FLIP 100.0/5.2/5.2 | detFAIL 100.0/100.0/100.0 | detPASS 6.2/6.2/4.2 | detPASS 5.8/5.8/5.8 |
| tsmc16 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL —/—/— | detFAIL 79.2/100.0/88.5 | detFAIL 77.2/65.2/65.2 | detPASS 7.3/7.3/7.3 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | FLIP 7.1/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 1.3/2.0/2.0 | detFAIL 100.0/100.0/100.0 | FLIP 5.4/100.0/100.0 | detFAIL 100.0/100.0/100.0 | FLIP 100.0/2.6/9.7 | detFAIL 100.0/100.0/100.0 | detPASS 6.2/6.2/6.2 | detFAIL 100.0/100.0/100.0 | detPASS 7.3/7.3/7.3 | detFAIL 100.0/100.0/100.0 | FLIP 100.0/7.5/7.5 |

#### ring_osc

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 | crit30a1 | crit30f | csobcrit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | detFAIL 12.7/12.7/12.7 | detFAIL 12.5/12.5/12.5 | detFAIL 13.5/13.5/13.5 | FLIP 5.2/5.2/4.7 | detPASS 4.7/4.7/4.7 | detPASS 4.6/4.6/4.6 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detFAIL 10.3/10.3/10.3 | detFAIL 11.6/11.6/11.6 | detFAIL 5.8/5.8/5.8 | detFAIL 6.1/6.1/6.1 | detFAIL 16.8/16.8/16.8 | detFAIL 12.6/12.6/12.6 | detFAIL 9.6/9.6/9.6 | detFAIL 11.8/11.8/11.8 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 |
| tsmc7 | detPASS 4.8/4.8/4.8 | detFAIL 7.3/7.3/7.3 | detFAIL 10.4/10.4/10.4 | detPASS 2.9/2.9/2.9 | detPASS 2.9/2.9/2.9 | detPASS 2.9/2.9/2.9 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 | detPASS 2.6/2.6/2.6 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 | detFAIL 5.1/5.1/5.1 | detFAIL 6.7/6.5/6.7 | detPASS 4.5/4.5/4.5 | detFAIL 6.7/6.7/6.7 | detFAIL 10.1/10.1/10.1 | detFAIL 7.2/7.2/7.2 | detFAIL 8.7/8.7/8.7 | detFAIL 7.2/7.2/7.2 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 | detPASS 2.4/2.4/2.4 |
| tsmc12 | detPASS 4.0/4.3/4.3 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/3.9 | detPASS 3.9/3.9/3.9 | detPASS 3.8/3.8/3.8 | detPASS 3.8/4.3/4.3 | detPASS 3.8/3.8/3.8 | detPASS 3.8/3.8/3.8 | detPASS 3.5/3.5/2.7 | detPASS 3.2/3.2/3.2 | detPASS 3.4/3.4/3.4 | detPASS 3.5/3.5/2.7 | detPASS 2.7/3.6/2.7 | detPASS 2.7/2.7/2.7 | detPASS 2.1/2.1/2.1 | detPASS 2.1/2.1/2.1 | detPASS 2.2/2.2/2.2 | detPASS 2.2/2.2/2.2 | detFAIL 5.7/5.7/5.7 | detPASS 4.2/4.2/3.9 | detPASS 2.8/2.8/2.8 | detPASS 3.4/3.4/3.4 | detPASS 3.8/3.8/3.8 | detPASS 2.7/3.5/3.5 | detPASS 2.7/2.7/2.7 |
| tsmc16 | detPASS 2.6/2.6/2.6 | detPASS 3.4/3.4/3.4 | detPASS 4.0/4.0/4.0 | detPASS 4.3/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 3.5/3.0/3.0 | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 2.9/2.9/2.9 | detPASS 3.2/3.2/3.2 | detPASS 2.2/2.2/2.2 | detPASS 2.2/2.2/2.2 | detPASS 2.3/2.3/2.3 | detPASS 3.6/3.6/3.6 | detPASS 2.4/2.4/2.4 | detPASS 3.4/3.4/3.4 | detPASS 2.4/2.4/2.2 | detPASS 3.5/3.0/3.0 | detPASS 2.9/2.9/2.9 | detPASS 2.8/2.8/2.8 |

### Aggregate accuracy per recipe

ring/SC/SRAM aggregates are means over the 4 techs (deterministic gates). The opamp columns count OMP-deterministic passes only (FLIP cells are unbankable, excluded from detPASS; mean gain_err is over detPASS cells). tsmc7-opamp (structural non-existence, fails everywhere) is excluded from the opamp mean so it doesn't drown the comparison.

| Recipe | ring mean period_err% | ring detPASS | opamp detPASS | opamp mean err% (det, excl tsmc7) | SRAM mean maxNRMSE% | SC mean charge_err% | SC max droop %alw |
|---|---|---|---|---|---|---|---|
| clean | 6.03 | 3/4 | 1/4 | 6.24 | 3.32 | 3.35 | 13 |
| invtripft | 6.81 | 2/4 | 2/4 | 2.63 | 3.17 | 3.35 | 11 |
| invtrip | 7.98 | 2/4 | 0/4 | — | 3.46 | 3.33 | 19 |
| cor | 4.04 | 3/4 | 0/4 | — | 18.74 | 1.80 | 70 |
| corft | 3.86 | 4/4 | 0/4 | — | 32.23 | 1.80 | 210 |
| corrft | 3.83 | 4/4 | 0/4 | — | 7.04 | 1.80 | 194 |
| corroft | 3.42 | 4/4 | 1/4 | 7.33 | 3.54 | 2.92 | 40 |
| corro15 | 3.24 | 4/4 | 0/4 | — | 3.27 | 2.93 | 35 |
| crit10 | 3.17 | 4/4 | 1/4 | 4.79 | 3.40 | 2.93 | 41 |
| crit15 | 3.10 | 4/4 | 1/4 | 6.32 | 3.40 | 2.93 | 34 |
| crit15m | 3.22 | 4/4 | 0/4 | — | 3.30 | 2.93 | 30 |
| crit15h | 3.17 | 4/4 | 1/4 | 6.52 | 3.54 | 2.94 | 36 |
| crit20 | 2.98 | 4/4 | 1/4 | 6.26 | 3.31 | 2.92 | 31 |
| crit30 | 3.00 | 4/4 | 2/4 | 3.42 | 3.52 | 2.93 | 32 |
| csob | 5.19 | 2/4 | 2/4 | 3.80 | 3.48 | 3.19 | 13 |
| cs7 | 5.66 | 2/4 | 1/4 | 0.43 | 2.92 | 3.26 | 22 |
| csobekv | 3.67 | 3/4 | 0/4 | — | 14.20 | 3.00 | 16 |
| ekv | 4.34 | 2/4 | 1/4 | 7.01 | 13.28 | 2.98 | 38 |
| sob | 9.06 | 1/4 | 0/4 | — | 3.34 | 6.41 | 109 |
| s7 | 6.60 | 2/4 | 1/4 | 0.15 | 3.22 | 3.35 | 13 |
| s17 | 6.14 | 2/4 | 1/4 | 6.16 | 3.63 | 3.11 | 9 |
| s123 | 6.19 | 2/4 | 0/4 | — | 3.19 | 3.29 | 14 |
| crit30a1 | 3.42 | 4/4 | 1/4 | 7.33 | 3.54 | 2.92 | 40 |
| crit30f | 3.00 | 4/4 | 2/4 | 2.89 | 3.42 | 2.93 | 32 |
| csobcrit | 2.97 | 4/4 | 1/4 | 5.80 | 3.58 | 2.85 | 18 |

---

## Appendix B — xl-tier isolated re-test (22 recipes)

Same isolated methodology, `xl` checkpoints (512×8 ~2.13M p, over-fit-boundary tier; V6.6.5 size-matrix recipe set).

### Derived summary — pass counts (xl)

| Recipe | single-run OMP=1 | strict all-OMP (opamp/ring ∈ {1,2,4}) |
|---|---|---|
| clean | 10/16 | 10/16 |
| invtripft | 10/16 | 10/16 |
| invtrip | 12/16 | 12/16 |
| cor | 5/16 | 5/16 |
| corft | 5/16 | 4/16 |
| corrft | 6/16 | 6/16 |
| corroft | 14/16 | 14/16 |
| corro15 | 13/16 | 13/16 |
| crit10 | 14/16 | 14/16 |
| crit15 | 13/16 | 13/16 |
| crit15m | 14/16 | 14/16 |
| crit15h | 12/16 | 12/16 |
| crit20 | 13/16 | 13/16 |
| crit30 | 12/16 | 12/16 |
| csob | 10/16 | 10/16 |
| cs7 | 10/16 | 10/16 |
| csobekv | 9/16 | 9/16 |
| ekv | 9/16 | 9/16 |
| sob | 10/16 | 10/16 |
| s7 | 10/16 | 10/16 |
| s17 | 12/16 | 12/16 |
| s123 | 10/16 | 10/16 |

### ring_osc — period_err % (gate ≤5) (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 13.50 FAIL | 13.36 FAIL | 12.39 FAIL | 4.73 PASS | 4.62 PASS | 4.63 PASS | 4.05 PASS | 4.06 PASS | 4.40 PASS | 4.17 PASS | 4.05 PASS | 4.05 PASS | 4.05 PASS | 4.05 PASS | 13.84 FAIL | 14.41 FAIL | 7.39 FAIL | 7.08 FAIL | 12.59 FAIL | 13.09 FAIL | 16.00 FAIL | 17.36 FAIL |
| tsmc7 | 14.31 FAIL | 14.12 FAIL | 16.45 FAIL | — CRASH | — CRASH | — CRASH | 2.54 PASS | 2.83 PASS | 3.84 PASS | 2.43 PASS | 3.61 PASS | 2.55 PASS | 3.74 PASS | 2.43 PASS | 12.68 FAIL | 7.17 FAIL | 5.39 FAIL | 7.27 FAIL | 12.07 FAIL | 17.09 FAIL | 14.40 FAIL | 16.42 FAIL |
| tsmc12 | 3.40 PASS | 3.54 PASS | 3.78 PASS | — CRASH | — CRASH | 4.63 PASS | 2.68 PASS | 2.68 PASS | 2.68 PASS | 2.68 PASS | 3.49 PASS | 3.50 PASS | 2.68 PASS | 3.45 PASS | 2.85 PASS | 2.84 PASS | 2.15 PASS | 2.11 PASS | 1.67 PASS | 4.32 PASS | 3.79 PASS | 4.45 PASS |
| tsmc16 | 3.05 PASS | 3.18 PASS | 2.20 PASS | 4.00 PASS | 4.00 PASS | — CRASH | 2.76 PASS | 2.77 PASS | 2.89 PASS | 3.06 PASS | 2.76 PASS | 2.76 PASS | 2.77 PASS | 2.77 PASS | 3.91 PASS | 3.84 PASS | 2.19 PASS | 2.22 PASS | 2.37 PASS | 3.56 PASS | 3.29 PASS | 3.23 PASS |

### opamp — gain_err % (gate ≤10) (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 100.00 FAIL | 100.00 FAIL | 7.06 PASS | 70.99 FAIL | 100.00 FAIL | 100.00 FAIL | 99.99 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 3.37 PASS | 100.00 FAIL | 3.54 PASS | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 4.13 PASS | 99.96 FAIL | 100.00 FAIL | 100.00 FAIL | 1.39 PASS | 100.00 FAIL |
| tsmc7 | 99.99 FAIL | 99.99 FAIL | 17.00 FAIL | — FAIL | — FAIL | — FAIL | 124.75 FAIL | 99.99 FAIL | 99.99 FAIL | 119.98 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL | 100.00 FAIL | 99.99 FAIL | 99.99 FAIL | 99.99 FAIL |
| tsmc12 | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | — FAIL | — FAIL | — FAIL | 6.22 PASS | 100.00 FAIL | 6.20 PASS | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 99.97 FAIL | 99.99 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 5.44 PASS | 100.00 FAIL | 100.00 FAIL | 99.99 FAIL |
| tsmc16 | 100.00 FAIL | 100.00 FAIL | 6.50 PASS | — FAIL | — FAIL | — FAIL | 6.66 PASS | 6.53 PASS | 6.66 PASS | 6.51 PASS | 6.47 PASS | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 100.00 FAIL | 7.73 PASS | 100.00 FAIL | 6.74 PASS | 100.00 FAIL |

### sram_snm — max lobe-NRMSE % (gate ≤10 + positivity) (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 5.89 PASS | 5.91 PASS | 6.30 PASS | 6.01 PASS | 6.21 PASS | 6.26 PASS | 6.20 PASS | 5.98 PASS | 6.01 PASS | 5.99 PASS | 6.00 PASS | 5.98 PASS | 5.96 PASS | 6.19 PASS | 5.86 PASS | 5.55 PASS | 7.57 PASS | 7.70 PASS | 7.97 PASS | 5.54 PASS | 5.89 PASS | 5.23 PASS |
| tsmc7 | 2.93 PASS | 2.65 PASS | 2.60 PASS | 51.19 FAIL | 7.25 PASS | 4.39 PASS | 1.80 PASS | 1.94 PASS | 2.80 PASS | 1.74 PASS | 3.01 PASS | 3.10 PASS | 3.74 PASS | 3.30 PASS | 4.00 PASS | 3.77 PASS | 3.55 PASS | 3.35 PASS | 6.78 PASS | 2.98 PASS | 3.70 PASS | 2.10 PASS |
| tsmc12 | 4.39 PASS | 4.34 PASS | 3.03 PASS | 102.23 FAIL | 140.92 FAIL | 4.31 PASS | 4.28 PASS | 4.27 PASS | 4.31 PASS | 4.29 PASS | 4.33 PASS | 4.34 PASS | 4.32 PASS | 4.32 PASS | 1.95 PASS | 1.52 PASS | 17.80 FAIL | 9.30 PASS | 13.76 FAIL | 2.06 PASS | 2.18 PASS | 3.73 PASS |
| tsmc16 | 2.22 PASS | 2.31 PASS | 1.73 PASS | 3.67 FAIL | 150.40 FAIL | 189.12 FAIL | 2.01 PASS | 2.11 PASS | 2.14 PASS | 2.04 PASS | 2.24 PASS | 2.25 PASS | 2.44 PASS | 1.78 PASS | 1.74 PASS | 1.79 PASS | 22.69 FAIL | 33.66 FAIL | 3.37 PASS | 2.30 PASS | 2.25 PASS | 1.75 PASS |

### switchcap — charge_err %VDD / droop %allowance (gates ≤5 / ≤100) (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 3.18/4 PASS | 3.24/5 PASS | 3.06/3 PASS | 0.88/16 PASS | 0.88/2 PASS | 0.88/0 PASS | 1.68/4 PASS | 2.22/7 PASS | 2.22/5 PASS | 2.22/7 PASS | 2.26/9 PASS | 2.28/7 PASS | 2.23/10 PASS | 1.69/5 PASS | 3.45/0 PASS | 3.70/5 PASS | 2.09/3 PASS | 1.97/25 PASS | 1.65/163 FAIL | 3.58/16 PASS | 3.48/2 PASS | 3.76/8 PASS |
| tsmc7 | 2.67/0 PASS | 2.69/0 PASS | 2.78/0 PASS | —/— CRASH | —/— CRASH | —/— CRASH | 2.26/0 PASS | 2.28/0 PASS | 2.29/0 PASS | 2.25/0 PASS | 2.28/0 PASS | 2.27/0 PASS | 2.27/0 PASS | 2.25/0 PASS | 2.65/0 PASS | 2.50/1 PASS | 2.18/0 PASS | 2.26/0 PASS | 3.65/4 PASS | 2.63/4 PASS | 2.74/22 PASS | 2.68/0 PASS |
| tsmc12 | 4.19/1 PASS | 4.19/5 PASS | 4.15/3 PASS | —/— CRASH | —/— CRASH | 2.54/172 FAIL | 4.22/6 PASS | 4.22/9 PASS | 4.22/4 PASS | 4.22/5 PASS | 4.22/8 PASS | 4.25/7 PASS | 4.22/6 PASS | 4.22/6 PASS | 4.15/0 PASS | 4.12/2 PASS | 4.10/0 PASS | 4.05/9 PASS | 3.47/56 PASS | 4.23/0 PASS | 4.17/4 PASS | 4.19/2 PASS |
| tsmc16 | 3.42/1 PASS | 3.42/0 PASS | 3.30/1 PASS | 2.01/0 PASS | 2.01/1417 FAIL | —/— CRASH | 3.48/4 PASS | 3.47/2 PASS | 3.47/11 PASS | 3.47/13 PASS | 3.48/17 PASS | 3.48/23 PASS | 3.48/10 PASS | 3.48/15 PASS | 3.32/1 PASS | 3.33/3 PASS | 3.26/8 PASS | 3.27/0 PASS | 1.46/0 PASS | 3.24/22 PASS | 3.32/10 PASS | 3.33/0 PASS |

### Waveform / locus NRMSE % (all circuits, lower = better) (xl)

#### ring_osc (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 61.95 | 62.29 | 64.18 | 73.83 | 74.13 | 74.11 | 74.40 | 74.41 | 74.03 | 74.71 | 74.41 | 74.41 | 74.41 | 74.40 | 60.29 | 59.08 | 60.86 | 61.73 | 64.27 | 63.70 | 59.31 | 57.83 |
| tsmc7 | 55.64 | 55.79 | 60.33 | — | — | — | 73.73 | 72.83 | 67.63 | 74.25 | 68.84 | 73.97 | 69.35 | 74.25 | 59.99 | 64.30 | 59.93 | 63.19 | 59.25 | 60.49 | 56.41 | 60.44 |
| tsmc12 | 61.95 | 62.49 | 66.71 | — | — | 70.92 | 60.80 | 60.81 | 60.80 | 60.81 | 68.32 | 67.50 | 60.79 | 68.34 | 62.42 | 62.88 | 54.55 | 54.00 | 47.78 | 72.07 | 67.56 | 71.20 |
| tsmc16 | 63.61 | 59.98 | 52.00 | 69.20 | 69.17 | — | 58.59 | 58.63 | 59.06 | 59.47 | 58.62 | 58.59 | 58.62 | 58.64 | 64.07 | 63.23 | 51.84 | 52.16 | 51.72 | 64.92 | 59.79 | 59.51 |

#### opamp (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 70.58 | 70.58 | 0.71 | 68.87 | 70.58 | 70.58 | 70.58 | 70.58 | 70.58 | 70.58 | 0.61 | 70.58 | 0.62 | 70.58 | 70.58 | 70.58 | 0.76 | 70.58 | 70.58 | 70.58 | 0.50 | 70.58 |
| tsmc7 | 70.08 | 70.08 | 66.73 | — | — | — | 69.65 | 70.07 | 70.07 | 69.66 | 70.07 | 70.07 | 70.07 | 70.07 | 70.08 | 70.08 | 70.07 | 70.07 | 70.08 | 70.08 | 70.08 | 70.08 |
| tsmc12 | 70.48 | 70.48 | 70.48 | — | — | — | 1.26 | 70.48 | 1.21 | 70.48 | 70.48 | 70.48 | 70.48 | 70.48 | 70.48 | 70.48 | 70.48 | 70.48 | 0.94 | 70.48 | 70.48 | 70.48 |
| tsmc16 | 70.43 | 70.43 | 1.16 | — | — | — | 1.40 | 1.30 | 1.39 | 1.22 | 1.24 | 70.43 | 70.43 | 70.43 | 70.43 | 70.43 | 70.43 | 70.43 | 1.53 | 70.43 | 1.39 | 70.43 |

#### sram_snm (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 5.89 | 5.91 | 6.30 | 6.01 | 6.21 | 6.26 | 6.20 | 5.98 | 6.01 | 5.99 | 6.00 | 5.98 | 5.96 | 6.19 | 5.86 | 5.55 | 7.57 | 7.70 | 7.97 | 5.54 | 5.89 | 5.23 |
| tsmc7 | 2.93 | 2.65 | 2.60 | 51.19 | 7.25 | 4.39 | 1.80 | 1.94 | 2.80 | 1.74 | 3.01 | 3.10 | 3.74 | 3.30 | 4.00 | 3.77 | 3.55 | 3.35 | 6.78 | 2.98 | 3.70 | 2.10 |
| tsmc12 | 4.39 | 4.34 | 3.03 | 102.23 | 140.92 | 4.31 | 4.28 | 4.27 | 4.31 | 4.29 | 4.33 | 4.34 | 4.32 | 4.32 | 1.95 | 1.52 | 17.80 | 9.30 | 13.76 | 2.06 | 2.18 | 3.73 |
| tsmc16 | 2.22 | 2.31 | 1.73 | 3.67 | 150.40 | 189.12 | 2.01 | 2.11 | 2.14 | 2.04 | 2.24 | 2.25 | 2.44 | 1.78 | 1.74 | 1.79 | 22.69 | 33.66 | 3.37 | 2.30 | 2.25 | 1.75 |

#### switchcap (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 5.98 | 6.00 | 5.96 | 2.47 | 2.50 | 2.54 | 3.96 | 4.88 | 4.95 | 4.87 | 4.95 | 4.96 | 4.86 | 3.98 | 6.43 | 6.89 | 4.16 | 4.25 | 5.06 | 6.63 | 6.33 | 6.67 |
| tsmc7 | 4.10 | 4.12 | 4.28 | — | — | — | 3.06 | 3.05 | 3.04 | 3.08 | 3.11 | 3.06 | 3.11 | 3.09 | 3.82 | 3.47 | 2.89 | 3.11 | 4.74 | 4.03 | 4.29 | 4.00 |
| tsmc12 | 5.52 | 5.52 | 5.46 | — | — | 3.53 | 5.42 | 5.44 | 5.49 | 5.48 | 5.48 | 5.51 | 5.44 | 5.47 | 5.52 | 5.43 | 5.35 | 5.28 | 3.70 | 5.53 | 5.55 | 5.55 |
| tsmc16 | 5.17 | 5.22 | 5.13 | 3.41 | 2.92 | — | 5.40 | 5.26 | 5.22 | 5.29 | 5.24 | 5.26 | 5.25 | 5.94 | 5.16 | 5.17 | 4.97 | 4.94 | 2.23 | 4.94 | 5.13 | 5.18 |

### Opamp open-loop AC — dc_gain_err dB / GBW ratio / PM err ° (gate ≤3dB, [0.6,1.67], ≤15°) (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | 86.1/—/— FAIL* | 8.6/13.50/30 FAIL* | 2.1/0.73/13 FAIL* | 16.6/0.36/57 FAIL* | 6.4/25.30/4 FAIL* | 108.9/—/— FAIL* | 118.8/—/— FAIL* | 10.9/10.90/36 FAIL* | 3.8/26.70/10 FAIL* | 5.9/26.80/6 FAIL* | 23.6/0.15/64 FAIL* | 95.3/—/— FAIL* | 2.9/0.83/0 FAIL* | 11.8/11.60/36 FAIL* | 18.5/0.74/59 FAIL* | 16.8/3.91/50 FAIL* | 3.7/0.99/6 FAIL* | 3.8/15.80/20 FAIL* | 20.1/8.35/52 FAIL* | 38.9/—/— FAIL* | 2.5/0.93/13 FAIL* | 14.7/5.32/45 FAIL* |
| tsmc7 | 19.5/4.64/24 FAIL* | 44.0/0.23/104 FAIL* | 70.3/—/— FAIL* | — | 52.7/—/— FAIL* | 111.3/—/— FAIL* | 64.9/—/— FAIL* | 20.5/4.66/21 FAIL* | 20.6/4.35/18 FAIL* | 39.0/0.34/81 FAIL* | 33.5/1.10/55 FAIL* | 144.3/—/— FAIL* | 50.3/—/— FAIL* | 36.4/0.66/69 FAIL* | 191.0/—/— FAIL* | 19.0/4.28/21 FAIL* | 31.3/1.46/44 FAIL* | 34.0/0.94/58 FAIL* | 35.1/0.40/71 FAIL* | 31.6/1.16/50 FAIL* | 100.0/—/— FAIL* | 21.8/3.52/4 FAIL* |
| tsmc12 | 42.7/0.12/115 FAIL* | 22.0/5.95/21 FAIL* | 112.3/—/— FAIL* | 28.7/0.69/50 FAIL* | 27.5/0.69/48 FAIL* | 173.6/—/— FAIL* | 31.2/2.24/46 FAIL* | 95.6/—/— FAIL* | 31.9/2.03/48 FAIL* | 23.9/5.32/28 FAIL* | 60.0/—/— FAIL* | 36.7/0.66/68 FAIL* | 30.4/1.12/51 FAIL* | 22.8/8.66/20 FAIL* | 33.3/1.59/54 FAIL* | 24.4/5.13/26 FAIL* | 165.4/—/— FAIL* | 5.0/19.50/39 FAIL* | 40.6/0.45/102 FAIL* | 4.5/20.70/45 FAIL* | 14.9/16.40/9 FAIL* | 20.8/6.53/18 FAIL* |
| tsmc16 | 99.8/—/— FAIL* | 121.0/—/— FAIL* | 7.5/1.04/1 FAIL | 87.3/—/— FAIL* | 66.4/—/— FAIL | 34.5/0.72/56 FAIL* | 7.6/1.00/1 FAIL | 7.8/0.99/3 FAIL | 36.5/1.94/53 FAIL* | 5.3/0.97/2 FAIL | 8.0/1.04/1 FAIL | 33.5/2.45/47 FAIL* | 41.5/0.31/69 FAIL* | 195.5/—/— FAIL* | 10.6/9.17/5 FAIL* | 26.5/6.63/21 FAIL* | 27.2/6.06/24 FAIL* | 26.8/6.03/24 FAIL* | 9.4/1.09/1 FAIL | 10.7/16.80/29 FAIL* | 7.9/1.04/0 FAIL | 8.4/17.30/43 FAIL* |

`*` = OP-MISBIAS (NN opamp output railed at the linearization point).

### OMP∈{1,2,4} determinism (opamp + ring) — detPASS / detFAIL / FLIP (xl)

FLIP = multistable coin-flip (unbankable, §9 discipline #3). Cell shows class + per-OMP headline err%.

#### opamp (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 7.1/7.1/7.1 | detFAIL 71.0/71.0/71.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 3.4/3.4/3.4 | detFAIL 100.0/100.0/100.0 | detPASS 3.5/3.5/3.5 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 4.1/3.4/0.7 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 1.4/1.4/1.4 | detFAIL 100.0/100.0/100.0 |
| tsmc7 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 17.0/21.3/45.4 | detFAIL —/—/— | detFAIL —/—/— | detFAIL —/—/— | detFAIL 124.8/124.8/124.8 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 120.0/42.4/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/119.5/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 |
| tsmc12 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL —/—/— | detFAIL —/—/— | detFAIL —/—/— | detPASS 6.2/6.2/6.2 | detFAIL 100.0/100.0/100.0 | detPASS 6.2/6.2/6.2 | detFAIL 100.0/100.0/100.0 | FLIP 100.0/6.2/0.3 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 5.4/5.4/5.4 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 |
| tsmc16 | detFAIL 100.0/100.0/100.0 | FLIP 100.0/6.2/6.2 | detPASS 6.5/6.5/6.5 | detFAIL —/—/— | detFAIL —/—/— | detFAIL —/—/— | detPASS 6.7/6.7/6.7 | detPASS 6.5/6.5/6.5 | detPASS 6.7/6.7/6.7 | detPASS 6.5/6.5/6.5 | detPASS 6.5/6.5/6.5 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detFAIL 100.0/100.0/100.0 | detPASS 7.7/7.7/7.7 | detFAIL 100.0/100.0/100.0 | detPASS 6.7/6.7/6.7 | detFAIL 100.0/100.0/100.0 |

#### ring_osc (xl)

| Tech | clean | invtripft | invtrip | cor | corft | corrft | corroft | corro15 | crit10 | crit15 | crit15m | crit15h | crit20 | crit30 | csob | cs7 | csobekv | ekv | sob | s7 | s17 | s123 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | detFAIL 13.5/14.3/13.5 | detFAIL 13.4/14.6/13.3 | detFAIL 12.4/13.1/12.1 | detPASS 4.7/4.7/4.7 | FLIP 4.6/4.9/5.1 | detPASS 4.6/4.6/4.6 | detPASS 4.0/4.0/4.0 | detPASS 4.1/4.1/4.1 | detPASS 4.4/4.4/4.3 | detPASS 4.2/4.2/4.2 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.0 | detFAIL 13.8/13.8/13.8 | detFAIL 14.4/14.4/14.4 | detFAIL 7.4/7.4/7.4 | detFAIL 7.1/8.1/7.4 | detFAIL 12.6/12.6/12.6 | detFAIL 13.1/13.1/12.8 | detFAIL 16.0/16.0/16.0 | detFAIL 17.4/15.7/16.1 |
| tsmc7 | detFAIL 14.3/13.6/14.0 | detFAIL 14.1/14.2/14.0 | detFAIL 16.4/16.4/16.4 | detFAIL —/—/— | detFAIL —/—/— | detFAIL —/—/— | detPASS 2.5/2.5/2.5 | detPASS 2.8/3.5/3.5 | detPASS 3.8/4.0/3.8 | detPASS 2.4/2.4/2.4 | detPASS 3.6/3.7/3.7 | detPASS 2.5/2.5/2.5 | detPASS 3.7/2.4/3.7 | detPASS 2.4/2.4/2.4 | detFAIL 12.7/12.7/12.7 | detFAIL 7.2/7.2/7.2 | detFAIL 5.4/5.4/5.4 | detFAIL 7.3/8.7/8.7 | detFAIL 12.1/12.1/12.1 | detFAIL 17.1/17.1/17.1 | detFAIL 14.4/14.4/14.4 | detFAIL 16.4/16.4/16.4 |
| tsmc12 | detPASS 3.4/3.4/3.4 | detPASS 3.5/3.5/3.5 | detPASS 3.8/4.7/4.2 | detFAIL —/—/— | detFAIL —/—/— | detPASS 4.6/3.8/3.8 | detPASS 2.7/2.7/2.7 | detPASS 2.7/2.7/2.7 | detPASS 2.7/3.5/3.5 | detPASS 2.7/2.7/2.7 | detPASS 3.5/3.5/3.5 | detPASS 3.5/2.7/3.5 | detPASS 2.7/2.7/2.7 | detPASS 3.5/3.5/2.7 | detPASS 2.9/2.9/3.8 | detPASS 2.8/2.8/2.8 | detPASS 2.1/2.1/2.1 | detPASS 2.1/2.9/2.9 | detPASS 1.7/1.7/1.7 | detPASS 4.3/4.1/4.3 | detPASS 3.8/3.8/4.2 | detPASS 4.5/4.7/4.2 |
| tsmc16 | detPASS 3.0/3.2/2.9 | detPASS 3.2/3.3/2.8 | detPASS 2.2/2.2/2.6 | detPASS 4.0/4.0/4.0 | detPASS 4.0/4.0/4.1 | detFAIL —/—/— | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 2.9/3.0/2.9 | detPASS 3.1/2.8/3.1 | detPASS 2.8/2.8/2.8 | detPASS 2.8/3.0/3.1 | detPASS 2.8/2.8/2.8 | detPASS 2.8/2.8/2.8 | detPASS 3.9/3.9/3.9 | detPASS 3.8/3.8/3.8 | detPASS 2.2/2.2/2.2 | detPASS 2.2/2.2/2.6 | detPASS 2.4/2.4/2.4 | detPASS 3.6/3.4/3.3 | detPASS 3.3/3.2/3.0 | detPASS 3.2/3.0/2.9 |

### Aggregate accuracy per recipe (xl)

ring/SC/SRAM aggregates are means over the 4 techs (deterministic gates). The opamp columns count OMP-deterministic passes only (FLIP cells are unbankable, excluded from detPASS; mean gain_err is over detPASS cells). tsmc7-opamp (structural non-existence, fails everywhere) is excluded from the opamp mean so it doesn't drown the comparison.

| Recipe | ring mean period_err% | ring detPASS | opamp detPASS | opamp mean err% (det, excl tsmc7) | SRAM mean maxNRMSE% | SC mean charge_err% | SC max droop %alw |
|---|---|---|---|---|---|---|---|
| clean | 8.56 | 2/4 | 0/4 | — | 3.86 | 3.37 | 4 |
| invtripft | 8.55 | 2/4 | 0/4 | — | 3.80 | 3.39 | 5 |
| invtrip | 8.71 | 2/4 | 2/4 | 6.78 | 3.42 | 3.32 | 3 |
| cor | 4.37 | 2/4 | 0/4 | — | 40.77 | 1.44 | 16 |
| corft | 4.31 | 1/4 | 0/4 | — | 76.19 | 1.44 | 1417 |
| corrft | 4.63 | 2/4 | 0/4 | — | 51.02 | 1.71 | 172 |
| corroft | 3.01 | 4/4 | 2/4 | 6.44 | 3.57 | 2.91 | 6 |
| corro15 | 3.08 | 4/4 | 1/4 | 6.53 | 3.58 | 3.05 | 9 |
| crit10 | 3.45 | 4/4 | 2/4 | 6.43 | 3.81 | 3.05 | 11 |
| crit15 | 3.08 | 4/4 | 1/4 | 6.50 | 3.52 | 3.04 | 13 |
| crit15m | 3.48 | 4/4 | 2/4 | 4.92 | 3.90 | 3.06 | 17 |
| crit15h | 3.21 | 4/4 | 0/4 | — | 3.92 | 3.07 | 23 |
| crit20 | 3.31 | 4/4 | 1/4 | 3.54 | 4.12 | 3.05 | 10 |
| crit30 | 3.17 | 4/4 | 0/4 | — | 3.90 | 2.91 | 15 |
| csob | 8.32 | 2/4 | 0/4 | — | 3.39 | 3.39 | 1 |
| cs7 | 7.06 | 2/4 | 0/4 | — | 3.16 | 3.41 | 5 |
| csobekv | 4.28 | 2/4 | 1/4 | 2.72 | 12.90 | 2.91 | 8 |
| ekv | 4.67 | 2/4 | 0/4 | — | 13.50 | 2.89 | 25 |
| sob | 7.17 | 2/4 | 2/4 | 6.59 | 7.97 | 2.56 | 163 |
| s7 | 9.52 | 2/4 | 0/4 | — | 3.22 | 3.42 | 22 |
| s17 | 9.37 | 2/4 | 2/4 | 4.07 | 3.50 | 3.43 | 22 |
| s123 | 10.37 | 2/4 | 0/4 | — | 3.20 | 3.49 | 8 |

## Appendix C — V6.6.5 recipe × size matrix (13 recipes × 4 sizes, device + AC suites)

Each recipe is a UNIFORM training addendum applied identically to all (tech × device × size) checkpoints on top of the V6.6.0 clean recipe (`--apply-filter off --swa-mode ema --seed 42`). `clean` is the control. Ground truth = NGSPICE BSIM-CMG (LEVEL=72), CPU-pinned. Gate matrix = 4 circuits × 4 techs = 16 complex gates per (recipe,size).

### Headline — complex-circuit gate pass-rate (X / 16)

Higher is better. Clean control: large 13/16, xl 10/16 (V6.6.0).

| Recipe | small | medium | large | xl |
|---|---|---|---|---|
| clean | 7/16 | 10/16 | 13/16 | 10/16 |
| cor | 11/16 | 12/16 | 11/16 | 5/16 |
| corft | 11/16 | 9/16 | 9/16 | 5/16 |
| invtripft | 9/16 | 10/16 | 12/16 | 10/16 |
| invtrip | 9/16 | 11/16 | 11/16 | 12/16 |
| csob | 9/16 | 10/16 | 12/16 | 10/16 |
| sob | 6/16 | 6/16 | 5/16 | 10/16 |
| ekv | 7/16 | 11/16 | 10/16 | 9/16 |
| csobekv | 9/16 | 9/16 | 10/16 | 9/16 |
| cs7 | 7/16 | 10/16 | 11/16 | 10/16 |
| s123 | 8/16 | 9/16 | 10/16 | 10/16 |
| s17 | 9/16 | 12/16 | 11/16 | 12/16 |
| s7 | 7/16 | 10/16 | 11/16 | 10/16 |

### AC device CS-amp pass-rate (X / 8)

Clean control: large 4/8-equiv (4/12 over all sizes in V6.6.0). Per (recipe,size) there are 8 device cells (4 techs × n/p).

| Recipe | small | medium | large | xl |
|---|---|---|---|---|
| clean | 5/12 | 4/12 | 4/12 | 4/12 |
| cor | 6/12 | 5/12 | 5/12 | 5/12 |
| corft | 6/12 | 5/12 | 5/12 | 4/12 |
| invtripft | 5/12 | 3/12 | 3/12 | 2/12 |
| invtrip | 7/12 | 5/12 | 4/12 | 2/12 |
| csob | 7/12 | 4/12 | 5/12 | 5/12 |
| sob | 5/12 | 5/12 | 4/12 | 4/12 |
| ekv | 5/12 | 5/12 | 5/12 | 3/12 |
| csobekv | 7/12 | 2/12 | 6/12 | 4/12 |
| cs7 | 5/12 | 4/12 | 4/12 | 3/12 |
| s123 | 6/12 | 4/12 | 4/12 | 3/12 |
| s17 | 4/12 | 5/12 | 5/12 | 4/12 |
| s7 | 5/12 | 4/12 | 4/12 | 3/12 |

### Device-level mean NRMSE% (lower = better fit)

| Recipe | small | medium | large | xl |
|---|---|---|---|---|
| clean | 1.63 | 1.29 | 1.7 | 2.17 |
| cor | 1.44 | 1.16 | 1.7 | 2.15 |
| corft | 1.49 | 1.2 | 1.6 | 2.0 |
| invtripft | 1.67 | 1.28 | 1.74 | 2.17 |
| invtrip | 1.59 | 1.24 | 1.84 | 2.01 |
| csob | 1.77 | 1.33 | 1.5 | 1.77 |
| sob | 1.92 | 1.38 | 1.95 | 2.22 |
| ekv | 1.99 | 1.72 | 2.38 | 2.18 |
| csobekv | 2.19 | 1.77 | 2.0 | 2.05 |
| cs7 | 1.64 | 1.28 | 1.74 | 1.8 |
| s123 | 1.63 | 1.37 | 1.97 | 2.3 |
| s17 | 1.62 | 1.48 | 1.73 | 2.01 |
| s7 | 1.66 | 1.37 | 1.91 | 2.15 |

### Complex-gate matrix — size = small

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | opamp | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc5 | switchcap | PASS | FAIL | FAIL | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | FAIL | FAIL | FAIL |
| tsmc7 | ring_osc | FAIL | PASS | PASS | PASS | FAIL | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL |
| tsmc7 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |
| tsmc7 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | switchcap | FAIL | PASS | PASS | FAIL | PASS | PASS | FAIL | FAIL | PASS | FAIL | FAIL | PASS | FAIL |
| tsmc12 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc12 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| tsmc12 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| tsmc12 | switchcap | FAIL | PASS | PASS | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc16 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc16 | opamp | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | FAIL |
| tsmc16 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| tsmc16 | switchcap | FAIL | PASS | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS |

#### Gate headline detail — size = small (period_err% / gain_err% / charge_err% / SNM)

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | period_err=8.06% | period_err=4.82% | period_err=4.83% | period_err=7.40% | period_err=6.02% | period_err=6.91% | period_err=6.83% | period_err=5.86% | period_err=7.19% | period_err=5.34% | period_err=7.54% | period_err=6.87% | period_err=7.90% |
| tsmc5 | opamp | gain_err=100.00% trip_shift=116.00mV | gain_err=100.00% trip_shift=-150.00mV | gain_err=100.00% trip_shift=-54.00mV | gain_err=0.35% trip_shift=22.00mV | gain_err=99.20% trip_shift=132.00mV | gain_err=0.42% trip_shift=-80.00mV | gain_err=11.80% trip_shift=-88.00mV | gain_err=0.50% trip_shift=0.00mV | gain_err=99.99% trip_shift=70.00mV | gain_err=100.00% trip_shift=126.00mV | gain_err=100.00% trip_shift=-122.00mV | gain_err=100.00% trip_shift=-150.00mV | gain_err=100.00% trip_shift=-78.00mV |
| tsmc5 | sram_snm | NG_SNM=189.2mV DN_SNM=174.4mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=279.1mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=204.8mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=166.3mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=164.5mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=173.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=182.1mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=187.9mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=165.3mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=168.4mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=165.4mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=183.5mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=180.6mV  force_ic=ok/ok |
| tsmc5 | switchcap | charge_err=1.72% | charge_err=0.76% | charge_err=0.91% | charge_err=1.27% | charge_err=2.05% | charge_err=0.21% | charge_err=2.48% | charge_err=1.80% | charge_err=1.77% | charge_err=1.50% | charge_err=1.68% | charge_err=2.06% | charge_err=2.19% |
| tsmc7 | ring_osc | period_err=5.94% | period_err=3.18% | period_err=4.23% | period_err=4.92% | period_err=5.27% | period_err=6.54% | period_err=5.83% | period_err=3.05% | period_err=4.19% | period_err=5.39% | period_err=4.34% | period_err=7.17% | period_err=6.13% |
| tsmc7 | opamp | gain_err=10.33% trip_shift=-146.00mV | gain_err=117.26% trip_shift=-148.00mV | gain_err=99.99% trip_shift=-38.00mV | gain_err=99.97% trip_shift=56.00mV | gain_err=12.52% trip_shift=-140.00mV | gain_err=11.36% trip_shift=-144.00mV | gain_err=124.44% trip_shift=-148.00mV | gain_err=120.31% trip_shift=-148.00mV | gain_err=99.99% trip_shift=100.00mV | gain_err=99.98% trip_shift=86.00mV | gain_err=99.99% trip_shift=82.00mV | gain_err=9.62% trip_shift=-46.00mV | gain_err=11.31% trip_shift=-136.00mV |
| tsmc7 | sram_snm | NG_SNM=187.4mV DN_SNM=323.1mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=207.3mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=265.6mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=255.1mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=182.4mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=186.3mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=280.0mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=286.2mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=184.8mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=302.5mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=246.8mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=183.1mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=330.3mV  force_ic=ok/ok |
| tsmc7 | switchcap | charge_err=2.34% | charge_err=1.75% | charge_err=1.86% | charge_err=2.54% | charge_err=1.85% | charge_err=2.06% | charge_err=5.48% | charge_err=2.62% | charge_err=2.81% | charge_err=2.54% | charge_err=3.38% | charge_err=2.67% | charge_err=1.90% |
| tsmc12 | ring_osc | period_err=1.95% | period_err=3.90% | period_err=3.99% | period_err=2.41% | period_err=3.84% | period_err=3.70% | period_err=4.20% | period_err=3.20% | period_err=3.22% | period_err=3.33% | period_err=2.53% | period_err=2.50% | period_err=3.34% |
| tsmc12 | opamp | gain_err=100.00% trip_shift=-12.00mV | gain_err=105.34% trip_shift=-22.00mV | gain_err=100.00% trip_shift=-140.00mV | gain_err=99.99% trip_shift=94.00mV | gain_err=100.00% trip_shift=114.00mV | gain_err=191.42% trip_shift=-136.00mV | gain_err=16.81% trip_shift=84.00mV | gain_err=100.00% trip_shift=142.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=-62.00mV | gain_err=8.65% trip_shift=-148.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=103.87% trip_shift=-148.00mV |
| tsmc12 | sram_snm | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=258.3mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=267.9mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=225.8mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=237.4mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=229.4mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=225.4mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=230.2mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=223.8mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=257.4mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=227.2mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=227.7mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=269.9mV  force_ic=ok/ok |
| tsmc12 | switchcap | charge_err=4.09% | charge_err=2.55% | charge_err=2.51% | charge_err=3.76% | charge_err=4.68% | charge_err=4.30% | charge_err=8.14% | charge_err=4.39% | charge_err=4.28% | charge_err=4.30% | charge_err=3.97% | charge_err=3.64% | charge_err=4.15% |
| tsmc16 | ring_osc | period_err=1.47% | period_err=4.10% | period_err=4.05% | period_err=2.32% | period_err=0.72% | period_err=1.19% | period_err=2.32% | period_err=0.99% | period_err=0.68% | period_err=1.56% | period_err=1.52% | period_err=1.69% | period_err=1.08% |
| tsmc16 | opamp | gain_err=10.23% trip_shift=0.00mV | gain_err=100.00% trip_shift=30.00mV | gain_err=8.27% trip_shift=-68.00mV | gain_err=111.02% trip_shift=0.00mV | gain_err=100.00% trip_shift=8.00mV | gain_err=100.00% trip_shift=150.00mV | gain_err=144.21% trip_shift=-142.00mV | gain_err=100.00% trip_shift=144.00mV | gain_err=5.31% trip_shift=-134.00mV | gain_err=100.00% trip_shift=-8.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=3.97% trip_shift=2.00mV | gain_err=100.00% trip_shift=-58.00mV |
| tsmc16 | sram_snm | NG_SNM=235.5mV DN_SNM=219.3mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=267.6mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=220.5mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=263.6mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=273.9mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=220.4mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.0mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=223.8mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=271.3mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=302.9mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=325.3mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=259.6mV  force_ic=ok/ok |
| tsmc16 | switchcap | charge_err=2.76% | charge_err=1.93% | charge_err=2.01% | charge_err=3.09% | charge_err=2.93% | charge_err=2.68% | charge_err=6.26% | charge_err=2.92% | charge_err=3.08% | charge_err=3.15% | charge_err=2.94% | charge_err=3.15% | charge_err=2.30% |

### Complex-gate matrix — size = medium

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | opamp | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |
| tsmc5 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc5 | switchcap | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | ring_osc | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | switchcap | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| tsmc12 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc12 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc12 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS |
| tsmc12 | switchcap | PASS | PASS | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS |
| tsmc16 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc16 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | PASS | FAIL |
| tsmc16 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| tsmc16 | switchcap | PASS | PASS | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |

#### Gate headline detail — size = medium (period_err% / gain_err% / charge_err% / SNM)

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | period_err=5.89% | period_err=4.58% | period_err=4.64% | period_err=6.02% | period_err=5.38% | period_err=10.16% | period_err=9.30% | period_err=4.67% | period_err=2.69% | period_err=6.70% | period_err=8.21% | period_err=7.85% | period_err=7.21% |
| tsmc5 | opamp | gain_err=100.00% trip_shift=74.00mV | gain_err=100.38% trip_shift=-150.00mV | gain_err=100.00% trip_shift=128.00mV | gain_err=100.00% trip_shift=70.00mV | gain_err=0.48% trip_shift=-50.00mV | gain_err=101.00% trip_shift=-150.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=68.00mV | gain_err=100.00% trip_shift=80.00mV | gain_err=100.00% trip_shift=130.00mV | gain_err=100.00% trip_shift=124.00mV | gain_err=0.80% trip_shift=-12.00mV | gain_err=100.00% trip_shift=126.00mV |
| tsmc5 | sram_snm | NG_SNM=189.2mV DN_SNM=180.1mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=176.3mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=178.7mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=179.8mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=177.1mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=172.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=179.6mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=174.6mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=176.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=172.3mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.0mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=194.4mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=177.8mV  force_ic=FAIL/FAIL |
| tsmc5 | switchcap | charge_err=1.56% | charge_err=0.90% | charge_err=0.89% | charge_err=1.59% | charge_err=2.04% | charge_err=2.82% | charge_err=9.34% | charge_err=1.23% | charge_err=0.09% | charge_err=2.26% | charge_err=1.86% | charge_err=2.24% | charge_err=2.46% |
| tsmc7 | ring_osc | period_err=10.86% | period_err=2.89% | period_err=2.89% | period_err=10.66% | period_err=11.83% | period_err=8.51% | period_err=34.58% | period_err=9.03% | period_err=7.36% | period_err=8.47% | period_err=8.85% | period_err=8.66% | period_err=10.07% |
| tsmc7 | opamp | gain_err=99.99% trip_shift=54.00mV | gain_err=99.99% trip_shift=130.00mV | gain_err=99.99% trip_shift=120.00mV | gain_err=100.00% trip_shift=-2.00mV | gain_err=99.99% trip_shift=134.00mV | gain_err=99.99% trip_shift=54.00mV | gain_err=100.00% trip_shift=-34.00mV | gain_err=99.99% trip_shift=94.00mV | gain_err=99.99% trip_shift=120.00mV | gain_err=100.00% trip_shift=2.00mV | gain_err=13.33% trip_shift=-128.00mV | gain_err=122.58% trip_shift=-148.00mV | gain_err=121.99% trip_shift=-148.00mV |
| tsmc7 | sram_snm | NG_SNM=187.4mV DN_SNM=268.3mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=279.7mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=243.2mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=214.8mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=187.6mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=218.4mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=239.4mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=190.9mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=221.2mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=263.0mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=206.5mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=192.8mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=282.0mV  force_ic=FAIL/FAIL |
| tsmc7 | switchcap | charge_err=2.81% | charge_err=1.79% | charge_err=1.78% | charge_err=2.84% | charge_err=2.68% | charge_err=2.72% | charge_err=0.75% | charge_err=2.72% | charge_err=2.54% | charge_err=2.75% | charge_err=2.75% | charge_err=2.63% | charge_err=2.90% |
| tsmc12 | ring_osc | period_err=2.26% | period_err=3.83% | period_err=3.86% | period_err=2.24% | period_err=2.31% | period_err=2.11% | period_err=nan% | period_err=2.32% | period_err=2.04% | period_err=2.08% | period_err=2.18% | period_err=2.14% | period_err=2.04% |
| tsmc12 | opamp | gain_err=100.00% trip_shift=-10.00mV | gain_err=110.22% trip_shift=-148.00mV | gain_err=100.00% trip_shift=146.00mV | gain_err=100.00% trip_shift=-78.00mV | gain_err=100.00% trip_shift=132.00mV | gain_err=99.99% trip_shift=148.00mV | gain_err=100.00% trip_shift=-28.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=4.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=111.84% trip_shift=0.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=148.00mV |
| tsmc12 | sram_snm | NG_SNM=239.1mV DN_SNM=226.2mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=276.4mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=245.2mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.0mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.2mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.3mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.6mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=227.3mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.2mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.0mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.6mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.5mV  force_ic=ok/ok |
| tsmc12 | switchcap | charge_err=4.19% | charge_err=2.54% | charge_err=2.54% | charge_err=4.15% | charge_err=4.16% | charge_err=4.10% | charge_err=52.49% | charge_err=4.05% | charge_err=3.45% | charge_err=4.19% | charge_err=4.27% | charge_err=4.01% | charge_err=4.23% |
| tsmc16 | ring_osc | period_err=2.22% | period_err=4.00% | period_err=3.99% | period_err=2.22% | period_err=2.23% | period_err=2.22% | period_err=nan% | period_err=2.24% | period_err=2.40% | period_err=2.28% | period_err=2.17% | period_err=2.18% | period_err=2.21% |
| tsmc16 | opamp | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=-22.00mV | gain_err=111.12% trip_shift=-148.00mV | gain_err=100.00% trip_shift=6.00mV | gain_err=12.37% trip_shift=0.00mV | gain_err=99.99% trip_shift=2.00mV | gain_err=4.74% trip_shift=0.00mV | gain_err=3.26% trip_shift=0.00mV | gain_err=100.00% trip_shift=14.00mV | gain_err=4.02% trip_shift=0.00mV | gain_err=100.00% trip_shift=150.00mV | gain_err=3.90% trip_shift=0.00mV | gain_err=100.00% trip_shift=-148.00mV |
| tsmc16 | sram_snm | NG_SNM=235.5mV DN_SNM=222.1mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=221.5mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=313.9mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.6mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=221.8mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.1mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=223.1mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=223.5mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.4mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=221.7mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.6mV  force_ic=FAIL/FAIL |
| tsmc16 | switchcap | charge_err=3.22% | charge_err=2.01% | charge_err=2.01% | charge_err=3.24% | charge_err=3.23% | charge_err=3.31% | charge_err=50.59% | charge_err=3.23% | charge_err=3.46% | charge_err=3.17% | charge_err=3.41% | charge_err=3.32% | charge_err=3.17% |

### Complex-gate matrix — size = large

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | opamp | PASS | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | PASS |
| tsmc5 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc5 | switchcap | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | ring_osc | PASS | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | switchcap | PASS | PASS | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc12 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc12 | opamp | PASS | PASS | FAIL | PASS | PASS | PASS | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc12 | sram_snm | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| tsmc12 | switchcap | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc16 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc16 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | FAIL | FAIL | PASS | FAIL |
| tsmc16 | sram_snm | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| tsmc16 | switchcap | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |

#### Gate headline detail — size = large (period_err% / gain_err% / charge_err% / SNM)

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | period_err=12.66% | period_err=5.15% | period_err=4.73% | period_err=12.46% | period_err=13.49% | period_err=10.31% | period_err=16.83% | period_err=6.10% | period_err=5.80% | period_err=11.57% | period_err=11.83% | period_err=9.64% | period_err=12.64% |
| tsmc5 | opamp | gain_err=2.10% trip_shift=-50.00mV | gain_err=100.00% trip_shift=120.00mV | gain_err=100.00% trip_shift=116.00mV | gain_err=0.67% trip_shift=-52.00mV | gain_err=100.00% trip_shift=110.00mV | gain_err=100.00% trip_shift=32.00mV | gain_err=101.73% trip_shift=-150.00mV | gain_err=1.58% trip_shift=0.00mV | gain_err=100.00% trip_shift=14.00mV | gain_err=0.43% trip_shift=0.00mV | gain_err=100.00% trip_shift=-14.00mV | gain_err=100.00% trip_shift=118.00mV | gain_err=0.15% trip_shift=0.00mV |
| tsmc5 | sram_snm | NG_SNM=189.2mV DN_SNM=181.0mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=178.1mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=177.7mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=182.6mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=179.5mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=183.4mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=178.3mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=180.5mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.9mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=179.0mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=182.8mV  force_ic=FAIL/FAIL |
| tsmc5 | switchcap | charge_err=3.48% | charge_err=0.89% | charge_err=0.89% | charge_err=3.47% | charge_err=3.31% | charge_err=2.92% | charge_err=9.19% | charge_err=2.32% | charge_err=2.36% | charge_err=3.23% | charge_err=3.40% | charge_err=2.53% | charge_err=3.56% |
| tsmc7 | ring_osc | period_err=4.82% | period_err=2.88% | period_err=2.87% | period_err=7.32% | period_err=10.40% | period_err=5.09% | period_err=10.13% | period_err=6.70% | period_err=4.51% | period_err=6.65% | period_err=7.15% | period_err=8.69% | period_err=7.15% |
| tsmc7 | opamp | gain_err=99.99% trip_shift=114.00mV | gain_err=28.35% trip_shift=-148.00mV | gain_err=99.99% trip_shift=42.00mV | gain_err=99.99% trip_shift=44.00mV | gain_err=99.99% trip_shift=42.00mV | gain_err=99.99% trip_shift=66.00mV | gain_err=161.90% trip_shift=-58.00mV | gain_err=99.99% trip_shift=128.00mV | gain_err=99.99% trip_shift=126.00mV | gain_err=99.99% trip_shift=98.00mV | gain_err=126.77% trip_shift=-148.00mV | gain_err=121.20% trip_shift=-148.00mV | gain_err=99.99% trip_shift=112.00mV |
| tsmc7 | sram_snm | NG_SNM=187.4mV DN_SNM=315.2mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=230.1mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=261.2mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=240.5mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=293.8mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=187.6mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=182.4mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=197.0mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=192.1mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=277.8mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=199.5mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=193.3mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=184.0mV  force_ic=ok/ok |
| tsmc7 | switchcap | charge_err=2.45% | charge_err=1.77% | charge_err=1.77% | charge_err=2.45% | charge_err=2.59% | charge_err=2.42% | charge_err=4.01% | charge_err=2.29% | charge_err=2.29% | charge_err=2.44% | charge_err=2.35% | charge_err=2.50% | charge_err=2.42% |
| tsmc12 | ring_osc | period_err=4.04% | period_err=3.85% | period_err=3.84% | period_err=4.05% | period_err=4.02% | period_err=2.12% | period_err=5.68% | period_err=2.24% | period_err=2.16% | period_err=2.15% | period_err=3.38% | period_err=2.83% | period_err=4.19% |
| tsmc12 | opamp | gain_err=6.25% trip_shift=0.00mV | gain_err=4.02% trip_shift=108.00mV | gain_err=16.10% trip_shift=-148.00mV | gain_err=0.35% trip_shift=0.00mV | gain_err=5.71% trip_shift=0.00mV | gain_err=5.82% trip_shift=0.00mV | gain_err=99.99% trip_shift=104.00mV | gain_err=7.01% trip_shift=0.00mV | gain_err=100.00% trip_shift=146.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=-18.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=148.00mV |
| tsmc12 | sram_snm | NG_SNM=239.1mV DN_SNM=226.9mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=302.4mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=244.6mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.9mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=227.0mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.9mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.9mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.6mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.7mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.6mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.6mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.7mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.7mV  force_ic=FAIL/FAIL |
| tsmc12 | switchcap | charge_err=4.14% | charge_err=2.54% | charge_err=2.54% | charge_err=4.15% | charge_err=4.14% | charge_err=4.08% | charge_err=5.17% | charge_err=4.05% | charge_err=4.19% | charge_err=4.08% | charge_err=4.14% | charge_err=4.11% | charge_err=4.12% |
| tsmc16 | ring_osc | period_err=2.59% | period_err=4.30% | period_err=4.00% | period_err=3.40% | period_err=4.01% | period_err=3.24% | period_err=3.61% | period_err=2.31% | period_err=2.19% | period_err=2.25% | period_err=2.41% | period_err=3.40% | period_err=2.43% |
| tsmc16 | opamp | gain_err=100.00% trip_shift=-60.00mV | gain_err=nan% trip_shift=nanmV | gain_err=79.15% trip_shift=86.00mV | gain_err=100.00% trip_shift=-54.00mV | gain_err=100.00% trip_shift=-30.00mV | gain_err=1.28% trip_shift=0.00mV | gain_err=99.99% trip_shift=136.00mV | gain_err=100.00% trip_shift=138.00mV | gain_err=5.41% trip_shift=0.00mV | gain_err=100.00% trip_shift=-24.00mV | gain_err=100.00% trip_shift=-30.00mV | gain_err=6.16% trip_shift=0.00mV | gain_err=100.00% trip_shift=-30.00mV |
| tsmc16 | sram_snm | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=326.5mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=890.4mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=223.0mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=223.3mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.6mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.6mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.8mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.7mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=223.0mV  force_ic=ok/ok |
| tsmc16 | switchcap | charge_err=3.32% | charge_err=2.01% | charge_err=2.01% | charge_err=3.33% | charge_err=3.27% | charge_err=3.35% | charge_err=7.26% | charge_err=3.25% | charge_err=3.17% | charge_err=3.30% | charge_err=3.29% | charge_err=3.31% | charge_err=3.30% |

### Complex-gate matrix — size = xl

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | PASS | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | opamp | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | PASS | FAIL |
| tsmc5 | sram_snm | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc5 | switchcap | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | ring_osc | FAIL | ? | ? | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | sram_snm | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc7 | switchcap | PASS | ? | ? | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc12 | ring_osc | PASS | ? | ? | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc12 | opamp | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| tsmc12 | sram_snm | PASS | FAIL | FAIL | PASS | PASS | PASS | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS |
| tsmc12 | switchcap | PASS | ? | ? | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc16 | ring_osc | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| tsmc16 | opamp | FAIL | FAIL | FAIL | FAIL | PASS | FAIL | PASS | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |
| tsmc16 | sram_snm | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS |
| tsmc16 | switchcap | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

#### Gate headline detail — size = xl (period_err% / gain_err% / charge_err% / SNM)

| Tech | Circuit | clean | cor | corft | invtripft | invtrip | csob | sob | ekv | csobekv | cs7 | s123 | s17 | s7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | period_err=13.50% | period_err=4.73% | period_err=4.62% | period_err=13.36% | period_err=12.39% | period_err=13.84% | period_err=12.59% | period_err=7.08% | period_err=7.39% | period_err=14.41% | period_err=17.36% | period_err=16.00% | period_err=13.09% |
| tsmc5 | opamp | gain_err=100.00% trip_shift=24.00mV | gain_err=70.99% trip_shift=-146.00mV | gain_err=100.00% trip_shift=72.00mV | gain_err=100.00% trip_shift=40.00mV | gain_err=7.06% trip_shift=-2.00mV | gain_err=100.00% trip_shift=120.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=99.96% trip_shift=96.00mV | gain_err=4.13% trip_shift=0.00mV | gain_err=100.00% trip_shift=0.00mV | gain_err=100.00% trip_shift=14.00mV | gain_err=1.39% trip_shift=0.00mV | gain_err=100.00% trip_shift=-24.00mV |
| tsmc5 | sram_snm | NG_SNM=189.2mV DN_SNM=181.8mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=177.7mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=177.6mV  force_ic=ok/ok | NG_SNM=189.2mV DN_SNM=181.8mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.4mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=182.3mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=177.0mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=174.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=175.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=182.4mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=183.1mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.2mV  force_ic=FAIL/FAIL | NG_SNM=189.2mV DN_SNM=181.8mV  force_ic=FAIL/FAIL |
| tsmc5 | switchcap | charge_err=3.18% | charge_err=0.88% | charge_err=0.88% | charge_err=3.24% | charge_err=3.06% | charge_err=3.45% | charge_err=1.65% | charge_err=1.97% | charge_err=2.09% | charge_err=3.70% | charge_err=3.76% | charge_err=3.48% | charge_err=3.58% |
| tsmc7 | ring_osc | period_err=14.31% | ? | ? | period_err=14.12% | period_err=16.45% | period_err=12.68% | period_err=12.07% | period_err=7.27% | period_err=5.39% | period_err=7.17% | period_err=16.42% | period_err=14.40% | period_err=17.09% |
| tsmc7 | opamp | gain_err=99.99% trip_shift=126.00mV | gain_err=nan% trip_shift=nanmV | gain_err=nan% trip_shift=nanmV | gain_err=99.99% trip_shift=40.00mV | gain_err=17.00% trip_shift=-136.00mV | gain_err=99.99% trip_shift=50.00mV | gain_err=100.00% trip_shift=150.00mV | gain_err=99.99% trip_shift=126.00mV | gain_err=99.99% trip_shift=82.00mV | gain_err=99.99% trip_shift=124.00mV | gain_err=99.99% trip_shift=110.00mV | gain_err=99.99% trip_shift=48.00mV | gain_err=99.99% trip_shift=82.00mV |
| tsmc7 | sram_snm | NG_SNM=187.4mV DN_SNM=195.4mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=229.2mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=244.3mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=281.9mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=283.9mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=227.2mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=229.3mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=207.8mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=205.5mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=339.4mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=210.9mV  force_ic=ok/ok | NG_SNM=187.4mV DN_SNM=191.9mV  force_ic=FAIL/FAIL | NG_SNM=187.4mV DN_SNM=235.8mV  force_ic=ok/ok |
| tsmc7 | switchcap | charge_err=2.67% | ? | ? | charge_err=2.69% | charge_err=2.78% | charge_err=2.65% | charge_err=3.65% | charge_err=2.26% | charge_err=2.18% | charge_err=2.50% | charge_err=2.68% | charge_err=2.74% | charge_err=2.63% |
| tsmc12 | ring_osc | period_err=3.40% | ? | ? | period_err=3.54% | period_err=3.78% | period_err=2.85% | period_err=1.67% | period_err=2.11% | period_err=2.15% | period_err=2.84% | period_err=4.45% | period_err=3.79% | period_err=4.32% |
| tsmc12 | opamp | gain_err=100.00% trip_shift=-8.00mV | gain_err=nan% trip_shift=nanmV | gain_err=nan% trip_shift=nanmV | gain_err=100.00% trip_shift=40.00mV | gain_err=100.00% trip_shift=40.00mV | gain_err=100.00% trip_shift=28.00mV | gain_err=5.44% trip_shift=0.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=100.00% trip_shift=28.00mV | gain_err=99.99% trip_shift=70.00mV | gain_err=100.00% trip_shift=0.00mV | gain_err=100.00% trip_shift=148.00mV |
| tsmc12 | sram_snm | NG_SNM=239.1mV DN_SNM=226.7mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=nanmV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=1106.6mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.9mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.3mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.6mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=FAIL/FAIL | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok |
| tsmc12 | switchcap | charge_err=4.19% | ? | ? | charge_err=4.19% | charge_err=4.15% | charge_err=4.15% | charge_err=3.47% | charge_err=4.05% | charge_err=4.10% | charge_err=4.12% | charge_err=4.19% | charge_err=4.17% | charge_err=4.23% |
| tsmc16 | ring_osc | period_err=3.05% | period_err=4.00% | period_err=4.00% | period_err=3.18% | period_err=2.20% | period_err=3.91% | period_err=2.37% | period_err=2.22% | period_err=2.19% | period_err=3.84% | period_err=3.23% | period_err=3.29% | period_err=3.56% |
| tsmc16 | opamp | gain_err=100.00% trip_shift=-16.00mV | gain_err=nan% trip_shift=nanmV | gain_err=nan% trip_shift=nanmV | gain_err=100.00% trip_shift=2.00mV | gain_err=6.50% trip_shift=0.00mV | gain_err=100.00% trip_shift=150.00mV | gain_err=7.73% trip_shift=0.00mV | gain_err=100.00% trip_shift=4.00mV | gain_err=100.00% trip_shift=44.00mV | gain_err=100.00% trip_shift=34.00mV | gain_err=100.00% trip_shift=148.00mV | gain_err=6.74% trip_shift=0.00mV | gain_err=100.00% trip_shift=148.00mV |
| tsmc16 | sram_snm | NG_SNM=235.5mV DN_SNM=223.0mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=nanmV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=1060.1mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=223.0mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.8mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.8mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.6mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.8mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=FAIL/FAIL | NG_SNM=235.5mV DN_SNM=223.1mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.7mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=222.8mV  force_ic=ok/ok | NG_SNM=235.5mV DN_SNM=223.0mV  force_ic=FAIL/FAIL |
| tsmc16 | switchcap | charge_err=3.42% | charge_err=2.01% | charge_err=2.01% | charge_err=3.42% | charge_err=3.30% | charge_err=3.32% | charge_err=1.46% | charge_err=3.27% | charge_err=3.26% | charge_err=3.33% | charge_err=3.33% | charge_err=3.32% | charge_err=3.24% |

---

# Part B — BSIM-AR complex matrix (V6.8.0 / V6.8.1, single-run)

## Appendix A — full complex matrix (single-run)

> ⚠ **Pre-fix (V6.8.0/V6.8.1) data, kept for provenance.** Every cell below was
> measured with the gds sign bug present. The post-fix matrix is in
> `by-tech.md` §3 and `by-recipe.md` §2, from `results/a3_regate/REPORT.md`.

| recipe/tier | tsmc5 | tsmc7 | tsmc12 | tsmc16 |
|---|---|---|---|---|
| clean/small | O✗ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| clean/medium | O✓ R✗ S✓ C✓ | O✓ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| clean/large | O✓ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corridor/large | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corroft/medium | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corroft/xl | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| crit15m/xl | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corro15/xl | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| clean/xl | O✓ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| csob/xl | O✓ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |

(O=opamp, R=ring_osc, S=sram_snm, C=switchcap; ✓ PASS / ✗ FAIL vs NGSPICE.)

