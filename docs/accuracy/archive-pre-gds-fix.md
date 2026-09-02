# Archive — the register of retracted claims

This file used to carry ~660 lines of frozen pre-fix data tables as well. They
were **removed in V7.3.0, not lost**: every table is in git history and prints
with

```bash
git show 37cef77:docs/accuracy/archive-pre-gds-fix.md
```

They were dropped because they can neither be used nor re-measured. Every one
was taken with the `gds` sign bug present (`methodology.md` §6), and most of the
checkpoints they describe no longer exist on disk — the V6.6.x recipe matrix
(208 checkpoints) and the V6.6.6 xl curriculum wave (72) were archived to
`/data2/shenshan/v66x_v670_retired_ckpts_2026-07-05.tar.gz` and
`/data2/shenshan/v6.5.9_production_specials.tar.gz`. Recipes with no surviving
checkpoint are also exactly the ones the V7.3.0 filter drops
(`*-recipes.md` §dead ends), so refreshing the tables was never an option and
keeping them invited citation.

**What survives here is the part that is still load-bearing:** which published
claims the re-gates killed, so nobody rediscovers them from an old plan or
CHANGELOG entry.

## If you do read a pre-fix number

Per §6's measured invariance:

| axis | pre-fix number still valid? |
|---|---|
| device **DC** (Id-Vgs NRMSE / MRE / R²) | **yes** — DC is exactly invariant, bit-identical |
| **ring**, **SRAM**, **switchcap** cells | **yes** — not one such cell moved in the re-gate |
| **opamp** cells | **no** — every cell the fix gained was an opamp |
| any **total** containing an opamp column | **no** — systematically pessimistic |
| **AC** (device CS-amp, opamp open-loop) | **no** — the axis the fix moved most |
| **transient** | partially — mean NRMSE improved 1.876 % → 1.512 % |
| OMP **FLIP** classifications | **no** — re-measure before believing one |

Beyond the bug, a second correction applies to *any* pre-V7.3.0 total:
denominators were **/16, /8, /4** over four techs, and are now **/20, /10, /5**
over five. A "13/16" and a "16/20" can be the same measurement.

---

## Register of claims retracted by the re-gates

| retracted claim | where it came from | what replaced it |
|---|---|---|
| "**tsmc7-opamp** is the universal ceiling cell for all three families — no capacity, tier, scope, seed or data recipe has ever reached it; only the V6.5.9 T3 differentiable-DC-solver fine-tune did." | DirectNet §6.2, BSIM-AR §4/§6, cross-family tables, CLAUDE.md | The `gds` floor was holding a railed operating point. V7.4 clean DirectNet passes it at every tier; clean BSIM-AR passes at small/medium/xl and rails with its TSMC6 twin at `large`, exposing the remaining training-basin variance. |
| "**PFN is the only flip-free family.**" | PFN §1/§10 | True when measured, then overtaken: after the fix every family was flip-free. **Now doubly retracted** — the V7.1.0 strict sweep recorded a flip at two PFN tiers, so "every group is flip-free" is also too strong. Treat a nonzero flip count as unbankable and re-measure. |
| "**BSIM-AR beats DirectNet by one cell** (tsmc16-opamp)." | cross-family tables | The one-cell framing is retired. V7.4 clean BSIM-AR `small` scores 18/20, versus DirectNet's current `large` 14/20 and best clean `xl` 15/20; historical recipes belong to V7.3 controls. |
| "BSIM-AR **capacity peaks at medium**." | BSIM-AR §3 | V7.4 clean BSIM-AR declines **18→17→15→13/20** from small→xl. The intermediate "flat across tiers" correction from V7.3 is itself retracted. |
| "The **strict best is medium, not large**" (BSIM-AR). | BSIM-AR §5 | It rested on an OMP FLIP that no longer exists. V7.4 clean best is `small`; V7.3 recipe best is shared by six 20/20 corridor groups. |
| "The **three-basin simultaneous hold (5+12+16) is the open target**." | DirectNet §6.2 | `crit15m@xl` holds all **four** opamp basins with no recipe change. |
| "`csob@large` is the documented complex-gate alternate." | DirectNet alternates | **Withdrawn** — post-fix it is the campaign's only regression and now fails the `tsmc16-opamp` production banks. Still the device/AC alternate. |
| "`crit10@xl` covers tsmc16-opamp." | DirectNet alternates | **Withdrawn** — post-fix it fails that exact cell. Superseded by `crit15m@xl`. |
| "**AC pass-rate peaks at SMALL**" — a dQ/dV pole property wanting the opposite capacity to DC fixed points. | DirectNet §8, BSIM-AR §7, PFN §7 | **Retracted.** Device CS-amp AC is saturated at every capacity. The pre-fix reading had both the level and the shape wrong. |
| "The opamp **open-loop AC** gate is 0/4 at every tier for every family." | DirectNet §8/§12.1, BSIM-AR §6, PFN §7 | **Falsified.** V7.5.16 corrected the gate's bias resolution and converged-OP contract; the current DirectNet/BSIM-AR reports own the replacement verdict, while PFN remains V7.3 historical evidence. |
| "**AC collapses at xl**" (BSIM-AR). | V6.8.1 | **Retracted.** Post-fix `xl` banks the TSMC16 opamp-AC cell, and the `tsmc7-opamp-AC` run recorded as never converging after ~6 h now completes. That pathology was the railed OP. |
| "BSIM-AR is the only family to bank the **tsmc6 opamp**, while tsmc7-opamp is the universal ceiling." | V6.11.0 | Void by construction: TSMC6 **is** TSMC7 (`methodology.md` §7), so it was the same measurement quoted twice. |
| "TSMC6 is a sixth technology." | V6.9.0 onboarding | **Retracted.** TSMC6 is TSMC7 relabelled, now on four independent lines of evidence. Its 9/9 DC and 14/14 transient onboarding gates told us nothing — they were TSMC7's gates. |
| "The **recipe decides which opamp basin you get**" (BSIM-AR). | V6.6.x recipe study | Retracted **for BSIM-AR only**: post-fix its `large` curricula agree with each other and its `xl` curricula agree with each other, so the recipe discriminates only on rings there. It still discriminates on opamps in DirectNet, same tier, same data. |
| "**`crit15m@xl` sweeps the matrix**" — DirectNet's full-sweep stem. | V6.13.0 re-gate, DirectNet alternates | **Retracted (V7.3.0).** It sweeps four techs and fails the fifth — `tsmc6-opamp`, whose data is `array_equal` to the `tsmc7-opamp` it passes. Nothing separates the two cells but the training run and Newton basin. Six historical V7.3 BSIM-AR corridor groups reached 20/20 while holding both copies; none was retrained or promoted in V7.4. |

## A note on why this register exists

Roughly half of the recipe rankings this project published between V6.4 and
V6.12 were ranking a wrong-signed Jacobian entry, and the other half were
ranking cells now known to carry ±4 pp of run-to-run scatter across a 5 % gate.

The sharpest single illustration: the V6.11.0 TSMC6 run and the V6.13.0 TSMC7
re-gate disagreed by **68.2 % vs 2.0 %** on SRAM SNM error at NFIN=2 — a gap the
project had been reading as per-tech difficulty. On identical data after the
fix, the same comparison lands at **5.2 % vs 6.2 %**. A 66.2 pp gap collapsed to
1.0 pp. *Part of the "training lottery" that recipes were ranked against was the
wrong Jacobian, not stochasticity.*
