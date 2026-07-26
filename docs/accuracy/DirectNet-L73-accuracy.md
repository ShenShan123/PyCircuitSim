# DirectNet (LEVEL=73) — family report

**What it is:** a feed-forward MLP compact model, netlist `LEVEL=73` — the
**production** NN fast path. 7-dim input (Vgs, Vds, Vbs, NFIN, L, T, tech_code
via `nn.Embedding`), 13 outputs; `gm`/`gds`/`gmb` are the **autograd Jacobian**
of the predicted `id`, and the AC capacitances are the `dQ/dV` autograd of the
predicted terminal charges.

**Status:** production = `crit30f@large`, **15/16 strict** across OMP ∈ {1,2,4}
with zero flips, at 0.92 M params and ~1.5 ms/eval.

Cross-cutting numbers live in the axis files — [`by-tech.md`](by-tech.md),
[`by-scale.md`](by-scale.md), [`by-recipe.md`](by-recipe.md) — and gate
definitions in [`methodology.md`](methodology.md). This file carries what is
specific to DirectNet, including the universal-scope study, which exists for no
other family.

**Covers:** V6.6.0 → V6.7.0 (all DirectNet accuracy campaigns) → V6.11.0 →
V6.13.0 (post-gds-fix re-gate) → V7.1.0 (device/AC re-gate). Frozen pre-fix data
tables: [`archive-pre-gds-fix.md`](archive-pre-gds-fix.md).

---

## 1. Production state

| metric | result |
|---|---|
| **Production checkpoint** | `tsmc{5,7,12,16}_dn_large_{nmos,pmos}` = the **crit30f** curriculum artifact (V6.6.4) |
| **Complex-circuit gates** | **15/16 PASS, strict across OMP ∈ {1,2,4}, zero FLIPs** |
| Open cell | **`tsmc7-opamp`, and only at `large`** — DirectNet passes it at `small` (1.81 %) and `xl` (4.20 %) |
| Device DC (`verify_nn_dc_tran`, resolver default) | 24/24 PASS; NMOS 1.53 % / PMOS 0.02 % NRMSE |
| Parametric DC / transient | 54/55 · 64/64 (the one DC fail is **bit-identical** pre/post-fix) |
| Inverter VTC + transient | 16/16 PASS at every size |
| Lifted-source canary (Rule 2) | 12/12 PASS (NRMSE ≤10 %) |
| Device AC (CS-amp) | **8/8 PASS** |
| Params / cost | large = 384×6 ≈ **0.92 M**, ~1.5 ms/eval (CPU, 1 thread) |

Cell margins at production (identical at all three OMP settings):
`tsmc5-opamp` 9.54 %, `tsmc12-opamp` 6.26 %, `tsmc16-opamp` 7.69 % (gate ≤10 %);
rings 2.40–4.04 % (gate ≤5 %); switchcap 2.04–4.17 % of VDD (gate ≤5 %); SRAM
lobes all positive, worst lobe NRMSE 6.32 %. **`tsmc5-opamp` at 9.54 % is the
thinnest margin in the matrix** — banked, but one recipe change from the gate.

**Production recipe (`crit30f`):** clean base + one identical curriculum
fine-tune — `--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4
--epochs 120 --patience 40 --init-from <own clean large>` on the ring-only
`corro` datasets. The clean originals are archived as
`tsmc{X}_dn_v660clean_large_*`; small / medium / xl stay clean.

### Documented alternates (env-pin only, never the resolver default)

| alternate | pin | why |
|---|---|---|
| **`crit15m@xl`** | `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}=tsmc{X}_dn_crit15m_xl_{dev}` | **16/16 strict, zero flips** — the only DirectNet stem that sweeps the matrix. Not promoted: 2.13 M params, 2.3× inference cost, no device-fidelity gain |
| `corroft@xl` | `…=tsmc{X}_dn_corroft_xl_{dev}` | 15/16 strict; deterministic `tsmc16-opamp` (6.69 %) |
| `csob@large` | `…=tsmc{X}_dn_csob_large_{dev}` | Best device NRMSE and charge-axis fidelity. ⚠ **Complex-gate rationale withdrawn** — 11/16 post-fix, and it now fails `tsmc16-opamp`, which production banks. Pin for device/AC work only |
| ~~`crit10@xl`~~ | — | ⚠ **Withdrawn** — post-fix it fails `tsmc16-opamp` outright (gain err 100 %), the exact cell it was documented to cover |
| `u716_dn_corroft_large` | `…=u716_dn_corroft_large_{dev}` | One-checkpoint multi-tech serving, 10/12 strict (§3) |

## 2. How production came to be — the curriculum breakthrough (V6.6.0 → V6.6.4)

V6.6.0 reset the V6.5.9 "16/16", which had been reached only by per-tech bespoke
interventions (tsmc5 ring corridor + seed 7, tsmc16 seed 17, tsmc7 T3
differentiable-DC-solver on an EKV core). Those answer *"can a hand-tuned
checkpoint pass this gate?"*, not *"how faithful is the model under one
recipe?"*. The honest uniform-recipe number was **13/16 (clean@large)**.

Nine uniform single-lever recipes, three seeds and two combos then failed to beat
it (`by-recipe.md` §5) — the ceiling was **mutually-exclusive value-surface
basins**: each recipe landed exactly two of the four opamp gates, a *different*
two. The corridor (`traj_corridor`) and inverter-trip (`inv_trip`) class weights
had only ever been tested **separately**, on opposite sides of that wall.
Combining them in a warm-start curriculum on the ring-only `corro` data broke it:

| step | recipe | result |
|---|---|---|
| V6.6.2 | `crit15` (corridor 1.5 + inv_trip 2.0) | 14/16 single-run, **13/16 strict** = clean + 1; the +1 is the deterministic tsmc5-ring opening (12.66 → 4.0 %) |
| V6.6.3 | `crit30` (corridor 3.0 + inv_trip 2.0) | **14/16 STRICT all-OMP** — best of all 22 on-disk recipes |
| V6.6.3 | `crit30f` | full-spec retrain (the original had been killed at heterogeneous epochs 30–92); reproduces the artifact cell for cell |
| V6.6.4 | **`crit30f` PROMOTED** | all 8 `tsmc{X}_dn_large_*` production slots replaced; clean archived as `v660clean_large` |
| V6.13.0 | *(same weights, gds fix)* | **15/16 strict**, adding `tsmc16-opamp` at 7.69 % |

`crit30` banks deterministically: all 4 rings (tsmc5 12.66 → 4.04 %), all SRAM +
switchcap, `tsmc12-opamp` 6.25 % and **`tsmc5-opamp` 0.21 %** (a coin-flip under
clean, det-FAIL under `crit15`). Device level ≥ clean everywhere (DC mean NRMSE
1.64 → 1.46 %).

**Two durable lessons:**

1. **A single-run opamp pass is unbankable.** tsmc5/tsmc16 opamps flipped
   0–8 % ↔ 100 % across OMP ∈ {1,2,4} under both clean and `crit15`.
   `opamp_sweep_def.sh` became the standing probe because of this. *(Post-fix the
   flips are gone — but the discipline is what surfaced the gds bug.)*
2. **The corridor-weight → basin map is NON-MONOTONE.** For tsmc5-opamp: w1.0
   FLIP, w1.5/2.0 det-FAIL, w3.0 det-PASS. The `inv_trip` anchor is what makes
   w3.0 safe where `corroft` alone railed.

## 3. Universal DirectNet + TSMC5 transfer (V6.7.0)

One 18-code-embedding DirectNet trained on TSMC16 + 12 + 7 concatenated
(`uni716_{dev}.npz`, nmos 6.92 M / pmos 7.26 M rows). **Production impact: none**
— all artifacts use `u716_*` / `u716f5_*` stems reachable only by explicit env
pin; the per-tech resolver cascade is untouched. Calibration bar: the per-tech
recipes all score 10/12 strict on the same 12 gates ({ring, opamp, SRAM, SC} ×
{TSMC7, 12, 16}); 11/12 is the realistic ceiling. Zero RESOLVER-MISS across 200
cells.

### 3.1 Universal recipe ranking (12 gates, strict OMP ∈ {1,2,4})

Post-fix re-gate of all 8 universal stems (`results/a3_regate_uni/REPORT.md`):

| stem | pre-fix strict | post-fix strict | Δ |
|---|---|---|---|
| `u716_dn_clean_xl` | 8/12 (1 FLIP) | **10/12** | **+2** |
| `u716_dn_corroft_xl` | 8/12 | **9/12** | **+1** |
| `u716_dn_crit30u_large` | 9/12 (1 FLIP) | **10/12** | **+1** |
| `u716_dn_csob_large` | 8/12 (1 FLIP) | **9/12** | **+1** |
| **`u716_dn_corroft_large`** | **10/12** | **10/12** | 0 |
| `u716f5_plain_nfull_large` | 3/4 | 3/4 | 0 |
| `u716_dn_clean_large` | 9/12 | **8/12** | **−1** |
| `u716f5_plain_n1000000_large` | 4/4 | **3/4** | **−1** |
| | | **net** | **+3** |

**All three pre-existing OMP FLIPs are gone; every stem is flip-free** — the more
durable result than the +3. Rings did not move at any stem, consistent with
rings being gds-invariant.

**Universal is viable:** device fidelity matches per-tech (id NRMSE ≤0.09 % per
variant, R² ≥0.996) and `corroft` **ties the per-tech calibration bar with zero
OMP flips**, which per-tech `large` never achieved pre-fix. The corridor is the
ring lever at universal scope too (tsmc7-ring 14.89 % → 3.61 %).

Pre-fix detail that still holds: at universal `xl` the mutual-exclusive-basin
wall reappears partitioned **opamps-XOR-rings** — `clean@xl` banks tsmc12- *and*
tsmc16-opamp but fails ALL rings; `corroft@xl` holds rings 3/3 but rails ALL
opamps. **`corroft@large` stands.**

### 3.2 TSMC5 onboarding by fine-tune — sample efficiency

| tier N | plain | crit |
|---|---|---|
| 2k / 10k / 50k / 200k | 0/4 | 0/4 |
| **1M** | **4/4 — STRICT** (3/4 post-fix) | 1/4 (ring only) |
| full (~2.02 M) | 3/4 (opamp det-FAIL) | 3/4 (opamp FAIL) |

Device level (DC NRMSE %, nmos — DC is gds-invariant, so these stand):

| tier | plain | crit | note |
|---|---|---|---|
| 2k | unusable | n/a | model wrecked by the norm refit on 1.8k train rows |
| 10k | ~1e69 (DIVERGED) | ~1e69 | normalizer-refit failure mode, not noise |
| 50k | 35.1 (ERROR) | 35.1 (ERROR) | |
| 200k | 10.1 FAIL | 5.9 PASS | `crit`'s class weights help at moderate N |
| 1M | 0.69 PASS | 0.57 PASS | |
| full | 0.42 PASS | 1.08 PASS | pmos full 0.64 / tran 0.52 / inv-tran 1.27 (plain) |

* **The gate threshold lags the device threshold** — 5.9 % DC NRMSE is a device
  PASS and nowhere near enough for complex gates; the value surface needs the
  ~1M-row tightening.
* **n1M > nfull at gates**, i.e. non-monotone in N: more TSMC5 data shifted the
  opamp basin *out*. Tier selection is itself a basin lottery.
* Zero-shot on untrained embedding rows: 0/4 — no free lunch from the shared trunk.

### 3.3 Retention — no free lunch

TSMC12 DC NRMSE % after a TSMC5-only fine-tune (the worst source tech):

| tier | plain | crit |
|---|---|---|
| 2k | 5.8 PASS | 5.8 PASS |
| 10k | 7.9 PASS | 7.8 PASS |
| 50k | 11.0 FAIL | 10.1 FAIL |
| 200k | 17.5 FAIL | 23.0 FAIL |
| 1M | **26,424 FAIL (blow-up)** | 5.9 PASS |
| full | 18.0 FAIL | 23.0 FAIL |

At gate level (12 retention gates, plain): n1M keeps 1/12, full keeps 3/12.
Forgetting grows with fine-tune N and is high-variance. Without replay the
fine-tune converts the universal model into a de-facto per-tech model:
**universal-base fine-tuning is a cheap onboarding path, not a multi-tech serving
path.**

## 4. What the AC gates actually diagnose

DirectNet's small-signal capacitances are autograd derivatives of its predicted
terminal charges (`cgd = ∂qg/∂Vd`, …), gated against NGSPICE `.ac` on the
identical L72 model. Pass counts by tier are in `by-scale.md` §5; the *diagnosis*
below survived the gds fix intact.

**What is right:** DC gain is excellent everywhere — device gain0 error <1.5 dB
in 24/24 cells (mean 0.86 dB), so the autograd `gm`/`gds` feeding the AC stamp
are accurate. The dominant cap-driven pole is faithful for well-fit cells
(f3db ratio ≈ 1.0).

**The genuine limits:**

1. The **Cgd-feedforward RHP zero** (high-frequency phase lag) is not reproduced
   — a clean, specific transcapacitance limitation. It is *reported* as a
   diagnostic in `verify_nn_ac` and deliberately not gated (phase deep beyond the
   corner is dominated by it).
2. Some cells **under-predict the output capacitance** (tsmc5 NMOS, tsmc12/16
   PMOS, f3db ratio 1.1–1.6) and miss the magnitude gate.
3. The opamp AC inherits the DC value-surface fragility — where the OP is in the
   good basin the NN nails GBW (0.97×) and phase margin (1.3°), confirming the
   *dynamics* are right and only the DC-gain *level* is the miss.

These are value-surface / feedforward limits, **not** a charge-derivative
deficiency — which would corrupt gain *and* pole everywhere.

## 5. Open frontier

| open | headline | reading |
|---|---|---|
| **`tsmc7-opamp` at `large`** | gain err 100 %, `vout` railed | Tier-specific basin, not a family-wide wall: DirectNet banks it at `small` (1.81 %) and `xl` (4.20 %), BSIM-AR at every size |
| **`tsmc5-opamp` margin** | 9.54 % against a 10 % gate | The thinnest margin in the production matrix; treat any recipe change as a threat to it |
| **Opamp open-loop AC** | see `by-scale.md` §5 | The OP un-rails post-fix; the full GBW/PM/magnitude gate is a separate question |
| **Low-VDD rings under clean recipes** | 5.9–13.1 % vs a 5 % gate | Closed only by the corridor curriculum; every clean tier fails at least one |

`tsmc16-opamp` **closed** in V6.13.0 (7.69 % strict) and is no longer open.

These remain **value-surface / fixed-point** properties (open-loop gain,
oscillation period at a sharp edge) — not device-current or charge-derivative
fidelity gaps. Device DC, inverter, switchcap, SRAM and AC-gain are all strong.

## 6. Artifacts and reproduction

```bash
# train one recipe wave
RECIPES="corroft crit30" SIZES=large GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
# gate it (isolated, CPU-pinned)
SIZE=large bash scripts/gate_matrix_iso.sh
# strict OMP determinism probe, one cell
bash scripts/recipe_multirun_gate.sh crit30 large TSMC16 verify_complex_opamp
```

* **Checkpoints** (gitignored): production `tsmc{X}_dn_large_{dev}`;
  `v660clean_large` (warm-start base) + `crit30f_large` (production provenance);
  alternates `csob@large`, `corroft`/`crit10`/`crit15m`@xl; universal
  `u716_dn_{clean,csob,corroft,crit30u}_large` + `_{clean,corroft}_xl` + TSMC5
  fine-tunes `u716f5_plain_n{1000000,full}_large`. Retired sets archived to
  `/data2/shenshan/v66x_v670_retired_ckpts_2026-07-05.tar.gz` and
  `/data2/shenshan/v6.5.9_production_specials.tar.gz`.
* **Raw runs:** `results/a3_regate/` (V6.13.0), `results/a3_regate_uni/`
  (universal), `results/v710_regate/` (V7.1.0), `results/recipe_bench/`
  (pre-fix matrix), `results/uni_bench/`.
* **Frozen pre-fix tables:** `archive-pre-gds-fix.md`. Machine-readable
  `results/recipe_bench/{retest_data,recipe_data}.json`.
* **Plans / narrative:** `docs/plans/2026-07-02-universal-nn-tsmc5-transfer.md`,
  `docs/CHANGELOG.md` (V6.6.0 → V7.1.0).
