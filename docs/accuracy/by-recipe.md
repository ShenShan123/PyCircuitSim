# Accuracy by recipe — what each training addendum actually buys

Cross-family view along the **recipe** axis. Companion pivots: `by-tech.md`
(technology), `by-scale.md` (capacity). Shared definitions and the code-state
ladder: `methodology.md`.

A **recipe** is one identical training addendum applied to every
(tech × device × size) checkpoint — never a per-tech or per-gate special
(`methodology.md` §5). Everything below therefore compares like with like: the
same data pipeline, the same 16 gates, one flag change.

---

## 1. The catalogue

Authoritative source: the `recipe_args` map in `scripts/recipe_train.sh`.
Curriculum recipes are 120-epoch fine-tunes at lr 3e-4, patience 40,
**warm-started from their own tier's clean checkpoint**, on a corridor dataset.

### Single-lever recipes (from-scratch, 800 epochs, on the base dataset)

| recipe | flags | lever |
|---|---|---|
| `clean` | *(none)* — `--apply-filter off --swa-mode ema --seed 42` | the control |
| `csob` | `--charge-sobolev` | supervises the autograd ∂q/∂V the AC/transient solvers consume |
| `sob` | `--sobolev --sobolev-corridor-only` | supervises ∂id/∂V slopes in the corridor |
| `sobf` | `--sobolev` | same, full-domain |
| `ekv` | `--ekv-core` | EKV analytic backbone + NN residual |
| `ekvhi` | `--ekv-core --ekv-lam-lo 0.12` | EKV with a raised r_o floor (opamp-targeted) |
| `s7` / `s17` / `s123` | `--seed 7 / 17 / 123` | pure basin probe — no objective change |
| `cs7` | `--charge-sobolev --seed 7` | combo |
| `csobekv` | `--charge-sobolev --ekv-core` | combo |
| `invtrip` | `--class-weights inv_trip=2.0` | upweights the per-tech Vth-centred inverter-trip band (sample class 7) |

### Curriculum recipes (warm-start fine-tunes on corridor data)

`traj_corridor` = a harvested bias tube along each tech's **own** ground-truth
circuit trajectories (sample class 12). Three corridor datasets exist — `cor`
(full), `corr` (ring + switchcap), `corro` (**ring-only**) — and the recipe name
selects one: `cor*` recipes take the matching variant, every `crit*` /
`csobcrit` recipe always takes **ring-only** `corro`.

| recipe | class weights | corridor data |
|---|---|---|
| `cor` / `corr` / `corro` | `traj_corridor=3.0`, from scratch | full / ring+SC / ring-only |
| `corft` / `corrft` / `corroft` | `traj_corridor=3.0`, fine-tune | full / ring+SC / ring-only |
| `corro15` | `traj_corridor=1.5`, fine-tune | ring-only |
| `invtripft` | `inv_trip=2.0`, fine-tune | base data |
| `crit10` | `traj_corridor=1.0, inv_trip=2.0` | ring-only |
| `crit15` | `traj_corridor=1.5, inv_trip=2.0` | ring-only |
| `crit15m` | `traj_corridor=1.5, inv_trip=3.0` | ring-only |
| `crit15h` | `traj_corridor=1.5, inv_trip=4.0` | ring-only |
| `crit20` | `traj_corridor=2.0, inv_trip=2.0` | ring-only |
| **`crit30`** | `traj_corridor=3.0, inv_trip=2.0` | ring-only |
| **`crit30f`** | identical to `crit30`, retrained to completion | ring-only |
| `crit30a1` | `traj_corridor=3.0, inv_trip=1.0` (half anchor) | ring-only |
| `csobcrit` | `--charge-sobolev` + `traj_corridor=3.0, inv_trip=2.0`, warm-started from `csob` | ring-only |
| `crit30u` | `crit30` at universal scope | ring-only |

`crit30f` exists because the original `crit30` stragglers were killed at
heterogeneous epochs 30–92: the on-disk `crit30` artifact was not a uniformly
executed recipe, and `crit30f` is the honest-contract re-run. It reproduces the
artifact cell for cell and is what production ships.

---

## 2. Recipe → complex gates, post-fix (V6.13.0, single-run OMP=1)

Only checkpoints still on disk could be re-measured; the rest are in
`archive-pre-gds-fix.md`. Source: `results/a3_regate/REPORT.md`.

### DirectNet (LEVEL=73)

| recipe | tier | /16 | pre-fix | failing cells |
|---|---|---|---|---|
| `clean` | small | **10** | 7 | tsmc5-ring, tsmc7-ring, tsmc12-opamp, tsmc7/12/16-switchcap |
| `clean` | medium | 10 | 10 | all 4 opamps, tsmc5-ring, tsmc7-ring |
| `v660clean` | large | 13 | 13 | tsmc5-ring, tsmc7-opamp, tsmc16-opamp |
| `clean` | xl | **12** | 10 | tsmc5-ring, tsmc7-ring, tsmc5-opamp, tsmc12-opamp |
| `csob` | large | **11** | 12 | tsmc5-ring, tsmc7-ring, tsmc5/7/16-opamp |
| **`crit30f`** | **large** | **15** | 14 | **tsmc7-opamp** |
| `corroft` | xl | **15** | 14 | tsmc5-opamp |
| `crit10` | xl | 14 | 14 | tsmc5-opamp, tsmc16-opamp |
| **`crit15m`** | **xl** | **16** | 14 | — |

`dn/clean/large` is the production slot, which has carried the `crit30f` weights
since V6.6.4; it and `dn/crit30f/large` are two independent re-measurements of
the same weights and agree cell for cell. The genuine clean@large is
`v660clean`.

### BSIM-AR (LEVEL=74)

| recipe | tier | /16 | pre-fix | failing cells |
|---|---|---|---|---|
| `clean` | small | **14** | 12 | tsmc5-ring 6.53 %, tsmc7-ring 5.97 % |
| `clean` | medium | 14 | 14 | tsmc5-ring 5.55 %, tsmc7-ring 7.41 % |
| `clean` | large | **14** | 13 | tsmc5-ring 7.38 %, tsmc7-ring 8.63 % |
| `clean` | xl | **14** | 13 | tsmc5-ring 7.61 %, tsmc7-ring 12.55 % |
| `csob` | xl | **14** | 13 | tsmc5-ring, tsmc7-ring |
| `invtrip` | large | **14** | 13 | tsmc5-ring, tsmc7-ring |
| `corroft` | large | 15 | 15 | tsmc7-opamp |
| `crit15m` | large | 15 | 15 | tsmc7-opamp |
| `crit30` | large | 15 | 15 | tsmc7-opamp |
| **`corroft`** | **medium** | **16** | 15 | — |
| **`corro15`** | **medium** | **16** | — | — |
| **`corroft`** | **xl** | **16** | 15 | — |
| **`crit15m`** | **xl** | **16** | 15 | — |
| **`crit30`** | **xl** | **16** | 14 | — |
| **`corro15`** | **xl** | **16** | 15 | — |

### PFN (LEVEL=75)

Only `clean` has ever been trained for PFN — the curriculum recipes are wired
(`MODEL=tabpfn` is supported by every driver) but untested.

| recipe | tier | /16 | pre-fix | failing cells |
|---|---|---|---|---|
| `clean` | small | 11 | 11 | tsmc5-ring/opamp, tsmc7-ring/opamp, tsmc12-switchcap |
| `clean` | medium | **11** | 10 | tsmc5-ring/opamp, tsmc7-ring, tsmc16-opamp, tsmc12-switchcap |
| `clean` | large | **9** | 8 | tsmc5-ring, tsmc7-ring/opamp, tsmc12-opamp/switchcap, tsmc16-opamp/switchcap |

---

## 3. What survives the re-gate — the durable recipe laws

**1. The corridor is the ring lever, and the only one.** Every recipe that
closes a low-VDD ring has `traj_corridor` in it; every recipe without it
(`clean`, `csob`, `invtrip`, seeds, `ekv`) fails tsmc5-ring and/or tsmc7-ring
in every family at every tier. On DirectNet the corridor takes tsmc5-ring from
12.66 % to 4.04 %; on BSIM-AR from 5.55–7.61 % to 3.33 %; at universal scope
tsmc7-ring from 14.89 % to 3.61 %. Rings are **gds-invariant** — not one ring
cell moved in the re-gate — so this is a genuine value-surface result and not an
artifact.

**2. `gds` moved opamps, the corridor moves rings, and the two levers are
independent.** This is the cleanest structural statement the campaign produced.
It also explains why the pre-fix recipe rankings were so noisy: half the signal
they were ranking on was a Jacobian sign error.

**3. The `inv_trip` anchor is family-dependent.**
* On **DirectNet** it composes: `crit30` (corridor 3.0 + anchor 2.0) beats
  `corroft` (corridor alone) by a deterministic tsmc5-opamp bank, and the
  corridor-weight → basin map is **non-monotone** — for tsmc5-opamp, w1.0 FLIP,
  w1.5/2.0 det-FAIL, w3.0 det-PASS. The anchor is what makes w3.0 safe where
  `corroft` alone railed.
* On **BSIM-AR** it is **inert**: `corroft` ≡ `crit30` to <0.5 % on every cell,
  and `invtrip` alone lands on exactly `clean`'s failure set. (Pre-fix it looked
  actively *harmful* — `invtrip@large` railed tsmc7-opamp at 99.99 % against
  clean's 12.78 % — but post-fix both pass. It was the gds floor railing the
  opamp, not the anchor.)
* The anchor is also **discontinuous in weight**: `crit30a1` (anchor 1.0)
  reproduces `corroft` (anchor 0) almost exactly, so the {16} → {5,12} basin hop
  happens between 1.0 and 2.0.

**4. Curricula *relocate* basins rather than composing them.** `csobcrit` —
`csob`'s tsmc16-opamp hold plus `crit30`'s tsmc5+tsmc12 hold — scored 13/16, not
15: csob's deterministic tsmc16 hold (1.28 %) degraded to a FLIP while tsmc12
was gained. The composition hypothesis is refuted; the V6.13.0 fix does not
revive it (the checkpoints are archived, but the mechanism — one shared `id`
head, mutually competitive basins — is unchanged).

**5. Recipe rankings do not transfer across capacity or scope.** `crit30` is the
best recipe at `large` and only 12/16 at xl pre-fix; `crit10`/`crit15m` peak at
xl. At universal scope `csob`'s tsmc16-opamp basin does not transfer at all, and
`crit30u`'s anchor relocates tsmc12-opamp into a FLIP. Three independent
demonstrations (tier, scope, family) that the weight→basin map is not portable.

**6. What the fix *changed* about recipe conclusions.** Post-fix, all three
BSIM-AR `large` curricula score exactly 15/16 with the same single miss, and all
four xl curricula score 16/16 — so **"the recipe decides which opamp basin you
get" is retracted for BSIM-AR**. The recipe now discriminates only on rings
there. On DirectNet the recipe still discriminates on opamps (`crit15m@xl` 16/16
vs `crit10@xl` 14/16, same tier, same data).

---

## 4. Device fidelity and AC by recipe

Device **DC** is gds-invariant, so the pre-fix per-recipe device table stands
(`archive-pre-gds-fix.md`, Appendix C):

| recipe | small | medium | large | xl | mean device NRMSE % |
|---|---|---|---|---|---|
| `clean` | 1.63 | **1.29** | 1.70 | 2.17 | 1.70 |
| **`csob`** | 1.77 | 1.33 | **1.50** | **1.77** | **1.59** |
| `cor` | 1.44 | 1.16 | 1.70 | 2.15 | 1.61 |
| `corft` | 1.49 | 1.20 | 1.60 | 2.00 | 1.57 |
| `invtrip` | 1.59 | 1.24 | 1.84 | 2.01 | 1.67 |
| `sob` | 1.92 | 1.38 | 1.95 | 2.22 | 1.87 |
| `ekv` | 1.99 | 1.72 | 2.38 | 2.18 | 2.07 |
| `csobekv` | 2.19 | 1.77 | 2.00 | 2.05 | 2.00 |

`csob` earns its keep on the device/charge axis and is retained as a documented
env-pin alternate for **device and AC work only** — its complex-gate rationale
is withdrawn (§2: 11/16 post-fix, and it now fails the tsmc16-opamp it was
documented to hold).

**AC by recipe is the axis the gds fix invalidated wholesale** and is being
re-measured in V7.1.0; see `by-scale.md` §5.

---

## 5. Dead ends — recorded because they cost real GPU time

| arm | verdict |
|---|---|
| **`sob` (id-Sobolev)** | KILL — 5/16 at `large`, the worst recipe measured. Reproduces the standing "derivative fidelity ⟂ opamp" result: sharpening ∂id/∂V collapses the value-owned high-gain fixed point. |
| **`ekv` / `csobekv`** | The EKV analytic core breaks tsmc12/16 SRAM and tsmc12 opamp, converges to a validation loss 10–40× worse than clean (the EKV core and charge-Sobolev fight over the shared `id` head), and nets 10/16. `csobekv` is nevertheless the only `large` checkpoint to pass tsmc7-ring **and** tsmc16-opamp together, and its tsmc5-ring 5.80 % is the closest any `large` checkpoint got to the 5 % gate pre-corridor. |
| **Seed sweeps (`s7`/`s17`/`s123`)** | Each lands exactly two opamp basins, a *different* two, and tsmc12-opamp passes only on seed 42. The seed axis cannot win; it only demonstrates the basins are a lottery. |
| **`invtrip` alone** | Inert-to-harmful on BSIM-AR, and on DirectNet it needs the corridor to matter. |
| **`csobcrit`, `crit30a1`** | Both 13/16 — the V6.6.7 "15/16 hunt round 1", both arms negative (§3, law 4). |
| **Full-corridor at xl (`cor`, `corft`, `corrft`)** | Collapses (5/16, 4/16, 6/16). The tsmc7 + tsmc12 xl checkpoints are structurally broken: the value surface overflows `sinh` → singular MNA → NR divergence at t = 2 ps. At xl the **ring-only** corridor is the only safe corridor. |
| **µA-band loss reweighting** | KILL (V6.6 campaign) — do not reweight the µA DC loss band. |
| **`crit` fine-tune for universal→TSMC5 transfer** | Better mid-tier device metrics, worse gates at the tier that matters (n1M: 1/4 vs plain 4/4). |
| **Tiny-tier fine-tunes (≤10k rows)** | Diverge outright through the tier-refit normalizer (device NRMSE ~1e69). Any few-shot regime below ~50k rows needs a frozen normalizer — a pipeline change, not a recipe. |
| **Zero-shot from the universal trunk** | 0/4. Embedding rows must be trained; there is no free lunch from the shared trunk. |

---

## 6. Recommendations

| use | checkpoint | why |
|---|---|---|
| **Production (fast path)** | `crit30f@large` (= the `tsmc{X}_dn_large_*` slots) | 15/16 strict, zero flips, 0.92 M params, ~1.5 ms/eval |
| **Full-sweep DirectNet** | `crit15m@xl` (env-pin) | 16/16 strict, zero flips — but 2.13 M params, 2.3× inference, no device-fidelity gain |
| **Highest fidelity** | `corroft@medium` (BSIM-AR) | 16/16 strict at 1.9 M params, device DC 44/44, comfortable margins — at ~40× DirectNet's per-eval cost |
| **Device / charge / AC work** | `csob@large` (env-pin) | best mean device NRMSE and charge-axis fidelity; **not** a complex-gate alternate |
| **One checkpoint, many techs** | `u716_dn_corroft_large` (env-pin) | 10/12 strict, zero flips at universal scope |
| **Research** | `clean@small` (PFN) | 11/16 strict at 0.69 M params — the strongest clean small tier on record |

Highest-value untried recipe arm: **the corridor curriculum on PFN**
(`MODEL=tabpfn RECIPES=corroft SIZES=small`) — fully wired, never run, and aimed
at exactly the 4 tsmc5/tsmc7 cells PFN fails.
