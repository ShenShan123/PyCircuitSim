# DirectNet target-trim + loss-reweight plan
**Date:** 2026-05-08
**Branch:** `refactor/nn-simple`
**Scope:** DirectNet (LEVEL=73) only. BSIMAR Transformer deferred until DN
results land.
**Owner:** Shen (ce-prepared by Claude)

## TL;DR

Of the 13 supervised targets, only 4 are consumed by the simulator. Drop
noisy/redundant ones and reweight the loss to emphasise load-bearing channels.
Validate against NGSPICE/PyCMG on the TSMC12 inverter (DC + transient). Three
experiments, ~30 min GPU each.

## Problem framing

### What the simulator actually reads from the NN

Looking at `pycircuitsim/models/mosfet_nn.py::_MOSFETNNBase._eval`:

| Target | Simulator's runtime use | Verdict |
|---|---|---|
| `id` | KCL residual on drain node (DC + tran) | **load-bearing** |
| `qg`, `qd`, `qb` | Charge state for transient cap currents (BE/Trap/BDF-2) | **load-bearing** |
| `qs` | Replaced with `−(qg+qd+qb)` for KCL (charge conservation) | **redundant; supervision is provably wrong** |
| `gm`, `gds`, `gmb` | Solver Jacobian, but rule 1 mandates **autograd ∂id/∂V**, never the supervised target | **unused at inference; only acts as a smoothness prior on `id`** |
| `cgg`, `cgd`, `cgs`, `cdg`, `cdd` | Same story — autograd ∂q/∂V is what gets stamped | **unused at inference; smoothness prior on `q*`** |

Supervised gm/gds/gmb/c\* targets earn their keep only as implicit
regularisers on surfaces of `id`, `qg`, `qd`. Second-order benefit paid for
with a 13-dim output head, 13-axis LDS computation, and 9 MAE terms competing
with the 4 we need.

### Why this is now an interesting trade

The post-refactor pipeline makes the loss boundary clean:
`MAELoss(weights=lds_per_target_per_sample)` is the one place where any
target's contribution is decided. Per-target mask is a 5-line change.
Output-head shrinkage is another 5 lines in `bsimar/models/direct_net.py`
and the simulator reads outputs by *name* through `_MOSFETNNBase._mcol`, so
a 4-output checkpoint works on the inference side — no new branch in eval glue.

## Hypothesis

**H1.** Removing supervision on `qs` is strictly an improvement: the
target is inconsistent with the analytical KCL identity the simulator
enforces, so the network is currently being penalised for matching what
the solver wants.

**H2.** Removing supervision on the 5 capacitances is neutral or
positive for circuit metrics. The autograd ∂q/∂V is what's stamped, and
the supervised c\* surface fights against the ∂q/∂V surface during
training (the project's own JAC postmortem documents the chain-rule
mismatch under asinh, and that's the same mismatch happening implicitly
under MAE today).

**H3.** Removing supervision on (`gm`, `gds`, `gmb`) hurts
deep-cutoff/strong-overshoot regions where `id` itself is near
the asinh floor — those are exactly the regimes where the ∂id/∂V
target carries information the `id` MAE cannot. Expected to *regress*
inverter VTC noise floor unless replaced with a mild reweighting.

**H4.** Down-weighting (`gm`, `gds`, `gmb`, `c*`) instead of dropping
them is the safer middle ground: keeps the smoothness prior, lets the
4 load-bearing targets dominate the gradient. Same checkpoint format,
same simulator path.

## Experiments

All on TSMC12 SVT (the tech the verifier hammers), L=12 nm, NFIN_n=10,
NFIN_p=20. DirectNet small preset (`hidden=128, layers=3,
batch=2048, epochs=80`), single-seed. Each run ≈4 min on one A100.

| Exp | Loss head | Output dim | Notes |
|---|---|---|---|
| **B0 — baseline** | uniform 13-target MAE+LDS | 13 | already trained: `refac_dn_small_{nmos,pmos}_best.pt` |
| **E1 — drop qs** | mask qs to 0 | 13 | minimum-risk; tests H1 in isolation |
| **E2 — id+q only (4-output)** | predict only `[id, qg, qd, qb]` | **4** | tests H2+H3 jointly; is the 9-target smoothness worth it? |
| **E3 — reweight 13** | weight = `[1.0, 0.1, 0.1, 0.1, 1.0, 1.0, 0, 1.0, 0.01×5]` | 13 | tests H4; same architecture, same checkpoint format |

E0 is the sentinel — same trainer, same data, no change. Needed on the
post-refactor pipeline so comparison isolates the loss change from any drift.

### Implementation deltas

1. **`bsimar/losses/bni_mae.py::MAELoss`** — accept a
   per-column-mask `column_weights: torch.Tensor` (length 13) and
   broadcast multiply. ~6 lines.

2. **`bsimar/training/trainer.py::_train_loop`** — add a
   `column_weights: Optional[torch.Tensor]` arg, threaded through the
   `MAELoss(weights=...)` call.

3. **`bsimar/models/direct_net.py::DirectNet`** — already accepts
   `output_dim`; the 4-output mode is just `output_dim=4`. Already
   parameterised; no architecture change needed.

4. **`bsimar/training/trainer.py::train_directnet`** — add
   `output_dim: int = 13` and `column_weights: Optional[Sequence[float]]`
   kwargs. When `output_dim < 13`, slice `train_ds.outputs` to the first
   `output_dim` columns of `OUTPUT_COLUMN_ORDER` (which puts
   `[id, gm, gds, gmb, qg, qd, qs, qb, ...]` in front; we'd actually
   want the first 4 to be `[id, qg, qd, qb]` — easiest fix is a fixed
   reorder once at load time).

5. **`bsimar/cli/train.py`** — `--output-dim {4,13}` and
   `--column-weights "id=1,gm=0.1,…"`. Tiny.

6. **Simulator side, `_MOSFETNNBase`** — already reads outputs by name via
   `_mcol`. For 4-output, model emits `[id, qg, qd, qb]`; unused names
   (gm, gds, gmb, c\*) are already handled by the autograd path inside
   `_eval`. Only `_out_col` map needs to change: 4-output maps
   `id→0, qg→1, qd→2, qb→3` and skips the rest. Detect via loaded state's
   last linear weight shape.

   For (gm, gds, gmb, c\*) we lose them as "predicted scalars" — never used
   as scalars; already reconstructed from autograd. Cleanly drops 9 columns
   of dead code.

### Acceptance gate

For each experiment, on the TSMC12 SVT inverter (`L=12n`,
`NFIN_n=10`, `NFIN_p=20`, `VDD=0.8`):

- **DC VTC** NRMSE vs PyCMG ≤ 12 % (matches v4 baseline)
- **Transient pulse** NRMSE vs NGSPICE ≤ 15 %
- **NR convergence:** ≤ 12 NR iters mean across 17 Vin sweep points;
  no divergence
- **NMOS DC NRMSE** at TSMC12 ≤ baseline + 1 pp

A run regressing on either DC or NR-iter count fails the gate; the
intermediate experiments don't have to be best-in-class, just not
worse than B0.

## Out of scope

- BSIMAR Transformer (deferred). The 4-output simplification could
  apply to it via the same name-based `_mcol` mechanism, but the AR
  ordering would need a redesign (drop the `id`-then-cap chain) and
  the project plans already document a long history of TF surprises.
  Wait until DN results land.
- ASAP7. The training set excludes ASAP7 (per CLAUDE.md), and the
  inverter verifier is on TSMC12.
- Jacobian-consistency loss (already deleted on this branch). If any
  of E1-E3 underperforms B0 *only* in the deep-cutoff / overshoot
  regions, then JAC becomes interesting again — but that's a follow-up
  branch, not this one.

## Order of operations

1. Wait for medium DN runs to finish (in flight, ETA ~25 min).
2. Run inverter verification on the existing **small** + **medium**
   13-target baselines (this is the B0 ground truth on the post-refactor
   pipeline). This is task #7 already.
3. Implement deltas 1–6 above.
4. Run E1, E2, E3 on the small preset (≈12 min total).
5. Inverter verify each; tabulate against B0.
6. Promote whichever wins (or doesn't regress) to medium-size, retrain,
   re-verify.
7. If the winner has the smaller head, update `_resolve_nn_checkpoint`
   cascade so it picks up the new prefix.

## Risks and mitigations

- **Risk:** deep-cutoff / overshoot Id-prediction worsens without the
  gm/gds smoothness prior, *exactly* in the regimes where rule 19's
  rail-restoring extrapolation is meant to mask the NN's flat-zero.
  **Mitigation:** acceptance gate explicitly checks NR-iter count;
  inverter VTC catches it before transient.
- **Risk:** 4-output DirectNet checkpoint silently fails the legacy
  parser cascade because `_resolve_nn_checkpoint` doesn't know about
  it. **Mitigation:** the cascade already prefers `refac_dn_*` over
  legacy v4; new checkpoints land under `refac_dn_e1_*` etc. so legacy
  paths are untouched.
- **Risk:** the medium retrain on the winning recipe regresses on
  another tech. **Mitigation:** universal training already covers all
  4 TSMC techs; the per-tech NRMSE breakdown the trainer prints will
  flag any single-tech blowup. If TSMC7 NMOS DC moves outside the
  known 14.72 % envelope (CLAUDE.md), that's a separate v4-re class
  bug, not an artifact of the trim.

## Deliverables

- `refac_dn_e1_{nmos,pmos}_best.pt` — drop-qs ablation
- `refac_dn_e2_{nmos,pmos}_best.pt` — 4-output ablation
- `refac_dn_e3_{nmos,pmos}_best.pt` — reweight ablation
- An inverter-verify table comparing B0 / E1 / E2 / E3 across:
  DC-VTC NRMSE, transient NRMSE, NR-iter count, NMOS DC NRMSE.
- One follow-up note recording which of {E1, E2, E3} actually moved
  the needle, and whether the 4-output simplification (E2) is shippable.
