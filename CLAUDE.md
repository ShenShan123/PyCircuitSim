# Project: PyCircuitSim

## Overview

Pure-Python SPICE-like circuit simulator emphasizing educational clarity and a
decoupled Solver ↔ Device-Model architecture. **Primary goal:** four coexisting
compact-model families on one solver, all gated against NGSPICE ground truth:

- **BSIM-CMG** (LEVEL=72) — PyCMG-wrapped OSDI FinFET model; the **ground truth**
  every NN trains against and is gated on.
- **DirectNet** (LEVEL=73) — feed-forward MLP; the **production** NN fast path.
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive Transformer; the validated
  **higher-fidelity** option (15/16 strict, ~30–100× slower AR inference).
  Un-parked in V6.8.0.
- **PFN / TabPFN** (LEVEL=75) — TabPFN-v3-style in-context transformer;
  **research** family (V6.10.0; clean small 11/16 strict, zero OMP flips; ~10× DN
  eval cost, 4× faster than BSIM-AR).

The three NN families share one data / normalization / loss / training / eval
pipeline via the unified `bsimar` package (`external_compact_models/bsimar/`).
Supports **`.op` / `.dc` / `.ac` / `.tran`** for all model types. Six techs:
ASAP7 + TSMC5/6/7/12/16.

**Core Principles:** pure Python; Solver ↔ Device Models decoupled; production-grade
compact models via PyCMG/OSDI; basic HSPICE netlist compatibility.

## Architecture

### Module Structure

```
pycircuitsim/
├── config.py           # Path configuration (OSDI binary, modelcards)
├── simulation.py       # Orchestration (run_simulation, run_dc_sweep, run_transient)
├── parser.py           # Two-pass netlist parsing, .model/.subckt/.ic directives
├── circuit.py          # Circuit topology
├── solver.py           # MNA matrix + Newton-Raphson; DC/Transient/AC solvers
├── logger.py           # HSPICE-like .lis output
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── base.py               # Component abstract base
    ├── passive.py            # R, C, V, I sources (PULSE)
    ├── mosfet_cmg.py         # BSIM-CMG (LEVEL=72) via PyCMG
    ├── mosfet_nn.py          # Shared _MOSFETNNBase (LEVEL=73/74/75) — voltage prep, autograd, Vds correction
    ├── mosfet_directnet.py   # DirectNet (LEVEL=73, production)
    ├── mosfet_bsimar.py      # BSIM-AR Transformer (LEVEL=74)
    └── mosfet_pfn.py         # TabPFN-style PFN (LEVEL=75)

external_compact_models/
├── bsimar/             # Unified NN compact model package (importable as `bsimar`)
│   ├── config.py                   # NNTechConfig + TECH_CODE_MAP + local-vocab helpers
│   ├── data/{normalize,dataset}.py
│   ├── models/{direct_net,transformer,tabpfn}.py   # nn.Embedding tech-code
│   ├── losses/bni_mae.py           # MAELoss + per-target LDS weights
│   ├── training/trainer.py
│   ├── eval/{metrics,loo_labels}.py
│   ├── cli/train.py                # python -m bsimar.cli.train --model {direct,transformer,tabpfn}
│   └── checkpoints/                # *.pt + _norm.npz (+ _config.npz for TF/PFN; gitignored)
└── PyCMG/              # BSIM-CMG OSDI wrapper (git submodule)
    ├── pycmg/{core,model,parser,osdi_types,tech}.py
    ├── build/osdi/bsimcmg.osdi
    └── modelcards/     # ASAP7/*.pm committed; TSMC{5,6,7,12,16}/cln*.l gitignored (IP)

main.py · examples/*.sp · results/
tests/
├── common/             # Shared infra: base.py, bsimcmg_{dc,tran}.py, nn{,_sweep}.py, complex{,_sweep,_ac}.py
├── references/         # NGSPICE reference netlists
└── verify_*.py         # DC/tran/AC/subckt/NN/complex gates
```

### Key Algorithms

* **MNA** — Sparse construction (scipy.sparse lil_matrix → CSR + spsolve).
* **Newton-Raphson** — SPICE-standard convergence (RELTOL=1e-4 + VNTOL=1e-7).
* **BE → Trap → BDF-2 integration** — Backward Euler step 1, Trapezoidal default, BDF-2 auto on stiffness.
* **Source + GMIN stepping** — homotopy; GMIN stepping opt-in for bistable.
* **LTE sub-stepping** — adaptive internal sub-steps (opt-in via `max_substeps`).
* **Bistable convergence** — DC oscillation detection, adaptive damping, hard `.ic` mode.
* **AC small-signal** — `ACSolver` linearizes about the DC OP and solves complex `Y = G + jωC` per frequency, including the full MOSFET transcapacitance stamp.

### Key Compact Models

Four families plug into the same solver (LEVEL 73/74/75 share the `bsimar`
data/normalization/eval pipeline). BSIM-CMG is the authoritative ground truth;
DirectNet is production; BSIM-AR is the higher-fidelity option; PFN is research.

| LEVEL | Model                  | Implementation                          | Role                                   |
| ----- | ---------------------- | --------------------------------------- | -------------------------------------- |
| 72    | **BSIM-CMG**           | `models/mosfet_cmg.py` via PyCMG/OSDI   | FinFET **ground truth**                |
| 73    | **DirectNet**          | `models/mosfet_directnet.py` (PyTorch)  | **Production** NN (fast path)          |
| 74    | **BSIM-AR Transformer**| `models/mosfet_bsimar.py` (PyTorch)     | Higher-fidelity AR NN (un-parked V6.8.0)|
| 75    | **PFN (TabPFN port)**  | `models/mosfet_pfn.py` (PyTorch)        | In-context NN (**research**, V6.10.0)  |

- **BSIM-CMG (72)** — the reference every NN trains against and is gated on;
  all 6 techs at DC <0.1 % / transient ~0.2 % NRMSE vs NGSPICE. Never substitute
  simplified equations for it.
- **DirectNet (73)** — single-shot MLP; 7-dim input (Vgs, Vds, Vbs, NFIN, L, T,
  tech_code with `nn.Embedding`). gm/gds/gmb are the **autograd Jacobian** of the
  predicted `id`; AC caps are the `dQ/dV` autograd of predicted charges; per-tech
  checkpoints use a local embedding vocab (Rule 16). **Production = uniform `large`
  tier with the crit30 curriculum (V6.6.4) = 14/16 complex gates, OMP-deterministic**
  — one identical recipe per (tech × device), no per-case specials. Report:
  `docs/accuracy/DirectNet-L73-accuracy.md` — the unified DirectNet record
  (V6.6.0 baseline → V6.6.1 recipes → V6.6.6 cross-tier → V6.7.0 universal → TSMC6;
  Part I = analysis + recommendation, Part II = the frozen data tables), CHANGELOG.
- **BSIM-AR (74)** — autoregressive Transformer sharing DirectNet's pipeline.
  Best config `corroft@medium` (corridor curriculum, 1.9M params) = **15/16 strict**,
  beating DN production (banks tsmc16-opamp + both rings; misses only tsmc7-opamp,
  the T3-solver-only cell). AR inference is ~30–100× slower on CPU, so DirectNet
  stays production. Per-tech `tsmc{X}_tf_{small,medium,large}_{nmos,pmos}` (+ recipe
  variants); parser LEVEL=74 preempt cascade + `PYCIRCUITSIM_NN_FORCE_LEVEL=74`
  hook. Report: `docs/accuracy/BSIM-AR-L74-accuracy.md`.
- **PFN / TabPFN (75)** — faithful scaled-down port of TabPFN-v3's in-context
  transformer (tech code = 8th column token, local vocab), with two deviations: a
  **frozen learned context** (stratified K-row buffer baked into the checkpoint,
  context-KV cached at inference) and a direct 13-output value head (NR needs smooth
  autograd). Clean `small` = 11/16 strict, the first family with zero OMP flips;
  capacity curve declines s→m→l (11/10/8). Per-tech `tsmc{X}_pfn_{small,medium,large}_{nmos,pmos}`;
  env pins `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`, hook
  `PYCIRCUITSIM_NN_FORCE_LEVEL=75`, drivers take `MODEL=tabpfn`. The `_config.npz`
  sidecar is **required** to rebuild the arch. Report: `docs/accuracy/PFN-L75-accuracy.md`.

## Supported Features

* **Devices:** R, C; NMOS/PMOS LEVEL=72/73/74/75; DC + AC voltage/current sources (`AC=mag phase`), PULSE.
* **Analyses:** `.op`, `.dc`, `.tran` (+ `uic`), `.ac`.
* **Directives:** `.model` (LEVEL=72/73/74/75), `.include`, `.ic`, `.subckt`/`.ends` + hierarchical `X` instances (V6.12.0).
* **Subcircuits (V6.12.0):** `.subckt <name> <ports...> [param=default ...]` … `.ends`;
  instance `X<id> <nodes...> <name> [param=val ...]`. **Flattening at parse time**
  (ngspice-style, so solver/circuit are untouched): internal nodes → `X1.n1`
  (nested `X1.X2.n1`), devices → `M.X1.Mp1` (type char preserved for first-char
  dispatch); ground `0`/`GND` stays global; ports map to connecting nodes. Params
  resolve as bare names or `{expr}`/`'expr'` arithmetic (`+ - * /`, unit suffixes);
  `.ic` in a body is node-remapped AND param-resolved, top-level `.ic V(X1.n1)=v`
  reaches internal nodes, `uic`/`force_ic` consume them unchanged; `.model`/`.include`
  in bodies are hoisted global; nested `.subckt` defs register globally. Loud errors
  on unknown subckt, port-count mismatch, recursion (>64). Gate: `tests/verify_subckt.py`
  (8 checks — subckt==flat bit-identical, L72 inverter + nested buffer vs NGSPICE).
* Legacy LEVEL=1 (Shichman-Hodges) removed.

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth within
reasonable numerical tolerance. Never use simplified/self-defined equations as reference.

> **Accuracy evidence lives in `docs/accuracy/` — one unified report per NN family**
> (`DirectNet-L73`, `BSIM-AR-L74`, `PFN-L75`; index in `docs/accuracy/README.md`).
> Shared methodology + the standing gds-sign-bug caveat: `DirectNet-L73-accuracy.md`
> §2 and §12.2.
>
> **Sprint history, version-by-version status, dead-ends, and the open known-issue
> roadmap live in `docs/CHANGELOG.md` + `MEMORY.md`** — not duplicated here.
> CLAUDE.md tracks durable architecture, rules, and how-to-run.

## Setup

```bash
conda create -n pycircuitsim python=3.10 -y
conda activate pycircuitsim
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch    # for LEVEL=73/74/75
git submodule update --init --recursive
```

**Prerequisites:** NGSPICE 45.2+ (`/usr/local/ngspice-45.2/bin/ngspice`), OpenVAF
23.5.0+ (`/usr/local/bin/openvaf`), BSIM-CMG OSDI binary
(`external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`).

## Quick Start

```bash
conda activate pycircuitsim
python main.py examples/bsimcmg_inverter_tran.sp           # -> results/
python main.py examples/rc_lowpass_ac.sp -o my_out -v      # custom output dir + verbose
```

The analysis is chosen by the directive **inside** the netlist (`.op`/`.dc`/`.tran`/`.ac`)
— `main.py` takes only the netlist path, an optional `-o/--output` dir (default
`results/`), and `-v/--verbose`. Ready-to-run decks live in `examples/`.

### Write a netlist (HSPICE-style)

Component lines + `.model` cards + one analysis directive, terminated by `.end`.
Node `0` is ground; `*` starts a comment. RC low-pass `.ac` (`examples/rc_lowpass_ac.sp`):

```spice
V1 in 0 DC=0 AC=1 0          * AC source: DC bias 0, |AC|=1V, phase 0 deg
R1 in out 1k
C1 out 0 159.155n
.ac dec 20 10 1e6            * 20 pts/decade, 10 Hz .. 1 MHz
.end
```

**MOSFETs — pick a compact model by LEVEL.** Card
`.model <name> {N|P}MOS (LEVEL=<72|73|74|75> [TECH=tsmc5 VT=lvt])`; instance
`M<id> <drain> <gate> <source> <bulk> <name> L=30n NFIN=10` (geometry `L`, `NFIN`,
optional `TFIN`/`HFIN`/`FPITCH`).

- **72** — BSIM-CMG ground truth (`examples/bsimcmg_inverter_dc.sp`).
- **73** — DirectNet; `TECH`/`VT` select the per-tech checkpoint (`examples/nn_inverter_dc.sp`).
- **74** — BSIM-AR (per-tech `tsmc{X}_tf_*` checkpoints; `examples/bsimar_inverter_dc.sp`).
- **75** — PFN/TabPFN (per-tech `tsmc{X}_pfn_*`; env-pin/`FORCE_LEVEL=75` driven; see above).

CMOS inverter DC sweep (LEVEL=72, `examples/bsimcmg_inverter_dc.sp`):

```spice
Vdd 1 0 1.0
Vin 2 0 0.0
Mp1 3 2 1 1 pmos1 L=30n NFIN=10     * drain gate source bulk
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)
.dc Vin 0 1.0 0.01
.end
```

### Analysis directives

- `.op` — DC operating point.
- `.dc <src> <start> <stop> <step>` — DC sweep.
- `.tran <tstep> <tstop> [uic]` — transient; drive with `PULSE v1 v2 td tr tf pw period`.
  `uic` (NGSPICE-style) starts from the `.ic` state — pins `.ic` nodes during the OP
  so a high-impedance node (e.g. a switched-cap hold node) starts at its `.ic` value.
  Default-off; non-`uic` decks are byte-identical.
- `.ac {dec|oct|lin} <N> <fstart> <fstop>` — small-signal; requires `AC=mag phase` on a source.
- `.ic V(node)=...` (hard initial condition; reaches subckt-internal nodes via
  `V(X1.n1)`) and `.include` are also supported.

### NN training (per-tech, LEVEL=73/74/75)

Dedicated per-tech NMOS/PMOS checkpoints for **TSMC5/6/7/12/16** (all three families
trained + gated at every scale in V6.11.0; addenda in the DN/TF/PFN accuracy reports +
CHANGELOG V6.11.0). `--tech-scope` ∈ `{tsmc5,tsmc6,tsmc7,tsmc12,tsmc16,universal}`;
`--size` ∈ `{small,medium,large,xl}`. Curriculum recipes (incl. the production crit30)
train via `scripts/recipe_train.sh` (warm-start from the clean same-size base — at
`large` the `v660clean` archive, injected automatically).

```bash
# 1. Per-tech data (one .npz per tech+device). --enable-inv-trip adds the inverter-trip
#    overlay; the grid sampler carries the reverse-Vds corridor. --tech ∈ {tsmc5,tsmc6,
#    tsmc7,tsmc12,tsmc16,asap7,all}. Repeat per tech.
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc5 --enable-inv-trip --n-workers 8

# 2. Train. --tech-scope auto-sets --exclude-techs, --num-tech-codes (local vocab +
#    UNKNOWN), the default --data path, and the save_prefix the parser resolver
#    recognizes (tsmc{X}_{dn,tf,pfn}_<size>_<dev>).
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model {direct,transformer,tabpfn} --size medium \
    --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc6,tsmc7,tsmc12,tsmc16} --cuda --overwrite
```

**Full capacity sweep** (DirectNet, 4 techs × N/P × 4 sizes = **32 ckpts**, one clean
recipe): `scripts/benchmark_gen_data.sh` → `scripts/benchmark_train_sml.sh`
(`GPUS="0 2"` pins a GPU subset, `NSTREAMS` sets concurrency) →
`scripts/benchmark_run_tests.sh` → `scripts/benchmark_collect.py` (`results/benchmark_sml/REPORT.md`).

**Checkpoints** (`external_compact_models/bsimar/checkpoints/`, each `*_best.pt` +
`_norm.npz`, plus `_config.npz` for TF/PFN):

- **DirectNet:** `tsmc{5,6,7,12,16}_dn_{small,medium,large,xl}_{nmos,pmos}` — each a
  local-vocab embedding (variants + 1 UNKNOWN; Rule 16). Production `large` carries
  **crit30** since V6.6.4 (clean originals archived `..._v660clean_large_*`; TSMC6 =
  clean only, so its production `large` = clean 3/4). Kept beyond production:
  `v660clean_large` (warm-start base), `crit30f_large` (production provenance),
  alternates `csob@large` (AC/device), `corroft`/`crit10`@xl + `crit15m@xl`
  (tsmc16-opamp coverage). Also `tsmc{X}_tf_*` (BSIM-AR) + `tsmc{X}_pfn_*` (PFN).
- **Universal DirectNet (V6.7.0):** `u716_dn_{clean,csob,corroft,crit30u}_large` +
  `_{clean,corroft}_xl` + TSMC5 fine-tunes `u716f5_plain_n{1000000,full}_large` —
  18-code vocab, env-pin-only. Best = `u716_dn_corroft_large` (10/12 strict, 0 FLIPs).
  See `docs/accuracy/DirectNet-L73-accuracy.md`.
- **Resolver cascade** (`pycircuitsim/parser.py`): env pin
  `PYCIRCUITSIM_NN_CHECKPOINT_{DN,PFN}_{NMOS,PMOS}` read FIRST (since V6.6.6 an absent
  pinned stem RAISES — no silent fallback); then per-tech `tsmc{X}_{dn,tf,pfn}_{large,
  medium,small,xl}` (**large-first**) preempts the dormant universal fallback.
  Completed runs carry a `*_best.pt.complete` marker (a bare `_best.pt` may be a
  killed run). Resolutions log `[NN-resolver] L73 <name> TECH=.. VT=.. -> <chk> ...`.

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with
`L=16n NFIN=10`. Parser auto-resolves the checkpoint + local-vocab tech_code via
`bsimar.config.local_variant_code(scope, tech, variant)`.

### Output files

`results/<circuit_name>/<analysis_type>/`: an HSPICE-like `*_simulation.lis` log, the
data (`*_dc_sweep.csv` / `*_transient.csv` / `*_ac_sweep.csv`), and a Matplotlib plot.

## Testing & Verification

All tests require `conda activate pycircuitsim`. Ground truth is **always** NGSPICE on
the identical BSIM-CMG (LEVEL=72) OSDI model — never a simplified/self-defined
reference. Gates are CPU-pinned (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`) and honor `NGSPICE_BIN` (repo `tools/ngspice-45.2/bin/ngspice`
when `/usr/local` is absent). Since V6.6.6 the complex/AC gate infra pins torch to 1
thread by default (`PYCIRCUITSIM_TORCH_THREADS` overrides — used by the OMP
multistability sweep).

**Shared infra** (`tests/common/`): `base.py`, `bsimcmg_{dc,tran}.py`,
`nn.py`+`nn_sweep.py`, `complex.py`+`complex_sweep.py`+`complex_ac.py`;
references in `tests/references/`.

- **Subcircuit hierarchy (V6.12.0):** `verify_subckt.py` — L1 linear subckt==flat
  equivalence (tran/AC/nested-OP/uic, no NGSPICE), L2 L72 inverter-in-subckt vs flat
  vs NGSPICE, L3 nested 2-inverter buffer (X-in-X + params + `.ic` on internal node).
  **8/8 PASS.** The PyCircuitSim-side test decks are now hierarchical (inverter,
  complex builders, NN inverter, CS-amp AC); probed nodes stay top-level (ports) so
  harness keys/baselines are unchanged; NGSPICE reference decks + single-device
  Id-Vgs decks stay flat.
- **BSIM-CMG (72):** OP `verify_bsimcmg_op.py` (<0.02%); DC L1 `verify_bsimcmg_dc.py` (2)
  · L2 `..._comprehensive.py` (81) · L3 `verify_multi_tech_dc.py` (53); tran L1
  `verify_bsimcmg_tran.py` (1) · L2 `..._comprehensive.py` (45) · L3
  `verify_multi_tech_tran.py` (86); AC `verify_ac.py` (2/2, ~machine precision).
- **DirectNet (73):** `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 [--inverter-only]`
  (baseline inverter 8/8, DC 55/55, tran 64/64); `verify_nn_multi_tech_{dc,tran}.py`
  (parametric, baseline-gated — pin OMP/MKL=1, VTC trip has ~±1% scatter);
  `verify_nn_ac.py` (CS-amp gain/f3db/mag-NRMSE); `verify_nn_lifted_source_dc.py`
  (NRMSE ≤10%, guards Rule 2).
- **Complex circuits (4 × 4 = 16 gates):** `verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py`
  + `complex.py` (scored vs NGSPICE: ring period, opamp gain, switchcap charge/droop,
  SRAM butterfly positivity + NRMSE). SRAM `force_ic` 6T-latch probe is a printed
  **diagnostic**, not a gate. Parametric mirrors `verify_complex_*_sweep.py` +
  `complex_sweep.py` (baseline-gated, sha256-pinned); `verify_complex_sweep_canaries.py`
  guards single-point ↔ sweep equivalence. Opamp AC `verify_complex_opamp_ac.py`
  (two-stage Miller open-loop; RO+SRAM AC-excluded).
- **Diagnostics** (`tests/diag_*.py`, **not** gates — L72-in-PyCircuitSim reference):
  `diag_l72_complex_control.py`, `diag_l72_switchcap_control.py`/`_uic_control.py`
  (prove L72-in-PyCircuitSim ≈ NGSPICE, isolating NN-surface gaps);
  `diag_nn_jacobian_consistency.py` (autograd-Jacobian self-consistency).

**Quick sanity:**

```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
python tests/verify_subckt.py
NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" python tests/verify_ac.py   # if /usr/local absent
```

---

## Development Guidelines

**Coding standards:** type hints on all signatures; clear names (`v_gate`, `i_drain`);
docstrings for complex algorithms; voltage clamping Vgs±5V, Vds±10V.

**Separation principle:** `solver.py` builds MNA + executes NR (no device equations);
`models/` computes current/conductances (no matrix ops); `simulation.py` orchestrates
(parse → solve → visualize); all devices inherit from `Component`.

**Key numerical techniques:**

- Sparse MNA: `lil_matrix` assembly, CSR + `spsolve`. O(n) memory, O(n·log n) solve.
- Convergence: `|ΔV| < VNTOL + RELTOL × max(|V_old|,|V_new|)` (RELTOL=1e-4, VNTOL=1e-7).
- GMIN (1e-12 S) prevents singular matrices. DC GMIN stepping opt-in
  (`use_gmin_stepping=True`, 2-level [1e-8, 1e-12]). NN circuits use
  `_solve_dc_with_retry` (fast path first, GMIN retry on `_last_solve_converged=False`);
  BSIM-CMG never enters the retry branch.
- BE → Trap → BDF-2: BE step 1, Trap step 2+, BDF-2 auto on stiffness (NR>20); one-way.
- Source stepping (20 steps); supply-relative adaptive damping with stuck-counter.
- DC oscillation detection: 5-snapshot ring, accepts averaged solution if variance < 10× tol.
- Hard `.ic` mode (`force_ic=True`): stamps `.ic` nodes as temporary V-source
  constraints, re-solves unconstrained. Required for SRAM latches.
- LTE sub-stepping (opt-in via `max_substeps`, default 1=disabled).

**Entry points:** CLI `main.py`; API `pycircuitsim.simulation.run_simulation()`;
module exports (Circuit, Parser, Visualizer, run_simulation).

**Environment & tools:** conda env `pycircuitsim` at
`/home/shenshan/.conda/envs/pycircuitsim`; PyTorch 2.10.0 (CPU); OpenVAF
`/usr/local/bin/openvaf`; NGSPICE `/usr/local/ngspice-45.2/bin/ngspice`.

---

## Critical Design Rules

These rules were learned from bugs. Violating them causes NR divergence or wrong results.

### Sign Convention for Device Models

1. **Use terminal current `id`, NOT channel `ids`** — `ids = id - is ≈ 2*id` (2× error).
2. **NMOS** `calculate_current()` returns `-result["id"]`; **PMOS** returns `result["id"]` (positive = leaving drain).
3. **Solver stamping** uses unified "current leaving drain" convention. All VCCS conductances (g_ds, g_m, g_mb) need full 4-entry stamps (drain,ctrl+; drain,ctrl-; source,ctrl-; source,ctrl+). An incomplete stamp breaks Jacobian symmetry.
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq    # same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
4. **gds floor** for stamping: `max(gds, 1e-12)`. Never `abs(gds)` — it flips large-negative to large-positive and diverges NR. Preserve gm/gmb signs.
5. **Update `_is_mosfet()`** in `solver.py` when adding new device types.
6. **Test both NMOS and PMOS** vs NGSPICE: single OP, DC sweep, inverter VTC, inverter transient.

### NN Model Rules (LEVEL=73 DirectNet production; 74 BSIM-AR; 75 PFN)

All three NN levels share the data pipeline and inference rules and use `nn.Embedding`
for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code). Rules
3–5, 9–10, 15 were deleted by the user (parked BSIMAR-specific structure — recover
from CHANGELOG/git if needed). Rule numbers are internal to this doc and no longer
cited in code.

1. Feel free to **re-generate datasets and re-train all models** as you want.
2. **Source-relative frame for BOTH device types** — shift all terminal voltages by
   -Vs before NN eval (`v_d_nn = v_d - v_s`, Vs ≡ 0). Training uses Vs=0; shift
   invariance makes this exact. Until V6.4.7 only PMOS was shifted — lifted-source
   NMOS saw phantom Vgs/Vds; the canary `verify_nn_lifted_source_dc.py` (NRMSE ≤10%)
   guards this permanently.
6. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to `nmos_rvt`/`pmos_rvt`.
7. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`, `TECH_CONFIGS`,
   `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config` (alias `TechConfig = NNTechConfig`).
   Training VDD may differ from PyCMG's runtime VDD; check `NNTechConfig.VDD` per tech.
8. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG
   `eval_dc` raises on internal-node convergence failure. NFIN=1 is excluded: unstable
   `(variant, NFIN=1)` bins fail OSDI convergence and are dropped per-bin, so NFIN≥2 trains.
11. **Unified CLI** — `python -m bsimar.cli.train --model {direct,transformer,tabpfn}
    --size {small,medium,large,xl} --device-type {nmos,pmos} --tech-scope {...} ...`
    (xl = 512×8 ~2.13M p, over-fit-boundary; production = large). Per-tech `--tech-scope`
    → default save_prefix `tsmc{X}_{dn,tf,pfn}_<size>_<device>`. Flags (all default-off /
    behavior-preserving): `--swa-mode {none,ema,swa}`+`--ema-decay`; `--apply-filter
    {on,off}`+`--class-weights`; `--enable-subvt-off`; loss terms `--sobolev` /
    `--subthresh` / `--charge-sobolev`; EKV backbone `--ekv-core`/`--ekv-alpha`/`--ekv-hidden`.
12. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)`
    analytically, even for 13-output models. Guarantees Kirchhoff conservation every timestep.
13. Always report MRE (%), R², NRMSE, Max error (mV) per tech.
14. **Exclude ASAP7** — out of scope (no checkpoints; see Overview).
16. **Per-tech models use a LOCAL embedding vocab.** For a per-tech `--tech-scope` the
    loader remaps universal tech codes to a 0-indexed vocab and the trainer instantiates
    `num_tech_codes=N, unknown_code_id=N-1`, N = variants+1 (TSMC5:5, TSMC6:4, TSMC7:4,
    TSMC12:6, TSMC16:6). Training-time `p_unknown` dropout writes `unknown_code_id`, so a
    misaligned UNKNOWN id → CUDA assert. **Derive `unknown_code_id` from `num_tech_codes`;
    do NOT hardcode the universal 17.** Parser remaps at inference via
    `bsimar.config.local_variant_code(scope, tech, variant)`; scope read from the ckpt stem.

> **Load-bearing code (do NOT delete):** the V6.4.2 `_MonotoneVgResidual` + `--monotonic`
> path (`bsimar/{cli/train,models/direct_net,training/trainer}.py`,
> `pycircuitsim/models/mosfet_directnet.py`) must stay committed — on-disk checkpoints
> carry `mono.*` state_dict keys and fail to load without it (stock checkpoints route
> `mono=None`, no inference change). The default-off EKV backbone (`_EKVCore`, `core.*`
> keys) and the Sobolev/subthreshold/charge-Sobolev loss terms are likewise kept
> recoverable; all leave stock checkpoints byte-identical.

---

## Important Paths

- **PyCMG submodule:** `external_compact_models/PyCMG/` (21 device variants).
- **OSDI binary:** `build/osdi/bsimcmg.osdi` (PyCMG-relative).
- **Modelcards:** `modelcards/` (PyCMG-relative); ASAP7 `*.pm` committed; TSMC raw PDK
  `cln*.l` gitignored/IP-protected — naive cards regenerated on-the-fly via
  `pycmg.tech.resolve_modelcard` into `build/modelcards/`. Never commit `modelcards.tar.gz`.
- **Results:** `results/<circuit_name>/<analysis_type>/`. **Test results:**
  `tests/verify_*_results/` (generated, not tracked).
- **Sprint history:** `docs/CHANGELOG.md`. House-cleans have pruned old plans, one-off
  scripts, gate-specific diagnostics, stale datasets, and result dirs — so some path
  references in older CHANGELOG entries are intentionally dangling (the narrative, not
  the file, is the record).

## Other Tips

* **Start every complex task in plan mode** — pour energy into the plan for 1-shot
  implementation. Re-plan the moment something goes sideways.
* If the plan has several stages, implement in sequence. Git commit before modifying;
  keep what makes progress, `git reset` what doesn't.
* On every version update, add a `docs/CHANGELOG.md` entry. **Record dead-end proposals
  (the ones reverted) — they are as important as the successful ones.**
* **Never be lazy** — never simplify code or skip tests. **NEVER** use simplified
  equations or self-defined CMG models as reference; ALWAYS use simulation results as
  ground truth.
* Use **GPUs in parallel** when you train NN models.
