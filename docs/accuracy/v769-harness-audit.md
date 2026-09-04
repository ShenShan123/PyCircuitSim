# V7.6.9 test-harness audit

> Historical audit: V7.7.0 removed the reduced LEVEL=73/74 implementation and
> the reduced-only tests referenced below.

Date: 2026-09-04

Audited baseline: `88f447e` (`V7.6.9`); fixes are in this follow-up commit.

Status: harness hardening and restored coverage. No diagnostic was promoted,
no threshold moved, and the frozen `simple-v1` `/20` denominator is unchanged.

Scope: the NN compact-model test harness in `circuit_templates/` and `tests/`.
The V7.6.8 audit ([`v768-template-harness-audit.md`](v768-template-harness-audit.md))
reviewed the catalog, its templates, and campaign fail-closed behaviour. This
pass asked a different question — *what is not tested at all, and what runs
only when someone remembers to run it* — and found four gaps that a review of
the catalog alone could not surface, plus one wrong answer the harness was
structurally unable to detect.

## Outcome

| Surface | Before | After |
|---|---|---|
| Collected `pytest -q tests` | 256 tests | 357 tests |
| Simulator-free gate suites in the collected run | 0 of 3 | 3 of 3 |
| Untested shipped simulator features | 3 | 0 |
| Deck cards checked for engine agreement | 0 | every template and rendered deck |
| Value scale factors matching NGSPICE | 8 of 10, `m` inverted by 1e9 | 10 of 10, measured |
| Dropped physics-changing directives | silent | warned, once per directive |
| Terminal/transcapacitance corner axis | nominal only | the full 14-corner matrix |
| AC analyses with no domain metric | 1 (`cascode_ac`) | 0 |
| Gates that ignore their argument vector | 10 | 0 |

## Gaps found and closed

### 1. Three hermetic gate suites never ran in the authoritative suite

`tests/README.md` calls `python -m pytest -q tests` the authoritative unit run.
It collected only the root `test_*.py` modules, so
`verify_simple_circuit_catalog.py` (33-case inventory, tiers, strict rendering,
corner matrix, repository placement), `verify_circuit_sweep_canaries.py` (4,854
render/topology-parity cells) and `verify_accuracy_campaign_tools.py` (600-job
campaign tooling, collector hygiene, solver residual checks) were reachable
only by running each script by hand. All three need no NGSPICE binary, no PDK
card and no checkpoint, and together take under 6 s.

`tests/test_hermetic_gate_suites.py` now runs all three from the collected
suite. They remain standalone scripts, because a campaign operator runs them
that way and reads their one-line verdict.

### 2. Five deleted V7.5 gates took three shipped features to zero coverage

`tests/common/core_gates.py` still advertised `verify_inductor.py`,
`verify_current_source_ngspice.py`, `verify_cmg_set_temperature.py`,
`verify_tran_branch_current.py` and `verify_tran_gear2.py`. None of those files
exist. The features they gated do:

| Feature | Where it lives | Test coverage found |
|---|---|---|
| `Inductor` (`L` cards, DC short, AC branch reactance, loud transient rejection) | `models/passive.py:969`, `parser.py:_parse_inductor` | none |
| `TransientSolver(integration_method=...)` — `auto` / `gear2` / `trap` | `solver.py:2282` | none |
| In-place `set_temperature()` on all three model families | `mosfet_cmg`, `mosfet_nn`, `mosfet_directnet_full` | indirect only, through `.temp` corner runs |
| Transient voltage-source branch currents | `solver.py:3357` | none directly; every supply-current metric reads it |
| Independent current-source sign convention | `models/passive.py:440` | none directly |

`tests/test_core_device_contracts.py` restores all five questions as hermetic
contracts. That choice is deliberate: each is a parser or solver seam whose
failure mode is structural — a device silently dropped, a DC short accepted in
a transient run, a stale voltage-keyed cache after a temperature change — so an
in-process assertion answers the same question as an NGSPICE comparison, runs
in the collected suite, and cannot rot behind a missing binary. The one
numerical check compares an RL divider against `jwL/(R + jwL)`, the closed form
of a linear network, which is an independent reference; no compact model is
compared against itself.

`core_gates.py` now names only its surviving caller and records what was lost.

### 3. Deck-to-deck parity cannot see engine disagreement

`physical_deck_mismatch` answers "are these two decks the same text?". It
cannot answer "do both engines read this text the same way?" — and both decks
render from the same template, so a card that NGSPICE honours and PyCircuitSim
drops yields two different problems and a clean parity report. Two such seams
exist:

- **Ignored directives.** `Parser.parse_line` deliberately falls through on
  every `.`-directive it does not implement. A template that gained
  `.options rshunt=1e12`, `.nodeset`, or `.measure` would change only the
  NGSPICE problem.
- **The `m` suffix.** `Parser.UNIT_SUFFIXES` mapped `m`/`M` to 1e6 with the
  comment "mega (milli is less common in circuits)". SPICE — and therefore the
  NGSPICE reference — reads `m` as milli. `Rload out 0 1m` was 1 mOhm to the
  reference and 1 MOhm to the candidate: a 1e9 divergence on a byte-identical
  deck that no parity check can see.

Both are now fixed rather than only guarded.

`Parser._parse_value` follows SPICE: scale factors are matched longest-first
and case-insensitively, `meg` and `mil` are recognized, `m` is milli, and
trailing unit text is ignored so `2.2kohm` is 2200. Every value in the table
was **measured on NGSPICE 45.2** by sourcing `V1 out 0 DC <token>` and reading
`v(out)`; `tests/common/parser_support.py` carries that measured table as the
reference, and the parser is held against it rather than against itself. No
deck, template, or tracked modelcard in the repository used the suffix, so no
existing result moves.

The parser now warns when it drops a directive that changes the circuit
(`Parser.PHYSICAL_DIRECTIVES`: `.options`, `.nodeset`, `.param`, `.global`,
`.noise`, `.pz`, `.sens`, `.disto`), once per directive per deck. Cards that
change only the output listing — `.print`, `.save`, `.measure`, `.end` — stay
silent, because warning about them would train the eye off the ones that
matter. Decks still parse: real corpora carry these cards, so this reports
rather than rejects.

`tests/test_deck_engine_compatibility.py` holds every template file and every
rendered catalog deck (both engines, all 79 analyses) against that surface. The
guard was verified by injecting `.options rshunt=1e12` and `R9 in 0 1m` into a
control template — both tests failed, both passed again on revert — and by
monkeypatching `m` back to 1e6, which the value scanner catches.

### 3a. The same bug, again, in the expression evaluator

Fixing `_parse_value` surfaced a second copy. `Parser._eval_expr` — the
`{...}` / `'...'` parameter evaluator that parameterized `.subckt` instances
go through — carried its own suffix regex with the same `m`-is-mega reading,
and could not evaluate `10n` or `1e-3` at all: the substituted float's own
exponent (`1e-08`) was then looked up as a parameter name and raised
"Unknown parameter 'e'". Two guards were added: the literal pattern refuses to
take a scientific-notation exponent as a scale factor, and the identifier pass
skips a letter that follows a digit or decimal point.

This is also where the audit's one **caught regression** came from. Changing
`UNIT_SUFFIXES` from a dict to an ordered tuple broke `_eval_expr`'s dict
lookup, and `verify_subckt.py` dropped from 11/11 to 7/8 — caught by re-running
the simulator-backed gates after the parser change, not by the unit suite. An
intermediate version of the fix also made `1e-3*2` evaluate to `-5` instead of
raising. Both are covered by contracts now, and the gate is back at 11/11. The
lesson is the audit's own: a unit suite that does not exercise the netlist path
end to end cannot certify a change to the netlist path.

`.op` is separated as an *implied* directive rather than silenced: it has no
branch in the dispatch, but it leaves `analysis_type` at `None`, which
`run_simulation` reads as "single DC operating point". The test pins both
halves, so a later change to that default cannot turn `.op` into a dropped
analysis.

### 4. The charge surface was measured at one corner

`verify_device_integrity` swept currents and derivatives across all fourteen
declared corners. `verify_terminal_integrity` — four-terminal currents and the
4x4 physical transcapacitance matrix, the surfaces the full-terminal LEVEL=75/76
families exist to provide — hard-coded `corner="nominal"` in four places and
had no corner flag. A charge model can be exact at 27 C / NFIN=2 and wrong at
125 C or NFIN=5, with no row to say so. The V7.6.8 audit listed this as
uncovered; it is now covered.

`--corner` is opt-in and defaults to `nominal`, so no existing campaign
denominator moves. Applicability delegates to `device_corner_applies`, so the
two single-device gates cannot disagree about which corners are no-ops for a
polarity, and a stress a technology cannot express (TSMC7 has no alternate
trained VT) yields an empty result rather than an infrastructure error.

## Enrichment

`cascode_ac` was the only AC profile in the catalog with an empty metric
contract, so the complementary cascode's small-signal gain — the quantity the
stage exists to produce — was scored as trace NRMSE alone. It now emits
per-polarity gain and headlines the worse of the two as
`cascode_gain_worst_error_pct`, which is also added to the case's
`required_metrics`. The branches are scored separately rather than averaged
because NMOS and PMOS are separate NN checkpoints, so one healthy polarity must
not cover for the other. Bandwidth is deliberately not required: the AC
branches carry a resistive load and no explicit output capacitance, so the
-3 dB point need not fall inside the declared sweep — the same correction
V7.6.8 had to make for the common-source diagnostics.

## Repairs

- Ten gate entry points ignored their argument vector entirely. `--tech TSMC5`
  on `verify_ac.py` was silently dropped and the full matrix ran, and `--help`
  on six of them launched NGSPICE instead of printing usage. All ten now use
  `tests/common/base.py:parse_no_options`, which answers `--help` and exits 2
  on an unknown flag — the same fail-closed rule the accuracy CLIs already
  apply to technology and analysis selections.
- Six references to a bundled `$PWD/tools/ngspice-45.2/bin/ngspice` remained in
  `README.md`, `docs/accuracy/methodology.md`, and four gate docstrings. That
  path does not exist in this checkout and `README.md` says so forty lines
  above one of the offending blocks; following the instruction would point
  `NGSPICE_BIN` at nothing. All now name `/usr/local/ngspice-45.2/bin/ngspice`,
  the code default in `tests/common/base.py`.
- `verify_bsimcmg_inverter_op.py` claimed its topology lived in
  `L1_primitives/inverter.spice.tmpl`; V7.6.8 moved it to `L2_stages/`.
- `models/mosfet_nn.py` pointed at `tests/verify_batched_tail.py`, which moved
  to `tests/perf/` in V7.5.9.
- Removed the empty `tests/tools/` directory.

## Merge and deletion decisions

Nothing was deleted. The two candidates were examined and both were kept:

- **`verify_nn_inverter` vs `verify_nn_multi_tech_tran`** and **`verify_nn_dc`
  vs `verify_nn_multi_tech_dc`.** The parametric gates' baselines are literally
  the same calls — `nn_sweep` imports `run_ngspice_inverter_vtc`,
  `run_pycircuitsim_nn_inverter_vtc`, `run_ngspice_nmos_dc` and the rest from
  `nn_gate` — so by configuration the fixed gates are a subset. They are kept
  because the *verdict* differs: `nn_gate` applies the tight qualification
  thresholds and `nn_sweep` the loose stress thresholds (`DC_NRMSE_PASS = 10.0`,
  "these are stress tests, not the tight inverter gate"). That is one gate per
  question, not two gates for one. `verify_nn_dc --tran-only` is additionally
  the only single-device transient anywhere in the tree.
- **`test_v760_full_terminal_directnet.py` vs
  `test_v761_full_terminal_bsimar.py`.** Four of their questions are the same
  asked of two families — closure reconstruction, the PMOS scalar-current
  stamp, explicit-family gating, and checksum-mutation rejection. Parametrizing
  them over family would remove roughly six duplicated tests. Left as-is: the
  merge is real but it is churn on 14 passing contracts, and it is recorded
  here as a recommendation rather than taken unilaterally.

## Still not covered

Carried forward from V7.6.8 and re-confirmed:

- `simple-v2` remains diagnostic; thresholds and three-repeat LEVEL=72
  stability are not frozen.
- The exhaustive five-technology, all-corner, four-family numerical campaign
  was not rerun here.
- Historical device-AC `/10` and opamp-AC `/5` totals still use the retired
  per-engine-bias definition.
- The generated-bias scale and cascode diagnostics remain NMOS-led; there is no
  complementary PMOS-only self-bias ladder.
- NN hierarchy is exercised dynamically through one nested inverter buffer;
  broader NN subcircuit DC and AC coverage is absent.
- `.noise`, `.pz`, `.disto`, `.sens` and controlled sources remain out of scope
  because the parser implements neither the analyses nor the elements.

New, and open:

- **`.options cshunt` / `.options rshunt` are still not applied.** `AGENTS.md`
  described them as an implemented contract; the only implementation was in the
  AnalogGym bench translator (`examples/.../pycircuitsim_bench/run_compare.py`,
  2,443 lines) deleted in V7.6.6. `AGENTS.md` now states this, keeps the
  measured V7.5.10 lesson (they are circuit elements worth 14% on an amplifier
  slew rate, and must be applied after subcircuit flattening) as the rule for
  any reimplementation, and the parser warns when it drops one. Building them
  is a simulator feature, not a harness repair, and is left open.
- Terminal integrity now *can* run every corner, but no full five-technology
  corner sweep has been scored. The V7.6.9 executable check covered TSMC12
  only.
- **The enriched `cascode_ac` metric has no live row yet.** `cascode_stack`
  does not converge on LEVEL=73 at TSMC5, TSMC12 or TSMC16 — all three
  analyses, including the two DC compliance sweeps the enrichment does not
  touch, return `ERROR` rows that keep their denominator slots. That is a
  reduced-family result, not a harness failure, and it predates this audit
  (V7.6.8 recorded the case converging on LEVEL=75). The new metric is verified
  through `compare_traces` on synthetic traces instead.

## Verification evidence

| Surface | Result |
|---|---|
| Collected unit suite | 314 passed (was 256), 2 CPU-only Torch warnings |
| `verify_simple_circuit_catalog` | 4 `simple-v1` + 29 `simple-v2` cases |
| `verify_circuit_sweep_canaries` | 4,854 applicable render/parity cells |
| `verify_accuracy_campaign_tools` | 600/600 unique clean jobs |
| Collected unit suite, after the parser fix | **357 passed** |
| Compatibility guard, negative controls | injected `.options` and `1m` caught; `m`-as-mega monkeypatch caught; clean on revert |
| Scale factors vs NGSPICE 45.2 | 15/15 tokens match the measured reference |
| `verify_subckt` regression caught and fixed | 11/11 -> 7/8 -> 11/11 |
| Terminal integrity, TSMC12 LEVEL=73, 3 corners x 2 devices | 54/54 rows characterized and converged, exit 0 |
| Gate `--help` / unknown-flag surface | all 33 standalone entry points answer `--help`; the 10 fixed-matrix gates reject an unknown flag with exit 2 |
| Import health | all 33 `verify_*`/`diag_*` modules import cleanly |

The terminal-integrity corner rows are diagnostics. They immediately show the
expected LEVEL=73 shape — sub-1 % four-terminal current error against a
35-51 % transcapacitance NRMSE with `gate_bulk_accuracy_supported: false` —
because the reduced 13-target family does not carry the gate/bulk charge
surface. That is the reduced family's declared limit, not a new failure, and
the rows exist so the full-terminal families can be measured against it.
