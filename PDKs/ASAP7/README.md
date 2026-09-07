# ASAP7 BSIM-CMG modelcards

This directory contains the tracked ASAP7 FinFET cards used for LEVEL=72
reference checks. The repository [README](../../README.md) owns setup,
commands, and release metadata. ASAP7 is outside the current LEVEL=75/76 NN
checkpoint scope; private TSMC cards belong in their separate ignored PDK
folders.

## Cards and model names

| Process corner | Dated source card | Companion card |
|---|---|---|
| Typical–typical | [`7nm_TT_160803.pm`](7nm_TT_160803.pm) | [`7nm_TT.pm`](7nm_TT.pm) |
| Slow–slow | [`7nm_SS_160803.pm`](7nm_SS_160803.pm) | [`7nm_SS.pm`](7nm_SS.pm) |
| Fast–fast | [`7nm_FF_160803.pm`](7nm_FF_160803.pm) | [`7nm_FF.pm`](7nm_FF.pm) |

Each dated card declares four NMOS and four PMOS models:

| Variant | NMOS | PMOS |
|---|---|---|
| Low threshold | `nmos_lvt` | `pmos_lvt` |
| Regular threshold | `nmos_rvt` | `pmos_rvt` |
| Super-low threshold | `nmos_slvt` | `pmos_slvt` |
| SRAM | `nmos_sram` | `pmos_sram` |

Read geometry and process parameters from the selected card. For example, the
TT card's `hfin=3.2e-8` is **32 nm**, not 3.2 nm. A technology label is not a
substitute for the instance's L/NFIN/TFIN or its selected process corner.

## Evaluator integration

The [PyCMG example](../../README.md#pycmg-reference-tools) loads the tracked TT
card with `Model(osdi_path, modelcard_path, model_name)` and reads the returned
dictionary's `id` terminal-current field. It uses the current `PDKs/ASAP7/`
and `build/osdi/` locations.

[`pycmg/parser.py`](../../external_compact_models/bsim_cmg/pycmg/parser.py)
infers `DEVTYPE` from the `.model` NMOS/PMOS type when the card omits it:
NMOS is 1, PMOS is 0. No separately patched `with_devtype` card is needed.
The source cards remain unchanged. Raw PyCMG currents and circuit-stamp
currents have different sign conventions; the
[PyCMG API guide](../../external_compact_models/bsim_cmg/README.md#evaluator-api)
explains the adapter conversion.

Ground truth is NGSPICE loading the same BSIM-CMG OSDI binary. Merely adding
an `.include` does not load the OSDI model; use the shared test renderer and
runner described in the [test guide](../../tests/README.md). Materialized
netlists, traces, and reports belong under root `results/`.

## Evidence

Bundling TT/SS/FF cards does not claim that every corner has been qualified.
The test registries select actual device, geometry, and analysis coverage;
[accuracy reports](../../docs/accuracy/README.md) identify measured campaigns.
Release outcomes and retired examples remain in the
[changelog and Git history](../../docs/CHANGELOG.md#v775--repository-cleanup).
