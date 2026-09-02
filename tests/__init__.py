"""Test harness root package.

Gates are grouped by the tier of the circuit they gate — `single_devices`
and `simple_circuits` — mirroring `examples/`; `perf` and
`diag` sit outside that axis on purpose (see their own `__init__`).
Shared infrastructure lives in `tests.common` (base helpers, BSIM-CMG DC/tran,
NN helpers). Every circuit a gate simulates lives in `examples/` and is
rendered by the gate — tests carry no netlists of their own.

**One gate per question** (V7.5.9). A gate whose configs are a subset of
another gate's matrix, or that differs from its neighbour only by a string,
is not extra coverage — it is the same measurement paid for twice. When two
gates converge on one question, merge them behind a flag and say in the
survivor's docstring what it absorbed.
"""
