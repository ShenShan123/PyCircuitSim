"""Test harness root package.

Verification scripts live flat in this directory (`verify_*.py`).
Shared infrastructure lives in `tests.common` (base helpers, BSIM-CMG DC/tran,
NN helpers). Every circuit a gate simulates lives in `examples/` and is
rendered by the gate — tests carry no netlists of their own.
"""
