"""Shared test infrastructure for PyCircuitSim verification suites.

Layout:
- `base`         — project paths, NGSPICE subprocess runner, TechProfile,
                    VtPair, generic orchestration helpers.
- `bsimcmg_dc`   — DC-specific runners, metrics, plots (BSIM-CMG LEVEL=72).
- `bsimcmg_tran` — Transient-specific runners, metrics, plots (BSIM-CMG LEVEL=72).
- `nn`           — Helpers shared across NN verification scripts
                    (DirectNet LEVEL=73 + BSIM-AR LEVEL=74).

Downstream verify_*.py scripts should import from
`tests.common.<module>` rather than from the old flat `tests.*_common`.
"""
