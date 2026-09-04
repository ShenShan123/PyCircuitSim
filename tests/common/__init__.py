"""Shared test infrastructure for PyCircuitSim verification suites.

Layout:
- `base`         — project paths, NGSPICE subprocess runner, TechProfile,
                    VtPair, generic orchestration helpers.
- `bsimcmg_dc`   — DC-specific runners, metrics, plots (BSIM-CMG LEVEL=72).
- `bsimcmg_tran` — Transient-specific runners, metrics, plots (BSIM-CMG LEVEL=72).
- `nn`           — helpers shared across full-terminal NN verification scripts.

Downstream verify_*.py scripts should import from
`tests.common.<module>` rather than from the old flat `tests.*_common`.
"""
