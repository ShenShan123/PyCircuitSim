"""Stdlib-only analysis contract for NN hierarchy equivalence."""
from __future__ import annotations

from tests.common.simple_circuit_catalog import AnalysisSpec

SUBCKT_ANALYSES: tuple[AnalysisSpec, ...] = (
    AnalysisSpec(
        "dc", "dc", "dc Vin 0 <VDD> 0.005", ("v(out)", "v(mid)"),
    ),
    AnalysisSpec(
        "tran", "tran", "tran 2p 4n uic", ("v(out)", "v(mid)"),
    ),
    AnalysisSpec(
        "ac", "ac", "ac dec 10 1e3 1e11", ("v(out)", "v(mid)"),
    ),
)
