"""Structured result records shared by simple-circuit gates and collectors."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


RESULT_MARKER = "===PYCIRCUITSIM_GATE_RESULT "


def _json_value(value: Any) -> Any:
    """Convert numpy-like scalars and non-finite floats to strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class GateResult:
    """One technology/corner/analysis outcome at the harness seam."""

    case_id: str
    tech: str
    corner: str
    analysis: str
    role: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    domain: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    reference_converged: bool = True
    candidate_converged: bool = True
    partial: bool = False

    def __post_init__(self) -> None:
        if self.role not in {"qualification", "diagnostic"}:
            raise ValueError(f"unknown gate role {self.role!r}")
        if self.status not in {"pass", "fail", "diagnostic", "error"}:
            raise ValueError(f"unknown gate status {self.status!r}")
        if self.role == "diagnostic" and self.status in {"pass", "fail"}:
            raise ValueError("diagnostic results cannot enter pass/fail scoring")
        if self.role == "qualification" and self.status == "diagnostic":
            raise ValueError("qualification results require pass/fail/error status")

    def payload(self) -> Dict[str, Any]:
        """Strict-JSON-compatible representation used by logs and reports."""
        return _json_value(asdict(self))

    def marker(self) -> str:
        """Single-line, schema-stable campaign log marker."""
        return RESULT_MARKER + json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ) + "==="


def parse_result_markers(text: str) -> list[Dict[str, Any]]:
    """Parse every structured result marker from a worker log."""
    results: list[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith(RESULT_MARKER) or not line.endswith("==="):
            continue
        raw = line[len(RESULT_MARKER):-3]
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("gate result marker payload must be an object")
        results.append(value)
    return results
