"""Structured result records shared by simple-circuit gates and collectors."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


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
    control_converged: Optional[bool] = None
    partial: bool = False
    execution_state: str = ""
    error_kind: str = ""
    model_family: str = ""
    model_level: int = 0
    checkpoint_pins: Dict[str, str] = field(default_factory=dict)
    campaign_manifest_sha256: str = ""
    thread_settings: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"qualification", "diagnostic"}:
            raise ValueError(f"unknown gate role {self.role!r}")
        if self.status not in {"pass", "fail", "diagnostic", "error"}:
            raise ValueError(f"unknown gate status {self.status!r}")
        if self.role == "diagnostic" and self.status in {"pass", "fail"}:
            raise ValueError("diagnostic results cannot enter pass/fail scoring")
        if self.role == "qualification" and self.status == "diagnostic":
            raise ValueError("qualification results require pass/fail/error status")
        execution_state = self.execution_state
        if not execution_state:
            if self.partial:
                execution_state = "partial"
            elif not self.reference_converged:
                execution_state = "reference_error"
            elif not self.candidate_converged:
                execution_state = "nonconverged"
            elif self.status == "error":
                execution_state = "error"
            else:
                execution_state = "complete"
            object.__setattr__(self, "execution_state", execution_state)
        allowed_states = {
            "complete", "partial", "nonconverged", "reference_error",
            "infrastructure_error", "unsupported", "error",
        }
        if execution_state not in allowed_states:
            raise ValueError(f"unknown execution state {execution_state!r}")
        if execution_state == "complete" and (
            self.partial
            or not self.reference_converged
            or not self.candidate_converged
            or self.control_converged is False
        ):
            raise ValueError(
                "complete execution requires complete, converged reference, "
                "candidate, and requested control solves"
            )
        if execution_state != "complete" and self.status != "error":
            raise ValueError(
                "incomplete execution must use error status; "
                f"got {execution_state!r} with {self.status!r}"
            )
        if execution_state == "complete" and self.status == "error":
            raise ValueError("error status requires a non-complete execution state")
        error_kind = self.error_kind
        if not error_kind and self.status == "error":
            if execution_state == "reference_error":
                error_kind = "reference"
            elif execution_state in {"partial", "nonconverged"}:
                error_kind = "candidate"
            else:
                error_kind = "unknown"
            object.__setattr__(self, "error_kind", error_kind)
        if error_kind not in {
            "", "candidate", "reference", "infrastructure", "unsupported",
            "result_schema", "unknown",
        }:
            raise ValueError(f"unknown error kind {error_kind!r}")
        if self.model_level and self.model_level not in {73, 74, 75, 76}:
            raise ValueError(f"unsupported NN model level {self.model_level}")
        if self.model_level and not self.model_family:
            raise ValueError("model family is required when model level is recorded")
        if self.campaign_manifest_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", self.campaign_manifest_sha256,
        ):
            raise ValueError("campaign manifest digest must be 64 lowercase hex chars")
        if any(value < 1 for value in self.thread_settings.values()):
            raise ValueError("recorded thread counts must be positive")

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


def result_exit_code(results: Sequence[GateResult]) -> int:
    """Return 0 success, 1 scientific miss, or 2 infrastructure failure."""
    if not results:
        return 2
    infrastructure = {
        "infrastructure", "reference", "result_schema", "unknown",
        "unsupported",
    }
    if any(
        result.status == "error" and result.error_kind in infrastructure
        for result in results
    ):
        return 2
    if any(result.status in {"fail", "error"} for result in results):
        return 1
    return 0
