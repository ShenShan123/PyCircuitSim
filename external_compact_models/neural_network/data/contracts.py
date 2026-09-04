"""Shared names and ordered schemas for neural compact-model contracts."""

from typing import Tuple


FULL_TERMINAL_OUTPUT_CONTRACT = "full-terminal"
FULL_TERMINAL_OUTPUT_COLUMN_ORDER: Tuple[str, ...] = (
    "i_d", "i_g", "i_b", "qd", "qg", "qb",
)
BSIMAR_FULL_TERMINAL_COLUMN_ORDER: Tuple[str, ...] = (
    "qg", "qb", "qd", "i_d", "i_g", "i_b",
)
CANONICAL_SAFETY_REJECTION_REASONS: Tuple[str, ...] = (
    "internal_node_solve_failed",
    "non_finite_output",
    "terminal_current_over_1A",
)


def dataset_filename(
    scope: str,
    device: str,
    version_tag: str = "",
) -> str:
    """Return the canonical six-surface dataset name."""
    parts = [scope]
    if version_tag:
        parts.append(version_tag)
    # ``dnf`` is the existing architecture-neutral full-terminal data tag.
    parts.append("dnf")
    parts.append(device)
    return "_".join(parts) + ".npz"


__all__ = [
    "FULL_TERMINAL_OUTPUT_CONTRACT",
    "FULL_TERMINAL_OUTPUT_COLUMN_ORDER",
    "BSIMAR_FULL_TERMINAL_COLUMN_ORDER",
    "CANONICAL_SAFETY_REJECTION_REASONS",
    "dataset_filename",
]
