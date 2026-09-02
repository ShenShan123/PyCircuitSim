"""Shared names and ordered schemas for neural compact-model contracts."""

from typing import Tuple


REDUCED_OUTPUT_CONTRACT = "reduced"
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
    output_contract: str,
    version_tag: str = "",
) -> str:
    """Return the isolated canonical dataset name for one output contract."""
    if output_contract not in (
        REDUCED_OUTPUT_CONTRACT, FULL_TERMINAL_OUTPUT_CONTRACT,
    ):
        raise ValueError(f"Unknown output contract: {output_contract}")
    parts = [scope]
    if version_tag:
        parts.append(version_tag)
    if output_contract == FULL_TERMINAL_OUTPUT_CONTRACT:
        parts.append("dnf")
    parts.append(device)
    return "_".join(parts) + ".npz"


__all__ = [
    "REDUCED_OUTPUT_CONTRACT",
    "FULL_TERMINAL_OUTPUT_CONTRACT",
    "FULL_TERMINAL_OUTPUT_COLUMN_ORDER",
    "BSIMAR_FULL_TERMINAL_COLUMN_ORDER",
    "CANONICAL_SAFETY_REJECTION_REASONS",
    "dataset_filename",
]
