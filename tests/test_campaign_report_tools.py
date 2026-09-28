"""Contracts for the campaign report tools (V7.7.6 case matrix)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from campaign_case_matrix import cell_text  # noqa: E402


def _row(status: str, nrmse: float | None = None) -> dict:
    metrics = {} if nrmse is None else {"nrmse_pct": nrmse}
    return {"status": status, "metrics": metrics}


def test_error_rows_keep_their_slot_but_never_enter_the_median() -> None:
    """Methodology §5: an ERROR row stays in the denominator, not the numbers.

    An unconverged row's recovered metrics must not pull the median; its count
    is shown instead, and a case with no converged row reads ``E``.
    """
    rows = [_row("diagnostic", 1.0), _row("diagnostic", 3.0),
            _row("error", 900.0), _row("fail", 2.0)]
    assert cell_text(rows) == "2.00 1E 1F"
    assert cell_text([_row("error", 5.0), _row("error")]) == "E"
    assert cell_text([_row("pass")]) == "—"
    assert cell_text(None) == "…"


def test_sram_gate_rows_use_their_worst_lobe_nrmse() -> None:
    row = {"status": "pass", "metrics": {"worst_nrmse_pct": 6.1}}
    assert cell_text([row]) == "6.10"


def test_matrix_is_structured_as_l0_to_l4_level_evaluations() -> None:
    """Each case sits under its template level; an unleveled suite is fatal.

    A level summary pools the level's rows, so an all-ERROR level reads as a
    0% converged share rather than disappearing from the table.
    """
    import pytest
    from campaign_case_matrix import render

    cells = {
        ("dnf", "small", "TSMC5", "verify_nn_multi_tech_dc"): [_row("pass", 1.0)],
        ("dnf", "small", "TSMC5",
         "verify_circuit_topologies__ldo_regulator"): [_row("error", 3.0)],
    }
    text = render(cells, "t")
    assert (text.index("## L0") < text.index("| device_dc |")
            < text.index("## L4") < text.index("| ldo_regulator |"))
    level4 = text[text.index("## L4"):]
    assert "| TSMC5 | — · 0% |" in level4
    with pytest.raises(SystemExit, match="without a level"):
        render({("dnf", "small", "TSMC5", "verify_unknown"): []}, "t")
