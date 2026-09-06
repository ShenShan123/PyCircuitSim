"""Retained LEVEL=72 gates must not hide execution or operating-point errors."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tests.common import base, bsimcmg_tran


@pytest.mark.parametrize("multi_tech", (False, True))
def test_mixed_pass_and_error_is_an_unsuccessful_suite(
    tmp_path: Path, multi_tech: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    """One successful configuration cannot hide a second failed execution."""
    tech = base.ALL_TECHS["ASAP7"]
    configs = [bsimcmg_tran.make_baseline(tech, config_name=name)
               for name in ("good", "broken")]
    saved: list[dict[str, Any]] = []

    def run_single(config: bsimcmg_tran.TestConfig, work_dir: Path) -> dict[str, Any]:
        if config.config_name == "broken":
            raise RuntimeError("controlled execution failure")
        return {"config": config, "passed": True}

    def save(rows: list[dict[str, Any]], path: Path) -> None:
        saved.extend(rows)

    kwargs = dict(
        results_dir=tmp_path, title="mixed verdicts", acceptance_msg="all cells",
        run_single_fn=run_single,
        print_summary_fn=lambda rows: (1, 0, 1),
        save_csv_fn=save, plot_bar_fn=lambda *_args: None,
    )
    if multi_tech:
        code = base.run_multi_tech_main(
            tech_names=["ASAP7"], make_baseline_fn=lambda _tech: configs[0],
            build_parametric_fn=lambda _tech: configs[1:], **kwargs,
        )
    else:
        code = base.run_test_suite(configs=configs, **kwargs)
    assert len(saved) == 2
    assert saved[0]["passed"] is True
    assert saved[1]["error"] == "controlled execution failure"
    assert code != 0
    if multi_tech:
        assert "ASAP7   : ERROR" in capsys.readouterr().out


def test_reference_transient_requires_a_converged_initial_operating_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned voltage mapping is insufficient evidence of a physical OP."""
    from pycircuitsim import parser, solver

    config = bsimcmg_tran.make_baseline(base.ALL_TECHS["ASAP7"])
    monkeypatch.setattr(bsimcmg_tran, "create_pycircuitsim_netlist",
                        lambda *_args: tmp_path / "unused.sp")
    monkeypatch.setattr(bsimcmg_tran, "get_merged_modelcard",
                        lambda *_args: tmp_path / "unused.lib")
    parsed = Mock(analysis_params={"tstep": 1e-12, "tstop": 1e-9})
    monkeypatch.setattr(parser, "Parser", Mock(return_value=parsed))
    op = Mock(_last_solve_converged=False)
    op.solve.return_value = {"in": 0.0, "out": 0.7}
    monkeypatch.setattr(solver, "DCSolver", Mock(return_value=op))
    transient = Mock()
    transient.return_value.solve.return_value = {"time": [], "in": [], "out": []}
    monkeypatch.setattr(solver, "TransientSolver", transient)

    with pytest.raises(RuntimeError, match="operating point.*converg"):
        bsimcmg_tran.run_pycircuitsim(config, tmp_path)
    transient.assert_not_called()
