"""Entry points nothing else in the collected suite reaches (V7.7.2 audit).

Three seams had no collected witness:

* ``main.py`` -> ``run_simulation`` -> ``Visualizer`` is the workflow
  README.md documents for running a netlist directly; the gates call the
  analysis runners underneath it and nothing called the dispatcher (C11).
* Training reproducibility: provenance was covered end to end, but nothing
  asserted that the same seed, configuration, and data produce the same
  checkpoint.  During a retraining campaign that is the difference between
  "the new model is worse" and "this run was noisier" (C11).
* The gate inventory: the V7.6.10 audit verified by hand that every
  ``verify_*`` module answers ``--help`` and rejects an unknown flag; nothing
  re-checked it as modules came and went (addition candidate).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

from tests.common.base import control_deck, render_template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.config import DirectNetConfig  # noqa: E402
from neural_network.data.contracts import FULL_TERMINAL_OUTPUT_COLUMN_ORDER  # noqa: E402
from neural_network.training import trainer  # noqa: E402
from neural_network.utils.seed import set_seed  # noqa: E402

FULL_COLUMNS = list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)


# ---------------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("command", ([], ["simulate"]))
def test_main_runs_a_control_deck_and_writes_its_transient_artifacts(
    tmp_path: Path, command: list[str],
) -> None:
    deck = tmp_path / "rc_lowpass.sp"
    deck.write_text(render_template(control_deck("rc_lowpass.spice.tmpl"), {
        "TEMP": "27", "INPUT_DC": "1", "INPUT_AC": "0", "INPUT_PHASE": "",
        "RESISTANCE": "1k", "CAPACITANCE": "1p",
        "ANALYSIS": ".tran 10p 1n",
    }))
    output = tmp_path / "out"
    env = {**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), *command, str(deck),
         "-o", str(output)],
        capture_output=True, text=True, env=env, cwd=PROJECT_ROOT, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    csv_files = list((output / "rc_lowpass" / "tran").glob("*.csv"))
    assert csv_files, sorted(str(p) for p in output.rglob("*"))
    header = csv_files[0].read_text().splitlines()[0].lower()
    assert "time" in header and "out" in header

    missing = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), *command, str(tmp_path / "absent.sp")],
        capture_output=True, text=True, env=env, cwd=PROJECT_ROOT, timeout=300,
    )
    assert missing.returncode == 1
    assert "not found" in missing.stderr.lower()


_RC_TOKENS = {
    "TEMP": "27", "INPUT_DC": "1", "INPUT_AC": "1", "INPUT_PHASE": "",
    "RESISTANCE": "1k", "CAPACITANCE": "1p",
}


@pytest.mark.parametrize(("analysis", "subdir", "artifact"), (
    (".tran 10p 1n", "tran", "rc_lowpass_transient.csv"),
    (".dc V1 0 1 0.25", "dc", "rc_lowpass_dc_sweep.csv"),
    (".ac dec 5 1k 1meg", "ac", "rc_lowpass_ac_sweep.csv"),
    (".op", "dc_op", "rc_lowpass_dc_op_point.txt"),
))
def test_run_simulation_dispatches_every_analysis_kind_in_process(
    tmp_path: Path, analysis: str, subdir: str, artifact: str,
) -> None:
    """``run_simulation`` routes each analysis card to its runner and writer.

    The subprocess smoke test above proves the CLI.  This in-process witness
    is the one the coverage tracer can see — the V7.7.2 second follow-up
    measured ``simulation.py`` at 11 % for exactly that reason — and it walks
    the three analysis kinds plus the bare operating point the dispatcher
    owns, each to the artifact a user would open.
    """
    import matplotlib
    matplotlib.use("Agg")
    from pycircuitsim.simulation import run_simulation

    deck = tmp_path / "rc_lowpass.sp"
    deck.write_text(render_template(
        control_deck("rc_lowpass.spice.tmpl"), {**_RC_TOKENS, "ANALYSIS": analysis},
    ))
    run_simulation(str(deck), output_dir=str(tmp_path / "out"))

    written = tmp_path / "out" / "rc_lowpass" / subdir / artifact
    assert written.is_file(), sorted(str(p) for p in tmp_path.rglob("*"))
    assert "out" in written.read_text().lower()


# ---------------------------------------------------------------------------
# Training determinism
# ---------------------------------------------------------------------------
def _write_smoke_dataset(root: Path) -> Path:
    rng = np.random.default_rng(772)
    data_path = root / "training.npz"
    n_rows = 48
    np.savez(
        data_path,
        inputs=rng.uniform(-0.5, 0.5, size=(n_rows, 4)),
        geometry=np.column_stack([
            np.full(n_rows, 2.0), np.full(n_rows, 16e-9),
            np.full(n_rows, 300.15), np.zeros((n_rows, 12)),
        ]),
        outputs=np.column_stack([
            rng.normal(scale=1e-4, size=(n_rows, 3)),
            rng.normal(scale=1e-15, size=(n_rows, 3)),
        ]),
        meta_output_columns=np.asarray(FULL_COLUMNS),
        sample_class=np.zeros(n_rows, dtype=np.int8),
    )
    data_path.with_suffix(".npz.complete").write_text(json.dumps({
        "dataset": data_path.name,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_commit": "a" * 40,
        "source_dirty": False,
    }))
    return data_path


def _train_once(data_path: Path, checkpoint_dir: Path, seed: int) -> dict[str, torch.Tensor]:
    set_seed(seed)
    trainer.train_directnet(
        str(data_path),
        config=DirectNetConfig(
            batch_size=16, trunk_hidden=8, trunk_layers=1,
            max_epochs=2, patience=2,
        ),
        save_prefix="dnf_repeat", device_str="cpu", overwrite=True,
        num_tech_codes=2, p_unknown=0.0, split_mode="random",
    )
    return torch.load(checkpoint_dir / "dnf_repeat_best.pt", map_location="cpu")


def test_same_seed_configuration_and_data_reproduce_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two CPU runs at one seed must be bit-identical; a new seed must not be."""
    data_path = _write_smoke_dataset(tmp_path)
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: np.zeros(48, dtype=int),
    )
    monkeypatch.setattr(trainer, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(trainer, "_NUM_WORKERS", 1)

    first = _train_once(data_path, tmp_path, seed=42)
    second = _train_once(data_path, tmp_path, seed=42)
    other = _train_once(data_path, tmp_path, seed=43)

    assert first.keys() == second.keys() == other.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert any(not torch.equal(first[key], other[key]) for key in first)


# ---------------------------------------------------------------------------
# Gate inventory
# ---------------------------------------------------------------------------
GATE_MODULES = sorted(
    f"tests.{path.parent.name}.{path.stem}"
    for tier in ("perf", "simple_circuits", "single_devices")
    for path in (PROJECT_ROOT / "tests" / tier).glob("verify_*.py")
)


def _exit_code(
    main: Callable[..., int], argv: list[str], monkeypatch: pytest.MonkeyPatch,
) -> int:
    """Run a gate's ``main`` on ``argv`` and normalize its exit to a code."""
    try:
        if "argv" in inspect.signature(main).parameters:
            return int(main(argv))
        monkeypatch.setattr(sys, "argv", ["gate", *argv])
        return int(main())
    except SystemExit as exit_request:
        code = exit_request.code
        return 0 if code is None else int(code)


def test_gate_inventory_is_enumerated_not_hand_counted() -> None:
    assert len(GATE_MODULES) == 29, GATE_MODULES


@pytest.mark.parametrize("module_name", GATE_MODULES)
def test_every_gate_answers_help_and_rejects_an_unknown_flag(
    module_name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)
    main = getattr(module, "main")

    assert _exit_code(main, ["--help"], monkeypatch) == 0
    assert "usage:" in capsys.readouterr().out.lower()
    assert _exit_code(main, ["--no-such-flag-v772"], monkeypatch) == 2
