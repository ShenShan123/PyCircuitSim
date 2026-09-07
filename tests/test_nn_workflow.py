"""The root workflow must preserve stage dependencies and campaign isolation."""
from __future__ import annotations

import argparse
import inspect
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

import main as cli


def test_generation_controls_are_forwarded_to_the_backend(tmp_path: Path) -> None:
    args = cli.parse_args([
        "data", "--run-dir", str(tmp_path), "--dry-run", "--device", "nmos",
        "--sampler", "lhs", "--n-lhs-samples", "123", "--temperatures", "280,300",
        "--voltage-box-factor", "1.8", "--variants", "lvt", "--data-seed", "7",
        "--data-version", "custom", "--no-data-enable-inv-trip",
    ])
    _, command = cli.generation_commands(args)[0]
    assert command[command.index("--sampler") + 1] == "lhs"
    assert command[command.index("--n-lhs-samples") + 1] == "123"
    assert command[command.index("--temperatures") + 1] == "280.0,300.0"
    assert command[command.index("--seed") + 1] == "7"
    assert "--enable-inv-trip" not in command
    assert cli.datasets(args)[0][2].name == "tsmc5_custom_dnf_nmos.npz"


def test_training_loss_and_rollout_options_reach_the_existing_trainer(tmp_path: Path) -> None:
    args = cli.parse_args([
        "train", "--dry-run", "--run-dir", str(tmp_path), "--model", "transformer",
        "--recipe", "ar3roll", "--subthresh", "--lam-subthresh", "0.2",
        "--class-weights", "subvt_off=3", "--swa-mode", "swa",
        "--lr", "0.0002", "--patience", "12", "--split-mode", "random",
    ])
    stem, command = cli.training_commands(args)[0]
    assert stem == "tsmc5_tff_ar3roll_small_nmos"
    assert "--autoregressive-training" in command and "--subthresh" in command
    assert command[command.index("--full-terminal-ar-targets") + 1] == "3"
    assert command[command.index("--lam-subthresh") + 1] == "0.2"
    assert command[command.index("--swa-mode") + 1] == "swa"


def test_named_evaluation_selects_only_requested_cases() -> None:
    args = cli.parse_args([
        "evaluate", "--circuit", "ring_osc", "current_mirror", "--dry-run",
        "--device-suite", "nn_multi_tech_dc",
    ])
    assert set(args.pools) == {"clean", "simple_v2"}
    assert set(args.selected_cases) == {"ring_osc", "current_mirror", "nn_multi_tech_dc"}


_BACKEND_SAMPLES = {
    "data": {
        "device": "both", "tech": "tsmc7", "variants": "lvt", "universal": True,
        "temperatures": "280,310", "n_lhs_samples": "123", "voltage_box_factor": "1.8",
        "sampler": "lhs", "grid_per_axis": "13", "vbs_levels": "3", "hot_per_axis": "7",
        "jitter_sigma_frac": "0.03", "n_workers": "2", "seed": "17", "finetune_size": "80",
        "version": "trial", "exclude_techs": "tsmc16", "enable_inv_trip": True,
        "enable_subvt_off": True, "dc_solve_tol": "1e-13", "max_l_ratio": "1.2",
        "allow_rejected_points": True, "allow_safety_rejections": True, "data_dir": "/tmp/data",
        "overshoot_per_axis": "4", "n_vbs_lhs": "11", "quiet": True,
    },
    "train": {
        "model": "transformer", "size": "large", "device_type": "both", "data": "/tmp/input.npz",
        "epochs": "3", "batch_size": "16", "lr": "0.002", "patience": "7", "max_rows": "100",
        "split_mode": "random", "cuda": True, "amp": True, "seed": "17",
        "exclude_techs": "tsmc16", "num_tech_codes": "4", "p_unknown": "0.2",
        "tech_scope": "tsmc7", "exp_name": "custom", "overwrite": True, "swa_mode": "swa",
        "ema_decay": "0.9", "class_weights": "subvt_off=3", "training_overlay_classes": "traj_corridor",
        "init_from": "/tmp/initial.pt", "full_terminal_ar_targets": "3", "autoregressive_training": True,
        "subthresh": True, "lam_subthresh": "0.2", "subthresh_s2": "2e-9",
        "subthresh_upper": "2e-6", "subthresh_floor": "2e-12", "subthresh_off_floor": "2e-10",
        "subthresh_ceiling_k": "2.0", "subthresh_ceiling_w": "3.0",
    },
}


@pytest.mark.parametrize("prefix", ("data", "train"))
def test_backend_forwarding_inventory_covers_every_option(prefix: str) -> None:
    backend = cli.generation_parser() if prefix == "data" else cli.training_parser()
    assert set(_BACKEND_SAMPLES[prefix]) == {a.dest for a in backend._actions if a.dest != "help"}


@pytest.mark.parametrize("mode", ("alias", "prefixed", "flow"))
@pytest.mark.parametrize(("prefix", "name", "value"), [
    (prefix, name, value) for prefix, samples in _BACKEND_SAMPLES.items() for name, value in samples.items()
])
def test_every_backend_option_survives_root_command_dispatch(
    prefix: str, name: str, value: str | bool, mode: str,
) -> None:
    """An accepted setting must reach the child, including both polarity jobs."""
    backend = cli.generation_parser() if prefix == "data" else cli.training_parser()
    shared = {
        "data": {"device", "tech", "data_dir", "n_workers"},
        "train": {"model", "size", "device_type", "tech_scope", "epochs", "batch_size", "seed"},
    }
    flag = "--" + name.replace("_", "-")
    if mode != "alias" and name not in shared[prefix]:
        flag = f"--{prefix}-{name.replace('_', '-')}"
    stage = "flow" if mode == "flow" else prefix
    argv = [stage, "--dry-run"]
    if prefix == "train":
        argv += ["--model", "transformer", "--device", "both" if stage == "flow" else "nmos"]
    argv += [flag] if value is True else [flag, str(value)]
    standalone_only = {"data": {"universal", "version", "allow_rejected_points"},
                       "train": {"data", "exp_name"}}
    if stage == "flow" and name in standalone_only[prefix]:
        with pytest.raises(SystemExit) as exc:
            cli.parse_args(argv)
        assert exc.value.code == 2
        return
    args = cli.parse_args(argv)
    commands = cli.generation_commands(args) if prefix == "data" else cli.training_commands(args)
    for _, command in commands:
        child = backend.parse_args(command[3:] if prefix == "data" else command[4:])
        expected = value
        if name in ("device", "device_type"):
            expected = "nmos" if command == commands[0][1] else "pmos"
        action = next(action for action in backend._actions if action.dest == name)
        if action.type:
            expected = action.type(str(expected))
        assert getattr(child, name) == expected
    if name in ("device", "device_type"):
        assert len(commands) == 2


def test_generator_public_inputs_have_cli_controls() -> None:
    from external_compact_models.bsim_cmg.scripts import generate_nn_data as generator

    controls = {action.dest for action in cli.generation_parser()._actions}
    aliases = {"device_type": "device", "variant_names": "variants", "verbose": "quiet"}
    for function in (generator.generate_dataset, generator.generate_universal_dataset):
        parameters = {aliases.get(name, name) for name in inspect.signature(function).parameters}
        assert parameters <= controls


@pytest.mark.parametrize(("prefix", "flag"), [
    (prefix, action.option_strings[0])
    for prefix, backend in (("data", cli.generation_parser()), ("train", cli.training_parser()))
    for action in backend._actions if isinstance(action, argparse._StoreTrueAction)
])
def test_boolean_options_can_be_explicitly_disabled(prefix: str, flag: str) -> None:
    argv = [prefix, "--dry-run", f"--{prefix}-{flag[2:]}", f"--no-{prefix}-{flag[2:]}"]
    if prefix == "train":
        argv += ["--model", "transformer"]
    args = cli.parse_args(argv)
    commands = cli.generation_commands(args) if prefix == "data" else cli.training_commands(args)
    assert all(flag not in command for _, command in commands)


@pytest.mark.parametrize("flag", ("--overshoot-per-axis", "--n-vbs-lhs"))
def test_experimental_overlay_counts_reject_negative_values(flag: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["data", "--dry-run", flag, "-1"])
    assert exc.value.code == 2
    args = cli.parse_args(["data", "--dry-run", flag, "0"])
    assert all(command[command.index(flag) + 1] == "0"
               for _, command in cli.generation_commands(args))


@pytest.mark.parametrize("universal", (False, True))
@pytest.mark.parametrize("options", ([], ["--overshoot-per-axis", "4", "--n-vbs-lhs", "11", "--quiet"]))
def test_generator_dispatches_overlay_and_verbosity_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, universal: bool, options: list[str],
) -> None:
    from external_compact_models.bsim_cmg.scripts import generate_nn_data as generator

    class CapturedGeneration(Exception):
        pass

    captured: dict[str, object] = {}

    def capture(*args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise CapturedGeneration

    monkeypatch.setattr(generator, "generate_dataset", capture)
    monkeypatch.setattr(generator, "generate_universal_dataset", capture)
    monkeypatch.setenv("NN_DC_SOLVE_TOL", "1e-12")
    monkeypatch.setattr(sys, "argv", ["generate_nn_data", "--data-dir", str(tmp_path),
                                      *(["--universal"] if universal else []), *options])
    with pytest.raises(CapturedGeneration):
        generator.main()
    assert captured["overshoot_per_axis"] == (4 if options else 0)
    assert captured["n_vbs_lhs"] == (11 if options else 0)
    assert captured["verbose"] is (not options)


def test_named_job_lists_keep_recipe_identity_and_selected_denominators(tmp_path: Path) -> None:
    from scripts import v710_regate_jobs as jobs

    assert jobs.main([
        str(tmp_path), "--tag", "tff", "--size", "small", "--tech", "TSMC5", "--omp", "1",
        "--recipe", "ar3", "ar3roll", "--case", "ring_osc", "current_mirror", "nn_multi_tech_dc",
    ]) == 0
    clean = (tmp_path / "jobs_clean.txt").read_text().splitlines()
    diagnostic = (tmp_path / "jobs_simple_v2.txt").read_text().splitlines()
    assert set(clean) == {
        f"tff {recipe}_small TSMC5 {suite} 1"
        for recipe in ("ar3", "ar3roll") for suite in ("verify_circuit_ring_osc", "verify_nn_multi_tech_dc")}
    assert set(diagnostic) == {
        f"tff {recipe}_small TSMC5 verify_circuit_topologies__current_mirror 1" for recipe in ("ar3", "ar3roll")}
    assert not (tmp_path / "jobs_canary.txt").exists()


def test_generation_supports_universal_and_diagnostic_modes_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(["data", "--universal", "--allow-rejected-points", "--finetune-size", "50",
                           "--run-dir", str(tmp_path), "--dry-run"])
    commands = cli.generation_commands(args)
    assert len(commands) == 2
    assert all("--universal" in command for _, command in commands)
    assert all(command[command.index("--exclude-techs") + 1] == "asap7" for _, command in commands)
    assert {path.name for _, _, path in cli.datasets(args)} == {"universal_dnf_nmos.npz", "universal_dnf_pmos.npz"}
    calls: list[str] = []
    args.dry_run = False
    monkeypatch.setattr(cli, "run_command", lambda name, command, env, root: calls.append(name))
    assert cli.execute(args, cli.environment(args)) == 0
    assert calls == ["data-universal-nmos", "data-universal-pmos"]


def test_universal_generation_honors_equals_style_scope_selection() -> None:
    args = cli.parse_args(["data", "--universal", "--tech=tsmc5", "--dry-run"])
    assert args.tech == ["tsmc5"]
    _, command = cli.generation_commands(args)[0]
    assert set(command[command.index("--exclude-techs") + 1].split(",")) == {
        "asap7", "tsmc6", "tsmc7", "tsmc12", "tsmc16"}


def test_custom_training_inputs_names_and_explicit_overrides(tmp_path: Path) -> None:
    args = cli.parse_args([
        "train", "--data", str(tmp_path / "custom.npz"), "--device", "nmos", "--tech-scope", "universal",
        "--exp-name", "experiment", "--recipe", "s7", "--seed", "9", "--overwrite", "--amp", "--dry-run",
        "--max-rows", "100", "--init-from", str(tmp_path / "initial.pt"), "--p-unknown", "0.2",
    ])
    stem, command = cli.training_commands(args)[0]
    assert stem == "experiment_nmos"
    assert command[command.index("--data") + 1] == str(tmp_path / "custom.npz")
    assert command[command.index("--seed") + 1] == "9"
    assert command[command.index("--exclude-techs") + 1] == "asap7"
    assert "--overwrite" in command and "--amp" in command


@pytest.mark.parametrize("options", (
    ["--circuit", "unknown"], ["--circuit", "opamp", "opamp"],
    ["--circuit", "current_mirror", "--pools", "clean"],
    ["--evaluation-group", "single_devices", "--circuit", "ring_osc"],
))
def test_invalid_evaluation_selection_fails_before_any_work(options: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--dry-run", *options])
    assert exc.value.code == 2


def test_explicit_geometry_case_does_not_check_unrequested_circuits() -> None:
    from tests.single_devices import verify_data_geometry_coverage as geometry

    points = geometry._simple_circuit_geometries(["current_mirror"])
    assert points
    assert all(":current_mirror:nominal:" in point.label for point in points)
    assert not geometry._simple_circuit_geometries([])


def test_flow_executes_the_same_commands_as_the_separate_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    def capture(name: str, command: list[str], env: dict[str, str], root: Path) -> None:
        calls.append((name, command, env))

    monkeypatch.setattr(cli, "run_command", capture)
    monkeypatch.setattr(cli, "evaluation_verdict", lambda *args: 0)
    for stage in ("data", "train", "evaluate", "flow"):
        args = cli.parse_args([stage, "--run-dir", str(tmp_path), "--dry-run"])
        args.dry_run = False
        assert cli.execute(args, cli.environment(args)) == 0
        if stage == "evaluate":
            separate = calls[:]
            calls.clear()
    # NGSPICE_BIN is needed only by the evaluation stage; commands and output
    # directories must otherwise agree, in the same dependency order.
    assert [(name, command) for name, command, _ in calls] == [
        (name, command) for name, command, _ in separate]
    for (_, _, flow_env), (_, _, stage_env) in zip(calls, separate):
        assert flow_env["BSIMAR_DATA_DIR"] == stage_env["BSIMAR_DATA_DIR"]
        assert flow_env["BSIMAR_CHECKPOINT_DIR"] == stage_env["BSIMAR_CHECKPOINT_DIR"]


def test_all_remains_an_alias_for_flow() -> None:
    assert cli.parse_args(["all", "--dry-run"]).stage == "flow"


@pytest.mark.parametrize("argv", (
    ["simulate", "circuit.sp", "-o", "results/run", "-v"],
    ["circuit.sp", "-o", "results/run", "-v"],
    ["-o", "results/run", "-v", "circuit.sp"],
))
def test_simulation_keeps_legacy_arguments_and_the_explicit_command(argv: list[str]) -> None:
    args = cli.parse_args(argv)
    assert args.stage == "simulate"
    assert args.netlist == "circuit.sp"
    assert args.output_dir == "results/run"
    assert args.verbose is True


@pytest.mark.parametrize("stage", ("data", "train", "evaluate", "flow"))
def test_preview_works_outside_repo_without_writing(
    tmp_path: Path, stage: str,
) -> None:
    output = tmp_path / "run with spaces"
    result = subprocess.run(
        [sys.executable, str(cli.ROOT / "main.py"), stage,
         "--run-dir", str(output), "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert str(output) in result.stdout
    assert not output.exists()
    if stage in ("evaluate", "flow"):
        assert "v710_regate.sh" in result.stdout
        assert "--require-manifest" in result.stdout
        assert "LEVEL=75" in result.stdout


@pytest.mark.parametrize("options", (
    ["--tech", "asap7"], ["--tech", "tsmc5", "tsmc5"],
    ["--model", "unknown"], ["--size", "large,"],
    ["--workers", "0"], ["--gpus", "0", "0"],
    ["--device", "nmos"], ["--unknown"],
))
def test_bad_flow_options_fail_before_any_work(options: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["flow", "--dry-run", *options])
    assert exc.value.code == 2


def test_child_environment_cannot_inherit_experiment_pins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_AR_CACHE", "1")
    monkeypatch.setenv("V710_TEST_BYPASS_MANIFEST", "1")
    monkeypatch.setenv("BSIMAR_CHECKPOINT_DIR", "/old/control")
    args = cli.parse_args(["evaluate", "--run-dir", str(tmp_path), "--dry-run"])
    env = cli.environment(args)
    assert "PYCIRCUITSIM_NN_AR_CACHE" not in env
    assert "V710_TEST_BYPASS_MANIFEST" not in env
    assert env["BSIMAR_CHECKPOINT_DIR"] == str(tmp_path / "checkpoints")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == env["PYCIRCUITSIM_TORCH_THREADS"] == "1"
    assert os.environ["BSIMAR_CHECKPOINT_DIR"] == "/old/control"


def test_failed_data_stage_never_starts_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "preflight", lambda args, env: None)

    def fail(name: str, command: list[str], env: dict[str, str], root: Path) -> None:
        calls.append(name)
        raise RuntimeError("generator failed")

    monkeypatch.setattr(cli, "run_command", fail)
    assert cli.main(["flow", "--run-dir", str(tmp_path / "results/run")]) == 2
    assert calls == ["data-tsmc5-nmos"]


def test_data_stage_refuses_to_replace_an_existing_dataset(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    existing = data / "tsmc5_dnf_nmos.npz"
    existing.write_bytes(b"preserved control")
    args = cli.parse_args(["data", "--run-dir", str(tmp_path), "--dry-run"])
    with pytest.raises(ValueError, match="already exists"):
        cli.check_outputs(args)
    assert existing.read_bytes() == b"preserved control"


def test_selected_gate_inventory_is_exact_and_does_not_change_defaults(
    tmp_path: Path,
) -> None:
    from scripts import v710_regate_jobs as jobs

    original = jobs.build_pools()
    assert jobs.main([
        str(tmp_path), "--tech", "TSMC5", "--tag", "tff",
        "--size", "small", "--omp", "1",
    ]) == 0
    for pool, lines in original.items():
        expected = [line for line in lines if line.split()[:3] == [
            "tff", "small", "TSMC5"] and line.split()[-1] == "1"]
        assert (tmp_path / f"jobs_{pool}.txt").read_text().splitlines() == expected
        assert expected
    assert jobs.build_pools() == original


def test_changed_gate_selection_cannot_replace_resume_inventory(tmp_path: Path) -> None:
    from scripts import v710_regate_jobs as jobs

    jobs.main([str(tmp_path)])
    original = (tmp_path / "jobs_clean.txt").read_bytes()
    with pytest.raises(SystemExit) as exc:
        jobs.main([str(tmp_path), "--tech", "TSMC5"])
    assert exc.value.code == 2
    assert (tmp_path / "jobs_clean.txt").read_bytes() == original


@pytest.mark.parametrize("selection", ("TSMC5,", "ASAP7", "TSMC5,TSMC5"))
def test_geometry_scope_rejects_a_smaller_accidental_denominator(selection: str) -> None:
    from tests.single_devices import verify_data_geometry_coverage as geometry

    with pytest.raises(SystemExit) as exc:
        geometry.main(["--tech", selection])
    assert exc.value.code == 2


def test_geometry_guard_checks_only_selected_technologies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from tests.single_devices import verify_data_geometry_coverage as geometry

    points = [geometry.EvalGeometry(tech, tech, "nmos", "lvt", 16e-9, 2, 300.15)
              for tech in ("tsmc5", "tsmc7")]
    monkeypatch.setattr(geometry, "_evaluation_geometries", lambda: points)
    monkeypatch.setattr(geometry, "_simple_circuit_geometries", lambda: [])
    monkeypatch.setattr(geometry, "_dataset_geometry", lambda *args: None)
    results = geometry.check(1.35, 2.0, tmp_path, techs=["TSMC5"])
    assert len(results) == 1
    assert results[0][0].startswith("tsmc5")
    assert results[0][1] is False  # Missing selected data still fails.
    assert len(geometry.check(1.35, 2.0, tmp_path)) == 2


def test_training_matrix_uses_shared_datasets_and_distinct_bundles(tmp_path: Path) -> None:
    args = cli.parse_args([
        "train", "--dry-run", "--run-dir", str(tmp_path),
        "--tech", *cli.TECHS, "--model", "direct", "transformer",
        "--size", *cli.SIZES, "--epochs", "2", "--batch-size", "32", "--gpus", "0",
    ])
    commands = cli.training_commands(args)
    assert len(commands) == len({stem for stem, _ in commands}) == 80
    data = {command[command.index("--data") + 1] for _, command in commands}
    assert len(data) == 10
    assert all("_dnf_" in path for path in data)
    for stem, command in commands:
        assert "--cuda" in command
        assert command[command.index("--epochs") + 1] == "2"
        assert command[command.index("--batch-size") + 1] == "32"
        assert ("_tff_" in stem) == ("transformer" in command)


def test_root_and_backends_accept_the_planned_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(["flow", "--run-dir", str(tmp_path), "--dry-run"])
    # Capture the exact launcher commands and ask each argparse backend for help.
    args.dry_run = False
    commands: list[list[str]] = []
    monkeypatch.setattr(cli, "run_command", lambda name, command, env, root: commands.append(command))
    monkeypatch.setattr(cli, "evaluation_verdict", lambda *args: 0)
    assert cli.execute(args, cli.environment(args)) == 0
    unique = {tuple(command[:4]) if "-m" in command else tuple(command[:2]): command
              for command in commands if "-c" not in command}
    for command in unique.values():
        result = subprocess.run([*command, "--help"], cwd=cli.ROOT,
                                env=cli.environment(args), capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("status", "rc", "expected"), (
    ("PASS", 0, 0), ("DIAGNOSTIC", 0, 0), ("FAIL", 1, 1), ("ERROR", 1, 1),
))
def test_scientific_verdicts_remain_distinct_from_execution_failures(
    tmp_path: Path, status: str, rc: int, expected: int,
) -> None:
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("dnf small TSMC5 verify_nn_ac 1\n")
    entry = {"result_complete": True, "status": status, "rc": rc}
    (tmp_path / "data.json").write_text(json.dumps({
        "dnf": {"small": {"verify_nn_ac": {"TSMC5": {"omp1": entry}}}},
    }))
    assert cli.evaluation_verdict(tmp_path, jobs) == expected
    # An absent requested cell can never become a successful partial report.
    jobs.write_text(jobs.read_text() + "dnf small TSMC7 verify_nn_ac 1\n")
    with pytest.raises(RuntimeError, match="incomplete evaluation"):
        cli.evaluation_verdict(tmp_path, jobs)


def test_child_failure_preserves_output_and_propagates(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exited 7"):
        cli.run_command("failure", [sys.executable, "-c",
                       "print('failure detail'); raise SystemExit(7)"], dict(os.environ), tmp_path)
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1
    assert "failure detail" in logs[0].read_text()


def test_preflight_requires_clean_source_before_expensive_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(["data", "--run-dir", str(tmp_path), "--dry-run"])
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *args, **kwargs: " M solver.py\n")
    with pytest.raises(ValueError, match="clean Git worktree"):
        cli.preflight(args, cli.environment(args))
    assert not (tmp_path / "data").exists()


def test_preflight_resolves_physical_gpus_and_pins_the_selected_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(["train", "--run-dir", str(tmp_path), "--dry-run", "--gpus", "0", "3"])
    calls: list[tuple[list[str], dict[str, str]]] = []

    def query(command: list[str], **kwargs: object) -> str:
        if command[0] == "git":
            return ""
        return "GPU-physical-" + command[1].split("=")[1]

    def child(command: list[str], *, env: dict[str, str], **kwargs: object) -> None:
        calls.append((command, env))

    monkeypatch.setattr(cli.subprocess, "check_output", query)
    monkeypatch.setattr(cli.subprocess, "run", child)
    monkeypatch.setattr(cli, "OSDI", tmp_path / "model.osdi")
    cli.OSDI.touch()
    env = cli.environment(args)
    cli.preflight(args, env)
    assert args.gpus == ["GPU-physical-0", "GPU-physical-3"]
    assert [child_env["CUDA_VISIBLE_DEVICES"] for _, child_env in calls] == [
        "", "", "GPU-physical-0", "GPU-physical-3"]
    assert all(command[0] == sys.executable for command, _ in calls)
    assert calls[1][0][-2:] == [str(path) for _, _, path in cli.datasets(args)]
    assert env["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("missing", ("osdi", "ngspice", "bundle", "none"))
def test_evaluation_preflight_fails_before_dispatch_when_inputs_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    args = cli.parse_args(["evaluate", "--run-dir", str(tmp_path), "--dry-run",
                          "--model", "transformer"])
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "OSDI", tmp_path / "model.osdi")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None if missing == "ngspice" else sys.executable)
    if missing != "osdi":
        cli.OSDI.touch()
    if missing == "none":
        args.checkpoint_dir.mkdir()
        for device in ("nmos", "pmos"):
            for suffix in ("_best.pt", "_norm.npz", "_best.pt.complete", "_config.npz"):
                (args.checkpoint_dir / f"tsmc5_tff_small_{device}{suffix}").touch()
        cli.preflight(args, cli.environment(args))
    else:
        with pytest.raises(ValueError):
            cli.preflight(args, cli.environment(args))
    assert not (tmp_path / "evaluation").exists()


def test_parallel_training_keeps_each_job_on_its_selected_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(["train", "--run-dir", str(tmp_path), "--dry-run", "--gpus", "0", "3"])
    args.dry_run = False
    seen: dict[str, str] = {}

    def child(name: str, command: list[str], env: dict[str, str], root: Path) -> None:
        seen[name] = env["CUDA_VISIBLE_DEVICES"]

    monkeypatch.setattr(cli, "run_command", child)
    assert cli.execute(args, cli.environment(args)) == 0
    assert seen == {"tsmc5_dnf_small_nmos": "0", "tsmc5_dnf_small_pmos": "3"}


@pytest.mark.parametrize("verdict", (0, 1))
def test_main_preserves_verdict_and_excludes_concurrent_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: int,
) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "preflight", lambda args, env: None)
    argv = ["evaluate", "--run-dir", "results/run"]

    def execute(args: object, env: dict[str, str]) -> int:
        assert cli.main(argv) == 2  # The outer workflow owns the run lock.
        return verdict

    monkeypatch.setattr(cli, "execute", execute)
    assert cli.main(argv) == verdict


def test_old_dataset_pin_is_validated_before_any_stage() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--dataset-source-commit", "typo", "--dry-run"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["flow", "--dataset-source-commit", "a" * 40, "--dry-run"])
    assert exc.value.code == 2
    args = cli.parse_args(["evaluate", "--dataset-source-commit", "a" * 40, "--dry-run"])
    assert cli.environment(args)["V710_DATASET_SOURCE_COMMIT"] == "a" * 40
