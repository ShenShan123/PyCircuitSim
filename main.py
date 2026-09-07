#!/usr/bin/env python3
"""PyCircuitSim: simulate netlists and run the NN compact-model workflow."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from external_compact_models.cli_options import (
    TRAINING_RECIPES, generation_parser, training_parser,
)
from scripts.v710_regate_jobs import evaluation_cases, select_cases

ROOT = Path(__file__).resolve().parent
TECHS = ("tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16")
MODELS = {"direct": ("dnf", 75), "transformer": ("tff", 76)}
SIZES = ("small", "medium", "large", "xl")
GENERATOR = "external_compact_models/bsim_cmg/scripts/generate_nn_data.py"
OSDI = ROOT / "external_compact_models/bsim_cmg/build/osdi/bsimcmg.osdi"


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_backend_options(
    parser: argparse.ArgumentParser, backend: argparse.ArgumentParser,
    prefix: str, *, aliases: bool, skip: set[str],
) -> None:
    """Reuse backend types/actions while keeping data and training inputs separate.

    Copy actions so argparse's internal registration never mutates the backend
    parser. Only explicitly supplied values override the workflow recipe.
    """
    group = parser.add_argument_group(f"{prefix} backend options")
    for original in backend._actions:
        if original.dest in skip or original.dest == "help":
            continue
        action = copy.copy(original)
        action.option_strings = [f"--{prefix}-{flag[2:]}" for flag in original.option_strings]
        if aliases:
            action.option_strings += [flag for flag in original.option_strings
                                      if flag not in parser._option_string_actions]
        action.dest = f"{prefix}_{original.dest}"
        action.default = argparse.SUPPRESS
        group._add_action(action)
        if isinstance(original, argparse._StoreTrueAction):
            group.add_argument(f"--no-{prefix}-{original.option_strings[0][2:]}",
                               dest=action.dest, action="store_false", default=argparse.SUPPRESS,
                               help=f"disable {original.option_strings[0]} for this stage")


def backend_values(args: argparse.Namespace, prefix: str) -> dict[str, object]:
    backend = generation_parser() if prefix == "data" else training_parser()
    return {action.dest: getattr(args, f"{prefix}_{action.dest}")
            for action in backend._actions if action.dest != "help"
            and hasattr(args, f"{prefix}_{action.dest}")}


def option_arguments(backend: argparse.ArgumentParser, values: dict[str, object]) -> list[str]:
    """Serialize typed options back to the unchanged backend CLI vocabulary."""
    result = []
    for action in backend._actions:
        value = values.get(action.dest)
        if value is None:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(action.option_strings[0])
        else:
            text = ",".join(str(item) for item in value) if isinstance(value, (tuple, list)) else str(value)
            result.extend((action.option_strings[0], text))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py simulate circuit.sp -o results/my-circuit
  python main.py circuit.sp -o results/my-circuit
  python main.py data --run-dir results/my-nn
  python main.py train --run-dir results/my-nn --gpus 0
  python main.py evaluate --run-dir results/my-nn
  python main.py flow --run-dir results/my-nn --gpus 0 --dry-run

Use <command> --help for stage-specific options.
""",
    )
    commands = parser.add_subparsers(dest="stage", required=True)
    simulation = commands.add_parser(
        "simulate", help="run a SPICE netlist", allow_abbrev=False,
    )
    simulation.add_argument("netlist", help="path to the HSPICE-format netlist file")
    simulation.add_argument("-o", "--output", dest="output_dir", default="results",
                            help="output directory for plots and results (default: results)")
    simulation.add_argument("-v", "--verbose", action="store_true",
                            help="enable verbose logging output")
    for stage, help_text in (
        ("data", "generate canonical six-surface datasets"),
        ("train", "train completed NMOS/PMOS checkpoint bundles"),
        ("evaluate", "run NGSPICE device and circuit gates and collect reports"),
        ("flow", "run data, training, then evaluation in order"),
    ):
        sub = commands.add_parser(stage, aliases=["all"] if stage == "flow" else [],
                                  help=help_text, description=help_text,
                                  allow_abbrev=False,
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        sub.set_defaults(stage=stage)
        sub.add_argument("--run-dir", type=Path, default=ROOT / "results/v773_nn",
                         help="isolated run directory; relative paths are repo-relative")
        sub.add_argument("--data-dir", type=Path, help="reuse a dataset directory")
        tech_choices = (*TECHS, "asap7", "all") if stage == "data" else (
            (*TECHS, "universal") if stage == "train" else TECHS)
        sub.add_argument("--tech", "--tech-scope", nargs="+", type=str.lower, choices=tech_choices,
                         default=["tsmc5"], help="space-separated technologies")
        sub.add_argument("--dry-run", action="store_true",
                         help="print commands and paths without executing or writing")
        if stage in ("data", "train", "flow"):
            sub.add_argument("--device", "--device-type", choices=("nmos", "pmos", "both"), default="both")
        if stage in ("data", "flow"):
            sub.add_argument("--workers", "--n-workers", type=positive, default=4,
                             help="generator workers per dataset")
            add_backend_options(sub, generation_parser(), "data", aliases=stage == "data",
                                skip={"device", "tech", "data_dir", "n_workers"})
        if stage in ("train", "evaluate", "flow"):
            sub.add_argument("--checkpoint-dir", type=Path, help="reuse a checkpoint directory")
            sub.add_argument("--model", nargs="+", choices=tuple(MODELS), default=["direct"])
            sub.add_argument("--size", nargs="+", choices=SIZES, default=["small"])
            sub.add_argument("--recipe", nargs="+", choices=tuple(TRAINING_RECIPES), default=["clean"])
        if stage in ("train", "flow"):
            sub.add_argument("--gpus", nargs="+", help="physical GPU IDs; one training job per GPU; default CPU")
            sub.add_argument("--epochs", "--train-epochs", type=positive, help="override the existing size preset")
            sub.add_argument("--batch-size", "--train-batch-size", type=positive, help="override the existing size preset")
            sub.add_argument("--seed", "--train-seed", type=int, default=None)
            add_backend_options(sub, training_parser(), "train", aliases=stage == "train",
                                skip={"model", "size", "device_type", "tech_scope", "epochs", "batch_size", "seed"})
        if stage in ("evaluate", "flow"):
            sub.add_argument("--pools", nargs="+", choices=("clean", "canary", "simple_v2"),
                             default=None)
            inventory = evaluation_cases()
            sub.add_argument("--evaluation-group", nargs="+", choices=("single_devices", "simple_circuits"))
            sub.add_argument("--device-suite", nargs="+", choices=tuple(
                name for name, case in inventory.items() if case["group"] == "single_devices"))
            sub.add_argument("--circuit", nargs="+", choices=tuple(
                name for name, case in inventory.items() if case["group"] == "simple_circuits"))
            sub.add_argument("--list-cases", action="store_true", help="list evaluation names without running")
            sub.add_argument("--parallel", type=positive, default=4, help="concurrent CPU gate jobs")
            sub.add_argument("--ngspice", default=os.environ.get(
                "NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice"))
            sub.set_defaults(dataset_source_commit=None)
        if stage == "evaluate":
            sub.add_argument("--dataset-source-commit",
                             help="explicit older dataset commit; existing equivalence checks still apply")
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original `main.py [-o output] netlist` invocation. Explicit
    # simulate also permits a netlist whose name matches a workflow command.
    if arguments and arguments[0] not in (
        "simulate", "data", "train", "evaluate", "flow", "all", "-h", "--help",
    ):
        arguments.insert(0, "simulate")
    args = parser.parse_args(arguments)
    if args.stage == "simulate":
        return args
    for name in ("tech", "model", "size", "pools", "gpus", "recipe", "evaluation_group", "device_suite", "circuit"):
        values = getattr(args, name, None)
        if values and len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates")
    if getattr(args, "gpus", None) and any(not gpu.isdigit() for gpu in args.gpus):
        parser.error("--gpus requires physical numeric GPU IDs")
    if args.stage == "flow" and args.device != "both":
        parser.error("flow requires --device both: circuit evaluation needs both polarities")
    if args.stage in ("data", "flow"):
        if "all" in args.tech:
            if len(args.tech) != 1:
                parser.error("--tech all cannot be combined with another technology")
            args.tech = ["asap7", *TECHS]
        data_options = backend_values(args, "data")
        if data_options.get("universal") and not any(
            argument.split("=", 1)[0] in ("--tech", "--tech-scope") for argument in arguments
        ):
            args.tech = list(TECHS)
        excluded = str(data_options.get("exclude_techs", ""))
        if excluded:
            names = [name.strip() for name in excluded.lower().split(",")]
            if len(set(names)) != len(names) or any(name not in (*TECHS, "asap7") for name in names):
                parser.error("--data-exclude-techs contains unknown, empty, or duplicate technologies")
            args.tech = [tech for tech in args.tech if tech not in names]
            if not args.tech:
                parser.error("technology exclusions leave no datasets to generate")
        version = str(data_options.get("version", "")).strip().strip("_")
        if version and not re.fullmatch(r"[A-Za-z0-9_]+", version):
            parser.error("--data-version must contain only letters, digits, and underscores")
        if args.stage == "flow" and any(data_options.get(name) for name in (
            "universal", "version", "allow_rejected_points",
        )):
            parser.error("flow evaluation requires canonical per-technology datasets; use data/train separately for universal, versioned, or diagnostic data")
    if args.stage in ("train", "flow"):
        train_options = backend_values(args, "train")
        if train_options.get("cuda") and not args.gpus:
            args.gpus = ["0"]
        if args.stage == "flow" and any(train_options.get(name) for name in ("data", "exp_name")):
            parser.error("flow owns dataset and checkpoint names; use standalone train for --train-data or --train-exp-name")
        if args.stage == "flow" and "corridor" in args.recipe:
            parser.error("corridor training requires prepared trajectory-overlay data; use standalone train with --data-dir")
        if train_options.get("data") and (len(args.tech) != 1 or args.device == "both"):
            parser.error("--train-data requires one technology and one polarity")
        if train_options.get("exp_name") and (len(args.tech) * len(args.model) * len(args.size) * len(args.recipe) != 1):
            parser.error("--train-exp-name requires one technology/model/size/recipe")
        if "direct" in args.model and (
            any(recipe in ("sub", "ar3", "ar3roll") for recipe in args.recipe)
            or any(train_options.get(name) for name in (
                "subthresh", "autoregressive_training", "full_terminal_ar_targets"))
            or any(name.startswith(("subthresh_", "lam_subthresh")) for name in train_options)
        ):
            parser.error("subthreshold and autoregressive strategies require --model transformer")
        for name in ("data", "init_from"):
            if train_options.get(name) and (name == "data" or "/" in str(train_options[name])):
                setattr(args, f"train_{name}", str((ROOT / str(train_options[name])).resolve()))
    if args.stage in ("evaluate", "flow"):
        requested = [*(args.device_suite or []), *(args.circuit or [])]
        defaults = ["clean", "canary"] if not requested and not args.evaluation_group and not args.list_cases else None
        try:
            args.pools, args.selected_cases = select_cases(args.pools or defaults, args.evaluation_group, requested)
        except ValueError as exc:
            parser.error(str(exc))
    source = getattr(args, "dataset_source_commit", None)
    if source is not None and not re.fullmatch(r"[0-9a-f]{40}", source):
        parser.error("--dataset-source-commit requires a full lowercase 40-character Git SHA")
    if hasattr(args, "ngspice") and "/" in args.ngspice:
        args.ngspice = str((ROOT / args.ngspice).resolve())
    args.run_dir = (ROOT / args.run_dir).resolve()
    args.data_dir = (ROOT / args.data_dir).resolve() if args.data_dir else args.run_dir / "data"
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    args.checkpoint_dir = ((ROOT / checkpoint_dir).resolve() if checkpoint_dir
                           else args.run_dir / "checkpoints")
    if not args.run_dir.is_relative_to(ROOT / "results") and not args.dry_run:
        parser.error("--run-dir must be under the repository results/ directory")
    return args


def environment(args: argparse.Namespace) -> dict[str, str]:
    """Prevent inherited experiment knobs from selecting a different problem."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYCIRCUITSIM_", "BSIMAR_", "V710_", "NN_"))}
    env.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "external_compact_models"))),
        "BSIMAR_DATA_DIR": str(args.data_dir),
        "BSIMAR_CHECKPOINT_DIR": str(args.checkpoint_dir),
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYCIRCUITSIM_TORCH_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
    })
    if args.stage in ("evaluate", "flow"):
        env["NGSPICE_BIN"] = args.ngspice
        if args.dataset_source_commit:
            env["V710_DATASET_SOURCE_COMMIT"] = args.dataset_source_commit
    return env


def datasets(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    device = getattr(args, "device", "both")
    devices = ("nmos", "pmos") if device == "both" else (device,)
    scopes = ["universal"] if getattr(args, "data_universal", False) else args.tech
    version = str(getattr(args, "data_version", "")).strip().strip("_")
    version_part = f"_{version}" if version else ""
    explicit = getattr(args, "train_data", None)
    return [(tech, dev, Path(explicit) if explicit else args.data_dir / f"{tech}{version_part}_dnf_{dev}.npz")
            for tech in scopes for dev in devices]


def generation_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    values = {"n_workers": args.workers, "enable_inv_trip": True,
              "enable_subvt_off": True, "allow_safety_rejections": True,
              **backend_values(args, "data")}
    if values.get("universal"):
        values["exclude_techs"] = ",".join(tech for tech in ("asap7", *TECHS) if tech not in args.tech)
    commands = []
    for tech, device, _ in datasets(args):
        selected = {**values, "device": device, "data_dir": args.data_dir}
        if tech != "universal":
            selected["tech"] = tech
        commands.append((f"data-{tech}-{device}", [sys.executable, "-u", GENERATOR,
                                                   *option_arguments(generation_parser(), selected)]))
    return commands


def checkpoint_variant(recipe: str, size: str) -> str:
    return size if recipe == "clean" else f"{recipe}_{size}"


def training_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands = []
    for model in args.model:
        for size in args.size:
            for recipe in args.recipe:
                for tech, device, data in datasets(args):
                    scope = "refac" if tech == "universal" else tech
                    prefix = f"{scope}_{MODELS[model][0]}_{checkpoint_variant(recipe, size)}"
                    if recipe == "corridor" and not getattr(args, "train_data", None):
                        data = args.data_dir / f"{tech}_dnf_corridor_{device}.npz"
                    values = {"model": model, "size": size, "device_type": device,
                              "tech_scope": tech, "data": str(data), "exp_name": prefix,
                              "swa_mode": "ema", "seed": 42,
                              **TRAINING_RECIPES[recipe], **backend_values(args, "train")}
                    if tech == "universal":
                        excluded = {name.strip().lower() for name in str(values.get("exclude_techs", "")).split(",") if name.strip()}
                        values["exclude_techs"] = ",".join(sorted(excluded | {"asap7"}))
                    for option in ("epochs", "batch_size", "seed"):
                        value = getattr(args, option)
                        if value is not None:
                            values[option] = value
                    if args.gpus:
                        values["cuda"] = True
                    stem = f"{values['exp_name']}_{device}"
                    command = [sys.executable, "-u", "-m", "neural_network.cli.train",
                               *option_arguments(training_parser(), values)]
                    commands.append((stem, command))
    return commands


def check_outputs(args: argparse.Namespace) -> None:
    """Catch accidental replacement before starting any expensive stage."""
    paths = []
    if args.stage in ("data", "flow"):
        for _, _, path in datasets(args):
            paths.extend((path, path.with_suffix(".npz.complete")))
            if getattr(args, "data_finetune_size", 0):
                fine = path.with_name(f"finetune_{path.name}")
                paths.extend((fine, fine.with_suffix(".npz.complete")))
    if args.stage in ("train", "flow") and not getattr(args, "train_overwrite", False):
        for stem, _ in training_commands(args):
            paths.extend(args.checkpoint_dir / f"{stem}{suffix}" for suffix in (
                "_best.pt", "_norm.npz", "_config.npz", "_best.pt.complete"))
    for path in paths:
        if path.exists():
            raise ValueError(f"artifact already exists: {path}; choose a new output directory")


def preflight(args: argparse.Namespace, env: dict[str, str]) -> None:
    check_outputs(args)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("canonical runs require a clean Git worktree; commit your changes first (or use --dry-run)")
    code = "import numpy, torch, scipy, matplotlib, sklearn"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)
    if not OSDI.is_file():
        raise ValueError(f"OSDI model not built: {OSDI}; see README.md setup")
    if args.stage in ("evaluate", "flow"):
        ngspice = shutil.which(args.ngspice)
        if ngspice is None:
            raise ValueError(f"NGSPICE executable not found: {args.ngspice}")
        args.ngspice = str(Path(ngspice).resolve())
        env["NGSPICE_BIN"] = args.ngspice
    if args.stage == "evaluate":
        for model in args.model:
            for size in [checkpoint_variant(recipe, size) for recipe in args.recipe for size in args.size]:
                for tech, device, _ in datasets(args):
                    stem = f"{tech}_{MODELS[model][0]}_{size}_{device}"
                    suffixes = ["_best.pt", "_norm.npz", "_best.pt.complete"]
                    if model == "transformer":
                        suffixes.append("_config.npz")
                    for suffix in suffixes:
                        path = args.checkpoint_dir / f"{stem}{suffix}"
                        if not path.is_file():
                            raise ValueError(f"incomplete checkpoint bundle: {path}")
    if args.stage in ("train", "evaluate"):
        code = ("import sys; from neural_network.data.dataset import validate_canonical_dataset; "
                "[validate_canonical_dataset(path) for path in sys.argv[1:]]")
        paths = ([command[command.index("--data") + 1] for _, command in training_commands(args)]
                 if args.stage == "train" else [str(p) for _, _, p in datasets(args)])
        subprocess.run([sys.executable, "-c", code, *dict.fromkeys(paths)],
                       cwd=ROOT, env=env, check=True)
    if getattr(args, "gpus", None):
        uuids = []
        for gpu in args.gpus:
            uuid = subprocess.check_output([
                "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader",
            ], text=True).strip()
            if not uuid.startswith("GPU-") or "\n" in uuid:
                raise ValueError(f"cannot identify physical GPU {gpu}")
            if uuid in uuids:
                raise ValueError(f"physical GPU {gpu} was selected more than once")
            gpu_env = {**env, "CUDA_VISIBLE_DEVICES": uuid}
            subprocess.run([sys.executable, "-c",
                            "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"],
                           env=gpu_env, check=True)
            uuids.append(uuid)
        args.gpus = uuids


def run_command(name: str, command: list[str], env: dict[str, str], root: Path) -> None:
    """Keep each attempt's output and exact command, including failed attempts."""
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.{time.time_ns()}.log"
    print(f"START {name}; log: {log}", flush=True)
    with log.open("w") as output:
        output.write(shlex.join(command) + "\n")
        recorded_env = {key: value for key, value in env.items()
                        if key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                   "PYTHONPATH", "NGSPICE_BIN", "PYCIRCUITSIM_TORCH_THREADS")
                        or key.startswith(("BSIMAR_", "V710_"))}
        output.write("Environment: " + json.dumps(recorded_env, sort_keys=True) + "\n")
        output.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see {log}")
    print(f"DONE {name}", flush=True)


def execute(args: argparse.Namespace, env: dict[str, str]) -> int:
    def run(name: str, command: list[str], extra: dict[str, str] | None = None) -> None:
        child_env = {**env, **(extra or {})}
        if args.dry_run:
            print(f"[{name}] {shlex.join(command)}")
            if extra:
                print("  " + shlex.join(f"{key}={value}" for key, value in extra.items()))
        else:
            run_command(name, command, child_env, args.run_dir)

    if args.stage in ("data", "flow"):
        for (name, command), (tech, device, path) in zip(generation_commands(args), datasets(args)):
            run(name, command)
            if getattr(args, "data_allow_rejected_points", False):
                print(f"Diagnostic dataset: {path}; excluded from training and scoring.")
                continue
            code = ("import sys; from neural_network.data.dataset import validate_canonical_dataset; "
                    "from neural_network.eval.loo_labels import get_or_build_tech_variant_labels; "
                    "validate_canonical_dataset(sys.argv[1]); "
                    "get_or_build_tech_variant_labels(sys.argv[1], sys.argv[2])")
            run(f"labels-{tech}-{device}", [sys.executable, "-c", code, str(path), device])
    if args.stage in ("train", "flow"):
        commands = training_commands(args)
        gpus = args.gpus or [""]

        def worker(index: int, gpu: str) -> None:
            for stem, command in commands[index::len(gpus)]:
                run(stem, command, {"CUDA_VISIBLE_DEVICES": gpu})

        if len(gpus) == 1 or args.dry_run:
            for index, gpu in enumerate(gpus):
                worker(index, gpu)
        else:
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [pool.submit(worker, index, gpu) for index, gpu in enumerate(gpus)]
                for future in futures:
                    future.result()
    verdict = 0
    if args.stage in ("evaluate", "flow"):
        run("geometry-coverage", [sys.executable,
            "tests/single_devices/verify_data_geometry_coverage.py",
            "--data-dir", str(args.data_dir), "--tech", ",".join(t.upper() for t in args.tech),
            "--case", *args.selected_cases])
        job_lists = args.run_dir / "evaluation/job_lists"
        run("gate-jobs", [sys.executable, "scripts/v710_regate_jobs.py", str(job_lists),
            "--tech", *[tech.upper() for tech in args.tech],
            "--tag", *[MODELS[model][0] for model in args.model],
            "--size", *args.size, "--omp", "1", "--recipe", *args.recipe,
            "--pools", *args.pools, "--case", *args.selected_cases])
        for pool in args.pools:
            output = args.run_dir / "evaluation" / pool
            run(f"evaluate-{pool}", ["bash", "scripts/v710_regate.sh"], {
                "NN_PY": sys.executable, "PAR": str(args.parallel),
                "JOBS": str(job_lists / f"jobs_{pool}.txt"),
                "V710_OUT": str(output), "V710_SCRATCH": str(output / "artifacts"),
                "V710_MANIFEST": str(output / "campaign_manifest.json"),
            })
            run(f"collect-{pool}", [sys.executable, "scripts/v710_regate_collect.py",
                                    "--root", str(output), "--require-manifest"])
            if not args.dry_run:
                verdict = max(verdict, evaluation_verdict(output, job_lists / f"jobs_{pool}.txt"))
                print(f"Report: {output / 'REPORT.md'}", flush=True)
    return verdict


def evaluation_verdict(output: Path, jobs: Path) -> int:
    """A report is complete only when every requested cell has valid evidence."""
    data = json.loads((output / "data.json").read_text())
    verdict = 0
    lines = jobs.read_text().splitlines()
    if not lines:
        raise RuntimeError(f"empty evaluation inventory: {jobs}")
    for line in lines:
        tag, size, tech, suite, omp = line.split()
        entry = data.get(tag, {}).get(size, {}).get(suite, {}).get(tech, {}).get(f"omp{omp}", {})
        if not entry.get("result_complete"):
            raise RuntimeError(f"incomplete evaluation evidence for {line}: {output / 'REPORT.md'}")
        if entry.get("status") in ("FAIL", "ERROR") or entry.get("rc") != 0:
            verdict = 1
    return verdict


def simulate(args: argparse.Namespace) -> int:
    """Run a netlist with the existing simulation interface."""
    # Import here to avoid errors if package not installed
    try:
        from pycircuitsim.simulation import run_simulation
    except ImportError:
        print("Error: PyCircuitSim package not found.", file=sys.stderr)
        print("Install the dependencies described in README.md.", file=sys.stderr)
        return 1

    # Run simulation
    try:
        run_simulation(
            netlist_path=args.netlist,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "simulate":
        return simulate(args)
    if getattr(args, "list_cases", False):
        inventory = evaluation_cases()
        for name in args.selected_cases:
            case = inventory[name]
            print(f"{case['group']:16s} {name:32s} {case['pool']:10s} {case['role']:14s} {case['description']}")
        return 0
    env = environment(args)
    print(f"NN workflow: {args.stage}; technologies: {', '.join(args.tech)}")
    if hasattr(args, "model"):
        print("Models: " + ", ".join(f"{m} (LEVEL={MODELS[m][1]})" for m in args.model))
        print("Recipes: " + ", ".join(args.recipe))
    print(f"Run: {args.run_dir}\nData: {args.data_dir}\nCheckpoints: {args.checkpoint_dir}")
    try:
        if args.dry_run:
            return execute(args, env)
        preflight(args, env)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / "workflow.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another workflow is active in {args.run_dir}") from exc
            check_outputs(args)
            verdict = execute(args, env)
        print("Evaluation contains failed/error gates." if verdict else "Requested stages completed.")
        return verdict
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; inspect logs before restarting.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
