#!/usr/bin/env python3
"""Fit one replay-anchored PMOS pass-device current-row correction.

The experiment is deliberately narrow: exact NGSPICE LEVEL=72 states from
the two ``ldo_1`` line-regulation decks supervise only the final linear row
that emits normalized ``i_d``.  Every hidden layer and every other output row
remain byte-identical to the parent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "external_compact_models"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.complex_circuits.pycircuitsim_bench import (  # noqa: E402
    run_compare,
    translate,
)
from examples.complex_circuits.pycircuitsim_bench.evaluator_probe import (  # noqa: E402
    align_devices,
)
from neural_network.config import TECH_CONFIGS  # noqa: E402
from neural_network.data.normalize import normalizer_from_stats  # noqa: E402
from pycircuitsim.models.mosfet_directnet_full import (  # noqa: E402
    _load_artifacts,
)
from pycircuitsim.solver import _is_pmos  # noqa: E402
from scripts.finetune_active_hermite import (  # noqa: E402
    _load_data,
    _model_metrics,
    _normalize_inputs,
    _normalize_outputs,
    _normalization_path,
    _scored_runtime_contract,
    evaluate_feasibility,
)
from scripts.generate_active_hermite_overlay import (  # noqa: E402
    OUTPUT_COLUMNS,
    _evaluate,
    sha256_file,
)


TRAIN_DECKS: tuple[str, ...] = ("tb_line_max.cir", "tb_line_min.cir")
TARGET_INSTANCE = "m.x1.mm8"
FINAL_WEIGHT = "net.12.weight"
FINAL_BIAS = "net.12.bias"
EXPECTED_CENTERS = 56
EXPECTED_SWEEPS = {
    "up": ("v1", 0.65, 0.715, 0.005, 14),
    "dn": ("v1", 0.65, 0.585, -0.005, 14),
}


def checkpoint_override_prefix(path: Path, device: str) -> str:
    """Return the parser's save-prefix form for one explicit checkpoint."""
    suffix = f"_{device}_best.pt"
    value = str(path.resolve())
    if not value.endswith(suffix):
        raise ValueError(f"checkpoint must end in {suffix!r}: {path}")
    return value.removesuffix("_best.pt")


def validate_line_sweep(plan: Any) -> int:
    """Validate and return one fixed line-regulation plan denominator."""
    expected = EXPECTED_SWEEPS.get(str(plan.label))
    if expected is None:
        raise ValueError(f"unexpected line-regulation plan {plan.label!r}")
    source, start, stop, step, points = expected
    actual = (str(plan.source).lower(), plan.start, plan.stop, plan.step)
    wanted = (source, start, stop, step)
    if plan.kind != "dc_source" or actual != wanted:
        raise ValueError(
            f"line-regulation plan {plan.label!r} changed: {actual!r}, "
            f"expected {wanted!r}")
    return points


def aligned_sweep_state(circuit: Any, sweep: Any, index: int) -> dict[str, float]:
    """Map one complete NGSPICE raw point onto native circuit node names."""
    table: dict[str, str] = {}
    for node in circuit.get_nodes():
        folded = str(node).lower()
        if folded in table and table[folded] != node:
            raise ValueError(
                f"circuit nodes {table[folded]!r}/{node!r} differ by case")
        table[folded] = str(node)
    supplied = {
        str(name).lower(): values
        for name, values in sweep.v.items()
        if "#" not in str(name)
    }
    missing = sorted(set(table) - set(supplied))
    extra = sorted(set(supplied) - set(table))
    if missing or extra:
        raise ValueError(
            f"NGSPICE state does not match circuit nodes: "
            f"missing={missing}, unknown={extra}")
    state: dict[str, float] = {}
    for folded, native in table.items():
        values = np.asarray(supplied[folded])
        if index < 0 or index >= len(values):
            raise IndexError(f"raw point {index} is outside node {folded!r}")
        value = float(np.real(values[index]))
        if not math.isfinite(value):
            raise ValueError(f"raw point {index} node {folded!r} is not finite")
        state[native] = value
    state["0"] = 0.0
    state["GND"] = 0.0
    return state


def solve_output_row_delta(
    local_features: np.ndarray,
    local_residual: np.ndarray,
    replay_features: np.ndarray,
    *,
    local_weight: float,
    anchor_weight: float,
    ridge_ratio: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Solve the predeclared float64 ridge correction for one output row."""
    local = np.asarray(local_features, dtype=np.float64)
    residual = np.asarray(local_residual, dtype=np.float64)
    replay = np.asarray(replay_features, dtype=np.float64)
    if local.ndim != 2 or replay.ndim != 2 or local.shape[1] != replay.shape[1]:
        raise ValueError("local and replay features must be aligned matrices")
    if residual.shape != (len(local),):
        raise ValueError("local residual must have one value per local row")
    if not all(np.all(np.isfinite(value)) for value in (local, residual, replay)):
        raise ValueError("ridge inputs contain NaN/Inf")
    if not len(local) or not len(replay):
        raise ValueError("ridge inputs must be non-empty")
    if local_weight < 0.0 or anchor_weight <= 0.0 or ridge_ratio < 0.0:
        raise ValueError("ridge weights are outside their declared domain")
    width = local.shape[1]
    local_normal = local.T @ local / len(local)
    replay_normal = replay.T @ replay / len(replay)
    normal = local_weight * local_normal + anchor_weight * replay_normal
    mean_diagonal = float(np.trace(normal) / width)
    ridge = ridge_ratio * mean_diagonal
    diagnostics: dict[str, float | bool] = {
        "local_weight": float(local_weight),
        "anchor_weight": float(anchor_weight),
        "ridge_ratio": float(ridge_ratio),
        "mean_diagonal": mean_diagonal,
        "ridge": ridge,
        "control_exact": local_weight == 0.0,
    }
    if local_weight == 0.0:
        return np.zeros(width, dtype=np.float64), diagnostics
    lhs = normal + ridge * np.eye(width, dtype=np.float64)
    rhs = local_weight * (local.T @ residual / len(local))
    delta = np.linalg.solve(lhs, rhs)
    if not np.all(np.isfinite(delta)):
        raise RuntimeError("ridge solve produced NaN/Inf")
    diagnostics["condition_number"] = float(np.linalg.cond(lhs))
    diagnostics["delta_l2"] = float(np.linalg.norm(delta))
    return delta, diagnostics


def minimum_local_scale(
    parent_prediction: np.ndarray,
    target: np.ndarray,
    full_effect: np.ndarray,
    *,
    target_ratio: float,
) -> float:
    """Return the smallest row scale meeting a local-training MAE target."""
    parent = np.asarray(parent_prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    effect = np.asarray(full_effect, dtype=np.float64)
    if parent.shape != truth.shape or parent.shape != effect.shape:
        raise ValueError("local scale arrays must align")
    if not 0.0 < target_ratio < 1.0:
        raise ValueError("local target ratio must be strictly between zero and one")
    baseline = float(np.mean(np.abs(truth - parent)))
    if baseline == 0.0:
        return 0.0

    def ratio(scale: float) -> float:
        return float(np.mean(np.abs(
            truth - (parent + scale * effect))) / baseline)

    grid = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    upper_index = next(
        (index for index in range(1, len(grid))
         if ratio(float(grid[index])) <= target_ratio),
        None,
    )
    if upper_index is None:
        raise ValueError("full row delta does not reach the local MAE target")
    lower = float(grid[upper_index - 1])
    upper = float(grid[upper_index])
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if ratio(middle) <= target_ratio:
            upper = middle
        else:
            lower = middle
    return upper


def apply_output_row_delta(
    state: dict[str, torch.Tensor],
    weight_key: str,
    bias_key: str,
    *,
    row: int,
    delta: np.ndarray,
) -> None:
    """Apply an augmented ``[weight..., bias]`` delta to one output row."""
    if weight_key not in state or bias_key not in state:
        raise KeyError("checkpoint lacks the declared final linear layer")
    weight = state[weight_key]
    bias = state[bias_key]
    values = np.asarray(delta, dtype=np.float64)
    if values.shape != (weight.shape[1] + 1,):
        raise ValueError("row delta width does not match the output layer")
    if row < 0 or row >= weight.shape[0] or row >= bias.shape[0]:
        raise IndexError("output row is outside the final layer")
    weight[row] += torch.as_tensor(
        values[:-1], dtype=weight.dtype, device=weight.device)
    bias[row] += torch.as_tensor(
        values[-1], dtype=bias.dtype, device=bias.device)


def candidate_gate(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    local_parent_mae: float,
    local_candidate_mae: float,
    local_ratio: float = 0.5,
    value_mae_ratio: float = 1.02,
    value_max_ratio: float = 1.05,
    jacobian_ratio: float = 1.05,
) -> dict[str, object]:
    """Apply the local-improvement and unchanged global safety gates."""
    value_gate = evaluate_feasibility(
        dict(baseline), dict(candidate),
        require_jacobian_improvement=False,
        value_mae_ratio=value_mae_ratio,
        value_max_ratio=value_max_ratio,
    )
    failures = list(value_gate["failures"])
    local_observed = (
        local_candidate_mae / local_parent_mae
        if local_parent_mae != 0.0
        else (1.0 if local_candidate_mae == 0.0 else float("inf"))
    )
    baseline_j = float(baseline["current_jacobian_mae"])
    candidate_j = float(candidate["current_jacobian_mae"])
    jacobian_observed = (
        candidate_j / baseline_j
        if baseline_j != 0.0
        else (1.0 if candidate_j == 0.0 else float("inf"))
    )
    tolerance = 1e-12
    if local_observed > local_ratio + tolerance:
        failures.append("local_i_d_mae")
    if jacobian_observed > jacobian_ratio + tolerance:
        failures.append("current_jacobian_mae")
    return {
        "eligible": not failures,
        "failures": failures,
        "ratios": {
            **dict(value_gate["ratios"]),
            "local_i_d_mae": float(local_observed),
            "current_jacobian_mae": float(jacobian_observed),
        },
        "limits": {
            "local_i_d_mae": local_ratio,
            "value_normalized_mae": value_mae_ratio,
            "value_physical_max_abs": value_max_ratio,
            "current_jacobian_mae": jacobian_ratio,
        },
    }


def _source_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return commit, bool(dirty)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object {path}")
    return value


def _hidden_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    codes: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Return augmented final-layer inputs in deterministic evaluation mode."""
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            embedding = model.tech_embedding(codes[start:stop])
            hidden = model.net[:-1](torch.cat((x[start:stop], embedding), dim=1))
            augmented = torch.cat(
                (hidden, torch.ones((len(hidden), 1), device=hidden.device)),
                dim=1,
            )
            rows.append(augmented.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def _raw_inputs(
    inputs: np.ndarray,
    nfin: np.ndarray,
    length: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    return np.column_stack((
        inputs[:, 0], inputs[:, 1], np.zeros(len(inputs)), inputs[:, 3],
        np.log2(np.maximum(nfin, 1.0)), length, temperature,
    )).astype(np.float64)


def _assert_support(raw: np.ndarray, stats: Any) -> None:
    lower = np.asarray(stats.input_min, dtype=np.float64)
    upper = np.asarray(stats.input_max, dtype=np.float64)
    outside = np.argwhere((raw < lower[None, :]) | (raw > upper[None, :]))
    if len(outside):
        row, column = (int(value) for value in outside[0])
        raise ValueError(
            f"LDO center row {row} input {column}={raw[row, column]} outside "
            f"parent support [{lower[column]}, {upper[column]}]")


def _harvest_centers(
    args: argparse.Namespace,
    stats: Any,
    source_commit: str,
    scored_runtime: Mapping[str, object],
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    """Harvest exact per-unit OSDI labels from the two training line decks."""
    run_compare._pin_design_tree(args.bench_root, "tsmc5")
    os.environ["PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS"] = (
        checkpoint_override_prefix(args.parent_nmos, "nmos"))
    os.environ["PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS"] = (
        checkpoint_override_prefix(args.parent_pmos, "pmos"))
    os.environ["PYCIRCUITSIM_NN_STRICT_TECH_CODE"] = "1"
    design_dir = (
        args.bench_root / "designs_tsmc5" / "ldo" / "ldo_1"
    )
    rows: dict[str, list[Any]] = {
        name: [] for name in (
            "inputs", "outputs", "jacobians", "tech_codes", "length",
            "nfin", "temperature", "instance_multiplier", "deck",
            "plan_label", "raw_point_index", "sweep_value", "state_id",
        )
    }
    provenance: list[dict[str, Any]] = []
    state_id = 0
    reference_seconds = 0.0
    reference_inputs: dict[str, Any] | None = None
    for deck in TRAIN_DECKS:
        td72 = translate.translate_deck(
            design_dir, deck, tech="tsmc5", category="ldo", model_level=72,
        )
        td75 = translate.translate_deck(
            design_dir, deck, tech="tsmc5", category="ldo", model_level=75,
        )
        if len(td72.plans) != len(EXPECTED_SWEEPS):
            raise ValueError(f"{deck} must contain exactly two fixed sweeps")
        expected_points = sum(validate_line_sweep(plan) for plan in td72.plans)
        if expected_points != EXPECTED_CENTERS // len(TRAIN_DECKS):
            raise ValueError(f"{deck} sweep denominator changed")
        current_reference = run_compare._reference_provenance(td72)
        current_inputs = {
            "osdi": current_reference["osdi"],
            "ngspice": current_reference["ngspice"],
        }
        if reference_inputs is None:
            reference_inputs = current_inputs
        elif current_inputs != reference_inputs:
            raise ValueError("LEVEL=72 executable provenance changed between decks")
        deck_work = args.work_dir / Path(deck).stem
        circuit72, netlist72 = run_compare.build_circuit(td72, deck_work / "l72")
        circuit75, netlist75 = run_compare.build_circuit(td75, deck_work / "l75")
        pairs = align_devices(
            run_compare._mosfets(circuit72), run_compare._mosfets(circuit75),
        )
        matches = [pair for pair in pairs
                   if str(pair[0].name).lower() == args.target_instance.lower()]
        if len(matches) != 1:
            raise ValueError(
                f"{deck} has {len(matches)} matches for {args.target_instance!r}")
        osdi_device, nn_device = matches[0]
        if not _is_pmos(osdi_device):
            raise ValueError("declared LDO pass instance is not PMOS")
        code = int(nn_device._tech_code)
        variants = tuple(TECH_CONFIGS["tsmc5"].variant_names)
        if not 0 <= code < len(variants):
            raise ValueError(f"pass-device local code {code} is not a TSMC5 VT")
        for plan_index, plan in enumerate(td72.plans):
            sweep, _own, seconds = run_compare.ngspice_sweep(
                td72, plan, deck_work / f"reference_{plan_index}",
                args.ng_timeout,
            )
            reference_seconds += seconds
            rawfile = Path(str(sweep.meta["rawfile"]))
            for point_index, sweep_value in enumerate(np.asarray(sweep.x)):
                state = aligned_sweep_state(circuit72, sweep, point_index)
                drain, gate, source, bulk = osdi_device.nodes
                source_value = state[source]
                point = np.asarray([
                    state[drain] - source_value,
                    state[gate] - source_value,
                    0.0,
                    state[bulk] - source_value,
                ], dtype=np.float64)
                values, jacobian = _evaluate(osdi_device._pycmg_instance, point)
                rows["inputs"].append(point)
                rows["outputs"].append(values)
                rows["jacobians"].append(jacobian)
                rows["tech_codes"].append(code)
                rows["length"].append(float(osdi_device.L))
                rows["nfin"].append(float(osdi_device.NFIN))
                rows["temperature"].append(float(osdi_device.temperature))
                rows["instance_multiplier"].append(float(osdi_device.m))
                rows["deck"].append(deck)
                rows["plan_label"].append(plan.label or f"plan-{plan_index}")
                rows["raw_point_index"].append(point_index)
                rows["sweep_value"].append(float(sweep_value))
                rows["state_id"].append(state_id)
                state_id += 1
            provenance.append({
                "deck": deck,
                "plan_index": plan_index,
                "plan_label": plan.label,
                "source_deck": {
                    "path": str((design_dir / deck).resolve()),
                    "sha256": sha256_file(design_dir / deck),
                },
                "translated_level72": {
                    "path": str(netlist72.resolve()),
                    "sha256": sha256_file(netlist72),
                },
                "translated_level75": {
                    "path": str(netlist75.resolve()),
                    "sha256": sha256_file(netlist75),
                },
                "rawfile": {
                    "path": str(rawfile.resolve()),
                    "sha256": sha256_file(rawfile),
                },
                "modelcard": {
                    "path": str(td72.modelcard_path.resolve()),
                    "sha256": sha256_file(td72.modelcard_path),
                },
                "points": int(len(sweep.x)),
            })
    if state_id != EXPECTED_CENTERS:
        raise ValueError(
            f"expected {EXPECTED_CENTERS} centers, harvested {state_id}")
    if reference_inputs is None:
        raise RuntimeError("no LEVEL=72 reference provenance was captured")
    arrays = {
        "inputs": np.asarray(rows["inputs"], dtype=np.float64),
        "outputs": np.asarray(rows["outputs"], dtype=np.float64),
        "jacobians": np.asarray(rows["jacobians"], dtype=np.float64),
        "tech_codes": np.asarray(rows["tech_codes"], dtype=np.int64),
        "length": np.asarray(rows["length"], dtype=np.float64),
        "nfin": np.asarray(rows["nfin"], dtype=np.float64),
        "temperature": np.asarray(rows["temperature"], dtype=np.float64),
        "instance_multiplier": np.asarray(
            rows["instance_multiplier"], dtype=np.float64),
        "deck": np.asarray(rows["deck"]),
        "plan_label": np.asarray(rows["plan_label"]),
        "raw_point_index": np.asarray(rows["raw_point_index"], dtype=np.int64),
        "sweep_value": np.asarray(rows["sweep_value"], dtype=np.float64),
        "state_id": np.asarray(rows["state_id"], dtype=np.int64),
    }
    raw = _raw_inputs(
        arrays["inputs"], arrays["nfin"], arrays["length"],
        arrays["temperature"],
    )
    _assert_support(raw, stats)
    if not all(np.all(np.isfinite(value)) for name, value in arrays.items()
               if name not in ("deck", "plan_label")):
        raise RuntimeError("LDO center artifact contains NaN/Inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "ldo_1_pass_pmos_centers.npz"
    marker_path = artifact.with_suffix(artifact.suffix + ".complete")
    if (artifact.exists() or marker_path.exists()) and not args.overwrite:
        raise FileExistsError(artifact)
    np.savez(
        artifact, **arrays,
        meta_output_columns=np.asarray(OUTPUT_COLUMNS),
        meta_jacobian_voltage_columns=np.asarray(("Vd", "Vg", "Vb")),
        meta_tech=np.asarray("tsmc5"),
        meta_design=np.asarray("ldo_1"),
        meta_target_instance=np.asarray(args.target_instance.lower()),
        meta_source_commit=np.asarray(source_commit),
        meta_source_dirty=np.asarray(False),
        meta_provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    marker = {
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "source_commit": source_commit,
        "source_dirty": False,
        "ground_truth": "NGSPICE LEVEL=72 identical BSIM-CMG OSDI",
        "tech": "tsmc5",
        "design": "ldo_1",
        "decks": list(TRAIN_DECKS),
        "excluded_designs": ["ldo_2"],
        "excluded_decks": ["tb_load.cir"],
        "target_instance": args.target_instance.lower(),
        "centers": state_id,
        "reference_seconds": reference_seconds,
        "parent_nmos_sha256": sha256_file(args.parent_nmos),
        "parent_pmos_sha256": sha256_file(args.parent_pmos),
        "replay_data_sha256": sha256_file(args.replay_data),
        "hermite_overlay_sha256": sha256_file(args.hermite_overlay),
        "osdi": reference_inputs["osdi"],
        "ngspice": reference_inputs["ngspice"],
        "runtime": dict(scored_runtime),
        "plans": provenance,
    }
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    return artifact, marker, arrays


def _only_id_row_changed(
    parent: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> bool:
    for name, value in parent.items():
        other = candidate[name]
        if name == FINAL_WEIGHT:
            if not torch.equal(value[1:], other[1:]):
                return False
        elif name == FINAL_BIAS:
            if not torch.equal(value[1:], other[1:]):
                return False
        elif not torch.equal(value, other):
            return False
    return True


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Harvest, solve, gate, and conditionally emit one PMOS candidate."""
    for name in (
        "parent_nmos", "parent_pmos", "replay_data", "hermite_overlay",
    ):
        path = Path(getattr(args, name)).resolve()
        setattr(args, name, path)
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.bench_root = args.bench_root.resolve()
    source_commit, source_dirty = _source_identity()
    if source_dirty:
        raise RuntimeError("LDO pass-current experiment source has tracked changes")
    scored_runtime = _scored_runtime_contract(args.device, args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    parent_model, stats, _num_codes, output_columns = _load_artifacts(
        args.parent_pmos)
    if tuple(output_columns) != OUTPUT_COLUMNS:
        raise ValueError("parent is not the six-surface DirectNet contract")
    parent_model = parent_model.to(device).eval()
    center_path, center_marker, centers = _harvest_centers(
        args, stats, source_commit, scored_runtime,
    )
    active, validation, replay, overlay_marker = _load_data(
        args.hermite_overlay, args.replay_data, args.parent_pmos, "pmos",
        stats, device,
    )
    del active
    raw_local = _raw_inputs(
        centers["inputs"], centers["nfin"], centers["length"],
        centers["temperature"],
    )
    local_x = torch.as_tensor(
        _normalize_inputs(raw_local, stats), dtype=torch.float32, device=device,
    )
    local_codes = torch.as_tensor(
        centers["tech_codes"], dtype=torch.long, device=device,
    )
    local_target = _normalize_outputs(centers["outputs"], stats)[:, 0]
    with torch.no_grad():
        local_parent_prediction = (
            parent_model(local_x, tech_codes=local_codes)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
    local_residual = local_target.astype(np.float64) - local_parent_prediction
    local_features = _hidden_features(
        parent_model, local_x, local_codes, args.feature_batch_size,
    )
    replay_features = _hidden_features(
        parent_model, replay["x"], replay["code"], args.feature_batch_size,
    )
    control_delta, control_diagnostics = solve_output_row_delta(
        local_features, local_residual, replay_features,
        local_weight=0.0, anchor_weight=args.anchor_weight,
        ridge_ratio=args.ridge_ratio,
    )
    if not np.array_equal(control_delta, np.zeros_like(control_delta)):
        raise RuntimeError("no-circuit analytic control changed the parent")
    delta, solve_diagnostics = solve_output_row_delta(
        local_features, local_residual, replay_features,
        local_weight=args.local_weight, anchor_weight=args.anchor_weight,
        ridge_ratio=args.ridge_ratio,
    )
    full_effect = local_features @ delta
    row_scale = (
        1.0 if args.step_policy == "full"
        else minimum_local_scale(
            local_parent_prediction, local_target.astype(np.float64),
            full_effect, target_ratio=args.local_ratio,
        )
    )
    applied_delta = row_scale * delta
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in parent_model.state_dict().items()
    }
    candidate_state = {
        name: value.clone() for name, value in parent_state.items()
    }
    apply_output_row_delta(
        candidate_state, FINAL_WEIGHT, FINAL_BIAS, row=0,
        delta=applied_delta,
    )
    if not _only_id_row_changed(parent_state, candidate_state):
        raise RuntimeError("candidate changed a tensor outside PMOS i_d row")
    candidate_model = copy.deepcopy(parent_model)
    candidate_model.load_state_dict(candidate_state)
    candidate_model = candidate_model.to(device).eval()
    with torch.no_grad():
        local_candidate_prediction = (
            candidate_model(local_x, tech_codes=local_codes)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
    local_parent_mae = float(np.mean(np.abs(local_residual)))
    local_candidate_mae = float(np.mean(np.abs(
        local_target.astype(np.float64) - local_candidate_prediction)))
    baseline = _model_metrics(
        parent_model, validation, replay, stats,
        args.value_batch_size, args.jacobian_batch_size,
    )
    candidate_metrics = _model_metrics(
        candidate_model, validation, replay, stats,
        args.value_batch_size, args.jacobian_batch_size,
    )
    gate = candidate_gate(
        baseline, candidate_metrics,
        local_parent_mae=local_parent_mae,
        local_candidate_mae=local_candidate_mae,
        local_ratio=args.local_ratio,
    )
    gate["only_pmos_i_d_row_changed"] = True
    delta_path = args.output_dir / "row_delta.npz"
    if delta_path.exists() and not args.overwrite:
        raise FileExistsError(delta_path)
    np.savez(
        delta_path,
        full_delta=delta,
        applied_delta=applied_delta,
        row_scale=np.asarray(row_scale),
        step_policy=np.asarray(args.step_policy),
        local_parent_prediction=local_parent_prediction,
        local_target=local_target,
        local_full_effect=full_effect,
    )
    summary: dict[str, object] = {
        "source_commit": source_commit,
        "parent_nmos": str(args.parent_nmos),
        "parent_nmos_sha256": sha256_file(args.parent_nmos),
        "parent_pmos": str(args.parent_pmos),
        "parent_pmos_sha256": sha256_file(args.parent_pmos),
        "parent_normalization_sha256": sha256_file(
            _normalization_path(args.parent_pmos)),
        "center_artifact": center_path.name,
        "center_artifact_sha256": center_marker["artifact_sha256"],
        "replay_data": str(args.replay_data),
        "replay_data_sha256": sha256_file(args.replay_data),
        "hermite_overlay": str(args.hermite_overlay),
        "hermite_overlay_sha256": sha256_file(args.hermite_overlay),
        "replay_rows": len(replay["x"]),
        "validation_rows": len(validation["x"]),
        "local_rows": len(local_x),
        "local_parent_i_d_mae": local_parent_mae,
        "local_candidate_i_d_mae": local_candidate_mae,
        "step_policy": args.step_policy,
        "row_scale": row_scale,
        "row_delta": delta_path.name,
        "row_delta_sha256": sha256_file(delta_path),
        "scored_runtime": scored_runtime,
        "control": control_diagnostics,
        "solve": solve_diagnostics,
        "baseline": baseline,
        "candidate": candidate_metrics,
        "gate": gate,
        "candidate_checkpoint": None,
    }
    if bool(gate["eligible"]):
        candidate_dir = args.output_dir / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = candidate_dir / args.parent_pmos.name
        norm = candidate_dir / _normalization_path(args.parent_pmos).name
        marker_path = Path(f"{checkpoint}.complete")
        if any(path.exists() for path in (checkpoint, norm, marker_path)) \
                and not args.overwrite:
            raise FileExistsError(checkpoint)
        torch.save(candidate_state, checkpoint)
        shutil.copy2(_normalization_path(args.parent_pmos), norm)
        if sha256_file(norm) != sha256_file(_normalization_path(args.parent_pmos)):
            raise RuntimeError("candidate normalization differs from parent")
        marker = {
            "family": "directnet-full",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "normalization": norm.name,
            "normalization_sha256": sha256_file(norm),
            "output_columns": list(OUTPUT_COLUMNS),
            "source_commit": source_commit,
            "parent_checkpoint": args.parent_pmos.name,
            "parent_checkpoint_sha256": sha256_file(args.parent_pmos),
            "center_artifact": center_path.name,
            "center_artifact_sha256": center_marker["artifact_sha256"],
            "replay_data_sha256": sha256_file(args.replay_data),
            "hermite_overlay_sha256": sha256_file(args.hermite_overlay),
            "row_delta": delta_path.name,
            "row_delta_sha256": sha256_file(delta_path),
            "fit": {
                "target": "PMOS i_d final row",
                "local_weight": args.local_weight,
                "anchor_weight": args.anchor_weight,
                "ridge_ratio": args.ridge_ratio,
                "step_policy": args.step_policy,
                "row_scale": row_scale,
                "local_ratio": args.local_ratio,
            },
            "gate": gate,
        }
        marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
        _load_artifacts(checkpoint)
        summary["candidate_checkpoint"] = str(checkpoint)
        summary["candidate_checkpoint_sha256"] = sha256_file(checkpoint)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-nmos", type=Path, required=True)
    parser.add_argument("--parent-pmos", type=Path, required=True)
    parser.add_argument("--replay-data", type=Path, required=True)
    parser.add_argument("--hermite-overlay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--bench-root", type=Path, default=run_compare.BENCH_ROOT,
    )
    parser.add_argument("--target-instance", default=TARGET_INSTANCE)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--ridge-ratio", type=float, default=1e-6)
    parser.add_argument(
        "--step-policy", choices=("full", "minimum-local"), default="full",
    )
    parser.add_argument("--local-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=767)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--value-batch-size", type=int, default=8192)
    parser.add_argument("--jacobian-batch-size", type=int, default=1024)
    parser.add_argument("--ng-timeout", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps({
        "eligible": summary["gate"]["eligible"],
        "candidate_checkpoint": summary["candidate_checkpoint"],
        "summary": str((args.output_dir / "summary.json").resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
