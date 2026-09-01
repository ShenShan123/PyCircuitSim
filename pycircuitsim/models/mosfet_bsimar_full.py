"""Full-terminal BSIM-AR compact model (LEVEL=76)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from pycircuitsim.models.mosfet_directnet_full import (
    _ArtifactBundle,
    _FullTerminalNNBase,
    _artifact_key,
    _sha256,
)

from neural_network.data.contracts import (
    BSIMAR_FULL_TERMINAL_COLUMN_ORDER,
    FULL_TERMINAL_OUTPUT_COLUMN_ORDER,
    FULL_TERMINAL_OUTPUT_CONTRACT,
)
from neural_network.data.normalize import NormStats
from neural_network.models.transformer import TransformerEncoderModel


_OUTPUT_COLUMNS = tuple(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)
_TARGET_COLUMNS = tuple(BSIMAR_FULL_TERMINAL_COLUMN_ORDER)
_ARTIFACT_CACHE: Dict[
    Tuple[
        Tuple[str, int, int], Tuple[str, int, int],
        Tuple[str, int, int], Tuple[str, int, int],
    ],
    _ArtifactBundle,
] = {}


def _required_artifact(checkpoint: Path, suffix: str, label: str) -> Path:
    path = checkpoint.parent / (
        checkpoint.stem.replace("_best", suffix) + ".npz")
    if not path.exists():
        raise FileNotFoundError(
            f"Full-terminal BSIM-AR {label} artifact not found: {path}")
    return path


def _scalar(config: np.lib.npyio.NpzFile, name: str) -> object:
    if name not in config.files:
        raise ValueError(f"Full-terminal BSIM-AR config is missing {name}")
    value = config[name]
    return value.item() if value.ndim == 0 else value


def _load_artifacts(checkpoint: Path) -> _ArtifactBundle:
    """Verify and load one immutable full-terminal BSIM-AR bundle."""
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Full-terminal BSIM-AR checkpoint not found: {checkpoint}")
    marker = Path(f"{checkpoint}.complete")
    if not marker.exists():
        raise FileNotFoundError(
            f"Full-terminal BSIM-AR completion marker not found: {marker}")
    norm_path = _required_artifact(checkpoint, "_norm", "normalization")
    config_path = _required_artifact(checkpoint, "_config", "configuration")
    cache_key = tuple(_artifact_key(path) for path in (
        checkpoint, norm_path, config_path, marker))
    cached = _ARTIFACT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        marker_data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid completion marker: {marker}") from exc
    if not isinstance(marker_data, dict):
        raise ValueError(f"Invalid completion marker schema: {marker}")
    expected_marker = {
        "family": "bsimar-full",
        "checkpoint": checkpoint.name,
        "normalization": norm_path.name,
        "configuration": config_path.name,
        "output_columns": list(_OUTPUT_COLUMNS),
        "target_columns": list(_TARGET_COLUMNS),
    }
    for name, expected in expected_marker.items():
        if marker_data.get(name) != expected:
            raise ValueError(
                f"Completion marker {marker} has invalid {name}: "
                f"expected {expected!r}")
    for name, path in (
        ("checkpoint", checkpoint),
        ("normalization", norm_path),
        ("configuration", config_path),
    ):
        checksum = marker_data.get(f"{name}_sha256")
        if not isinstance(checksum, str) or _sha256(path) != checksum:
            raise ValueError(f"Completion marker {name} SHA-256 mismatch: {path}")

    norm_stats = NormStats.load(str(norm_path))
    if tuple(norm_stats.output_columns or ()) != _OUTPUT_COLUMNS:
        raise ValueError(
            "Full-terminal BSIM-AR norm output columns must be "
            f"{list(_OUTPUT_COLUMNS)}")
    if norm_stats.mode not in ("zscore", "asinh"):
        raise ValueError(
            f"Unsupported normalization mode {norm_stats.mode!r}")

    with np.load(config_path, allow_pickle=False) as config:
        output_contract = str(_scalar(config, "output_contract"))
        target_columns = tuple(
            str(value) for value in _scalar(config, "target_columns"))
        if output_contract != FULL_TERMINAL_OUTPUT_CONTRACT:
            raise ValueError(
                "Full-terminal BSIM-AR config has the wrong output contract")
        if target_columns != _TARGET_COLUMNS:
            raise ValueError(
                "Full-terminal BSIM-AR target columns must be "
                f"{list(_TARGET_COLUMNS)}")
        kwargs = {
            name: int(_scalar(config, name))
            for name in (
                "input_dim", "target_dim", "d_model", "nhead",
                "num_layers", "dim_feedforward", "num_tech_codes",
                "ar_target_dim",
            )
        }
        dropout = float(_scalar(config, "dropout"))

    if kwargs["target_dim"] != len(_TARGET_COLUMNS):
        raise ValueError("Full-terminal BSIM-AR target_dim must be 6")
    if kwargs["ar_target_dim"] not in (3, len(_TARGET_COLUMNS)):
        raise ValueError(
            "Full-terminal BSIM-AR must autoregress 3 or all 6 targets")
    if marker_data.get("ar_target_dim") != kwargs["ar_target_dim"]:
        raise ValueError(
            "Full-terminal BSIM-AR completion marker ar_target_dim does "
            "not match its configuration")
    model = TransformerEncoderModel(
        **kwargs,
        dropout=dropout,
        tech_embed_dropout=0.0,
        unknown_code_id=kwargs["num_tech_codes"] - 1,
    )
    state = torch.load(str(checkpoint), weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    loaded = (
        model, norm_stats, kwargs["num_tech_codes"], _TARGET_COLUMNS)
    _ARTIFACT_CACHE[cache_key] = loaded
    return loaded


class NMOS_TFF(_FullTerminalNNBase):
    """N-channel full-terminal BSIM-AR (LEVEL=76)."""

    _artifact_loader = staticmethod(_load_artifacts)
    _family_label = "BSIM-AR-Full"


class PMOS_TFF(_FullTerminalNNBase):
    """P-channel full-terminal BSIM-AR (LEVEL=76)."""

    _artifact_loader = staticmethod(_load_artifacts)
    _family_label = "BSIM-AR-Full"

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Return the project PMOS scalar sign for comparison consumers."""
        return -super().calculate_current(voltages)


__all__ = ["NMOS_TFF", "PMOS_TFF"]
