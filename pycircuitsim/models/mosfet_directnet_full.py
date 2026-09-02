"""Full-terminal DirectNet compact model (LEVEL=75).

This family learns three independent solver-positive terminal currents
(``i_d``, ``i_g``, ``i_b``) and three independent terminal charges
(``qd``, ``qg``, ``qb``).  Source values follow from closure.  The source
Jacobian column follows from voltage-translation invariance and the source
row follows from KCL, yielding full 4x4 current and charge stamps.
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pycircuitsim.models.base import Component


PROJECT_ROOT = Path(__file__).parent.parent.parent
_NN_PARENT = PROJECT_ROOT / "external_compact_models"
if str(_NN_PARENT) not in sys.path:
    sys.path.insert(0, str(_NN_PARENT))

from neural_network.data.contracts import (  # noqa: E402
    FULL_TERMINAL_OUTPUT_COLUMN_ORDER,
)
from neural_network.data.normalize import NormStats  # noqa: E402


_OUTPUT_COLUMNS = tuple(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)
_ArtifactBundle = Tuple[torch.nn.Module, NormStats, int, Tuple[str, ...]]
_ARTIFACT_CACHE: Dict[
    Tuple[Tuple[str, int, int], Tuple[str, int, int], Tuple[str, int, int]],
    _ArtifactBundle,
] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_key(path: Path) -> Tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _load_artifacts(
    checkpoint: Path,
) -> _ArtifactBundle:
    """Verify and load one immutable full-terminal artifact bundle."""
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Full-terminal DirectNet checkpoint not found: {checkpoint}")
    marker = Path(f"{checkpoint}.complete")
    if not marker.exists():
        raise FileNotFoundError(
            f"Full-terminal DirectNet completion marker not found: {marker}")
    norm_path = checkpoint.parent / (
        checkpoint.stem.replace("_best", "_norm") + ".npz")
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Full-terminal DirectNet norm artifact not found: {norm_path}")

    cache_key = (
        _artifact_key(checkpoint), _artifact_key(norm_path),
        _artifact_key(marker),
    )
    cached = _ARTIFACT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        marker_data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid completion marker: {marker}") from exc
    if not isinstance(marker_data, dict):
        raise ValueError(f"Invalid completion marker schema: {marker}")
    if marker_data.get("family") != "directnet-full":
        raise ValueError(
            f"Completion marker {marker} is not family=directnet-full")
    if marker_data.get("checkpoint") != checkpoint.name:
        raise ValueError(
            f"Completion marker {marker} names a different checkpoint")
    checkpoint_sha256 = marker_data.get("checkpoint_sha256")
    if (not isinstance(checkpoint_sha256, str)
            or _sha256(checkpoint) != checkpoint_sha256):
        raise ValueError(
            f"Completion marker checkpoint SHA-256 mismatch: {checkpoint}")
    if marker_data.get("normalization") != norm_path.name:
        raise ValueError(
            f"Completion marker {marker} names a different normalization")
    normalization_sha256 = marker_data.get("normalization_sha256")
    if (not isinstance(normalization_sha256, str)
            or _sha256(norm_path) != normalization_sha256):
        raise ValueError(
            f"Completion marker normalization SHA-256 mismatch: {norm_path}")
    if tuple(marker_data.get("output_columns") or ()) != _OUTPUT_COLUMNS:
        raise ValueError(
            f"Completion marker output columns must be "
            f"{list(_OUTPUT_COLUMNS)}")

    norm_stats = NormStats.load(str(norm_path))
    if tuple(norm_stats.output_columns or ()) != _OUTPUT_COLUMNS:
        raise ValueError(
            "Full-terminal DirectNet norm output columns must be "
            f"{list(_OUTPUT_COLUMNS)}")
    if norm_stats.mode not in ("zscore", "asinh"):
        raise ValueError(
            f"Unsupported normalization mode {norm_stats.mode!r}")

    state = torch.load(str(checkpoint), weights_only=True, map_location="cpu")
    if any(key.startswith(("mono.", "core.")) for key in state):
        raise ValueError(
            "Full-terminal DirectNet requires the plain six-surface "
            "architecture")
    from neural_network.models.direct_net import DirectNet

    net_keys = sorted(
        (key for key in state
         if key.startswith("net.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not net_keys or "tech_embedding.weight" not in state:
        raise ValueError(f"Invalid DirectNet checkpoint: {checkpoint}")
    output_dim = int(state[net_keys[-1]].shape[0])
    if output_dim != len(_OUTPUT_COLUMNS):
        raise ValueError(
            f"Full-terminal DirectNet head must have 6 outputs, got "
            f"{output_dim}")
    hidden_dim = int(state[net_keys[-1]].shape[1])
    n_layers = len(net_keys) - 1
    num_tech_codes, tech_embed_dim = (
        int(value) for value in state["tech_embedding.weight"].shape)
    input_dim = int(state[net_keys[0]].shape[1]) - tech_embed_dim
    model = DirectNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_dim=output_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=tech_embed_dim,
        tech_embed_dropout=0.0,
        unknown_code_id=num_tech_codes - 1,
    )
    model.load_state_dict(state)
    model.eval()
    loaded = (model, norm_stats, num_tech_codes, _OUTPUT_COLUMNS)
    _ARTIFACT_CACHE[cache_key] = loaded
    return loaded


class _FullTerminalNNBase(Component):
    """Architecture-neutral runtime for a six-surface terminal model."""

    _artifact_loader = staticmethod(_load_artifacts)
    _family_label = "DirectNet-Full"

    def __init__(
        self,
        name: str,
        nodes: List[str],
        model_path: str,
        L: float,
        NFIN: float,
        temperature: float = 300.15,
        tech_code: Optional[int] = None,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__(name, nodes, None)
        if len(nodes) != 4:
            raise ValueError(
                f"Full-terminal DirectNet must have 4 nodes, got {len(nodes)}")
        if L <= 0.0 or NFIN <= 0.0 or multiplier <= 0.0:
            raise ValueError("L, NFIN, and instance multiplier must be positive")

        self.L = float(L)
        self.NFIN = float(NFIN)
        self.m = float(multiplier)
        self.temperature = float(temperature)

        (self._nn_model, self._norm_stats, num_tech_codes,
         model_output_columns) = self._artifact_loader(Path(model_path))
        self._model_output_columns = tuple(model_output_columns)
        if (len(self._model_output_columns) != len(_OUTPUT_COLUMNS)
                or set(self._model_output_columns) != set(_OUTPUT_COLUMNS)):
            raise ValueError(
                f"{self._family_label} model output columns must be a "
                f"permutation of {list(_OUTPUT_COLUMNS)}, got "
                f"{list(self._model_output_columns)}")

        self._tech_code = (
            num_tech_codes - 1 if tech_code is None else int(tech_code))
        if not 0 <= self._tech_code < num_tech_codes:
            raise ValueError(
                f"tech_code {self._tech_code} is outside checkpoint vocab")
        self._tech_code_tensor = torch.tensor(
            [self._tech_code], dtype=torch.long)

        self._input_std = np.asarray(
            self._norm_stats.input_std, dtype=np.float64).copy()
        self._input_std[self._input_std < 1e-12] = 1.0
        self._stats_idx = {
            name: index for index, name in enumerate(_OUTPUT_COLUMNS)}
        self._cache_voltages: Optional[Tuple[float, ...]] = None
        self._eval_cache: Optional[Tuple[np.ndarray, np.ndarray,
                                         np.ndarray,
                                         Optional[np.ndarray]]] = None
        self._caps_required = False
        self._q_prev: Optional[Dict[str, float]] = None
        self._q_prev2: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate = 0.0
        self._i_prev_drain = 0.0
        self._i_prev_source = 0.0
        self._i_prev_bulk = 0.0

    def _forward_model(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the checkpoint; kept as a seam for focused tests."""
        return self._nn_model(x, tech_codes=self._tech_code_tensor)

    def _raw_inputs(self, voltages: Dict[str, float]) -> np.ndarray:
        v_d, v_g, v_s, v_b = (
            float(voltages.get(node, 0.0)) for node in self.nodes)
        return np.asarray([
            v_d - v_s,
            v_g - v_s,
            0.0,
            v_b - v_s,
            math.log2(max(self.NFIN, 1.0)),
            self.L,
            self.temperature,
        ], dtype=np.float64)

    def _check_support(self, raw: np.ndarray) -> None:
        lower = np.asarray(self._norm_stats.input_min, dtype=np.float64)
        upper = np.asarray(self._norm_stats.input_max, dtype=np.float64)
        outside = np.flatnonzero((raw < lower) | (raw > upper))
        if outside.size:
            index = int(outside[0])
            raise ValueError(
                f"{self.name} is outside certified support at input "
                f"{index}: {raw[index]} not in [{lower[index]}, "
                f"{upper[index]}]")

    def _denorm_value(self, name: str, value: float) -> float:
        index = self._stats_idx[name]
        u = (value * float(self._norm_stats.output_std[index])
             + float(self._norm_stats.output_mean[index]))
        if self._norm_stats.mode == "asinh":
            scales = self._norm_stats.asinh_scale
            if scales is None:
                raise ValueError("asinh norm artifact is missing asinh_scale")
            return float(scales[index]) * float(np.sinh(u))
        return u

    def _denorm_derivative(
        self,
        name: str,
        input_index: int,
        derivative: float,
        physical_value: float,
    ) -> float:
        index = self._stats_idx[name]
        factor = 1.0
        if self._norm_stats.mode == "asinh":
            scales = self._norm_stats.asinh_scale
            if scales is None:
                raise ValueError("asinh norm artifact is missing asinh_scale")
            scale = float(scales[index])
            factor = math.sqrt(scale * scale + physical_value * physical_value)
        return (float(derivative)
                * float(self._norm_stats.output_std[index])
                * factor / float(self._input_std[input_index]))

    @staticmethod
    def _closed_values(independent: np.ndarray) -> np.ndarray:
        result = np.empty(4, dtype=np.float64)
        result[[0, 1, 3]] = independent
        result[2] = -float(independent.sum())
        return result

    @staticmethod
    def _closed_jacobian(independent: np.ndarray) -> np.ndarray:
        """Build a 4x4 invariant/KCL-closed matrix from three d/d(Vd,Vg,Vb)."""
        result = np.zeros((4, 4), dtype=np.float64)
        result[np.ix_([0, 1, 3], [0, 1, 3])] = independent
        result[[0, 1, 3], 2] = -independent.sum(axis=1)
        result[2, :] = -result[[0, 1, 3], :].sum(axis=0)
        return result

    def _eval(
        self, voltages: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        key = tuple(float(voltages.get(node, 0.0)) for node in self.nodes)
        if (self._cache_voltages == key and self._eval_cache is not None
                and (not self._caps_required
                     or self._eval_cache[3] is not None)):
            return self._eval_cache

        raw = self._raw_inputs(voltages)
        self._check_support(raw)
        normalized = (
            (raw - np.asarray(self._norm_stats.input_mean, dtype=np.float64))
            / self._input_std)
        x = torch.tensor(
            normalized, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        output = self._forward_model(x)
        if output.shape != (1, len(self._model_output_columns)):
            raise RuntimeError(
                f"{self._family_label} returned shape {tuple(output.shape)}")

        values: Dict[str, float] = {
            name: self._denorm_value(name, float(output[0, index].detach()))
            for index, name in enumerate(self._model_output_columns)
        }
        derivatives: Dict[str, np.ndarray] = {}
        differentiated = ["i_d", "i_g", "i_b"]
        if self._caps_required:
            differentiated.extend(["qd", "qg", "qb"])
        for gradient_index, name in enumerate(differentiated):
            index = self._model_output_columns.index(name)
            physical = values[name]
            gradient = torch.autograd.grad(
                output[0, index], x,
                retain_graph=gradient_index + 1 < len(differentiated),
            )[0][0]
            derivatives[name] = np.asarray([
                self._denorm_derivative(
                    name, input_index, float(gradient[input_index]), physical)
                for input_index in (0, 1, 3)
            ], dtype=np.float64)

        current_names = ("i_d", "i_g", "i_b")
        charge_names = ("qd", "qg", "qb")
        currents = self._closed_values(np.asarray(
            [values[name] for name in current_names])) * self.m
        current_jacobian = self._closed_jacobian(np.stack(
            [derivatives[name] for name in current_names])) * self.m
        charges = self._closed_values(np.asarray(
            [values[name] for name in charge_names])) * self.m
        charge_jacobian = None
        if self._caps_required:
            charge_jacobian = self._closed_jacobian(np.stack(
                [derivatives[name] for name in charge_names])) * self.m

        finite_values = [currents, current_jacobian, charges]
        if charge_jacobian is not None:
            finite_values.append(charge_jacobian)
        if not all(np.all(np.isfinite(value)) for value in finite_values):
            raise RuntimeError(f"{self._family_label} produced NaN/Inf")
        self._cache_voltages = key
        self._eval_cache = (
            currents, current_jacobian, charges, charge_jacobian)
        return self._eval_cache

    def get_terminal_stamp(
        self, voltages: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        currents, jacobian, _, _ = self._eval(voltages)
        return currents.copy(), jacobian.copy()

    def get_charge_stamp(
        self, voltages: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._caps_required:
            self.require_capacitance_jacobians()
        _, _, charges, jacobian = self._eval(voltages)
        assert jacobian is not None
        return charges.copy(), jacobian.copy()

    def require_capacitance_jacobians(self) -> None:
        """Declare a charge-Jacobian consumer and invalidate a DC-only cache."""
        if self._caps_required:
            return
        self._caps_required = True
        if self._eval_cache is None or self._eval_cache[3] is None:
            self.clear_cache()

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(
        self, matrix: np.ndarray, node_map: Dict[str, int],
    ) -> None:
        pass

    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        pass

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return float(self._eval(voltages)[0][0])

    def get_conductance(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float]:
        jacobian = self._eval(voltages)[1]
        return (float(jacobian[0, 0]), float(jacobian[0, 1]),
                float(jacobian[0, 3]))

    def get_charges(self, voltages: Dict[str, float]) -> Dict[str, float]:
        charges = self._eval(voltages)[2]
        return dict(zip(("qd", "qg", "qs", "qb"), map(float, charges)))

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        charges = self.get_charges(voltages)
        self._q_prev = charges.copy()
        self._q_prev2 = charges.copy()
        self._v_prev_tran = {
            key: float(voltages.get(node, 0.0))
            for key, node in zip(("d", "g", "s", "b"), self.nodes)
        }
        self._i_prev_gate = 0.0
        self._i_prev_drain = 0.0
        self._i_prev_source = 0.0
        self._i_prev_bulk = 0.0

    def update_charge_state(
        self,
        voltages: Dict[str, float],
        cap_currents: Optional[Dict[str, float]] = None,
    ) -> None:
        charges = self.get_charges(voltages)
        self._q_prev2 = (
            self._q_prev.copy() if self._q_prev is not None
            else charges.copy())
        self._q_prev = charges.copy()
        self._v_prev_tran = {
            key: float(voltages.get(node, 0.0))
            for key, node in zip(("d", "g", "s", "b"), self.nodes)
        }
        if cap_currents is not None:
            self._i_prev_gate = cap_currents.get("i_gate", 0.0)
            self._i_prev_drain = cap_currents.get("i_drain", 0.0)
            self._i_prev_source = cap_currents.get("i_source", 0.0)
            self._i_prev_bulk = cap_currents.get("i_bulk", 0.0)

    def set_temperature(self, temperature_kelvin: float) -> None:
        if temperature_kelvin <= 200.0:
            raise ValueError(
                f"Temperature must be in Kelvin (> 200 K), got "
                f"{temperature_kelvin}")
        self.temperature = float(temperature_kelvin)
        self.clear_cache()
        self._q_prev = None
        self._q_prev2 = None
        self._v_prev_tran = None

    def clear_cache(self) -> None:
        self._cache_voltages = None
        self._eval_cache = None


class NMOS_DNF(_FullTerminalNNBase):
    """N-channel full-terminal DirectNet (LEVEL=75)."""


class PMOS_DNF(_FullTerminalNNBase):
    """P-channel full-terminal DirectNet (LEVEL=75)."""

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """Return the project PMOS scalar sign for comparison consumers."""
        return -super().calculate_current(voltages)


# Compatibility for private imports in downstream diagnostic scripts.
_DirectNetFullBase = _FullTerminalNNBase


__all__ = ["NMOS_DNF", "PMOS_DNF"]
