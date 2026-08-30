#!/usr/bin/env python3
"""Bounded DirectNet-Full fine-tune with an unrolled LDO DC solve.

The circuit residual follows the LEVEL=75 device boundary exactly: the model
predicts solver-positive ``i_d``, ``i_g`` and ``i_b``; source current follows
from KCL; and every terminal contributes to its node equation.  This script is
an experiment driver, not a runtime solver implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "external_compact_models"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pycircuitsim.models.mosfet_directnet_full import (  # noqa: E402
    _load_artifacts,
)
from scripts import finetune_active_hermite as hermite  # noqa: E402


SCHEMA_VERSION = 1
SCHEMA_NAME = "v768-dnf-unrolled-ldo"
EXPECTED_STATES = 28
EXPECTED_GROUPS = 28
EXPECTED_DEVICES = 9
EPOCHS = 8
LEARNING_RATE = 1e-6
REPLAY_ROWS = 262_144
REPLAY_BATCH_SIZE = 4096
GRADIENT_CLIP = 1.0
LM_STEPS = 8
LM_LAMBDA = 0.05
LM_STEP = 1.0
TRUST_CLIP_V = 0.1
RESIDUAL_LOSS_WEIGHT = 0.1
VALUE_MAE_RATIO = 1.02
VALUE_MAX_RATIO = 1.05
JACOBIAN_MAX_RATIO = 1.05
VOUT_MAE_RATIO = 0.50
PHYSICAL_GMIN = 1e-12
ARM_SCALE_FLOOR = 1e-7
LOSS_DENOMINATOR_FLOOR = 1e-12


class SupportError(RuntimeError):
    """A differentiable solve attempted to evaluate outside model support."""


class ArtifactError(ValueError):
    """A harvested topology artifact violates its declared schema."""


@dataclass(frozen=True)
class CircuitArtifact:
    """Tensor form of one harvested DC topology and its LEVEL=72 states."""

    free_nodes: tuple[str, ...]
    fixed_nodes: tuple[str, ...]
    output_node: str
    state_group: torch.Tensor
    group_segment_offsets: torch.Tensor
    group_sweep: torch.Tensor
    free_voltage_l72: torch.Tensor
    fixed_voltage_l72: torch.Tensor
    arm_scale: torch.Tensor
    mos_is_pmos: torch.Tensor
    mos_term_free: torch.Tensor
    mos_term_fixed: torch.Tensor
    mos_nfin: torch.Tensor
    mos_length: torch.Tensor
    mos_temperature: torch.Tensor
    mos_multiplier: torch.Tensor
    mos_code: torch.Tensor
    mos_support_min: torch.Tensor
    mos_support_max: torch.Tensor
    mos_support_min_fractional_margin: torch.Tensor
    resistor_term_free: torch.Tensor
    resistor_term_fixed: torch.Tensor
    resistor_conductance: torch.Tensor
    current_term_free: torch.Tensor
    current_term_fixed: torch.Tensor
    current_value: torch.Tensor
    gmin: float

    @property
    def output_index(self) -> int:
        return self.free_nodes.index(self.output_node)

    @property
    def states(self) -> int:
        return int(self.free_voltage_l72.shape[0])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "CircuitArtifact":
        with np.load(path, allow_pickle=False) as source:
            mapping = {name: np.asarray(source[name]) for name in source.files}
        return cls.from_mapping(mapping, device=device, dtype=dtype)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "CircuitArtifact":
        """Validate and tensorize the shared V7.6.8 topology schema."""

        def required(name: str) -> np.ndarray:
            if name not in mapping:
                raise ArtifactError(f"missing artifact field {name}")
            return np.asarray(mapping[name])

        version = int(required("schema_version"))
        if version != SCHEMA_VERSION:
            raise ArtifactError(
                f"schema_version must be {SCHEMA_VERSION}, got {version}")
        schema_name = str(required("schema_name"))
        if schema_name != SCHEMA_NAME:
            raise ArtifactError(
                f"schema_name must be {SCHEMA_NAME!r}, got {schema_name!r}")
        free_nodes = tuple(str(value) for value in required("free_nodes"))
        fixed_nodes = tuple(str(value) for value in required("fixed_nodes"))
        output_node = str(required("output_node"))
        if not free_nodes or len(set(free_nodes)) != len(free_nodes):
            raise ArtifactError("free_nodes must be non-empty and unique")
        if len(set(fixed_nodes)) != len(fixed_nodes):
            raise ArtifactError("fixed_nodes must be unique")
        if set(free_nodes).intersection(fixed_nodes):
            raise ArtifactError("free and fixed nodes overlap")
        if output_node not in free_nodes:
            raise ArtifactError("output_node must be a free node")

        def real(name: str) -> torch.Tensor:
            value = required(name)
            if not np.all(np.isfinite(value)):
                raise ArtifactError(f"{name} contains NaN/Inf")
            return torch.as_tensor(value, dtype=dtype, device=device)

        def integer(name: str) -> torch.Tensor:
            return torch.as_tensor(
                required(name), dtype=torch.long, device=device)

        free_voltage = real("free_voltage_l72")
        fixed_voltage = real("fixed_voltage_l72")
        arm_scale = real("arm_scale")
        if free_voltage.ndim != 2 or free_voltage.shape[1] != len(free_nodes):
            raise ArtifactError("free_voltage_l72 has the wrong shape")
        states = int(free_voltage.shape[0])
        if fixed_voltage.shape != (states, len(fixed_nodes)):
            raise ArtifactError("fixed_voltage_l72 has the wrong shape")
        if (arm_scale.shape != free_voltage.shape
                or torch.any(arm_scale < ARM_SCALE_FLOOR)):
            raise ArtifactError(
                "arm_scale must be state-aligned and respect its 1e-7 A floor")

        state_group = integer("state_group")
        offsets = integer("group_segment_offsets")
        group_sweep = real("group_sweep")
        if state_group.shape != (states,):
            raise ArtifactError("state_group must contain one entry per state")
        groups = int(group_sweep.numel())
        if group_sweep.ndim != 1 or offsets.shape != (groups + 1,):
            raise ArtifactError("group sweep/segment offsets are inconsistent")
        if (int(offsets[0]) != 0 or int(offsets[-1]) != states
                or torch.any(offsets[1:] < offsets[:-1])):
            raise ArtifactError("group_segment_offsets is not valid CSR")
        if states and (
            int(state_group.min()) < 0 or int(state_group.max()) >= groups
        ):
            raise ArtifactError("state_group is outside the group range")
        for group in range(groups):
            start, stop = int(offsets[group]), int(offsets[group + 1])
            if stop > start and not torch.all(state_group[start:stop] == group):
                raise ArtifactError("state_group disagrees with CSR offsets")

        mos_is_pmos = torch.as_tensor(
            required("mos_is_pmos"), dtype=torch.bool, device=device)
        devices = int(mos_is_pmos.numel())
        mos_term_free = integer("mos_term_free")
        mos_term_fixed = integer("mos_term_fixed")
        if mos_term_free.shape != (devices, 4):
            raise ArtifactError("mos_term_free must have shape (D, 4)")
        if mos_term_fixed.shape != (devices, 4):
            raise ArtifactError("mos_term_fixed must have shape (D, 4)")

        def device_real(name: str) -> torch.Tensor:
            value = real(name)
            if value.shape != (devices,):
                raise ArtifactError(f"{name} must have shape (D,)")
            return value

        mos_nfin = device_real("mos_nfin")
        mos_length = device_real("mos_length")
        mos_temperature = device_real("mos_temperature")
        mos_multiplier = device_real("mos_multiplier")
        mos_code = integer("mos_code")
        if mos_code.shape != (devices,):
            raise ArtifactError("mos_code must have shape (D,)")
        mos_support_min = real("mos_support_min")
        mos_support_max = real("mos_support_max")
        mos_support_margin = device_real(
            "mos_support_min_fractional_margin")
        if (mos_support_min.shape != (devices, 7)
                or mos_support_max.shape != (devices, 7)):
            raise ArtifactError("MOS support bounds must have shape (D, 7)")
        if (torch.any(mos_support_max < mos_support_min)
                or torch.any(mos_support_margin < 0)):
            raise ArtifactError("MOS support bounds or margins are invalid")
        if (torch.any(mos_nfin <= 0) or torch.any(mos_length <= 0)
                or torch.any(mos_temperature <= 0)
                or torch.any(mos_multiplier <= 0)):
            raise ArtifactError("MOS geometry and multiplier must be positive")

        def endpoint_fields(prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
            free = integer(f"{prefix}_term_free")
            fixed = integer(f"{prefix}_term_fixed")
            if free.ndim != 2 or free.shape[1] != 2 or fixed.shape != free.shape:
                raise ArtifactError(f"{prefix} endpoints must have shape (N, 2)")
            return free, fixed

        resistor_free, resistor_fixed = endpoint_fields("resistor")
        current_free, current_fixed = endpoint_fields("current")
        resistor_conductance = real("resistor_conductance")
        current_value = real("current_value")
        resistors = int(resistor_free.shape[0])
        currents = int(current_free.shape[0])
        if resistor_conductance.shape != (resistors,):
            raise ArtifactError("resistor_conductance must have shape (R,)")
        if torch.any(resistor_conductance <= 0):
            raise ArtifactError("resistor conductance must be positive")
        if current_value.shape not in ((currents,), (states, currents)):
            raise ArtifactError("current_value must have shape (I,) or (S, I)")

        def validate_map(
            name: str, free: torch.Tensor, fixed: torch.Tensor,
        ) -> None:
            if not torch.all((free >= 0) ^ (fixed >= 0)):
                raise ArtifactError(
                    f"every {name} endpoint must map to exactly one node set")
            if torch.any(free >= len(free_nodes)):
                raise ArtifactError(f"{name} has an invalid free-node index")
            if torch.any(fixed >= len(fixed_nodes)):
                raise ArtifactError(f"{name} has an invalid fixed-node index")

        validate_map("MOS terminal", mos_term_free, mos_term_fixed)
        validate_map("resistor", resistor_free, resistor_fixed)
        validate_map("current source", current_free, current_fixed)
        gmin = float(required("gmin"))
        if not np.isfinite(gmin) or abs(gmin - PHYSICAL_GMIN) > 1e-24:
            raise ArtifactError(
                f"gmin must be the physical {PHYSICAL_GMIN:.0e} S")
        if "vout_index" in mapping and int(np.asarray(mapping["vout_index"])) \
                != free_nodes.index(output_node):
            raise ArtifactError("vout_index disagrees with output_node")
        if "vout_target" in mapping:
            target = np.asarray(mapping["vout_target"], dtype=np.float64)
            expected_target = np.asarray(mapping["free_voltage_l72"])[
                :, free_nodes.index(output_node)]
            if target.shape != (states,) or not np.array_equal(
                target, expected_target,
            ):
                raise ArtifactError(
                    "vout_target disagrees with free_voltage_l72")
        if "terminal_order" in mapping:
            terminal_order = tuple(
                str(value) for value in np.asarray(mapping["terminal_order"]))
            if terminal_order != ("d", "g", "s", "b"):
                raise ArtifactError("terminal_order must be [d,g,s,b]")
        if "gmin_terminal_pairs" in mapping:
            pairs = np.asarray(mapping["gmin_terminal_pairs"], dtype=np.int64)
            if not np.array_equal(pairs, np.asarray(((0, 2), (0, 3), (2, 3)))):
                raise ArtifactError("gmin terminal-pair contract changed")
        if "gmin_scales_with_multiplier" in mapping and bool(
            np.asarray(mapping["gmin_scales_with_multiplier"])
        ):
            raise ArtifactError("physical GMIN must not scale with multiplier")

        return cls(
            free_nodes=free_nodes,
            fixed_nodes=fixed_nodes,
            output_node=output_node,
            state_group=state_group,
            group_segment_offsets=offsets,
            group_sweep=group_sweep,
            free_voltage_l72=free_voltage,
            fixed_voltage_l72=fixed_voltage,
            arm_scale=arm_scale,
            mos_is_pmos=mos_is_pmos,
            mos_term_free=mos_term_free,
            mos_term_fixed=mos_term_fixed,
            mos_nfin=mos_nfin,
            mos_length=mos_length,
            mos_temperature=mos_temperature,
            mos_multiplier=mos_multiplier,
            mos_code=mos_code,
            mos_support_min=mos_support_min,
            mos_support_max=mos_support_max,
            mos_support_min_fractional_margin=mos_support_margin,
            resistor_term_free=resistor_free,
            resistor_term_fixed=resistor_fixed,
            resistor_conductance=resistor_conductance,
            current_term_free=current_free,
            current_term_fixed=current_fixed,
            current_value=current_value,
            gmin=gmin,
        )


def _stats_tensor(
    stats: Any, name: str, reference: torch.Tensor,
) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(getattr(stats, name)),
        dtype=reference.dtype,
        device=reference.device,
    )


def denormalize_outputs(normalized: torch.Tensor, stats: Any) -> torch.Tensor:
    """Differentiably invert the checkpoint's exact output transform."""
    inner = (
        normalized * _stats_tensor(stats, "output_std", normalized)
        + _stats_tensor(stats, "output_mean", normalized)
    )
    if stats.mode == "zscore":
        return inner
    if stats.mode != "asinh" or stats.asinh_scale is None:
        raise ValueError(f"unsupported normalization mode {stats.mode!r}")
    return _stats_tensor(stats, "asinh_scale", normalized) * torch.sinh(inner)


def close_terminal_currents(
    independent: torch.Tensor,
    multiplier: torch.Tensor,
) -> torch.Tensor:
    """Return solver-positive ``[d,g,s,b]`` currents with KCL closure."""
    if independent.shape[-1] != 3:
        raise ValueError("independent currents must be [i_d, i_g, i_b]")
    source = -independent.sum(dim=-1, keepdim=True)
    closed = torch.cat(
        (independent[..., :2], source, independent[..., 2:3]), dim=-1,
    )
    shape = (1,) * (closed.ndim - 2) + (len(multiplier), 1)
    return closed * multiplier.reshape(shape)


def _gather_endpoints(
    free_voltage: torch.Tensor,
    fixed_voltage: torch.Tensor,
    free_index: torch.Tensor,
    fixed_index: torch.Tensor,
) -> torch.Tensor:
    free_value = free_voltage[:, torch.clamp(free_index, min=0)]
    fixed_value = fixed_voltage[:, torch.clamp(fixed_index, min=0)]
    return torch.where(free_index.unsqueeze(0) >= 0, free_value, fixed_value)


def _incidence(
    free_index: torch.Tensor,
    free_nodes: int,
    *,
    terminal_currents: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if terminal_currents:
        result = torch.zeros(
            (*free_index.shape, free_nodes), dtype=dtype, device=device)
        for element in range(free_index.shape[0]):
            for terminal in range(free_index.shape[1]):
                node = int(free_index[element, terminal])
                if node >= 0:
                    result[element, terminal, node] += 1.0
        return result
    result = torch.zeros(
        (free_index.shape[0], free_nodes), dtype=dtype, device=device)
    for element in range(free_index.shape[0]):
        positive, negative = map(int, free_index[element])
        if positive >= 0:
            result[element, positive] += 1.0
        if negative >= 0:
            result[element, negative] -= 1.0
    return result


class FullTerminalResidual:
    """Differentiable all-terminal KCL residual for one harvested topology."""

    def __init__(
        self,
        artifact: CircuitArtifact,
        *,
        models: Mapping[str, torch.nn.Module],
        stats: Mapping[str, Any],
    ) -> None:
        if set(models) != {"nmos", "pmos"} or set(stats) != {"nmos", "pmos"}:
            raise ValueError("models and stats must contain nmos and pmos")
        self.artifact = artifact
        self.models = models
        self.stats = stats
        dtype = artifact.free_voltage_l72.dtype
        device = artifact.free_voltage_l72.device
        nodes = len(artifact.free_nodes)
        self._mos_incidence = _incidence(
            artifact.mos_term_free, nodes,
            terminal_currents=True, dtype=dtype, device=device,
        )
        self._resistor_incidence = _incidence(
            artifact.resistor_term_free, nodes,
            terminal_currents=False, dtype=dtype, device=device,
        )
        self._current_incidence = _incidence(
            artifact.current_term_free, nodes,
            terminal_currents=False, dtype=dtype, device=device,
        )
        pairs = ((0, 2), (0, 3), (2, 3))
        pair_free = torch.stack([
            artifact.mos_term_free[:, [first, second]]
            for first, second in pairs
        ], dim=1).reshape(-1, 2)
        self._gmin_incidence = _incidence(
            pair_free, nodes,
            terminal_currents=False, dtype=dtype, device=device,
        )

    def _terminal_voltage(self, free_voltage: torch.Tensor) -> torch.Tensor:
        return _gather_endpoints(
            free_voltage, self.artifact.fixed_voltage_l72,
            self.artifact.mos_term_free, self.artifact.mos_term_fixed,
        )

    @staticmethod
    def _check_support(
        raw: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        polarity: str,
        device_indices: torch.Tensor,
    ) -> None:
        outside = (raw < lower.unsqueeze(0)) | (raw > upper.unsqueeze(0))
        if bool(torch.any(outside).detach().cpu()):
            first = torch.nonzero(outside, as_tuple=False)[0]
            selected_index = int(first[1])
            device_index = int(device_indices[selected_index])
            feature = int(first[-1])
            value = float(raw[tuple(int(item) for item in first)].detach().cpu())
            raise SupportError(
                f"{polarity} device {device_index} input {feature}={value:.9g} "
                f"is outside [{float(lower[selected_index, feature]):.9g}, "
                f"{float(upper[selected_index, feature]):.9g}]")

    def _mos_kcl(
        self,
        free_voltage: torch.Tensor,
        terminal_voltage: torch.Tensor,
    ) -> torch.Tensor:
        states = free_voltage.shape[0]
        result = free_voltage.new_zeros(
            (states, len(self.artifact.free_nodes)))
        for polarity, flag in (("nmos", False), ("pmos", True)):
            selected = torch.nonzero(
                self.artifact.mos_is_pmos == flag, as_tuple=False,
            ).flatten()
            if not len(selected):
                continue
            terminal = terminal_voltage[:, selected]
            source = terminal[..., 2]
            raw = torch.stack((
                terminal[..., 0] - source,
                terminal[..., 1] - source,
                torch.zeros_like(source),
                terminal[..., 3] - source,
                torch.log2(torch.clamp(self.artifact.mos_nfin[selected], min=1.0))
                .unsqueeze(0).expand(states, -1),
                self.artifact.mos_length[selected].unsqueeze(0).expand(states, -1),
                self.artifact.mos_temperature[selected].unsqueeze(0).expand(states, -1),
            ), dim=-1)
            stats = self.stats[polarity]
            self._check_support(
                raw,
                self.artifact.mos_support_min[selected],
                self.artifact.mos_support_max[selected],
                polarity,
                selected,
            )
            scale = _stats_tensor(stats, "input_std", raw).clone()
            scale = torch.where(scale < 1e-12, torch.ones_like(scale), scale)
            normalized = (raw - _stats_tensor(stats, "input_mean", raw)) / scale
            flat = normalized.reshape(-1, 7)
            codes = self.artifact.mos_code[selected].unsqueeze(0).expand(
                states, -1).reshape(-1)
            predicted = self.models[polarity](flat, tech_codes=codes)
            if predicted.shape != (len(flat), 6):
                raise RuntimeError(
                    f"{polarity} model returned {tuple(predicted.shape)}, "
                    "expected six surfaces")
            physical = denormalize_outputs(predicted, stats).reshape(
                states, len(selected), 6)
            currents = close_terminal_currents(
                physical[..., :3], self.artifact.mos_multiplier[selected])
            result = result + torch.einsum(
                "sdt,dtn->sn", currents, self._mos_incidence[selected])
        return result

    def __call__(self, free_voltage: torch.Tensor) -> torch.Tensor:
        if free_voltage.shape != self.artifact.free_voltage_l72.shape:
            raise ValueError("free voltage shape does not match artifact states")
        terminal = self._terminal_voltage(free_voltage)
        result = self._mos_kcl(free_voltage, terminal)

        if len(self.artifact.resistor_conductance):
            endpoint = _gather_endpoints(
                free_voltage, self.artifact.fixed_voltage_l72,
                self.artifact.resistor_term_free,
                self.artifact.resistor_term_fixed,
            )
            branch = (
                (endpoint[..., 0] - endpoint[..., 1])
                * self.artifact.resistor_conductance.unsqueeze(0)
            )
            result = result + torch.einsum(
                "se,en->sn", branch, self._resistor_incidence)

        if len(self.artifact.current_term_free):
            current = self.artifact.current_value
            if current.ndim == 1:
                current = current.unsqueeze(0).expand(free_voltage.shape[0], -1)
            result = result + torch.einsum(
                "se,en->sn", current, self._current_incidence)

        pairs = ((0, 2), (0, 3), (2, 3))
        gmin_branch = torch.stack([
            self.artifact.gmin * (terminal[..., first] - terminal[..., second])
            for first, second in pairs
        ], dim=2).reshape(free_voltage.shape[0], -1)
        result = result + torch.einsum(
            "se,en->sn", gmin_branch, self._gmin_incidence)
        if not bool(torch.all(torch.isfinite(result)).detach().cpu()):
            raise RuntimeError("circuit residual produced NaN/Inf")
        return result


@dataclass(frozen=True)
class LMResult:
    voltages: torch.Tensor
    scaled_residual: torch.Tensor
    max_abs_update: torch.Tensor


def unroll_lm(
    residual: Callable[[torch.Tensor], torch.Tensor],
    initial: torch.Tensor,
    arm_scale: torch.Tensor,
    *,
    steps: int = LM_STEPS,
    lm_lambda: float = LM_LAMBDA,
    step: float = LM_STEP,
    trust_clip: float = TRUST_CLIP_V,
    create_graph: bool,
) -> LMResult:
    """Unroll a true row-scaled LM solve while preserving model gradients."""
    if steps < 1 or lm_lambda <= 0.0 or step <= 0.0 or trust_clip <= 0.0:
        raise ValueError("LM controls must be positive")
    if initial.shape != arm_scale.shape or torch.any(arm_scale <= 0):
        raise ValueError("arm_scale must be positive and match initial")
    voltage = initial.detach().clone().requires_grad_(True)
    identity = torch.eye(
        initial.shape[1], dtype=torch.float64, device=initial.device)
    updates: list[torch.Tensor] = []
    for _ in range(steps):
        scaled = residual(voltage) / arm_scale
        rows: list[torch.Tensor] = []
        for row in range(scaled.shape[1]):
            derivative = torch.autograd.grad(
                scaled[:, row].sum(), voltage,
                create_graph=create_graph, retain_graph=True,
            )[0]
            rows.append(derivative)
        jacobian = torch.stack(rows, dim=1)
        # Preserve the probed mixed-precision contract: DirectNet and its
        # voltage derivatives remain float32, while the ill-conditioned
        # normal equations are formed and solved in float64.
        jacobian64 = jacobian.to(torch.float64)
        scaled64 = scaled.to(torch.float64)
        transpose = jacobian64.transpose(1, 2)
        matrix = transpose @ jacobian64 + lm_lambda * identity
        right = transpose @ scaled64.unsqueeze(-1)
        try:
            delta64 = torch.linalg.solve(matrix, right).squeeze(-1)
        except RuntimeError as exc:
            raise RuntimeError("unrolled LM linear solve failed") from exc
        accepted = torch.clamp(
            delta64, -trust_clip, trust_clip).to(voltage.dtype)
        updates.append(torch.max(torch.abs(step * accepted)))
        voltage = voltage - step * accepted
        if not create_graph:
            voltage = voltage.detach().requires_grad_(True)
    final_scaled = residual(voltage) / arm_scale
    return LMResult(
        voltages=voltage,
        scaled_residual=final_scaled,
        max_abs_update=torch.stack(updates).max(),
    )


def circuit_weight(arm: str) -> float:
    """Return the only two registered circuit-loss arms."""
    if arm == "control":
        return 0.0
    if arm == "treatment":
        return 1.0
    raise ValueError(f"unknown arm {arm!r}; expected control or treatment")


def replay_polarities(arm: str) -> tuple[str, str]:
    """Return the fixed replay order for one control or treatment epoch."""
    circuit_weight(arm)
    return ("nmos", "pmos")


def should_save_candidate(arm: str, gate: Mapping[str, object]) -> bool:
    """Only a feasible treatment may create checkpoint artifacts."""
    circuit_weight(arm)
    return arm == "treatment" and bool(gate.get("feasible", False))


def normalized_circuit_loss(
    result: LMResult,
    artifact: CircuitArtifact,
    *,
    baseline_curve_mse: float,
    baseline_residual_mse: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the single predeclared dimensionless circuit objective."""
    vout = result.voltages[:, artifact.output_index]
    target = artifact.free_voltage_l72[:, artifact.output_index]
    curve_mse = torch.mean((vout - target) ** 2)
    residual_mse = torch.mean(result.scaled_residual ** 2)
    curve_denominator = max(baseline_curve_mse, LOSS_DENOMINATOR_FLOOR)
    residual_denominator = max(baseline_residual_mse, LOSS_DENOMINATOR_FLOOR)
    loss = (
        curve_mse / curve_denominator
        + RESIDUAL_LOSS_WEIGHT * residual_mse / residual_denominator
    )
    return loss, {"curve_mse": curve_mse, "residual_mse": residual_mse}


def evaluate_candidate_gate(
    baselines: Mapping[str, dict[str, object]],
    candidates: Mapping[str, dict[str, object]],
    *,
    baseline_vout_mae: float,
    candidate_vout_mae: float,
) -> dict[str, object]:
    """Apply both-polarity device preservation and local circuit gates."""
    failures: list[str] = []
    details: dict[str, object] = {}
    for polarity in ("nmos", "pmos"):
        device_gate = hermite.evaluate_feasibility(
            baselines[polarity], candidates[polarity],
            require_jacobian_improvement=False,
            value_mae_ratio=VALUE_MAE_RATIO,
            value_max_ratio=VALUE_MAX_RATIO,
        )
        details[polarity] = device_gate
        failures.extend(
            f"{polarity}.{failure}" for failure in device_gate["failures"])
        baseline_j = float(baselines[polarity]["current_jacobian_mae"])
        candidate_j = float(candidates[polarity]["current_jacobian_mae"])
        if not hermite._within_ratio(
            candidate_j, baseline_j, JACOBIAN_MAX_RATIO,
        ):
            failures.append(f"{polarity}.current_jacobian_mae")
    if not hermite._within_ratio(
        candidate_vout_mae, baseline_vout_mae, VOUT_MAE_RATIO,
    ):
        failures.append("local_vout_mae")
    vout_ratio = (
        candidate_vout_mae / baseline_vout_mae
        if baseline_vout_mae != 0.0
        else (1.0 if candidate_vout_mae == 0.0 else float("inf"))
    )
    return {
        "feasible": not failures,
        "failures": failures,
        "device": details,
        "local_vout_mae_ratio": vout_ratio,
        "limits": {
            "value_normalized_mae": VALUE_MAE_RATIO,
            "value_physical_max_abs": VALUE_MAX_RATIO,
            "current_jacobian_mae": JACOBIAN_MAX_RATIO,
            "local_vout_mae": VOUT_MAE_RATIO,
        },
    }


@dataclass
class _DeviceRun:
    checkpoint: Path
    model: torch.nn.Module
    stats: Any
    validation_cpu: dict[str, torch.Tensor]
    replay_cpu: dict[str, torch.Tensor]
    replay_parent_cpu: torch.Tensor
    baseline_metrics: dict[str, object]
    overlay_marker: dict[str, Any]


def _completion_path(checkpoint: Path) -> Path:
    return Path(f"{checkpoint}.complete")


def _verify_topology_provenance(
    artifact_path: Path,
    checkpoints: Mapping[str, Path],
    artifact: CircuitArtifact,
) -> dict[str, Any]:
    marker_path = Path(f"{artifact_path}.complete")
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = hermite._read_json(marker_path)
    if int(marker.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("topology marker schema_version mismatch")
    if marker.get("schema_name") != SCHEMA_NAME:
        raise ValueError("topology marker schema_name mismatch")
    if marker.get("artifact") != artifact_path.name:
        raise ValueError("topology marker names another artifact")
    if marker.get("artifact_sha256") != hermite.sha256_file(artifact_path):
        raise ValueError("topology artifact checksum mismatch")
    expected_counts = {
        "states": EXPECTED_STATES,
        "groups": EXPECTED_GROUPS,
        "devices": EXPECTED_DEVICES,
    }
    for name, expected in expected_counts.items():
        if int(marker.get(name, -1)) != expected:
            raise ValueError(f"topology marker {name} must be {expected}")
    actual_counts = {
        "states": artifact.states,
        "groups": int(artifact.group_sweep.numel()),
        "devices": int(artifact.mos_is_pmos.numel()),
    }
    for name, expected in expected_counts.items():
        if actual_counts[name] != expected:
            raise ValueError(f"topology artifact {name} must be {expected}")
    expected_sweep = torch.as_tensor(
        [0.65 + 0.005 * index for index in range(14)]
        + [0.65 - 0.005 * index for index in range(14)],
        dtype=artifact.group_sweep.dtype,
        device=artifact.group_sweep.device,
    )
    if not torch.allclose(
        artifact.group_sweep, expected_sweep, rtol=0.0, atol=1e-7,
    ):
        raise ValueError("topology artifact sweep coordinates changed")
    expected_groups = torch.arange(
        EXPECTED_GROUPS, device=artifact.state_group.device)
    if (not torch.equal(artifact.state_group, expected_groups)
            or not torch.equal(
                artifact.group_segment_offsets,
                torch.arange(
                    EXPECTED_GROUPS + 1,
                    device=artifact.group_segment_offsets.device,
                ),
            )):
        raise ValueError("topology artifact must contain one state per group")
    recorded = marker.get("parent_checkpoints")
    if not isinstance(recorded, dict):
        raise ValueError("topology marker lacks parent_checkpoints")
    for polarity, checkpoint in checkpoints.items():
        info = recorded.get(polarity)
        if not isinstance(info, dict):
            raise ValueError(f"topology marker lacks {polarity} parent")
        norm = hermite._normalization_path(checkpoint)
        completion = _completion_path(checkpoint)
        expected = {
            "checkpoint_sha256": hermite.sha256_file(checkpoint),
            "normalization_sha256": hermite.sha256_file(norm),
            "completion_sha256": hermite.sha256_file(completion),
        }
        for name, value in expected.items():
            if info.get(name) != value:
                raise ValueError(
                    f"topology {polarity} provenance mismatch for {name}")
    return marker


def _batched_parent_predictions(
    model: torch.nn.Module,
    replay: Mapping[str, torch.Tensor],
    *,
    batch_size: int = 8192,
) -> torch.Tensor:
    model.eval()
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(replay["x"]), batch_size):
            stop = min(start + batch_size, len(replay["x"]))
            parts.append(model(
                replay["x"][start:stop],
                tech_codes=replay["code"][start:stop],
            ).detach().cpu())
    return torch.cat(parts, dim=0)


def _prepare_device(
    polarity: str,
    checkpoint: Path,
    overlay: Path,
    replay_path: Path,
) -> _DeviceRun:
    loaded, stats, _num_codes, output_columns = _load_artifacts(checkpoint)
    if tuple(output_columns) != tuple(hermite.OUTPUT_COLUMNS):
        raise ValueError(f"{polarity} parent is not the six-surface contract")
    model = copy.deepcopy(loaded).cpu().eval()
    _active, validation, replay, overlay_marker = hermite._load_data(
        overlay, replay_path, checkpoint, polarity,
        stats, torch.device("cpu"),
    )
    baseline = hermite._model_metrics(
        model, validation, replay, stats,
        value_batch_size=8192, jacobian_batch_size=512,
    )
    if len(replay["x"]) != REPLAY_ROWS:
        raise ValueError(
            f"{polarity} replay must contain exactly {REPLAY_ROWS} rows")
    parent = _batched_parent_predictions(model, replay)
    return _DeviceRun(
        checkpoint=checkpoint,
        model=model,
        stats=stats,
        validation_cpu=validation,
        replay_cpu=replay,
        replay_parent_cpu=parent,
        baseline_metrics=baseline,
        overlay_marker=overlay_marker,
    )


def _to_device(
    values: Mapping[str, torch.Tensor], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in values.items()}


def _circuit_metrics(
    artifact: CircuitArtifact,
    models: Mapping[str, torch.nn.Module],
    stats: Mapping[str, Any],
    *,
    create_graph: bool,
) -> tuple[LMResult, dict[str, float]]:
    for model in models.values():
        model.eval()
    result = unroll_lm(
        FullTerminalResidual(artifact, models=models, stats=stats),
        artifact.free_voltage_l72,
        artifact.arm_scale,
        steps=LM_STEPS,
        lm_lambda=LM_LAMBDA,
        step=LM_STEP,
        trust_clip=TRUST_CLIP_V,
        create_graph=create_graph,
    )
    vout = result.voltages[:, artifact.output_index]
    target = artifact.free_voltage_l72[:, artifact.output_index]
    metrics = {
        "vout_mae": float(torch.mean(torch.abs(vout - target)).detach().cpu()),
        "curve_mse": float(torch.mean((vout - target) ** 2).detach().cpu()),
        "residual_mse": float(
            torch.mean(result.scaled_residual ** 2).detach().cpu()),
        "residual_rms": float(
            torch.sqrt(torch.mean(result.scaled_residual ** 2)).detach().cpu()),
        "max_abs_update": float(result.max_abs_update.detach().cpu()),
    }
    return result, metrics


def _candidate_metrics_cpu(
    runs: Mapping[str, _DeviceRun],
    models: Mapping[str, torch.nn.Module],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for polarity in ("nmos", "pmos"):
        model = models[polarity].cpu().eval()
        run = runs[polarity]
        result[polarity] = hermite._model_metrics(
            model, run.validation_cpu, run.replay_cpu, run.stats,
            value_batch_size=8192, jacobian_batch_size=512,
        )
    return result


def _stem(
    checkpoint: Path,
    polarity: str,
    requested: str | None,
    arm: str,
) -> str:
    if polarity not in {"nmos", "pmos"}:
        raise ValueError(f"unknown checkpoint polarity {polarity!r}")
    suffix = f"_{polarity}_best.pt"
    if not checkpoint.name.endswith(suffix):
        raise ValueError(
            f"{polarity} checkpoint must end in {suffix}: {checkpoint.name}")
    if requested:
        if not requested.endswith(f"_{polarity}"):
            raise ValueError(
                f"{polarity} stem must end in _{polarity}: {requested!r}")
        return requested
    parent = checkpoint.name.removesuffix(suffix)
    return f"{parent}_unrolled_{arm}_{polarity}"


def _schedule_record() -> dict[str, float | int]:
    return {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "replay_batch_size": REPLAY_BATCH_SIZE,
        "gradient_clip": GRADIENT_CLIP,
        "lm_steps": LM_STEPS,
        "lm_lambda": LM_LAMBDA,
        "lm_step": LM_STEP,
        "trust_clip_v": TRUST_CLIP_V,
        "residual_loss_weight": RESIDUAL_LOSS_WEIGHT,
    }


def _save_candidate_pair(
    models: Mapping[str, torch.nn.Module],
    runs: Mapping[str, _DeviceRun],
    args: argparse.Namespace,
    *,
    epoch: int,
    source_commit: str,
    runtime: Mapping[str, object],
    topology_marker: Mapping[str, Any],
    device_metrics: Mapping[str, dict[str, object]],
    circuit_metrics: Mapping[str, float],
    gate: Mapping[str, object],
) -> dict[str, str]:
    epoch_dir = args.output_dir / f"epoch_{epoch:02d}"
    staging_dir = args.output_dir / f".epoch_{epoch:02d}.tmp-{os.getpid()}"
    backup_dir = args.output_dir / f".epoch_{epoch:02d}.bak-{os.getpid()}"
    names = {
        "nmos": _stem(
            args.nmos_checkpoint, "nmos", args.nmos_stem, args.arm),
        "pmos": _stem(
            args.pmos_checkpoint, "pmos", args.pmos_stem, args.arm),
    }
    paths = {
        polarity: staging_dir / f"{names[polarity]}_best.pt"
        for polarity in ("nmos", "pmos")
    }
    if epoch_dir.exists() and not args.overwrite:
        raise FileExistsError(epoch_dir)
    if staging_dir.exists() or backup_dir.exists():
        raise FileExistsError(staging_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()

    try:
        for polarity, checkpoint in paths.items():
            state = {
                name: value.detach().cpu()
                for name, value in models[polarity].state_dict().items()
            }
            torch.save(state, checkpoint)
            norm = staging_dir / f"{names[polarity]}_norm.npz"
            parent_norm = hermite._normalization_path(runs[polarity].checkpoint)
            shutil.copy2(parent_norm, norm)
            if hermite.sha256_file(norm) != hermite.sha256_file(parent_norm):
                raise RuntimeError("saved normalizer differs from parent")

        trainer_sha256 = hermite.sha256_file(Path(__file__))
        for polarity, checkpoint in paths.items():
            norm = staging_dir / f"{names[polarity]}_norm.npz"
            parent_norm = hermite._normalization_path(runs[polarity].checkpoint)
            other = "pmos" if polarity == "nmos" else "nmos"
            marker = {
                "family": "directnet-full",
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": hermite.sha256_file(checkpoint),
                "normalization": norm.name,
                "normalization_sha256": hermite.sha256_file(norm),
                "output_columns": list(hermite.OUTPUT_COLUMNS),
                "source_commit": source_commit,
                "trainer_sha256": trainer_sha256,
                "seed": args.seed,
                "runtime": dict(runtime),
                "parent_checkpoint_sha256": hermite.sha256_file(
                    runs[polarity].checkpoint),
                "parent_normalization_sha256": hermite.sha256_file(
                    parent_norm),
                "parent_completion_sha256": hermite.sha256_file(
                    _completion_path(runs[polarity].checkpoint)),
                "hermite_overlay": runs[polarity].overlay_marker["artifact"],
                "hermite_overlay_sha256": runs[polarity].overlay_marker[
                    "artifact_sha256"],
                "replay_dataset_sha256": runs[polarity].overlay_marker[
                    "dataset_sha256"],
                "replay_rows": REPLAY_ROWS,
                "companion_polarity": other,
                "companion_checkpoint": paths[other].name,
                "topology_artifact": args.artifact.name,
                "topology_artifact_sha256": topology_marker[
                    "artifact_sha256"],
                "arm": args.arm,
                "epoch": epoch,
                "schedule": _schedule_record(),
                "device_metrics": device_metrics,
                "circuit_metrics": circuit_metrics,
                "feasibility": gate,
            }
            _completion_path(checkpoint).write_text(
                json.dumps(marker, sort_keys=True, indent=2) + "\n")

        if epoch_dir.exists():
            os.replace(epoch_dir, backup_dir)
        try:
            os.replace(staging_dir, epoch_dir)
        except BaseException:
            if backup_dir.exists() and not epoch_dir.exists():
                os.replace(backup_dir, epoch_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return {
        polarity: str(epoch_dir / path.name)
        for polarity, path in paths.items()
    }


def _write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")


def finetune(args: argparse.Namespace) -> dict[str, object]:
    """Execute one registered control or treatment arm."""
    args.artifact = args.artifact.resolve()
    args.nmos_checkpoint = args.nmos_checkpoint.resolve()
    args.pmos_checkpoint = args.pmos_checkpoint.resolve()
    args.nmos_overlay = args.nmos_overlay.resolve()
    args.pmos_overlay = args.pmos_overlay.resolve()
    args.nmos_replay = args.nmos_replay.resolve()
    args.pmos_replay = args.pmos_replay.resolve()
    args.output_dir = args.output_dir.resolve()
    summary_path = args.output_dir / f"unrolled_{args.arm}_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    weight = circuit_weight(args.arm)
    source_commit, dirty = hermite._source_identity()
    if dirty:
        raise RuntimeError("unrolled trainer source has tracked changes")
    checkpoints = {
        "nmos": args.nmos_checkpoint,
        "pmos": args.pmos_checkpoint,
    }
    for checkpoint in checkpoints.values():
        for path in (
            checkpoint, hermite._normalization_path(checkpoint),
            _completion_path(checkpoint),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
    artifact_cpu = CircuitArtifact.load(
        args.artifact, device=torch.device("cpu"), dtype=torch.float32)
    topology_marker = _verify_topology_provenance(
        args.artifact, checkpoints, artifact_cpu)

    scored_runtime = hermite._scored_runtime_contract("cpu", 1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    train_device = torch.device(args.device)
    if train_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    if train_device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    runtime: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "training_device": str(train_device),
        "gate_device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "scored_runtime": scored_runtime,
    }
    if train_device.type == "cuda":
        runtime["cuda_device"] = torch.cuda.get_device_name(train_device)

    runs = {
        "nmos": _prepare_device(
            "nmos", args.nmos_checkpoint,
            args.nmos_overlay, args.nmos_replay),
        "pmos": _prepare_device(
            "pmos", args.pmos_checkpoint,
            args.pmos_overlay, args.pmos_replay),
    }
    parent_models = {name: run.model for name, run in runs.items()}
    stats = {name: run.stats for name, run in runs.items()}
    _baseline_result, baseline_circuit = _circuit_metrics(
        artifact_cpu, parent_models, stats, create_graph=False)
    baselines = {
        name: run.baseline_metrics for name, run in runs.items()}

    models = {
        name: copy.deepcopy(run.model).to(train_device)
        for name, run in runs.items()
    }
    replay_train = {
        name: _to_device(run.replay_cpu, train_device)
        for name, run in runs.items()
    }
    parent_targets = {
        name: run.replay_parent_cpu.to(train_device)
        for name, run in runs.items()
    }
    artifact_train = CircuitArtifact.load(
        args.artifact, device=train_device, dtype=torch.float32)
    parameters = [
        parameter
        for model in models.values()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=0.0,
        fused=train_device.type == "cuda",
    )
    history: list[dict[str, object]] = []
    saved: list[dict[str, str]] = []

    for epoch in range(1, EPOCHS + 1):
        replay_losses: dict[str, float] = {}
        for polarity in replay_polarities(args.arm):
            model = models[polarity]
            data = replay_train[polarity]
            model.train(mode=bool(weight))
            order = hermite.epoch_order(
                len(data["x"]), seed=args.seed + (0 if polarity == "nmos" else 1),
                epoch=epoch,
            )
            total = 0.0
            seen = 0
            for indices_np in hermite._batches(order, REPLAY_BATCH_SIZE):
                indices = torch.as_tensor(indices_np, device=train_device)
                prediction = model(
                    data["x"][indices], tech_codes=data["code"][indices])
                loss = torch.mean(torch.abs(
                    prediction - parent_targets[polarity][indices]))
                if weight:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
                    optimizer.step()
                total += float(loss.detach()) * len(indices_np)
                seen += len(indices_np)
            if seen != len(data["x"]):
                raise RuntimeError(f"{polarity} replay sweep was incomplete")
            replay_losses[polarity] = total / seen

        if not weight:
            for polarity, model in models.items():
                parent_state = runs[polarity].model.state_dict()
                for name, value in model.state_dict().items():
                    if not torch.equal(value.detach().cpu(), parent_state[name]):
                        raise RuntimeError(
                            f"control changed {polarity} parameter {name}")

        circuit_training: dict[str, float] | None = None
        if weight:
            result, raw_metrics = _circuit_metrics(
                artifact_train, models, stats, create_graph=True)
            loss, components = normalized_circuit_loss(
                result, artifact_train,
                baseline_curve_mse=baseline_circuit["curve_mse"],
                baseline_residual_mse=baseline_circuit["residual_mse"],
            )
            optimizer.zero_grad(set_to_none=True)
            (weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            circuit_training = {
                **raw_metrics,
                "loss": float(loss.detach().cpu()),
                "curve_component": float(components["curve_mse"].detach().cpu()),
                "residual_component": float(
                    components["residual_mse"].detach().cpu()),
            }

        # The scored candidate gates are always evaluated on one-thread CPU.
        candidate_models = {
            name: copy.deepcopy(model).cpu().eval()
            for name, model in models.items()
        }
        candidate_device = _candidate_metrics_cpu(runs, candidate_models)
        _candidate_result, candidate_circuit = _circuit_metrics(
            artifact_cpu, candidate_models, stats, create_graph=False)
        gate = evaluate_candidate_gate(
            baselines, candidate_device,
            baseline_vout_mae=baseline_circuit["vout_mae"],
            candidate_vout_mae=candidate_circuit["vout_mae"],
        )
        epoch_row: dict[str, object] = {
            "epoch": epoch,
            "replay_distillation_loss": replay_losses,
            "circuit_training": circuit_training,
            "device_metrics": candidate_device,
            "circuit_metrics": candidate_circuit,
            "gate": gate,
        }
        if should_save_candidate(args.arm, gate):
            checkpoint_paths = _save_candidate_pair(
                candidate_models, runs, args,
                epoch=epoch, source_commit=source_commit, runtime=runtime,
                topology_marker=topology_marker,
                device_metrics=candidate_device,
                circuit_metrics=candidate_circuit,
                gate=gate,
            )
            epoch_row["checkpoints"] = checkpoint_paths
            saved.append(checkpoint_paths)
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)

    summary: dict[str, object] = {
        "source_commit": source_commit,
        "trainer_sha256": hermite.sha256_file(Path(__file__)),
        "seed": args.seed,
        "runtime": runtime,
        "arm": args.arm,
        "artifact": args.artifact.name,
        "artifact_sha256": topology_marker["artifact_sha256"],
        "parents": {
            name: {
                "checkpoint": str(run.checkpoint),
                "checkpoint_sha256": hermite.sha256_file(run.checkpoint),
                "normalization_sha256": hermite.sha256_file(
                    hermite._normalization_path(run.checkpoint)),
                "overlay_sha256": run.overlay_marker["artifact_sha256"],
                "replay_dataset_sha256": run.overlay_marker["dataset_sha256"],
                "replay_rows": REPLAY_ROWS,
            }
            for name, run in runs.items()
        },
        "schedule": _schedule_record(),
        "baseline_device_metrics": baselines,
        "baseline_circuit_metrics": baseline_circuit,
        "history": history,
        "saved_checkpoints": saved,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune DirectNet-Full with one bounded unrolled LDO arm")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--nmos-checkpoint", type=Path, required=True)
    parser.add_argument("--pmos-checkpoint", type=Path, required=True)
    parser.add_argument("--nmos-overlay", type=Path, required=True)
    parser.add_argument("--pmos-overlay", type=Path, required=True)
    parser.add_argument("--nmos-replay", type=Path, required=True)
    parser.add_argument("--pmos-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nmos-stem")
    parser.add_argument("--pmos-stem")
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=768)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    finetune(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
