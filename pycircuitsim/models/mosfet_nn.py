"""Shared base class for NN-based MOSFET compact models.

Used by ``mosfet_directnet`` (LEVEL=73, MLP) and ``mosfet_bsimar``
(LEVEL=74, autoregressive Transformer). Both models share:

* terminal voltage prep (PMOS source-shift + softplus clamp + z-score)
* a 1-sample autograd pass that gives id/qg/qd plus their Jacobians
* normalised → physical chain rule (delegated to the normalizer)
* analytical Vds correction including rail-restoring extrapolation
* charge state + caching used by the transient solver

Subclasses provide:

1. ``model_factory(state)`` returning the un-loaded ``nn.Module``.
2. ``output_layout`` selecting how columns are read from the model
   output: ``"standard"`` reads ``OUTPUT_COLUMN_ORDER`` directly
   (DirectNet); ``"bsimar"`` permutes from ``BSIMAR_COLUMN_ORDER``
   back to ``OUTPUT_COLUMN_ORDER`` (Transformer).
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Make the neural compact-model package importable regardless of cwd.
PROJECT_ROOT = Path(__file__).parent.parent.parent
_NN_PARENT = PROJECT_ROOT / "external_compact_models"
if str(_NN_PARENT) not in sys.path:
    sys.path.insert(0, str(_NN_PARENT))

from pycircuitsim.models.base import Component
from neural_network.config import UNKNOWN_CODE_ID
from neural_network.data.normalize import (
    NormStats, normalizer_from_stats,
    OUTPUT_COLUMN_ORDER, BSIMAR_COLUMN_ORDER,
)


_logger = logging.getLogger(__name__)
_NN_DEVICE: Optional[torch.device] = None

# V7.0.3 — opt-in closed-form Jacobian (DirectNet only; see
# ``DirectNet.forward_with_jacobian``). Replaces 1 forward + 3 backward
# weight streams with a single fused pass: 1610 us -> 748 us per eval at
# DirectNet-large / batch 1.
#
# DEFAULT OFF, and it must stay off until a full 16-gate complex re-gate
# clears it. The mathematics is identical but the summation order is not
# (values ~5e-7 abs, d(id)/dV ~4e-6 relative), and this repo has repeatedly
# seen a high-gain circuit turn a last-bit NN perturbation into a different
# NR basin — the same reason ``NN_BATCHED_EVAL=0`` exists in solver.py.
_FUSED_JAC = os.environ.get("PYCIRCUITSIM_NN_FUSED_JAC", "0") == "1"

# gds negative-branch guard coefficient (audit 2026-07-21 §A3-guard, "guard F").
# Only ever applied to a *negative* gds candidate: k = 0.02 S/A == 1/(50 V), and
# 50 V sits above the 43.4 V maximum true Early voltage measured anywhere in the
# training box, so this can never bind on a physically correct value. It is a
# physical bound, not a tuning knob — env-overridable for diagnostics only.
#
# History: the shipped V6.4.8 form was a two-sided floor max(gds, |id|·0.5), i.e.
# an asserted V_A ≤ 2 V, below the true median Early voltage of every device. It
# overrode the learned output conductance at 90.9% of amplifying points and was
# load-bearing only because it masked the gds sign bug below (audit §A3-measured
# arm D). Its docstring carried a pre-registered prediction that floor-k is
# "Jacobian-only … NOT a shipping accuracy lever" — true for DC, falsified for AC.
try:
    _GDS_GUARD_K = float(os.environ.get("PYCIRCUITSIM_GDS_GUARD_K", "0.02"))
except (TypeError, ValueError):
    _GDS_GUARD_K = 0.02

if "PYCIRCUITSIM_GDS_FLOOR_K" in os.environ:
    # Loud, not silent: the old knob tuned a two-sided floor that no longer
    # exists, so honouring it would mean something different than its setter
    # intended (audit §B6 silent-green class).
    _logger.warning(
        "PYCIRCUITSIM_GDS_FLOOR_K is obsolete and IGNORED — the two-sided gds "
        "floor was replaced by the negative-only guard F. Use "
        "PYCIRCUITSIM_GDS_GUARD_K to vary the guard coefficient.")

# V6.5.2 reverse-conduction taper window (fractions of VDD_train). Default
# 0.20/0.30 == the S7-bisected committed window; env-overridable for the
# tg_corridor-retrain A/B (see ``_reverse_taper``). Unset → unchanged.
try:
    _REV_TAPER_X0 = float(os.environ.get("PYCIRCUITSIM_REV_TAPER_X0", "0.20"))
    _REV_TAPER_X1 = float(os.environ.get("PYCIRCUITSIM_REV_TAPER_X1", "0.30"))
except (TypeError, ValueError):
    _REV_TAPER_X0, _REV_TAPER_X1 = 0.20, 0.30

# Process-level shared-module cache. Devices that load the *same*
# checkpoint file share one ``nn.Module`` instance so the Phase-5
# batched path (``batch_eval``) can group them into a single stacked
# forward. The module is held in ``eval()`` mode and its forward is
# purely functional — weights are immutable — so sharing is safe and
# bit-identical to per-device instantiation. Keyed on (abs path, mtime,
# size) so a re-trained checkpoint at the same path is not aliased.
_SHARED_NN_MODULES: Dict[Tuple[str, int, int], torch.nn.Module] = {}

# V7.2.0 Phase 1a — shared ``NormStats`` per norm.npz, same key scheme as
# ``_SHARED_NN_MODULES``. The stats arrays are read-only by contract: every
# simulator-side consumer copies before mutating (``_setup_gpu``,
# ``_geo_norm``), so N devices sharing one object is bit-identical to N
# private loads of the same file.
_SHARED_NORM_STATS: Dict[Tuple[str, int, int], NormStats] = {}

# V7.2.0 Phase 1c — ``Path.resolve()`` walks the filesystem (realpath
# syscalls) and was called twice per device to build the cache keys:
# ~1.2 s of a 32x32 parse. The str->resolved mapping is stable within a
# process; staleness is caught by the (mtime, size) part of the keys.
_RESOLVED_PATH_CACHE: Dict[str, str] = {}

# V7.2.0 Phase 2c — shared per-(norm file, device) inference tensors.
# ``_setup_gpu`` used to allocate 6 identical small tensors per device
# instance (~43k redundant allocations at 6144 devices). All are
# read-only inputs to elementwise ops, so sharing is bit-identical.
_SHARED_NORM_TENSORS: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}
_SHARED_GEO_TENSORS: Dict[Tuple, torch.Tensor] = {}
_SHARED_CODE_TENSORS: Dict[Tuple, torch.Tensor] = {}

# V7.2.0 Phase 2c — per-model cache of the stacked (N, 3) geometry and
# (N,) tech-code tensors ``batch_eval`` rebuilt from Python lists every
# NR iteration (plan §3.3b). Keyed on the row VALUES (NFIN, L, T,
# tech_code per member, in order), not on device identity: group
# membership varies per iteration (warm-cached devices are skipped), and
# a value key makes any same-shaped membership share one entry — on a
# geometry-uniform SRAM array every subset of size N collapses to a
# single key. The stats behind ``_geo_norm_t`` are fixed per model
# (checkpoint and norm file are 1:1), so the key fully determines the
# rows and a hit is bit-identical to a rebuild.
_STACK_CACHE_CAP = 64


def _stacked_group_inputs(
    model: torch.nn.Module, devs: List["_MOSFETNNBase"],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    # V7.2.0 Phase 2a-full: the cached value also carries the per-member
    # ``_is_pmos`` flags (as a bool ndarray) for the batched denorm tail.
    # Groups are keyed on model identity, and in practice one checkpoint
    # serves one device type — but an env double-pin CAN alias NMOS and
    # PMOS onto one module, so the tail takes the flags per member rather
    # than assuming uniformity. Read-only, like the tensors.
    key = tuple(
        (m.NFIN, m.L, m.temperature, m._tech_code, m._is_pmos)
        for m in devs)
    cache = getattr(model, "_pcs_stack_cache", None)
    if cache is None:
        cache = {}
        model._pcs_stack_cache = cache
    hit = cache.get(key)
    if hit is None:
        if len(cache) >= _STACK_CACHE_CAP:
            cache.clear()
        hit = (
            torch.stack([m._geo_norm_t for m in devs], dim=0),
            torch.cat([m._tech_code_tensor for m in devs], dim=0),
            np.fromiter(
                (m._is_pmos for m in devs), dtype=bool, count=len(devs)),
        )
        cache[key] = hit
    return hit


def _resolve_path_cached(path: Path) -> str:
    s = str(path)
    r = _RESOLVED_PATH_CACHE.get(s)
    if r is None:
        r = str(path.resolve())
        _RESOLVED_PATH_CACHE[s] = r
    return r


def _get_nn_device() -> torch.device:
    """Resolve the NN eval device once per process (V7.2.0 Phase 1b).

    Default **CPU**, even when CUDA is visible. Before V7.2.0 this
    silently picked CUDA whenever it was available, so the *same command
    line* produced different floats (and potentially a different NR basin
    in a bistable circuit) depending on which box it ran on — a
    provenance bug, since every gate result is defined as a CPU property.
    GPU eval is now an explicit opt-in: ``PYCIRCUITSIM_NN_DEVICE=cuda``
    (or ``cuda:N``). A requested-but-unavailable CUDA device raises —
    no silent fallback (audit §B6 silent-green class).
    """
    global _NN_DEVICE
    if _NN_DEVICE is None:
        want = os.environ.get("PYCIRCUITSIM_NN_DEVICE", "cpu").strip().lower()
        if want in ("", "cpu"):
            _NN_DEVICE = torch.device("cpu")
        elif want.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"PYCIRCUITSIM_NN_DEVICE={want} requested but CUDA is "
                    "unavailable; refusing silent CPU fallback")
            # §8.4 T0 determinism pins, enforced by the runtime rather than
            # left to the gate harness: TF32 would silently trade the
            # mantissa bits the T1/T4 tiers measure, and nondeterministic
            # kernels would make "same hardware, same result" false. Set
            # BEFORE the first cuBLAS call; asserted so a torch upgrade
            # that renames a knob fails loud, not silently fast-and-loose.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert torch.are_deterministic_algorithms_enabled()
            _NN_DEVICE = torch.device(want)
        else:
            raise ValueError(
                f"PYCIRCUITSIM_NN_DEVICE={want!r} not recognised "
                "(use 'cpu', 'cuda' or 'cuda:N')")
        print(f"[NN-device] NN eval device: {_NN_DEVICE}")
    return _NN_DEVICE


# Column indices into OUTPUT_COLUMN_ORDER (the canonical order the
# normalizer's stats are stored in).
_OC = {n: i for i, n in enumerate(OUTPUT_COLUMN_ORDER)}
_OC_ID = _OC["id"]
_OC_QG = _OC["qg"]
_OC_QD = _OC["qd"]
_OC_QB = _OC["qb"]


class _MOSFETNNBase(Component):
    """Shared NN-MOSFET implementation used by LEVEL=73 and LEVEL=74."""

    # Subclasses (DirectNet / BSIMAR) set these.
    _is_pmos: bool = False
    _output_layout: str = "standard"   # "standard" or "bsimar"

    # V7.0.1 — does this analysis consume capacitances?
    #
    # The 5 caps are the ONLY consumers of the qg / qd autograd Jacobians,
    # and they are read in exactly two places: ``TransientSolver`` and
    # ``ACSolver``. A ``.dc`` / ``.op`` run computed both backward passes
    # and threw them away — two thirds of the eval cost (measured: 1610 us
    # -> 784 us per eval at DirectNet-large, batch 1).
    #
    # Default False so a pure DC run is fast without any solver having to
    # opt in; the two cap consumers call ``require_caps`` at construction.
    # ``get_capacitances`` self-heals if anything else asks for caps on a
    # DC-warmed cache, so the flag is a performance hint, never a
    # correctness precondition.
    _caps_required: bool = False

    def __init__(
        self,
        name: str,
        nodes: List[str],
        model_path: str,
        L: float,
        NFIN: float,
        temperature: float = 300.15,
        tech_code: Optional[int] = None,
        *,
        model_factory: Optional[
            Callable[[Dict[str, torch.Tensor]], torch.nn.Module]
        ] = None,
        output_layout: str = "standard",
    ) -> None:
        super().__init__(name, nodes, None)
        if len(nodes) != 4:
            raise ValueError(
                f"NN MOSFET must have 4 nodes, got {len(nodes)}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")
        if NFIN <= 0:
            raise ValueError(f"NFIN must be positive, got {NFIN}")

        self.L = float(L)
        self.NFIN = float(NFIN)
        self.temperature = float(temperature)
        self._output_layout = output_layout

        # ── Resolve the checkpoint, norm.npz, and arch config ─────────
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"NN model not found: {model_path_obj}")

        base_stem = model_path_obj.stem
        for sfx in (".phys", ".ar"):
            if base_stem.endswith(sfx):
                base_stem = base_stem[: -len(sfx)]
                break
        norm_path = model_path_obj.parent / (
            base_stem.replace("_best", "_norm") + ".npz")
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Norm stats not found: {norm_path}")

        # ── Build the model (subclass-supplied) and load weights ──────
        # Reuse a shared module for identical checkpoints so the Phase-5
        # batched path can stack devices that load the same .pt into one
        # forward. The module is stateless in eval() mode → safe to share.
        #
        # V7.2.0 Phase 1a: consult the cache BEFORE ``torch.load``. The
        # old order deserialised the checkpoint unconditionally and threw
        # it away on a hit — 55 % of parse wall at array sizes (one
        # ``torch.load`` per device; ~22 GB of redundant unpickling at
        # 6144 devices). Parse is now O(devices + checkpoints).
        assert model_factory is not None, (
            "_MOSFETNNBase requires model_factory")
        st = model_path_obj.stat()
        self._model_key: Tuple[str, int, int] = (
            _resolve_path_cached(model_path_obj),
            int(st.st_mtime_ns), int(st.st_size))
        cached = _SHARED_NN_MODULES.get(self._model_key)
        if cached is not None:
            self._nn_model = cached
        else:
            state = torch.load(
                str(model_path_obj), weights_only=True, map_location="cpu")
            self._nn_model = model_factory(state)
            self._nn_model.load_state_dict(state)
            self._nn_model.eval()
            # V7.2.0 Phase 1c: move the module once, when it is built.
            # ``_setup_gpu`` used to call ``model.to(device)`` per DEVICE
            # INSTANCE — a no-op data-wise on a warm module, but the
            # module-tree ``_apply`` walk alone was ~2.6 s of a 32x32
            # parse (6,144 walks over the same 18 modules).
            self._nn_model.to(_get_nn_device())
            _SHARED_NN_MODULES[self._model_key] = self._nn_model

        # ── Norm stats + normalizer ───────────────────────────────────
        # V7.2.0 Phase 1a: shared per-file (was one ``NormStats.load``
        # per device, ~8 s of parse at 6144 devices).
        nst = norm_path.stat()
        self._norm_key: Tuple[str, int, int] = (
            _resolve_path_cached(norm_path),
            int(nst.st_mtime_ns), int(nst.st_size))
        stats = _SHARED_NORM_STATS.get(self._norm_key)
        if stats is None:
            stats = NormStats.load(str(norm_path))
            _SHARED_NORM_STATS[self._norm_key] = stats
        self._norm_stats: NormStats = stats
        self._normalizer = normalizer_from_stats(self._norm_stats)

        # ── Tech code ────────────────────────────────────────────────
        self._tech_code = (
            tech_code if tech_code is not None else UNKNOWN_CODE_ID)
        self._tech_code_tensor = torch.tensor(
            [self._tech_code], dtype=torch.long)

        # ── Pre-compute normalised geometry (constant per device) ────
        self._refresh_geometry()

        # Derive the model-output → column-name lookup. Three cases:
        #
        # (1) Transformer (BSIMAR layout): the MODEL emits
        #     BSIMAR_COLUMN_ORDER regardless of what the norm stats carry —
        #     modern norm.npz files store ``output_columns`` in CANONICAL
        #     order (they describe the stats arrays, which are fitted before
        #     the trainer's BSIMAR reorder). This branch must win over (2):
        #     until V6.8 it did not, so a LEVEL=74 checkpoint whose norm file
        #     carried ``output_columns`` had every output misread (qg denormed
        #     as id, ~5x current error) while ``_stats_col`` stayed correct.
        # (2) E2 4-output head (DirectNet): norm.npz declares a SUBSET
        #     ``output_columns`` (e.g. ["id", "qg", "qd", "qb"]) matching the
        #     model head. Map names to that subset's indices.
        # (3) Standard 13-output DirectNet: OUTPUT_COLUMN_ORDER.
        if self._output_layout == "bsimar":
            self._out_col = {
                n: BSIMAR_COLUMN_ORDER.index(n) for n in OUTPUT_COLUMN_ORDER
            }
        elif self._norm_stats.output_columns is not None:
            cols = self._norm_stats.output_columns
            self._out_col = {n: cols.index(n) for n in cols}
        else:
            self._out_col = {
                n: OUTPUT_COLUMN_ORDER.index(n) for n in OUTPUT_COLUMN_ORDER
            }

        # ── VDD estimate from the training-domain box ────────────────
        vd_range = max(
            abs(float(self._norm_stats.input_max[0])),
            abs(float(self._norm_stats.input_min[0])))
        self._vdd_estimate = vd_range / 2.0

        # ── Pre-resolved denorm constants (V7.0.1) ───────────────────
        # ``_stats_col`` used to do a ``list.index`` per scalar, 13x per
        # eval, to rediscover indices fixed at construction. Resolve them
        # once here. The arithmetic in ``_denorm`` / ``_denorm_deriv`` is
        # left in exactly its original order — only the lookups moved.
        stats_cols = self._norm_stats.output_columns or OUTPUT_COLUMN_ORDER
        self._stats_idx: Dict[str, int] = {
            n: stats_cols.index(n) for n in stats_cols}
        s = self._norm_stats
        self._out_std_f: Dict[str, float] = {
            n: float(s.output_std[i]) for n, i in self._stats_idx.items()}
        self._out_mean_f: Dict[str, float] = {
            n: float(s.output_mean[i]) for n, i in self._stats_idx.items()}
        self._asinh_f: Dict[str, float] = (
            {n: float(s.asinh_scale[i]) for n, i in self._stats_idx.items()}
            if s.mode == "asinh" and s.asinh_scale is not None else {})
        self._in_std_f: List[float] = [float(v) for v in s.input_std[:4]]

        # ── Cache + transient state ──────────────────────────────────
        self._eval_cache: Optional[Dict[str, float]] = None
        self._cache_voltages: Optional[Tuple[float, ...]] = None
        # V7.0.1: whether the cached result carries the capacitance block.
        self._cache_has_caps: bool = False
        self._q_prev: Optional[Dict[str, float]] = None
        self._q_prev2: Optional[Dict[str, float]] = None
        self._v_prev_tran: Optional[Dict[str, float]] = None
        self._i_prev_gate: float = 0.0
        self._i_prev_drain: float = 0.0

        self._setup_gpu()

    # ── GPU setup ─────────────────────────────────────────────────────

    def _setup_gpu(self) -> None:
        self._device = _get_nn_device()
        # The shared module was already moved to this device when it was
        # built (Phase 1c) — the device is a process singleton, so no
        # per-instance ``model.to`` is needed.
        dev_key = str(self._device)

        # V7.2.0 Phase 2c: all constant tensors below are shared per
        # distinct value key — they used to be allocated per device
        # instance. Read-only by contract (elementwise-op inputs only).
        code_key = (self._tech_code, dev_key)
        code_t = _SHARED_CODE_TENSORS.get(code_key)
        if code_t is None:
            code_t = self._tech_code_tensor.to(self._device)
            _SHARED_CODE_TENSORS[code_key] = code_t
        self._tech_code_tensor = code_t

        self._bind_geometry_tensor()

        s = self._norm_stats
        norm_key = (self._norm_key, dev_key)
        shared = _SHARED_NORM_TENSORS.get(norm_key)
        if shared is None:
            v_std = s.input_std[:4].copy()
            v_std[v_std < 1e-12] = 1.0
            v_mean = torch.tensor(
                s.input_mean[:4], dtype=torch.float32, device=self._device)
            v_std_t = torch.tensor(
                v_std, dtype=torch.float32, device=self._device)
            v_min = torch.tensor(
                s.input_min[:4], dtype=torch.float32, device=self._device)
            v_max = torch.tensor(
                s.input_max[:4], dtype=torch.float32, device=self._device)
            v_range = torch.clamp(v_max - v_min, min=0.01)
            # Smooth-clamp sharpness; margin = 5% of per-dim training range
            clamp_beta = (1.0 / (0.05 * v_range)).to(self._device)
            shared = (v_mean, v_std_t, v_min, v_max, clamp_beta)
            _SHARED_NORM_TENSORS[norm_key] = shared
        (self._v_mean, self._v_std_t, self._v_min, self._v_max,
         self._clamp_beta) = shared

    # ── Voltage prep: source shift + smooth clamp + z-score ──────────

    def _raw_voltages(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float, float]:
        """Terminal voltages in the NN frame (source-referenced, Vs ≡ 0).

        Training data is generated exclusively at Vs=0, so BOTH device
        types must be source-shifted; shift invariance makes this exact.
        """
        v_d = voltages.get(self.nodes[0], 0.0)
        v_g = voltages.get(self.nodes[1], 0.0)
        v_s = voltages.get(self.nodes[2], 0.0)
        v_b = voltages.get(self.nodes[3], 0.0)

        return v_d - v_s, v_g - v_s, 0.0, v_b - v_s

    def _clamp_norm_voltages(self, v_raw: torch.Tensor) -> torch.Tensor:
        """Softplus-clamp ``v_raw`` to [v_min, v_max] then z-score.

        Fully elementwise — broadcasts over any leading batch dim, so a
        stacked ``(N, 4)`` input yields rows bit-identical to N separate
        ``(1, 4)`` calls.
        """
        beta = self._clamp_beta
        bx_lo = beta * (v_raw - self._v_min)
        v_clamped = self._v_min + torch.where(
            bx_lo > 20.0, v_raw - self._v_min,
            torch.log1p(torch.exp(bx_lo)) / beta)
        bx_hi = beta * (self._v_max - v_clamped)
        v_clamped = self._v_max - torch.where(
            bx_hi > 20.0, self._v_max - v_clamped,
            torch.log1p(torch.exp(bx_hi)) / beta)
        return (v_clamped - self._v_mean) / self._v_std_t

    def _prep_voltages(
        self, voltages: Dict[str, float],
    ) -> Tuple[torch.Tensor, float, float]:
        """Returns (x_full normalised, v_d_nn, v_s_nn)."""
        v_d_nn, v_g_nn, v_s_nn, v_b_nn = self._raw_voltages(voltages)
        v_raw = torch.tensor(
            [v_d_nn, v_g_nn, v_s_nn, v_b_nn],
            dtype=torch.float32, device=self._device)
        v_norm = self._clamp_norm_voltages(v_raw)
        x = torch.cat([v_norm, self._geo_norm_t]).unsqueeze(0)
        return x, v_d_nn, v_s_nn

    # ── Core eval: forward + autograd + denorm ───────────────────────

    def _v_tuple(self, voltages: Dict[str, float]) -> Tuple[float, ...]:
        """Cache key: the 4 terminal voltages in (d, g, s, b) order."""
        return (
            voltages.get(self.nodes[0], 0.0),
            voltages.get(self.nodes[1], 0.0),
            voltages.get(self.nodes[2], 0.0),
            voltages.get(self.nodes[3], 0.0),
        )

    def _unpack_eval(
        self,
        out_row: torch.Tensor,
        grad_id_row: torch.Tensor,
        grad_qg_row: Optional[torch.Tensor],
        grad_qd_row: Optional[torch.Tensor],
        v_d_nn: float,
        v_s_nn: float,
    ) -> Dict[str, float]:
        """Denormalise one forward+autograd result into the physical
        result dict and apply the Vds correction.

        ``out_row`` is a 1-D tensor of model outputs; ``grad_*_row`` are
        the 1-D autograd derivatives of the named output w.r.t. the 4
        voltage inputs. Shared verbatim by the per-device ``_eval`` and
        the batched ``batch_eval`` path so both produce identical numbers.

        V7.0.1: ``grad_qg_row`` / ``grad_qd_row`` may be ``None`` when the
        analysis consumes no capacitances (see ``_caps_required``). The
        returned dict then omits the 5 cap keys; every other value is
        bit-identical to the full path, because nothing in the id / charge
        chain or the Vds correction reads a cap.
        """
        # One ``.tolist()`` per tensor instead of per-element ``.item()``
        # — same float32 values, far fewer host/device syncs.
        out = out_row.tolist()
        gi = grad_id_row.tolist()
        with_caps = grad_qg_row is not None and grad_qd_row is not None

        # Scalar predictions → physical units. The normalizer's stats
        # are stored in OUTPUT_COLUMN_ORDER, so look up by name.
        id_phys = self._denorm("id", out[self._mcol("id")])
        qg_phys = self._denorm("qg", out[self._mcol("qg")])
        qd_phys = self._denorm("qd", out[self._mcol("qd")])
        qb_phys = self._denorm("qb", out[self._mcol("qb")])
        qs_phys = -(qg_phys + qd_phys + qb_phys)  # charge conservation

        # Conductances from autograd. The NN predicts id in PyCMG sign
        # convention (negative for NMOS ON), so d(id)/dV is negative;
        # negate ALL THREE so the solver's "current leaving drain" frame
        # gets always-positive conductances.
        #
        # gds is negated for the same reason gm and gmb are: gm = d(id)/dVg and
        # gds = d(id)/dVd are both derivatives of the same signed id, so the sign
        # comes from id's convention, not from which variable is differentiated.
        # Until 2026-07-21 gds alone was left un-negated, following commit
        # 930c274's "gds is the diagonal so no flip" rule — wrong for this stored
        # convention, and already refuted in losses/bni_mae.py:100-113, which
        # negates all three. Verified: autograd d(id)/dVd vs -gds_head = 0.12 rel
        # err, vs +gds_head = 2.08 (a rel err of exactly 2.0 is the signature of a
        # pure sign flip); OSDI -d(id)/dVd is positive at 100.0000% of conducting
        # points over 111,630 evals. See the 2026-07-21 systematic audit §A3
        # (docs/CHANGELOG.md V6.13.0).
        gm_phys = -self._denorm_deriv(
            "id", in_col=1, deriv_norm=gi[1], phys_val=id_phys)
        gds_phys = -self._denorm_deriv(
            "id", in_col=0, deriv_norm=gi[0], phys_val=id_phys)
        gmb_phys = -self._denorm_deriv(
            "id", in_col=3, deriv_norm=gi[3], phys_val=id_phys)

        gds_phys = self._guard_gds(id_phys, gds_phys)

        result = {
            "id": id_phys, "gm": gm_phys, "gds": gds_phys, "gmb": gmb_phys,
            "qg": qg_phys, "qd": qd_phys, "qs": qs_phys, "qb": qb_phys,
        }
        if with_caps:
            gqg = grad_qg_row.tolist()
            gqd = grad_qd_row.tolist()
            result["cgg"] = self._denorm_deriv(
                "qg", in_col=1, deriv_norm=gqg[1], phys_val=qg_phys)
            result["cgd"] = self._denorm_deriv(
                "qg", in_col=0, deriv_norm=gqg[0], phys_val=qg_phys)
            result["cgs"] = self._denorm_deriv(
                "qg", in_col=2, deriv_norm=gqg[2], phys_val=qg_phys)
            result["cdg"] = self._denorm_deriv(
                "qd", in_col=1, deriv_norm=gqd[1], phys_val=qd_phys)
            result["cdd"] = self._denorm_deriv(
                "qd", in_col=0, deriv_norm=gqd[0], phys_val=qd_phys)

        return self._apply_vds_correction(result, vds=v_d_nn - v_s_nn)

    def _eval(self, voltages: Dict[str, float]) -> Dict[str, float]:
        v_tuple = self._v_tuple(voltages)
        if self._cache_voltages == v_tuple and self._eval_cache is not None:
            if self._cache_has_caps or not self._caps_required:
                return self._eval_cache

        need_caps = self._caps_required
        x, v_d_nn, v_s_nn = self._prep_voltages(voltages)

        if self._fused_jac_available(self._nn_model):
            with torch.no_grad():
                out, grad_id, grad_qg, grad_qd = self._fused_eval(
                    self._nn_model, x, self._tech_code_tensor,
                    self._mcol("id"), self._mcol("qg"), self._mcol("qd"),
                    need_caps)
            result = self._unpack_eval(
                out[0], grad_id[0],
                grad_qg[0] if grad_qg is not None else None,
                grad_qd[0] if grad_qd is not None else None,
                v_d_nn, v_s_nn)
            self._eval_cache = result
            self._cache_voltages = v_tuple
            self._cache_has_caps = need_caps
            return result

        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        with torch.enable_grad():
            out = self._forward_model(x_full)
            grad_id = torch.autograd.grad(
                out[:, self._mcol("id")].sum(), x_v,
                create_graph=False, retain_graph=need_caps)[0]
            grad_qg = grad_qd = None
            if need_caps:
                grad_qg = torch.autograd.grad(
                    out[:, self._mcol("qg")].sum(), x_v,
                    create_graph=False, retain_graph=True)[0]
                grad_qd = torch.autograd.grad(
                    out[:, self._mcol("qd")].sum(), x_v,
                    create_graph=False, retain_graph=False)[0]

        result = self._unpack_eval(
            out[0], grad_id[0],
            grad_qg[0] if grad_qg is not None else None,
            grad_qd[0] if grad_qd is not None else None,
            v_d_nn, v_s_nn)

        self._eval_cache = result
        self._cache_voltages = v_tuple
        self._cache_has_caps = need_caps
        return result

    @staticmethod
    def batch_eval(
        mosfets: List["_MOSFETNNBase"],
        voltages: Dict[str, float],
    ) -> None:
        """Pre-populate every NN MOSFET's ``_eval_cache`` with one stacked
        forward + autograd call per distinct checkpoint (perf, plan
        Phase 5).

        DirectNet's MLP is row-independent, so a single forward over a
        stacked input and a single ``autograd.grad`` of the column-sum
        give per-row gradients with no cross-device coupling. Devices are
        grouped by their ``_nn_model`` identity because each per-tech /
        per-device-type checkpoint is a separate ``nn.Module``; the
        process-level ``_SHARED_NN_MODULES`` cache makes devices loading
        the same checkpoint share one module so they group together.

        After this call, the subsequent per-device ``_stamp_mosfet_dc``
        path hits a warm cache (``_eval`` returns ``_eval_cache``), so the
        stamping code is unchanged. Devices already holding a valid cache
        for ``voltages`` are skipped; the per-device ``_eval`` remains a
        correct fallback for anything not pre-computed here.

        Accuracy: a group of ONE device is exactly bit-identical to the
        per-device path (same GEMV, same autograd). A group of N>1 runs a
        stacked GEMM whose accumulation order differs from N separate
        GEMVs at the last bit (~1e-8 in the NN output) — pure float
        noise on its own, but a high-gain circuit can amplify it; see
        the ``_batch_eval_nn_mosfets`` note and the ``NN_BATCHED_EVAL``
        opt-out in ``solver.py``.
        """
        # Group devices that still need an eval by shared model identity.
        # Devices in one group share a checkpoint → identical voltage
        # clamp/norm params, so their inputs are normalised in one
        # batched op (the per-device geometry rows still differ).
        groups: Dict[int, List["_MOSFETNNBase"]] = {}
        raw_v: Dict[int, List[Tuple[float, float, float, float]]] = {}
        v_tuples: Dict[int, List[Tuple[float, ...]]] = {}
        for m in mosfets:
            v_tuple = m._v_tuple(voltages)
            if (m._cache_voltages == v_tuple and m._eval_cache is not None
                    and (m._cache_has_caps or not m._caps_required)):
                continue  # already warm
            key = id(m._nn_model)
            groups.setdefault(key, []).append(m)
            raw_v.setdefault(key, []).append(m._raw_voltages(voltages))
            v_tuples.setdefault(key, []).append(v_tuple)

        for key, devs in groups.items():
            ref = devs[0]
            # One stacked (N,4) tensor build for the whole group, then a
            # batched clamp+z-score with the group-shared norm params.
            #
            # V7.2.0 Phase 3a data-movement spec (§3.2/§3.4): stage
            # through a contiguous numpy array — ``torch.tensor(list,
            # device=cuda)`` marshals element-by-element through Python.
            # Bit-identical either way (both are IEEE f64->f32 casts;
            # verified bit-equal over 16k random rows), so the CPU
            # default path is unchanged in value.
            v_raw = torch.from_numpy(
                np.asarray(raw_v[key], dtype=np.float32))
            if ref._device.type != "cpu":
                v_raw = v_raw.to(ref._device)
            v_norm = ref._clamp_norm_voltages(v_raw)
            x_v = v_norm.detach().requires_grad_(True)
            x_g, tech_codes, pmos_arr = _stacked_group_inputs(
                ref._nn_model, devs)
            x_full = torch.cat([x_v, x_g], dim=1)

            need_caps = any(m._caps_required for m in devs)
            id_col = ref._mcol("id")
            qg_col = ref._mcol("qg")
            qd_col = ref._mcol("qd")

            if ref._fused_jac_available(ref._nn_model):
                with torch.no_grad():
                    out, grad_id, grad_qg, grad_qd = ref._fused_eval(
                        ref._nn_model, x_full, tech_codes,
                        id_col, qg_col, qd_col, need_caps)
                out, grad_id, grad_qg, grad_qd = ref._to_host_block(
                    out, grad_id, grad_qg, grad_qd)
                results = ref._unpack_eval_batch(
                    ref, out, grad_id, grad_qg, grad_qd,
                    raw_v[key], pmos_arr)
                tuples = v_tuples[key]
                for i, m in enumerate(devs):
                    m._eval_cache = results[i]
                    m._cache_voltages = tuples[i]
                    m._cache_has_caps = need_caps
                continue

            with torch.enable_grad():
                out = ref._nn_model(x_full, tech_codes=tech_codes)
                # V7.0.1: the qg / qd sweeps below are skipped outright
                # when no capacitance consumer is attached (DC / OP) —
                # a 2x saving, not a reordering. The forward and the id
                # sweep are untouched, so DC results are bit-identical.
                # Three separate backward sweeps — one per output
                # column. NOT collapsed via ``is_grads_batched``: that
                # vmap path changes the reduction order and is not
                # bit-identical to the per-device autograd (verified —
                # it shifted the inverter VTC by up to 0.25 V). The
                # accuracy-neutral gate outranks the extra speedup.
                grad_id = torch.autograd.grad(
                    out[:, id_col].sum(), x_v,
                    create_graph=False, retain_graph=need_caps)[0]
                grad_qg = grad_qd = None
                if need_caps:
                    grad_qg = torch.autograd.grad(
                        out[:, qg_col].sum(), x_v,
                        create_graph=False, retain_graph=True)[0]
                    grad_qd = torch.autograd.grad(
                        out[:, qd_col].sum(), x_v,
                        create_graph=False, retain_graph=False)[0]

            out, grad_id, grad_qg, grad_qd = ref._to_host_block(
                out, grad_id, grad_qg, grad_qd)
            results = ref._unpack_eval_batch(
                ref, out, grad_id, grad_qg, grad_qd, raw_v[key], pmos_arr)
            tuples = v_tuples[key]
            for i, m in enumerate(devs):
                m._eval_cache = results[i]
                m._cache_voltages = tuples[i]
                m._cache_has_caps = need_caps

    @staticmethod
    def _unpack_eval_batch(
        ref: "_MOSFETNNBase",
        out: torch.Tensor,
        grad_id: torch.Tensor,
        grad_qg: Optional[torch.Tensor],
        grad_qd: Optional[torch.Tensor],
        raw_rows: List[Tuple[float, float, float, float]],
        is_pmos: np.ndarray,
    ) -> List[Dict[str, float]]:
        """V7.2.0 Phase 2a-full: the whole ``_unpack_eval`` tail for a
        group, vectorised in float64 numpy — bit-identical per element to
        N calls of the scalar tail (gated by ``tests/verify_batched_tail.py``).

        ``ref`` supplies the group-shared denorm constants (all derived
        from the checkpoint/norm file that keyed the group); the only
        per-device non-voltage input, ``_is_pmos``, arrives as an array.
        Every arithmetic expression below reproduces the scalar tail's
        association order — do not "simplify" (regrouping rounds
        differently, and this feeds every stamped current).

        The two §8.1 constraints are acceptance criteria, not style:

        1. The Vds-correction exponential is evaluated with per-element
           libm ``math.exp`` over the boolean-masked subset that reaches
           the exp branch. ``np.exp`` / any SIMD exponential is
           PROHIBITED here — it mismatches libm by 1 ULP on ~4.6 % of
           arguments, and the downstream ``1 − exp`` cancellation
           amplifies that ~60× exactly in the SRAM off-device regime.
           (``np.sinh`` in the denorm is fine: measured bit-equal to the
           scalar path, which itself uses ``np.sinh``.)
        2. Everything is cast to float64 BEFORE any arithmetic. The
           tensors arrive float32; under NEP-50 a float32 array times a
           Python float STAYS float32, which silently runs the chain at
           ~2e-7 relative error — worse than VNTOL. The dtype asserts are
           load-bearing.
        """
        o = out.detach().cpu().numpy().astype(np.float64)
        gi = grad_id.detach().cpu().numpy().astype(np.float64)
        with_caps = grad_qg is not None and grad_qd is not None
        if with_caps:
            gqg = grad_qg.detach().cpu().numpy().astype(np.float64)
            gqd = grad_qd.detach().cpu().numpy().astype(np.float64)
        raw = np.asarray(raw_rows, dtype=np.float64)
        vds = raw[:, 0] - raw[:, 2]
        n = o.shape[0]
        assert (o.dtype == np.float64 and gi.dtype == np.float64
                and vds.dtype == np.float64), "batched tail must be float64"

        out_std = ref._out_std_f
        out_mean = ref._out_mean_f
        asinh = ref._asinh_f
        in_std_f = ref._in_std_f

        def _dn(name: str, col: np.ndarray) -> np.ndarray:
            # mirrors ``_denorm``: u = v*std + mean; asinh → scale*sinh(u)
            u = col * out_std[name] + out_mean[name]
            if asinh:
                return asinh[name] * np.sinh(u)
            return u

        def _dd(out_name: str, in_col: int,
                dcol: np.ndarray, phys: np.ndarray) -> np.ndarray:
            # mirrors ``_denorm_deriv``: d*out_std*factor/in_std, with
            # factor ≡ 1.0 (exact identity) in the non-asinh mode.
            in_s = in_std_f[in_col]
            if in_s < 1e-12:
                return np.zeros_like(dcol)
            o_s = out_std[out_name]
            if asinh:
                sc = asinh[out_name]
                fac = np.sqrt(sc * sc + phys * phys)
                return dcol * o_s * fac / in_s
            return dcol * o_s / in_s

        id_p = _dn("id", o[:, ref._mcol("id")])
        qg_p = _dn("qg", o[:, ref._mcol("qg")])
        qd_p = _dn("qd", o[:, ref._mcol("qd")])
        qb_p = _dn("qb", o[:, ref._mcol("qb")])
        qs_p = -(qg_p + qd_p + qb_p)  # charge conservation

        # Conductances: negate all three (see the scalar tail's sign note).
        gm = -_dd("id", 1, gi[:, 1], id_p)
        gds = -_dd("id", 0, gi[:, 0], id_p)
        gmb = -_dd("id", 3, gi[:, 3], id_p)
        # guard F (negative-only; positives pass through bit-identical)
        gds = np.where(
            gds > 0.0, gds,
            np.maximum(np.abs(id_p) * _GDS_GUARD_K, 1e-12))

        if with_caps:
            cgg = _dd("qg", 1, gqg[:, 1], qg_p)
            cgd = _dd("qg", 0, gqg[:, 0], qg_p)
            cgs = _dd("qg", 2, gqg[:, 2], qg_p)
            cdg = _dd("qd", 1, gqd[:, 1], qd_p)
            cdd = _dd("qd", 0, gqd[:, 0], qd_p)

        # ── ``_apply_vds_correction``, vectorised ────────────────────
        VDD_train = ref._vdd_estimate
        VT = max(0.06 * VDD_train, 0.026)
        a = np.abs(vds)
        normal = np.where(is_pmos, vds < 0.0, vds > 0.0)

        # (a) rail-restoring extrapolation
        m_ext = a > VDD_train
        if m_ext.any():
            overshoot = a - VDD_train
            g_max = 1.0e-3
            x_ref = 0.5 * VDD_train
            x_cap = 5.0 * x_ref
            lin = overshoot > x_cap
            id_extra = np.where(
                lin,
                0.5 * g_max * x_cap * x_cap / x_ref
                + g_max * x_cap / x_ref * (overshoot - x_cap),
                0.5 * g_max * overshoot * overshoot / x_ref)
            g_extra = np.where(
                lin, g_max * x_cap / x_ref, g_max * overshoot / x_ref)
            delta_ext = np.where(is_pmos, id_extra, -id_extra)
            id_p = np.where(m_ext & normal, id_p + delta_ext, id_p)
            gds = np.where(m_ext, np.maximum(gds, g_extra), gds)

        # Fast path: well into the normal-direction regime — those rows
        # keep their post-(a) values (composed via ``fast`` at the end).
        fast = normal & (a > 20.0 * VT)

        # (b) §8.1 Constraint 1: masked per-element libm math.exp.
        exp_sym = np.zeros(n)
        need = np.flatnonzero(a <= 20.0 * VT)
        if need.size:
            args = ((-a[need]) / VT).tolist()
            exp_sym[need] = [math.exp(v) for v in args]
        f_sym = 1.0 - exp_sym

        # reverse-conduction taper, mirroring ``_reverse_taper``'s branch
        # order (a ≤ x0 wins over a ≥ x1, relevant only if the env knobs
        # invert the window)
        x0 = _REV_TAPER_X0 * VDD_train
        x1 = _REV_TAPER_X1 * VDD_train
        taper = np.ones(n)
        gt0 = a > x0
        taper[gt0 & (a >= x1)] = 0.0
        mid = gt0 & (a < x1)
        if mid.any():
            u = (a[mid] - x0) / (x1 - x0)
            taper[mid] = 1.0 - u * u * (3.0 - 2.0 * u)
        f_id = np.where(normal, f_sym, f_sym * taper)

        id_new = id_p * f_id
        gm_new = gm * f_id
        gmb_new = gmb * f_id
        # (c) symmetric gds factor + linear-region term, then guard F
        gds_new = gds * f_sym + np.abs(id_p) * exp_sym / VT
        gds_new = np.where(
            gds_new > 0.0, gds_new,
            np.maximum(np.abs(id_new) * _GDS_GUARD_K, 1e-12))

        # (d) wrong-sign clamp, scoped by direction
        neg = id_new < 0.0
        pos = id_new > 0.0
        wrong = np.where(
            is_pmos,
            np.where(normal, neg, pos),
            np.where(normal, pos, neg))
        id_new = np.where(wrong, 0.0, id_new)
        gm_new = np.where(wrong, 0.0, gm_new)
        gmb_new = np.where(wrong, 0.0, gmb_new)

        id_f = np.where(fast, id_p, id_new)
        gm_f = np.where(fast, gm, gm_new)
        gmb_f = np.where(fast, gmb, gmb_new)
        gds_f = np.where(fast, gds, gds_new)
        assert (id_f.dtype == np.float64 and gds_f.dtype == np.float64
                and qg_p.dtype == np.float64), "batched tail must be float64"

        # ── Per-device result dicts (same keys, same order, Python
        # floats via one C-level ``tolist`` per column) ────────────────
        cols = [id_f, gm_f, gds_f, gmb_f, qg_p, qd_p, qs_p, qb_p]
        if with_caps:
            cols += [cgg, cgd, cgs, cdg, cdd]
        lists = [c.tolist() for c in cols]
        if with_caps:
            return [
                {"id": v0, "gm": v1, "gds": v2, "gmb": v3,
                 "qg": v4, "qd": v5, "qs": v6, "qb": v7,
                 "cgg": v8, "cgd": v9, "cgs": v10, "cdg": v11, "cdd": v12}
                for v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12
                in zip(*lists)
            ]
        return [
            {"id": v0, "gm": v1, "gds": v2, "gmb": v3,
             "qg": v4, "qd": v5, "qs": v6, "qb": v7}
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*lists)
        ]

    # — small helpers —

    @staticmethod
    def _to_host_block(
        out: torch.Tensor,
        grad_id: torch.Tensor,
        grad_qg: Optional[torch.Tensor],
        grad_qd: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """V7.2.0 Phase 2a-lite: one D2H move for the whole result block.

        The per-device unpack loop calls ``.tolist()`` per row; on a CUDA
        tensor every row is a separate device sync — the measured reason
        the unmodified latent GPU path gained ~nothing (plan §3.3). Here
        the (N, 13) outputs and the (N, 4) Jacobian rows are concatenated
        and moved to the host in a single transfer, then sliced back.

        On CPU this returns the inputs untouched (no copy, no reorder).
        The CUDA path copies values exactly (a transfer, not arithmetic),
        so unpacked results are bit-identical to per-row readback.
        """
        if out.device.type == "cpu":
            return out, grad_id, grad_qg, grad_qd
        ncols = out.shape[1]
        parts = [out.detach(), grad_id.detach()]
        if grad_qg is not None and grad_qd is not None:
            parts += [grad_qg.detach(), grad_qd.detach()]
        blk = torch.cat(parts, dim=1).cpu()
        out_h = blk[:, :ncols]
        grad_id_h = blk[:, ncols:ncols + 4]
        if grad_qg is not None and grad_qd is not None:
            return (out_h, grad_id_h,
                    blk[:, ncols + 4:ncols + 8], blk[:, ncols + 8:ncols + 12])
        return out_h, grad_id_h, None, None

    def _forward_model(self, x_full: torch.Tensor) -> torch.Tensor:
        """Override in BSIMAR subclass to call the AR-inference forward."""
        return self._nn_model(x_full, tech_codes=self._tech_code_tensor)

    @staticmethod
    def _fused_jac_available(model: torch.nn.Module) -> bool:
        """Whether ``model`` can serve the V7.0.3 closed-form Jacobian.

        Requires the opt-in flag AND a model that implements it AND an
        instance whose ``id`` column is not re-composed by an EKV core or
        monotone residual. False for BSIM-AR and PFN, whose forwards are
        not plain MLPs.
        """
        return (
            _FUSED_JAC
            and hasattr(model, "forward_with_jacobian")
            and model.supports_fused_jacobian())

    @staticmethod
    def _fused_eval(
        model: torch.nn.Module,
        x_full: torch.Tensor,
        tech_codes: torch.Tensor,
        id_col: int, qg_col: int, qd_col: int,
        need_caps: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Closed-form (out, grad_id, grad_qg, grad_qd) — no autograd.

        Slices the (B, 4, out_dim) Jacobian into the same per-column (B, 4)
        tensors the autograd path produces, so ``_unpack_eval`` is shared
        verbatim between the two.
        """
        out, jac = model.forward_with_jacobian(x_full, tech_codes=tech_codes)
        return (
            out,
            jac[:, :, id_col],
            jac[:, :, qg_col] if need_caps else None,
            jac[:, :, qd_col] if need_caps else None,
        )

    def _mcol(self, name: str) -> int:
        """Model-output column index for ``name``."""
        return self._out_col[name]

    def _stats_col(self, name: str) -> int:
        """Index of column ``name`` in the normalizer's stats arrays."""
        return self._stats_idx[name]

    def _denorm(self, name: str, val_norm: float) -> float:
        """Physical value of a single scalar output column.

        V7.0.1: the per-column stats are resolved once in ``__init__``
        (``_out_std_f`` / ``_out_mean_f`` / ``_asinh_f``) instead of via a
        ``list.index`` + numpy element read per call. The arithmetic is
        unchanged — ``float(np.float64)`` is exact, and ``np.sinh`` is kept
        rather than ``math.sinh`` because libm and numpy may disagree in
        the last ulp and this feeds every stamped current.
        """
        u = (float(val_norm) * self._out_std_f[name]
             + self._out_mean_f[name])
        if self._asinh_f:
            return self._asinh_f[name] * float(np.sinh(u))
        return u

    def _denorm_deriv(
        self, out_name: str, in_col: int, deriv_norm: float, phys_val: float,
    ) -> float:
        """Chain-rule denormalise a derivative.

        V7.0.1: inlined from ``_NormalizerBase.denormalize_derivative`` with
        the constants pre-resolved. The expression is reproduced in its
        original association order — ``d * out_std * factor / in_std``
        regrouped (e.g. folding ``out_std / in_std``) would round
        differently. ``math.sqrt`` is IEEE-correctly-rounded, so it is
        bit-identical to the ``np.sqrt`` it replaces.
        """
        in_std = self._in_std_f[in_col]
        if in_std < 1e-12:
            return 0.0
        out_std = self._out_std_f[out_name]
        if self._asinh_f:
            scale = self._asinh_f[out_name]
            out_factor = math.sqrt(scale * scale + phys_val * phys_val)
        else:
            out_factor = 1.0
        return float(deriv_norm) * out_std * out_factor / in_std

    @staticmethod
    def _guard_gds(id_phys: float, gds_phys: float) -> float:
        """Negative-only gds guard ("guard F", audit §A3-guard).

        Positives pass through **bit-identical** to the raw autograd Jacobian, so
        this provably cannot perturb a correct value — the point, given how
        sensitive opamp operating points are to Jacobian changes. Only a negative
        candidate is clamped, to ``max(|id|·k, 1e-12)`` with k = 1/(50 V).

        A negative small-signal output conductance is never physical here: OSDI
        ``-d(id)/dVd`` is positive at 100.0000% of conducting points across
        111,630 evals on all 10 production devices, forward and reverse. Every
        negative is model error, so clamping (rather than passing it to the
        solver, which is the Rule 4 divergence mode) is always right. Landing at
        1/(50 V) keeps the error set within ~1.3-3x of truth instead of stamping
        r_o = 1e12 ohm — an essentially open drain — where truth is ~4.5e-05 S.
        """
        if gds_phys > 0.0:
            return gds_phys
        return max(abs(id_phys) * _GDS_GUARD_K, 1e-12)

    @staticmethod
    def _reverse_taper(abs_vds: float, vdd_train: float) -> float:
        """C¹ roll-off of the reverse-conduction blend.

        Soft-blend window (V6.4.7 S7 bisection): 1 inside |Vds| ≤
        0.20·VDD_train, smoothstep to 0 by 0.30·VDD_train. Window rule:
        the LARGEST window that breaks no protected gate — the full
        trained corridor (taper at 0.30–0.40·VDD) regressed the TSMC5
        opamp; 0.10–0.20 broke nothing but lost the TSMC12 SC flip.

        V6.5.2 diagnostic knob (default-off, S7-window preserving): the
        window edges are read from ``PYCIRCUITSIM_REV_TAPER_X0`` /
        ``PYCIRCUITSIM_REV_TAPER_X1`` (fractions of VDD_train, default
        0.20 / 0.30). The TG-corridor retrain teaches the deep-reverse
        conduction the original taper zeros; widening the window lets that
        learned surface reach the switchcap pass-device regime. When the env
        vars are unset the window is exactly 0.20/0.30 → committed behaviour
        is unchanged. Only valid in tandem with a tg_corridor-trained
        checkpoint; widening it on a stock checkpoint injects the raw (~0)
        reverse surface as garbage.
        """
        x0 = _REV_TAPER_X0 * vdd_train
        x1 = _REV_TAPER_X1 * vdd_train
        if abs_vds <= x0:
            return 1.0
        if abs_vds >= x1:
            return 0.0
        u = (abs_vds - x0) / (x1 - x0)
        return 1.0 - u * u * (3.0 - 2.0 * u)

    # ── Vds correction ───────────────────────────────────────────────

    def _apply_vds_correction(
        self, result: Dict[str, float], vds: float,
    ) -> Dict[str, float]:
        """Enforce Id(Vds=0)=0 and Id=0 for reverse Vds, plus rail-
        restoring extrapolation past the training Vds range.

        Four-part correction (order matters):

        (a) Quadratic-then-linear ramp when |Vds| > VDD_train. Replicates
            PyCMG's restoring leakage so NR converges to the true rail
            instead of locking on the NN's flat-zero plateau.
        (b) One-sided 1−exp(−|Vds|/VT) factor on Id/gm/gmb in the normal
            direction; in the reverse direction the same factor times a
            C¹ taper (1 inside the trained |Vds| ≤ 0.30·VDD_train
            reverse_vds corridor, 0 past 0.40·VDD_train) — V6.4.7 S7/P2
            relaxation; the raw reverse surface is sign-correct and
            ~25–35 % conservative on the corridor (S7 probe).
        (c) Symmetric Vds factor on gds plus a linear-region term so the
            Jacobian has finite slope even when Id is forced to zero.
        (d) Sign enforcement scoped by direction: normal keeps the
            original guard (NMOS id≤0, PMOS id≥0); reverse allows the
            physically flipped sign and clamps forward-sign noise.
        """
        VDD_train = self._vdd_estimate
        VT = max(0.06 * VDD_train, 0.026)
        abs_vds = abs(vds)
        normal_dir = (vds < 0.0) if self._is_pmos else (vds > 0.0)

        # (a) Rail-restoring extrapolation
        if abs_vds > VDD_train:
            overshoot = abs_vds - VDD_train
            g_max = 1.0e-3       # 1 mS scale
            x_ref = 0.5 * VDD_train
            x_cap = 5.0 * x_ref  # transition to linear past 5·x_ref
            if overshoot <= x_cap:
                id_extra = 0.5 * g_max * overshoot * overshoot / x_ref
                g_extra = g_max * overshoot / x_ref
            else:
                id_at_cap = 0.5 * g_max * x_cap * x_cap / x_ref
                g_at_cap = g_max * x_cap / x_ref
                id_extra = id_at_cap + g_at_cap * (overshoot - x_cap)
                g_extra = g_at_cap
            # Sign convention for restoring leakage (V6.2 fix):
            # In PyCMG sign convention, NMOS conducting id < 0 (current
            # leaving drain in CMG's frame) and PMOS conducting id > 0. At
            # rail-overshoot, the physical restoring leakage drives id in
            # the *same* direction as conducting (more |id|), pulling the
            # drain node back toward the source rail via the device. The
            # original V4-re ship used the opposite sign here; the
            # wrong-sign clamp at (d) then wiped the contribution inside
            # the band VDD_train < |Vds| < 20·VT, creating a current-free
            # dead-band where V(out) could settle at ~±100 mV outside the
            # rails (the V6.1 TSMC7 transient bottleneck).
            if normal_dir:
                if self._is_pmos:
                    result["id"] += id_extra      # PMOS: id more positive
                else:
                    result["id"] -= id_extra      # NMOS: id more negative
            result["gds"] = max(result["gds"], g_extra)

        # Fast path: well into the normal-direction regime.
        if normal_dir and abs_vds > 20.0 * VT:
            return result

        exp_sym = math.exp(-abs_vds / VT) if abs_vds <= 20.0 * VT else 0.0
        f_sym = 1.0 - exp_sym
        if normal_dir:
            f_id = f_sym
        else:
            f_id = f_sym * self._reverse_taper(abs_vds, VDD_train)

        id_raw = result["id"]
        result["id"] = id_raw * f_id
        result["gm"] *= f_id
        result["gmb"] *= f_id
        result["gds"] = result["gds"] * f_sym + abs(id_raw) * exp_sym / VT
        result["gds"] = self._guard_gds(result["id"], result["gds"])

        # (d) wrong-sign clamp, scoped by direction: reverse conduction
        # physically flips the sign, so the reverse branch clamps
        # forward-sign noise instead of all conduction.
        if normal_dir:
            wrong = (
                (self._is_pmos and result["id"] < 0.0)
                or (not self._is_pmos and result["id"] > 0.0))
        else:
            wrong = (
                (self._is_pmos and result["id"] > 0.0)
                or (not self._is_pmos and result["id"] < 0.0))
        if wrong:
            result["id"] = 0.0
            result["gm"] = 0.0
            result["gmb"] = 0.0

        return result

    # ── Solver-side interface ────────────────────────────────────────

    def get_nodes(self) -> List[str]:
        return self.nodes

    def stamp_conductance(self, matrix, node_map):  # noqa: D401
        pass  # solver stamps MOSFETs directly

    def stamp_rhs(self, rhs, node_map):
        pass

    def get_conductance(
        self, voltages: Dict[str, float],
    ) -> Tuple[float, float, float]:
        r = self._eval(voltages)
        return r["gds"], r["gm"], r["gmb"]

    @staticmethod
    def require_caps(components) -> None:
        """Mark every NN MOSFET in ``components`` as cap-consuming.

        Called once by the two analyses that read capacitances
        (``TransientSolver`` / ``ACSolver``) so their evals compute the
        qg / qd Jacobians. A ``.dc`` / ``.op`` run never calls this and so
        never pays for them (V7.0.1). Any stale DC-warmed cache is
        dropped, so the first eval after this call is a full one.
        """
        for c in components:
            if isinstance(c, _MOSFETNNBase):
                c._caps_required = True
                if not c._cache_has_caps:
                    c.clear_cache()

    def get_capacitances(
        self, voltages: Dict[str, float],
    ) -> Dict[str, float]:
        # Self-heal: something is asking for caps on a device that was
        # never marked cap-consuming (a direct API/test call, or an
        # analysis that forgot ``require_caps``). Latch the flag on and
        # recompute rather than KeyError — the flag is a performance
        # hint, never a correctness precondition.
        if not self._caps_required:
            self._caps_required = True
            self.clear_cache()
        r = self._eval(voltages)
        return {k: r[k] for k in ("cgg", "cgd", "cgs", "cdg", "cdd")}

    def get_charges(
        self, voltages: Dict[str, float],
    ) -> Dict[str, float]:
        r = self._eval(voltages)
        return {k: r[k] for k in ("qg", "qd", "qs", "qb")}

    # ── Transient charge state ───────────────────────────────────────

    def init_charge_state(self, voltages: Dict[str, float]) -> None:
        q = self.get_charges(voltages)
        self._q_prev = q.copy()
        self._q_prev2 = q.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        self._i_prev_gate = 0.0
        self._i_prev_drain = 0.0

    def update_charge_state(
        self,
        voltages: Dict[str, float],
        cap_currents: Optional[Dict[str, float]] = None,
    ) -> None:
        q = self.get_charges(voltages)
        self._q_prev2 = (
            self._q_prev.copy() if self._q_prev is not None else q.copy())
        self._q_prev = q.copy()
        self._v_prev_tran = {
            "d": voltages.get(self.nodes[0], 0.0),
            "g": voltages.get(self.nodes[1], 0.0),
            "s": voltages.get(self.nodes[2], 0.0),
            "b": voltages.get(self.nodes[3], 0.0),
        }
        if cap_currents is not None:
            self._i_prev_gate = cap_currents.get("i_gate", 0.0)
            self._i_prev_drain = cap_currents.get("i_drain", 0.0)

    def set_temperature(self, temperature_kelvin: float) -> None:
        """Rebind the temperature feature used by NN inference.

        Temperature is part of the constant geometry tensor, not the voltage
        tuple used as the evaluation-cache key.  A temperature sweep therefore
        has to rebuild that tensor and clear both the current cache and the
        transient charge history.
        """
        if temperature_kelvin <= 200.0:
            raise ValueError(
                f"Temperature must be in Kelvin (> 200 K), got "
                f"{temperature_kelvin}. Use temp_K = temp_C + 273.15.")

        self.temperature = float(temperature_kelvin)
        self._refresh_geometry()
        self._bind_geometry_tensor()

        self.clear_cache()
        self._q_prev = None
        self._q_prev2 = None
        self._v_prev_tran = None

    def _refresh_geometry(self) -> None:
        """Recompute normalized geometry after a geometry feature changes."""
        nfin_log = float(np.log2(max(self.NFIN, 1.0)))
        geo_raw = np.array(
            [nfin_log, self.L, self.temperature], dtype=np.float64)
        geo_std = self._norm_stats.input_std[4:7].copy()
        geo_std[geo_std < 1e-12] = 1.0
        self._geo_norm = (
            (geo_raw - self._norm_stats.input_mean[4:7]) / geo_std)

    def _bind_geometry_tensor(self) -> None:
        """Share the normalized geometry tensor for this device geometry."""
        dev_key = str(self._device)
        geo_key = (
            self._norm_key, dev_key, self.NFIN, self.L, self.temperature)
        geo_t = _SHARED_GEO_TENSORS.get(geo_key)
        if geo_t is None:
            geo_t = torch.tensor(
                self._geo_norm, dtype=torch.float32, device=self._device)
            _SHARED_GEO_TENSORS[geo_key] = geo_t
        self._geo_norm_t = geo_t

    def clear_cache(self) -> None:
        self._eval_cache = None
        self._cache_voltages = None
        self._cache_has_caps = False
