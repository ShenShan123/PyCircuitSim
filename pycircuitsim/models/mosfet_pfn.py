"""TabPFN-style compact model (LEVEL=75).

Wraps the TabPFN-v3-style in-context regressor (``bsimar.models.tabpfn``)
into the same ``_MOSFETNNBase`` used by DirectNet (LEVEL=73) and BSIM-AR
(LEVEL=74). The model is one-shot and emits the canonical
``OUTPUT_COLUMN_ORDER``, so the base's "standard" layout applies; its
frozen context lives in checkpoint buffers and the context-side
activations are cached inside the module after the first eval.

Architecture is read from the sibling ``*_config.npz`` (saved by the
trainer); checkpoint stem suffixes ``.phys`` / ``.ar`` are stripped so the
matching ``norm`` and ``config`` files are found.

Terminal order: [drain, gate, source, bulk]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from pycircuitsim.models.mosfet_nn import _MOSFETNNBase

__all__ = ["NMOS_PFN", "PMOS_PFN", "_MOSFETPFNBase"]


class _MOSFETPFNBase(_MOSFETNNBase):
    """LEVEL=75 base. Loads a TabPFNCompact checkpoint."""

    def __init__(
        self,
        name: str,
        nodes: List[str],
        model_path: str,
        L: float,
        NFIN: float,
        temperature: float = 300.15,
        tech_code: Optional[int] = None,
    ):
        from bsimar.models.tabpfn import TabPFNCompact

        model_path_obj = Path(model_path)
        base_stem = model_path_obj.stem
        for sfx in (".phys", ".ar"):
            if base_stem.endswith(sfx):
                base_stem = base_stem[: -len(sfx)]
                break

        config_path = model_path_obj.parent / (
            base_stem.replace("_best", "_config") + ".npz")
        if not config_path.exists():
            raise FileNotFoundError(
                f"TabPFN architecture config not found: {config_path}")

        def _build(state: Dict[str, torch.Tensor]) -> torch.nn.Module:
            # V7.2.0 Phase 1a: loaded lazily — the factory only runs on a
            # shared-module cache miss, so N devices sharing a checkpoint
            # read the sidecar once instead of N times (the existence
            # check above stays eager so a missing sidecar fails loud).
            cfg = np.load(str(config_path))
            model = TabPFNCompact(
                input_dim=int(cfg["input_dim"]),
                output_dim=int(cfg["output_dim"]),
                embed_dim=int(cfg["embed_dim"]),
                n_inducing=int(cfg["n_inducing"]),
                dist_blocks=int(cfg["dist_blocks"]),
                dist_heads=int(cfg["dist_heads"]),
                agg_blocks=int(cfg["agg_blocks"]),
                agg_heads=int(cfg["agg_heads"]),
                n_cls_tokens=int(cfg["n_cls_tokens"]),
                icl_num_blocks=int(cfg["icl_num_blocks"]),
                icl_heads=int(cfg["icl_heads"]),
                ctx_len=int(cfg["ctx_len"]),
                num_tech_codes=int(cfg["num_tech_codes"]),
                use_rope=bool(int(cfg["use_rope"])),
                ff_factor=int(cfg["ff_factor"]),
                feature_group_size=int(cfg["feature_group_size"]),
            )
            # The frozen context is data, not architecture — cross-check.
            assert state["ctx_x"].shape[0] == int(cfg["ctx_len"]), (
                f"ctx_len mismatch: config {int(cfg['ctx_len'])} vs "
                f"checkpoint {state['ctx_x'].shape[0]}")
            return model

        super().__init__(
            name=name, nodes=nodes, model_path=model_path,
            L=L, NFIN=NFIN, temperature=temperature, tech_code=tech_code,
            model_factory=_build,
            output_layout="standard",
        )
        # Sanity-check the normaliser: TabPFN was trained with asinh.
        assert self._norm_stats.mode == "asinh", (
            f"TabPFN LEVEL=75 expects asinh-mode norm stats, "
            f"got mode={self._norm_stats.mode}")


class NMOS_PFN(_MOSFETPFNBase):
    """N-channel TabPFN MOSFET (LEVEL=75)."""

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return -self._eval(voltages)["id"]


class PMOS_PFN(_MOSFETPFNBase):
    """P-channel TabPFN MOSFET (LEVEL=75). Source-relative frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return self._eval(voltages)["id"]
