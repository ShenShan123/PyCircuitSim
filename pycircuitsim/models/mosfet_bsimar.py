"""BSIM-AR Transformer compact model (LEVEL=74).

Wraps the BSIMAR Transformer into the same ``_MOSFETNNBase`` used by
DirectNet (LEVEL=73). The only differences are the model class and
the output column layout (the Transformer emits its targets in
``BSIMAR_COLUMN_ORDER``, so the base looks up columns by name).

Terminal order: [drain, gate, source, bulk]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from pycircuitsim.models.mosfet_nn import _MOSFETNNBase

__all__ = ["NMOS_BSIMAR", "PMOS_BSIMAR", "_MOSFETBSIMARBase"]


class _MOSFETBSIMARBase(_MOSFETNNBase):
    """LEVEL=74 base. Loads a BSIMAR Transformer checkpoint.

    Architecture is read from the sibling ``*_config.npz`` (saved by
    the trainer); checkpoint stem suffixes ``.phys`` / ``.ar`` are
    stripped so the matching ``norm`` and ``config`` files are found.
    """

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
        from bsimar.models.transformer import TransformerEncoderModel

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
                f"BSIMAR architecture config not found: {config_path}")
        cfg = np.load(str(config_path))

        def _build(_state: Dict[str, torch.Tensor]) -> torch.nn.Module:
            return TransformerEncoderModel(
                input_dim=int(cfg["input_dim"]),
                target_dim=int(cfg["target_dim"]),
                d_model=int(cfg["d_model"]),
                nhead=int(cfg["nhead"]),
                num_layers=int(cfg["num_layers"]),
                dim_feedforward=int(cfg["dim_feedforward"]),
                dropout=float(cfg["dropout"]),
                num_tech_codes=(
                    int(cfg["num_tech_codes"])
                    if "num_tech_codes" in cfg.files else 22),
            )

        super().__init__(
            name=name, nodes=nodes, model_path=model_path,
            L=L, NFIN=NFIN, temperature=temperature, tech_code=tech_code,
            model_factory=_build,
            output_layout="bsimar",
        )
        # Sanity-check the normaliser: BSIMAR was trained with asinh.
        assert self._norm_stats.mode == "asinh", (
            f"BSIMAR LEVEL=74 expects asinh-mode norm stats, "
            f"got mode={self._norm_stats.mode}")


class NMOS_BSIMAR(_MOSFETBSIMARBase):
    """N-channel BSIMAR MOSFET (LEVEL=74)."""

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return -self._eval(voltages)["id"]


class PMOS_BSIMAR(_MOSFETBSIMARBase):
    """P-channel BSIMAR MOSFET (LEVEL=74). Source-relative frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_pmos = True

    def calculate_current(self, voltages: Dict[str, float]) -> float:
        return self._eval(voltages)["id"]
