"""Configuration for NN-based compact model training."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_MODEL_DIR = PROJECT_ROOT / "nn_model"
CHECKPOINT_DIR = NN_MODEL_DIR / "checkpoints"
DATA_DIR = NN_MODEL_DIR / "data" / "datasets"

# PyCMG paths
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
OSDI_PATH = str(PYCMG_DIR / "build-deep-verify" / "osdi" / "bsimcmg.osdi")

# ASAP7 technology config
ASAP7_MODELCARD = str(PYCMG_DIR / "tech_model_cards" / "ASAP7" / "7nm_TT_160803.pm")
ASAP7_VDD = 0.7
ASAP7_L = 30e-9  # 30nm channel length

# Default temperature
DEFAULT_TEMPERATURE = 300.15  # 27°C in Kelvin


@dataclass
class TechConfig:
    """Technology-specific configuration for data generation."""
    name: str
    modelcard_path: str
    nmos_model_name: str
    pmos_model_name: str
    vdd: float
    L: float
    nfin_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 15, 20])
    temperature: float = DEFAULT_TEMPERATURE
    # For techs with separate NMOS/PMOS modelcard files
    nmos_modelcard_path: Optional[str] = None
    pmos_modelcard_path: Optional[str] = None
    # For techs with asymmetric L (TSMC: L_nmos != L_pmos)
    L_nmos: Optional[float] = None
    L_pmos: Optional[float] = None

    def get_L(self, device_type: str) -> float:
        """Get channel length for a specific device type."""
        if device_type == "nmos" and self.L_nmos is not None:
            return self.L_nmos
        if device_type == "pmos" and self.L_pmos is not None:
            return self.L_pmos
        return self.L

    def get_modelcard_path(self, device_type: str) -> str:
        """Get modelcard path for a specific device type."""
        if device_type == "nmos" and self.nmos_modelcard_path is not None:
            return self.nmos_modelcard_path
        if device_type == "pmos" and self.pmos_modelcard_path is not None:
            return self.pmos_modelcard_path
        return self.modelcard_path


# Pre-defined technology configs
ASAP7_CONFIG = TechConfig(
    name="ASAP7",
    modelcard_path=ASAP7_MODELCARD,
    nmos_model_name="nmos_rvt",
    pmos_model_name="pmos_rvt",
    vdd=ASAP7_VDD,
    L=ASAP7_L,
)

# TSMC technology modelcard base
TSMC_MODELCARDS = PYCMG_DIR / "tech_model_cards"

TSMC5_CONFIG = TechConfig(
    name="TSMC5",
    modelcard_path="",  # unused, per-device paths below
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_lvt_mac",
    vdd=0.65,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC5" / "naive" / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC5" / "naive" / "pch_lvt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
)

TSMC7_CONFIG = TechConfig(
    name="TSMC7",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_svt_mac",  # LVT PMOS has PDIBL2_i bug; use SVT
    vdd=0.75,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC7" / "naive" / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC7" / "naive" / "pch_svt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
)

TSMC12_CONFIG = TechConfig(
    name="TSMC12",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_lvt_mac",
    vdd=0.80,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC12" / "naive" / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC12" / "naive" / "pch_lvt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
)

TSMC16_CONFIG = TechConfig(
    name="TSMC16",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_lvt_mac",
    vdd=0.80,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC16" / "naive" / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(TSMC_MODELCARDS / "TSMC16" / "naive" / "pch_lvt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
)

# Registry of all technologies
TECH_CONFIGS: Dict[str, TechConfig] = {
    "asap7": ASAP7_CONFIG,
    "tsmc5": TSMC5_CONFIG,
    "tsmc7": TSMC7_CONFIG,
    "tsmc12": TSMC12_CONFIG,
    "tsmc16": TSMC16_CONFIG,
}


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    batch_size: int = 1024
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Architecture
    trunk_hidden: int = 128
    trunk_layers: int = 3
    head_hidden: int = 64

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 50

    # Loss weights
    w_id: float = 1.0
    w_gm: float = 0.5
    w_gds: float = 0.5
    w_gmb: float = 0.3
    w_charges: float = 0.5
    w_caps: float = 0.3
    w_zero_bias: float = 5.0


# Output column names (13 values the solver consumes)
OUTPUT_COLUMNS = [
    "id", "gm", "gds", "gmb",
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

# Input column names
INPUT_COLUMNS = ["Vd", "Vg", "Vs", "Vb", "NFIN", "T"]
