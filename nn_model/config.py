"""Configuration for NN-based compact model training."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_MODEL_DIR = PROJECT_ROOT / "nn_model"
CHECKPOINT_DIR = NN_MODEL_DIR / "checkpoints"
DATA_DIR = NN_MODEL_DIR / "data" / "datasets"

# PyCMG paths (submodule: 21 device variants, 5 process nodes)
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
OSDI_PATH = str(PYCMG_DIR / "build" / "osdi" / "bsimcmg.osdi")

# Ensure PyCMG submodule is importable
_PYCMG_PYPATH = str(PYCMG_DIR)
if _PYCMG_PYPATH not in sys.path:
    sys.path.insert(0, _PYCMG_PYPATH)

from pycmg.tech import TECH_REGISTRY as _PYCMG_REGISTRY, resolve_modelcard

# Default temperature
DEFAULT_TEMPERATURE = 300.15  # 27°C in Kelvin

# Process parameter names used as NN input features (order matters!)
PROCESS_PARAM_NAMES = ["PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW"]


@dataclass
class ProcessParams:
    """BSIM-CMG process parameters used as NN input features.

    These 7 parameters are the top discriminators across technologies
    and device variants, based on sensitivity analysis (CV, uniqueness).
    """
    phig: float   # Gate workfunction [V] — variant discriminator
    u0: float     # Low-field mobility [m²/(V·s)] — strongest overall discriminator
    vsat: float   # Saturation velocity [m/s] — tech discriminator
    eot: float    # Equivalent oxide thickness [m] — tech node discriminator
    eta0: float   # DIBL coefficient — high variation
    cit: float    # Interface trap charge [F/m] — moderate variation
    rdsw: float   # S/D parasitic resistance [Ω·μm] — ASAP7 vs TSMC

    def as_array(self) -> list:
        """Return as ordered list matching PROCESS_PARAM_NAMES."""
        return [self.phig, self.u0, self.vsat, self.eot,
                self.eta0, self.cit, self.rdsw]

    def as_dict(self) -> Dict[str, float]:
        """Return as dict keyed by lowercase parameter name."""
        return {
            "phig": self.phig, "u0": self.u0, "vsat": self.vsat,
            "eot": self.eot, "eta0": self.eta0, "cit": self.cit,
            "rdsw": self.rdsw,
        }


@dataclass
class NNVariantConfig:
    """NN-specific variant config: pairs a PyCMG device with process params."""
    name: str
    nmos_process: ProcessParams
    pmos_process: ProcessParams

    def get_process_params(self, device_type: str) -> ProcessParams:
        return self.nmos_process if device_type == "nmos" else self.pmos_process


@dataclass
class NNTechConfig:
    """NN training config wrapping PyCMG's tech registry.

    Delegates device structure (model names, modelcard resolution) to PyCMG's
    TECH_REGISTRY. Stores NN-specific data: training VDD, ProcessParams,
    and fixed L values (NN was trained with specific L, not auto-detected).
    """
    pycmg_name: str
    vdd_train: float
    nfin_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 15, 20])
    temperature: float = DEFAULT_TEMPERATURE
    variants: Dict[str, NNVariantConfig] = field(default_factory=dict)
    default_variant: str = ""
    L_nmos: Optional[float] = None
    L_pmos: Optional[float] = None

    @property
    def name(self) -> str:
        return self.pycmg_name

    @property
    def pycmg_tech(self):
        return _PYCMG_REGISTRY[self.pycmg_name]

    @property
    def vdd(self) -> float:
        return self.vdd_train

    @property
    def tfin(self) -> float:
        return self.pycmg_tech.tfin

    @property
    def L(self) -> float:
        """Default L (uses NMOS L if asymmetric)."""
        if self.L_nmos is not None:
            return self.L_nmos
        if self.L_pmos is not None:
            return self.L_pmos
        return 7e-9

    def get_L(self, device_type: str) -> float:
        if device_type == "nmos" and self.L_nmos is not None:
            return self.L_nmos
        if device_type == "pmos" and self.L_pmos is not None:
            return self.L_pmos
        return self.L

    def get_model_name(self, device_type: str,
                       variant: Optional[str] = None) -> str:
        """Get model name from PyCMG registry."""
        vname = variant or self.default_variant
        canon = f"{device_type}_{vname}"
        dev = self.pycmg_tech.get_device(canon)
        return dev.model_name

    def get_modelcard_path(self, device_type: str,
                           variant: Optional[str] = None) -> str:
        """Resolve modelcard path.

        Priority:
        1. ASAP7: static modelcard from DeviceConfig.modelcard
        2. TSMC: pre-baked naive modelcard in submodule
        3. Fallback: resolve_modelcard (needs full PDK file)
        """
        vname = variant or self.default_variant
        canon = f"{device_type}_{vname}"
        dev = self.pycmg_tech.get_device(canon)
        L = self.get_L(device_type)

        if dev.modelcard is not None:
            return str(PYCMG_DIR / dev.modelcard)

        if dev.pdk_device is not None:
            L_nm = int(L * 1e9)
            naive_path = (PYCMG_DIR / "modelcards" / self.pycmg_name
                          / "naive" / f"{dev.pdk_device}_l{L_nm}nm.l")
            if naive_path.exists():
                return str(naive_path)

        return resolve_modelcard(dev, self.pycmg_tech, L)

    def get_phig(self, device_type: str,
                 variant: Optional[str] = None) -> float:
        return self.get_process_params(device_type, variant).phig

    def get_process_params(self, device_type: str,
                           variant: Optional[str] = None) -> ProcessParams:
        vname = variant or self.default_variant
        if vname and vname in self.variants:
            return self.variants[vname].get_process_params(device_type)
        raise ValueError(
            f"No process params for tech={self.name}, "
            f"device={device_type}, variant={vname}")


# ============================================================================
# Pre-defined technology configs with process parameters
# Process params extracted from modelcards via sensitivity analysis
# ============================================================================

ASAP7_CONFIG = NNTechConfig(
    pycmg_name="ASAP7",
    vdd_train=0.7,
    L_nmos=7e-9, L_pmos=7e-9,
    default_variant="rvt",
    variants={
        "rvt": NNVariantConfig("rvt",
            nmos_process=ProcessParams(phig=4.372, u0=0.0252, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.8108, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.307, u0=0.0283, vsat=70000.0, eot=1.0e-9, eta0=0.068, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.8681, u0=0.0227, vsat=60000.0, eot=1.0e-9, eta0=0.093, cit=0.0, rdsw=200.0)),
        "slvt": NNVariantConfig("slvt",
            nmos_process=ProcessParams(phig=4.2466, u0=0.0303, vsat=70000.0, eot=1.0e-9, eta0=0.070, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.9278, u0=0.0237, vsat=60000.0, eot=1.0e-9, eta0=0.094, cit=0.0, rdsw=200.0)),
        "sram": NNVariantConfig("sram",
            nmos_process=ProcessParams(phig=4.45, u0=0.025, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.78, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0)),
    },
)

TSMC5_CONFIG = NNTechConfig(
    pycmg_name="TSMC5",
    vdd_train=0.65,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.534, u0=0.0369, vsat=61358.327, eot=1.06e-9, eta0=0.0309, cit=-3.17e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.56, u0=0.1288, vsat=46420.011, eot=1.10e-9, eta0=0.0097, cit=-6.66e-4, rdsw=17.0)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.41, u0=0.0328, vsat=65370.07, eot=1.06e-9, eta0=0.0052, cit=-9.81e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.671, u0=0.0655, vsat=58093.912, eot=1.10e-9, eta0=0.0072, cit=-1.6e-3, rdsw=17.0)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.361, u0=0.0736, vsat=65204.0, eot=1.06e-9, eta0=0.0569, cit=-1.409e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.735, u0=0.0570, vsat=52073.0, eot=1.10e-9, eta0=5.25e-4, cit=-1.428e-3, rdsw=17.0)),
        "elvt": NNVariantConfig("elvt",
            nmos_process=ProcessParams(phig=4.361, u0=0.0519, vsat=39832.0, eot=1.06e-9, eta0=-2.104e-3, cit=-1.591e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.735, u0=0.0567, vsat=48826.0, eot=1.10e-9, eta0=-1.054e-3, cit=-1.428e-3, rdsw=17.0)),
    },
)

TSMC7_CONFIG = NNTechConfig(
    pycmg_name="TSMC7",
    vdd_train=0.75,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.461, u0=0.1444, vsat=122241.86, eot=1.16e-9, eta0=0.0167, cit=2.25e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.63088, u0=0.1187, vsat=44499.92, eot=1.11e-9, eta0=0.0135, cit=-1.68e-4, rdsw=17.0)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.402, u0=0.1167, vsat=62485.775, eot=1.16e-9, eta0=-0.0128, cit=9.42e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.692727, u0=0.1895, vsat=106620.77, eot=1.11e-9, eta0=0.0241, cit=-1.2e-3, rdsw=17.0)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.347, u0=0.1168, vsat=53284.0, eot=1.16e-9, eta0=-0.0128, cit=9.42e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.740, u0=0.1021, vsat=78600.0, eot=1.11e-9, eta0=0.035, cit=-1.15e-3, rdsw=17.0)),
    },
)

TSMC12_CONFIG = NNTechConfig(
    pycmg_name="TSMC12",
    vdd_train=0.80,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.51, u0=0.0921, vsat=69148.329, eot=1.46e-9, eta0=-0.2506, cit=-9.77e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.57, u0=0.0411, vsat=99928.366, eot=1.42e-9, eta0=0.0636, cit=-1.4e-3, rdsw=17.0)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.4189, u0=0.099, vsat=69307.07, eot=1.46e-9, eta0=-0.0473, cit=2.5e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.665, u0=0.0584, vsat=182615.78, eot=1.42e-9, eta0=0.0274, cit=-4.1e-3, rdsw=17.0)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.330, u0=0.0825, vsat=76587.0, eot=1.46e-9, eta0=-3.817e-3, cit=-2.125e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.768, u0=0.1061, vsat=82119.0, eot=1.42e-9, eta0=-0.0707, cit=-3.279e-3, rdsw=17.0)),
        "hvt": NNVariantConfig("hvt",
            nmos_process=ProcessParams(phig=4.580, u0=0.0889, vsat=70621.0, eot=1.46e-9, eta0=-0.2518, cit=-3.389e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.520, u0=0.1323, vsat=70223.0, eot=1.42e-9, eta0=0.027, cit=-3.099e-3, rdsw=17.0)),
        "lnvt": NNVariantConfig("lnvt",
            nmos_process=ProcessParams(phig=4.250, u0=0.0834, vsat=63143.0, eot=1.46e-9, eta0=0.080, cit=8.936e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.882, u0=0.0629, vsat=61621.0, eot=1.42e-9, eta0=0.1664, cit=3.513e-3, rdsw=17.0)),
    },
)

TSMC16_CONFIG = NNTechConfig(
    pycmg_name="TSMC16",
    vdd_train=0.80,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.47, u0=0.2081, vsat=119428.57, eot=1.46e-9, eta0=-0.1429, cit=-2.6e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.57, u0=0.1368, vsat=75228.571, eot=1.42e-9, eta0=0.039, cit=-3.1e-3, rdsw=17.0)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.4189, u0=0.0418, vsat=44005.862, eot=1.46e-9, eta0=-0.0145, cit=-2.1e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.665, u0=0.0764, vsat=23571.429, eot=1.42e-9, eta0=-0.1494, cit=-2.0e-3, rdsw=17.0)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.330, u0=0.0733, vsat=74829.0, eot=1.46e-9, eta0=-0.0607, cit=-4.8e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.768, u0=0.0664, vsat=11991.0, eot=1.42e-9, eta0=-1.79e-3, cit=-4.307e-3, rdsw=17.0)),
        "hvt": NNVariantConfig("hvt",
            nmos_process=ProcessParams(phig=4.580, u0=0.1361, vsat=128229.0, eot=1.46e-9, eta0=-0.1914, cit=-2.017e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.520, u0=0.0753, vsat=88857.0, eot=1.42e-9, eta0=-0.010, cit=-3.419e-3, rdsw=17.0)),
        "lnvt": NNVariantConfig("lnvt",
            nmos_process=ProcessParams(phig=4.250, u0=0.0725, vsat=78557.0, eot=1.46e-9, eta0=0.1557, cit=6.064e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.882, u0=0.0783, vsat=60271.0, eot=1.42e-9, eta0=0.1243, cit=5.814e-3, rdsw=17.0)),
    },
)

# Registry of all technologies
TECH_CONFIGS: Dict[str, NNTechConfig] = {
    "asap7": ASAP7_CONFIG,
    "tsmc5": TSMC5_CONFIG,
    "tsmc7": TSMC7_CONFIG,
    "tsmc12": TSMC12_CONFIG,
    "tsmc16": TSMC16_CONFIG,
}

# Backward compatibility aliases — test files import these by name.
TechConfig = NNTechConfig
VariantConfig = NNVariantConfig


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

# Input column names (13 features: voltages + geometry + process params)
INPUT_COLUMNS = [
    "Vd", "Vg", "Vs", "Vb", "NFIN", "T",
    "PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW",
]
