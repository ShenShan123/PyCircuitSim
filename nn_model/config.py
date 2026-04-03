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
# Union of DirectNet (7) and BSIM-AR (9) process params = 12 unique params
PROCESS_PARAM_NAMES = [
    "PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW",
    "CFS", "TOXP", "CGSL", "UA", "EU",
]


@dataclass
class ProcessParams:
    """BSIM-CMG process parameters used as NN input features.

    Union of DirectNet (7) and BSIM-AR (9) process params = 12 unique.
    Top discriminators across technologies and device variants.
    """
    # Original 7 (DirectNet)
    phig: float   # Gate workfunction [V] — variant discriminator
    u0: float     # Low-field mobility [m²/(V·s)] — strongest overall discriminator
    vsat: float   # Saturation velocity [m/s] — tech discriminator
    eot: float    # Equivalent oxide thickness [m] — tech node discriminator
    eta0: float   # DIBL coefficient — high variation
    cit: float    # Interface trap charge [F/m] — moderate variation
    rdsw: float   # S/D parasitic resistance [Ω·μm] — ASAP7 vs TSMC
    # Additional 5 (from BSIM-AR)
    cfs: float    # Fringing capacitance [F/m] — tech discriminator
    toxp: float   # Physical oxide thickness [m] — differs from EOT
    cgsl: float   # Gate-source overlap capacitance [F/m]
    ua: float     # Mobility degradation coefficient — high variation
    eu: float     # Mobility temperature exponent — high variation

    def as_array(self) -> list:
        """Return as ordered list matching PROCESS_PARAM_NAMES (12 elements)."""
        return [self.phig, self.u0, self.vsat, self.eot,
                self.eta0, self.cit, self.rdsw,
                self.cfs, self.toxp, self.cgsl, self.ua, self.eu]

    def as_dict(self) -> Dict[str, float]:
        """Return as dict keyed by lowercase parameter name (12 keys)."""
        return {
            "phig": self.phig, "u0": self.u0, "vsat": self.vsat,
            "eot": self.eot, "eta0": self.eta0, "cit": self.cit,
            "rdsw": self.rdsw, "cfs": self.cfs, "toxp": self.toxp,
            "cgsl": self.cgsl, "ua": self.ua, "eu": self.eu,
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
            nmos_process=ProcessParams(phig=4.372, u0=0.0252, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=0.55, eu=1.2),
            pmos_process=ProcessParams(phig=4.8108, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=1.133, eu=0.05)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.307, u0=0.0283, vsat=70000.0, eot=1.0e-9, eta0=0.068, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=0.55, eu=1.2),
            pmos_process=ProcessParams(phig=4.8681, u0=0.0227, vsat=60000.0, eot=1.0e-9, eta0=0.093, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=1.133, eu=0.05)),
        "slvt": NNVariantConfig("slvt",
            nmos_process=ProcessParams(phig=4.2466, u0=0.0303, vsat=70000.0, eot=1.0e-9, eta0=0.070, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=0.55, eu=1.2),
            pmos_process=ProcessParams(phig=4.9278, u0=0.0237, vsat=60000.0, eot=1.0e-9, eta0=0.094, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=1.133, eu=0.05)),
        "sram": NNVariantConfig("sram",
            nmos_process=ProcessParams(phig=4.45, u0=0.025, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=0.55, eu=1.2),
            pmos_process=ProcessParams(phig=4.78, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0,
                                       cfs=0.0, toxp=2.1e-9, cgsl=0.0, ua=1.133, eu=0.05)),
    },
)

TSMC5_CONFIG = NNTechConfig(
    pycmg_name="TSMC5",
    vdd_train=0.65,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.534, u0=0.0369, vsat=61358.327, eot=1.06e-9, eta0=0.0309, cit=-3.17e-4, rdsw=15.0,
                                       cfs=7.7005125e-18, toxp=8.774684e-10, cgsl=9.122016e-11, ua=1.2417607, eu=1.421806),
            pmos_process=ProcessParams(phig=4.56, u0=0.1288, vsat=46420.011, eot=1.10e-9, eta0=0.0097, cit=-6.66e-4, rdsw=17.0,
                                       cfs=8.14525625e-18, toxp=8.566677e-10, cgsl=1.365667e-10, ua=11.922426, eu=0.65684975)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.41, u0=0.0328, vsat=65370.07, eot=1.06e-9, eta0=0.0052, cit=-9.81e-4, rdsw=15.0,
                                       cfs=7.7005125e-18, toxp=8.872776e-10, cgsl=8.016279e-11, ua=0.93822431, eu=1.5863304),
            pmos_process=ProcessParams(phig=4.671, u0=0.0655, vsat=58093.912, eot=1.10e-9, eta0=0.0072, cit=-1.6e-3, rdsw=17.0,
                                       cfs=8.14525625e-18, toxp=8.724651e-10, cgsl=1.070484e-10, ua=5.3292601, eu=1.5768162)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.361, u0=0.0736, vsat=65204.0, eot=1.06e-9, eta0=0.0569, cit=-1.409e-3, rdsw=15.0,
                                       cfs=7.7005125e-18, toxp=8.837233e-10, cgsl=6.511122e-11, ua=3.5356256, eu=1.7378881),
            pmos_process=ProcessParams(phig=4.735, u0=0.0570, vsat=52073.0, eot=1.10e-9, eta0=5.25e-4, cit=-1.428e-3, rdsw=17.0,
                                       cfs=8.145256e-18, toxp=8.680445e-10, cgsl=7.715184e-11, ua=6.1840168, eu=0.97892949)),
        "elvt": NNVariantConfig("elvt",
            nmos_process=ProcessParams(phig=4.361, u0=0.0519, vsat=39832.0, eot=1.06e-9, eta0=-2.104e-3, cit=-1.591e-3, rdsw=15.0,
                                       cfs=7.7005125e-18, toxp=8.840481e-10, cgsl=6.344311e-11, ua=3.0286137, eu=1.2118134),
            pmos_process=ProcessParams(phig=4.735, u0=0.0567, vsat=48826.0, eot=1.10e-9, eta0=-1.054e-3, cit=-1.428e-3, rdsw=17.0,
                                       cfs=8.145256e-18, toxp=8.792781e-10, cgsl=7.412749e-11, ua=6.1840168, eu=0.97892949)),
    },
)

TSMC7_CONFIG = NNTechConfig(
    pycmg_name="TSMC7",
    vdd_train=0.75,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.461, u0=0.1444, vsat=122241.86, eot=1.16e-9, eta0=0.0167, cit=2.25e-4, rdsw=15.0,
                                       cfs=5.857141e-18, toxp=1.002253e-9, cgsl=1.357872e-10, ua=4.8823968, eu=1.0300001),
            pmos_process=ProcessParams(phig=4.63088, u0=0.1187, vsat=44499.92, eot=1.11e-9, eta0=0.0135, cit=-1.68e-4, rdsw=17.0,
                                       cfs=6.218746e-18, toxp=9.899438e-10, cgsl=1.765979e-10, ua=12.466666, eu=0.90666646)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.402, u0=0.1167, vsat=62485.775, eot=1.16e-9, eta0=-0.0128, cit=9.42e-4, rdsw=15.0,
                                       cfs=5.857141e-18, toxp=1.012111e-9, cgsl=1.112567e-10, ua=3.0542664, eu=1.0300001),
            pmos_process=ProcessParams(phig=4.692727, u0=0.1895, vsat=106620.77, eot=1.11e-9, eta0=0.0241, cit=-1.2e-3, rdsw=17.0,
                                       cfs=6.218746e-18, toxp=9.967284e-10, cgsl=1.411137e-10, ua=-0.7718089, eu=0.12665036)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.347, u0=0.1168, vsat=53284.0, eot=1.16e-9, eta0=-0.0128, cit=9.42e-4, rdsw=15.0,
                                       cfs=5.857141e-18, toxp=1.019443e-9, cgsl=1.072315e-10, ua=4.0116818, eu=0.99170409),
            pmos_process=ProcessParams(phig=4.740, u0=0.1021, vsat=78600.0, eot=1.11e-9, eta0=0.035, cit=-1.15e-3, rdsw=17.0,
                                       cfs=6.218746e-18, toxp=9.676798e-10, cgsl=1.237862e-10, ua=16.366669, eu=0.72999983)),
    },
)

TSMC12_CONFIG = NNTechConfig(
    pycmg_name="TSMC12",
    vdd_train=0.80,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.51, u0=0.0921, vsat=69148.329, eot=1.46e-9, eta0=-0.2506, cit=-9.77e-3, rdsw=15.0,
                                       cfs=2.068449e-10, toxp=1.043239e-9, cgsl=1.457558e-10, ua=1.8662313, eu=1.240302),
            pmos_process=ProcessParams(phig=4.57, u0=0.0411, vsat=99928.366, eot=1.42e-9, eta0=0.0636, cit=-1.4e-3, rdsw=17.0,
                                       cfs=2.23683e-10, toxp=9.075222e-10, cgsl=2.273192e-10, ua=-0.85888163, eu=3.5654541)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.4189, u0=0.099, vsat=69307.07, eot=1.46e-9, eta0=-0.0473, cit=2.5e-3, rdsw=15.0,
                                       cfs=2.068449e-10, toxp=1.02383e-9, cgsl=1.518138e-10, ua=0.82142857, eu=-0.51428571),
            pmos_process=ProcessParams(phig=4.665, u0=0.0584, vsat=182615.78, eot=1.42e-9, eta0=0.0274, cit=-4.1e-3, rdsw=17.0,
                                       cfs=2.23683e-10, toxp=9.3e-10, cgsl=2.211889e-10, ua=-2.0747107, eu=1.4714284)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.330, u0=0.0825, vsat=76587.0, eot=1.46e-9, eta0=-3.817e-3, cit=-2.125e-3, rdsw=15.0,
                                       cfs=2.068449e-10, toxp=1.056e-9, cgsl=1.617466e-10, ua=0.63767393, eu=1.836392),
            pmos_process=ProcessParams(phig=4.768, u0=0.1061, vsat=82119.0, eot=1.42e-9, eta0=-0.0707, cit=-3.279e-3, rdsw=17.0,
                                       cfs=2.23683e-10, toxp=9.670895e-10, cgsl=1.909258e-10, ua=2.0843168, eu=0.94564765)),
        "hvt": NNVariantConfig("hvt",
            nmos_process=ProcessParams(phig=4.580, u0=0.0889, vsat=70621.0, eot=1.46e-9, eta0=-0.2518, cit=-3.389e-3, rdsw=15.0,
                                       cfs=2.068449e-10, toxp=1.00893e-9, cgsl=1.7565e-10, ua=5.2888682, eu=1.5428571),
            pmos_process=ProcessParams(phig=4.520, u0=0.1323, vsat=70223.0, eot=1.42e-9, eta0=0.027, cit=-3.099e-3, rdsw=17.0,
                                       cfs=2.23683e-10, toxp=8.624149e-10, cgsl=2.444424e-10, ua=4.3632341, eu=-1.894734)),
        "lnvt": NNVariantConfig("lnvt",
            nmos_process=ProcessParams(phig=4.250, u0=0.0834, vsat=63143.0, eot=1.46e-9, eta0=0.080, cit=8.936e-3, rdsw=15.0,
                                       cfs=2.068449e-10, toxp=1.09318e-9, cgsl=9.96109e-11, ua=1.8142857, eu=1.1),
            pmos_process=ProcessParams(phig=4.882, u0=0.0629, vsat=61621.0, eot=1.42e-9, eta0=0.1664, cit=3.513e-3, rdsw=17.0,
                                       cfs=2.23683e-10, toxp=9.996244e-10, cgsl=2.196626e-10, ua=1.8377453, eu=1.2306612)),
    },
)

TSMC16_CONFIG = NNTechConfig(
    pycmg_name="TSMC16",
    vdd_train=0.80,
    L_nmos=16e-9, L_pmos=20e-9,
    default_variant="svt",
    variants={
        "svt": NNVariantConfig("svt",
            nmos_process=ProcessParams(phig=4.47, u0=0.2081, vsat=119428.57, eot=1.46e-9, eta0=-0.1429, cit=-2.6e-3, rdsw=15.0,
                                       cfs=2.2938e-10, toxp=1.07464e-9, cgsl=1.44757e-10, ua=6.4714286, eu=0.55),
            pmos_process=ProcessParams(phig=4.57, u0=0.1368, vsat=75228.571, eot=1.42e-9, eta0=0.039, cit=-3.1e-3, rdsw=17.0,
                                       cfs=2.47765e-10, toxp=9.02158e-10, cgsl=2.26086e-10, ua=6.0469183, eu=-1.4685402)),
        "lvt": NNVariantConfig("lvt",
            nmos_process=ProcessParams(phig=4.4189, u0=0.0418, vsat=44005.862, eot=1.46e-9, eta0=-0.0145, cit=-2.1e-3, rdsw=15.0,
                                       cfs=2.2938e-10, toxp=1.02383e-9, cgsl=1.55405e-10, ua=-0.20332625, eu=1.2868053),
            pmos_process=ProcessParams(phig=4.665, u0=0.0764, vsat=23571.429, eot=1.42e-9, eta0=-0.1494, cit=-2.0e-3, rdsw=17.0,
                                       cfs=2.47765e-10, toxp=9.19788e-10, cgsl=2.33539e-10, ua=0.79285384, eu=1.4428571)),
        "ulvt": NNVariantConfig("ulvt",
            nmos_process=ProcessParams(phig=4.330, u0=0.0733, vsat=74829.0, eot=1.46e-9, eta0=-0.0607, cit=-4.8e-3, rdsw=15.0,
                                       cfs=2.2938e-10, toxp=1.056e-9, cgsl=1.76193e-10, ua=1.161424, eu=2.7114289),
            pmos_process=ProcessParams(phig=4.768, u0=0.0664, vsat=11991.0, eot=1.42e-9, eta0=-1.79e-3, cit=-4.307e-3, rdsw=17.0,
                                       cfs=2.47765e-10, toxp=9.60661e-10, cgsl=2.26997e-10, ua=1.1242857, eu=1.1171429)),
        "hvt": NNVariantConfig("hvt",
            nmos_process=ProcessParams(phig=4.580, u0=0.1361, vsat=128229.0, eot=1.46e-9, eta0=-0.1914, cit=-2.017e-3, rdsw=15.0,
                                       cfs=2.2938e-10, toxp=1.00893e-9, cgsl=1.7565e-10, ua=5.3142857, eu=1.5428571),
            pmos_process=ProcessParams(phig=4.520, u0=0.0753, vsat=88857.0, eot=1.42e-9, eta0=-0.010, cit=-3.419e-3, rdsw=17.0,
                                       cfs=2.47765e-10, toxp=8.6452e-10, cgsl=2.44444e-10, ua=-1.4285714, eu=0.8)),
        "lnvt": NNVariantConfig("lnvt",
            nmos_process=ProcessParams(phig=4.250, u0=0.0725, vsat=78557.0, eot=1.46e-9, eta0=0.1557, cit=6.064e-3, rdsw=15.0,
                                       cfs=2.2938e-10, toxp=1.10518e-9, cgsl=9.96109e-11, ua=1.2714286, eu=1.2071429),
            pmos_process=ProcessParams(phig=4.882, u0=0.0783, vsat=60271.0, eot=1.42e-9, eta0=0.1243, cit=5.814e-3, rdsw=17.0,
                                       cfs=2.47765e-10, toxp=9.9065e-10, cgsl=2.38114e-10, ua=1.7357143, eu=1.1124776)),
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

# Input column names (18 features: voltages + geometry + 12 process params)
INPUT_COLUMNS = [
    "Vd", "Vg", "Vs", "Vb", "NFIN", "T",
    "PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW",
    "CFS", "TOXP", "CGSL", "UA", "EU",
]
