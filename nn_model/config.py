"""Configuration for NN-based compact model training."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_MODEL_DIR = PROJECT_ROOT / "nn_model"
CHECKPOINT_DIR = NN_MODEL_DIR / "checkpoints"
DATA_DIR = NN_MODEL_DIR / "data" / "datasets"

# PyCMG paths (pycmg-wrapper: 21 device variants, 5 process nodes)
PYCMG_DIR = Path("/home/shenshan/pycmg-wrapper")
OSDI_PATH = str(PYCMG_DIR / "build-deep-verify" / "osdi" / "bsimcmg.osdi")

# ASAP7 technology config
ASAP7_MODELCARD = str(PYCMG_DIR / "tech_model_cards" / "ASAP7" / "7nm_TT_160803.pm")
ASAP7_VDD = 0.7
ASAP7_L = 7e-9  # 7nm drawn gate length (matching pycmg-wrapper)

# Default temperature
DEFAULT_TEMPERATURE = 300.15  # 27°C in Kelvin

# TSMC technology modelcard base
TSMC_MODELCARDS = PYCMG_DIR / "tech_model_cards"

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
class VariantConfig:
    """Device variant configuration (SVT, LVT, RVT, etc.)."""
    name: str
    nmos_model_name: str
    pmos_model_name: str
    nmos_process: ProcessParams
    pmos_process: ProcessParams
    # Optional per-variant modelcard paths (for TSMC where each variant has its own file)
    nmos_modelcard_path: Optional[str] = None
    pmos_modelcard_path: Optional[str] = None

    # --- Backward compatibility properties ---
    @property
    def nmos_phig(self) -> float:
        return self.nmos_process.phig

    @property
    def pmos_phig(self) -> float:
        return self.pmos_process.phig

    def get_model_name(self, device_type: str) -> str:
        return self.nmos_model_name if device_type == "nmos" else self.pmos_model_name

    def get_phig(self, device_type: str) -> float:
        return self.get_process_params(device_type).phig

    def get_process_params(self, device_type: str) -> ProcessParams:
        return self.nmos_process if device_type == "nmos" else self.pmos_process

    def get_modelcard_path(self, device_type: str) -> Optional[str]:
        if device_type == "nmos":
            return self.nmos_modelcard_path
        return self.pmos_modelcard_path


@dataclass
class TechConfig:
    """Technology-specific configuration for data generation."""
    name: str
    modelcard_path: str
    nmos_model_name: str
    pmos_model_name: str
    vdd: float
    L: float
    tfin: float = 6e-9  # Fin thickness [m] (ASAP7: 6.5nm, TSMC: 6nm)
    nfin_values: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 15, 20])
    temperature: float = DEFAULT_TEMPERATURE
    # For techs with separate NMOS/PMOS modelcard files
    nmos_modelcard_path: Optional[str] = None
    pmos_modelcard_path: Optional[str] = None
    # For techs with asymmetric L (TSMC: L_nmos != L_pmos)
    L_nmos: Optional[float] = None
    L_pmos: Optional[float] = None
    # Multi-variant support
    variants: Dict[str, VariantConfig] = field(default_factory=dict)
    default_variant: str = ""

    def get_L(self, device_type: str) -> float:
        """Get channel length for a specific device type."""
        if device_type == "nmos" and self.L_nmos is not None:
            return self.L_nmos
        if device_type == "pmos" and self.L_pmos is not None:
            return self.L_pmos
        return self.L

    def get_modelcard_path(self, device_type: str,
                           variant: Optional[str] = None) -> str:
        """Get modelcard path for a specific device type and variant."""
        if variant and variant in self.variants:
            vc = self.variants[variant]
            vpath = vc.get_modelcard_path(device_type)
            if vpath is not None:
                return vpath
        if device_type == "nmos" and self.nmos_modelcard_path is not None:
            return self.nmos_modelcard_path
        if device_type == "pmos" and self.pmos_modelcard_path is not None:
            return self.pmos_modelcard_path
        return self.modelcard_path

    def get_model_name(self, device_type: str,
                       variant: Optional[str] = None) -> str:
        """Get model name for a specific device type and variant."""
        if variant and variant in self.variants:
            return self.variants[variant].get_model_name(device_type)
        return self.nmos_model_name if device_type == "nmos" else self.pmos_model_name

    def get_phig(self, device_type: str, variant: Optional[str] = None) -> float:
        """Get PHIG for a specific device type and variant."""
        vname = variant or self.default_variant
        if vname and vname in self.variants:
            return self.variants[vname].get_phig(device_type)
        raise ValueError(f"No PHIG for tech={self.name}, device={device_type}, variant={vname}")

    def get_process_params(self, device_type: str,
                           variant: Optional[str] = None) -> ProcessParams:
        """Get process parameters for a specific device type and variant."""
        vname = variant or self.default_variant
        if vname and vname in self.variants:
            return self.variants[vname].get_process_params(device_type)
        raise ValueError(
            f"No process params for tech={self.name}, device={device_type}, variant={vname}")


# ============================================================================
# Pre-defined technology configs with process parameters
# Process params extracted from modelcards via sensitivity analysis
# ============================================================================

# --- ASAP7 (7nm academic PDK) ---
# All models in single file: 7nm_TT_160803.pm
# Symmetric L=7nm, VDD=0.7V, T=300.15K, TFIN=6.5nm
ASAP7_CONFIG = TechConfig(
    name="ASAP7",
    modelcard_path=ASAP7_MODELCARD,
    nmos_model_name="nmos_rvt",
    pmos_model_name="pmos_rvt",
    vdd=ASAP7_VDD,
    L=ASAP7_L,
    tfin=6.5e-9,
    default_variant="rvt",
    variants={
        "rvt": VariantConfig(
            "rvt", "nmos_rvt", "pmos_rvt",
            nmos_process=ProcessParams(phig=4.372, u0=0.0252, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.8108, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0),
        ),
        "lvt": VariantConfig(
            "lvt", "nmos_lvt", "pmos_lvt",
            nmos_process=ProcessParams(phig=4.307, u0=0.0283, vsat=70000.0, eot=1.0e-9, eta0=0.068, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.8681, u0=0.0227, vsat=60000.0, eot=1.0e-9, eta0=0.093, cit=0.0, rdsw=200.0),
        ),
        "slvt": VariantConfig(
            "slvt", "nmos_slvt", "pmos_slvt",
            nmos_process=ProcessParams(phig=4.2466, u0=0.0303, vsat=70000.0, eot=1.0e-9, eta0=0.070, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.9278, u0=0.0237, vsat=60000.0, eot=1.0e-9, eta0=0.094, cit=0.0, rdsw=200.0),
        ),
        "sram": VariantConfig(
            "sram", "nmos_sram", "pmos_sram",
            nmos_process=ProcessParams(phig=4.45, u0=0.025, vsat=70000.0, eot=1.0e-9, eta0=0.062, cit=0.0, rdsw=200.0),
            pmos_process=ProcessParams(phig=4.78, u0=0.0209, vsat=60000.0, eot=1.0e-9, eta0=0.090, cit=0.0, rdsw=200.0),
        ),
    },
)

# --- TSMC5 (5nm FinFET) ---
# Asymmetric L: NMOS=16nm, PMOS=20nm, VDD=0.65V
_tsmc5_naive = TSMC_MODELCARDS / "TSMC5" / "naive"
TSMC5_CONFIG = TechConfig(
    name="TSMC5",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_svt_mac",
    vdd=0.65,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(_tsmc5_naive / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(_tsmc5_naive / "pch_svt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
    default_variant="svt",
    variants={
        "svt": VariantConfig(
            "svt", "nch_svt_mac", "pch_svt_mac",
            nmos_process=ProcessParams(phig=4.534, u0=0.0369, vsat=61358.327, eot=1.06e-9, eta0=0.0309, cit=-3.17e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.56, u0=0.1288, vsat=46420.011, eot=1.10e-9, eta0=0.0097, cit=-6.66e-4, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc5_naive / "nch_svt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc5_naive / "pch_svt_mac_l20nm.l"),
        ),
        "lvt": VariantConfig(
            "lvt", "nch_lvt_mac", "pch_lvt_mac",
            nmos_process=ProcessParams(phig=4.41, u0=0.0328, vsat=65370.07, eot=1.06e-9, eta0=0.0052, cit=-9.81e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.671, u0=0.0655, vsat=58093.912, eot=1.10e-9, eta0=0.0072, cit=-1.6e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc5_naive / "nch_lvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc5_naive / "pch_lvt_mac_l20nm.l"),
        ),
        "ulvt": VariantConfig(
            "ulvt", "nch_ulvt_mac", "pch_ulvt_mac",
            nmos_process=ProcessParams(phig=4.361, u0=0.0736, vsat=65204.0, eot=1.06e-9, eta0=0.0569, cit=-1.409e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.735, u0=0.0570, vsat=52073.0, eot=1.10e-9, eta0=5.25e-4, cit=-1.428e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc5_naive / "nch_ulvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc5_naive / "pch_ulvt_mac_l20nm.l"),
        ),
        "elvt": VariantConfig(
            "elvt", "nch_elvt_mac", "pch_elvt_mac",
            nmos_process=ProcessParams(phig=4.361, u0=0.0519, vsat=39832.0, eot=1.06e-9, eta0=-2.104e-3, cit=-1.591e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.735, u0=0.0567, vsat=48826.0, eot=1.10e-9, eta0=-1.054e-3, cit=-1.428e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc5_naive / "nch_elvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc5_naive / "pch_elvt_mac_l20nm.l"),
        ),
    },
)

# --- TSMC7 (7nm FinFET) ---
# Asymmetric L: NMOS=16nm, PMOS=20nm, VDD=0.75V
_tsmc7_naive = TSMC_MODELCARDS / "TSMC7" / "naive"
TSMC7_CONFIG = TechConfig(
    name="TSMC7",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_svt_mac",
    vdd=0.75,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(_tsmc7_naive / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(_tsmc7_naive / "pch_svt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
    default_variant="svt",
    variants={
        "svt": VariantConfig(
            "svt", "nch_svt_mac", "pch_svt_mac",
            nmos_process=ProcessParams(phig=4.461, u0=0.1444, vsat=122241.86, eot=1.16e-9, eta0=0.0167, cit=2.25e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.63088, u0=0.1187, vsat=44499.92, eot=1.11e-9, eta0=0.0135, cit=-1.68e-4, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc7_naive / "nch_svt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc7_naive / "pch_svt_mac_l20nm.l"),
        ),
        "lvt": VariantConfig(
            "lvt", "nch_lvt_mac", "pch_lvt_mac",
            nmos_process=ProcessParams(phig=4.402, u0=0.1167, vsat=62485.775, eot=1.16e-9, eta0=-0.0128, cit=9.42e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.692727, u0=0.1895, vsat=106620.77, eot=1.11e-9, eta0=0.0241, cit=-1.2e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc7_naive / "nch_lvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc7_naive / "pch_lvt_mac_l20nm.l"),
        ),
        "ulvt": VariantConfig(
            "ulvt", "nch_ulvt_mac", "pch_ulvt_mac",
            nmos_process=ProcessParams(phig=4.347, u0=0.1168, vsat=53284.0, eot=1.16e-9, eta0=-0.0128, cit=9.42e-4, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.740, u0=0.1021, vsat=78600.0, eot=1.11e-9, eta0=0.035, cit=-1.15e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc7_naive / "nch_ulvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc7_naive / "pch_ulvt_mac_l20nm.l"),
        ),
    },
)

# --- TSMC12 (12nm FinFET, also called 14nm equivalent) ---
# Asymmetric L: NMOS=16nm, PMOS=20nm, VDD=0.80V
_tsmc12_naive = TSMC_MODELCARDS / "TSMC12" / "naive"
TSMC12_CONFIG = TechConfig(
    name="TSMC12",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_svt_mac",
    vdd=0.80,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(_tsmc12_naive / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(_tsmc12_naive / "pch_svt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
    default_variant="svt",
    variants={
        "svt": VariantConfig(
            "svt", "nch_svt_mac", "pch_svt_mac",
            nmos_process=ProcessParams(phig=4.51, u0=0.0921, vsat=69148.329, eot=1.46e-9, eta0=-0.2506, cit=-9.77e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.57, u0=0.0411, vsat=99928.366, eot=1.42e-9, eta0=0.0636, cit=-1.4e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc12_naive / "nch_svt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc12_naive / "pch_svt_mac_l20nm.l"),
        ),
        "lvt": VariantConfig(
            "lvt", "nch_lvt_mac", "pch_lvt_mac",
            nmos_process=ProcessParams(phig=4.4189, u0=0.099, vsat=69307.07, eot=1.46e-9, eta0=-0.0473, cit=2.5e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.665, u0=0.0584, vsat=182615.78, eot=1.42e-9, eta0=0.0274, cit=-4.1e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc12_naive / "nch_lvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc12_naive / "pch_lvt_mac_l20nm.l"),
        ),
        "ulvt": VariantConfig(
            "ulvt", "nch_ulvt_mac", "pch_ulvt_mac",
            nmos_process=ProcessParams(phig=4.330, u0=0.0825, vsat=76587.0, eot=1.46e-9, eta0=-3.817e-3, cit=-2.125e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.768, u0=0.1061, vsat=82119.0, eot=1.42e-9, eta0=-0.0707, cit=-3.279e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc12_naive / "nch_ulvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc12_naive / "pch_ulvt_mac_l20nm.l"),
        ),
        "hvt": VariantConfig(
            "hvt", "nch_hvt_mac", "pch_hvt_mac",
            nmos_process=ProcessParams(phig=4.580, u0=0.0889, vsat=70621.0, eot=1.46e-9, eta0=-0.2518, cit=-3.389e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.520, u0=0.1323, vsat=70223.0, eot=1.42e-9, eta0=0.027, cit=-3.099e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc12_naive / "nch_hvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc12_naive / "pch_hvt_mac_l20nm.l"),
        ),
        "lnvt": VariantConfig(
            "lnvt", "nch_lnvt_mac", "pch_lnvt_mac",
            nmos_process=ProcessParams(phig=4.250, u0=0.0834, vsat=63143.0, eot=1.46e-9, eta0=0.080, cit=8.936e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.882, u0=0.0629, vsat=61621.0, eot=1.42e-9, eta0=0.1664, cit=3.513e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc12_naive / "nch_lnvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc12_naive / "pch_lnvt_mac_l20nm.l"),
        ),
    },
)

# --- TSMC16 (16nm FinFET) ---
# Asymmetric L: NMOS=16nm, PMOS=20nm, VDD=0.80V
_tsmc16_naive = TSMC_MODELCARDS / "TSMC16" / "naive"
TSMC16_CONFIG = TechConfig(
    name="TSMC16",
    modelcard_path="",
    nmos_model_name="nch_svt_mac",
    pmos_model_name="pch_svt_mac",
    vdd=0.80,
    L=16e-9,
    L_nmos=16e-9,
    L_pmos=20e-9,
    nmos_modelcard_path=str(_tsmc16_naive / "nch_svt_mac_l16nm.l"),
    pmos_modelcard_path=str(_tsmc16_naive / "pch_svt_mac_l20nm.l"),
    nfin_values=[1, 2, 5, 10, 15, 20],
    default_variant="svt",
    variants={
        "svt": VariantConfig(
            "svt", "nch_svt_mac", "pch_svt_mac",
            nmos_process=ProcessParams(phig=4.47, u0=0.2081, vsat=119428.57, eot=1.46e-9, eta0=-0.1429, cit=-2.6e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.57, u0=0.1368, vsat=75228.571, eot=1.42e-9, eta0=0.039, cit=-3.1e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc16_naive / "nch_svt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc16_naive / "pch_svt_mac_l20nm.l"),
        ),
        "lvt": VariantConfig(
            "lvt", "nch_lvt_mac", "pch_lvt_mac",
            nmos_process=ProcessParams(phig=4.4189, u0=0.0418, vsat=44005.862, eot=1.46e-9, eta0=-0.0145, cit=-2.1e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.665, u0=0.0764, vsat=23571.429, eot=1.42e-9, eta0=-0.1494, cit=-2.0e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc16_naive / "nch_lvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc16_naive / "pch_lvt_mac_l20nm.l"),
        ),
        "ulvt": VariantConfig(
            "ulvt", "nch_ulvt_mac", "pch_ulvt_mac",
            nmos_process=ProcessParams(phig=4.330, u0=0.0733, vsat=74829.0, eot=1.46e-9, eta0=-0.0607, cit=-4.8e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.768, u0=0.0664, vsat=11991.0, eot=1.42e-9, eta0=-1.79e-3, cit=-4.307e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc16_naive / "nch_ulvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc16_naive / "pch_ulvt_mac_l20nm.l"),
        ),
        "hvt": VariantConfig(
            "hvt", "nch_hvt_mac", "pch_hvt_mac",
            nmos_process=ProcessParams(phig=4.580, u0=0.1361, vsat=128229.0, eot=1.46e-9, eta0=-0.1914, cit=-2.017e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.520, u0=0.0753, vsat=88857.0, eot=1.42e-9, eta0=-0.010, cit=-3.419e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc16_naive / "nch_hvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc16_naive / "pch_hvt_mac_l20nm.l"),
        ),
        "lnvt": VariantConfig(
            "lnvt", "nch_lnvt_mac", "pch_lnvt_mac",
            nmos_process=ProcessParams(phig=4.250, u0=0.0725, vsat=78557.0, eot=1.46e-9, eta0=0.1557, cit=6.064e-3, rdsw=15.0),
            pmos_process=ProcessParams(phig=4.882, u0=0.0783, vsat=60271.0, eot=1.42e-9, eta0=0.1243, cit=5.814e-3, rdsw=17.0),
            nmos_modelcard_path=str(_tsmc16_naive / "nch_lnvt_mac_l16nm.l"),
            pmos_modelcard_path=str(_tsmc16_naive / "pch_lnvt_mac_l20nm.l"),
        ),
    },
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

# Input column names (13 features: voltages + geometry + process params)
INPUT_COLUMNS = [
    "Vd", "Vg", "Vs", "Vb", "NFIN", "T",
    "PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW",
]
