"""NN compact model configuration: process parameters and technology configs.

Defines ProcessParams, NNTechConfig, and the TECH_CONFIGS registry used by
nn_generate.py for training data generation.

Lives inside PyCMG so all data generation is self-contained in the submodule.
nn_model/config.py re-exports from here for backward compatibility.

Key design: no hardcoded ProcessParams per variant. Instead,
extract_process_params() reads them on-the-fly from the actual modelcard
for each (L, NFIN) bin during data generation. This ensures per-bin accuracy.

Geometry array layout (15 columns):
    [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]

Input feature vector (19 features = 4 voltages + 15 geometry):
    [Vd, Vg, Vs, Vb, log2(NFIN), L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW,
     CFS, TOXP, CGSL, UA, EU]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .tech import TECH_REGISTRY as _PYCMG_REGISTRY, resolve_modelcard, TechConfig
from .sweep import NN_OUTPUT_COLUMNS

# PyCMG root: two levels up from pycmg/nn_config.py  →  PyCMG/
PYCMG_DIR: Path = Path(__file__).resolve().parents[1]
OSDI_PATH: str = str(PYCMG_DIR / "build" / "osdi" / "bsimcmg.osdi")

DEFAULT_TEMPERATURE: float = 300.15  # 27 °C in Kelvin

# Process parameter names used as NN input features (12 total, order fixed)
PROCESS_PARAM_NAMES: List[str] = [
    "PHIG", "U0", "VSAT", "EOT", "ETA0", "CIT", "RDSW",
    "CFS", "TOXP", "CGSL", "UA", "EU",
]

# NN output targets — re-exported from sweep for convenience
OUTPUT_COLUMNS: List[str] = NN_OUTPUT_COLUMNS

# Full NN input feature names (19 total)
INPUT_COLUMNS: List[str] = [
    "Vd", "Vg", "Vs", "Vb",   # 4 terminal voltages (source-relative)
    "NFIN", "L", "T",          # 3 geometric / operating-condition scalars
    *PROCESS_PARAM_NAMES,       # 12 process parameters
]

# Default NFIN values for techs without PDK binning (e.g., ASAP7).
# NFIN=1 IS now included as part of Phase D6: nn_generate.py handles
# OSDI convergence failure per-bin via a smoke-test eval at instance
# creation, so unstable (variant, NFIN=1) bins are skipped cleanly while
# stable ones are kept. Variants known to fail include tsmc5:ulvt and
# tsmc16:lnvt — see CLAUDE.md "NFIN=1 causes convergence failure".
DEFAULT_NFIN_VALUES: List[int] = [1, 2, 3, 5, 10, 15, 20, 24]

# Default ASAP7 channel-length sweep used by Phase D7 — paper §3 trains
# on L ∈ [8, 240] nm. ASAP7 modelcards expose a single device-template L,
# so we override per-bin via the Instance L parameter using this list.
# TSMC variants enumerate L from PDK bin boundaries automatically.
DEFAULT_ASAP7_L_VALUES: List[float] = [
    7e-9, 14e-9, 30e-9, 60e-9, 120e-9, 240e-9,
]


@dataclass
class ProcessParams:
    """BSIM-CMG process parameters used as NN input features (12 total).

    These are NOT stored in config — they are extracted on-the-fly from
    modelcards via extract_process_params(). This dataclass is the canonical
    output format for the extraction.
    """
    phig: float   # Gate workfunction [V] — variant discriminator
    u0: float     # Low-field mobility [m²/(V·s)]
    vsat: float   # Saturation velocity [m/s]
    eot: float    # Equivalent oxide thickness [m]
    eta0: float   # DIBL coefficient
    cit: float    # Interface trap charge [F/m]
    rdsw: float   # S/D parasitic resistance [Ω·μm]
    cfs: float    # Fringing capacitance [F/m]
    toxp: float   # Physical oxide thickness [m]
    cgsl: float   # Gate-source overlap capacitance [F/m]
    ua: float     # Mobility degradation coefficient
    eu: float     # Mobility temperature exponent

    def as_array(self) -> List[float]:
        """Ordered list matching PROCESS_PARAM_NAMES (12 elements)."""
        return [self.phig, self.u0, self.vsat, self.eot,
                self.eta0, self.cit, self.rdsw,
                self.cfs, self.toxp, self.cgsl, self.ua, self.eu]

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(PROCESS_PARAM_NAMES, self.as_array()))


def extract_process_params(modelcard_params: Dict[str, float]) -> ProcessParams:
    """Extract the 12 NN process parameters from a parsed modelcard.

    Reads lowercase-keyed parameters from the modelcard (as returned by
    ``Model.modelcard_params`` or ``parse_modelcard().params``) and returns
    a ProcessParams instance. Missing parameters default to 0.0.

    This is the single source of truth for mapping modelcard → NN features.
    Both data generation and inference should use this function.

    Args:
        modelcard_params: Dict of modelcard parameters (lowercase keys),
            e.g., from ``Model(osdi, modelcard, name).modelcard_params``.

    Returns:
        ProcessParams with the 12 NN-relevant process parameters.
    """
    return ProcessParams(
        phig=modelcard_params.get("phig", 0.0),
        u0=modelcard_params.get("u0", 0.0),
        vsat=modelcard_params.get("vsat", 0.0),
        eot=modelcard_params.get("eot", 0.0),
        eta0=modelcard_params.get("eta0", 0.0),
        cit=modelcard_params.get("cit", 0.0),
        rdsw=modelcard_params.get("rdsw", 0.0),
        cfs=modelcard_params.get("cfs", 0.0),
        toxp=modelcard_params.get("toxp", 0.0),
        cgsl=modelcard_params.get("cgsl", 0.0),
        ua=modelcard_params.get("ua", 0.0),
        eu=modelcard_params.get("eu", 0.0),
    )


@dataclass
class NNTechConfig:
    """NN training config that wraps PyCMG's tech registry.

    Stores NN-specific data: training VDD, variant name list, temperature.
    Does NOT store ProcessParams — those come from the PDK:
      - Legal (L, NFIN) combos: ``DeviceConfig.get_geometry_combos(pdk_path)``
      - Process params: ``extract_process_params(model.modelcard_params)``

    For ASAP7 (no PDK binning), ``fallback_nfin_values`` and
    ``fallback_l_values`` provide explicit lists that get crossed to
    enumerate (L, NFIN) bins.
    """
    pycmg_name: str
    vdd_train: float
    variant_names: List[str]
    temperature: float = DEFAULT_TEMPERATURE
    default_variant: str = ""
    fallback_nfin_values: Optional[List[int]] = None
    fallback_l_values: Optional[List[float]] = None

    @property
    def name(self) -> str:
        return self.pycmg_name

    @property
    def pycmg_tech(self) -> TechConfig:
        return _PYCMG_REGISTRY[self.pycmg_name]

    @property
    def vdd(self) -> float:
        return self.vdd_train

    @property
    def tfin(self) -> float:
        return self.pycmg_tech.tfin

    def get_geometry_combos(
        self, device_type: str, variant: str,
    ) -> List[Tuple[float, float]]:
        """Return legal (L, NFIN) combinations for this tech/device/variant.

        For TSMC techs: delegates to DeviceConfig.get_geometry_combos(pdk_path),
            which scans the PDK file for bin boundaries.
        For ASAP7 (no PDK binning): crosses ``fallback_l_values`` with
            ``fallback_nfin_values``. If only one of the two is set, the
            other defaults to a sensible single-element list.

        D6: NFIN=1 is *not* filtered out here. Bins that fail OSDI
        convergence (e.g. tsmc5:ulvt NFIN=1) are dropped at the
        instance-creation smoke test in ``nn_generate.generate_dataset``.

        D7: ASAP7 ``fallback_l_values`` defaults to
        ``DEFAULT_ASAP7_L_VALUES`` (≈ paper's [8, 240] nm range) so the
        single-L modelcards still cover long-channel geometry via
        Instance-level L overrides.
        """
        dev = self.pycmg_tech.get_device(f"{device_type}_{variant}")
        pdk_path = self.pycmg_tech.pdk_path

        if dev.pdk_device is not None and pdk_path is not None:
            combos = dev.get_geometry_combos(pdk_path=pdk_path)
        else:
            nfin_list = self.fallback_nfin_values or DEFAULT_NFIN_VALUES
            l_list = (self.fallback_l_values
                      or DEFAULT_ASAP7_L_VALUES
                      or [dev.get_min_l(pdk_path)])
            combos = [(float(L), float(nfin))
                      for L in l_list
                      for nfin in nfin_list]

        return list(combos)

    def get_model_name(self, device_type: str, variant: str) -> str:
        """Get model name from PyCMG registry."""
        dev = self.pycmg_tech.get_device(f"{device_type}_{variant}")
        return dev.model_name

    def resolve_modelcard(
        self, device_type: str, variant: str,
        L: float, NFIN: Optional[float] = None,
    ) -> str:
        """Resolve the modelcard path for a specific (L, NFIN) bin.

        For ASAP7: returns the static modelcard (same for all L/NFIN).
        For TSMC: generates/caches an NFIN-aware naive modelcard.
        """
        dev = self.pycmg_tech.get_device(f"{device_type}_{variant}")
        return resolve_modelcard(dev, self.pycmg_tech, L, NFIN)


# ---------------------------------------------------------------------------
# Technology configs (5 process nodes)
# ---------------------------------------------------------------------------

ASAP7_CONFIG = NNTechConfig(
    pycmg_name="ASAP7",
    vdd_train=0.7,
    variant_names=["rvt", "lvt", "slvt", "sram"],
    default_variant="rvt",
    # D6: include NFIN=1; failures are caught per-bin in nn_generate.
    fallback_nfin_values=[1, 2, 3, 5, 10, 15, 20, 24],
    # D7: long-channel L sweep — paper §3 trains over [8, 240] nm.
    fallback_l_values=DEFAULT_ASAP7_L_VALUES,
)

TSMC5_CONFIG = NNTechConfig(
    pycmg_name="TSMC5",
    vdd_train=0.65,
    variant_names=["svt", "lvt", "ulvt", "elvt"],
    default_variant="svt",
)

TSMC6_CONFIG = NNTechConfig(
    pycmg_name="TSMC6",
    vdd_train=0.75,
    variant_names=["svt", "lvt", "ulvt"],
    default_variant="svt",
)

TSMC7_CONFIG = NNTechConfig(
    pycmg_name="TSMC7",
    vdd_train=0.75,
    variant_names=["svt", "lvt", "ulvt"],
    default_variant="svt",
)

TSMC12_CONFIG = NNTechConfig(
    pycmg_name="TSMC12",
    vdd_train=0.80,
    variant_names=["svt", "lvt", "ulvt", "hvt", "lnvt"],
    default_variant="svt",
)

TSMC16_CONFIG = NNTechConfig(
    pycmg_name="TSMC16",
    vdd_train=0.80,
    variant_names=["svt", "lvt", "ulvt", "hvt", "lnvt"],
    default_variant="svt",
)

TECH_CONFIGS: Dict[str, NNTechConfig] = {
    "asap7": ASAP7_CONFIG,
    "tsmc5": TSMC5_CONFIG,
    "tsmc6": TSMC6_CONFIG,
    "tsmc7": TSMC7_CONFIG,
    "tsmc12": TSMC12_CONFIG,
    "tsmc16": TSMC16_CONFIG,
}
