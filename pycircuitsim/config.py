"""Configuration for BSIM-CMG compact model integration.

This module defines default paths for OSDI binaries and modelcards,
with support for environment variable overrides.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# OSDI binary location for BSIM-CMG model
# Can be overridden by setting BSIMCMG_OSDI environment variable
BSIMCMG_OSDI_PATH = os.environ.get(
    "BSIMCMG_OSDI",
    str(PROJECT_ROOT / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi")
)

# ASAP7 modelcard directory (for production PDK)
# Can be overridden by setting ASAP7_MODELCARD environment variable
ASAP7_MODELCARD_DIR = os.environ.get(
    "ASAP7_MODELCARD",
    str(PROJECT_ROOT / "PyCMG" / "tech_model_cards" / "asap7_pdk_r1p7" / "models" / "hspice")
)

# Generic BSIM-CMG modelcard directory (for testing/benchmarks)
# These are the benchmark modelcards from the BSIM-CMG VA distribution
GENERIC_MODELCARD_DIR = str(PROJECT_ROOT / "PyCMG" / "bsim-cmg-va" / "benchmark_test")

# Default temperature in Kelvin (27°C)
DEFAULT_TEMPERATURE = 300.15


def verify_osdi_binary() -> bool:
    """Verify that the OSDI binary exists and is accessible.

    Returns:
        True if OSDI binary exists, False otherwise
    """
    osdi_path = Path(BSIMCMG_OSDI_PATH)
    return osdi_path.exists() and osdi_path.is_file()


def get_modelcard_path(modelcard_name: str, use_asap7: bool = True) -> str:
    """Get the full path to a modelcard file.

    Args:
        modelcard_name: Name of the modelcard file (e.g., "7nm_TT.pm" or "modelcard.nmos.1")
        use_asap7: If True, search ASAP7 directory first, else use generic directory

    Returns:
        Full path to the modelcard file

    Raises:
        FileNotFoundError: If modelcard file is not found
    """
    if use_asap7:
        asap7_path = Path(ASAP7_MODELCARD_DIR) / modelcard_name
        if asap7_path.exists():
            return str(asap7_path)

    generic_path = Path(GENERIC_MODELCARD_DIR) / modelcard_name
    if generic_path.exists():
        return str(generic_path)

    raise FileNotFoundError(
        f"Modelcard '{modelcard_name}' not found in ASAP7 ({ASAP7_MODELCARD_DIR}) "
        f"or generic ({GENERIC_MODELCARD_DIR}) directories"
    )
