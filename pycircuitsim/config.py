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
    str(
        PROJECT_ROOT
        / "external_compact_models"
        / "bsim_cmg"
        / "build"
        / "osdi"
        / "bsimcmg.osdi"
    ),
)

# ASAP7 modelcard directory (for production PDK)
# Can be overridden by setting ASAP7_MODELCARD environment variable
ASAP7_MODELCARD_DIR = os.environ.get(
    "ASAP7_MODELCARD",
    str(PROJECT_ROOT / "PDKs" / "ASAP7"),
)

# Generic BSIM-CMG modelcard directory (for testing/benchmarks)
# These are the benchmark modelcards from the BSIM-CMG VA distribution
GENERIC_MODELCARD_DIR = str(
    PROJECT_ROOT
    / "external_compact_models"
    / "bsim_cmg"
    / "bsim-cmg-va"
    / "benchmark_test"
)

# Default temperature in Kelvin (27°C)
DEFAULT_TEMPERATURE = 300.15
