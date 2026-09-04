"""Public release metadata must identify one version."""
from __future__ import annotations

import re
from pathlib import Path

from pycircuitsim import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_readme_release() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    match = re.search(r"Current release: \*\*V([^*]+)\*\*", readme)

    assert match is not None
    assert match.group(1) == __version__
