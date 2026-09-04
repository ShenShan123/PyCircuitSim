"""Collect the simulator-free gate suites into the authoritative unit run.

Three ``verify_*.py`` entry points need no NGSPICE binary, no OSDI model, no
PDK card and no NN checkpoint: they are pure contract checks over the catalog,
the renderer and the campaign tooling. Until V7.6.9 they were reachable only by
running each script by hand, so ``python -m pytest -q tests`` — the run
``tests/README.md`` calls authoritative — reported green while several thousand
of their assertions had not executed.

They stay standalone scripts because a campaign operator runs them that way and
reads their one-line verdict. This module only adds them to the collected run;
it owns no assertions of its own, so a failure here is always a failure in the
suite named by the test, not in the wiring.

The NGSPICE- and checkpoint-dependent gates are deliberately *not* wired in:
collection must stay runnable on a checkout with no simulator and no private
PDK cards.
"""
from __future__ import annotations

from typing import Callable

import pytest

from tests.simple_circuits.verify_accuracy_campaign_tools import (
    main as accuracy_campaign_tools_main,
)
from tests.simple_circuits.verify_circuit_sweep_canaries import (
    main as circuit_sweep_canaries_main,
)
from tests.simple_circuits.verify_simple_circuit_catalog import (
    main as simple_circuit_catalog_main,
)


@pytest.mark.parametrize(
    ("suite", "entry_point"),
    (
        ("verify_simple_circuit_catalog", simple_circuit_catalog_main),
        ("verify_circuit_sweep_canaries", circuit_sweep_canaries_main),
        ("verify_accuracy_campaign_tools", accuracy_campaign_tools_main),
    ),
)
def test_simulator_free_gate_suite_passes(
    suite: str,
    entry_point: Callable[[], int],
) -> None:
    """A hermetic gate suite must return 0 from the collected unit run."""
    assert entry_point() == 0, f"{suite} reported failures; see captured output"
