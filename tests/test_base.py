"""
Tests for the Component abstract base class.

This module verifies that the Component ABC enforces its interface properly.
"""
import pytest
from abc import ABC
from pycircuitsim.models.base import Component


def test_component_is_abstract():
    """Component should not be instantiable directly."""
    with pytest.raises(TypeError):
        Component()


def test_component_requires_interface():
    """Subclass must implement all abstract methods."""
    class IncompleteComponent(Component):
        def get_nodes(self):
            return ["n1", "n2"]

    with pytest.raises(TypeError):
        IncompleteComponent()


def test_complete_component_can_be_instantiated():
    """A complete implementation of Component should be instantiable."""
    import numpy as np

    class CompleteComponent(Component):
        """Mock implementation for testing."""
        def __init__(self, name: str, nodes: list[str]):
            self.name = name
            self._nodes = nodes

        def get_nodes(self) -> list[str]:
            return self._nodes

        def stamp_conductance(self, matrix: np.ndarray, node_map: dict) -> None:
            """Mock implementation - does nothing."""
            pass

        def stamp_rhs(self, rhs: np.ndarray, node_map: dict) -> None:
            """Mock implementation - does nothing."""
            pass

        def calculate_current(self, voltages: dict) -> float:
            """Mock implementation - returns 0."""
            return 0.0

    # Should not raise any exception
    comp = CompleteComponent("R1", ["n1", "n2"])
    assert comp.name == "R1"
    assert comp.get_nodes() == ["n1", "n2"]
