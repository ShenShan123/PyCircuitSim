"""
Abstract base class for all circuit components.

This module defines the Component ABC that all device models inherit from.
The Solver only knows about Components - it never contains device physics.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class Component(ABC):
    """
    Abstract base class for all circuit components.

    All devices (resistors, capacitors, sources, MOSFETs) inherit from this.
    The Solver only knows about Components - it never contains device physics.

    This enforces the separation principle: device physics are encapsulated
    in subclasses, while the solver works only with the Component interface.
    """

    def __init__(self, name: str, nodes: List[str], value: Any = None):
        """
        Initialize a component.

        Args:
            name: Component identifier (e.g., 'R1', 'M1')
            nodes: List of node names this component connects to
            value: Component value (resistance, capacitance, etc.)
        """
        self.name = name
        self.nodes = nodes
        self.value = value

    @abstractmethod
    def get_nodes(self) -> List[str]:
        """
        Return list of node names this component connects to.

        Returns:
            List of node names (e.g., ['n1', 'n2'])
        """
        pass

    @abstractmethod
    def stamp_conductance(self, matrix: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add conductance terms to the MNA matrix (G part).

        This method stamps the component's conductance contributions to the
        Modified Nodal Analysis matrix. For linear components, this is fixed.
        For non-linear components (MOSFETs), this changes with operating point.

        Args:
            matrix: The MNA matrix to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        pass

    @abstractmethod
    def stamp_rhs(self, rhs: np.ndarray, node_map: Dict[str, int]) -> None:
        """
        Add current/source terms to the RHS vector (z part).

        This method stamps the component's contributions to the right-hand side
        of the MNA equation Ax = b.

        Args:
            rhs: The RHS vector to modify (in-place)
            node_map: Mapping from node names to matrix indices
        """
        pass

    @abstractmethod
    def calculate_current(self, voltages: Dict[str, float]) -> float:
        """
        Calculate device current given terminal voltages.

        Args:
            voltages: Dictionary mapping node names to voltages

        Returns:
            Current flowing through the device (conventional current direction:
            from first node to second node for 2-terminal devices)
        """
        pass

    def __repr__(self) -> str:
        """String representation of the component."""
        return f"{self.__class__.__name__}({self.name}, nodes={self.nodes})"
