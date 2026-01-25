"""
Circuit container class for managing circuit topology.

This module provides the Circuit class, which acts as a container for
all components in a circuit and manages node discovery and mapping.
The Circuit class is responsible for:
- Maintaining a list of all components
- Auto-discovering unique nodes from component connections
- Creating node mappings for MNA matrix construction
- Counting voltage sources for matrix sizing
"""
from typing import List, Set, Dict
from pycircuitsim.models.base import Component


class Circuit:
    """
    Container for circuit topology and components.

    The Circuit class manages all components in a circuit and provides
    methods for node discovery and mapping. It serves as the central
    data structure that the Solver uses to construct the MNA matrix.

    Attributes:
        components: List of all components in the circuit
        nodes: Set of all unique node names (including ground)
    """

    def __init__(self):
        """
        Initialize an empty circuit.

        Creates a new circuit with no components and no nodes.
        Components and nodes are added as components are registered.
        """
        self.components: List[Component] = []
        self.nodes: Set[str] = set()

    def add_component(self, component: Component) -> None:
        """
        Add a component to the circuit and auto-discover its nodes.

        When a component is added, all nodes it connects to are automatically
        added to the circuit's node set. This enables automatic topology
        discovery without manual node registration.

        Args:
            component: Component instance to add to the circuit
        """
        self.components.append(component)

        # Auto-discover nodes from the component
        for node in component.get_nodes():
            self.nodes.add(node)

    def get_nodes(self) -> List[str]:
        """
        Return list of unique nodes excluding ground.

        Ground nodes ("0" and "GND") are excluded from the returned list
        as they are treated as the reference potential (0V) and do not
        appear in the MNA matrix.

        Returns:
            List of node names (excluding ground), sorted for consistency
        """
        ground_nodes = {"0", "GND"}
        return sorted([node for node in self.nodes if node not in ground_nodes])

    def get_node_map(self) -> Dict[str, int]:
        """
        Create mapping from node names to matrix indices.

        This method creates a dictionary that maps each non-ground node name
        to a sequential integer index. This mapping is used by the Solver
        to stamp component contributions into the correct matrix positions.

        Ground nodes ("0" and "GND") are excluded from the mapping as they
        do not appear in the MNA matrix (ground is the reference).

        Returns:
            Dictionary mapping node names to indices (0-based sequential)
        """
        nodes = self.get_nodes()
        return {node: idx for idx, node in enumerate(nodes)}

    def count_voltage_sources(self) -> int:
        """
        Count the number of voltage sources in the circuit.

        Voltage sources require special handling in MNA formulation as they
        add rows and columns to the matrix (for the unknown currents through
        each voltage source). The Solver uses this count to determine the
        size of the augmented MNA matrix.

        Returns:
            Number of voltage source components in the circuit
        """
        from pycircuitsim.models.passive import VoltageSource

        count = 0
        for component in self.components:
            if isinstance(component, VoltageSource):
                count += 1
        return count

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns:
            String showing circuit statistics (component count, node count)
        """
        num_components = len(self.components)
        num_non_ground_nodes = len(self.get_nodes())
        return (
            f"Circuit(components={num_components}, "
            f"non_ground_nodes={num_non_ground_nodes})"
        )
