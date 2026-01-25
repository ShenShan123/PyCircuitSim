"""
HSPICE-like netlist parser for PyCircuitSim.

This module provides the Parser class which reads .sp netlist files and
constructs Circuit objects with appropriate components. The parser supports
HSPICE-like syntax for components and analysis commands.

Supported components:
- Resistors: R<name> <n1> <n2> <value>
- Capacitors: C<name> <n1> <n2> <value>
- Voltage sources: V<name> <n+> <n-> <value>
- Current sources: I<name> <n+> <n-> <value>
- MOSFETs: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>

Supported analysis:
- DC sweep: .dc <source> <start> <stop> <step>
- Transient: .tran <tstep> <tstop>

Value suffixes supported:
- k/K: kilo (1e3)
- u/U: micro (1e-6)
- n/N: nano (1e-9)
- p/P: pico (1e-12)
"""
from typing import Dict, Optional, Tuple
import re

from pycircuitsim.circuit import Circuit
from pycircuitsim.models import (
    Resistor,
    Capacitor,
    VoltageSource,
    CurrentSource,
    NMOS,
    PMOS,
)


class Parser:
    """
    HSPICE-like netlist parser.

    The Parser reads .sp files line by line and constructs a Circuit object
    containing all components and analysis commands from the netlist.

    Attributes:
        circuit: Circuit object containing all parsed components
        analysis_type: Type of analysis ('dc', 'tran', or None)
        analysis_params: Dictionary of analysis parameters
    """

    # Unit suffix multipliers
    UNIT_SUFFIXES = {
        'k': 1e3,
        'K': 1e3,
        'u': 1e-6,
        'U': 1e-6,
        'n': 1e-9,
        'N': 1e-9,
        'p': 1e-12,
        'P': 1e-12,
    }

    def __init__(self):
        """Initialize an empty parser."""
        self.circuit = Circuit()
        self.analysis_type: Optional[str] = None
        self.analysis_params: Dict[str, float] = {}

    def parse_file(self, filename: str) -> None:
        """
        Parse a netlist file and populate the circuit.

        Reads the specified .sp file line by line, parsing each line to
        extract components and analysis commands.

        Args:
            filename: Path to the .sp netlist file

        Raises:
            FileNotFoundError: If the netlist file doesn't exist
            ValueError: If the netlist contains invalid syntax
        """
        with open(filename, 'r') as f:
            for line in f:
                self.parse_line(line.strip())

    def parse_line(self, line: str) -> None:
        """
        Parse a single line from the netlist.

        Dispatches to the appropriate parsing method based on the first
        character of the line. Ignores comments (lines starting with '*')
        and empty lines.

        Args:
            line: A single line from the netlist file

        Raises:
            ValueError: If the line contains invalid syntax
        """
        # Skip empty lines and comments
        if not line or line.startswith('*'):
            return

        # Skip .end directive
        if line.lower().startswith('.end'):
            return

        # Dispatch based on first character
        first_char = line[0].upper()

        if first_char == 'R':
            self._parse_resistor(line)
        elif first_char == 'C':
            self._parse_capacitor(line)
        elif first_char == 'V':
            self._parse_voltage_source(line)
        elif first_char == 'I':
            self._parse_current_source(line)
        elif first_char == 'M':
            self._parse_mosfet(line)
        elif line.startswith('.dc'):
            self._parse_dc(line)
        elif line.startswith('.tran'):
            self._parse_tran(line)
        # Ignore other directives (.option, .measure, etc.)

    def _parse_value(self, value_str: str) -> float:
        """
        Convert a value string with optional unit suffix to a float.

        Args:
            value_str: Value string (e.g., "1k", "10u", "3.3", "100p")

        Returns:
            Floating point value

        Examples:
            >>> parser._parse_value("1k")
            1000.0
            >>> parser._parse_value("10n")
            1e-08
            >>> parser._parse_value("3.3")
            3.3
        """
        # Check if the last character is a unit suffix
        if len(value_str) > 1 and value_str[-1] in self.UNIT_SUFFIXES:
            multiplier = self.UNIT_SUFFIXES[value_str[-1]]
            return float(value_str[:-1]) * multiplier

        # No suffix, just convert to float
        return float(value_str)

    def _parse_resistor(self, line: str) -> None:
        """
        Parse a resistor line: R<name> <n1> <n2> <value>.

        Args:
            line: Resistor definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid resistor syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]
        value = self._parse_value(parts[3])

        resistor = Resistor(name, nodes, value)
        self.circuit.add_component(resistor)

    def _parse_capacitor(self, line: str) -> None:
        """
        Parse a capacitor line: C<name> <n1> <n2> <value>.

        Args:
            line: Capacitor definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid capacitor syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]
        value = self._parse_value(parts[3])

        capacitor = Capacitor(name, nodes, value)
        self.circuit.add_component(capacitor)

    def _parse_voltage_source(self, line: str) -> None:
        """
        Parse a voltage source line: V<name> <n+> <n-> <value>.

        Args:
            line: Voltage source definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid voltage source syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]
        value = self._parse_value(parts[3])

        voltage_source = VoltageSource(name, nodes, value)
        self.circuit.add_component(voltage_source)

    def _parse_current_source(self, line: str) -> None:
        """
        Parse a current source line: I<name> <n+> <n-> <value>.

        Args:
            line: Current source definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid current source syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]
        value = self._parse_value(parts[3])

        current_source = CurrentSource(name, nodes, value)
        self.circuit.add_component(current_source)

    def _parse_mosfet(self, line: str) -> None:
        """
        Parse a MOSFET line: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>.

        Args:
            line: MOSFET definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        # MOSFET line format: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>
        # We need to handle L and W which are key=value pairs
        parts = line.split()

        if len(parts) < 7:
            raise ValueError(f"Invalid MOSFET syntax: {line}")

        name = parts[0]
        nodes = parts[1:5]  # [drain, gate, source, bulk]
        model = parts[5].upper()  # NMOS or PMOS

        # Extract L and W parameters
        L = None
        W = None

        for part in parts[6:]:
            if part.startswith('L='):
                L = self._parse_value(part[2:])
            elif part.startswith('W='):
                W = self._parse_value(part[2:])

        if L is None or W is None:
            raise ValueError(f"MOSFET missing L or W parameter: {line}")

        # Create appropriate transistor type
        if model == "NMOS":
            mosfet = NMOS(name, nodes, L=L, W=W)
        elif model == "PMOS":
            mosfet = PMOS(name, nodes, L=L, W=W)
        else:
            raise ValueError(f"Unknown MOSFET model: {model}")

        self.circuit.add_component(mosfet)

    def _parse_dc(self, line: str) -> None:
        """
        Parse a DC sweep analysis line: .dc <source> <start> <stop> <step>.

        Args:
            line: DC sweep analysis line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid .dc syntax: {line}")

        self.analysis_type = "dc"
        self.analysis_params = {
            "source": parts[1],
            "start": self._parse_value(parts[2]),
            "stop": self._parse_value(parts[3]),
            "step": self._parse_value(parts[4]),
        }

    def _parse_tran(self, line: str) -> None:
        """
        Parse a transient analysis line: .tran <tstep> <tstop>.

        Args:
            line: Transient analysis line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid .tran syntax: {line}")

        self.analysis_type = "tran"
        self.analysis_params = {
            "tstep": self._parse_value(parts[1]),
            "tstop": self._parse_value(parts[2]),
        }
