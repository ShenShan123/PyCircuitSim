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
- AC analysis: .ac <sweep_type> <num_points> <fstart> <fstop>

Supported directives:
- Initial conditions: .ic V(<node>)=<value> ...
- Model definitions: .model <name> <type> <params>
- Include files: .include <filename>

Value suffixes supported:
- k/K: kilo (1e3)
- u/U: micro (1e-6)
- n/N: nano (1e-9)
- p/P: pico (1e-12)
"""
from typing import Dict, Optional, Tuple
import re
from pathlib import Path

from pycircuitsim.circuit import Circuit
from pycircuitsim.models import (
    Resistor,
    Capacitor,
    VoltageSource,
    CurrentSource,
    NMOS,
    PMOS,
)
from pycircuitsim.config import BSIMCMG_OSDI_PATH, GENERIC_MODELCARD_DIR


class Parser:
    """
    HSPICE-like netlist parser.

    The Parser reads .sp files line by line and constructs a Circuit object
    containing all components and analysis commands from the netlist.

    Attributes:
        circuit: Circuit object containing all parsed components
        analysis_type: Type of analysis ('dc', 'tran', or None)
        analysis_params: Dictionary of analysis parameters
        models: Dictionary of model definitions (name -> type + params)
    """

    # Unit suffix multipliers
    UNIT_SUFFIXES = {
        't': 1e12,  # tera
        'T': 1e12,
        'g': 1e9,   # giga
        'G': 1e9,
        'm': 1e6,   # mega (milli is less common in circuits)
        'M': 1e6,
        'k': 1e3,   # kilo
        'K': 1e3,
        'u': 1e-6,  # micro
        'U': 1e-6,
        'n': 1e-9,  # nano
        'N': 1e-9,
        'p': 1e-12, # pico
        'P': 1e-12,
        'f': 1e-15, # femto
        'F': 1e-15,
    }

    def __init__(self, osdi_path: Optional[str] = None, modelcard_base_dir: Optional[str] = None):
        """Initialize an empty parser.

        Args:
            osdi_path: Path to BSIM-CMG OSDI binary (defaults to config value)
            modelcard_base_dir: Base directory for modelcard files (defaults to generic modelcards)
        """
        self.circuit = Circuit()
        self.analysis_type: Optional[str] = None
        self.analysis_params: Dict[str, float] = {}
        self.models: Dict[str, Dict[str, any]] = {}  # Model definitions
        self._osdi_path = osdi_path or BSIMCMG_OSDI_PATH
        self._modelcard_base_dir = modelcard_base_dir or GENERIC_MODELCARD_DIR

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
        # Store current file for .include resolution
        self._current_file = str(Path(filename).resolve())

        with open(filename, 'r') as f:
            lines = f.readlines()

        # First pass: handle line continuations and collect models/includes
        processed_lines = []
        continued_line = ""
        in_model = False  # Track if we're in a .model definition

        for raw_line in lines:
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('*'):
                continue

            # Handle line continuations (lines starting with '+')
            if line.startswith('+'):
                continuation = line[1:].strip()
                continuation = continuation.replace(' = ', '=').replace('= ', '=')
                import re
                continuation = re.sub(r'\s*=\s*', '=', continuation)
                continued_line += " " + continuation
                continue

            # If we have a continued line, add it to processed lines
            if continued_line:
                processed_lines.append(continued_line)
                continued_line = ""
                in_model = False

            # Check if this is a new .model or .include or analysis line
            if line.lower().startswith('.model') or line.lower().startswith('.include') or \
               line.lower().startswith('.dc') or line.lower().startswith('.tran') or \
               line.lower().startswith('.ac') or line.lower().startswith('.ic') or line.lower().startswith('.end'):
                line = line.replace(' = ', '=').replace('= ', '=')
                line = ' '.join(line.split())

                # For .model lines, start accumulating continuations
                if line.lower().startswith('.model'):
                    continued_line = line
                    in_model = True
                else:
                    processed_lines.append(line)
            else:
                # Regular line (component definition)
                line = line.replace(' = ', '=').replace('= ', '=')
                line = ' '.join(line.split())
                processed_lines.append(line)

        # Process any remaining continued line
        if continued_line:
            processed_lines.append(continued_line)

        # Pre-pass: collect all .model and .include directives first
        # This ensures models are available before components that reference them
        for line in processed_lines:
            if line.lower().startswith('.model'):
                self._parse_model(line)
            elif line.lower().startswith('.include'):
                # Includes may add more models, so process them
                self.parse_line(line)

        # Second pass: parse all remaining lines (components, analysis, etc.)
        for line in processed_lines:
            # Skip .model and .include (already processed)
            if not line.lower().startswith(('.model', '.include')):
                self.parse_line(line)

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
        elif line.startswith('.ac'):
            self._parse_ac(line)
        elif line.startswith('.ic'):
            self._parse_ic(line)
        elif line.lower().startswith('.model'):
            self._parse_model(line)
        elif line.lower().startswith('.include'):
            self._parse_include(line)
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
        Parse a voltage source line: V<name> <n+> <n-> <value> or V<name> <n+> <n-> PULSE <params>.

        Supports:
        - DC voltage source: V1 1 0 3.3
        - PULSE source: V1 1 0 PULSE 0 3.3 1n 0.1n 0.1n 5n 10n
        - AC voltage source: V1 1 0 DC=1.0 AC=0.1 0 (DC bias, AC magnitude, AC phase in degrees)

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

        # Check if it's a PULSE source
        if len(parts) >= 4 and parts[3].upper() == 'PULSE':
            # PULSE source: V1 n+ n- PULSE V1 V2 TD TR TF PW PER
            if len(parts) < 11:
                raise ValueError(f"PULSE source requires 8 parameters: {line}")

            from pycircuitsim.models.passive import PulseVoltageSource

            v1 = self._parse_value(parts[4])
            v2 = self._parse_value(parts[5])
            td = self._parse_value(parts[6])
            tr = self._parse_value(parts[7])
            tf = self._parse_value(parts[8])
            pw = self._parse_value(parts[9])
            per = self._parse_value(parts[10])

            pulse_source = PulseVoltageSource(name, nodes, v1, v2, td, tr, tf, pw, per)
            self.circuit.add_component(pulse_source)
        else:
            # Check if it's an AC specification: DC=x AC=y phase
            dc_value = None
            ac_magnitude = 0.0
            ac_phase = 0.0

            # Look for DC=, AC= keywords
            for i, part in enumerate(parts[3:], start=3):
                if part.upper().startswith('DC='):
                    dc_value = self._parse_value(part[3:])
                elif part.upper().startswith('AC='):
                    ac_magnitude = self._parse_value(part[3:])
                    # Check if phase follows AC magnitude
                    if i + 1 < len(parts) and not parts[i + 1].upper().startswith(('DC=', 'AC=')):
                        try:
                            ac_phase = float(parts[i + 1])
                        except ValueError:
                            pass  # Not a phase value, skip
                elif dc_value is None and not part.upper().startswith(('DC=', 'AC=')):
                    # No DC= keyword, treat first value as DC value
                    dc_value = self._parse_value(part)

            # Default DC value to 0 if only AC specified
            if dc_value is None:
                dc_value = 0.0

            voltage_source = VoltageSource(name, nodes, dc_value, ac_magnitude=ac_magnitude, ac_phase=ac_phase)
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
        Parse a MOSFET line: M<name> <d> <g> <s> <b> <model> L=<l> W=<w> [NFIN=<nf> ...].

        Supports both Level 1 (L, W) and Level 72/BSIM-CMG (L, NFIN, TFIN, HFIN, FPITCH).

        Args:
            line: MOSFET definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        # MOSFET line format: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>
        # BSIM-CMG format: M<name> <d> <g> <s> <b> <model> L=<l> NFIN=<nf> [TFIN=<tf>] ...
        parts = line.split()

        if len(parts) < 7:
            raise ValueError(f"Invalid MOSFET syntax: {line}")

        name = parts[0]
        nodes = parts[1:5]  # [drain, gate, source, bulk]
        model = parts[5].upper()  # NMOS or PMOS

        # Extract geometric parameters (Level 1: L, W; BSIM-CMG: L, NFIN, TFIN, HFIN, FPITCH)
        L = None
        W = None
        NFIN = None
        TFIN = None
        HFIN = None
        FPITCH = None

        for part in parts[6:]:
            if part.startswith('L='):
                L = self._parse_value(part[2:])
            elif part.startswith('W='):
                W = self._parse_value(part[2:])
            elif part.startswith('NFIN='):
                NFIN = float(part[5:])  # Number of fins (integer or float)
            elif part.startswith('TFIN='):
                TFIN = self._parse_value(part[5:])
            elif part.startswith('HFIN='):
                HFIN = self._parse_value(part[5:])
            elif part.startswith('FPITCH='):
                FPITCH = self._parse_value(part[7:])

        # L is always required
        if L is None:
            raise ValueError(f"MOSFET missing L parameter: {line}")

        # Check if model name references a .model definition
        model_name = parts[5]  # Keep case for model lookup

        # Look up model in .model definitions first
        # If not found, fall back to direct NMOS/PMOS keyword (backward compatibility)
        if model_name not in self.models:
            # No .model definition found, check if it's a direct NMOS/PMOS keyword
            if model_name.upper() == "NMOS":
                if W is None:
                    raise ValueError(f"Level 1 NMOS missing W parameter: {line}")
                mosfet = NMOS(name, nodes, L=L, W=W)
                self.circuit.add_component(mosfet)
                return
            elif model_name.upper() == "PMOS":
                if W is None:
                    raise ValueError(f"Level 1 PMOS missing W parameter: {line}")
                mosfet = PMOS(name, nodes, L=L, W=W)
                self.circuit.add_component(mosfet)
                return
            else:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

        model_def = self.models[model_name]
        model_type = model_def['type']
        model_params = model_def['params']

        # Check model level (1 = Level 1 Shichman-Hodges, 72 = BSIM-CMG)
        level = model_params.get('LEVEL', 1)

        if level == 1:
            # Level 1 Shichman-Hodges model
            if W is None:
                raise ValueError(f"Level 1 MOSFET missing W parameter: {line}")

            # Extract model parameters (VTO, KP, LAMBDA, etc.)
            # Use defaults from NMOS/PMOS class if not specified
            VTO = model_params.get('VTO', None)
            KP = model_params.get('KP', None)
            LAMBDA = model_params.get('LAMBDA', None)

            if model_type.upper() == 'NMOS':
                # Build kwargs dict with only non-None parameters
                kwargs = {'L': L, 'W': W}
                if VTO is not None:
                    kwargs['VTO'] = VTO
                if KP is not None:
                    kwargs['KP'] = KP
                mosfet = NMOS(name, nodes, **kwargs)
            elif model_type.upper() == 'PMOS':
                # Build kwargs dict with only non-None parameters
                kwargs = {'L': L, 'W': W}
                if VTO is not None:
                    kwargs['VTO'] = VTO
                if KP is not None:
                    kwargs['KP'] = KP
                mosfet = PMOS(name, nodes, **kwargs)
            else:
                raise ValueError(f"Unknown MOSFET model type: {model_type}")

        elif level == 72:
            # BSIM-CMG compact model
            if NFIN is None:
                raise ValueError(f"BSIM-CMG (LEVEL=72) MOSFET missing NFIN parameter: {line}")

            # Import BSIM-CMG models
            try:
                from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
            except ImportError as e:
                raise ImportError(
                    f"Failed to import BSIM-CMG models: {e}. "
                    "Ensure PyCMG is built and OSDI binary exists."
                )

            # Resolve modelcard path
            # For BSIM-CMG, the modelcard is embedded in the .model directive
            # We need to create a temporary modelcard file or use an existing one
            # For now, we'll use the model name to look up a modelcard file
            modelcard_filename = f"{model_name}.1"  # e.g., "nmos1.1"
            modelcard_path = Path(self._modelcard_base_dir) / f"modelcard.{model_type.lower()}.1"

            if not modelcard_path.exists():
                raise FileNotFoundError(
                    f"BSIM-CMG modelcard not found: {modelcard_path}. "
                    f"Expected modelcard file for model '{model_name}' in {self._modelcard_base_dir}"
                )

            if model_type.upper() == 'NMOS':
                mosfet = NMOS_CMG(
                    name=name,
                    nodes=nodes,
                    osdi_path=self._osdi_path,
                    modelcard_path=str(modelcard_path),
                    model_name=model_name,
                    L=L,
                    NFIN=NFIN,
                    TFIN=TFIN,
                    HFIN=HFIN,
                    FPITCH=FPITCH,
                )
            elif model_type.upper() == 'PMOS':
                mosfet = PMOS_CMG(
                    name=name,
                    nodes=nodes,
                    osdi_path=self._osdi_path,
                    modelcard_path=str(modelcard_path),
                    model_name=model_name,
                    L=L,
                    NFIN=NFIN,
                    TFIN=TFIN,
                    HFIN=HFIN,
                    FPITCH=FPITCH,
                )
            else:
                raise ValueError(f"Unknown MOSFET model type: {model_type}")

        else:
            raise ValueError(
                f"Unsupported MOSFET LEVEL={level}. "
                f"Supported levels: LEVEL=1 (Shichman-Hodges), LEVEL=72 (BSIM-CMG)"
            )

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

    def _parse_ac(self, line: str) -> None:
        """
        Parse an AC analysis line: .ac <sweep_type> <num_points> <fstart> <fstop>.

        Sweep types:
        - dec: decade sweep (logarithmic, num_points per decade)
        - lin: linear sweep (num_points total between fstart and fstop)
        - oct: octave sweep (logarithmic, num_points per octave)

        Args:
            line: AC analysis line (e.g., ".ac dec 10 1k 10e6")

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid .ac syntax: {line}")

        sweep_type = parts[1].lower()
        if sweep_type not in ['dec', 'lin', 'oct']:
            raise ValueError(f"Invalid AC sweep type: {sweep_type}. Must be 'dec', 'lin', or 'oct'")

        self.analysis_type = "ac"
        self.analysis_params = {
            "sweep_type": sweep_type,
            "num_points": int(parts[2]),
            "fstart": self._parse_value(parts[3]),
            "fstop": self._parse_value(parts[4]),
        }

    def _parse_ic(self, line: str) -> None:
        """
        Parse an initial condition line: .ic V(<node>)=<value> V(<node>)=<value> ...

        Sets initial voltages for specified nodes, which is useful for
        defining the initial state of bistable circuits like SRAM cells.

        Args:
            line: Initial condition line (e.g., ".ic V(2)=3.3 V(3)=0")

        Raises:
            ValueError: If the line has invalid syntax

        Examples:
            .ic V(2)=3.3 V(3)=0
            .ic V(node1)=1.8 V(node2)=0.5
        """
        # Remove ".ic" prefix
        ic_spec = line[3:].strip()

        # Pattern to match V(node)=value or V(node)=value, with multiple assignments
        # Supports: V(2)=3.3, V(2)=3.3 V(3)=0, V(node1)=1.8 V(node2)=0.5
        pattern = r'V\(\s*([^)]+)\s*\)\s*=\s*([0-9.eE+-]+[kKuUnNpP]?)'

        matches = re.findall(pattern, ic_spec)

        if not matches:
            raise ValueError(f"Invalid .ic syntax: {line}")

        for node_str, value_str in matches:
            node = node_str.strip()
            value = self._parse_value(value_str)
            self.circuit.initial_conditions[node] = value

    def _parse_model(self, line: str) -> None:
        """
        Parse a .model line: .model <name> NMOS/PMOS <params>

        Args:
            line: Model definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        # Remove ".model" prefix and get parts
        model_spec = line[6:].strip()

        # Remove parentheses if present (HSPICE style: .model name TYPE (params))
        model_spec = model_spec.replace('(', ' ').replace(')', ' ')
        parts = model_spec.split()

        if len(parts) < 2:
            raise ValueError(f"Invalid .model syntax: {line}")

        model_name = parts[0]
        model_type = parts[1].upper()

        # Parse parameters (supports key=value format)
        params = {}
        for part in parts[2:]:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key and value:  # Skip empty keys or values
                    try:
                        params[key.upper()] = self._parse_value(value)
                    except ValueError:
                        # Skip parameters that can't be parsed (e.g., empty values)
                        pass

        # Store model definition
        self.models[model_name] = {
            'type': model_type,
            'params': params
        }

    def _parse_include(self, line: str) -> None:
        """
        Parse an .include directive: .include <filename>

        Args:
            line: Include directive line

        Raises:
            ValueError: If the line has invalid syntax
            FileNotFoundError: If the included file doesn't exist
        """
        # Remove ".include" prefix
        include_spec = line[8:].strip()
        included_file = include_spec.strip('"\'')  # Remove quotes

        # Resolve path relative to current file
        current_file = getattr(self, '_current_file', None)
        if current_file:
            current_dir = Path(current_file).parent
            included_path = current_dir / included_file
        else:
            included_path = Path(included_file)

        if not included_path.exists():
            raise FileNotFoundError(f"Included file not found: {included_path}")

        # Parse the included file using parse_file (handles line continuations)
        self.parse_file(str(included_path))
