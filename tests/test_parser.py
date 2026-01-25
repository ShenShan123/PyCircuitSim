"""
Tests for the netlist parser.

This module tests the Parser class which reads HSPICE-like .sp files
and constructs Circuit objects with appropriate components.
"""
import pytest
from pycircuitsim.parser import Parser
from pycircuitsim.models import Resistor, VoltageSource, NMOS, PMOS, Capacitor


class TestParserVoltageSource:
    """Tests for parsing voltage sources."""

    def test_parse_dc_voltage_source(self):
        """Test parsing a simple DC voltage source."""
        parser = Parser()
        parser.parse_line("V1 1 0 5.0")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert isinstance(component, VoltageSource)
        assert component.name == "V1"
        assert component.nodes == ["1", "0"]
        assert component.voltage == 5.0

    def test_parse_voltage_source_with_decimals(self):
        """Test parsing voltage source with decimal value."""
        parser = Parser()
        parser.parse_line("Vdd vdd 0 3.3")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert isinstance(component, VoltageSource)
        assert component.name == "Vdd"
        assert component.nodes == ["vdd", "0"]
        assert component.voltage == 3.3


class TestParserResistor:
    """Tests for parsing resistors."""

    def test_parse_resistor_basic(self):
        """Test parsing a basic resistor."""
        parser = Parser()
        parser.parse_line("R1 1 2 1000")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert isinstance(component, Resistor)
        assert component.name == "R1"
        assert component.nodes == ["1", "2"]
        assert component.resistance == 1000.0

    def test_parse_resistor_with_k_suffix(self):
        """Test parsing resistor with 'k' suffix (kilo)."""
        parser = Parser()
        parser.parse_line("R1 1 2 10k")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert component.resistance == 10e3


class TestParserCapacitor:
    """Tests for parsing capacitors."""

    def test_parse_capacitor_basic(self):
        """Test parsing a basic capacitor."""
        parser = Parser()
        parser.parse_line("C1 1 2 1e-9")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert isinstance(component, Capacitor)
        assert component.name == "C1"
        assert component.nodes == ["1", "2"]
        assert component.capacitance == 1e-9

    def test_parse_capacitor_with_p_suffix(self):
        """Test parsing capacitor with 'p' suffix (pico)."""
        parser = Parser()
        parser.parse_line("C1 1 2 100p")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert component.capacitance == 100e-12


class TestParserMOSFET:
    """Tests for parsing MOSFET transistors."""

    def test_parse_nmos_basic(self):
        """Test parsing a basic NMOS transistor."""
        parser = Parser()
        parser.parse_line("M1 3 2 0 0 NMOS L=1u W=10u")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert isinstance(component, NMOS)
        assert component.name == "M1"
        assert component.nodes == ["3", "2", "0", "0"]
        assert component.L == pytest.approx(1e-6)
        assert component.W == pytest.approx(10e-6)

    def test_parse_pmos_basic(self):
        """Test parsing a basic PMOS transistor."""
        parser = Parser()
        parser.parse_line("Mp1 vout vin vdd vdd PMOS L=1u W=20u")

        assert len(parser.circuit.components) == 1
        component = parser.circuit.components[0]
        assert isinstance(component, PMOS)
        assert component.name == "Mp1"
        assert component.nodes == ["vout", "vin", "vdd", "vdd"]
        assert component.L == pytest.approx(1e-6)
        assert component.W == pytest.approx(20e-6)


class TestParserAnalysis:
    """Tests for parsing analysis commands."""

    def test_parse_dc_sweep(self):
        """Test parsing .dc sweep analysis."""
        parser = Parser()
        parser.parse_line(".dc Vin 0 3.3 0.1")

        assert parser.analysis_type == "dc"
        assert parser.analysis_params["source"] == "Vin"
        assert parser.analysis_params["start"] == 0.0
        assert parser.analysis_params["stop"] == 3.3
        assert parser.analysis_params["step"] == 0.1

    def test_parse_tran_analysis(self):
        """Test parsing .tran transient analysis."""
        parser = Parser()
        parser.parse_line(".tran 10n 1u")

        assert parser.analysis_type == "tran"
        assert parser.analysis_params["tstep"] == 10e-9
        assert parser.analysis_params["tstop"] == 1e-6


class TestParserFullNetlist:
    """Tests for parsing complete netlist files."""

    def test_parse_inverter_netlist(self, tmp_path):
        """Test parsing the inverter netlist file."""
        # Create a temporary netlist file
        netlist_content = """* CMOS Inverter Test Circuit
Vdd vdd 0 3.3
Vin vin 0 0
Mp1 vout vin vdd vdd PMOS L=1u W=10u
Mn1 vout vin 0 0 NMOS L=1u W=5u
Cload vout 0 100p

.dc vin 0 3.3 0.1
.end
"""
        netlist_file = tmp_path / "test_inverter.sp"
        netlist_file.write_text(netlist_content)

        # Parse the file
        parser = Parser()
        parser.parse_file(str(netlist_file))

        # Check components
        assert len(parser.circuit.components) == 5

        # Check voltage sources
        vdd = next(c for c in parser.circuit.components if c.name == "Vdd")
        assert isinstance(vdd, VoltageSource)
        assert vdd.voltage == 3.3

        vin = next(c for c in parser.circuit.components if c.name == "Vin")
        assert isinstance(vin, VoltageSource)
        assert vin.voltage == 0.0

        # Check transistors
        mp1 = next(c for c in parser.circuit.components if c.name == "Mp1")
        assert isinstance(mp1, PMOS)
        assert mp1.W == pytest.approx(10e-6)

        mn1 = next(c for c in parser.circuit.components if c.name == "Mn1")
        assert isinstance(mn1, NMOS)
        assert mn1.W == pytest.approx(5e-6)

        # Check capacitor
        cload = next(c for c in parser.circuit.components if c.name == "Cload")
        assert isinstance(cload, Capacitor)
        assert cload.capacitance == 100e-12

        # Check analysis
        assert parser.analysis_type == "dc"
        assert parser.analysis_params["source"] == "vin"

    def test_parse_actual_inverter_file(self):
        """Test parsing the actual inverter.sp test file."""
        parser = Parser()
        parser.parse_file("/home/shenshan/NN_SPICE/tests/test_circuits/inverter.sp")

        # Verify components were parsed
        assert len(parser.circuit.components) == 5
        assert parser.analysis_type == "dc"


class TestValueParsing:
    """Tests for value parsing with unit suffixes."""

    def test_parse_value_with_suffixes(self):
        """Test parsing values with various unit suffixes."""
        parser = Parser()

        # Test various suffixes
        assert parser._parse_value("1k") == 1e3
        assert parser._parse_value("10k") == 10e3
        assert parser._parse_value("1u") == 1e-6
        assert parser._parse_value("10n") == 10e-9
        assert parser._parse_value("100p") == 100e-12
        assert parser._parse_value("1.5k") == 1.5e3
        assert parser._parse_value("3.3") == 3.3
        assert parser._parse_value("100") == 100.0

    def test_parse_value_case_insensitive(self):
        """Test that suffixes are case-insensitive."""
        parser = Parser()

        assert parser._parse_value("1K") == 1e3
        assert parser._parse_value("1U") == 1e-6
        assert parser._parse_value("1N") == 1e-9
        assert parser._parse_value("1P") == 1e-12


class TestParserCommentsAndEmptyLines:
    """Tests for handling comments and empty lines."""

    def test_ignore_comment_lines(self):
        """Test that comment lines are ignored."""
        parser = Parser()
        parser.parse_line("* This is a comment")
        parser.parse_line("* Another comment")

        assert len(parser.circuit.components) == 0

    def test_ignore_empty_lines(self):
        """Test that empty lines are ignored."""
        parser = Parser()
        parser.parse_line("")
        parser.parse_line("   ")

        assert len(parser.circuit.components) == 0

    def test_mixed_content(self):
        """Test parsing mixed comments, empty lines, and components."""
        parser = Parser()
        parser.parse_line("* Comment")
        parser.parse_line("")
        parser.parse_line("V1 1 0 5.0")
        parser.parse_line("* Another comment")
        parser.parse_line("R1 1 2 1k")

        assert len(parser.circuit.components) == 2
