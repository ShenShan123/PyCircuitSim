"""
Tests for the main CLI entry point.
"""

import pytest
from pathlib import Path
from pycircuitsim.main import run_simulation


class TestMain:
    """Test suite for main CLI functionality."""

    def test_run_simulation_dc_sweep(self, tmp_path):
        """Test running a DC sweep simulation."""
        # Create a simple RC circuit netlist
        netlist_content = """* Simple RC Circuit
V1 1 0 5.0
R1 1 2 1k
C1 2 0 1p
.dc V1 0 5 0.5
.end
"""
        netlist_file = tmp_path / "test_rc.sp"
        netlist_file.write_text(netlist_content)

        output_dir = tmp_path / "output"

        # Run simulation
        run_simulation(
            netlist_path=str(netlist_file),
            output_dir=str(output_dir)
        )

        # Verify output files were created
        assert output_dir.exists()
        # Check for plot files
        plot_files = list(output_dir.glob("*.png"))
        assert len(plot_files) > 0, "No plot files were generated"

    def test_run_simulation_transient(self, tmp_path):
        """Test running a transient simulation."""
        netlist_content = """* RC Circuit Transient
V1 1 0 5.0
R1 1 2 1k
C1 2 0 1p
.tran 1n 50n
.end
"""
        netlist_file = tmp_path / "test_tran.sp"
        netlist_file.write_text(netlist_content)

        output_dir = tmp_path / "output_tran"

        run_simulation(
            netlist_path=str(netlist_file),
            output_dir=str(output_dir)
        )

        assert output_dir.exists()
        plot_files = list(output_dir.glob("*.png"))
        assert len(plot_files) > 0

    def test_run_simulation_invalid_file(self, tmp_path):
        """Test that invalid netlist file raises an error."""
        non_existent = tmp_path / "nonexistent.sp"

        with pytest.raises(FileNotFoundError):
            run_simulation(
                netlist_path=str(non_existent),
                output_dir=str(tmp_path / "output")
            )

    def test_run_simulation_creates_output_dir(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        netlist_content = """* Simple Voltage Divider
V1 1 0 10
R1 1 2 1k
R2 2 0 1k
.dc V1 0 10 1
.end
"""
        netlist_file = tmp_path / "test_divider.sp"
        netlist_file.write_text(netlist_content)

        nested_output = tmp_path / "deeply" / "nested" / "output"

        run_simulation(
            netlist_path=str(netlist_file),
            output_dir=str(nested_output)
        )

        assert nested_output.exists()

    def test_run_simulation_no_analysis(self, tmp_path):
        """Test netlist without analysis directive."""
        netlist_content = """* Circuit without analysis
V1 1 0 5
R1 1 2 1k
R2 2 0 2k
.end
"""
        netlist_file = tmp_path / "test_no_anal.sp"
        netlist_file.write_text(netlist_content)

        output_dir = tmp_path / "output_no_anal"

        # Should not crash, but may not generate plots
        run_simulation(
            netlist_path=str(netlist_file),
            output_dir=str(output_dir)
        )

        # Output directory should still be created
        assert output_dir.exists()
