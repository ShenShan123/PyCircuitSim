"""
Test suite for Logger class.

Tests the logging functionality for circuit simulation results.
"""

import pytest
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

from pycircuitsim.logger import Logger, IterationInfo


class TestIterationInfo:
    """Test IterationInfo dataclass."""

    def test_create_iteration_info(self):
        """Test creating an IterationInfo instance."""
        info = IterationInfo(
            iteration=1,
            voltages={"1": 3.3, "2": 1.5},
            deltas={"1": 0.01, "2": 0.005},
            currents={"V1": 0.001},
            conductances={"M1": {"gm": 0.01, "gds": 0.001}}
        )

        assert info.iteration == 1
        assert info.voltages["1"] == 3.3
        assert info.deltas["2"] == 0.005
        assert info.currents["V1"] == 0.001
        assert info.conductances["M1"]["gm"] == 0.01

    def test_iteration_info_empty_dicts(self):
        """Test IterationInfo with empty dictionaries."""
        info = IterationInfo(
            iteration=0,
            voltages={},
            deltas={},
            currents={},
            conductances={}
        )

        assert info.iteration == 0
        assert len(info.voltages) == 0
        assert len(info.deltas) == 0
        assert len(info.currents) == 0
        assert len(info.conductances) == 0


class TestLoggerContextManager:
    """Test Logger context manager functionality."""

    def test_context_manager_creates_file(self):
        """Test that context manager creates and closes file properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                assert logger is not None
                assert output_file.exists()

            # File should still exist after context exit
            assert output_file.exists()

    def test_context_manager_write_operations(self):
        """Test that write operations work within context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_header("DC Analysis", {"source": "V1", "start": 0, "stop": 3.3, "step": 0.1})

            # Check file was written
            content = output_file.read_text()
            assert "DC Analysis" in content
            assert "test_netlist.sp" in content


class TestLoggerHeader:
    """Test Logger header writing methods."""

    def test_log_header_dc_analysis(self):
        """Test logging header for DC analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_header(
                    "DC Analysis",
                    {"source": "V1", "start": 0.0, "stop": 3.3, "step": 0.1}
                )

            content = output_file.read_text()
            assert "PyCircuitSim Simulation Log" in content
            assert "test_netlist.sp" in content
            assert "DC Analysis" in content
            assert "V1" in content
            assert "0.0" in content
            assert "3.3" in content

    def test_log_header_transient_analysis(self):
        """Test logging header for transient analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_header(
                    "Transient Analysis",
                    {"tstep": 1e-9, "tstop": 1e-6}
                )

            content = output_file.read_text()
            assert "Transient Analysis" in content
            assert "1e-09" in content or "1e-9" in content
            assert "1e-06" in content or "1e-6" in content

    def test_log_circuit_summary(self):
        """Test logging circuit summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_circuit_summary(
                    component_count=5,
                    node_count=3,
                    vsource_count=1
                )

            content = output_file.read_text()
            assert "Circuit Summary" in content
            assert "5" in content  # component count
            assert "3" in content  # node count
            assert "1" in content  # voltage source count


class TestLoggerSweepLogging:
    """Test Logger sweep point logging methods."""

    def test_log_sweep_point_start(self):
        """Test logging sweep point start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_sweep_point_start(point_num=1, sweep_value=1.65)

            content = output_file.read_text()
            assert "Sweep Point" in content
            assert "1" in content
            assert "1.65" in content

    def test_log_iteration(self):
        """Test logging iteration information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                info = IterationInfo(
                    iteration=2,
                    voltages={"1": 3.3, "2": 1.65},
                    deltas={"1": 0.001, "2": 0.0005},
                    currents={"V1": 0.0001},
                    conductances={"M1": {"gm": 0.01, "gds": 0.001}}
                )
                logger.log_iteration(point_num=1, iter_info=info)

            content = output_file.read_text()
            assert "Iteration 2" in content
            assert "3.3" in content
            assert "1.65" in content
            assert "0.001" in content or "1e-03" in content
            assert "V1" in content
            assert "gm" in content
            assert "gds" in content

    def test_log_convergence_success(self):
        """Test logging successful convergence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_convergence(
                    point_num=1,
                    converged=True,
                    iterations=5,
                    tolerance=1e-6
                )

            content = output_file.read_text()
            assert "CONVERGED" in content
            assert "5" in content
            assert "1e-06" in content or "1e-6" in content

    def test_log_convergence_failure(self):
        """Test logging convergence failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_convergence(
                    point_num=1,
                    converged=False,
                    iterations=20,
                    tolerance=1e-6
                )

            content = output_file.read_text()
            assert "FAILED TO CONVERGE" in content
            assert "20" in content

    def test_log_final_results(self):
        """Test logging final operating point results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                results = {
                    "1": 3.3000,
                    "2": 1.6500,
                    "3": 0.0000
                }
                logger.log_final_results(results, "Final Operating Point")

            content = output_file.read_text()
            assert "Final Operating Point" in content
            assert "3.3" in content
            assert "1.65" in content
            assert "0 V" in content  # Format is "0 V" not "0.0"


class TestLoggerMultipleSweepPoints:
    """Test Logger with multiple sweep points."""

    def test_log_multiple_sweep_points(self):
        """Test logging multiple sweep points with iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_header("DC Sweep", {"source": "Vin", "start": 0, "stop": 3.3, "step": 1.1})
                logger.log_circuit_summary(4, 3, 1)

                # Point 1
                logger.log_sweep_point_start(1, 0.0)
                info1 = IterationInfo(
                    iteration=1,
                    voltages={"1": 3.3, "2": 0.0},
                    deltas={"1": 0.0, "2": 0.0},
                    currents={"Vdd": 0.00033},
                    conductances={}
                )
                logger.log_iteration(1, info1)
                logger.log_convergence(1, True, 1, 1e-6)

                # Point 2
                logger.log_sweep_point_start(2, 1.1)
                info2 = IterationInfo(
                    iteration=3,
                    voltages={"1": 3.3, "2": 0.5},
                    deltas={"1": 0.0001, "2": 0.0001},
                    currents={"Vdd": 0.00028},
                    conductances={"M1": {"gm": 0.005, "gds": 0.0005}}
                )
                logger.log_iteration(2, info2)
                logger.log_convergence(2, True, 3, 1e-6)

            content = output_file.read_text()
            # Verify both points are logged
            assert "Sweep Point 1" in content
            assert "Sweep Point 2" in content
            assert "0 V" in content or "0.0" in content
            assert "1.1" in content
            assert "CONVERGED" in content


class TestLoggerEdgeCases:
    """Test Logger edge cases."""

    def test_empty_iteration_info(self):
        """Test logging with empty iteration info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                info = IterationInfo(
                    iteration=1,
                    voltages={},
                    deltas={},
                    currents={},
                    conductances={}
                )
                logger.log_iteration(1, info)

            # Should not raise exception
            content = output_file.read_text()
            assert "Iteration 1" in content

    def test_log_empty_final_results(self):
        """Test logging empty final results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            with Logger("test_netlist.sp", str(output_file)) as logger:
                logger.log_final_results({}, "Empty Results")

            content = output_file.read_text()
            assert "Empty Results" in content

    def test_long_netlist_name(self):
        """Test logger with very long netlist name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.lis"

            long_name = "very_long_netlist_name_with_many_descriptive_characters_and_numbers_12345.sp"
            with Logger(long_name, str(output_file)) as logger:
                logger.log_header("DC Analysis", {})

            content = output_file.read_text()
            assert long_name in content
