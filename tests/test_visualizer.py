"""
Tests for the visualizer module.
"""

import pytest
import matplotlib
import os
from pathlib import Path

# Use Agg backend for non-interactive testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pycircuitsim.visualizer import Visualizer


class TestVisualizer:
    """Test suite for Visualizer class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create results directory if it doesn't exist
        self.results_dir = Path("/home/shenshan/NN_SPICE/results")
        self.results_dir.mkdir(exist_ok=True)

        # Clean up any existing test plot files
        for f in self.results_dir.glob("test_*.png"):
            f.unlink()

    def teardown_method(self):
        """Cleanup after tests."""
        # Close all matplotlib figures
        plt.close('all')

    def test_plot_dc_sweep_creates_file(self):
        """Test that plot_dc_sweep creates an output file."""
        # Sample DC sweep data
        sweep_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        results = {
            'v(3)': [3.0, 2.8, 2.1, 1.5, 0.9, 0.4, 0.1]
        }

        viz = Visualizer()
        output_path = self.results_dir / "test_dc_sweep.png"

        viz.plot_dc_sweep(
            sweep_values=sweep_values,
            results=results,
            sweep_variable='Vin',
            output_path=str(output_path)
        )

        # Verify file was created
        assert output_path.exists(), f"Plot file {output_path} was not created"
        assert output_path.stat().st_size > 0, "Plot file is empty"

    def test_plot_dc_sweep_multiple_traces(self):
        """Test plotting multiple node voltages in DC sweep."""
        sweep_values = [0.0, 1.0, 2.0, 3.0]
        results = {
            'v(2)': [0.0, 1.0, 2.0, 3.0],
            'v(3)': [3.0, 2.0, 1.0, 0.0],
            'v(4)': [1.5, 1.5, 1.5, 1.5]
        }

        viz = Visualizer()
        output_path = self.results_dir / "test_dc_sweep_multiple.png"

        viz.plot_dc_sweep(
            sweep_values=sweep_values,
            results=results,
            sweep_variable='Vsource',
            output_path=str(output_path)
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_transient_creates_file(self):
        """Test that plot_transient creates an output file."""
        # Sample transient data
        time_points = [0.0, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9]
        results = {
            'v(2)': [0.0, 1.5, 2.8, 3.1, 3.2, 3.2],
            'v(3)': [3.3, 2.5, 1.8, 1.2, 0.8, 0.5]
        }

        viz = Visualizer()
        output_path = self.results_dir / "test_transient.png"

        viz.plot_transient(
            time_points=time_points,
            results=results,
            output_path=str(output_path)
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_transient_with_current(self):
        """Test plotting transient analysis with currents."""
        time_points = [0.0, 1e-6, 2e-6, 3e-6]
        results = {
            'v(out)': [0.0, 1.2, 2.4, 3.0],
            'i(R1)': [0.0, 0.001, 0.002, 0.003]
        }

        viz = Visualizer()
        output_path = self.results_dir / "test_transient_current.png"

        viz.plot_transient(
            time_points=time_points,
            results=results,
            output_path=str(output_path)
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_with_empty_results(self):
        """Test that empty results dictionary is handled gracefully."""
        viz = Visualizer()
        output_path = self.results_dir / "test_empty.png"

        # Should not crash, just create empty plot
        viz.plot_dc_sweep(
            sweep_values=[0.0, 1.0, 2.0],
            results={},
            sweep_variable='V',
            output_path=str(output_path)
        )

        # File should still be created
        assert output_path.exists()

    def test_plot_creates_directory_if_needed(self):
        """Test that output directory is created if it doesn't exist."""
        viz = Visualizer()
        nested_dir = self.results_dir / "nested" / "directory"
        output_path = nested_dir / "test.png"

        viz.plot_dc_sweep(
            sweep_values=[0.0, 1.0],
            results={'v(1)': [0.0, 1.0]},
            sweep_variable='V',
            output_path=str(output_path)
        )

        assert output_path.exists()
        # Cleanup created directory
        output_path.unlink()
        output_path.parent.rmdir()
        output_path.parent.parent.rmdir()
