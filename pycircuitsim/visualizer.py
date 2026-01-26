"""
Visualization module for circuit simulation results.

This module provides plotting capabilities for DC sweep and transient analysis
results using matplotlib.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class Visualizer:
    """
    Visualizer for circuit simulation results.

    Provides methods to plot DC sweep and transient analysis results.
    All plots are saved to disk as image files.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use for plots
        """
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if specified style not available
            plt.style.use('default')

    def plot_dc_sweep(
        self,
        sweep_values: List[float],
        results: Dict[str, List[float]],
        sweep_variable: str,
        output_path: str,
        title: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> None:
        """
        Plot DC sweep analysis results with separate voltage and current subplots.

        Args:
            sweep_values: List of swept source values (e.g., voltage or current)
            results: Dictionary mapping node names to their value lists
                    (e.g., {'v(3)': [0.0, 1.5, ...], 'i(R1)': [...]})
            sweep_variable: Name of the swept variable (for x-axis label)
            output_path: Path where the plot image will be saved
            title: Optional plot title. If None, auto-generated
            figsize: Figure size as (width, height) in inches (default taller for 2 subplots)
        """
        # Create figure with 2 subplots (top and bottom)
        fig, (ax_voltage, ax_current) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if not results:
            # No data case
            for ax in [ax_voltage, ax_current]:
                ax.text(0.5, 0.5, 'No data to plot',
                       ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
            ax_voltage.set_ylabel('Voltage (V)')
            ax_current.set_ylabel('Current (A)')
            ax_current.set_xlabel(f'{sweep_variable}')
        else:
            # Separate voltages and currents
            voltages = {}
            currents = {}

            for node_name, values in results.items():
                if len(values) != len(sweep_values):
                    print(f"Warning: Length mismatch for {node_name}. Skipping.")
                    continue

                # Check node name prefix for voltage/current classification
                if node_name.startswith('i(') or node_name.startswith('I('):
                    currents[node_name] = values
                elif node_name.startswith('v(') or node_name.startswith('V(') or node_name.startswith('node_'):
                    voltages[node_name] = values
                else:
                    # Heuristic: check magnitude to guess
                    max_val = max(abs(v) for v in values) if values else 0
                    if max_val < 1.0:  # Likely current
                        currents[node_name] = values
                    else:  # Likely voltage
                        voltages[node_name] = values

            # Plot voltages on top subplot
            if voltages:
                for node_name, values in voltages.items():
                    ax_voltage.plot(sweep_values, values, marker='o', markersize=3, label=node_name)
                ax_voltage.set_ylabel('Voltage (V)')
                ax_voltage.legend(loc='best')
                ax_voltage.grid(True, alpha=0.3)
            else:
                ax_voltage.text(0.5, 0.5, 'No voltage data',
                              ha='center', va='center',
                              transform=ax_voltage.transAxes, fontsize=12)
                ax_voltage.set_ylabel('Voltage (V)')

            # Plot currents on bottom subplot
            if currents:
                for node_name, values in currents.items():
                    ax_current.plot(sweep_values, values, marker='o', markersize=3, label=node_name)
                ax_current.set_ylabel('Current (A)')
                ax_current.legend(loc='best')
                ax_current.grid(True, alpha=0.3)
            else:
                ax_current.text(0.5, 0.5, 'No current data',
                              ha='center', va='center',
                              transform=ax_current.transAxes, fontsize=12)
                ax_current.set_ylabel('Current (A)')

            ax_current.set_xlabel(f'{sweep_variable}')

        # Set title
        if title is None:
            title = f'DC Sweep Analysis - {sweep_variable}'
        fig.suptitle(title, y=0.98)  # Overall title for both subplots

        # Tight layout and save
        plt.tight_layout()

        # Create directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"DC sweep plot saved to: {output_path}")

    def plot_transient(
        self,
        time_points: List[float],
        results: Dict[str, List[float]],
        output_path: str,
        title: Optional[str] = None,
        figsize: tuple = (12, 6),
        time_scale: str = 'auto'
    ) -> None:
        """
        Plot transient analysis results.

        Args:
            time_points: List of time points in seconds
            results: Dictionary mapping node names to their value lists
                    (e.g., {'v(3)': [0.0, 1.5, ...], 'i(R1)': [...]})
            output_path: Path where the plot image will be saved
            title: Optional plot title. If None, auto-generated
            figsize: Figure size as (width, height) in inches
            time_scale: Time scale for x-axis ('auto', 's', 'ms', 'us', 'ns')
        """
        fig, ax = plt.subplots(figsize=figsize)

        if not results:
            ax.text(
                0.5, 0.5,
                'No data to plot',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=12
            )
            ax.set_xlabel('Time (s)')
        else:
            # Convert time to appropriate scale
            time_array = np.array(time_points)
            scale_factor = 1.0
            time_unit = 's'

            if time_scale == 'auto':
                # Auto-detect appropriate scale
                max_time = np.max(np.abs(time_array))
                if max_time < 1e-6:
                    scale_factor = 1e9
                    time_unit = 'ns'
                elif max_time < 1e-3:
                    scale_factor = 1e6
                    time_unit = 'us'
                elif max_time < 1.0:
                    scale_factor = 1e3
                    time_unit = 'ms'
            else:
                # Use specified scale
                scales = {
                    's': 1.0,
                    'ms': 1e3,
                    'us': 1e6,
                    'ns': 1e9
                }
                scale_factor = scales.get(time_scale, 1.0)
                time_unit = time_scale

            scaled_time = time_array * scale_factor

            # Plot each node/branch variable
            for node_name, values in results.items():
                if len(values) != len(time_points):
                    print(f"Warning: Length mismatch for {node_name}. Skipping.")
                    continue

                ax.plot(scaled_time, values, linewidth=1.5, label=node_name)

            ax.set_xlabel(f'Time ({time_unit})')
            ax.set_ylabel('Voltage (V) / Current (A)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        # Set title
        if title is None:
            title = 'Transient Analysis'
        ax.set_title(title)

        # Tight layout and save
        plt.tight_layout()

        # Create directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Transient plot saved to: {output_path}")
