"""
Logger module for PyCircuitSim.

Provides logging functionality for circuit simulation results,
following HSPICE-like .lis file format.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class IterationInfo:
    """
    Dataclass containing information about a single Newton-Raphson iteration.

    Attributes
    ----------
    iteration : int
        Iteration number (0-indexed)
    voltages : Dict[str, float]
        Node voltages (node name -> voltage in Volts)
    deltas : Dict[str, float]
        Voltage changes from previous iteration (node name -> delta in Volts)
    currents : Dict[str, float]
        Device currents (device name -> current in Amps)
    conductances : Dict[str, Dict[str, float]]
        Device conductances (device name -> {"gm": value, "gds": value, ...})
    """
    iteration: int
    voltages: Dict[str, float]
    deltas: Dict[str, float]
    currents: Dict[str, float]
    conductances: Dict[str, Dict[str, float]]


class Logger:
    """
    Logger for circuit simulation results.

    Generates HSPICE-like .lis files with detailed simulation information
    including headers, iteration data, convergence status, and final results.

    The Logger is a context manager that handles file opening/closing automatically.

    Examples
    --------
    >>> with Logger("circuit.sp", "output.lis") as logger:
    ...     logger.log_header("DC Analysis", {"source": "V1", "start": 0, "stop": 3.3})
    ...     logger.log_iteration(1, iter_info)
    ...     logger.log_convergence(1, True, 5, 1e-6)
    """

    def __init__(self, netlist: str, output_file: str):
        """
        Initialize the Logger.

        Parameters
        ----------
        netlist : str
            Name of the input netlist file
        output_file : str
            Path to the output log file (.lis file)
        """
        self.netlist = netlist
        self.output_file = Path(output_file)
        self.file_handle: Optional[object] = None

    def __enter__(self) -> 'Logger':
        """
        Enter the context manager and open the output file.

        Returns
        -------
        Logger
            The logger instance
        """
        self.file_handle = open(self.output_file, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and close the output file.

        Parameters
        ----------
        exc_type : type
            Exception type if an error occurred
        exc_val : Exception
            Exception value if an error occurred
        exc_tb : traceback
            Exception traceback if an error occurred
        """
        if self.file_handle:
            self.file_handle.close()
        return False  # Don't suppress exceptions

    def _write(self, text: str) -> None:
        """
        Write text to the log file.

        Parameters
        ----------
        text : str
            Text to write
        """
        if self.file_handle:
            self.file_handle.write(text + "\n")

    def _write_separator(self, char: str = "=", length: int = 70) -> None:
        """
        Write a separator line.

        Parameters
        ----------
        char : str
            Character to use for separator (default: '=')
        length : int
            Length of separator line (default: 70)
        """
        self._write(char * length)

    def log_header(self, analysis_type: str, analysis_params: Dict[str, any]) -> None:
        """
        Write the simulation header with analysis type and parameters.

        Parameters
        ----------
        analysis_type : str
            Type of analysis (e.g., "DC Analysis", "Transient Analysis")
        analysis_params : Dict[str, any]
            Analysis parameters (e.g., source, start, stop, step for DC sweep)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._write_separator()
        self._write("PyCircuitSim Simulation Log")
        self._write(f"Netlist: {self.netlist}")
        self._write(f"Timestamp: {timestamp}")
        self._write_separator()
        self._write(f"Analysis Type: {analysis_type}")

        if analysis_params:
            self._write("Analysis Parameters:")
            for key, value in analysis_params.items():
                self._write(f"  {key}: {value}")

        self._write_separator()
        self._write("")  # Empty line for readability

    def log_circuit_summary(self, component_count: int, node_count: int,
                           vsource_count: int) -> None:
        """
        Write circuit summary statistics.

        Parameters
        ----------
        component_count : int
            Total number of components in the circuit
        node_count : int
            Number of circuit nodes (excluding ground)
        vsource_count : int
            Number of voltage sources
        """
        self._write("Circuit Summary")
        self._write("-" * 70)
        self._write(f"  Total Components: {component_count}")
        self._write(f"  Circuit Nodes: {node_count}")
        self._write(f"  Voltage Sources: {vsource_count}")
        self._write("-" * 70)
        self._write("")

    def log_sweep_point_start(self, point_num: int, sweep_value: float) -> None:
        """
        Write header for a new sweep point.

        Parameters
        ----------
        point_num : int
            Sweep point number (1-indexed)
        sweep_value : float
            Value of the sweep variable at this point
        """
        self._write_separator("-")
        self._write(f"Sweep Point {point_num}: Sweep Value = {sweep_value:.6g}")
        self._write_separator("-")
        self._write("")

    def log_iteration(self, point_num: int, iter_info: IterationInfo) -> None:
        """
        Write detailed iteration information.

        Parameters
        ----------
        point_num : int
            Sweep point number (1-indexed)
        iter_info : IterationInfo
            Iteration information to log
        """
        self._write(f"  Iteration {iter_info.iteration}:")

        # Log voltages
        if iter_info.voltages:
            self._write("    Node Voltages:")
            for node, voltage in sorted(iter_info.voltages.items()):
                self._write(f"      {node:>6s}: {voltage:12.6g} V")
        else:
            self._write("    Node Voltages: (none)")

        # Log deltas
        if iter_info.deltas:
            self._write("    Voltage Changes:")
            for node, delta in sorted(iter_info.deltas.items()):
                self._write(f"      {node:>6s}: {delta:12.6g} V")

        # Log currents
        if iter_info.currents:
            self._write("    Device Currents:")
            for device, current in sorted(iter_info.currents.items()):
                self._write(f"      {device:>6s}: {current:12.6g} A")

        # Log conductances
        if iter_info.conductances:
            self._write("    Device Conductances:")
            for device, cond_dict in sorted(iter_info.conductances.items()):
                self._write(f"      {device}:")
                for param, value in sorted(cond_dict.items()):
                    self._write(f"        {param:>4s}: {value:12.6g} S")

        self._write("")

    def log_convergence(self, point_num: int, converged: bool,
                       iterations: int, tolerance: float) -> None:
        """
        Write convergence status for a sweep point.

        Parameters
        ----------
        point_num : int
            Sweep point number (1-indexed)
        converged : bool
            True if solution converged, False otherwise
        iterations : int
            Number of iterations required
        tolerance : float
            Final tolerance achieved
        """
        self._write_separator("-")

        if converged:
            self._write(f"Point {point_num}: CONVERGED in {iterations} iterations")
        else:
            self._write(f"Point {point_num}: FAILED TO CONVERGE after {iterations} iterations")

        self._write(f"Final tolerance: {tolerance:.6g}")
        self._write_separator("-")
        self._write("")

    def log_final_results(self, results: Dict[str, any], title: str = "Final Results") -> None:
        """
        Write final operating point results.

        Parameters
        ----------
        results : Dict[str, float] or Dict[str, List[float]]
            Final node voltages (node name -> voltage) or sweep results (node name -> list of voltages)
        title : str, optional
            Title for the results section (default: "Final Results")
        """
        self._write_separator("=")
        self._write(title)
        self._write_separator("=")

        if not results:
            self._write("No results to display")
        else:
            # Check if this is a sweep result (list of values) or single operating point (single value)
            first_value = next(iter(results.values()))
            is_sweep = isinstance(first_value, list)

            if is_sweep:
                # DC sweep results
                self._write("DC Sweep Results Summary")
                self._write(f"  Total sweep points: {len(first_value)}")
                self._write("")
                self._write("Final node voltages (at last sweep point):")
                for node, voltages in sorted(results.items()):
                    if voltages:  # Check if list is not empty
                        final_voltage = voltages[-1]
                        self._write(f"  {node:>6s}: {final_voltage:12.6g} V")
            else:
                # Single operating point
                self._write("Node Voltages:")
                for node, voltage in sorted(results.items()):
                    self._write(f"  {node:>6s}: {voltage:12.6g} V")

        self._write_separator("=")
        self._write("")
