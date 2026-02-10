"""
Main entry point for PyCircuitSim simulations.

This module provides the high-level simulation orchestration, connecting
parsing, circuit solving, and visualization.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

from pycircuitsim.parser import Parser
from pycircuitsim.circuit import Circuit
from pycircuitsim.visualizer import Visualizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simulation(
    netlist_path: str,
    output_dir: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Run a complete circuit simulation from a netlist file.

    This function orchestrates the entire simulation workflow:
    1. Parse the netlist
    2. Build the circuit
    3. Run appropriate analyses (DC sweep, transient)
    4. Generate and save plots

    Args:
        netlist_path: Path to the HSPICE-format netlist file
        output_dir: Directory where results and plots will be saved
                   (default: 'results' in current directory)
        verbose: Enable verbose logging output

    Raises:
        FileNotFoundError: If netlist file doesn't exist
        ValueError: If netlist contains invalid syntax
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate netlist file exists
    netlist_file = Path(netlist_path)
    if not netlist_file.exists():
        raise FileNotFoundError(f"Netlist file not found: {netlist_path}")

    logger.info(f"Loading netlist: {netlist_path}")

    # Parse netlist first to determine analysis type
    parser = Parser()
    try:
        parser.parse_file(str(netlist_file))
        circuit = parser.circuit
        logger.info(f"Parsed {len(circuit.components)} components")
    except Exception as e:
        raise ValueError(f"Failed to parse netlist: {e}")

    # Set output directory with analysis type subdirectory
    # Extract circuit name from netlist filename (e.g., "inverter" from "inverter.sp")
    circuit_name = netlist_file.stem  # Gets filename without extension
    if output_dir is None:
        output_dir = "results"

    # Create analysis type subdirectory
    if parser.analysis_type == "dc":
        analysis_subdir = "dc"
    elif parser.analysis_type == "tran":
        analysis_subdir = "tran"
    elif parser.analysis_type == "ac":
        analysis_subdir = "ac"
    else:
        analysis_subdir = "dc_op"  # Single DC operating point

    output_path = Path(output_dir) / circuit_name / analysis_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = Visualizer()

    # Run DC sweep analysis if present
    if parser.analysis_type == "dc":
        logger.info("Running DC sweep analysis...")
        dc_results = run_dc_sweep(circuit, parser.analysis_params, visualizer, output_path, circuit_name)
        logger.info(f"DC sweep complete: {len(dc_results)} points computed")

    # Run transient analysis if present
    elif parser.analysis_type == "tran":
        logger.info("Running transient analysis...")
        tran_results = run_transient(circuit, parser.analysis_params, visualizer, output_path, circuit_name)
        logger.info(f"Transient analysis complete: {len(tran_results)} time points")

    # Run AC analysis if present
    elif parser.analysis_type == "ac":
        logger.info("Running AC analysis...")
        ac_results = run_ac_sweep(circuit, parser.analysis_params, visualizer, output_path, circuit_name)
        logger.info(f"AC analysis complete: {len(ac_results['frequency'])} frequency points")

    # If no analysis specified, run a single DC operating point
    if parser.analysis_type is None:
        logger.info("No analysis specified. Running single DC operating point.")
        run_dc_op_point(circuit, output_path, circuit_name)

    logger.info(f"Simulation complete. Results saved to: {output_path}")


def run_dc_sweep(
    circuit: Circuit,
    analysis_params: Dict,
    visualizer: Visualizer,
    output_path: Path,
    circuit_name: str
) -> Dict[str, List[float]]:
    """
    Run DC sweep analysis and generate plots.

    Uses two-stage analysis:
    1. Stage 1: Compute DC operating point at initial conditions
    2. Stage 2: Perform DC sweep using OP solution as initial guess

    Args:
        circuit: Circuit object
        analysis_params: DC sweep parameters (source, start, stop, step)
        visualizer: Visualizer instance for plotting
        output_path: Directory to save plots
        circuit_name: Name of the circuit (for file naming)

    Returns:
        Dictionary mapping node names to their value lists
    """
    from pycircuitsim.solver import DCSolver

    source_name = analysis_params['source']
    start = analysis_params['start']
    stop = analysis_params['stop']
    step = analysis_params['step']

    # Find the source component to modify
    source_component = None
    for comp in circuit.components:
        if comp.name == source_name:
            source_component = comp
            break

    if source_component is None:
        raise ValueError(f"Source {source_name} not found in circuit")

    # Store original value
    original_value = source_component.value

    # Setup output file for logging
    output_file = output_path / f"{circuit_name}_simulation.lis"

    # STAGE 1: Run single DC operating point first
    logger.info("Stage 1: Computing DC operating point...")
    # Use .ic initial conditions if provided, otherwise None (solver will use 0V guess)
    initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
    op_solver = DCSolver(circuit, output_file=output_file, initial_guess=initial_guess, use_source_stepping=True)
    with op_solver:
        op_solution = op_solver.solve()
    logger.info(f"DC operating point computed: {len(op_solution)} nodes")

    # STAGE 2: Use OP solution as initial guess for sweep
    logger.info("Stage 2: Running DC sweep with OP initial guess...")

    # Generate sweep values
    # Handle both increasing (step > 0) and decreasing (step < 0) sweeps
    sweep_values = []
    if step > 0:
        # Increasing sweep: start < stop
        current_value = start
        while current_value <= stop:
            sweep_values.append(current_value)
            current_value += step
    elif step < 0:
        # Decreasing sweep: start > stop
        current_value = start
        while current_value >= stop:
            sweep_values.append(current_value)
            current_value += step
    else:
        raise ValueError(f"DC sweep step cannot be zero: {step}")

    # Run sweep with logging
    all_results = {}

    # Use context manager to enable logging for sweep
    # Disable source stepping during sweep (use continuation method instead)
    with DCSolver(circuit, output_file=output_file, use_source_stepping=False) as solver:
        # Log header with sweep parameters
        if solver.logger:
            solver.logger.log_header("DC Sweep Analysis", analysis_params)
            num_nodes = len(circuit.get_nodes())
            num_vsources = circuit.count_voltage_sources()
            solver.logger.log_circuit_summary(
                component_count=len(circuit.components),
                node_count=num_nodes,
                vsource_count=num_vsources
            )

        # Initialize previous solution tracker
        prev_solution = op_solution.copy()

        for point_num, current_value in enumerate(sweep_values):
            # Update source value
            source_component.value = current_value

            # Log sweep point start
            if solver.logger:
                solver.logger.log_sweep_point_start(point_num=point_num, sweep_value=current_value)

            # Create solver with appropriate initial guess
            # Use reduced source stepping (5 steps) for faster convergence during sweep
            # This balances performance (fewer steps than Stage 1's 20) with convergence stability
            if point_num == 0:
                # First point: use OP solution
                point_solver = DCSolver(circuit, initial_guess=op_solution, logger=solver.logger,
                                       use_source_stepping=True, source_stepping_steps=5)
            else:
                # Subsequent points: use previous solution (continuation method)
                point_solver = DCSolver(circuit, initial_guess=prev_solution, logger=solver.logger,
                                       use_source_stepping=True, source_stepping_steps=5)

            # Solve at this point
            solution = point_solver.solve(skip_header=True)

            # Store this solution for next point's initial guess
            prev_solution = solution.copy()

            # Store node voltages
            for node, node_value in solution.items():
                if node not in all_results:
                    all_results[node] = []
                all_results[node].append(node_value)

            # Calculate and store device currents
            for comp in circuit.components:
                try:
                    current = comp.calculate_current(solution)
                    # Use format: i(comp_name) for currents
                    current_key = f"i({comp.name})"
                    if current_key not in all_results:
                        all_results[current_key] = []
                    all_results[current_key].append(current)
                except (NotImplementedError, AttributeError):
                    # Skip components that don't support current calculation
                    pass

        # Log final results
        if solver.logger:
            solver.logger.log_final_results(all_results, title="DC Sweep Final Results")

    # Restore original value
    source_component.value = original_value

    # Generate plot
    plot_path = output_path / f"{circuit_name}_dc_sweep.png"
    visualizer.plot_dc_sweep(
        sweep_values=sweep_values,
        results=all_results,
        sweep_variable=f"{source_name} (V)" if source_name.startswith('V') else f"{source_name} (A)",
        output_path=str(plot_path)
    )

    # Save waveform data to CSV
    import csv
    csv_path = output_path / f"{circuit_name}_dc_sweep.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = [f"{source_name} (V)"] + list(all_results.keys())
        writer.writerow(header)

        # Write data
        for i, sweep_val in enumerate(sweep_values):
            row = [f"{sweep_val:.6f}"]
            for key in all_results.keys():
                row.append(f"{all_results[key][i]:.6e}")
            writer.writerow(row)

    logger.info(f"Waveform data saved to: {csv_path}")

    return all_results


def run_transient(
    circuit: Circuit,
    analysis_params: Dict,
    visualizer: Visualizer,
    output_path: Path,
    circuit_name: str
) -> Dict[str, List[float]]:
    """
    Run transient analysis and generate plots.

    Uses two-stage analysis:
    1. Stage 1: Compute DC operating point for initial conditions
    2. Stage 2: Perform transient analysis using OP solution as initial guess

    Args:
        circuit: Circuit object
        analysis_params: Transient parameters (tstep, tstop)
        visualizer: Visualizer instance for plotting
        output_path: Directory to save plots
        circuit_name: Name of the circuit (for file naming)

    Returns:
        Dictionary mapping node names to their value lists over time
    """
    from pycircuitsim.solver import DCSolver, TransientSolver
    import numpy as np

    time_step = analysis_params['tstep']
    final_time = analysis_params['tstop']

    # STAGE 1: Run DC operating point first for initial guess
    logger.info("Stage 1: Computing DC operating point for transient initialization...")
    # Use .ic initial conditions if provided, otherwise None (solver will use 0V guess)
    initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
    op_solver = DCSolver(circuit, initial_guess=initial_guess)
    op_solution = op_solver.solve()
    logger.info(f"DC operating point computed: {len(op_solution)} nodes")

    # STAGE 2: Use OP solution as initial guess for transient
    logger.info("Stage 2: Running transient analysis...")
    solver = TransientSolver(circuit, t_stop=final_time, dt=time_step,
                            initial_guess=op_solution)
    results = solver.solve()

    # Convert numpy arrays to lists for plotting
    time_points = results['time'].tolist()
    all_results = {}
    for node, voltages in results.items():
        if node == 'time':
            continue
        all_results[node] = voltages.tolist()

    # Generate plot
    plot_path = output_path / f"{circuit_name}_transient.png"
    visualizer.plot_transient(
        time_points=time_points,
        results=all_results,
        output_path=str(plot_path)
    )

    return all_results


def run_dc_op_point(
    circuit: Circuit,
    output_path: Path,
    circuit_name: str
) -> None:
    """
    Run a single DC operating point analysis.

    Args:
        circuit: Circuit object to analyze
        output_path: Directory to save results
        circuit_name: Name of the circuit (for file naming)
    """
    from pycircuitsim.solver import DCSolver

    # Use .ic initial conditions if provided, otherwise None (solver will use 0V guess)
    initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
    solver = DCSolver(circuit, initial_guess=initial_guess)
    solution = solver.solve()

    # Save results to text file
    result_file = output_path / f"{circuit_name}_dc_op_point.txt"
    with open(result_file, 'w') as f:
        f.write("DC Operating Point Results\n")
        f.write("=" * 40 + "\n")
        for node, value in solution.items():
            f.write(f"{node}: {value:.6f}\n")

    logger.info(f"DC operating point saved to: {result_file}")


def run_ac_sweep(
    circuit: Circuit,
    params: Dict,
    visualizer: Visualizer,
    output_path: Path,
    circuit_name: str
) -> Dict[str, np.ndarray]:
    """
    Run AC (small-signal frequency domain) analysis.

    This function:
    1. Computes DC operating point
    2. Generates frequency sweep array based on sweep type
    3. Solves small-signal circuit at each frequency
    4. Saves results and generates Bode plots

    Args:
        circuit: Circuit object to analyze
        params: Dictionary with 'sweep_type', 'num_points', 'fstart', 'fstop'
        visualizer: Visualizer object for plotting
        output_path: Path where results should be saved
        circuit_name: Base name for output files

    Returns:
        Dictionary with frequency array and complex voltages for each node
    """
    from pycircuitsim.solver import ACSolver
    import numpy as np
    import pandas as pd

    logger.info("Computing DC operating point for AC analysis...")

    # Step 1: Compute DC operating point
    # For AC analysis, all AC sources are set to 0 during DC solve
    from pycircuitsim.solver import DCSolver
    from pycircuitsim.logger import Logger

    # Create logger for DC operating point
    dc_log_file = output_path / f"{circuit_name}_dc_op_simulation.lis"
    dc_logger = Logger(circuit_name, dc_log_file)

    with dc_logger:
        dc_solver = DCSolver(circuit, logger=dc_logger)
        with dc_solver:
            dc_solution = dc_solver.solve()

    logger.info("DC operating point computed")
    for node, voltage in dc_solution.items():
        if node not in ["0", "GND"]:
            logger.info(f"  V({node}) = {voltage:.6f} V")

    # Step 2: Generate frequency array
    sweep_type = params["sweep_type"]
    num_points = params["num_points"]
    fstart = params["fstart"]
    fstop = params["fstop"]

    if sweep_type == "dec":
        # Decade sweep: num_points per decade
        num_decades = np.log10(fstop / fstart)
        total_points = int(num_points * num_decades)
        frequencies = np.logspace(np.log10(fstart), np.log10(fstop), total_points)
    elif sweep_type == "oct":
        # Octave sweep: num_points per octave
        num_octaves = np.log2(fstop / fstart)
        total_points = int(num_points * num_octaves)
        frequencies = np.logspace(np.log10(fstart), np.log10(fstop), total_points)
    else:  # "lin"
        # Linear sweep: num_points total between fstart and fstop
        frequencies = np.linspace(fstart, fstop, num_points)

    logger.info(f"AC sweep: {sweep_type.upper()} {len(frequencies)} points from {fstart:.3e} Hz to {fstop:.3e} Hz")

    # Step 3: Solve AC circuit at each frequency
    ac_solver = ACSolver(circuit, dc_solution=dc_solution)
    ac_results = ac_solver.solve(frequencies)

    logger.info(f"AC analysis complete: {len(frequencies)} frequency points")

    # Step 4: Save results to CSV
    csv_file = output_path / f"{circuit_name}_ac_sweep.csv"

    # Convert complex voltages to magnitude and phase
    data = {"frequency": frequencies}

    for node in circuit.get_nodes():
        if node not in ["0", "GND"]:
            v_complex = ac_results[node]
            v_mag = np.abs(v_complex)
            v_phase_rad = np.angle(v_complex)
            v_phase_deg = np.rad2deg(v_phase_rad)

            data[f"V({node})_mag"] = v_mag
            data[f"V({node})_phase"] = v_phase_deg

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    logger.info(f"AC results saved to: {csv_file}")

    # Step 5: Generate Bode plots
    png_file = output_path / f"{circuit_name}_ac_bode.png"
    visualizer.plot_bode(ac_results, circuit.get_nodes(), str(png_file))
    logger.info(f"Bode plot saved to: {png_file}")

    return ac_results
