"""
Main entry point for PyCircuitSim simulations.

This module provides the high-level simulation orchestration, connecting
parsing, circuit solving, and visualization.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

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

    # Set output directory
    if output_dir is None:
        output_dir = "results"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse netlist
    parser = Parser()
    try:
        parser.parse_file(str(netlist_file))
        circuit = parser.circuit
        logger.info(f"Parsed {len(circuit.components)} components")
    except Exception as e:
        raise ValueError(f"Failed to parse netlist: {e}")

    # Initialize visualizer
    visualizer = Visualizer()

    # Run DC sweep analysis if present
    if parser.analysis_type == "dc":
        logger.info("Running DC sweep analysis...")
        dc_results = run_dc_sweep(circuit, parser.analysis_params, visualizer, output_path)
        logger.info(f"DC sweep complete: {len(dc_results)} points computed")

    # Run transient analysis if present
    if parser.analysis_type == "tran":
        logger.info("Running transient analysis...")
        tran_results = run_transient(circuit, parser.analysis_params, visualizer, output_path)
        logger.info(f"Transient analysis complete: {len(tran_results)} time points")

    # If no analysis specified, run a single DC operating point
    if parser.analysis_type is None:
        logger.info("No analysis specified. Running single DC operating point.")
        run_dc_op_point(circuit, output_path)

    logger.info(f"Simulation complete. Results saved to: {output_path}")


def run_dc_sweep(
    circuit: Circuit,
    analysis_params: Dict,
    visualizer: Visualizer,
    output_path: Path
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
    output_file = output_path / "simulation.lis"

    # STAGE 1: Run single DC operating point first
    logger.info("Stage 1: Computing DC operating point...")
    op_solver = DCSolver(circuit, output_file=output_file)
    with op_solver:
        op_solution = op_solver.solve()
    logger.info(f"DC operating point computed: {len(op_solution)} nodes")

    # STAGE 2: Use OP solution as initial guess for sweep
    logger.info("Stage 2: Running DC sweep with OP initial guess...")

    # Generate sweep values
    sweep_values = []
    current_value = start
    while current_value <= stop:
        sweep_values.append(current_value)
        current_value += step

    # Run sweep with logging
    all_results = {}

    # Use context manager to enable logging for sweep
    with DCSolver(circuit, output_file=output_file) as solver:
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

        for point_num, current_value in enumerate(sweep_values):
            # Update source value
            source_component.value = current_value

            # Log sweep point start
            if solver.logger:
                solver.logger.log_sweep_point_start(point_num=point_num, sweep_value=current_value)

            # Create solver with initial guess from OP (reuses logger context)
            point_solver = DCSolver(circuit, initial_guess=op_solution, logger=solver.logger)

            # Solve at this point
            solution = point_solver.solve(skip_header=True)

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
    plot_path = output_path / "dc_sweep.png"
    visualizer.plot_dc_sweep(
        sweep_values=sweep_values,
        results=all_results,
        sweep_variable=f"{source_name} (V)" if source_name.startswith('V') else f"{source_name} (A)",
        output_path=str(plot_path)
    )

    # Save waveform data to CSV
    import csv
    csv_path = output_path / "dc_sweep.csv"
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
    output_path: Path
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

    Returns:
        Dictionary mapping node names to their value lists over time
    """
    from pycircuitsim.solver import DCSolver, TransientSolver
    import numpy as np

    time_step = analysis_params['tstep']
    final_time = analysis_params['tstop']

    # STAGE 1: Run DC operating point first for initial guess
    logger.info("Stage 1: Computing DC operating point for transient initialization...")
    op_solver = DCSolver(circuit)
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
    plot_path = output_path / "transient.png"
    visualizer.plot_transient(
        time_points=time_points,
        results=all_results,
        output_path=str(plot_path)
    )

    return all_results


def run_dc_op_point(
    circuit: Circuit,
    output_path: Path
) -> None:
    """
    Run a single DC operating point analysis.

    Args:
        circuit: Circuit object to analyze
        output_path: Directory to save results
    """
    from pycircuitsim.solver import DCSolver

    solver = DCSolver(circuit)
    solution = solver.solve()

    # Save results to text file
    result_file = output_path / "dc_op_point.txt"
    with open(result_file, 'w') as f:
        f.write("DC Operating Point Results\n")
        f.write("=" * 40 + "\n")
        for node, value in solution.items():
            f.write(f"{node}: {value:.6f}\n")

    logger.info(f"DC operating point saved to: {result_file}")
