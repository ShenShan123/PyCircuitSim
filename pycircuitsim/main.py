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

    # Run sweep
    sweep_values = []
    all_results = {}

    current_value = start
    while current_value <= stop:
        # Update source value
        source_component.value = current_value

        # Solve at this point
        solver = DCSolver(circuit)
        solution = solver.solve()

        # Store results
        sweep_values.append(current_value)
        for node, node_value in solution.items():
            if node not in all_results:
                all_results[node] = []
            all_results[node].append(node_value)

        # Increment
        current_value += step

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

    return all_results


def run_transient(
    circuit: Circuit,
    analysis_params: Dict,
    visualizer: Visualizer,
    output_path: Path
) -> Dict[str, List[float]]:
    """
    Run transient analysis and generate plots.

    Args:
        circuit: Circuit object
        analysis_params: Transient parameters (tstep, tstop)
        visualizer: Visualizer instance for plotting
        output_path: Directory to save plots

    Returns:
        Dictionary mapping node names to their value lists over time
    """
    from pycircuitsim.solver import TransientSolver
    import numpy as np

    time_step = analysis_params['tstep']
    final_time = analysis_params['tstop']

    # Create solver and run simulation
    solver = TransientSolver(circuit, t_stop=final_time, dt=time_step)
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
