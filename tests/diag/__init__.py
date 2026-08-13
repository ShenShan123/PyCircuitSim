"""Diagnostics — NOT pass/fail gates.

These use LEVEL=72-in-PyCircuitSim as the reference rather than NGSPICE, which
is what makes them controls: they isolate an NN-surface gap from a solver gap.
Never quote a diag_* result as a gate result.
"""
