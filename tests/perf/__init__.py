"""Perf-path gates. These run NO NGSPICE.

They guard that an optimization is bit-identical (or that an opt-in perturbing
flag stays behind its env switch), not that physics is right — so they are
grouped by what they check rather than by circuit tier.
"""
