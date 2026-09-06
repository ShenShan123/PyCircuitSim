"""Time-derivative coefficients shared by passive and terminal-charge stamps."""


def bdf2_coefficients(dt: float, previous_dt: float | None) -> tuple[float, float, float]:
    """Return a0, a1, a2 for dq/dt = a0*q - a1*q_prev + a2*q_prev2.

    Differentiating the quadratic through the three accepted-time samples
    gives variable-step BDF-2. The preceding interval must be the last
    accepted piece, never a rejected Newton/LTE attempt.
    """
    if previous_dt is None or dt == previous_dt:
        return 1.5 / dt, 2.0 / dt, 0.5 / dt
    ratio = dt / previous_dt
    return (
        (1.0 + 2.0 * ratio) / ((1.0 + ratio) * dt),
        (1.0 + ratio) / dt,
        ratio * ratio / ((1.0 + ratio) * dt),
    )
