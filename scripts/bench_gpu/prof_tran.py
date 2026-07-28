"""Transient scaling profile: where does a real SRAM write .tran spend its time?

Same instrumentation set as prof_scale.py (nn batch eval, per-device stamping,
tocsr, spsolve, counters n_solve/n_batch), but on a write transient:
wl0 pulsed, bl0 driven low to flip column-0 cells — every other cell holds.
Mirrors run_transient's production two-stage path (DC OP seeded from .ic,
then TransientSolver seeded from the OP) and reports the two numbers the
V7.2.0 plan's transient budget assumed rather than measured: NR iterations
per accepted timestep, and the per-NR-iteration bucket split.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("external_compact_models"))

from gen_sram_array import sram_array  # noqa: E402


def sram_write_tran(rows, cols, vdd=0.75, tstep="0.05n", tstop="10n"):
    """Take the .op deck and rewrite the row/col-0 sources into a write op."""
    deck = sram_array(rows, cols, vdd=vdd, analysis=f".tran {tstep} {tstop}")
    out = []
    for ln in deck.splitlines():
        if ln.startswith("Vwl0 "):
            # wordline pulse inside the bitline drive window
            out.append(f"Vwl0 wl0 0 PULSE 0 {vdd} 2n 0.1n 0.1n 5n 20n")
        elif ln.startswith("Vbl0 "):
            # drive bl0 low to write q=0/qb=1 into every cell of column 0
            out.append(f"Vbl0 bl0 0 PULSE {vdd} 0 1n 0.1n 0.1n 8n 20n")
        else:
            out.append(ln)
    return "\n".join(out) + "\n"


class _Instrument:
    """Patch the module-level solver hooks; accumulate into a dict."""

    def __init__(self, S):
        self.S = S
        self.c = {"stamp": 0.0, "batch": 0.0, "solve": 0.0, "tocsr": 0.0,
                  "n_solve": 0, "n_batch": 0}
        self._orig = (S._stamp_mosfet_dc, S._batch_eval_nn_mosfets,
                      S._solve_mna)

    def __enter__(self):
        S, c = self.S, self.c
        orig_stamp, orig_batch, orig_solve = self._orig

        def stamp(*a, **k):
            t = time.perf_counter()
            r = orig_stamp(*a, **k)
            c["stamp"] += time.perf_counter() - t
            return r

        def batch(*a, **k):
            t = time.perf_counter()
            r = orig_batch(*a, **k)
            c["batch"] += time.perf_counter() - t
            c["n_batch"] += 1
            return r

        def solve(m, rhs):
            t = time.perf_counter()
            if S.issparse(m):
                m.tocsr()
            c["tocsr"] += time.perf_counter() - t
            t = time.perf_counter()
            r = orig_solve(m, rhs)
            c["solve"] += time.perf_counter() - t
            c["n_solve"] += 1
            return r

        S._stamp_mosfet_dc = stamp
        S._batch_eval_nn_mosfets = batch
        S._solve_mna = solve
        return c

    def __exit__(self, *exc):
        (self.S._stamp_mosfet_dc, self.S._batch_eval_nn_mosfets,
         self.S._solve_mna) = self._orig
        return False


def _report(tag, c, wall, nsteps=None):
    other = wall - c["stamp"] - c["batch"] - c["solve"] - c["tocsr"]
    nb = max(1, c["n_batch"])
    line = (f"  {tag:<5} wall={wall:8.2f}s | nn={c['batch']:7.2f} "
            f"stamp={c['stamp']:7.2f} tocsr={c['tocsr']:6.2f} "
            f"spsolve={c['solve']:7.2f} other={other:7.2f} | "
            f"nsolve={c['n_solve']} nbatch={c['n_batch']} "
            f"LM_extra={c['n_solve'] - c['n_batch']}")
    if nsteps:
        line += f" steps={nsteps} NR/step={c['n_batch'] / nsteps:.2f}"
    line += (f" | per-iter: nn={1e3 * c['batch'] / nb:6.1f}ms "
             f"stamp={1e3 * c['stamp'] / nb:6.1f}ms "
             f"spsolve={1e3 * c['solve'] / nb:5.2f}ms "
             f"other={1e3 * other / nb:5.1f}ms")
    print(line, flush=True)


def run(rows, cols, tmpdir, tstep="0.05n", tstop="10n"):
    deck_path = os.path.join(tmpdir, f"sram_tran_{rows}x{cols}.sp")
    with open(deck_path, "w") as f:
        f.write(sram_write_tran(rows, cols, tstep=tstep, tstop=tstop))

    from pycircuitsim.parser import Parser
    import pycircuitsim.solver as S

    t0 = time.perf_counter()
    p = Parser()
    p.parse_file(deck_path)
    circuit = p.circuit
    t_parse = time.perf_counter() - t0

    ndev = sum(1 for c in circuit.components if S._is_mosfet(c))
    nnodes = len(circuit.get_nodes())
    params = p.analysis_params
    dt, t_stop = params["tstep"], params["tstop"]
    nsteps = round(t_stop / dt)
    print(f"{rows}x{cols} dev={ndev} nodes={nnodes} parse={t_parse:.2f}s "
          f"steps={nsteps}", flush=True)

    # Stage 1: DC OP exactly as run_transient's fast path builds it.
    with _Instrument(S) as c_op:
        op = S.DCSolver(circuit,
                        initial_guess=circuit.initial_conditions or None,
                        use_source_stepping=True, use_gmin_stepping=False)
        t0 = time.perf_counter()
        try:
            op_solution = op.solve(skip_header=True)
            ok = "OK"
        except Exception as e:  # noqa: BLE001
            op_solution, ok = None, f"FAIL:{type(e).__name__}"
        wall_op = time.perf_counter() - t0
    print(f"  OP {ok}", flush=True)
    _report("op", c_op, wall_op)

    if op_solution is None:
        return None

    # Stage 2: transient with run_transient's production knobs.
    with _Instrument(S) as c_tr:
        tran = S.TransientSolver(circuit, t_stop=t_stop, dt=dt,
                                 initial_guess=op_solution,
                                 use_gmin_stepping=True,
                                 gmin_initial=1e-8, gmin_final=1e-12,
                                 gmin_steps=10,
                                 use_pseudo_transient=True,
                                 pseudo_transient_steps=10,
                                 pseudo_transient_cap=1e-12)
        t0 = time.perf_counter()
        try:
            tran.solve()
            ok = "OK"
        except Exception as e:  # noqa: BLE001
            ok = f"FAIL:{type(e).__name__}"
        wall_tr = time.perf_counter() - t0
    print(f"  TRAN {ok}", flush=True)
    _report("tran", c_tr, wall_tr, nsteps=nsteps)
    return c_tr


if __name__ == "__main__":
    tmpdir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tmpdir)
    sizes = [(4, 4), (8, 8), (16, 16)]
    if len(sys.argv) > 1:
        sizes = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]]
    for r, c in sizes:
        run(r, c, tmpdir)
