"""Scaling profile: where does .op time go as the SRAM array grows?

Breaks the DC solve into: NN batch eval, per-device Python stamping,
lil->csr conversion, spsolve.
"""
import os
import sys
import time
import cProfile
import pstats
import io

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("external_compact_models"))

from gen_sram_array import sram_array  # noqa: E402


def run(rows, cols, tmpdir):
    deck = os.path.join(tmpdir, f"sram_{rows}x{cols}.sp")
    with open(deck, "w") as f:
        f.write(sram_array(rows, cols))

    from pycircuitsim.parser import Parser
    import pycircuitsim.solver as S

    t0 = time.perf_counter()
    p = Parser()
    p.parse_file(deck)
    circuit = p.circuit
    t_parse = time.perf_counter() - t0

    ndev = sum(1 for c in circuit.components if S._is_mosfet(c))
    nnodes = len(circuit.get_nodes())

    # instrument
    counters = {"stamp": 0.0, "batch": 0.0, "solve": 0.0, "n_solve": 0,
                "n_batch": 0, "tocsr": 0.0}

    orig_stamp = S._stamp_mosfet_dc
    orig_batch = S._batch_eval_nn_mosfets
    orig_solve = S._solve_mna

    def stamp(*a, **k):
        t = time.perf_counter()
        r = orig_stamp(*a, **k)
        counters["stamp"] += time.perf_counter() - t
        return r

    def batch(*a, **k):
        t = time.perf_counter()
        r = orig_batch(*a, **k)
        counters["batch"] += time.perf_counter() - t
        counters["n_batch"] += 1
        return r

    def solve(m, rhs):
        t = time.perf_counter()
        csr = m.tocsr() if S.issparse(m) else m
        counters["tocsr"] += time.perf_counter() - t
        t = time.perf_counter()
        r = orig_solve(m, rhs)
        counters["solve"] += time.perf_counter() - t
        counters["n_solve"] += 1
        return r

    S._stamp_mosfet_dc = stamp
    S._batch_eval_nn_mosfets = batch
    S._solve_mna = solve
    # DCSolver captured them at module import time? they are module-level
    # lookups inside methods, so patching the module attr works.

    solver = S.DCSolver(circuit)
    t0 = time.perf_counter()
    try:
        solver.solve(skip_header=True)
        ok = "OK"
    except Exception as e:  # noqa: BLE001
        ok = f"FAIL:{type(e).__name__}"
    t_solve = time.perf_counter() - t0

    other = t_solve - counters["stamp"] - counters["batch"] - counters["solve"] \
        - counters["tocsr"]
    print(f"{rows}x{cols:<5} dev={ndev:<6} nodes={nnodes:<6} {ok:<12} "
          f"parse={t_parse:7.2f}s solve={t_solve:8.2f}s | "
          f"nn={counters['batch']:7.2f} stamp={counters['stamp']:7.2f} "
          f"tocsr={counters['tocsr']:6.2f} spsolve={counters['solve']:7.2f} "
          f"other={other:7.2f} | nsolve={counters['n_solve']} "
          f"nbatch={counters['n_batch']}", flush=True)
    return counters


if __name__ == "__main__":
    tmpdir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tmpdir)
    sizes = [(2, 2), (4, 4), (8, 8), (11, 12), (16, 16)]
    if len(sys.argv) > 1:
        sizes = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]]
    for r, c in sizes:
        run(r, c, tmpdir)
