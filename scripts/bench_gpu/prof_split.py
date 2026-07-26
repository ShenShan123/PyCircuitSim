"""Split batch_eval into (tensor compute) vs (per-device Python tail).

The tail = _unpack_eval loop: denorm, guard_gds, Vds correction. It is
scalar Python and CANNOT be moved to GPU without changing rounding, so it
caps the achievable speedup from a GPU port of the forward+autograd.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("external_compact_models"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_sram_array import sram_array  # noqa: E402


def run(rows, cols, tmpdir):
    deck = os.path.join(tmpdir, f"sram_{rows}x{cols}.sp")
    with open(deck, "w") as f:
        f.write(sram_array(rows, cols))

    from pycircuitsim.parser import Parser
    from pycircuitsim.models.mosfet_nn import _MOSFETNNBase
    import pycircuitsim.solver as S

    p = Parser()
    p.parse_file(deck)
    circuit = p.circuit
    ndev = sum(1 for c in circuit.components if S._is_mosfet(c))

    acc = {"tail": 0.0, "n": 0}
    orig_unpack = _MOSFETNNBase._unpack_eval

    def unpack(self, *a, **k):
        t = time.perf_counter()
        r = orig_unpack(self, *a, **k)
        acc["tail"] += time.perf_counter() - t
        acc["n"] += 1
        return r

    _MOSFETNNBase._unpack_eval = unpack

    tot = {"batch": 0.0}
    orig_batch = S._batch_eval_nn_mosfets

    def batch(*a, **k):
        t = time.perf_counter()
        r = orig_batch(*a, **k)
        tot["batch"] += time.perf_counter() - t
        return r

    S._batch_eval_nn_mosfets = batch

    solver = S.DCSolver(circuit)
    t0 = time.perf_counter()
    solver.solve(skip_header=True)
    t_solve = time.perf_counter() - t0

    tensor = tot["batch"] - acc["tail"]
    print(f"{rows}x{cols:<5} dev={ndev:<6} solve={t_solve:7.2f}s "
          f"batch_eval={tot['batch']:6.2f}s = tensor {tensor:6.2f}s "
          f"({tensor/tot['batch']*100:4.1f}%) + python-tail {acc['tail']:6.2f}s "
          f"({acc['tail']/tot['batch']*100:4.1f}%)  "
          f"[{acc['n']} unpack calls, "
          f"{acc['tail']/max(acc['n'],1)*1e6:.1f} us each]", flush=True)


if __name__ == "__main__":
    tmpdir = os.path.dirname(os.path.abspath(__file__))
    for a in sys.argv[1:] or ["8x8"]:
        r, c = (int(x) for x in a.split("x"))
        run(r, c, tmpdir)
