"""V7.2.0 Phase 4a' prerequisite: re-measure the §5.2 ordering claim on
REAL assembled MNA matrices (the rev-3/4 numbers came from a synthetic
structural matrix — `bench_spsolve.build`).

Captures matrices by monkeypatching `pycircuitsim.solver._solve_mna`
during real runs (SRAM .op at several sizes + a short .tran for the
transcap-stamped pattern), then benches per matrix:

  - spsolve as shipped (the flag-off path)
  - splu with COLAMD / MMD_AT_PLUS_A / NATURAL: factor wall, solve
    wall, fill (nnz(L)+nnz(U))/nnz(A)
  - refactor-free repeated-solve wall (the KLU-class 4b upside bound)
  - max rel deviation splu vs spsolve (the perturbation size 8.4/T1
    would gate)

Counts and fill are exact regardless of box load; walls carry the
standing +-40% contended-box caveat. Not a gate; a plan-input
measurement.

Usage:
    conda run -n pycircuitsim python scripts/bench_gpu/bench_ordering_real.py
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

sys.path.insert(0, "/data2/shenshan/PyCircuitSim")
sys.path.insert(0, "/data2/shenshan/PyCircuitSim/scripts/bench_gpu")

import torch  # noqa: E402
torch.set_num_threads(1)

from gen_sram_array import sram_array  # noqa: E402
from prof_tran import sram_write_tran  # noqa: E402
import pycircuitsim.solver as solver_mod  # noqa: E402
from pycircuitsim.parser import Parser  # noqa: E402

SP = os.environ.get("TMPDIR_DECKS", "/tmp/claude-1001/bench_ordering")
os.makedirs(SP, exist_ok=True)

SAVE_AT = {1, 2, 4, 8, 16, 32, 64}  # call indices to snapshot


class _EnoughCaptured(Exception):
    pass


def capture(deck_text: str, tag: str, analysis: str, max_call: int,
            save_at=None):
    """Run a deck with _solve_mna patched; return [(call#, A_csr, rhs)].

    A snapshot is kept when the call index is in ``save_at`` OR when the
    matrix nnz changes vs the previous call (a new stamping regime — the
    transcap block appearing marks entry into the true transient)."""
    save_at = save_at if save_at is not None else SAVE_AT
    deck_path = os.path.join(SP, f"{tag}.sp")
    with open(deck_path, "w") as f:
        f.write(deck_text)

    parser = Parser()
    parser.parse_file(deck_path)
    circuit = parser.circuit
    caps = []
    orig = solver_mod._solve_mna
    state = {"n": 0, "nnz": -1}

    def patched(mna_matrix, rhs):
        state["n"] += 1
        if sp.issparse(mna_matrix):
            csr = mna_matrix.tocsr()
            if state["n"] in save_at or (
                    csr.nnz != state["nnz"] and len(caps) < 12):
                caps.append((state["n"], csr.copy(), rhs.copy()))
            state["nnz"] = csr.nnz
        out = orig(mna_matrix, rhs)
        if state["n"] >= max_call:
            raise _EnoughCaptured()
        return out

    solver_mod._solve_mna = patched
    t0 = time.perf_counter()
    try:
        op = solver_mod.DCSolver(
            circuit, initial_guess=circuit.initial_conditions or None,
            use_source_stepping=True, use_gmin_stepping=False)
        op_solution = op.solve(skip_header=True)
        if analysis != "op":
            params = parser.analysis_params
            tran = solver_mod.TransientSolver(
                circuit, t_stop=params["tstop"], dt=params["tstep"],
                initial_guess=op_solution,
                use_gmin_stepping=True, gmin_initial=1e-8,
                gmin_final=1e-12, gmin_steps=10,
                use_pseudo_transient=True, pseudo_transient_steps=10,
                pseudo_transient_cap=1e-12)
            tran.solve()
    except _EnoughCaptured:
        pass
    finally:
        solver_mod._solve_mna = orig
    print(f"[{tag}] captured {len(caps)} matrices "
          f"({state['n']} solve calls, {time.perf_counter()-t0:.1f} s)",
          flush=True)
    return caps


def bench_matrix(tag: str, call_no: int, A: sp.csr_matrix, b: np.ndarray):
    N = A.shape[0]
    print(f"\n== {tag} call#{call_no}: N={N}, nnz={A.nnz} "
          f"({A.nnz/N:.1f}/row) ==")

    def med(f, k=3):
        ts = []
        for _ in range(k):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[k // 2]

    x_ref = spsolve(A.tocsr(), b)
    t_spsolve = med(lambda: spsolve(A.tocsr(), b))
    print(f"   spsolve (shipped)     {t_spsolve*1e3:9.2f} ms")

    Ac = A.tocsc()
    for spec in ("COLAMD", "MMD_AT_PLUS_A", "NATURAL"):
        try:
            t_f = med(lambda: splu(Ac, permc_spec=spec))
            lu = splu(Ac, permc_spec=spec)
            t_s = med(lambda: lu.solve(b), k=10)
            x = lu.solve(b)
            denom = np.maximum(np.abs(x_ref), 1e-30)
            rel = float(np.max(np.abs(x - x_ref) / denom))
            fill = (lu.L.nnz + lu.U.nnz) / A.nnz
            print(f"   splu {spec:14s} factor {t_f*1e3:9.2f} ms  "
                  f"solve {t_s*1e3:7.3f} ms  fill {fill:6.1f}x  "
                  f"vs-spsolve rel {rel:.2e}  "
                  f"({t_spsolve/(t_f+t_s):5.1f}x vs shipped)")
        except Exception as e:
            print(f"   splu {spec:14s} FAILED: {type(e).__name__}: {e}")


def main():
    cases = [
        ("sram16x16_op", sram_array(16, 16), "op", 9, None),
        ("sram32x32_op", sram_array(32, 32), "op", 9, None),
        ("sram64x32_op", sram_array(64, 32), "op", 9, None),
        # write transient: OP (source stepping, ~40+ solves) runs first;
        # nnz-change capture picks up the first true transient matrix
        # (transcap block adds gate-row entries), high indices sample it.
        ("sram16x16_tran",
         sram_write_tran(16, 16, tstep="0.05n", tstop="1n"), "tran", 62,
         {1, 45, 50, 55, 60}),
    ]
    if os.environ.get("BIG", "0") == "1":
        cases = [("sram128x64_op", sram_array(128, 64), "op", 5, None)]

    for tag, deck, analysis, max_call, save_at in cases:
        try:
            caps = capture(deck, tag, analysis, max_call, save_at)
        except Exception as e:
            print(f"[{tag}] capture FAILED: {type(e).__name__}: {e}")
            continue
        for call_no, A, b in caps:
            bench_matrix(tag, call_no, A, b)


if __name__ == "__main__":
    main()
