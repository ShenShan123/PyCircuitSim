"""How does the MNA linear solve scale for an R x C 6T SRAM array?

Builds the true MNA sparsity/structure (node count verified against the
parser: nodes = 2*cells + rows + 2*cols + 1) and times:
  - spsolve            : what the code does today, every NR iteration
  - splu factor+solve  : same work, explicit
  - splu solve only    : the win available from reusing the factorization
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu


def build(rows, cols, seed=0):
    """MNA matrix for an R x C array: 2 storage nodes/cell, R wordlines,
    2C bitlines, 1 vdd; each a voltage source with a branch-current row."""
    rng = np.random.default_rng(seed)
    cells = rows * cols
    # node indexing
    n_store = 2 * cells                 # q, qb per cell
    n_wl = rows
    n_bl = 2 * cols
    n_vdd = 1
    n_nodes = n_store + n_wl + n_bl + n_vdd
    n_vs = n_wl + n_bl + n_vdd          # every rail is a voltage source
    N = n_nodes + n_vs

    wl0 = n_store
    bl0 = wl0 + n_wl
    vdd = bl0 + n_bl

    I, J = [], []

    def link(a, b):
        I.extend([a, a, b, b])
        J.extend([a, b, b, a])

    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            q, qb = 2 * k, 2 * k + 1
            link(q, qb)                 # cross-coupled inverter pair
            link(q, vdd)                # pull-up / pull-down to rails
            link(qb, vdd)
            link(q, bl0 + 2 * c)        # access transistor -> bitline
            link(qb, bl0 + 2 * c + 1)
            link(q, wl0 + r)            # gate coupling (gm stamp)
            link(qb, wl0 + r)

    # voltage-source B / C blocks
    for i, node in enumerate(list(range(wl0, wl0 + n_wl))
                             + list(range(bl0, bl0 + n_bl)) + [vdd]):
        br = n_nodes + i
        I.extend([node, br])
        J.extend([br, node])

    V = rng.uniform(0.5, 1.5, size=len(I))
    A = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    A = A + sp.eye(N, format="csr") * 10.0   # diagonally dominant -> stable
    return A.tocsc(), N


def timeit(fn, reps=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    print(f"{'array':>10} {'cells':>8} {'N':>9} {'nnz':>10} "
          f"{'spsolve ms':>12} {'splu f+s ms':>13} {'splu solve ms':>15} "
          f"{'LU nnz':>11} {'fill':>7}")
    specs = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
    if len(sys.argv) > 1:
        specs = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]]
    for r, c in specs:
        A, N = build(r, c)
        b = np.random.default_rng(1).uniform(size=N)
        reps = 3 if N < 200000 else 1
        t_sp = timeit(lambda: spsolve(A, b), reps)
        t_lu = timeit(lambda: splu(A).solve(b), reps)
        lu = splu(A)
        t_so = timeit(lambda: lu.solve(b), 10)
        lunnz = lu.L.nnz + lu.U.nnz
        print(f"{f'{r}x{c}':>10} {r*c:>8} {N:>9} {A.nnz:>10} "
              f"{t_sp*1e3:>12.2f} {t_lu*1e3:>13.2f} {t_so*1e3:>15.3f} "
              f"{lunnz:>11} {lunnz/A.nnz:>7.2f}", flush=True)
