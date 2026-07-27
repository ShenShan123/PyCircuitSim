"""Is MMD_AT_PLUS_A robust to node ordering? Real parses sort node names
alphabetically (circuit.get_nodes -> sorted), which scrambles any natural order."""
import sys,time
sys.path.insert(0,'/data2/shenshan/PyCircuitSim/scripts/bench_gpu')
from bench_spsolve import build
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
rng=np.random.default_rng(7)
for r,c in [(128,64)]:
    A,N=build(r,c); b=rng.uniform(size=N)
    P=rng.permutation(N)                      # worst case: random node order
    Perm=sp.csc_matrix((np.ones(N),(np.arange(N),P)),shape=(N,N))
    Ap=(Perm.T@A@Perm).tocsc()
    # alphabetical-ish order: sort node indices by their string name
    names=sorted(range(N), key=lambda i: str(i))
    Q=sp.csc_matrix((np.ones(N),(np.arange(N),np.array(names))),shape=(N,N))
    Aa=(Q.T@A@Q).tocsc()
    for tag,M in (("natural-order",A),("ALPHABETICAL (real)",Aa),("random-permuted",Ap)):
        print(f"\n-- {r}x{c}, {tag} --")
        for spec in ("COLAMD","MMD_AT_PLUS_A","NATURAL"):
            t=time.perf_counter(); lu=splu(M,permc_spec=spec); tf=time.perf_counter()-t
            nz=lu.L.nnz+lu.U.nnz
            print(f"   {spec:15s} factor {tf*1e3:9.1f}ms  fill {nz/M.nnz:6.1f}x")
