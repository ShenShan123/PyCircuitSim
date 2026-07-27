"""Is the fill-in explosion inherent, or just scipy's default COLAMD ordering?
MNA matrices are structurally symmetric; MMD_AT_PLUS_A is the textbook choice."""
import sys,time
sys.path.insert(0,'/data2/shenshan/PyCircuitSim/scripts/bench_gpu')
from bench_spsolve import build
import numpy as np
from scipy.sparse.linalg import splu
for r,c in [(64,32),(128,64),(192,96)]:
    A,N=build(r,c); b=np.random.default_rng(1).uniform(size=N)
    print(f"\n== {r}x{c}  N={N}  nnz={A.nnz} ==")
    for spec in ("COLAMD","MMD_AT_PLUS_A","MMD_ATA","NATURAL"):
        try:
            t=time.perf_counter(); lu=splu(A,permc_spec=spec); tf=time.perf_counter()-t
            t=time.perf_counter()
            for _ in range(3): lu.solve(b)
            ts=(time.perf_counter()-t)/3
            nz=lu.L.nnz+lu.U.nnz
            print(f"  {spec:15s} factor {tf*1e3:9.1f}ms  solve {ts*1e3:7.3f}ms  "
                  f"LUnnz {nz:11d}  fill {nz/A.nnz:6.1f}x")
        except Exception as e:
            print(f"  {spec:15s} FAILED: {str(e)[:40]}")
