"""Scaling at the USER'S REAL target sizes: 32x16 .. 256x128."""
import sys, time
sys.path.insert(0,'/data2/shenshan/PyCircuitSim/scripts/bench_gpu')
from bench_spsolve import build
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

TARGETS=[(32,16),(64,32),(96,48),(128,64),(160,80),(192,96),(256,128)]
print(f"{'array':>9} {'cells':>7} {'devices':>8} {'N':>7} {'nnz':>9} "
      f"{'spsolve':>10} {'splu fact':>10} {'solve-only':>10} {'LU nnz':>12} {'fill':>7}")
for r,c in TARGETS:
    cells=r*c; dev=6*cells
    A,N=build(r,c)
    nnz=A.nnz
    b=np.random.default_rng(1).uniform(-1,1,N)
    if N>40000:
        print(f"{r}x{c:<5} {cells:7d} {dev:8d} {N:7d} {nnz:9d}   -- skipped (too large, see note) --")
        continue
    t=time.perf_counter(); spsolve(A,b); t_sp=time.perf_counter()-t
    t=time.perf_counter(); lu=splu(A); t_f=time.perf_counter()-t
    t=time.perf_counter()
    for _ in range(5): lu.solve(b)
    t_s=(time.perf_counter()-t)/5
    lun=lu.L.nnz+lu.U.nnz
    print(f"{r}x{c:<5} {cells:7d} {dev:8d} {N:7d} {nnz:9d} "
          f"{t_sp*1e3:9.1f}ms {t_f*1e3:9.1f}ms {t_s*1e3:9.3f}ms {lun:12d} {lun/nnz:6.1f}x")
