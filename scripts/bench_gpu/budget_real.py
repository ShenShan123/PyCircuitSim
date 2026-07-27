"""End-to-end budget at the USER'S real sizes. All per-device terms scale O(devices)
from the measured 6144-device (32x32) profile; NN-CPU uses the measured growth curve.
Solver: measured synthetic /12, COLAMD (today) vs MMD_AT_PLUS_A (fix)."""
import numpy as np
IT=41
B={'parse':44.4,'tensor':17.2,'tail':9.5,'stamp':11.6,'other':1.9}   # at 6144 devices
def cpu_us(n):
    xs=[384,1536,6144,24576]; ys=[50.5,55.4,85.4,201.3]
    return float(np.interp(n,xs,ys)) if n<=24576 else 201.3*(n/24576)**0.72
SIZES=[("32x16",3072,14.9,0.9,1.65),("64x32",12288,410.0,24.75,2.07),
       ("128x64",49152,27747.,526.6,6.17),("192x96",110592,131513.,1784.,13.04),
       ("256x128",196608,401000.,4208.,22.24)]
print(f"{'array':>9} {'dev':>7} | {'TODAY total':>12} | {'ALL PHASES':>11} "
      f"| {'parse':>8} {'solver':>8} {'NNgpu':>7} {'stamp':>7} {'other':>7} | parse%")
for a,dev,sp_col,sp_mmd,gpu in SIZES:
    k=dev/6144
    parse0=B['parse']*k;  parse1=18.4*k
    nn_cpu=cpu_us(dev)*dev/1e6*IT
    nn_gpu=gpu*1.5/1e3*IT
    tail0=B['tail']*k;    tail1=tail0/58
    stamp0=B['stamp']*k;  stamp1=stamp0/10
    other=B['other']*k
    solv0=sp_col/12/1e3*IT; solv1=sp_mmd/12/1e3*IT
    today=parse0+nn_cpu+tail0+stamp0+other+solv0
    after=parse1+nn_gpu+tail1+stamp1+other+solv1
    print(f"{a:>9} {dev:7d} | {today/60:9.1f}min | {after:8.0f}s "
          f"| {parse1:7.0f}s {solv1:7.1f}s {nn_gpu:6.2f}s {stamp1:6.1f}s {other:6.1f}s "
          f"| {100*parse1/after:3.0f}%")
print("\n--- 100-run MC at 256x128 (array-level, parse once) ---")
dev=196608; k=dev/6144
parse1=18.4*k; per_run=4208/12/1e3*IT + 22.24*1.5/1e3*IT + (9.5*k)/58 + (11.6*k)/10 + 1.9*k
print(f"  parse ONCE                 : {parse1/60:8.1f} min")
print(f"  per additional MC run      : {per_run/60:8.1f} min")
print(f"  100 runs, parse once       : {(parse1+100*per_run)/3600:8.1f} h   <-- feasible")
print(f"  100 runs, re-parsing each  : {100*(parse1+per_run)/3600:8.1f} h")
today_run=44.4*k + cpu_us(dev)*dev/1e6*IT + 9.5*k + 11.6*k + 1.9*k + 401000/12/1e3*IT
print(f"  100 runs, TODAY            : {100*today_run/86400:8.1f} DAYS")
