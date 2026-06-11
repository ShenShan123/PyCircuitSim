# S6=P1 swap matrix — mode `id_q_np` (TSMC7 RO, window 0.600 ns, settle 0.300 ns)

Start-state: S5b constrained-.ic (uic-equivalent), commit 7454034.

| mode | period (ps) | NG (ps) | err% | Δ vs pre-S5b 50.83 | NRMSE% | R2 | partial | reached (ns) | wall (s) | osdi calls/hits/fail |
|------|------------:|--------:|-----:|-------------------:|-------:|---:|:-------:|-------------:|---------:|---------------------:|
| id_q_np | 93.01 | 46.65 | 99.38 | +42.18 | 61.39 | -1.1057 | False | 0.6000 | 4152 | 725462/2207408/0 |

NN-vs-OSDI at injected biases: injected_evals=731700 fallbacks=0  max|Δqg|=50.534 aC  max|Δqd|=34.851 aC  max|Δid|=105.479 uA  max rel id (|id|>1uA)=131.2% at bias=(np.float64(0.784), np.float64(0.233), 0.0, 0.0)

**VERDICT: 93.01 ps vs NG 46.65 ps (99.4% off; P0-I id-only gave 92.30 ps) — INDICTS solver/harness**
