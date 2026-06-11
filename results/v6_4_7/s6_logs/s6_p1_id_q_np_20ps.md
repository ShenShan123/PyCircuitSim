# S6=P1 swap matrix — mode `id_q_np` (TSMC7 RO, window 0.020 ns, settle 0.005 ns)

Start-state: S5b constrained-.ic (uic-equivalent), commit 7454034.

| mode | period (ps) | NG (ps) | err% | Δ vs pre-S5b 50.83 | NRMSE% | R2 | partial | reached (ns) | wall (s) | osdi calls/hits/fail |
|------|------------:|--------:|-----:|-------------------:|-------:|---:|:-------:|-------------:|---------:|---------------------:|
| id_q_np | nan | nan | nan | +nan | 32.53 | -0.1170 | False | 0.0200 | 148 | 25458/79312/0 |

NN-vs-OSDI at injected biases: injected_evals=26120 fallbacks=0  max|Δqg|=50.534 aC  max|Δqd|=34.851 aC  max|Δid|=105.479 uA  max rel id (|id|>1uA)=131.2% at bias=(np.float64(0.784), np.float64(0.233), 0.0, 0.0)

**VERDICT: period not measurable in this window (NaN) — extend tstop or judge from half-period diagnostics**
