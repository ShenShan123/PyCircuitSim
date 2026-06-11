# S6 control — native LEVEL=72 TSMC7 RO vs NGSPICE (window 0.050 ns, settle 0.010 ns, tstep 2 ps)

Circuit: 5-stage RO, PMOS pch_ulvt_mac L=20n + NMOS nch_ulvt_mac L=16n, NFIN=2, TFIN=6.0n, 0.5 fF/stage, VDD=0.75, alternating .ic, S5b constrained-.ic start. NO device monkeypatch — production NMOS_CMG/PMOS_CMG.

Modelcard sources (both sides):
- NMOS `/data2/home/shenshan/NN_SPICE-refactor-nn/external_compact_models/PyCMG/build/modelcards/TSMC7/nch_ulvt_mac_l16nm_nfin2.l` sha256 `a2c40baa7e1d3c5c06ad021eb35aae9c270c1cd939a7fdf8b2d6a32e1402096a`
- PMOS `/data2/home/shenshan/NN_SPICE-refactor-nn/external_compact_models/PyCMG/build/modelcards/TSMC7/pch_ulvt_mac_l20nm_nfin2.l` sha256 `c2854c853809bd1a8dd500d23fb293ef8f1f68269da2ad53fbbb584016fc8f92`

| run | period (ps) | ratio vs NG | err% | NRMSE% | R2 | partial | reached (ns) | wall (s) |
|-----|------------:|------------:|-----:|-------:|---:|:-------:|-------------:|---------:|
| PyCircuitSim L72 | nan | nan | nan | 12.65 | 0.9081 | False | 0.0500 | 24 |
| NGSPICE (truth) | nan | 1.000 | — | — | — | — | 0.050 | — |
| S6=P1 id+q injection (context) | 93.01 | nan | — | — | — | — | — | — |

L72 wave: vmin=-0.012 vmax=0.772 swing=0.784 V, n_rise=1, n_fall=1

half-periods (ps): [22.9]

**VERDICT: L72 period not measurable (NaN) — no oscillation in window or non-periodic; judge from waveform/half-periods**
