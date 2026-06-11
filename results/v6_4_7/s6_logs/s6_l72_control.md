# S6 control — native LEVEL=72 TSMC7 RO vs NGSPICE (window 0.600 ns, settle 0.300 ns, tstep 2 ps)

Circuit: 5-stage RO, PMOS pch_ulvt_mac L=20n + NMOS nch_ulvt_mac L=16n, NFIN=2, TFIN=6.0n, 0.5 fF/stage, VDD=0.75, alternating .ic, S5b constrained-.ic start. NO device monkeypatch — production NMOS_CMG/PMOS_CMG.

Modelcard sources (both sides):
- NMOS `/data2/home/shenshan/NN_SPICE-refactor-nn/external_compact_models/PyCMG/build/modelcards/TSMC7/nch_ulvt_mac_l16nm_nfin2.l` sha256 `a2c40baa7e1d3c5c06ad021eb35aae9c270c1cd939a7fdf8b2d6a32e1402096a`
- PMOS `/data2/home/shenshan/NN_SPICE-refactor-nn/external_compact_models/PyCMG/build/modelcards/TSMC7/pch_ulvt_mac_l20nm_nfin2.l` sha256 `c2854c853809bd1a8dd500d23fb293ef8f1f68269da2ad53fbbb584016fc8f92`

| run | period (ps) | ratio vs NG | err% | NRMSE% | R2 | partial | reached (ns) | wall (s) |
|-----|------------:|------------:|-----:|-------:|---:|:-------:|-------------:|---------:|
| PyCircuitSim L72 | 46.64 | 1.000 | 0.02 | 9.96 | 0.9445 | False | 0.6000 | 129 |
| NGSPICE (truth) | 46.65 | 1.000 | — | — | — | — | 0.600 | — |
| S6=P1 id+q injection (context) | 93.01 | 1.994 | — | — | — | — | — | — |

L72 wave: vmin=-0.017 vmax=0.772 swing=0.789 V, n_rise=7, n_fall=6

half-periods (ps): [22.9,23.7,22.9,23.7,23. ,23.7,22.9,23.7,22.9,23.7,23. ,23.7]

**VERDICT: L72 46.64 ps vs NG 46.65 ps (ratio 1.000) — solver/harness EXONERATED; the ~93 ps S6=P1 (and P0-I ~92 ps) injection numbers are injection-scheme artifacts**
