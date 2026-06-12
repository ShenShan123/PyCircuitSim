# Lifted-source NMOS Id-Vgs sweep — s7vc (V6.4.7 P0)

DirectNet LEVEL=73 vs NGSPICE OSDI BSIM-CMG, absolute terminals Vs=vs0, Vb=0, Vd=VDD, Vg=0..VDD (5 mV grid). `_raw_voltages` NMOS source-frame canary (mosfet_nn.py:232-243).
Note: Vd=VDD here; the 55-config grounded gate biases Vds=VDD/2, so vs0=0 is a qualitative (not bit-exact) control.

| tech | vs0/VDD | vs0 (V) | NRMSE (%) | MRE (%) | R2 | MaxErr (uA) | verdict |
|------|---------|---------|-----------|---------|----|----------|---------|
| TSMC5 | 0.0 | 0.000 | 3.05 | 13.63 | 0.98896 | 6.449 | PASS |
| TSMC5 | 0.1 | 0.065 | 2.72 | 15.09 | 0.98973 | 4.461 | PASS |
| TSMC5 | 0.2 | 0.130 | 2.51 | 18.48 | 0.98908 | 2.753 | PASS |
| TSMC7 | 0.0 | 0.000 | 4.55 | 15.15 | 0.97959 | 13.316 | PASS |
| TSMC7 | 0.1 | 0.075 | 4.42 | 16.13 | 0.97904 | 11.046 | PASS |
| TSMC7 | 0.2 | 0.150 | 4.41 | 17.18 | 0.97637 | 8.953 | PASS |
| TSMC12 | 0.0 | 0.000 | 0.04 | 0.52 | 1.00000 | 0.121 | PASS |
| TSMC12 | 0.1 | 0.080 | 0.05 | 0.68 | 1.00000 | 0.108 | PASS |
| TSMC12 | 0.2 | 0.160 | 0.07 | 0.93 | 0.99999 | 0.089 | PASS |
| TSMC16 | 0.0 | 0.000 | 0.04 | 0.44 | 1.00000 | 0.092 | PASS |
| TSMC16 | 0.1 | 0.080 | 0.06 | 0.60 | 1.00000 | 0.100 | PASS |
| TSMC16 | 0.2 | 0.160 | 0.08 | 0.86 | 0.99999 | 0.106 | PASS |
