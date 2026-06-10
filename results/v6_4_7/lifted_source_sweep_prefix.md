# Lifted-source NMOS Id-Vgs sweep — PRE-FIX (V6.4.7 P0 diagnostic)

DirectNet LEVEL=73 vs NGSPICE OSDI BSIM-CMG, absolute terminals Vs=vs0, Vb=0, Vd=VDD, Vg=0..VDD (5 mV). Captured BEFORE fixing the `_raw_voltages` NMOS source-frame bug (mosfet_nn.py:232-243).
Note: Vd=VDD here; the 55-config grounded gate biases Vds=VDD/2, so vs0=0 is a qualitative (not bit-exact) control.

| tech | vs0/VDD | vs0 (V) | NRMSE (%) | MRE (%) | R2 | MaxErr (uA) |
|------|---------|---------|-----------|---------|----|-------------|
| TSMC5 | 0.0 | 0.000 | 3.05 | 13.63 | 0.98896 | 6.449 |
| TSMC5 | 0.1 | 0.065 | 24.31 | 200.89 | 0.17790 | 29.507 |
| TSMC5 | 0.2 | 0.130 | 63.83 | 809.40 | -6.05126 | 50.192 |
| TSMC7 | 0.0 | 0.000 | 4.55 | 15.15 | 0.97959 | 13.316 |
| TSMC7 | 0.1 | 0.075 | 10.07 | 81.01 | 0.89117 | 17.143 |
| TSMC7 | 0.2 | 0.150 | 30.95 | 324.68 | -0.16639 | 42.525 |
| TSMC12 | 0.0 | 0.000 | 0.04 | 0.52 | 1.00000 | 0.121 |
| TSMC12 | 0.1 | 0.080 | 19.00 | 158.42 | 0.53417 | 26.580 |
| TSMC12 | 0.2 | 0.160 | 51.84 | 651.91 | -3.31518 | 51.853 |
| TSMC16 | 0.0 | 0.000 | 0.04 | 0.44 | 1.00000 | 0.092 |
| TSMC16 | 0.1 | 0.080 | 18.91 | 155.33 | 0.54125 | 25.869 |
| TSMC16 | 0.2 | 0.160 | 51.37 | 633.65 | -3.14402 | 49.979 |
