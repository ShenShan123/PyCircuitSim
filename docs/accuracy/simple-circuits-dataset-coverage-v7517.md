# V7.5.17 simple-circuit dataset coverage gate

This gate answers one question: does every scored V7.5.17 single-device,
inverter, and circuit configuration have nearby training data from the
exact technology/VT label, temperature, and legal PDK `(L, NFIN)` bin?

The gate enumerates the same configurations as the simple-circuit campaign,
loads the exact per-technology NMOS/PMOS NPZ and label sidecars, and rejects
marginal-only matches that do not satisfy the joint geometry constraints.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n pycircuitsim python \
  tests/single_devices/verify_data_geometry_coverage.py
```

Result on 2026-08-25: **339/339 PASS**. No configuration was skipped. This is
a dataset-support gate, not circuit-accuracy evidence; the complete NGSPICE
comparison campaign is reported separately in the generated clean-family
reports in this directory.
