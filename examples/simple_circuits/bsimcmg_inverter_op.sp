* CMOS inverter operating point — BSIM-CMG LEVEL=72 (ASAP7 RVT, VDD=0.7 V)
*
* The scored half of the inverter-OP gate
* (`tests/simple_circuits/verify_bsimcmg_inverter_op.py`), checked against the
* NGSPICE ground truth in `bsimcmg_inverter_op.cir`.
*
* L=30n / NFIN=10 deliberately match the single-device OP gates in
* `tests/single_devices/verify_bsimcmg_op.py`: same geometry, same rail, so a
* disagreement here is about the inverter and not about the device.
*
* The harness rewrites the `Vin` line to bias the input at each end of the
* rail (0 V, then VDD) and reads V(out) — the two points where the inverter
* is fully switched and the answer is unambiguous.

Vdd 1 0 0.7
Vin 2 0 0.0

Xinv 2 3 1 inv

.subckt inv i o vdd
Mp1 o i vdd vdd pmos1 L=30n NFIN=10
Mn1 o i 0 0 nmos1 L=30n NFIN=10
.ends

.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

.op

.end
