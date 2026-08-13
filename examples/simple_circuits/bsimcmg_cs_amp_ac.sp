* BSIM-CMG NMOS Common-Source Amplifier — AC (small-signal) Analysis
*
* The scored half of the AC gate's Level 2
* (`tests/simple_circuits/verify_ac.py`), checked against the NGSPICE ground
* truth in `bsimcmg_cs_amp_ac.cir`. Editing this deck changes what that gate
* simulates.
*
* Single-transistor common-source gain stage with resistive load RD:
*   - Vin DC=0.35 V biases the NMOS in saturation (Vout ~ 0.51 V, mid-rail).
*   - AC=1 injects the small-signal stimulus at the gate.
* The output node "out" shows the low-frequency voltage gain -gm*(RD||ro)
* rolling off through the Miller (Cgd) feedback plus the RD/Cload output pole
* — the device capacitances ARE the roll-off, which is what makes this the
* smallest circuit that gates the AC transcapacitance stamp.
* Bulk is tied to source (0) so the source-referenced small-signal
* capacitance matrix is exact.
*
* ASAP7 RVT at L=30 nm / NFIN=2 / VDD=0.7 V. The gate bakes its NGSPICE
* modelcard from the same ASAP7 baseline profile, so a geometry edited here
* and not there shows up as a gain disagreement, loudly.
*
* Hierarchical (.subckt): the gain stage is one instance, `out` stays
* top-level through the port. `verify_subckt.py` proves subckt = flat
* bit-identical, so the hierarchy costs the comparison nothing.

VDD vdd 0 0.7
Vin in 0 DC=0.35 AC=1 0

Xamp in out vdd csamp

Cload out 0 5f

.subckt csamp in out vdd
RD vdd out 50k
Mn1 out in 0 0 nmos_rvt L=30n NFIN=2 TFIN=6.5n
.ends

.model nmos_rvt NMOS (LEVEL=72)

* AC sweep: 20 points/decade, 1 kHz to 1 THz
.ac dec 20 1e3 1e12

.end
