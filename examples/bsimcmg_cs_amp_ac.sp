* BSIM-CMG NMOS Common-Source Amplifier - AC (small-signal) Analysis
* Single-transistor common-source gain stage with resistive load RD.
*   - Vin DC=0.35V biases the NMOS in saturation (Vout ~ 0.51V, mid-rail).
*   - AC=1 injects the small-signal stimulus at the gate.
* The output node "out" exhibits the low-frequency voltage gain
* (-gm*(RD||ro)) rolling off through the Miller (Cgd) feedback + the
* RD/Cload output pole -- the device capacitances ARE the roll-off.
* Bulk is tied to source (0) so the source-referenced small-signal
* capacitance matrix is exact.

VDD vdd 0 0.7
Vin in 0 DC=0.35 AC=1 0
RD vdd out 50k

* NMOS (drain=out, gate=in, source=0, bulk=0) -- BSIM-CMG LEVEL=72
Mn1 out in 0 0 nmos1 L=30n NFIN=2

* Load capacitance
Cload out 0 5f

.model nmos1 NMOS (LEVEL=72)

* AC sweep: 20 points/decade, 1 kHz to 1 THz
.ac dec 20 1e3 1e12

.end
