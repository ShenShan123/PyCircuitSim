* NMOS Level 1 (Shichman-Hodges) Model Test
* Simple test using direct NMOS keyword (no .model directive)
* This tests the Level 1 implementation

* Voltage sources
Vds 2 0 1.0
Vgs 1 0 1.0

* NMOS transistor (drain=2, gate=1, source=0, bulk=0)
* Using default parameters: VTO=0.7V, KP=20uA/V^2
Mn1 2 1 0 0 NMOS L=1u W=10u

* DC sweep: Vgs from 0 to 2V
.dc Vgs 0 2 0.05

.end
