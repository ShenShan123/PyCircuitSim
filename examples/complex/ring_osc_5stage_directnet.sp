* 5-stage CMOS ring oscillator -- DirectNet LEVEL=73
* Benchmark 3a (docs/plans/2026-05-15-directnet-complex-circuits.md)
*
* Five inverter stages in a loop; n5 feeds back to n1. Alternating .ic seeds
* the latch out of its (unstable) DC operating point so oscillation starts.
* TECH=/VT= are placeholders -- the verify harness rewrites this netlist
* per technology and resolves the per-tech DirectNet checkpoint.
*
* L_n=16n / L_p=20n / NFIN=2 match the tsmc*_dn_medium checkpoints.

Vdd vdd 0 0.80
.ic V(n1)=0.0 V(n2)=0.80 V(n3)=0.0 V(n4)=0.80 V(n5)=0.0

* Hierarchical (V6.12.0): one ringinv .subckt instantiated 5x; the stage
* nodes n1..n5 stay top-level through the ports.
Xinv1 n5 n1 vdd ringinv
Xinv2 n1 n2 vdd ringinv
Xinv3 n2 n3 vdd ringinv
Xinv4 n3 n4 vdd ringinv
Xinv5 n4 n5 vdd ringinv

.subckt ringinv i o vdd
Mp o i vdd vdd pmos_nn L=20n NFIN=2
Mn o i 0   0   nmos_nn L=16n NFIN=2
Cl o 0 0.5f
.ends

.model nmos_nn NMOS (LEVEL=73 TECH=tsmc12 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc12 VT=svt)

.tran 1p 5n uic

.end
