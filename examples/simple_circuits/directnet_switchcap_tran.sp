* Switched-capacitor unit cell -- DirectNet LEVEL=73
* Benchmark 3d (docs/plans/2026-05-15-directnet-complex-circuits.md)
*
* A CMOS transmission gate samples a DC input Vin onto Csample. The clock
* phi (PULSE) closes the TG during the sample phase; during the hold phase
* the TG is open and Csample should retain its charge -- residual droop is
* sub-threshold leakage. A clock-inverter generates the complementary phib.
*
* The harness measures (i) charge-transfer accuracy at the end of a sample
* window and (ii) hold-phase droop. TECH=/VT=/VDD rewritten per technology.
* PULSE uses PyCircuitSim's space-separated syntax: V1 V2 TD TR TF PW PER.
* L_n=16n / L_p=20n / NFIN=2 match the tsmc*_dn_medium checkpoints.

Vdd vdd 0 0.80
Vin vin 0 0.48
Vphi phi 0 PULSE 0 0.80 0.5n 0.1n 0.1n 1.9n 4n

* --- hierarchical (V6.12.0): clock inverter + transmission gate are
* --- .subckt instances; phi/phib/vsamp stay top-level via the ports ---
Xck phi phib vdd ckinv
Xtg vin vsamp phi phib vdd tgate

Csample vsamp 0 100f

.ic V(vsamp)=0.0 V(phib)=0.80

* --- clock inverter: phi -> phib ---
.subckt ckinv i o vdd
Mpc o i vdd vdd pmos_nn L=20n NFIN=2
Mnc o i 0   0   nmos_nn L=16n NFIN=2
.ends

* --- CMOS transmission gate: a <-> b ---
.subckt tgate a b phi phib vdd
Mnt a phi  b 0   nmos_nn L=16n NFIN=2
Mpt a phib b vdd pmos_nn L=20n NFIN=2
.ends

.model nmos_nn NMOS (LEVEL=73 TECH=tsmc12 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc12 VT=svt)

.tran 5p 12n uic

.end
