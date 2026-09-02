* Switched-capacitor unit cell — NN candidate
*
* A CMOS transmission gate samples a DC input Vin onto Csample. The clock
* phi (PULSE) closes the TG during the sample phase; during the hold phase
* the TG is open and Csample should retain its charge -- residual droop is
* sub-threshold leakage. A clock-inverter generates the complementary phib.
*
* The harness measures (i) charge-transfer accuracy at the end of a sample
* window and (ii) hold-phase droop. Tokens are rendered per technology.
* PULSE uses PyCircuitSim's space-separated syntax: V1 V2 TD TR TF PW PER.
* Geometry and clock parameters are explicit tokens.

Vdd vdd 0 <VDD>
Vin vin 0 <VIN>
Vphi phi 0 PULSE 0 <VDD> <TD> <SLEW> <SLEW> <PW> <PER>

* --- clock inverter + transmission gate (flat authoritative topology) ---
Mpc phib phi vdd vdd pmos_nn L=<LP> NFIN=<NFP>
Mnc phib phi 0 0 nmos_nn L=<LN> NFIN=<NFN>
Mnt vin phi vsamp 0 nmos_nn L=<LN> NFIN=<NFN>
Mpt vin phib vsamp vdd pmos_nn L=<LP> NFIN=<NFP>
Csample vsamp 0 <CSAMPLE>
.ic V(vsamp)=0.0 V(phib)=<VDD>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>

.end
