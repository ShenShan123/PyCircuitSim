* 6T SRAM cell -- DirectNet LEVEL=73
* Benchmark 3c (docs/plans/2026-05-15-directnet-complex-circuits.md)
*
* Standard 6T bitcell: two cross-coupled inverters (Mpl/Mnl, Mpr/Mnr) and
* two NMOS access transistors (Mal/Mar) to the bit lines. Read condition:
* WL asserted, both bit lines precharged to VDD.
*
* The read static-noise-margin (SNM) butterfly is traced by the verify
* harness, which breaks the cross-coupled feedback and sweeps one storage
* node while reading the other (and vice-versa) -- see verify_complex_sram_snm.py.
* The full cross-coupled netlist below is solved with force_ic=True (hard .ic
* mode) to land each storage state, exercising the same path SRAM latches use.
*
* L_n=16n / L_p=20n / NFIN=2 match the tsmc*_dn_medium checkpoints.

Vdd vdd 0 0.80
Vwl wl 0 0.80
Vbl bl 0 0.80
Vblb blb 0 0.80
.ic V(q)=0.80 V(qb)=0.0

* --- left inverter: q -> qb ... cross-coupled ---
Mpl qb q vdd vdd pmos_nn L=20n NFIN=2
Mnl qb q 0   0   nmos_nn L=16n NFIN=2
* --- right inverter: qb -> q ---
Mpr q qb vdd vdd pmos_nn L=20n NFIN=2
Mnr q qb 0   0   nmos_nn L=16n NFIN=2
* --- access transistors ---
Mal bl  wl q  0 nmos_nn L=16n NFIN=2
Mar blb wl qb 0 nmos_nn L=16n NFIN=2

.model nmos_nn NMOS (LEVEL=73 TECH=tsmc12 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc12 VT=svt)

.op

.end
