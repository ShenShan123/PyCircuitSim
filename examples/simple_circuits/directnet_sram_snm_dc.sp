* 6T SRAM read-SNM half-cell — NN candidate
*
* The scored half of the SRAM static-noise-margin gate.
* `tests/simple_circuits/verify_circuit_sram_snm.py` renders this deck per technology and per
* fin count and scores it point-by-point against the NGSPICE BSIM-CMG lobe in
* `bsimcmg_sram_snm_dc.cir`.
*
* The read-SNM butterfly is a pair of voltage-transfer curves. This deck is
* one of them: the cross-coupled feedback is BROKEN — q is driven by an ideal
* source and qb is read — because a closed latch has no transfer curve to
* trace. The mirror lobe comes from reflecting this one.
*
* Read bias: word line asserted, bit line precharged high.
*
* The strict renderer supplies technology, VT, supply, geometry, temperature,
* and fin count explicitly. Nominal L_n=16n / L_p=20n matches the clean
* per-technology checkpoint geometry.

Vdd vdd 0 <VDD>
Vwl wl 0 <WL>
Vbl bl 0 <VDD>
Vq q 0 0.0

* --- storage inverter q -> qb, plus the access transistor bl <-> qb ---
Mpl qb q vdd vdd pmos_nn L=<LP> NFIN=<NFN>
Mnl qb q 0 0 nmos_nn L=<LN> NFIN=<NFN>
Mna bl wl qb 0 nmos_nn L=<LN> NFIN=<NFN>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>

.end
