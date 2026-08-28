* 6T SRAM read-SNM half-cell — DirectNet LEVEL=73
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
* Authored at the TSMC12 rail (svt, NFIN=2). The harness rewrites TECH= / VT=
* / every supply rail / the fin count per run, so editing those values here
* changes what the gate simulates. L_n=16n / L_p=20n match the per-tech
* DirectNet checkpoints and are the same across all five techs.

Vdd vdd 0 0.80
Vwl wl 0 0.80
Vbl bl 0 0.80
Vq q 0 0.0

* --- storage inverter q -> qb, plus the access transistor bl <-> qb ---
Xinv q qb vdd sraminv NF=2
Mna bl wl qb 0 nmos_nn L=16n NFIN=2

.subckt sraminv i o vdd NF=1
Mpl o i vdd vdd pmos_nn L=20n NFIN=NF
Mnl o i 0   0   nmos_nn L=16n NFIN=NF
.ends

.model nmos_nn NMOS (LEVEL=73 TECH=tsmc12 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc12 VT=svt)

.dc Vq 0 0.80 0.005

.end
