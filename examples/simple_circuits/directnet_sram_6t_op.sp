* 6T SRAM cell, hold state — DirectNet LEVEL=73
*
* The full cross-coupled bitcell, used by the `force_ic` retention probe in
* `tests/verify_complex_sram_snm.py`. That probe is a printed DIAGNOSTIC, not
* a scored gate — read stability is gated by the butterfly curves in
* `directnet_sram_snm_dc.sp` instead.
*
* Two cross-coupled inverters (Xl / Xr, each an sraminv instance) and two
* NMOS access transistors (Mal / Mar) to the bit lines. The cell is solved
* with force_ic=True (hard `.ic` mode): the `.ic` nodes are stamped as
* temporary voltage-source constraints, then the cell is re-solved
* unconstrained to check the latch holds the state on its own. This is the
* same solver path real SRAM latches exercise.
*
* WORD LINE OFF. This is the hold condition, and it is the one the probe
* runs. An earlier version of this deck asserted the word line with both bit
* lines forced high — that is a read-disturb, not a hold, and V6.4.7 showed
* it rejects ground-truth physics: native LEVEL=72 BSIM-CMG fails it 0/8 and
* passes the hold 8/8, identically to the NN. The harness can still drive the
* read bias for diagnostics (`wl_on=True`).
*
* Authored at the TSMC12 rail (svt, NFIN=2). The harness rewrites TECH= / VT=
* / every supply rail, and replaces the `.ic` line per storage state, so
* editing those values here changes what the probe runs.

Vdd vdd 0 0.80
Vwl wl 0 0.0
Vbl bl 0 0.80
Vblb blb 0 0.80

.ic V(q)=0.80 V(qb)=0.0

* --- the cross-coupled pair: two instances of one sraminv cell.
* --- q/qb stay top-level via the ports, so probes are unchanged.
Xl q qb vdd sraminv NF=2
Xr qb q vdd sraminv NF=2

* --- access transistors ---
Mal bl  wl q  0 nmos_nn L=16n NFIN=2
Mar blb wl qb 0 nmos_nn L=16n NFIN=2

.subckt sraminv i o vdd NF=1
Mpl o i vdd vdd pmos_nn L=20n NFIN=NF
Mnl o i 0   0   nmos_nn L=16n NFIN=NF
.ends

.model nmos_nn NMOS (LEVEL=73 TECH=tsmc12 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc12 VT=svt)

.op

.end
