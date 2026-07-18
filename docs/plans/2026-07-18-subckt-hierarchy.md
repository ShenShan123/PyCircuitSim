# V6.12.0 — .subckt hierarchical netlists + test-circuit conversion

**Status: DONE (2026-07-18) — feature landed + circuits converted (commit
1744b28); full campaign re-run with ZERO regressions: verify_subckt 8/8 (new),
DC L2 81/81, tran L2 45/45, L3 DC 52/53+1 documented pre-existing ERROR, L3
tran 86/86, NN 24/24, complex 17/20 (= exact V6.11.0 production state; the 3
opamp misses tsmc6/7/16 are the documented ones). Full table in CHANGELOG
V6.12.0. NN parametric mirrors + lifted-source ran as follow-up JobD.**

## Scope

1. Parser support for `.subckt`/`.ends` definitions and hierarchical `X`
   instances (nested defs + X-in-X), with parameter passing (`{expr}`
   arithmetic), ngspice-style flattening (`X1.n1` nodes, `M.X1.Mp1` devices),
   and `.ic` support: cards inside subckt bodies (node remap + param values)
   and top-level hierarchical `.ic V(X1.n1)=v`.
2. Convert the project's test circuits to `.subckt` hierarchy and re-run the
   verification suites to collect pass rates.

## Design decisions (durable)

- **Flattening at parse time** in `parse_file`: collect defs → model/include
  pre-pass → expand X lines → normal second pass. Node names are opaque
  strings end-to-end, so no solver/circuit changes were needed.
- **Probed nodes stay top-level** in every converted test circuit (they are
  subckt ports), so all harness result keys, baselines, and NGSPICE
  comparisons are unchanged. Deep internal-node hierarchy is exercised by
  `tests/verify_subckt.py` L3 instead.
- **Single-device Id-Vgs decks stay flat**: the DC harness probes device
  currents by name (`i(Mn1)`); renaming via hierarchy would break the probe
  keys for no structural gain.
- **NGSPICE reference decks stay flat** — ground truth is never restructured.
- Sweep canaries forced template ↔ builder lockstep: both sides emit the
  same subckt line-sets (verified: ALL CANARIES PASS, 5 techs).

## Campaign (2026-07-18, cluster loadavg ~1330)

- JobA: verify_subckt, OP, DC L1, TRAN L1, AC, DC L2 (67), TRAN L2 (37)
- JobB: multi-tech DC L3 (53), multi-tech TRAN L3 (86)
- JobC: verify_nn_dc_tran (TSMC5/7/12/16), complex gates
  (ring/opamp/switchcap/SRAM × TSMC5/6/7/12/16 = 20 gates)

Results → CHANGELOG V6.12.0 entry once collected.
