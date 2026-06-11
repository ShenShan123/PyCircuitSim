# S5b — uic-equivalent `.ic` start in the DN complex-benchmark runner (V6.4.7, 2026-06-11)

**Change (`tests/common/complex.py` `run_directnet_transient`):** when the
netlist carries `.ic`, the DC OP is solved with each `.ic` node pinned by a
temporary ideal V-source (skipping V-source-constrained nodes and nodes
absent from the node map, mirroring solver force_ic's guards); pins are
removed in a `finally`; the transient starts from the **constrained**
solution. Previously `.ic` was only the OP's NR guess — the OP converged to
the NN's unconstrained leakage equilibrium, so DN started at vsamp(0)=0.390 V
(TSMC5, =Vin) / 0.704 V (TSMC16) while NGSPICE ran `tran ... uic` from
vsamp=0. The benchmark compared two different experiments; under the old
protocol **a bit-perfect model would still have failed** (TSMC5's old
"14.65 %" equals (Vin−NG_chg)/VDD exactly — pure protocol artifact).

## A/B (post-P0 code, repaired droop gate)

| tech | ChgErr% old → new | droop old → new | verdict old → new |
|------|-------------------|------------------|---------------------|
| TSMC5 | 14.65 → 11.96 | 0.000 → 0.000 mV | FAIL → FAIL (overshoot: NN TG over-conducts forward) |
| TSMC7 | 3.06 → 3.40 | 2.208 → 0.541 mV | FAIL → **PASS** (droop now within the principled floor) |
| TSMC12 | 10.29 → 8.14 | 0.001 → 0.136 mV | FAIL → FAIL (now UNDershoot: forward id too weak) |
| TSMC16 | 13.14 → 6.20 | 0.002 → **3.852 mV** | FAIL → FAIL (charge + a REAL hold leak newly exposed, 481 % of allowance) |

SC census 0/4 → 1/4; failures are now genuine forward-conduction id errors +
real hold leaks (campaign territory: P4/P3), not artifact recovery.
**RO blind veto HELD:** periods bit-identical to baseline at log precision
(75.41/50.83/83.85/92.67 ps; 3/4; TSMC7 8.98 %) — only ungated startup-phase
waveform metrics moved. Butterfly/opamp/inverter suites use other runners
(unaffected). **Headline: 9/16 → 10/16** (with the fragility caveat below).

## Adversarial review (agent ab281c3437251b92f): CORRECTION, with conditions

- uic-equivalence is **exact for both affected netlists** — every non-source
  node is covered by `.ic`, so the constrained OP pins all unknowns (NR only
  resolves branch currents/charge state at the .ic bias, same as NGSPICE
  uic). Residual t=0 mismatch ≤37 µV vs a 32–40 mV gate resolution.
- Implementation verified: op_sol carries node-voltage keys only; GMIN retry
  runs with pins installed; `finally` removal on exceptions; no name
  collisions; PULSE sources correctly in the constrained-node scan.
  node-map guard added per review.
- An engineered change does not ADD a 481 % failure (TSMC16 leak) — both
  directions moved.

### Required caveats (all recorded here)

1. **TSMC7 robustness probe = 2/3 → fragile PASS, no S5b improvement
   claim.** (`scripts/v6_4_7_s5b_tsmc7_robustness.py`) Droop — the
   adversary's actual concern — held at ALL Vin points (0.703/0.541/0.107 mV
   at 0.55/0.60/0.65·VDD, monotone-improving). The cell-level FAIL at
   0.65·VDD is the **charge** sub-gate (5.36 % > 5) — the same
   forward-under-conduction error failing TSMC12/16 at default Vin, growing
   with Vin. TSMC7 SC counts in the headline under the defined gate config
   (Vin=0.6·VDD) but is marked **fragile**; an off-default-Vin SC variant is
   mandatory in the S19 blind holdout set, and any campaign winner must hold
   it.
2. **Production `.ic` semantics known-issue:** `pycircuitsim/simulation.py`
   `run_transient` (~:536) still feeds an unconstrained OP into the
   transient — production users get exactly the artifact-start behavior the
   harness just abandoned. Out of S5b scope (harness-only measurement fix);
   needs a `uic` option decision in a future iteration.
3. **TSMC16 equilibrium artifact recorded:** the NN's unconstrained OP put
   vsamp at 0.704 V — a physically impossible equilibrium (both Rule-15
   corrections zero id at Vds=0, so the true equilibrium is vsamp=vin). No
   longer visible to the SC gate; the off-state id-magnitude error class
   stays exposed via SRAM force_ic (0/8) and the new TSMC16 droop FAIL.
4. **REV-clamp coverage note:** SC's sample window is now forward-conduction,
   so the P2 reverse-recovery defect (+11.6 fC withheld, S5 dump) is no
   longer exercised by any passing-relevant SC path; P2's evidence base is
   unchanged (S5 dump + SRAM Mpr) but its SC-side EV shrinks.
