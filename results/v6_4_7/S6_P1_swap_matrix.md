# S6 = P1 — causal swap matrix completed + LEVEL=72 native control: simulator EXONERATED, P0-I RETRACTED (V6.4.7, 2026-06-11)

**Status:** diagnostic resolved. No production code changed. The plan's P1
decision branch is answered — by the *control*, not the injection: the RO
error is **model-owned**; every model-side RO lever stays live; **P5 is
funded**; **P0-I's "id-VALUE non-separable from charge" is RETRACTED** as an
injection-scheme artifact, re-arming all id-only levers (P4 Sobolev, P8a,
frozen-base LoRA).

## Commands / artifacts

- `scripts/v6_4_7_s6_p1_swap_matrix.py` (injection, extends P0-I v2 to the
  full 13-key consistent OSDI op-point; FD-verified conventions) — logs
  `s6_logs/s6_p1_{baseline,id_q_np,id_q_nmos_only}.log` + waveform npz.
- `scripts/v6_4_7_s6_l72_ro_control.py` (the decisive native control) — log
  `s6_logs/s6_l72_control.log`.
- `scripts/v6_4_7_s6_artifact_probe.py` (class-method-level device diff) and
  `scripts/v6_4_7_s6_capsign_experiment.py` (within-NN cap-convention flip)
  — log `s6_logs/s6_capsign_experiment.log`.

## The measurement matrix (TSMC7 5-stage RO, S5b uic start, settle 0.3 ns, NG 46.64–46.65 ps)

| Cell | id path | charges | caps | period (ps) |
|------|---------|---------|------|-------------|
| NN baseline (anchor, = pre-S5b bit-for-bit) | NN | NN | NN (+∂Q/∂V) | 50.83 |
| NN with off-diag caps flipped to L72 conv | NN | NN | −NN | 51.49 |
| P0-I id-only injection (V6.4.6) | OSDI-mapped | NN | NN | 92.30 |
| S6 id+q NMOS-only injection | OSDI-mapped (N) | OSDI (N) | OSDI-mapped (N) | 92.91 |
| S6 id+q N+P injection (headline) | OSDI-mapped | OSDI | OSDI-mapped | 93.01 |
| **Native LEVEL=72 (control)** | OSDI native | OSDI native | OSDI native | **46.64 — ratio 1.000 vs NG** |

Control details: same solver, same runner (S5b constrained-`.ic`), same
window/estimator, same resolved TSMC7 ULVT cards on both sides (sha256s in
the control log); pycircuitsim err vs NGSPICE = **0.02 %**.

## Findings

1. **Simulator/harness EXONERATED.** The production LEVEL=72 path reproduces
   NGSPICE through the identical stamps/integrator/start-state. A universal
   solver bug was already implausible (three NN ROs pass at ~3 %); now it is
   measured out. The §2 row "P1 → ~92 ps ⇒ pause all model-side RO levers"
   resolves AGAINST the pause.
2. **The ~92–93 ps injection numbers are artifacts of the injection id-path
   mapping, not physics.** Adding exact charges+caps to the id injection
   moved it 92.30 → 93.01 ps (nothing); the within-NN cap-flip experiment
   bounds the entire cap-convention question at ±1.3 % (second-order, and
   the NN's native +∂Q/∂V convention is the *better* one — flipping worsens
   all four techs). By elimination the artifact lives in the id-path mapping
   (`gds = floor(−OSDI gds)` → |id|/2 everywhere conducting (3–6× low vs
   true OSDI gds per the artifact probe), Rule-15 bypass, possible residual
   sign subtleties). Exact line item NOT identified — recorded as an open
   curiosity, NOT load-bearing: no decision depends on it.
3. **P0-I RETRACTED.** Its "id-VALUE not separable from charge — id-only
   fixes no longer de-risked, must consider joint id+charge correction"
   conclusion rested on the 92 ps number now shown to be scheme-borne. The
   V6.4.6 record gets an annotation. What P0-I *did* prove survives: the
   naive inconsistent swap diverges; injection-style causal probes on this
   codebase are convention-fragile (use the native L72 device as the
   exact-physics endpoint instead — it is cheaper too: 129 s vs ~4,400 s).
4. **The artifact-probe device diff is a permanent reference**: NN vs L72
   class methods agree on id (after the class sign maps), gm, gmb, and ALL
   charges (+1.000) and disagree on off-diag cap sign (convention, benign)
   and gds magnitude policy (floors — Jacobian-only).
5. **RO ownership after S6:** NGSPICE-matching needs the OSDI id surface;
   the NN's documented ~20 % dynamic peak pull-down under-prediction
   (P0-G/P0-H) is the prime — now unclouded — suspect, with charges already
   proven exact (P0-H) and integration/truncation ~0.4 ps (P0-G).

## Consequences for the campaign

- **P5 funded** (corridor harvest from NGSPICE waveforms), re-scoped toward
  the **id surface along trajectories** (charges are exact; the joint-(id,q)
  premise is void).
- **P4 / P8a / LoRA id-only levers re-armed**; every retrain arm still
  scores the live RO period early (cheap, and the cancellation caveat is
  moot now that the 92 ps evidence is retracted).
- The deferred V6.4.6 split-head cap-head remains unnecessary for RO
  (P0-C + the cap-flip experiment both bound cap effects at second order).
