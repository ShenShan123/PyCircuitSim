# DirectNet-Full LEVEL=75 — V7.6.0 recovery evidence

## Verdict

V7.6.0 adds a separate full-terminal DirectNet family because the exact OSDI
reduced-boundary control failed materially on a high-temperature analog case.
The implementation is experimental: no LEVEL=75 checkpoint has passed the
truth-surface, simple-circuit, AnalogGym, multi-seed, or performance promotion
gates. DirectNet LEVEL=73 remains the production NN path.

Ground truth in every circuit comparison below is NGSPICE LEVEL=72 using the
identical BSIM-CMG OSDI model. Scored inference was CPU-pinned to one OpenMP,
MKL, and Torch thread. Local evidence is under ignored `results/v760-*` roots;
the attribution harness commits are `e7f1c88`, `794de6d`, and `eaab66e`.

## Attribution results

The instance-multiplier defect was independently real but not sufficient. On
the TSMC5 LDO line sweep, propagating `m` through the NN current, conductance,
charge, and capacitance boundary reduced maximum node error from 12.3250 V to
0.306632 V; the deck still scored 0/3. The high-temperature Fan sweep improved
but remained 6/15, and the charge pump still failed.

| TSMC5 diagnostic | full OSDI | exact reduced OSDI | conclusion |
|---|---:|---:|---|
| LDO line sweep | 3/3, 96.3 µV max | 3/3, 179.7 µV max | Reduced boundary is adequate here |
| Fan temperature sweep | 15/15, 115 µV max | 12/15, 817.3 V max | Omitted terminal rows are material |

The exact reduced boundary was therefore retained as a diagnostic and rejected
as a production contract. It is a structural prerequisite, not a demonstrated
sole cause: it passed the LDO and paired-ring controls, while full production
DirectNet remained inaccurate.

Removing production corrections was also rejected. On the same LDO sweep,
raw DirectNet reduced the maximum error from 0.306632 V to 0.142730 V only over
the comparable states, but converged at 9/28 points instead of 28/28 and still
scored 0/3. Paired ring results moved in both directions. Correction activation
was observed, but this ablation did not establish it as the initiating cause.

The same-state evaluator probe aligns LEVEL=72 and LEVEL=73 devices by folded
topology, geometry, and instance multiplier, evaluates full OSDI, reduced
OSDI, raw DirectNet, and production DirectNet at identical explicit states,
and solves the augmented voltage-source branch-current tail before reporting
current-row residuals. It passed 9/9 device alignments on the real TSMC5 LDO.

## Dataset controls

The unchanged current generator produced only 7/10 canonical technology and
polarity artifacts. TSMC5/6/7 PMOS failed loudly with 1,370/335/335 rejected
rows, all classified as terminal current above 1 A. No diagnostic artifact was
used for training. TSMC5/6/7 NMOS and both polarities for TSMC12/16 completed.
This is an incomplete procedure control, not accuracy evidence.

For the new six-surface contract, a real TSMC5 nominal-envelope probe found
132 PMOS rejections across 44 bins. Every rejected coordinate belonged to the
`vds_zero` class at `Vd=0`, `Vgs=-1.3 V` (`-2×VDD`), despite the requested
1.0×VDD envelope. After making only the full-terminal `vds_zero` class honor
the declared envelope, the same OSDI run retained 5,008,380/5,008,380 rows and
780/780 bins with zero rejects; NMOS independently retained the same row and
bin counts. The legacy reduced generator keeps its 2×VDD class unchanged.

Those nominal-envelope runs had dirty-source completion metadata and are
diagnostic only. They justify retaining the code change but do not clear the
artifact gate. No training or circuit score is claimed from them.

## Implemented LEVEL=75 contract

- Dataset and trainer use six independent targets: solver-positive `i_d`,
  `i_g`, `i_b`, plus `qd`, `qg`, and `qb`.
- Runtime derives source current and charge analytically, reconstructs the
  source Jacobian column from voltage-translation invariance, and reconstructs
  the source row from KCL/charge closure.
- DC/OP differentiates only the three current surfaces. AC/transient explicitly
  request the three charge Jacobians; a direct charge consumer self-heals.
- Parser selection requires `LEVEL=75 FAMILY=directnet-full TECH=... VT=...`.
  LEVEL=73 campaign decks can be explicitly retargeted with
  `PYCIRCUITSIM_NN_FORCE_LEVEL=75`.
- Checkpoint loading requires the distinct `dnf` family stem plus matching
  normalization and completion-marker names, schemas, and SHA-256 hashes.
- Generator and trainer defaults isolate six-surface datasets and checkpoints
  under `dnf` names; scored campaigns reject diagnostic evaluator/trace rows.
- Inference is source-relative and fails explicitly outside the artifact's
  recorded input bounds. It has no inherited LEVEL=73 correction policy.

## Promotion status

Artifact generation, three independent training seeds, truth-surface metrics,
the full simple/complex gate matrix, the untouched AnalogGym holdout and full
basket, and CPU performance remain outstanding. Until those complete without
model-induced errors or denominator shrinkage, LEVEL=75 is an implementation
and research boundary only.
