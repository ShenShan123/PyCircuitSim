# Plan: TSMC6 (CLN6) onboarding + TSMC PDK parse audit — V6.9.0

**Status: EXECUTED 2026-07-12** (branch `feat/tsmc6-onboarding`, main repo +
PyCMG submodule). Results: `docs/V6.9.0-tsmc6-onboarding-pdk-parse-audit.md`
+ CHANGELOG V6.9.0. This file records the routing and the live follow-up
frontier.

## Scope (user-confirmed)

Full LEVEL=72 (BSIM-CMG ground truth) support for the new tech TSMC6
(`/data2/shenshan/TN06CLSP001W1_1_0_2P2A/.../cln6_1d8_sp_v1d0_2p2.l` →
`PyCMG/modelcards/TSMC6/`, gitignored IP) + NN config plumbing + dataset
generation. **NN checkpoint training deferred.** Plus: audit PyCMG parsing of
all TSMC cards.

## Executed

- Phase A: worktree `.claude/worktrees/tsmc6-onboarding`, submodule branch,
  gitignored assets (PDK cards ×5, OSDI binary, tools symlink) copied in.
- Phase B: PyCMG registry/config/CLI/test wiring (314-test suite).
- Phase C: bsimar universal codes 22–24 tail-append (NUM_TOTAL_CODES=25,
  legacy vocab untouched), tsmc6 scope + local vocab, labeller
  fingerprint-collision fix (tech_filter by stem + collision-raise).
- Phase D: `ALL_TECHS["TSMC6"]` full profile — NO pruning needed (all 3 VTs ×
  L16/20/24 pass, unlike TSMC7).
- Phase E: `PyCMG/scripts/audit_pdk_parse.py` — AUDIT PASS, 40 devices,
  0 round-trip mismatches; verdicts in the report.
- Phase F: `tsmc6_{nmos,pmos}.npz` (1.82M/2.19M rows) + sidecars
  (codes 22/23/24), smoke-loaded with tech_scope=tsmc6; copied to main
  checkout.
- Verification: PyCMG 314/314; TSMC6 DC 9/9, tran 14/14; L1 trio PASS;
  full 6-tech DC/tran regression run at close (legacy techs reproduce).

## Follow-up frontier (next campaigns)

1. **TSMC6 NN training campaign** — ✅ **DONE in V6.11.0** (2026-07-14/17). All
   three families' CLEAN capacity sweeps trained + gated vs NGSPICE (curriculum
   recipes deferred — user scoped to "all scales", clean only): DirectNet
   s/m/l/xl (peaks **large 3/4**, banks ring), PFN s/m/l (flat **2/4**), BSIM-AR
   s/m/l/xl (peaks **medium 3/4**, the only family to bank the tsmc6 opamp 9.83%).
   All the wiring in this bullet was done — `complex.py` BENCH_TECHS + VT map,
   `nn_sweep.py` NN_TECHS, `verify_nn_dc_tran.py` tables + `tsmc6_dn_`/`tsmc6_tf_`
   sentinels, the 5 collectors, `recipe_eval.sh` (TECHS-overridable) — plus a
   `tech_code_in_vocab` fix (rejected tsmc6 univ codes 22–24). **Prediction
   CONFIRMED: tsmc6 tracks tsmc7's basket** — the opamp is the tsmc7-family hard
   cell, cracked only by BSIM-AR's opamp basin; the ring by DN-large (opamp-XOR-ring
   split). See CHANGELOG V6.11.0 + the DN/TF/PFN report addenda. Env caveat: 2
   BSIM-AR large/xl opamp cells indeterminate under cluster overload (re-gate-when-idle).
2. **Universal + tsmc6** — requires `num_tech_codes=NUM_TOTAL_CODES` (25) at
   train time and extending `uni_concat_npz.py` (code-subset validation is
   pinned to the V6.7.0 {4..16} build). Fingerprint labelling CANNOT
   distinguish tsmc6 from tsmc7 (108/108 collisions) — universal sidecars
   must come from per-tech sidecar concatenation.
3. **Optional card hygiene** — stripping OSDI-unknown (tmi*/stat/LOD) numerics
   from naive cards would shrink them ~4× with no behavioral change; touches
   all cached legacy cards, so do it only alongside a full re-gate.
