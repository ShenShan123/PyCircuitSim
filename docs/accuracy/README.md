# Accuracy reports — one per compact-model family

Each file is the **single unified accuracy record** for one NN compact-model family:
every campaign that measured it, consolidated, with the frozen data tables carried
verbatim. Ground truth is always NGSPICE on the identical BSIM-CMG (LEVEL=72) OSDI
model.

| LEVEL | Family | Report | Best config | Complex gates (strict) |
|---|---|---|---|---|
| 73 | **DirectNet** (production) | [`DirectNet-L73-accuracy.md`](DirectNet-L73-accuracy.md) | `crit30f@large`, 0.92 M | **15/16** (16/16 at `crit15m@xl`) |
| 74 | **BSIM-AR Transformer** (higher-fidelity) | [`BSIM-AR-L74-accuracy.md`](BSIM-AR-L74-accuracy.md) | `corroft@medium`, 1.9 M | **16/16** |
| 75 | **PFN / TabPFN** (research) | [`PFN-L75-accuracy.md`](PFN-L75-accuracy.md) | `clean@small`, 0.69 M | 11/16 |

BSIM-CMG (LEVEL=72) is the ground truth, not a graded family; its own gate record lives
in `../CHANGELOG.md` and `../V6.9.0-tsmc6-onboarding-pdk-parse-audit.md`.

**Read these first, whatever the family:**

- **`DirectNet-L73-accuracy.md` §2** — the shared methodology (gate definitions, strict
  OMP discipline, isolation, CPU pinning) that every number in all three files obeys.
- **`DirectNet-L73-accuracy.md` §12.2** — the **gds sign bug**: every number in every
  report was measured with it present, and the measured fix moves AC, `force_ic` and one
  complex gate. Not shipped as of 2026-07-24.
- **TSMC6 ≡ TSMC7 relabelled** (`../2026-07-21-systematic-audit.md` §D1) — TSMC6 rows are
  a second training run on the TSMC7 data, not a sixth technology. Flagged in each report.

**Consolidated 2026-07-24** from `V6.6.0`, `V6.6.1`, `V6.6.6`, `V6.7.0` (→ DirectNet),
`V6.8.0`/`V6.8.1` (→ BSIM-AR), and `V6.10.0` (→ PFN), plus the V6.11.0 TSMC6 addenda.
Those per-version files were removed; git history keeps them (last present at `1fe1cdb`).

All counts are strict across OMP∈{1,2,4} and were re-measured in **V6.13.0**
(2026-07-24) after the `gds` sign + guard fix; every family is now flip-free.
See `DirectNet-L73-accuracy.md` §12.2 for the fix and what it retracts.
