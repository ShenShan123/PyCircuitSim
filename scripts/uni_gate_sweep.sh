#!/usr/bin/env bash
# V6.7.0 universal-transfer campaign — Phase 2/4 evaluation driver
# (plan docs/plans/2026-07-02-universal-nn-tsmc5-transfer.md §6/§8).
#
# For each checkpoint stem (default: the Core-4 recipes at $SIZE) it
#   1. verifies BOTH device checkpoints exist AND carry .complete markers,
#   2. env-pins PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS} to the stem,
#   3. runs the selected sections, each cell CPU-pinned
#      (CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1) with an
#      isolated PYCIRCUITSIM_COMPLEX_RESULTS dir and a hard `timeout`:
#        gates    — 12 ranking gates: verify_complex_{ring_osc,opamp,sram_snm,
#                   switchcap} × $TECHS (verdict = exit code)
#        zeroshot — the same 4 circuits at TSMC5 (diagnostic; expected fail
#                   for base universal ckpts — embedding rows 0-3 untrained)
#        dev      — verify_nn_dc_tran.py --tech $DEV_TECHS
#        ac       — verify_nn_ac.py per tech in $TECHS
#        omp      — determinism sweep: opamp+ring × $TECHS ×
#                   OMP/MKL/PYCIRCUITSIM_TORCH_THREADS = N ∈ {2,4}
#                   (N=1 is the base `gates` run — same pinning standard)
#   4. post-checks every cell log for the `[NN-resolver] ... scope=universal`
#      line on BOTH device stems — a cell that ran on the wrong checkpoint is
#      re-verdicted RESOLVER-MISS (campaign-invalid), never counted.
#
# Verdicts land in results/uni_bench/<stem>/SUMMARY.tsv:
#   section circuit tech ompN verdict exit_code wall_s log_path
#
# Usage:
#   bash scripts/uni_gate_sweep.sh                                   # Phase 2
#   STEMS="u716f5_plain_n2000_large" TECHS="TSMC5" SECTIONS="gates dev ac" \
#     bash scripts/uni_gate_sweep.sh                                 # Phase 4
#   RECIPES="clean" SECTIONS="dev" bash scripts/uni_gate_sweep.sh    # anchor
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
CKPT="$ROOT/external_compact_models/bsimar/checkpoints"
OUT="${UNI_OUT:-$ROOT/results/uni_bench}"
SIZE="${SIZE:-large}"
NPAR="${NPAR:-6}"
GATE_TIMEOUT="${GATE_TIMEOUT:-5400}"
ZS_TIMEOUT="${ZS_TIMEOUT:-2700}"
DEV_TIMEOUT="${DEV_TIMEOUT:-10800}"
TECHS="${TECHS:-TSMC7 TSMC12 TSMC16}"
DEV_TECHS="${DEV_TECHS:-$(echo "$TECHS" | tr ' ' ',')}"
SECTIONS="${SECTIONS:-gates zeroshot dev ac omp}"
# Direct env-python (NOT `conda run`): `timeout` must signal python itself —
# conda run interposes a wrapper process that can orphan the grandchild on
# SIGTERM, leaving a runaway solver after a TIMEOUT verdict.
PYBIN="${PYBIN:-/data1/shenshan/.conda/envs/pycircuitsim/bin/python}"
export PYTHONPATH="$ROOT/external_compact_models${PYTHONPATH:+:$PYTHONPATH}"
# repo ngspice when /usr/local is absent (tests/common/base.py convention)
if [ -z "${NGSPICE_BIN:-}" ]; then
  if [ -x /usr/local/ngspice-45.2/bin/ngspice ]; then
    NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice
  else
    NGSPICE_BIN="$ROOT/tools/ngspice-45.2/bin/ngspice"
  fi
fi
export NGSPICE_BIN

# ---- single-cell worker: _cell <stem> <section> <script> <tech> <ompN> ----
if [ "${1:-}" = "_cell" ]; then
  stem="$2"; section="$3"; vscript="$4"; tech="$5"; ompn="$6"
  rdir="$OUT/$stem"
  tag="${section}_${vscript}_${tech}_omp${ompn}"
  log="$rdir/logs/${tag}.log"
  mkdir -p "$rdir/logs"
  case "$section" in
    zeroshot) tmo="$ZS_TIMEOUT" ;;
    dev|ac)   tmo="$DEV_TIMEOUT" ;;
    *)        tmo="$GATE_TIMEOUT" ;;
  esac
  # isolated per (stem, section, ompN): concurrent cells of the same circuit
  # at different techs share this dir — per-tech work dirs inside must not
  # collide (post-checked by the resolver grep + audited).
  cres="$rdir/complex_omp${ompn}"
  start=$SECONDS
  env CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS="$ompn" MKL_NUM_THREADS="$ompn" \
      PYCIRCUITSIM_TORCH_THREADS="$ompn" \
      PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS="${stem}_nmos" \
      PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS="${stem}_pmos" \
      PYCIRCUITSIM_COMPLEX_RESULTS="$cres" \
      timeout --kill-after=60 "$tmo" \
        "$PYBIN" -u "$ROOT/tests/${vscript}.py" --tech "$tech" \
        > "$log" 2>&1
  rc=$?
  wall=$((SECONDS - start))
  if [ $rc -eq 0 ]; then verdict="PASS"
  elif [ $rc -eq 124 ]; then verdict="TIMEOUT"
  else verdict="FAIL"; fi
  # resolver post-check: BOTH device stems must have resolved with
  # scope=universal (dev/ac suites parse many netlists; one line each is
  # enough). A miss means the cell ran on the WRONG checkpoint.
  for dev in nmos pmos; do
    if ! grep -q "NN-resolver.*${stem}_${dev}.*scope=universal" "$log"; then
      verdict="RESOLVER-MISS"
    fi
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$section" "$vscript" "$tech" "$ompn" "$verdict" "$rc" "$wall" \
    >> "$rdir/SUMMARY.tsv"
  echo "[uni-eval] $stem $tag -> $verdict (rc=$rc, ${wall}s)"
  # audit B3 — PASS and FAIL are both scientific verdicts (exit 0). TIMEOUT
  # (rc 124) and RESOLVER-MISS (the cell ran on the WRONG checkpoint, so it is
  # campaign-invalid) are NOT verdicts, nor is rc >= 126 (cannot-exec / signal):
  # exit 3 so the dispatcher fails loudly. Never 255 — xargs aborts on 255.
  case "$verdict" in
    TIMEOUT|RESOLVER-MISS) exit 3 ;;
  esac
  if [ "$rc" -ge 126 ]; then exit 3; fi
  exit 0
fi

# ---- dispatcher ----
read -r -a stems_arr <<< "${STEMS:-}"
if [ ${#stems_arr[@]} -eq 0 ]; then
  read -r -a recipes <<< "${RECIPES:-clean csob corroft crit30u}"
  stems_arr=()
  for r in "${recipes[@]}"; do stems_arr+=("u716_dn_${r}_${SIZE}"); done
fi

mkdir -p "$OUT"
cells=()
ran_stems=()
for stem in "${stems_arr[@]}"; do
  ok=1
  for dev in nmos pmos; do
    c="$CKPT/${stem}_${dev}_best.pt"
    if [ ! -f "$c" ]; then echo "[uni-eval] SKIP $stem: missing ${c##*/}"; ok=0
    elif [ ! -f "$c.complete" ]; then echo "[uni-eval] SKIP $stem: ${c##*/} has NO .complete marker (killed run?)"; ok=0
    elif [ ! -f "$CKPT/${stem}_${dev}_norm.npz" ]; then echo "[uni-eval] SKIP $stem: missing ${stem}_${dev}_norm.npz"; ok=0
    fi
  done
  [ $ok -eq 1 ] || continue
  # audit B3 — only stems that actually dispatch may enter the completeness
  # count below; a SKIPped stem's rows are stale leftovers from an earlier
  # dispatch (the section prune happens after this guard) and counting them
  # inflates `recorded` enough to mask a real shortfall.
  ran_stems+=("$stem")
  rdir="$OUT/$stem"; mkdir -p "$rdir"
  # Drop only the rows for the sections this dispatch re-runs; keep every
  # other section's rows. Truncating the whole file here meant a follow-up
  # `SECTIONS=omp` dispatch silently erased the `gates` verdicts it was
  # supposed to be compared against (same clobber class as the one fixed in
  # gate_matrix_iso.sh). Cells still append.
  if [ -f "$rdir/SUMMARY.tsv" ]; then
    keep="$rdir/.SUMMARY.keep"; : > "$keep"
    while IFS= read -r line; do
      sec="${line%%	*}"
      drop=0
      for s in $SECTIONS; do [ "$sec" = "$s" ] && drop=1; done
      [ $drop -eq 1 ] || printf '%s\n' "$line" >> "$keep"
    done < "$rdir/SUMMARY.tsv"
    mv "$keep" "$rdir/SUMMARY.tsv"
  else
    : > "$rdir/SUMMARY.tsv"
  fi
  for section in $SECTIONS; do
    case "$section" in
      gates)
        for circ in ring_osc opamp sram_snm switchcap; do
          for tech in $TECHS; do
            cells+=("$stem gates verify_complex_${circ} $tech 1"); done; done ;;
      zeroshot)
        for circ in ring_osc opamp sram_snm switchcap; do
          cells+=("$stem zeroshot verify_complex_${circ} TSMC5 1"); done ;;
      dev)
        cells+=("$stem dev verify_nn_dc_tran $DEV_TECHS 1") ;;
      ac)
        for tech in $TECHS; do
          cells+=("$stem ac verify_nn_ac $tech 1"); done ;;
      omp)
        for ompn in 2 4; do
          for circ in ring_osc opamp; do
            for tech in $TECHS; do
              cells+=("$stem omp verify_complex_${circ} $tech $ompn"); done; done; done ;;
      *) echo "[uni-eval] UNKNOWN section $section"; exit 1 ;;
    esac
  done
done

if [ ${#cells[@]} -eq 0 ]; then echo "[uni-eval] nothing to run"; exit 1; fi
echo "[uni-eval] launching ${#cells[@]} cells, $NPAR concurrent (sections: $SECTIONS; stems: ${stems_arr[*]})"
# audit B3 — capture xargs BEFORE the trailing echo / print loop, which would
# otherwise overwrite $? with 0. Verdict is decided after the tables print.
printf '%s\n' "${cells[@]}" | \
  SIZE="$SIZE" OUT="$OUT" PYBIN="$PYBIN" NGSPICE_BIN="$NGSPICE_BIN" \
  GATE_TIMEOUT="$GATE_TIMEOUT" ZS_TIMEOUT="$ZS_TIMEOUT" DEV_TIMEOUT="$DEV_TIMEOUT" \
  xargs -P "$NPAR" -L1 "$SELF" _cell
xrc=$?
echo "[uni-eval] ALL CELLS DONE"
for stem in "${stems_arr[@]}"; do
  [ -f "$OUT/$stem/SUMMARY.tsv" ] || continue
  echo "--- $stem ---"
  sort "$OUT/$stem/SUMMARY.tsv" | column -t
done

# audit B3/B5f — a cell that appended no row leaves SUMMARY.tsv short, and every
# downstream ratio then reads over a silently smaller denominator. Count the rows
# belonging to the sections this dispatch re-ran (they were pruned before the run,
# so what is there now is exactly what the cells wrote) against the cells launched.
recorded=0
for stem in "${ran_stems[@]}"; do
  f="$OUT/$stem/SUMMARY.tsv"
  [ -f "$f" ] || continue
  while IFS= read -r line; do
    sec="${line%%	*}"
    for s in $SECTIONS; do
      if [ "$sec" = "$s" ]; then recorded=$((recorded + 1)); break; fi
    done
  done < "$f"
done
if [ "$xrc" -ne 0 ] || [ "$recorded" -lt "${#cells[@]}" ]; then
  echo "[uni-eval] INFRASTRUCTURE FAILURE: $recorded/${#cells[@]} cells recorded a verdict" \
       "(xargs rc=$xrc) — TIMEOUT / RESOLVER-MISS / missing rows are NOT results" >&2
  exit 1
fi
