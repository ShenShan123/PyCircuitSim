#!/usr/bin/env python
"""S14c — force_ic seed sweep. Runs the (fast) force_ic_probe for one tech on
whatever checkpoint the resolver picks (set BSIMAR_CHECKPOINT_DIR to an isolated
{tech}_dn_medium install). Prints a parseable RESULT line with rail_ok + the
released q/qb so a wrapping sweep can spot a seed that fully rails.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/data2/shenshan/PyCircuitSim")
import tests.verify_complex_sram_snm as sram  # noqa: E402

tech = sys.argv[1]
stem = sys.argv[2] if len(sys.argv) > 2 else "?"
bt = sram.BENCH[tech]
with tempfile.TemporaryDirectory() as wd:
    res = sram.force_ic_probe(bt, Path(wd))
n_pass = int(res.get("state1", False)) + int(res.get("state0", False))
print(f"RESULT {stem} {tech} pass={n_pass}/2 "
      f"state1={res.get('state1')} state0={res.get('state0')}")
