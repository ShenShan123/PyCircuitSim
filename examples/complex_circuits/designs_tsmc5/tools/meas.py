"""Run an ngspice deck against the BSIM-CMG OSDI binary and read its .meas output.

ngspice must execute ``osdi <binary>`` before the netlist is parsed, so every
deck here is sourced from a generated runner that does exactly that.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from pycmg_lib import OSDI_PATH

NGSPICE = os.environ.get("NGSPICE", "/usr/local/ngspice-45.2/bin/ngspice")

# ``dcgain = 6.68e+01`` / ``maxval = 8.0e-01 at= -3.7e+01`` / ``x = failed``
_MEAS_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*(failed|[-+0-9.eE]+)\s*(?:\S+=.*)?$"
)
_ERR_RE = re.compile(
    r"(doAnalyses:\s+(?:TRAN|AC|DC)|Transient op failed|"
    r"operating point could not be simulated successfully|Fatal:|"
    r"no such vector|Error on line)", re.IGNORECASE
)


class SimError(RuntimeError):
    """ngspice did not complete, or completed without producing measurements."""


def run_deck(deck: Path, control: str, work: Path, tag: str,
             timeout: float = 300.0) -> Dict[str, Optional[float]]:
    """Run *deck* under *control*, returning every ``.meas`` result by name.

    A measurement ngspice reports as ``failed`` maps to ``None`` -- that is a
    real outcome (a gain that never crosses 0 dB has no GBW), not an error.
    """
    work.mkdir(parents=True, exist_ok=True)
    # Runner and log go in a per-process directory.  Two passes over the same
    # design otherwise write the same run_<tag>.cir: one truncates the file
    # while the other's ngspice is reading it, and every deck fails with rc=1
    # and an empty log -- which looks exactly like a circuit that does not
    # simulate.  The log is also copied to the stable path callers expect.
    # Resolved so the paths survive ngspice's cwd (deck.parent): a caller
    # passing a *relative* work dir otherwise hands ngspice a log path it
    # cannot create from inside the design dir -- rc=1 with an empty log,
    # indistinguishable from a circuit that does not simulate.
    private = (work / f".run-{os.getpid()}").resolve()
    private.mkdir(parents=True, exist_ok=True)
    runner = private / f"run_{tag}.cir"
    log = private / f"{tag}.log"
    log.unlink(missing_ok=True)

    runner.write_text(
        f"* tsmc16 runner ({tag})\n"
        f".control\n"
        f"set noaskquit\n"
        # ngspice parallelises device evaluation with OpenMP and sets the thread
        # count itself -- OMP_NUM_THREADS and `.option numthreads` are both
        # ignored, and each run takes 300-600 % CPU.  Only the `set` variable is
        # honoured.  Measured on one AC deck: 1.70 s at 319 % CPU unpinned
        # versus 0.19 s at 60 % pinned, because the threads were fighting each
        # other rather than the work.  This single line is worth ~9x.
        f"set num_threads=1\n"
        f"osdi {OSDI_PATH}\n"
        f"source {deck.resolve()}\n"
        f"{control}\n"
        f".endc\n"
        f".end\n"
    )

    # ngspice parallelises internally; left alone it takes ~6 cores per run and
    # N designs sized concurrently thrash.  One thread per run keeps the
    # scheduling ours.
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        proc = subprocess.run(
            [NGSPICE, "-b", "-o", str(log), str(runner)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(deck.parent), env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SimError(f"{tag}: ngspice timed out after {timeout}s") from exc

    text = log.read_text() if log.exists() else ""
    if text:
        try:
            (work / f"{tag}.log").write_text(text)
        except OSError:
            pass
    if proc.returncode != 0:
        raise SimError(
            f"{tag}: ngspice rc={proc.returncode}\n{text[-800:]}"
        )
    fatal = [line.strip() for line in text.splitlines() if _ERR_RE.search(line)]
    if fatal:
        raise SimError(f"{tag}: ngspice analysis failed: {fatal[-1]}")

    out: Dict[str, Optional[float]] = {}
    for line in text.splitlines():
        m = _MEAS_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2)
        if val == "failed":
            out.setdefault(key, None)
        else:
            try:
                out[key] = float(val)
            except ValueError:
                continue
    return out


def fatal_errors(log: Path) -> List[str]:
    """Return simulator-level error lines found in a run log."""
    if not log.exists():
        return ["missing log"]
    return [ln.strip() for ln in log.read_text().splitlines()
            if _ERR_RE.search(ln)]
