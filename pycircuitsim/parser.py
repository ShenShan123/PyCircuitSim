"""
HSPICE-like netlist parser for PyCircuitSim.

This module provides the Parser class which reads .sp netlist files and
constructs Circuit objects with appropriate components. The parser supports
HSPICE-like syntax for components and analysis commands.

Supported components:
- Resistors: R<name> <n1> <n2> <value>
- Capacitors: C<name> <n1> <n2> <value>
- Voltage sources: V<name> <n+> <n-> <value>
- Current sources: I<name> <n+> <n-> <value>
- MOSFETs: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>

Supported analysis:
- DC sweep: .dc <source> <start> <stop> <step>
- Transient: .tran <tstep> <tstop>
- AC analysis: .ac <sweep_type> <num_points> <fstart> <fstop>

Supported directives:
- Initial conditions: .ic V(<node>)=<value> ...  (hierarchical nodes like
  V(X1.n1)=... are accepted; .ic cards inside .subckt bodies are rewritten
  to the instance-prefixed nodes at expansion time)
- Model definitions: .model <name> <type> <params>
- Include files: .include <filename>
- Subcircuits: .subckt <name> <port1> ... [param=default ...] / .ends
  with hierarchical X-instance lines: X<name> <node1> ... <subckt> [param=val ...]

Subcircuit semantics (flattening expansion, ngspice-style):
- Instances are expanded recursively at parse time; internal nodes become
  "<inst_path>.<node>" (e.g. X1.n1, X1.X2.n1); ground ("0"/"GND") stays global.
- Flattened device names become "<type_char>.<inst_path>.<orig_name>"
  (e.g. M.X1.Mp1) so first-character dispatch still works.
- Parameters: .subckt-line defaults, overridden per instance; referenced in
  the body as bare names or {expr} / 'expr' arithmetic (unit suffixes OK).
- .model/.include inside a body are hoisted to global scope; nested .subckt
  definitions are registered globally (ngspice-flat scoping).
- Node/port/param matching is case-sensitive for nodes, case-insensitive
  for subckt names and parameter names.

Value suffixes supported:
- k/K: kilo (1e3)
- u/U: micro (1e-6)
- n/N: nano (1e-9)
- p/P: pico (1e-12)
"""
from typing import Any, Dict, Optional, Tuple
import os
import re
import sys
from pathlib import Path

# Make the `bsimar` package importable for LEVEL=73 / LEVEL=74 resolution below.
_BSIMAR_PARENT = Path(__file__).resolve().parent.parent / "external_compact_models"
if str(_BSIMAR_PARENT) not in sys.path:
    sys.path.insert(0, str(_BSIMAR_PARENT))

from pycircuitsim.circuit import Circuit
from pycircuitsim.models import (
    Resistor,
    Capacitor,
    VoltageSource,
    CurrentSource,
)
from pycircuitsim.config import BSIMCMG_OSDI_PATH, GENERIC_MODELCARD_DIR, ASAP7_MODELCARD_DIR


def _resolve_nn_checkpoint(
    *,
    level: int,
    device_key: str,
    tech_key: str,
    vt_key: str,
    explicit_path: Optional[str],
    netlist_name: str,
) -> Tuple[str, int]:
    """Resolve checkpoint path and tech code for LEVEL=73/74/75.

    Cascade: explicit ``MODEL_PATH`` > v4 universal > per-tech > bare.
    For LEVEL=74 (BSIMAR) the universal cascade prefers ``_best.phys.pt``
    over ``_best.pt`` only when the matching ``_norm.npz`` was saved with
    the median-aggregated phys-score (post-2026-05-03 fix); pre-fix
    files default to ``_best.pt`` to avoid the AR-rollout id-column
    blowup.
    """
    from bsimar.config import (
        CHECKPOINT_DIR, tech_variant_to_code, UNKNOWN_CODE_ID,
        LOCAL_VARIANT_CODES, local_variant_code,
    )

    # ── Env-var override (V5 Phase C): force a specific exp prefix ──────
    # Per-(level, polarity) env vars, then per-polarity, then global:
    #   PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS / _DN_PMOS  — LEVEL=73, polarity
    #   PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS / _TF_PMOS  — LEVEL=74, polarity
    #   PYCIRCUITSIM_NN_CHECKPOINT_PFN_NMOS / _PFN_PMOS — LEVEL=75, polarity
    #   PYCIRCUITSIM_NN_CHECKPOINT_NMOS / _PMOS        — all levels, polarity
    #   PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE            — all levels, both polarities
    # The override prefix is treated as the trainer save_prefix; when it
    # ends in "_nmos"/"_pmos" the polarity is honoured.
    import os
    level_tag = {73: "DN", 74: "TF", 75: "PFN"}[level]
    level_polarity_env = (
        f"PYCIRCUITSIM_NN_CHECKPOINT_{level_tag}_{device_key.upper()}")
    per_polarity_env = (
        f"PYCIRCUITSIM_NN_CHECKPOINT_{device_key.upper()}")
    _override = (os.environ.get(level_polarity_env)
                 or os.environ.get(per_polarity_env)
                 or os.environ.get("PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE"))
    # The env var takes priority over explicit_path (MODEL_PATH= in netlist)
    # so V5 Phase C verify can swap checkpoints without re-generating netlists.
    if _override:
        ovr = _override.strip()
        if ovr.endswith(f"_{device_key}"):
            base = ovr
        elif ovr.endswith("_nmos") or ovr.endswith("_pmos"):
            base = None
        else:
            base = f"{ovr}_{device_key}"

        if base is not None:
            if level in (73, 75):
                ovr_path = CHECKPOINT_DIR / f"{base}_best.pt"
                if ovr_path.exists():
                    explicit_path = str(ovr_path)
                else:
                    # a pinned-but-absent checkpoint must fail LOUDLY: falling
                    # through to the resolver cascade would silently evaluate
                    # a different model (typically production `large`) under
                    # the pinned recipe's name
                    raise FileNotFoundError(
                        f"NN checkpoint override '{ovr}' -> {ovr_path} "
                        "does not exist; refusing silent fallback to the "
                        "resolver cascade")
            else:  # level == 74
                phys_path = CHECKPOINT_DIR / f"{base}_best.phys.pt"
                plain_path = CHECKPOINT_DIR / f"{base}_best.pt"
                norm_path = CHECKPOINT_DIR / f"{base}_norm.npz"
                phys_trustworthy = False
                if phys_path.exists() and norm_path.exists():
                    try:
                        from bsimar.data.normalize import BSIMARNormStats
                        _ns = BSIMARNormStats.load(str(norm_path))
                        phys_trustworthy = (
                            getattr(_ns, "phys_best_metric", "legacy_mean")
                            == "median")
                    except Exception:
                        phys_trustworthy = False
                if phys_trustworthy:
                    explicit_path = str(phys_path)
                elif plain_path.exists():
                    explicit_path = str(plain_path)
                elif phys_path.exists():
                    explicit_path = str(phys_path)
                else:
                    raise FileNotFoundError(
                        f"NN checkpoint override '{ovr}' -> {plain_path} "
                        f"(or {phys_path.name}) does not exist; refusing "
                        "silent fallback to the resolver cascade")

    if explicit_path is not None:
        path = explicit_path
    elif level == 73:
        # Cascade: per-tech dedicated > refactor universal presets >
        # v4-re universal > legacy v4 universal > per-tech-bare > bare.
        #
        # Per-tech dedicated slots (`tsmc{5,7,12,16}_dn_{size}_{dev}_best.pt`)
        # preempt the universal cascade when the netlist's tech matches.
        # The trained model uses a SHRUNK local-vocab embedding; the
        # tech_code remap below keys off the file's `tsmc{X}_dn_` prefix.
        #
        # V6.6.0: `large` is the uniform production capacity tier (the best
        # uniform-recipe gate pass-rate, 13/16, vs medium's 10/16; capacity
        # peaks at large then over-fits at xl). So the bare resolver default
        # prefers `large` first, then falls back medium > small > xl. The
        # benchmark/gate harness still pins any specific tier explicitly via
        # PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}, so this only sets the
        # no-override production default.
        per_tech_preempt: list = []
        if tech_key in LOCAL_VARIANT_CODES:
            per_tech_preempt = [
                CHECKPOINT_DIR / f"{tech_key}_dn_large_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_dn_medium_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_dn_small_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_dn_xl_{device_key}_best.pt",
            ]
        candidates = per_tech_preempt + [
            CHECKPOINT_DIR / f"refac_dn_medium_{device_key}_best.pt",
            CHECKPOINT_DIR / f"refac_dn_small_{device_key}_best.pt",
            CHECKPOINT_DIR / f"refac_dn_large_{device_key}_best.pt",
            CHECKPOINT_DIR / f"v4_re_dn_universal_{device_key}_best.pt",
            CHECKPOINT_DIR / f"v4_dn_universal_{device_key}_best.pt",
            CHECKPOINT_DIR / f"{tech_key}_{device_key}_best.pt",
            CHECKPOINT_DIR / f"{device_key}_best.pt",
        ]
        path = next((str(p) for p in candidates if p.exists()),
                    str(candidates[-1]))
    elif level == 75:
        # V6.9: per-tech dedicated TabPFN slots
        # (`tsmc{5,7,12,16}_pfn_{size}_{dev}_best.pt`) preempt when the
        # netlist's tech matches — the exact mirror of the LEVEL=73
        # cascade (same large-first production ordering; local-vocab
        # embedding keyed off the `tsmc{X}_pfn_` stem below).
        per_tech_preempt = []
        if tech_key in LOCAL_VARIANT_CODES:
            per_tech_preempt = [
                CHECKPOINT_DIR / f"{tech_key}_pfn_large_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_pfn_medium_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_pfn_small_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_pfn_xl_{device_key}_best.pt",
            ]
        candidates = per_tech_preempt + [
            CHECKPOINT_DIR / f"refac_pfn_medium_{device_key}_best.pt",
            CHECKPOINT_DIR / f"refac_pfn_small_{device_key}_best.pt",
            CHECKPOINT_DIR / f"refac_pfn_large_{device_key}_best.pt",
            CHECKPOINT_DIR / f"pfn_{tech_key}_{device_key}_best.pt",
            CHECKPOINT_DIR / f"pfn_{device_key}_best.pt",
        ]
        path = next((str(p) for p in candidates if p.exists()),
                    str(candidates[-1]))
    else:  # level == 74
        # Cascade: per-tech dedicated > v4-re universal > legacy v4 universal
        # > per-tech-bare > bare.
        #
        # V6.8: per-tech dedicated Transformer slots
        # (`tsmc{5,7,12,16}_tf_{size}_{dev}_best.pt`) preempt the universal
        # cascade when the netlist's tech matches — the exact mirror of the
        # LEVEL=73 preempt. Same large-first production ordering; the trained
        # model uses a SHRUNK local-vocab embedding (Rule 16) keyed off the
        # `tsmc{X}_tf_` stem below.
        #
        # For each universal candidate prefer `_best.phys.pt` only when the
        # norm.npz declares `phys_best_metric == "median"` (post-2026-05-03 fix).
        def _select(prefix: str) -> Optional[str]:
            phys_path = CHECKPOINT_DIR / f"{prefix}_{device_key}_best.phys.pt"
            plain_path = CHECKPOINT_DIR / f"{prefix}_{device_key}_best.pt"
            norm_path = CHECKPOINT_DIR / f"{prefix}_{device_key}_norm.npz"
            phys_trustworthy = False
            if phys_path.exists() and norm_path.exists():
                try:
                    from bsimar.data.normalize import BSIMARNormStats
                    _ns = BSIMARNormStats.load(str(norm_path))
                    phys_trustworthy = (
                        getattr(_ns, "phys_best_metric", "legacy_mean")
                        == "median")
                except Exception:
                    phys_trustworthy = False
            if phys_trustworthy:
                return str(phys_path)
            if plain_path.exists():
                return str(plain_path)
            if phys_path.exists():
                # Last-resort fallback when no plain best.pt is on disk.
                return str(phys_path)
            return None

        per_tech_path = CHECKPOINT_DIR / f"ar_{tech_key}_{device_key}_best.pt"
        bare_path = CHECKPOINT_DIR / f"ar_{device_key}_best.pt"

        per_tech_preempt = []
        if tech_key in LOCAL_VARIANT_CODES:
            per_tech_preempt = [
                CHECKPOINT_DIR / f"{tech_key}_tf_large_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_tf_medium_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_tf_small_{device_key}_best.pt",
                CHECKPOINT_DIR / f"{tech_key}_tf_xl_{device_key}_best.pt",
            ]

        # Cascade: per-tech preempt > refactor presets > v4-re > legacy v4
        # > per-tech-bare > bare.
        path = next((str(p) for p in per_tech_preempt if p.exists()), None)
        if path is None:
            path = (
                _select("refac_tf_medium")
                or _select("refac_tf_small")
                or _select("refac_tf_large")
                or _select("v4_re_universal")
                or _select("v4_universal")
            )
        if path is None:
            if per_tech_path.exists():
                path = str(per_tech_path)
            else:
                path = str(bare_path)

    # Determine vocab scope from the resolved checkpoint name.
    # `tsmc{X}_dn_*` (DirectNet) / `tsmc{X}_tf_*` (BSIMAR Transformer) /
    # `tsmc{X}_pfn_*` (TabPFN) => local per-tech vocab; else universal.
    chk_name = Path(path).name
    scope = "universal"
    for s in LOCAL_VARIANT_CODES:
        if (chk_name.startswith(f"{s}_dn_")
                or chk_name.startswith(f"{s}_tf_")
                or chk_name.startswith(f"{s}_pfn_")):
            scope = s
            break
    tech_code = local_variant_code(scope, tech_key, vt_key)

    # Fail loud: log every NN checkpoint resolution so the .lis /
    # stdout makes the universal-vs-per-tech choice and tech_code visible.
    print(f"[NN-resolver] L{level} {netlist_name} TECH={tech_key} VT={vt_key} "
          f"-> {chk_name} (scope={scope}, tech_code={tech_code})")

    if scope == "universal" and tech_code == UNKNOWN_CODE_ID:
        import warnings
        warnings.warn(
            f"MOSFET {netlist_name}: TECH={tech_key} VT={vt_key} maps to "
            f"UNKNOWN tech code ({UNKNOWN_CODE_ID}). "
            "Predictions may be less accurate."
        )
    return path, tech_code


class Parser:
    """
    HSPICE-like netlist parser.

    The Parser reads .sp files line by line and constructs a Circuit object
    containing all components and analysis commands from the netlist.

    Attributes:
        circuit: Circuit object containing all parsed components
        analysis_type: Type of analysis ('dc', 'tran', or None)
        analysis_params: Dictionary of analysis parameters
        models: Dictionary of model definitions (name -> type + params)
    """

    # Unit suffix multipliers
    UNIT_SUFFIXES = {
        't': 1e12,  # tera
        'T': 1e12,
        'g': 1e9,   # giga
        'G': 1e9,
        'm': 1e6,   # mega (milli is less common in circuits)
        'M': 1e6,
        'k': 1e3,   # kilo
        'K': 1e3,
        'u': 1e-6,  # micro
        'U': 1e-6,
        'n': 1e-9,  # nano
        'N': 1e-9,
        'p': 1e-12, # pico
        'P': 1e-12,
        'f': 1e-15, # femto
        'F': 1e-15,
    }

    # ASAP7 modelcard filenames (from ASAP7 PDK)
    ASAP7_MODELCARD_FILES = [
        "7nm_TT_160803.pm",  # Typical-Typical corner
        "7nm_FF.pm",          # Fast-Fast corner
        "7nm_SS.pm",           # Slow-Slow corner
    ]

    def __init__(
        self,
        osdi_path: Optional[str] = None,
        modelcard_base_dir: Optional[str] = None,
        modelcard_path: Optional[str] = None,
        model_name_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize an empty parser.

        Args:
            osdi_path: Path to BSIM-CMG OSDI binary (defaults to config value)
            modelcard_base_dir: Base directory for modelcard files (defaults to generic modelcards)
            modelcard_path: Explicit path to a modelcard file (bypasses auto-discovery).
                Useful for non-ASAP7 technologies with separate NMOS/PMOS files
                that have been merged into a single file.
            model_name_map: Mapping from device type ("NMOS"/"PMOS") to the model
                name inside the modelcard file (e.g. {"NMOS": "nch_svt_mac",
                "PMOS": "pch_lvt_mac"}). If None, uses ASAP7 auto-detection
                or falls back to no remapping.
        """
        self.circuit = Circuit()
        self.analysis_type: Optional[str] = None
        self.analysis_params: Dict[str, float] = {}
        self.models: Dict[str, Dict[str, Any]] = {}  # Model definitions
        # Subcircuit definitions: UPPER name -> {name, ports, params, body}
        self.subckts: Dict[str, Dict[str, Any]] = {}
        self._osdi_path = osdi_path or BSIMCMG_OSDI_PATH
        self._modelcard_base_dir = modelcard_base_dir or GENERIC_MODELCARD_DIR
        self._explicit_modelcard = modelcard_path
        self._model_name_map = model_name_map

        # Allow override of ASAP7 modelcard directory via environment variable
        self._asap7_modelcard_dir = os.environ.get("ASAP7_MODELCARD_DIR", ASAP7_MODELCARD_DIR)

    def parse_file(self, filename: str) -> None:
        """
        Parse a netlist file and populate the circuit.

        Reads the specified .sp file line by line, parsing each line to
        extract components and analysis commands.

        Args:
            filename: Path to the .sp netlist file

        Raises:
            FileNotFoundError: If the netlist file doesn't exist
            ValueError: If the netlist contains invalid syntax
        """
        # Store current file for .include resolution
        self._current_file = str(Path(filename).resolve())

        with open(filename, 'r') as f:
            lines = f.readlines()

        # First pass: handle line continuations and collect models/includes
        processed_lines = []
        continued_line = ""
        in_model = False  # Track if we're in a .model definition

        for raw_line in lines:
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('*'):
                continue

            # Handle line continuations (lines starting with '+')
            if line.startswith('+'):
                continuation = line[1:].strip()
                continuation = continuation.replace(' = ', '=').replace('= ', '=')
                continuation = re.sub(r'\s*=\s*', '=', continuation)
                continued_line += " " + continuation
                continue

            # If we have a continued line, add it to processed lines
            if continued_line:
                processed_lines.append(continued_line)
                continued_line = ""
                in_model = False

            # Check if this is a new .model or .include or analysis line
            if line.lower().startswith('.model') or line.lower().startswith('.include') or \
               line.lower().startswith('.dc') or line.lower().startswith('.tran') or \
               line.lower().startswith('.ac') or line.lower().startswith('.ic') or line.lower().startswith('.end'):
                line = line.replace(' = ', '=').replace('= ', '=')
                line = ' '.join(line.split())

                # For .model lines, start accumulating continuations
                if line.lower().startswith('.model'):
                    continued_line = line
                    in_model = True
                else:
                    processed_lines.append(line)
            else:
                # Regular line (component definition)
                line = line.replace(' = ', '=').replace('= ', '=')
                line = ' '.join(line.split())
                processed_lines.append(line)

        # Process any remaining continued line
        if continued_line:
            processed_lines.append(continued_line)

        # Subckt pass: extract .subckt/.ends definition blocks (registers
        # them in self.subckts; hoists .model/.include out of bodies).
        # Runs before the model/include pre-pass so hoisted cards are seen.
        component_lines = self._collect_subckt_defs(processed_lines)

        # Pre-pass: collect all .model and .include directives first
        # This ensures models are available before components that reference
        # them. Included files register their .subckt definitions on this
        # parser too, so top-level X instances can use library subckts.
        for line in component_lines:
            if line.lower().startswith('.model'):
                self._parse_model(line)
            elif line.lower().startswith('.include'):
                # Includes may add more models, so process them
                self.parse_line(line)

        # Expansion pass: flatten X subcircuit instances recursively.
        flat_lines = []
        for line in component_lines:
            if line[0].upper() == 'X':
                flat_lines.extend(self._expand_instance(
                    line, node_map={}, path="", params={}, depth=0))
            else:
                flat_lines.append(line)

        # Second pass: parse all remaining lines (components, analysis, etc.)
        for line in flat_lines:
            # Skip .model and .include (already processed)
            if not line.lower().startswith(('.model', '.include')):
                self.parse_line(line)

    def parse_line(self, line: str) -> None:
        """
        Parse a single line from the netlist.

        Dispatches to the appropriate parsing method based on the first
        character of the line. Ignores comments (lines starting with '*')
        and empty lines.

        Args:
            line: A single line from the netlist file

        Raises:
            ValueError: If the line contains invalid syntax
        """
        # Skip empty lines and comments
        if not line or line.startswith('*'):
            return

        # Skip .end directive
        if line.lower().startswith('.end'):
            return

        # Dispatch based on first character
        first_char = line[0].upper()

        if first_char == 'R':
            self._parse_resistor(line)
        elif first_char == 'C':
            self._parse_capacitor(line)
        elif first_char == 'V':
            self._parse_voltage_source(line)
        elif first_char == 'I':
            self._parse_current_source(line)
        elif first_char == 'M':
            self._parse_mosfet(line)
        elif first_char == 'X':
            # X instances are expanded (flattened) inside parse_file; one
            # reaching parse_line means the subckt machinery was bypassed.
            raise ValueError(
                f"Unexpanded subcircuit instance '{line}': X lines are only "
                "supported through parse_file (.subckt expansion)")
        elif line.lower().startswith('.dc'):
            self._parse_dc(line)
        elif line.lower().startswith('.tran'):
            self._parse_tran(line)
        elif line.lower().startswith('.ac'):
            self._parse_ac(line)
        elif line.lower().startswith('.ic'):
            self._parse_ic(line)
        elif line.lower().startswith('.model'):
            self._parse_model(line)
        elif line.lower().startswith('.include'):
            self._parse_include(line)
        # Ignore other directives (.option, .measure, etc.)

    def _parse_value(self, value_str: str) -> float:
        """
        Convert a value string with optional unit suffix to a float.

        Args:
            value_str: Value string (e.g., "1k", "10u", "3.3", "100p")

        Returns:
            Floating point value

        Examples:
            >>> parser._parse_value("1k")
            1000.0
            >>> parser._parse_value("10n")
            1e-08
            >>> parser._parse_value("3.3")
            3.3
        """
        # Check if the last character is a unit suffix
        if len(value_str) > 1 and value_str[-1] in self.UNIT_SUFFIXES:
            multiplier = self.UNIT_SUFFIXES[value_str[-1]]
            return float(value_str[:-1]) * multiplier

        # No suffix, just convert to float
        return float(value_str)

    def _parse_resistor(self, line: str) -> None:
        """
        Parse a resistor line: R<name> <n1> <n2> <value>.

        Args:
            line: Resistor definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid resistor syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]
        value = self._parse_value(parts[3])

        resistor = Resistor(name, nodes, value)
        self.circuit.add_component(resistor)

    def _parse_capacitor(self, line: str) -> None:
        """
        Parse a capacitor line: C<name> <n1> <n2> <value>.

        Args:
            line: Capacitor definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid capacitor syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]
        value = self._parse_value(parts[3])

        capacitor = Capacitor(name, nodes, value)
        self.circuit.add_component(capacitor)

    def _parse_voltage_source(self, line: str) -> None:
        """
        Parse a voltage source line: V<name> <n+> <n-> <value> or V<name> <n+> <n-> PULSE <params>.

        Supports:
        - DC voltage source: V1 1 0 3.3
        - PULSE source: V1 1 0 PULSE 0 3.3 1n 0.1n 0.1n 5n 10n
        - AC voltage source: V1 1 0 DC=1.0 AC=0.1 0 (DC bias, AC magnitude, AC phase in degrees)

        Args:
            line: Voltage source definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid voltage source syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]

        # Check if it's a PULSE source
        if len(parts) >= 4 and parts[3].upper() == 'PULSE':
            # PULSE source: V1 n+ n- PULSE V1 V2 TD TR TF PW PER
            if len(parts) < 11:
                raise ValueError(f"PULSE source requires 8 parameters: {line}")

            from pycircuitsim.models.passive import PulseVoltageSource

            v1 = self._parse_value(parts[4])
            v2 = self._parse_value(parts[5])
            td = self._parse_value(parts[6])
            tr = self._parse_value(parts[7])
            tf = self._parse_value(parts[8])
            pw = self._parse_value(parts[9])
            per = self._parse_value(parts[10])

            pulse_source = PulseVoltageSource(name, nodes, v1, v2, td, tr, tf, pw, per)
            self.circuit.add_component(pulse_source)
        else:
            # Check if it's an AC specification: DC=x AC=y phase
            dc_value = None
            ac_magnitude = 0.0
            ac_phase = 0.0

            # Look for DC=, AC= keywords
            for i, part in enumerate(parts[3:], start=3):
                if part.upper().startswith('DC='):
                    dc_value = self._parse_value(part[3:])
                elif part.upper().startswith('AC='):
                    ac_magnitude = self._parse_value(part[3:])
                    # Check if phase follows AC magnitude
                    if i + 1 < len(parts) and not parts[i + 1].upper().startswith(('DC=', 'AC=')):
                        try:
                            ac_phase = float(parts[i + 1])
                        except ValueError:
                            pass  # Not a phase value, skip
                elif dc_value is None and not part.upper().startswith(('DC=', 'AC=')):
                    # No DC= keyword, treat first value as DC value
                    dc_value = self._parse_value(part)

            # Default DC value to 0 if only AC specified
            if dc_value is None:
                dc_value = 0.0

            voltage_source = VoltageSource(name, nodes, dc_value, ac_magnitude=ac_magnitude, ac_phase=ac_phase)
            self.circuit.add_component(voltage_source)

    def _parse_current_source(self, line: str) -> None:
        """
        Parse a current source line.

        Supports:
        - DC current source: I1 n+ n- 1m
        - AC current source: I1 n+ n- DC=0 AC=1 0 (DC bias, AC magnitude, AC phase in degrees)

        The AC keyword form mirrors the voltage-source parser so that `.ac`
        analysis can be driven by a current stimulus (e.g. transimpedance).

        Args:
            line: Current source definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid current source syntax: {line}")

        name = parts[0]
        nodes = [parts[1], parts[2]]

        # Check for the AC specification form: DC=x AC=y phase
        dc_value = None
        ac_magnitude = 0.0
        ac_phase = 0.0

        for i, part in enumerate(parts[3:], start=3):
            if part.upper().startswith('DC='):
                dc_value = self._parse_value(part[3:])
            elif part.upper().startswith('AC='):
                ac_magnitude = self._parse_value(part[3:])
                # Optional phase value immediately following AC=magnitude
                if i + 1 < len(parts) and not parts[i + 1].upper().startswith(('DC=', 'AC=')):
                    try:
                        ac_phase = float(parts[i + 1])
                    except ValueError:
                        pass  # Not a phase value, skip
            elif dc_value is None and not part.upper().startswith(('DC=', 'AC=')):
                # No DC= keyword: treat the first bare value as the DC current
                dc_value = self._parse_value(part)

        # Default DC value to 0 if only AC specified
        if dc_value is None:
            dc_value = 0.0

        current_source = CurrentSource(name, nodes, dc_value,
                                       ac_magnitude=ac_magnitude, ac_phase=ac_phase)
        self.circuit.add_component(current_source)

    def _parse_mosfet(self, line: str) -> None:
        """
        Parse a MOSFET line: M<name> <d> <g> <s> <b> <model> L=<l> W=<w> [NFIN=<nf> ...].

        Supports Level 72/BSIM-CMG (L, NFIN, TFIN, HFIN, FPITCH) and Level 73/NN.

        Args:
            line: MOSFET definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        # MOSFET line format: M<name> <d> <g> <s> <b> <model> L=<l> W=<w>
        # BSIM-CMG format: M<name> <d> <g> <s> <b> <model> L=<l> NFIN=<nf> [TFIN=<tf>] ...
        parts = line.split()

        if len(parts) < 7:
            raise ValueError(f"Invalid MOSFET syntax: {line}")

        name = parts[0]
        nodes = parts[1:5]  # [drain, gate, source, bulk]
        model = parts[5].upper()  # NMOS or PMOS

        # Extract geometric parameters (BSIM-CMG: L, NFIN, TFIN, HFIN, FPITCH; NN: L, NFIN)
        L = None
        W = None
        NFIN = None
        TFIN = None
        HFIN = None
        FPITCH = None

        for part in parts[6:]:
            if part.startswith('L='):
                L = self._parse_value(part[2:])
            elif part.startswith('W='):
                W = self._parse_value(part[2:])
            elif part.startswith('NFIN='):
                NFIN = float(part[5:])  # Number of fins (integer or float)
            elif part.startswith('TFIN='):
                TFIN = self._parse_value(part[5:])
            elif part.startswith('HFIN='):
                HFIN = self._parse_value(part[5:])
            elif part.startswith('FPITCH='):
                FPITCH = self._parse_value(part[7:])

        # L is always required
        if L is None:
            raise ValueError(f"MOSFET missing L parameter: {line}")

        # Check if model name references a .model definition
        model_name = parts[5]  # Keep case for model lookup

        # Look up model in .model definitions
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

        model_def = self.models[model_name]
        model_type = model_def['type']
        model_params = model_def['params']

        # Check model level (72 = BSIM-CMG, 73 = NN)
        level = model_params.get('LEVEL', 72)

        if level == 72:
            # BSIM-CMG compact model
            if NFIN is None:
                raise ValueError(f"BSIM-CMG (LEVEL=72) MOSFET missing NFIN parameter: {line}")

            # Import BSIM-CMG models
            try:
                from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG
            except ImportError as e:
                raise ImportError(
                    f"Failed to import BSIM-CMG models: {e}. "
                    "Ensure PyCMG is built and OSDI binary exists."
                )

            # Resolve modelcard path
            # Priority: explicit path > ASAP7 auto-discovery > generic naming
            modelcard_path = None

            if self._explicit_modelcard:
                # Use explicitly provided modelcard path (e.g., merged TSMC file)
                modelcard_path = Path(self._explicit_modelcard)
                if not modelcard_path.exists():
                    raise FileNotFoundError(
                        f"Explicit modelcard not found: {modelcard_path}"
                    )
            else:
                # Try ASAP7 naming first
                for asap7_file in self.ASAP7_MODELCARD_FILES:
                    asap7_path = Path(self._asap7_modelcard_dir) / asap7_file
                    if asap7_path.exists():
                        modelcard_path = asap7_path
                        break

                # Fall back to generic naming
                if modelcard_path is None:
                    generic_path = Path(self._modelcard_base_dir) / f"modelcard.{model_type.lower()}.1"
                    if generic_path.exists():
                        modelcard_path = generic_path

            if modelcard_path is None:
                raise FileNotFoundError(
                    f"BSIM-CMG modelcard not found. Tried:\n"
                    f"  - ASAP7 directory: {self._asap7_modelcard_dir} (files: {self.ASAP7_MODELCARD_FILES})\n"
                    f"  - Generic directory: {self._modelcard_base_dir} (modelcard.{model_type.lower()}.1)\n"
                    f"Model referenced: '{model_name}' (type={model_type}, level={level})\n"
                    f"Hint: Set ASAP7_MODELCARD_DIR environment variable if using ASAP7 PDK,\n"
                    f"or pass modelcard_path= to Parser() for non-ASAP7 technologies."
                )

            # Determine the model_card_name (name inside the modelcard file).
            # Priority: explicit map > ASAP7 auto-detection > no remapping
            model_card_name = None
            if self._model_name_map:
                model_card_name = self._model_name_map.get(model_type.upper())
            elif str(modelcard_path).startswith(str(self._asap7_modelcard_dir)):
                # ASAP7 modelcards define models like "nmos_rvt", "pmos_rvt" etc.,
                # which differ from the user's netlist model name (e.g., "nmos1").
                # Default to RVT (regular Vth) variant if using ASAP7.
                if model_type.upper() == 'NMOS':
                    model_card_name = "nmos_rvt"
                elif model_type.upper() == 'PMOS':
                    model_card_name = "pmos_rvt"

            if model_type.upper() == 'NMOS':
                mosfet = NMOS_CMG(
                    name=name,
                    nodes=nodes,
                    osdi_path=self._osdi_path,
                    modelcard_path=str(modelcard_path),
                    model_name=model_name,
                    L=L,
                    NFIN=NFIN,
                    TFIN=TFIN,
                    HFIN=HFIN,
                    FPITCH=FPITCH,
                    model_card_name=model_card_name,
                )
            elif model_type.upper() == 'PMOS':
                mosfet = PMOS_CMG(
                    name=name,
                    nodes=nodes,
                    osdi_path=self._osdi_path,
                    modelcard_path=str(modelcard_path),
                    model_name=model_name,
                    L=L,
                    NFIN=NFIN,
                    TFIN=TFIN,
                    HFIN=HFIN,
                    FPITCH=FPITCH,
                    model_card_name=model_card_name,
                )
            else:
                raise ValueError(f"Unknown MOSFET model type: {model_type}")

        elif level in (73, 74, 75):
            # NN compact model: LEVEL=73 (DirectNet), LEVEL=74 (BSIMAR
            # Transformer), or LEVEL=75 (TabPFN).
            #
            # V6.8 harness hook: PYCIRCUITSIM_NN_FORCE_LEVEL={73,74,75}
            # retargets every NN model card at parse time, so the ENTIRE
            # gate/sweep/AC harness (whose netlists carry LEVEL=73 tokens)
            # can run the BSIMAR Transformer or TabPFN without rendering
            # parallel decks. Loud per-card log below; NGSPICE reference
            # decks (LEVEL=72) are untouched.
            import os as _os
            _force = _os.environ.get("PYCIRCUITSIM_NN_FORCE_LEVEL")
            if _force and int(_force) in (73, 74, 75) and int(_force) != level:
                print(f"[NN-resolver] FORCE LEVEL {level}->{_force} "
                      f"(PYCIRCUITSIM_NN_FORCE_LEVEL) for {name}")
                level = int(_force)
            label = {73: "NN", 74: "BSIM-AR", 75: "TabPFN"}[level]
            if NFIN is None:
                raise ValueError(
                    f"{label} (LEVEL={level}) MOSFET missing NFIN parameter: {line}")

            try:
                if level == 73:
                    from pycircuitsim.models.mosfet_directnet import (
                        NMOS_NN as _NMOS, PMOS_NN as _PMOS,
                    )
                elif level == 75:
                    from pycircuitsim.models.mosfet_pfn import (
                        NMOS_PFN as _NMOS, PMOS_PFN as _PMOS,
                    )
                else:
                    from pycircuitsim.models.mosfet_bsimar import (
                        NMOS_BSIMAR as _NMOS, PMOS_BSIMAR as _PMOS,
                    )
            except ImportError:
                raise ImportError(
                    f"{label} MOSFET model requires PyTorch. "
                    "Install: pip install torch"
                )

            tech_key = (model_params.get('TECH') or "asap7").lower()
            vt_key = (model_params.get('VT') or "svt").lower()
            device_key = model_type.lower()

            nn_model_path, nn_tech_code = _resolve_nn_checkpoint(
                level=level,
                device_key=device_key,
                tech_key=tech_key,
                vt_key=vt_key,
                explicit_path=model_params.get('MODEL_PATH'),
                netlist_name=name,
            )

            nn_kwargs = dict(
                name=name, nodes=nodes, model_path=nn_model_path,
                L=L, NFIN=NFIN, tech_code=nn_tech_code,
            )

            if model_type.upper() == 'NMOS':
                mosfet = _NMOS(**nn_kwargs)
            elif model_type.upper() == 'PMOS':
                mosfet = _PMOS(**nn_kwargs)
            else:
                raise ValueError(f"Unknown MOSFET model type: {model_type}")

        else:
            raise ValueError(
                f"Unsupported MOSFET LEVEL={level}. "
                f"Supported levels: LEVEL=72 (BSIM-CMG), LEVEL=73 (NN), "
                f"LEVEL=74 (BSIM-AR), LEVEL=75 (TabPFN)"
            )

        self.circuit.add_component(mosfet)

    def _parse_dc(self, line: str) -> None:
        """
        Parse a DC sweep analysis line: .dc <source> <start> <stop> <step>.

        Args:
            line: DC sweep analysis line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid .dc syntax: {line}")

        self.analysis_type = "dc"
        self.analysis_params = {
            "source": parts[1],
            "start": self._parse_value(parts[2]),
            "stop": self._parse_value(parts[3]),
            "step": self._parse_value(parts[4]),
        }

    def _parse_tran(self, line: str) -> None:
        """
        Parse a transient analysis line: .tran <tstep> <tstop>.

        Args:
            line: Transient analysis line

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid .tran syntax: {line}")

        # `uic` (use-initial-conditions, NGSPICE-style): start the transient
        # from the `.ic` state instead of the unconstrained DC operating point.
        # Without it, a high-impedance node (e.g. a switched-cap hold node) seeds
        # at its off-device leakage equilibrium rather than the `.ic` value —
        # see run_transient's uic pinning. Any token after tstep/tstop.
        uic = any(p.lower() == "uic" for p in parts[3:])

        self.analysis_type = "tran"
        self.analysis_params = {
            "tstep": self._parse_value(parts[1]),
            "tstop": self._parse_value(parts[2]),
            "uic": uic,
        }

    def _parse_ac(self, line: str) -> None:
        """
        Parse an AC analysis line: .ac <sweep_type> <num_points> <fstart> <fstop>.

        Sweep types:
        - dec: decade sweep (logarithmic, num_points per decade)
        - lin: linear sweep (num_points total between fstart and fstop)
        - oct: octave sweep (logarithmic, num_points per octave)

        Args:
            line: AC analysis line (e.g., ".ac dec 10 1k 10e6")

        Raises:
            ValueError: If the line has invalid syntax
        """
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid .ac syntax: {line}")

        sweep_type = parts[1].lower()
        if sweep_type not in ['dec', 'lin', 'oct']:
            raise ValueError(f"Invalid AC sweep type: {sweep_type}. Must be 'dec', 'lin', or 'oct'")

        self.analysis_type = "ac"
        self.analysis_params = {
            "sweep_type": sweep_type,
            "num_points": int(parts[2]),
            "fstart": self._parse_value(parts[3]),
            "fstop": self._parse_value(parts[4]),
        }

    def _parse_ic(self, line: str) -> None:
        """
        Parse an initial condition line: .ic V(<node>)=<value> V(<node>)=<value> ...

        Sets initial voltages for specified nodes, which is useful for
        defining the initial state of bistable circuits like SRAM cells.

        Args:
            line: Initial condition line (e.g., ".ic V(2)=3.3 V(3)=0")

        Raises:
            ValueError: If the line has invalid syntax

        Examples:
            .ic V(2)=3.3 V(3)=0
            .ic V(node1)=1.8 V(node2)=0.5
        """
        # Remove ".ic" prefix
        ic_spec = line[3:].strip()

        # Pattern to match V(node)=value or V(node)=value, with multiple assignments
        # Supports: V(2)=3.3, V(2)=3.3 V(3)=0, V(node1)=1.8 V(node2)=0.5,
        # and hierarchical (subckt-expanded) nodes: V(X1.n1)=0.5
        pattern = r'V\(\s*([^)]+?)\s*\)\s*=\s*([0-9.eE+-]+[kKuUnNpP]?)'

        matches = re.findall(pattern, ic_spec, flags=re.IGNORECASE)

        if not matches:
            raise ValueError(f"Invalid .ic syntax: {line}")

        for node_str, value_str in matches:
            node = node_str.strip()
            value = self._parse_value(value_str)
            self.circuit.initial_conditions[node] = value

    def _parse_model(self, line: str) -> None:
        """
        Parse a .model line: .model <name> NMOS/PMOS <params>

        Args:
            line: Model definition line

        Raises:
            ValueError: If the line has invalid syntax
        """
        # Remove ".model" prefix and get parts
        model_spec = line[6:].strip()

        # Remove parentheses if present (HSPICE style: .model name TYPE (params))
        model_spec = model_spec.replace('(', ' ').replace(')', ' ')
        parts = model_spec.split()

        if len(parts) < 2:
            raise ValueError(f"Invalid .model syntax: {line}")

        model_name = parts[0]
        model_type = parts[1].upper()

        # Parse parameters (supports key=value format)
        # String-valued params (TECH, VT, MODEL_PATH) are stored as-is;
        # numeric params are converted via _parse_value.
        _STRING_PARAMS = {"TECH", "VT", "MODEL_PATH"}
        params = {}
        for part in parts[2:]:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().upper()
                value = value.strip()
                if key and value:
                    if key in _STRING_PARAMS:
                        params[key] = value
                    else:
                        try:
                            params[key] = self._parse_value(value)
                        except ValueError:
                            # Store as string for unknown params
                            params[key] = value

        # Store model definition
        self.models[model_name] = {
            'type': model_type,
            'params': params
        }

    def _parse_include(self, line: str) -> None:
        """
        Parse an .include directive: .include <filename>

        Args:
            line: Include directive line

        Raises:
            ValueError: If the line has invalid syntax
            FileNotFoundError: If the included file doesn't exist
        """
        # Remove ".include" prefix
        include_spec = line[8:].strip()
        included_file = include_spec.strip('"\'')  # Remove quotes

        # Resolve path relative to current file
        current_file = getattr(self, '_current_file', None)
        if current_file:
            current_dir = Path(current_file).parent
            included_path = current_dir / included_file
        else:
            included_path = Path(included_file)

        if not included_path.exists():
            raise FileNotFoundError(f"Included file not found: {included_path}")

        # Parse the included file using parse_file (handles line continuations).
        # Save/restore _current_file so sibling includes after this one still
        # resolve relative to the including file, not the included one.
        try:
            self.parse_file(str(included_path))
        finally:
            if current_file is not None:
                self._current_file = current_file

    # ── Subcircuit (.subckt / .ends / X instance) machinery ──────────────
    #
    # Hierarchy is supported by flattening at parse time (ngspice-style):
    # definitions are collected into self.subckts, then every X line is
    # recursively expanded into ordinary component/.ic lines with
    # instance-prefixed internal node names ("X1.n1", "X1.X2.n1") and
    # device names ("M.X1.Mp1" — first char preserved for dispatch).

    #: Maximum instantiation depth (guards recursive subckt cycles).
    MAX_SUBCKT_DEPTH = 64

    #: Node-token count per component type letter (nodes come right after
    #: the name token; everything later is values/params/model refs).
    _NODE_COUNT = {'R': 2, 'C': 2, 'V': 2, 'I': 2, 'M': 4}

    def _collect_subckt_defs(self, lines: list) -> list:
        """Extract .subckt/.ends blocks from a processed-line stream.

        Registers each definition in ``self.subckts`` (keyed by upper-cased
        name) and returns the remaining top-level lines. Nested .subckt
        definitions are allowed and registered globally (ngspice-flat
        scoping). ``.model``/``.include`` cards found inside a body are
        hoisted to the top-level stream so the model pre-pass sees them.

        Args:
            lines: Continuation-folded, whitespace-normalized netlist lines

        Returns:
            Top-level lines with all .subckt blocks removed

        Raises:
            ValueError: On malformed/unbalanced .subckt/.ends structure
        """
        top_lines: list = []
        def_stack: list = []  # innermost definition last

        for line in lines:
            low = line.lower()
            if low.startswith('.subckt'):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid .subckt syntax: {line}")
                ports = []
                params: Dict[str, Any] = {}
                for tok in parts[2:]:
                    if '=' in tok:
                        key, val = tok.split('=', 1)
                        params[key.strip().upper()] = \
                            self._resolve_param_value(val, {})
                    else:
                        ports.append(tok)
                def_stack.append({
                    'name': parts[1],
                    'ports': ports,
                    'params': params,
                    'body': [],
                })
            elif low.startswith('.ends'):
                if not def_stack:
                    raise ValueError(f"'.ends' without matching .subckt: {line}")
                defn = def_stack.pop()
                parts = line.split()
                if len(parts) > 1 and parts[1].upper() != defn['name'].upper():
                    raise ValueError(
                        f"'.ends {parts[1]}' does not close "
                        f".subckt {defn['name']}")
                self.subckts[defn['name'].upper()] = defn
            elif def_stack:
                if low.startswith('.model') or low.startswith('.include'):
                    # Models/includes are global in this parser — hoist.
                    top_lines.append(line)
                else:
                    def_stack[-1]['body'].append(line)
            else:
                top_lines.append(line)

        if def_stack:
            raise ValueError(
                f".subckt '{def_stack[-1]['name']}' has no matching .ends")
        return top_lines

    def _expand_instance(self, line: str, node_map: Dict[str, str],
                         path: str, params: Dict[str, Any],
                         depth: int) -> list:
        """Recursively flatten one X subcircuit-instance line.

        Args:
            line: ``X<name> <node1> ... <subckt_name> [param=val ...]``
            node_map: Enclosing scope's local-node -> flat-node mapping
            path: Enclosing instance path ("" at top level)
            params: Enclosing scope's parameter environment
            depth: Current instantiation depth (cycle guard)

        Returns:
            List of flattened component / .ic lines

        Raises:
            ValueError: Unknown subckt, port-count mismatch, or recursion
        """
        if depth > self.MAX_SUBCKT_DEPTH:
            raise ValueError(
                f"Subcircuit nesting deeper than {self.MAX_SUBCKT_DEPTH} "
                f"levels at '{line}' — recursive .subckt definitions?")

        parts = line.split()
        inst_name = parts[0]
        positional = [p for p in parts[1:] if '=' not in p]
        if len(positional) < 1:
            raise ValueError(f"Invalid subcircuit instance syntax: {line}")
        subckt_name = positional[-1]
        raw_nodes = positional[:-1]

        defn = self.subckts.get(subckt_name.upper())
        if defn is None:
            raise ValueError(
                f"Subcircuit '{subckt_name}' not found for instance "
                f"'{inst_name}'. Defined subckts: "
                f"{[d['name'] for d in self.subckts.values()]}")
        if len(raw_nodes) != len(defn['ports']):
            raise ValueError(
                f"Instance '{inst_name}' connects {len(raw_nodes)} nodes "
                f"but .subckt {defn['name']} declares "
                f"{len(defn['ports'])} ports ({defn['ports']})")

        # Map connection nodes through the ENCLOSING scope first.
        ext_nodes = [self._map_node(n, node_map, path) for n in raw_nodes]

        # Parameter environment: defaults overridden per instance; override
        # values are resolved in the enclosing scope (may reference parent
        # params or {expr} arithmetic).
        child_params = dict(defn['params'])
        for tok in parts[1:]:
            if '=' in tok:
                key, val = tok.split('=', 1)
                child_params[key.strip().upper()] = \
                    self._resolve_param_value(val, params)

        inst_path = f"{path}.{inst_name}" if path else inst_name
        child_map = dict(zip(defn['ports'], ext_nodes))

        flat: list = []
        for body_line in defn['body']:
            flat.extend(self._expand_body_line(
                body_line, child_map, inst_path, child_params, depth + 1))
        return flat

    def _expand_body_line(self, line: str, node_map: Dict[str, str],
                          path: str, params: Dict[str, Any],
                          depth: int) -> list:
        """Flatten a single .subckt body line within an instance context.

        Component lines get instance-prefixed names and mapped nodes plus
        parameter substitution; nested X lines recurse; ``.ic`` cards are
        rewritten to the flat node names. Any other directive is an error
        (``.model``/``.include`` were hoisted at collection time).
        """
        if line.startswith('.'):
            if line.lower().startswith('.ic'):
                # Rewrite every V(node)=value assignment: node mapped to its
                # flat name, value resolved in the parameter environment
                # (bare param names / {expr} allowed inside bodies).
                def _rewrite_ic(m: "re.Match") -> str:
                    node = self._map_node(m.group(2).strip(), node_map, path)
                    value = self._format_param(
                        self._resolve_param_value(m.group(4), params))
                    return f"{m.group(1)}{node}{m.group(3)}{value}"
                return [re.sub(
                    r'([Vv]\(\s*)([^)]+?)(\s*\)\s*=\s*)(\{[^}]*\}|\S+)',
                    _rewrite_ic, line)]
            raise ValueError(
                f"Directive '{line.split()[0]}' is not allowed inside a "
                f".subckt body (only components, X instances and .ic)")

        first_char = line[0].upper()
        if first_char == 'X':
            return self._expand_instance(line, node_map, path, params, depth)

        if first_char not in self._NODE_COUNT:
            raise ValueError(
                f"Unsupported component '{line}' inside .subckt body "
                f"(supported: R, C, V, I, M, X)")

        parts = line.split()
        n_nodes = self._NODE_COUNT[first_char]
        if len(parts) < 1 + n_nodes + 1:
            raise ValueError(f"Invalid component line in .subckt body: {line}")

        new_name = f"{parts[0][0]}.{path}.{parts[0]}"
        mapped_nodes = [self._map_node(tok, node_map, path)
                        for tok in parts[1:1 + n_nodes]]
        if first_char == 'M':
            # Token 5 is the .model reference — never substituted.
            tail = [parts[5]] + [self._subst_token(tok, params)
                                 for tok in parts[6:]]
        else:
            tail = [self._subst_token(tok, params)
                    for tok in parts[1 + n_nodes:]]
        return [' '.join([new_name] + mapped_nodes + tail)]

    def _map_node(self, node: str, node_map: Dict[str, str],
                  path: str) -> str:
        """Map a local node token to its flat (global) name.

        Ports map to the connecting nodes, ground ("0"/"GND") is global,
        anything else is an internal node prefixed with the instance path.
        """
        if node in node_map:
            return node_map[node]
        if node in ("0", "GND") or not path:
            return node
        return f"{path}.{node}"

    def _subst_token(self, token: str, params: Dict[str, Any]) -> str:
        """Apply parameter substitution to one value/param token.

        ``KEY=val`` tokens have their value part resolved (bare parameter
        names, ``{expr}``/``'expr'`` arithmetic, plain literals); bare
        tokens are replaced only when they name a parameter or expression.
        """
        if '=' in token:
            key, val = token.split('=', 1)
            return f"{key}={self._format_param(self._resolve_param_value(val, params))}"
        resolved = self._resolve_param_value(token, params)
        if isinstance(resolved, float) or token.upper() in params \
                or token.startswith(('{', "'")):
            return self._format_param(resolved)
        return token

    def _resolve_param_value(self, raw: str, params: Dict[str, Any]):
        """Resolve a parameter value string in the given environment.

        Order: ``{expr}`` / ``'expr'`` arithmetic > bare parameter-name
        lookup (case-insensitive) > numeric literal with unit suffix >
        raw string passthrough.
        """
        raw = raw.strip()
        if (raw.startswith('{') and raw.endswith('}')) or \
                (raw.startswith("'") and raw.endswith("'") and len(raw) > 1):
            return self._eval_expr(raw[1:-1], params)
        if raw.upper() in params:
            return params[raw.upper()]
        try:
            return self._parse_value(raw)
        except ValueError:
            return raw

    def _eval_expr(self, expr: str, params: Dict[str, Any]) -> float:
        """Safely evaluate a {…} parameter expression.

        Supports + - * / ( ), numeric literals with unit suffixes, and
        parameter names. No builtins/attribute access can be reached: all
        identifier tokens are substituted (or rejected) before eval and
        the residue is charset-checked.
        """
        def _num(m: "re.Match") -> str:
            return repr(float(m.group(1)) * self.UNIT_SUFFIXES[m.group(2)])

        # Numeric literals with unit suffixes -> plain floats
        substituted = re.sub(
            r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)([tTgGmMkKuUnNpPfF])\b',
            _num, expr)

        def _ident(m: "re.Match") -> str:
            key = m.group(0).upper()
            if key in params:
                val = params[key]
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Parameter {key}={val!r} is not numeric in "
                        f"expression '{expr}'")
                return f"({float(val)!r})"
            raise ValueError(
                f"Unknown parameter '{m.group(0)}' in expression '{expr}'")

        substituted = re.sub(r'[A-Za-z_][A-Za-z_0-9]*', _ident, substituted)
        if not re.fullmatch(r"[-+*/(). 0-9eE]*", substituted):
            raise ValueError(f"Unsupported syntax in expression '{expr}'")
        try:
            return float(eval(substituted, {"__builtins__": {}}, {}))
        except ZeroDivisionError:
            raise ValueError(f"Division by zero in expression '{expr}'")

    @staticmethod
    def _format_param(val) -> str:
        """Format a resolved parameter back into a netlist token."""
        if isinstance(val, float):
            if val.is_integer() and abs(val) < 1e15:
                return str(int(val))
            return repr(val)
        return str(val)
