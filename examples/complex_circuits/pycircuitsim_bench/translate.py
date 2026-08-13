"""AnalogGym deck -> PyCircuitSim netlist + analysis plan (owner: I2).

This is the only module that knows AnalogGym / ngspice deck syntax.  It never
imports ``pycircuitsim`` and never runs a simulator, so it is unit-testable
without the OSDI binary.

The translation is a sequence of rules, each of which was a measured blocker
against PyCircuitSim's parser (``pycircuitsim/parser.py``).  The ones that are
easy to get wrong silently:

* **Values are always emitted in plain e-notation.**  ``Parser._parse_value``
  inspects only the LAST character and maps ``'m'``/``'M'`` to 1e6 (MEGA); it
  has no ``meg``.  So ``100n`` must not survive into the emitted deck, and
  ``10m`` would silently become 1e7.  Every literal this module writes goes
  through :func:`_fmt`, which asserts the result ends in a digit.
* **Node tokens are lower-cased here, not in core.**  The decks declare
  ``.subckt op gnda vdda ...`` in lower case and reference ``VDDA`` in the
  body; ``Parser._map_node`` is a case-sensitive dict lookup, so an
  unfolded port becomes a dead internal net (``Xop1.VDDA``) and the matrix
  goes singular.  Folding inside the parser instead would permute
  ``Circuit.get_nodes()``'s sort order and move existing gate numbers, so it
  lives here.
* **Geometry is scraped from the model card, never from the instance line.**
  ngspice's OSDI binding takes no instance parameters, so L/NFIN/TFIN are
  baked into per-geometry ``.model <name> bsimcmg (...)`` blocks.  PyCMG's own
  ``parse_modelcard`` deliberately forces ``nfin -> 1.0``, so it must not be
  used to read them back.
* **A ``.subckt`` port named ``GND``** is rejected outright by PyCircuitSim
  (ground cannot be a port), so it is renamed to ``gnd_p`` throughout the
  body.  The instance already ties it to node 0, so the circuit is unchanged.

Analysis directives are NEVER emitted into the netlist: the injected ngspice
control line is parsed into :class:`AnalysisPlan` data that ``run_compare``
executes against the solvers directly.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import (
    AMP_TB_DC_NARROW,
    AMP_TB_DC_RECOVERY,
    CONTROLS,
    STABILITY_DECKS,
    AnalysisPlan,
    DeckOptions,
    MeasCard,
    TranslateError,
    TranslatedDeck,
)

# ── Constants ────────────────────────────────────────────────────────────

#: ngspice scale factors.  Deliberately NOT PyCircuitSim's table: here 'm' is
#: milli and 'meg' exists, which is what the source decks are written in.
_NG_SCALES: Dict[str, float] = {
    't': 1e12, 'g': 1e9, 'k': 1e3, 'm': 1e-3,
    'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15,
}

#: A number with an optional ngspice scale factor / trailing unit junk.
_NUM_RE = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)$')

#: Quoted expression, braced expression, or a bare token.  Used everywhere so
#: a quoted expression containing spaces can never be split in half.
_TOKEN_RE = re.compile(r"'[^']*'|\{[^}]*\}|\S+")

#: ``V(node)=value`` assignment as it appears on .nodeset / .ic cards.
_NODE_ASSIGN_RE = re.compile(
    r"[Vv]\(\s*([^)\s]+)\s*\)\s*=\s*('[^']*'|\{[^}]*\}|\S+)")

#: Node spellings that PyCircuitSim's ``Parser._canon_node`` collapses to "0".
_GROUND_ALIASES = frozenset({'0', 'gnd'})

#: Replacement for a ``.subckt`` port that names ground (translation rule 11).
_GND_PORT_ALIAS = 'gnd_p'

#: Component letters this translator may emit (rule 13's self-validation set).
_EMITTABLE = frozenset({'R', 'C', 'L', 'V', 'I', 'M', 'X', '.'})

#: ``.options`` keys we carry into :class:`DeckOptions`, with their types.
_OPTION_KEYS: Dict[str, str] = {
    'cshunt': 'float', 'rshunt': 'float', 'reltol': 'float',
    'vntol': 'float', 'abstol': 'float', 'gmin': 'float',
    'method': 'str', 'maxord': 'int',
}

#: ``(TECH, design)`` pairs whose amplifier transient control carries an
#: explicit tmax cap.  Verbatim from every tree's ``tools/build_amp.py``.
_TRAN_MAXSTEP: Dict[Tuple[str, str], str] = {
    ('TSMC5', 'qu_lec_pin_3'): '2n',
}

#: Deepest ``.include`` chain we will follow (the corpus uses exactly one).
_MAX_INCLUDE_DEPTH = 8

#: Parsed model-card geometry, keyed by resolved models-file path.  The file is
#: 444 KB and shared by all seven decks of a design, so scan it once.
_MODEL_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}


# ── Low-level text helpers ───────────────────────────────────────────────

def strip_comment(line: str) -> str:
    """Truncate a logical line at its first unquoted ``$`` or ``;`` tail.

    ngspice treats ``;`` as an end-of-line comment anywhere and ``$`` as one
    when it follows whitespace or starts the line.  The AnalogGym decks use
    ``  $ gmf_PMOS`` role annotations on nearly every device line; those
    currently survive into PyCircuitSim's device parsing by luck (they land
    past the last token it reads), so they are removed here explicitly.

    Args:
        line: One raw (already continuation-folded or not) netlist line

    Returns:
        The line with any comment tail removed and trailing space stripped
    """
    in_quote = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == ';':
            return line[:i].rstrip()
        if ch == '$' and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()


def _tokenize(line: str) -> List[str]:
    """Split a line into tokens, keeping ``'expr'`` / ``{expr}`` groups whole."""
    return _TOKEN_RE.findall(line)


def _ng_value(token: str) -> float:
    """Convert an ngspice numeric literal (with scale factor) to a float.

    ngspice's own scale factors, not PyCircuitSim's: ``meg`` is 1e6, a bare
    ``m`` is milli, and trailing unit junk ("1kohm") is ignored.  The scaling
    is done as ``mantissa * scale`` — the same two-step arithmetic ngspice's
    reader performs — so ``100n`` becomes 1.0000000000000001e-07, one ulp above
    ``1e-7``, and BIT-IDENTICAL to the value ngspice will use.  Prettier
    parsing here would put a 1-ulp wedge between the two simulators.

    Args:
        token: Numeric literal, e.g. "1T", "100n", "2.2e-05", "1meg", "10G"

    Returns:
        The value as a float

    Raises:
        TranslateError: If the token is not a number with an optional suffix
    """
    m = _NUM_RE.match(token.strip())
    if m is None:
        raise TranslateError(f"Not a numeric literal: {token!r}")
    mantissa, suffix = m.group(1), m.group(2).lower()
    if not suffix:
        return float(mantissa)
    if suffix.startswith('meg'):
        return float(mantissa) * 1e6
    if suffix.startswith('mil'):
        return float(mantissa) * 25.4e-6
    return float(mantissa) * _NG_SCALES.get(suffix[0], 1.0)


def _fmt(value: float) -> str:
    """Format a value as a plain e-notation literal PyCircuitSim can read.

    ``repr`` gives the shortest round-tripping form and never ends in a
    letter for a finite float, which is exactly the property that matters:
    ``Parser._parse_value`` would read a trailing 'm' as MEGA (1e6).

    Raises:
        TranslateError: If the value is not finite (the formatted token would
            end in a letter and be silently re-scaled).
    """
    text = repr(float(value))
    if not text[-1].isdigit():
        raise TranslateError(
            f"Refusing to emit non-finite / suffixed literal {text!r}: "
            f"Parser._parse_value would re-scale its last character")
    return text


def _fmt_count(value: float) -> str:
    """Format an integral quantity (NFIN, m) without a spurious ``.0``."""
    if float(value).is_integer():
        return str(int(round(value)))
    return _fmt(value)


# ── Expression evaluation (.param env, quoted device values) ─────────────

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CALLS = {'abs': abs, 'max': max, 'min': min}


def _eval_expr(expr: str, env: Mapping[str, float]) -> float:
    """Evaluate a deck arithmetic expression against a case-folded env.

    Supports ``+ - * / **``, unary minus, parentheses, ``abs``/``max``/``min``
    and identifiers resolved (case-insensitively) from ``env``.  Numeric
    literals may carry an ngspice scale factor, which Python's tokenizer would
    reject, so a bare literal is tried through :func:`_ng_value` first.

    ``Parser._eval_expr`` is deliberately not reused: it rejects identifiers
    absent from its own params dict and charset-checks the residue.
    """
    body = expr.strip()
    if body.startswith("'") and body.endswith("'"):
        body = body[1:-1].strip()
    elif body.startswith('{') and body.endswith('}'):
        body = body[1:-1].strip()
    if not body:
        raise TranslateError(f"Empty expression: {expr!r}")

    # Fast path: a plain literal, possibly with an ngspice scale factor.
    try:
        return _ng_value(body)
    except TranslateError:
        pass

    try:
        tree = ast.parse(body, mode='eval')
    except SyntaxError as exc:
        raise TranslateError(f"Cannot parse expression {expr!r}: {exc}") from exc

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise TranslateError(f"Non-numeric constant in {expr!r}")
        if isinstance(node, ast.Name):
            key = node.id.lower()
            if key not in env:
                raise TranslateError(f"Unknown identifier '{node.id}' in {expr!r}")
            return float(env[key])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return float(_BIN_OPS[type(node.op)](visit(node.left), visit(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return float(_UNARY_OPS[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id.lower()
            if fname in _CALLS and not node.keywords:
                return float(_CALLS[fname](*[visit(a) for a in node.args]))
        raise TranslateError(
            f"Unsupported syntax {type(node).__name__} in expression {expr!r}")

    return visit(tree)


def resolve_params(lines: Sequence[str]) -> Dict[str, float]:
    """Collect and resolve every ``.param`` / ``.PARAM`` assignment.

    Assignments may reference earlier ones (``.PARAM STEP_TIME =
    '10/GBW_ideal'``), so unresolved ones are retried until the environment
    stops growing.  Keys are lower-cased: the decks write ``supply_voltage``
    and ``GBW_ideal`` and reference both spellings.

    Args:
        lines: Continuation-folded, comment-stripped logical lines

    Returns:
        ``{lower_case_name: value}``

    Raises:
        TranslateError: If an assignment can never be resolved
    """
    pending: List[Tuple[str, str]] = []
    for line in lines:
        if not line.lower().startswith('.param'):
            continue
        # Normalise "name = value" to "name=value" outside quotes so the
        # assignment survives tokenisation as one token.
        body = _normalise_equals(line.split(None, 1)[1] if ' ' in line else '')
        for token in _tokenize(body):
            if '=' not in token:
                raise TranslateError(f"Malformed .param assignment in: {line}")
            name, value = token.split('=', 1)
            pending.append((name.strip().lower(), value.strip()))

    env: Dict[str, float] = {}
    while pending:
        progressed = False
        deferred: List[Tuple[str, str]] = []
        for name, raw in pending:
            try:
                env[name] = _eval_expr(raw, env)
                progressed = True
            except TranslateError:
                deferred.append((name, raw))
        if not progressed:
            unresolved = ', '.join(f"{n}={v}" for n, v in deferred)
            raise TranslateError(f"Unresolvable .param assignments: {unresolved}")
        pending = deferred
    return env


def _normalise_equals(text: str) -> str:
    """Collapse whitespace around ``=`` signs that are outside quotes."""
    out: List[str] = []
    in_quote = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            in_quote = not in_quote
            out.append(ch)
            i += 1
            continue
        if ch == '=' and not in_quote:
            while out and out[-1].isspace():
                out.pop()
            out.append('=')
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


# ── Deck loading (.include following, continuation folding) ──────────────

def _fold(text: str) -> List[str]:
    """Strip comments and fold ``+`` continuations into logical lines."""
    logical: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('*'):
            continue
        line = strip_comment(line)
        if not line:
            continue
        if line.startswith('+'):
            if not logical:
                raise TranslateError(f"Continuation line with no predecessor: {raw}")
            logical[-1] = f"{logical[-1]} {line[1:].strip()}"
            continue
        logical.append(' '.join(line.split()))
    return logical


def _load_logical_lines(path: Path, depth: int = 0
                        ) -> Tuple[List[str], Optional[Path]]:
    """Load a deck's logical lines, inlining ``.include``d netlists.

    ``*_models.spice`` is NOT inlined: it is 18 850 lines of ``.model
    bsimcmg`` cards that PyCircuitSim must never see.  Its path is returned so
    the caller can hand it to ``Parser(modelcard_path=...)``, which makes PyCMG
    resolve the real card by the netlist's own model token.

    Returns:
        ``(logical_lines, models_file_path_or_None)``
    """
    if depth > _MAX_INCLUDE_DEPTH:
        raise TranslateError(f"'.include' nested deeper than "
                             f"{_MAX_INCLUDE_DEPTH} levels at {path}")
    if not path.exists():
        raise TranslateError(f"Deck not found: {path}")

    out: List[str] = []
    models: Optional[Path] = None
    for line in _fold(path.read_text()):
        if not line.lower().startswith('.include'):
            out.append(line)
            continue
        target = _tokenize(line)[-1].strip('"\'')
        resolved = (path.parent / target).resolve()
        if resolved.name.endswith('_models.spice'):
            if models is not None and models != resolved:
                raise TranslateError(
                    f"{path.name} includes two different model libraries: "
                    f"{models} and {resolved}")
            models = resolved
            continue
        nested, nested_models = _load_logical_lines(resolved, depth + 1)
        out.extend(nested)
        if nested_models is not None:
            models = models or nested_models
    return out, models


# ── Model-card geometry (translation rule 4) ─────────────────────────────

def _scan_model_cards(models_path: Path) -> Dict[str, Dict[str, float]]:
    """Scrape ``l`` / ``nfin`` / ``tfin`` / ``devtype`` per ``.model`` block.

    The blocks are ``.model <name> bsimcmg (`` followed by ~2350 ``+ key =
    value`` continuation lines and a closing ``)``.  Only the four keys we
    need are kept; matching is exact (lower-cased) so the ~30 decoy keys
    containing "nfin" (``_nfin_cj``, ``she_c_nfin``, ``nfinfbd2_low``, ...)
    cannot be picked up by accident.
    """
    key = str(models_path)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    wanted = {'l', 'nfin', 'tfin', 'devtype'}
    cards: Dict[str, Dict[str, float]] = {}
    current: Optional[Dict[str, float]] = None
    for raw in models_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('*'):
            continue
        low = line.lower()
        if low.startswith('.model'):
            parts = line.split()
            if len(parts) < 2:
                raise TranslateError(f"Malformed .model card in {models_path}: {line}")
            current = {}
            cards[parts[1]] = current
            continue
        if current is None:
            continue
        if line.startswith(')'):
            current = None
            continue
        if not line.startswith('+'):
            continue
        assign = _normalise_equals(line[1:].strip())
        if '=' not in assign:
            continue
        name, value = assign.split('=', 1)
        name = name.strip().lower()
        if name in wanted:
            current[name] = _ng_value(value.strip())

    _MODEL_CACHE[key] = cards
    return cards


def _polarity(card_name: str, geom: Mapping[str, float]) -> str:
    """Return "NMOS"/"PMOS" from ``devtype``, cross-checked against the name.

    BSIM-CMG ``devtype`` is 1.0 for n-channel and 0.0 for p-channel; every
    card in the corpus is additionally named ``<n|p><vt>_l<L>_f<NFIN>``.  A
    disagreement means the models file and the netlist disagree about the
    device, which would silently invert a stage.
    """
    devtype = geom.get('devtype')
    if devtype is None:
        raise TranslateError(f"Model card '{card_name}' has no devtype")
    by_devtype = 'NMOS' if devtype >= 0.5 else 'PMOS'
    initial = card_name[:1].lower()
    if initial in ('n', 'p'):
        by_name = 'NMOS' if initial == 'n' else 'PMOS'
        if by_name != by_devtype:
            raise TranslateError(
                f"Model card '{card_name}' names a {by_name} device but "
                f"devtype={devtype} says {by_devtype}")
    return by_devtype


# ── Node-name handling (translation rules 6 and 11) ──────────────────────

class _Scope:
    """Node-token rewriting context for one ``.subckt`` body (or top level)."""

    def __init__(self, name: str, gnd_ports: Iterable[str]) -> None:
        self.name = name
        #: lower-cased port spellings that must become ``gnd_p``
        self.gnd_ports = frozenset(gnd_ports)


def _lower_node(token: str, scope: Optional[_Scope],
                warnings: List[str]) -> str:
    """Lower-case a node token and apply the ``GND``-port rename.

    Rule 6: PyCircuitSim's ``_map_node`` is a case-sensitive dict lookup, so
    every node token this module emits must be lower-case.
    Rule 11: inside a ``.subckt`` whose port list named ground, that port (and
    every body reference to it) becomes ``gnd_p``.
    """
    low = token.lower()
    if scope is not None and low in scope.gnd_ports:
        return _GND_PORT_ALIAS
    if low == 'gnd':
        # Not a renamed port: PyCircuitSim's _canon_node collapses this to "0"
        # while ngspice keeps it as an ordinary net.  Never seen in the corpus.
        warnings.append(
            f"node '{token}' will be collapsed to ground by PyCircuitSim "
            f"(_canon_node) but is an ordinary net for ngspice")
    return low


# ── Value resolution on emitted cards (translation rule 3) ──────────────

def _value_token(token: str, env: Mapping[str, float]) -> str:
    """Resolve one value token to a plain e-notation literal."""
    return _fmt(_eval_expr(token, env))


# ── Card translators ─────────────────────────────────────────────────────

def _translate_passive(tokens: Sequence[str], env: Mapping[str, float],
                       scope: Optional[_Scope], warnings: List[str]) -> str:
    """Translate an R / C / L card: lower the nodes, literalise the value."""
    if len(tokens) < 4:
        raise TranslateError(f"Malformed passive card: {' '.join(tokens)}")
    name = tokens[0]
    nodes = [_lower_node(t, scope, warnings) for t in tokens[1:3]]
    value = _value_token(tokens[3], env)
    if len(tokens) > 4:
        warnings.append(f"ignored trailing tokens on {name}: {tokens[4:]}")
    return f"{name} {nodes[0]} {nodes[1]} {value}"


def _translate_source(tokens: Sequence[str], env: Mapping[str, float],
                      scope: Optional[_Scope], warnings: List[str]) -> str:
    """Translate a V / I card into the exact spellings PyCircuitSim parses.

    Output forms (rule 7):

    * ``<name> <n+> <n-> DC=<v> [AC=<m> [<phase>]]``
    * ``<name> <n+> <n-> PULSE <i1> <i2> <td> <tr> <tf> <pw> <per>``

    The parenthesised ``pulse(...)`` spelling is normalised to the
    space-separated one for BOTH source types; PyCircuitSim's parser only
    accepts the space form.
    """
    if len(tokens) < 4:
        raise TranslateError(f"Malformed source card: {' '.join(tokens)}")
    name = tokens[0]
    nodes = [_lower_node(t, scope, warnings) for t in tokens[1:3]]
    rest = list(tokens[3:])

    if rest[0].lower().startswith('pulse'):
        args = _pulse_args(rest)
        if len(args) != 7:
            raise TranslateError(
                f"PULSE needs 7 parameters, got {len(args)}: {' '.join(tokens)}")
        values = ' '.join(_value_token(a, env) for a in args)
        return f"{name} {nodes[0]} {nodes[1]} PULSE {values}"

    dc_value: Optional[float] = None
    ac_mag: Optional[float] = None
    ac_phase: Optional[float] = None
    i = 0
    while i < len(rest):
        token = rest[i]
        low = token.lower()
        if low == 'dc':
            dc_value = _eval_expr(rest[i + 1], env)
            i += 2
        elif low.startswith('dc='):
            dc_value = _eval_expr(token.split('=', 1)[1], env)
            i += 1
        elif low == 'ac' or low.startswith('ac='):
            if low == 'ac':
                ac_mag = _eval_expr(rest[i + 1], env)
                i += 2
            else:
                ac_mag = _eval_expr(token.split('=', 1)[1], env)
                i += 1
            if i < len(rest) and '=' not in rest[i] \
                    and rest[i].lower() not in ('dc', 'ac'):
                ac_phase = _eval_expr(rest[i], env)
                i += 1
        elif dc_value is None:
            dc_value = _eval_expr(token, env)
            i += 1
        else:
            warnings.append(f"ignored token {token!r} on source {name}")
            i += 1

    parts = [f"DC={_fmt(dc_value if dc_value is not None else 0.0)}"]
    if ac_mag is not None:
        parts.append(f"AC={_fmt(ac_mag)}")
        if ac_phase is not None:
            parts.append(_fmt(ac_phase))
    return f"{name} {nodes[0]} {nodes[1]} {' '.join(parts)}"


def _pulse_args(rest: Sequence[str]) -> List[str]:
    """Extract the seven PULSE arguments from either spelling."""
    joined = ' '.join(rest)
    match = re.match(r'(?i)pulse\s*\((.*)\)\s*$', joined)
    if match is not None:                       # pulse(a b td tr tf pw per)
        return _tokenize(match.group(1))
    if rest[0].lower() != 'pulse':              # e.g. "pulse(0.1" without ')'
        raise TranslateError(f"Malformed PULSE specification: {joined}")
    return list(rest[1:])                       # PULSE a b td tr tf pw per


def _translate_mosfet(tokens: Sequence[str], geoms: Mapping[str, Dict[str, float]],
                      scope: Optional[_Scope], used_models: Dict[str, str],
                      multipliers: Dict[str, float],
                      warnings: List[str]) -> str:
    """Translate an ``N``-prefix OSDI device card into a PyCircuitSim M card.

    The leading ``N`` becomes ``M`` (first character only), the four terminals
    are lower-cased, the model token keeps its exact case (``self.models`` is a
    case-sensitive dict) and L/NFIN/TFIN scraped from that model's card are
    appended together with the deck's own ``m=``.
    """
    if len(tokens) < 6:
        raise TranslateError(f"Malformed device card: {' '.join(tokens)}")
    name = 'M' + tokens[0][1:]
    nodes = [_lower_node(t, scope, warnings) for t in tokens[1:5]]
    model = tokens[5]

    geom = geoms.get(model)
    if geom is None:
        matches = [k for k in geoms if k.lower() == model.lower()]
        if len(matches) == 1:
            warnings.append(f"model token '{model}' matched card "
                            f"'{matches[0]}' case-insensitively")
            model, geom = matches[0], geoms[matches[0]]
        else:
            raise TranslateError(
                f"Model '{model}' has no .model block in the models library "
                f"(device {tokens[0]})")
    for key in ('l', 'nfin', 'tfin'):
        if key not in geom:
            raise TranslateError(
                f"Model card '{model}' does not bake in '{key}'; geometry "
                f"cannot be recovered (device {tokens[0]})")

    mult = 1.0
    for token in tokens[6:]:
        if token.lower().startswith('m='):
            mult = _ng_value(token.split('=', 1)[1])
        else:
            warnings.append(f"ignored instance parameter {token!r} on {tokens[0]}")
    if mult <= 0:
        raise TranslateError(f"Non-positive multiplier on {tokens[0]}: m={mult}")

    used_models[model] = _polarity(model, geom)
    # Instance names are only unique within their .subckt (the charge pump has
    # an Npm11 in both of its blocks), so the audit key carries the scope.
    # PyCircuitSim's own flat name is "M.<instance path>.<name>", which is not
    # knowable here because one .subckt may be instantiated more than once.
    multipliers[f"{scope.name}:{name}" if scope is not None else name] = mult
    return (f"{name} {' '.join(nodes)} {model} "
            f"L={_fmt(geom['l'])} NFIN={_fmt_count(geom['nfin'])} "
            f"TFIN={_fmt(geom['tfin'])} m={_fmt_count(mult)}")


def _translate_instance(tokens: Sequence[str], scope: Optional[_Scope],
                        warnings: List[str]) -> str:
    """Translate an ``X`` subcircuit instance: lower every connection node.

    The instance name and the trailing subckt-name reference keep their case
    (the subckt table is keyed on the upper-cased name, and the flattened
    node prefix must match what ``.nodeset V(x1.net42)`` refers to).
    """
    positional = [t for t in tokens[1:] if '=' not in t]
    params = [t for t in tokens[1:] if '=' in t]
    if len(positional) < 2:
        raise TranslateError(f"Malformed subcircuit instance: {' '.join(tokens)}")
    nodes = [_lower_node(t, scope, warnings) for t in positional[:-1]]
    return ' '.join([tokens[0]] + nodes + [positional[-1]] + params)


def _translate_subckt_header(tokens: Sequence[str], warnings: List[str]
                             ) -> Tuple[str, _Scope]:
    """Translate a ``.subckt`` header and open its scope.

    A port whose name is a ground spelling is renamed to ``gnd_p``: PyCircuitSim
    rejects a ground-named port outright (audit C6i), and since the instance
    already ties that port to node 0 the flattened circuit is identical.
    """
    if len(tokens) < 2:
        raise TranslateError(f"Malformed .subckt: {' '.join(tokens)}")
    name = tokens[1]
    ports: List[str] = []
    gnd_ports: List[str] = []
    tail: List[str] = []
    for token in tokens[2:]:
        if '=' in token:
            tail.append(token)
            continue
        low = token.lower()
        if low in _GROUND_ALIASES:
            gnd_ports.append(low)
            ports.append(_GND_PORT_ALIAS)
            warnings.append(
                f".subckt {name}: port '{token}' names ground; renamed to "
                f"'{_GND_PORT_ALIAS}' (the instance ties it to node 0)")
        else:
            ports.append(low)
    header = ' '.join(['.subckt', name] + ports + tail)
    return header, _Scope(name, gnd_ports)


# ── Analysis-control parsing ─────────────────────────────────────────────

def parse_control(control: str) -> List[AnalysisPlan]:
    """Parse an injected ngspice control block into :class:`AnalysisPlan`s.

    Recognised forms::

        ac dec <n_per_decade> <f_start> <f_stop>
        dc temp <start> <stop> <step>
        dc <source> <start> <stop> <step>
        tran <t_step> <t_stop> [<t_start> [<t_max>]]

    A multi-line block (``size_ldo.line_control``) yields one plan per ``dc``
    line, labelled ``"up"`` and ``"dn"`` in order; its ``let``/``echo`` lines
    are recognised and dropped because ``measure.combine_line_segments``
    reimplements that vector algebra.

    Raises:
        TranslateError: On an unrecognised control line
    """
    plans: List[AnalysisPlan] = []
    for raw in control.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = line.split()[0].lower()
        if head in ('let', 'echo', 'set', 'print'):
            continue
        if head == 'ac':
            plans.append(_plan_ac(line))
        elif head == 'dc':
            plans.append(_plan_dc(line))
        elif head == 'tran':
            plans.append(_plan_tran(line))
        else:
            raise TranslateError(f"Unrecognised analysis control line: {line}")
    if not plans:
        raise TranslateError(f"Control block yielded no analysis: {control!r}")

    dc_source = [i for i, p in enumerate(plans) if p.kind == 'dc_source']
    if len(plans) > 1 and len(dc_source) == 2:
        for idx, label in zip(dc_source, ('up', 'dn')):
            plans[idx] = replace(plans[idx], label=label)
    elif len(plans) > 1:
        raise TranslateError(
            f"Multi-analysis control block is not the two-segment line sweep: "
            f"{control!r}")
    return plans


def _plan_ac(line: str) -> AnalysisPlan:
    parts = line.split()
    if len(parts) != 5 or parts[1].lower() != 'dec':
        raise TranslateError(f"Only 'ac dec <n> <f1> <f2>' is supported: {line}")
    return AnalysisPlan(kind='ac', control=line,
                        n_per_decade=int(_ng_value(parts[2])),
                        f_start=_ng_value(parts[3]),
                        f_stop=_ng_value(parts[4]))


def _plan_dc(line: str) -> AnalysisPlan:
    parts = line.split()
    if len(parts) != 5:
        raise TranslateError(
            f"Only 'dc <source|temp> <start> <stop> <step>' is supported: {line}")
    source = parts[1]
    start, stop, step = (_ng_value(p) for p in parts[2:5])
    if step == 0.0:
        raise TranslateError(f"Zero sweep step: {line}")
    if (stop - start) * step < 0:
        raise TranslateError(f"Sweep step points away from the stop value: {line}")
    kind = 'dc_temp' if source.lower() == 'temp' else 'dc_source'
    return AnalysisPlan(kind=kind, control=line,
                        source='temp' if kind == 'dc_temp' else source,
                        start=start, stop=stop, step=step)


def _plan_tran(line: str) -> AnalysisPlan:
    parts = line.split()
    if len(parts) not in (3, 4, 5):
        raise TranslateError(
            f"Only 'tran <tstep> <tstop> [tstart [tmax]]' is supported: {line}")
    t_step, t_stop = _ng_value(parts[1]), _ng_value(parts[2])
    t_max: Optional[float] = None
    if len(parts) >= 4:
        t_start = _ng_value(parts[3])
        if t_start != 0.0:
            raise TranslateError(
                f"A non-zero transient start time is not supported "
                f"(AnalysisPlan carries no t_start): {line}")
    if len(parts) == 5:
        t_max = _ng_value(parts[4])
    return AnalysisPlan(kind='tran', control=line,
                        t_step=t_step, t_stop=t_stop, t_max=t_max)


# ── Control tables ───────────────────────────────────────────────────────

def _tech_of(design_dir: Path) -> str:
    """Recover the tech tag ("TSMC5") from a design directory path."""
    for part in reversed(design_dir.resolve().parts):
        match = re.fullmatch(r'designs_(tsmc\d+)', part, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    raise TranslateError(f"Cannot infer tech from design path: {design_dir}")


def build_controls(category: str, design_dir: Path) -> Dict[str, str]:
    """Return ``{deck: ngspice control}`` for one design.

    The two computed families are filled in exactly as ``tools/finalize.py``
    computes them:

    * amplifier ``tb_tran.cir``: ``step_time = 10 / design.json["gbw_ideal"]``,
      ``tran {step_time/2000:.6g} {2.2*step_time:.6g}``, plus ``" 0 <cap>"``
      for the one ``(TECH, design)`` pair in ``build_amp.TRAN_MAXSTEP``.
    * ldo ``tb_load.cir`` / ``tb_line_{max,min}.cir``: from the design's own
      load range and rail (``size_ldo.line_control``).

    ``tb_tran_altns.cir`` is added for amplifiers with the same control as
    ``tb_tran.cir`` so :func:`translate_deck` can be called on it directly;
    ``finalize.py`` reuses the tb_tran control for the fallback the same way.

    Raises:
        TranslateError: For an unknown category
    """
    if category not in CONTROLS:
        raise TranslateError(f"Unknown category '{category}'")
    controls: Dict[str, Optional[str]] = dict(CONTROLS[category])

    if category == 'amplifier':
        tran = _amp_tran_control(design_dir)
        controls['tb_tran.cir'] = tran
        controls['tb_tran_altns.cir'] = tran
    elif category == 'ldo':
        design = _design_json(design_dir)
        imin, imax = design['iload_min'], design['iload_max']
        controls['tb_load.cir'] = (
            f"dc Iload {imin:g} {imax:g} {(imax - imin) / 100:g}")
        controls['tb_line_max.cir'] = _line_control('max', design['vdd'])
        controls['tb_line_min.cir'] = _line_control('min', design['vdd'])

    missing = [deck for deck, ctl in controls.items() if ctl is None]
    if missing:
        raise TranslateError(
            f"Category '{category}' left controls unfilled for {missing}")
    return {deck: ctl for deck, ctl in controls.items() if ctl is not None}


def _design_json(design_dir: Path) -> Dict[str, Any]:
    path = design_dir / 'design.json'
    if not path.exists():
        raise TranslateError(f"design.json missing for {design_dir}")
    return json.loads(path.read_text())


def _amp_tran_control(design_dir: Path) -> str:
    """Verbatim port of ``finalize._amp_tran_control``."""
    design = _design_json(design_dir)
    step_time = 10.0 / design.get('gbw_ideal', 1e6)
    control = f"tran {step_time / 2000:.6g} {2.2 * step_time:.6g}"
    cap = _TRAN_MAXSTEP.get((_tech_of(design_dir), design_dir.name.lower()))
    return f"{control} 0 {cap}" if cap else control


def _line_control(tag: str, vdd: float) -> str:
    """Verbatim port of ``size_ldo.line_control`` (including its let/echo tail).

    The ``let``/``echo`` lines are kept so the block stays byte-comparable with
    the reference control; :func:`parse_control` drops them, and
    ``measure.combine_line_segments`` reimplements the vector algebra.
    """
    lo, hi = 0.9 * vdd, 1.1 * vdd
    return "\n".join([
        f"dc V1 {vdd:g} {hi:g} 0.005",
        f"dc V1 {vdd:g} {lo:g} -0.005",
        "let seg_up_max = vecmax(dc1.v(vo))",
        "let seg_dn_max = vecmax(dc2.v(vo))",
        "let seg_up_min = vecmin(dc1.v(vo))",
        "let seg_dn_min = vecmin(dc2.v(vo))",
        "let seg_max = (seg_up_max + seg_dn_max + abs(seg_up_max - seg_dn_max)) / 2",
        "let seg_min = (seg_up_min + seg_dn_min - abs(seg_up_min - seg_dn_min)) / 2",
        "let seg_avg = (mean(dc1.v(vo)) + mean(dc2.v(vo))) / 2",
        "let seg_pp = seg_max - seg_min",
        "let seg_lnr = seg_pp / seg_avg / 0.2",
        f"echo lnr_avg{tag} = $&seg_avg",
        f"echo lnr_pp{tag} = $&seg_pp",
        f"echo lnr{tag} = $&seg_lnr",
    ])


# ── The translator ───────────────────────────────────────────────────────

def translate_deck(design_dir: Path, deck: str, *, tech: str, category: str,
                   control: Optional[str] = None) -> TranslatedDeck:
    """Translate one AnalogGym deck into a :class:`TranslatedDeck`.

    Args:
        design_dir: The design directory (contains the deck, netlist.spice,
            the models library, design.json and result.json)
        deck: Deck file name, e.g. ``"tb_gain.cir"``
        tech: Tech tag for the artifact names, e.g. ``"tsmc5"``
        category: One of the :data:`CONTROLS` keys
        control: Override the injected ngspice control block

    Raises:
        TranslateError: On any construct the translator cannot represent
            faithfully.  It never degrades silently — a dropped device or an
            unfolded node produces a plausible wrong answer, which is the one
            failure mode this whole design exists to prevent.
    """
    design_dir = Path(design_dir)
    deck_path = design_dir / deck
    warnings: List[str] = []

    lines, models_path = _load_logical_lines(deck_path)
    if models_path is None:
        raise TranslateError(f"{deck_path} includes no *_models.spice library")
    geoms = _scan_model_cards(models_path)
    params = resolve_params(lines)

    cards: List[str] = []
    used_models: Dict[str, str] = {}
    multipliers: Dict[str, float] = {}
    nodesets: Dict[str, float] = {}
    ic: Dict[str, float] = {}
    raw_meas: List[str] = []
    options: Dict[str, Any] = {}
    temp_c: Optional[float] = None
    source_devices = 0
    scope_stack: List[_Scope] = []

    for line in lines:
        scope = scope_stack[-1] if scope_stack else None
        low = line.lower()

        if low.startswith('.'):
            directive = low.split()[0]
            if directive == '.subckt':
                header, new_scope = _translate_subckt_header(
                    _tokenize(line), warnings)
                cards.append(header)
                scope_stack.append(new_scope)
            elif directive == '.ends':
                if not scope_stack:
                    raise TranslateError(f"'.ends' without .subckt: {line}")
                # Name the .ends so PyCircuitSim's own pairing check applies.
                cards.append(f".ends {scope_stack.pop().name}")
            elif directive == '.nodeset':
                nodesets.update(_node_assignments(line, params, scope, warnings))
            elif directive == '.ic':
                ic.update(_node_assignments(line, params, scope, warnings))
            elif directive in ('.option', '.options'):
                _collect_options(line, options, warnings)
            elif directive == '.temp':
                temp_c = _ng_value(_tokenize(line)[1])
            elif directive in ('.meas', '.measure'):
                raw_meas.append(line)
            elif directive in ('.param', '.end', '.title', '.include'):
                pass                      # consumed above / deliberately dropped
            else:
                warnings.append(f"dropped unsupported directive: {line}")
            continue

        tokens = _tokenize(line)
        first = tokens[0][0].upper()
        if first == 'N':
            source_devices += 1
            cards.append(_translate_mosfet(tokens, geoms, scope, used_models,
                                           multipliers, warnings))
        elif first in ('R', 'C', 'L'):
            cards.append(_translate_passive(tokens, params, scope, warnings))
        elif first in ('V', 'I'):
            cards.append(_translate_source(tokens, params, scope, warnings))
        elif first == 'X':
            cards.append(_translate_instance(tokens, scope, warnings))
        else:
            raise TranslateError(
                f"Unsupported device letter '{tokens[0][0]}' in {deck_path}: "
                f"{line}")

    if scope_stack:
        raise TranslateError(
            f"Unterminated .subckt '{scope_stack[-1].name}' in {deck_path}")

    stubs = [f".model {name} {kind} (LEVEL=72)"
             for name, kind in used_models.items()]
    header = [
        f"* {tech} / {category} / {design_dir.name} / {deck}",
        "* Translated for PyCircuitSim by pycircuitsim_bench.translate.",
        f"* Geometry (L/NFIN/TFIN) scraped from {models_path.name}; analysis",
        "* directives are carried as AnalysisPlan data, not as cards.",
    ]
    netlist_text = '\n'.join(header + stubs + cards) + '\n'

    if control is None:
        control = build_controls(category, design_dir)[deck]
    plans = parse_control(control)

    if options.get('gmin') is not None:
        warnings.append(
            f"ngspice option gmin={options['gmin']} recorded but NOT applied: "
            f"PyCircuitSim's gmin is a per-device gds floor, not a node shunt")

    _validate(cards, used_models, source_devices, deck_path)

    td = TranslatedDeck(
        tech=tech, category=category, design=design_dir.name, deck=deck,
        design_dir=design_dir, netlist_text=netlist_text,
        modelcard_path=models_path, plans=plans, meas=_parse_meas(raw_meas),
        nodesets=nodesets, ic=ic, params=params,
        options=DeckOptions(**options), devices=source_devices,
        multipliers=multipliers, stability=STABILITY_DECKS.get(deck),
        temp_c=temp_c, warnings=warnings)
    return td


def _parse_meas(raw_lines: Sequence[str]) -> List[MeasCard]:
    """Hand the raw ``.meas`` cards to I3's parser.

    Imported lazily and by module attribute so ``translate.py`` stays
    importable (and unit-testable) while ``measure.py`` is being written in
    parallel — this is the only cross-module call in the split.
    """
    from . import measure
    return list(measure.parse_meas_cards(list(raw_lines)))


def _node_assignments(line: str, env: Mapping[str, float],
                      scope: Optional[_Scope],
                      warnings: List[str]) -> Dict[str, float]:
    """Parse ``V(node)=value`` pairs off a ``.nodeset`` / ``.ic`` card."""
    found = _NODE_ASSIGN_RE.findall(line)
    if not found:
        raise TranslateError(f"No V(node)=value assignment found in: {line}")
    return {_lower_node(node, scope, warnings): _eval_expr(value, env)
            for node, value in found}


def _collect_options(line: str, options: Dict[str, Any],
                     warnings: List[str]) -> None:
    """Merge one ``.option``/``.options`` card into the DeckOptions kwargs."""
    for token in _tokenize(_normalise_equals(line))[1:]:
        if '=' not in token:
            warnings.append(f"dropped valueless option token {token!r}")
            continue
        key, value = token.split('=', 1)
        key = key.strip().lower()
        kind = _OPTION_KEYS.get(key)
        if kind is None:
            warnings.append(f"dropped unsupported option {key}={value}")
            continue
        if kind == 'str':
            options[key] = value.strip().lower()
        elif kind == 'int':
            options[key] = int(_ng_value(value))
        else:
            options[key] = _ng_value(value)


def _validate(cards: Sequence[str], used_models: Mapping[str, str],
              source_devices: int, deck_path: Path) -> None:
    """Rule 13 self-validation: fail loud rather than emit a plausible deck."""
    emitted_models = 0
    referenced: List[str] = []
    for card in cards:
        letter = card[0].upper()
        if letter not in _EMITTABLE:
            raise TranslateError(
                f"Emitted card starts with unsupported letter {card[0]!r} "
                f"({deck_path}): {card}")
        if letter == 'M':
            emitted_models += 1
            referenced.append(card.split()[5])
    if emitted_models != source_devices:
        raise TranslateError(
            f"{deck_path}: {source_devices} source 'N' devices but "
            f"{emitted_models} emitted 'M' cards")
    missing = sorted(set(referenced) - set(used_models))
    if missing:
        raise TranslateError(
            f"{deck_path}: M cards reference un-emitted .model tokens {missing}")


# ── Derived deck variants ────────────────────────────────────────────────

def recovery_decks(td: TranslatedDeck) -> List[TranslatedDeck]:
    """Amplifier ``tb_dc.cir`` recovery passes (``finalize.py`` lines 171-219).

    The hot/cold pair re-runs the temperature sweep outward from the qualified
    25 C point (same coverage, different Newton continuation path) and the
    narrow pass recovers power/vos25/vout25/ivdd25 only.  Returns ``[]`` for
    every other deck.
    """
    if td.category != 'amplifier' or td.deck != 'tb_dc.cir':
        return []
    out: List[TranslatedDeck] = []
    segments = list(AMP_TB_DC_RECOVERY) + [('narrow', AMP_TB_DC_NARROW)]
    for label, control in segments:
        plans = [replace(p, label=label) for p in parse_control(control)]
        out.append(replace(td, plans=plans))
    return out


def altns_deck(td: TranslatedDeck) -> Optional[TranslatedDeck]:
    """Translate the alternate-nodeset twin of an amplifier ``tb_tran.cir``.

    The two decks differ only in ``.nodeset``; some designs' transient
    operating point solves from only one of the two seeds, so which seed won
    is recorded rather than hidden.
    """
    if td.category != 'amplifier' or td.deck != 'tb_tran.cir':
        return None
    alt = td.design_dir / 'tb_tran_altns.cir'
    if not alt.exists():
        return None
    return translate_deck(td.design_dir, 'tb_tran_altns.cir', tech=td.tech,
                          category=td.category, control=td.plans[0].control)


def write_netlist(td: TranslatedDeck, out_dir: Path) -> Path:
    """Write ``td.netlist_text`` as the audit artifact run_compare parses."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{td.tech}_{td.design}_{Path(td.deck).stem}.sp"
    path.write_text(td.netlist_text)
    return path
