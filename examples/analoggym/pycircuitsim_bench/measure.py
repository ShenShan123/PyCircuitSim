"""The AnalogGym ``.meas`` language: one parser, one evaluator, two simulators.

Everything the migration scores is a ``.meas`` card or a derived quantity built
on top of one, and the whole point of this module is that a PyCircuitSim sweep
and an NGSPICE sweep go through the SAME code: a disagreement between the two
metric columns is then a physics or numerics difference, never a difference in
what "AVG from=-40 to=125" was taken to mean.

The implemented form list is a census of every ``.meas``/``.measure`` card in
``designs_tsmc{5,6,7,12,16}`` (2058 cards over two trees, 1029 per tree), not a
reading of a SPICE manual.  Grouped by form, with the two-tree counts:

===== ======================================================================
 428  ``.meas tran <n> param='<expr>'``          (+ 84/70/54/34/2 dc & ac)
 214  ``.meas tran <n> avg v(x) from=<a> to=<b>``
 272  ``.measure dc <n> {min,max} v(x) from=<a> to=<b>``
 132  ``.measure dc <n> find v(x) at=<x0>``      (+ 54 ``.meas`` spelling)
 112  ``.measure dc <n> avg v(x) from=<a> to=<b>``
  68  ``.measure dc <n> pp v(x) from=<a> to=<b>``
 176  ``.meas[ure] ac <n> find vdb(x) at=<f0>``
 136  ``.meas tran <n> when v(x)=<val> {rise,fall}=<k> td=<t>``
  56  ``.meas[ure] ac <n> when vdb(x)=<val>``
  56  ``.meas[ure] ac <n> find vp(x) when vdb(x)=<val>``
  56  ``.meas dc <n> find i(<src>) at=<x0>``
  12  ``.measure tran <n> {min,max,avg} i(<src>) from=<a> to=<b>``
===== ======================================================================

Proven ABSENT from the corpus and therefore not implemented: TRIG/TARG, DERIV,
INTEG, RMS, CROSS=, and measurements on an expression of two nodes.  An
unrecognised card raises rather than being skipped -- a silently dropped
measurement reads as ngspice's "failed", which is a legitimate outcome, so the
two would be indistinguishable.

Semantics that are load-bearing and were verified against real runs:

* ``when`` on an AC sweep interpolates linearly in the measured quantity AND in
  ``log10(f)``.  That is ``tools/acstab.py``'s definition and it is the one the
  scored artifacts use; interpolating linearly in ``f`` is wrong by ~0.15 % on
  the pilot amplifier.
* ``avg`` is the trapezoidal integral over the window divided by the window
  width -- a range-weighted mean, not ``sum/N``.  The plain arithmetic mean
  that the LDO line-regulation control block uses lives in
  :func:`combine_line_segments`, deliberately not here.
* window endpoints are interpolated into the trace before reducing, and the
  window is direction-independent (the amplifier sweeps 125 -> -40 while its
  cards say ``from=-40 to=125``).
* ``vp()`` is RADIANS (principal value).  ``vdb()`` is ``20*log10|V|``.
* ``i(<src>)`` is NGSPICE's sign: current into the ``+`` terminal, negative
  when the source delivers.
* a measurement with no solution is ``None``, never ``NaN`` and never ``0.0``.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import Condition, MeasCard, MetricDict, Signal, SweepResult

__all__ = [
    "spice_number",
    "parse_meas_cards",
    "eval_signal",
    "measure",
    "eval_param_expr",
    "stability_metrics",
    "sfe_sweep_metrics",
    "combine_line_segments",
    "recombine_temp_segments",
    "compare_metrics",
]

# --------------------------------------------------------------------------
# Lexical layer
# --------------------------------------------------------------------------

_ANALYSES = ("ac", "dc", "tran")
_REDUCTIONS = ("max", "min", "avg", "pp")
_SIGNAL_FUNCS = ("v", "vdb", "vp", "i")

# ngspice scale factors.  NOTE the deliberate difference from PyCircuitSim's
# ``Parser._parse_value``, which maps a trailing 'm' to 1e6: these are ngspice
# decks, so 'm' is milli and 'meg' is mega.  The corpus only uses 'u' (``from=1u
# to=1.9u`` and friends in the LDO transient bench), but getting 'm' wrong here
# would be a silent 1e9 error, so the full table is spelled out.
_SPICE_SCALE: Dict[str, float] = {
    "t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3,
    "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18,
}

# ``np.trapz`` is removed in NumPy 2.x, ``np.trapezoid`` absent before it.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

_NUMBER_RE = re.compile(
    r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)([a-zA-Z]*)$")
_SIGNAL_RE = re.compile(r"^(v|vdb|vp|i)\(([^()]*)\)$", re.IGNORECASE)
_CONDITION_RE = re.compile(
    r"^((?:v|vdb|vp|i)\([^()]*\))=(.+)$", re.IGNORECASE)


def spice_number(token: str) -> float:
    """Parse an ngspice numeric literal, honouring its scale suffixes.

    ``1e-06`` -> 1e-6, ``22u`` -> 2.2e-5, ``4meg`` -> 4e6.  Trailing unit
    letters after the scale factor are ignored the way ngspice ignores them
    (``20ns`` is 2e-8); an alpha tail that is not a scale factor at all is
    treated as a bare unit (factor 1.0).
    """
    m = _NUMBER_RE.match(token.strip())
    if m is None:
        raise ValueError(f"not a number: {token!r}")
    mantissa = float(m.group(1))
    suffix = m.group(2).lower()
    if not suffix:
        return mantissa
    if suffix.startswith("meg"):
        return mantissa * 1e6
    if suffix.startswith("mil"):
        return mantissa * 25.4e-6
    return mantissa * _SPICE_SCALE.get(suffix[0], 1.0)


def _strip_comment(line: str) -> str:
    """Truncate a logical line at an unquoted ``$`` or ``;`` comment tail."""
    out: List[str] = []
    quote: Optional[str] = None
    for i, ch in enumerate(line):
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
            continue
        if ch == ";":
            break
        if ch == "$" and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out)


def _normalise_eq(line: str) -> str:
    """Collapse whitespace around every unquoted ``=``.

    The decks write ``at = 0.1``, ``param = '...'`` and ``PP  V(vout6)``
    interchangeably with the tight spellings; normalising here means the token
    grammar below never has to care.
    """
    out: List[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "=":
            while out and out[-1].isspace():
                out.pop()
            out.append("=")
            i += 1
            while i < n and line[i].isspace():
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _tokenise(line: str) -> List[str]:
    """Whitespace-split *line*, keeping quoted spans (the param expr) intact."""
    tokens: List[str] = []
    cur: List[str] = []
    quote: Optional[str] = None
    for ch in line:
        if quote is not None:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            cur.append(ch)
            continue
        if ch.isspace():
            if cur:
                tokens.append("".join(cur))
                cur = []
            continue
        cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return tokens


def _unquote(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _parse_signal(token: str) -> Signal:
    m = _SIGNAL_RE.match(token.strip())
    if m is None:
        raise ValueError(f"unrecognised output function: {token!r}")
    return Signal(func=m.group(1).lower(), arg=m.group(2).strip().lower())


def _parse_condition(tokens: Sequence[str]) -> Condition:
    """Parse ``<signal>=<value> [rise=k|fall=k] [td=t]`` (already ``=``-tight)."""
    if not tokens:
        raise ValueError("missing condition")
    m = _CONDITION_RE.match(tokens[0])
    if m is None:
        raise ValueError(f"unrecognised condition: {tokens[0]!r}")
    signal = _parse_signal(m.group(1))
    value = spice_number(m.group(2))
    edge: Optional[str] = None
    nth = 1
    td = 0.0
    for tok in tokens[1:]:
        low = tok.lower()
        if low.startswith("rise=") or low.startswith("fall="):
            edge = low.split("=", 1)[0]
            nth = int(round(spice_number(low.split("=", 1)[1])))
        elif low.startswith("td="):
            td = spice_number(low[3:])
        else:
            raise ValueError(f"unrecognised condition qualifier: {tok!r}")
    return Condition(signal=signal, value=value, edge=edge, nth=nth, td=td)


def _parse_window(tokens: Sequence[str]
                  ) -> Tuple[Optional[float], Optional[float]]:
    lo: Optional[float] = None
    hi: Optional[float] = None
    for tok in tokens:
        low = tok.lower()
        if low.startswith("from="):
            lo = spice_number(low[5:])
        elif low.startswith("to="):
            hi = spice_number(low[3:])
        else:
            raise ValueError(f"unrecognised window qualifier: {tok!r}")
    return (lo, hi)


def _parse_card(line: str) -> Optional[MeasCard]:
    raw = line.rstrip()
    body = _strip_comment(line).strip()
    if not body or body.startswith("*"):
        return None
    tokens = _tokenise(_normalise_eq(body))
    if not tokens:
        return None
    if tokens[0].lower() not in (".meas", ".measure"):
        raise ValueError(f"not a .meas card: {raw!r}")
    if len(tokens) < 4:
        raise ValueError(f"truncated .meas card: {raw!r}")
    analysis = tokens[1].lower()
    if analysis not in _ANALYSES:
        raise ValueError(f"unrecognised .meas analysis {analysis!r}: {raw!r}")
    name = tokens[2].lower()
    keyword = tokens[3].lower()

    if keyword.startswith("param="):
        expr = _unquote(tokens[3][len("param="):])
        if not expr:
            raise ValueError(f"empty param expression: {raw!r}")
        return MeasCard(name=name, analysis=analysis, kind="param",
                        expr=expr, raw=raw)

    if keyword == "find":
        if len(tokens) < 6:
            raise ValueError(f"truncated FIND card: {raw!r}")
        signal = _parse_signal(tokens[4])
        tail = tokens[5:]
        if tail[0].lower().startswith("at="):
            if len(tail) != 1:
                raise ValueError(f"trailing tokens after AT=: {raw!r}")
            return MeasCard(name=name, analysis=analysis, kind="find_at",
                            signal=signal, at=spice_number(tail[0][3:]),
                            raw=raw)
        if tail[0].lower() == "when":
            return MeasCard(name=name, analysis=analysis, kind="find_when",
                            signal=signal, cond=_parse_condition(tail[1:]),
                            raw=raw)
        raise ValueError(f"unrecognised FIND card: {raw!r}")

    if keyword == "when":
        return MeasCard(name=name, analysis=analysis, kind="when",
                        cond=_parse_condition(tokens[4:]), raw=raw)

    if keyword in _REDUCTIONS:
        if len(tokens) < 5:
            raise ValueError(f"truncated {keyword.upper()} card: {raw!r}")
        return MeasCard(name=name, analysis=analysis, kind=keyword,
                        signal=_parse_signal(tokens[4]),
                        window=_parse_window(tokens[5:]), raw=raw)

    raise ValueError(f"unrecognised .meas keyword {keyword!r}: {raw!r}")


def parse_meas_cards(lines: Sequence[str]) -> List[MeasCard]:
    """Parse ``.meas``/``.measure`` cards, in deck order.

    Blank lines and ``*`` comments are dropped; anything else that is not a
    card this module implements raises ``ValueError``.  Never skip silently: a
    dropped card is indistinguishable from ngspice's "failed", which is a real
    outcome the comparison has to be able to trust.
    """
    cards: List[MeasCard] = []
    for line in lines:
        card = _parse_card(line)
        if card is not None:
            cards.append(card)
    return cards


# --------------------------------------------------------------------------
# Signal extraction
# --------------------------------------------------------------------------

def eval_signal(result: SweepResult, signal: Signal) -> Optional[np.ndarray]:
    """Materialise one ``.meas`` output function over a whole sweep.

    ``v`` on an AC sweep is the complex phasor (no magnitude applied); on
    dc/tran it is the real trace.  ``vdb`` is ``20*log10|V|``, ``vp`` is
    ``np.angle`` in RADIANS (principal value), ``i`` is the branch current in
    NGSPICE sign.  Returns ``None`` -- never raises -- when the vector the card
    asks for is not in *result*.
    """
    table = result.i if signal.func == "i" else result.v
    raw = table.get(signal.arg)
    if raw is None:
        return None
    arr = np.asarray(raw)
    if signal.func == "vdb":
        with np.errstate(divide="ignore", invalid="ignore"):
            return 20.0 * np.log10(np.abs(arr.astype(np.complex128)))
    if signal.func == "vp":
        return np.angle(arr.astype(np.complex128))
    if result.kind == "ac":
        return arr.astype(np.complex128)
    return np.real(arr).astype(np.float64)


def _abscissa(result: SweepResult) -> np.ndarray:
    return np.asarray(result.x, dtype=np.float64)


def _logx(result: SweepResult) -> bool:
    """AC abscissae are decade-spaced frequencies; everything else is linear."""
    return result.kind == "ac"


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Points every given array can describe (the reference dumps drop these)."""
    good: Optional[np.ndarray] = None
    for arr in arrays:
        col = np.isfinite(np.asarray(arr, dtype=np.complex128))
        good = col if good is None else (good & col)
    assert good is not None
    return good


def _finite_pair(x: np.ndarray, y: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    good = _finite_mask(x, y)
    return x[good], np.asarray(y)[good]


def _scalar(value: Any) -> Optional[float]:
    """Coerce an interpolated sample to a finite float, or ``None``."""
    if value is None:
        return None
    if isinstance(value, complex) or np.iscomplexobj(value):
        value = abs(complex(value))
    out = float(value)
    return out if math.isfinite(out) else None


def _weight(a: float, b: float, x0: float, logx: bool) -> float:
    if logx and a > 0.0 and b > 0.0:
        la, lb = math.log10(a), math.log10(b)
        if lb == la:
            return 0.0
        return (math.log10(x0) - la) / (lb - la)
    if b == a:
        return 0.0
    return (x0 - a) / (b - a)


def _interp_at(x: np.ndarray, y: np.ndarray, x0: float,
               logx: bool) -> Optional[Any]:
    """Sample *y* at abscissa *x0*, interpolating between bracketing points.

    Works on an ascending or a descending sweep.  An abscissa within
    ``1e-9*max(1,|x0|)`` of a solved point is an exact hit -- the AC grid puts
    ``at=0.1`` exactly on point 0 -- otherwise the two bracketing points are
    interpolated (in ``log10(x)`` for a frequency abscissa).  ``x0`` outside the
    solved range is ngspice's "out of interval": ``None``.
    """
    n = len(x)
    if n == 0:
        return None
    tol = 1e-9 * max(1.0, abs(x0))
    hits = np.nonzero(np.abs(x - x0) <= tol)[0]
    if hits.size:
        return y[hits[0]]
    for i in range(n - 1):
        a, b = float(x[i]), float(x[i + 1])
        if (a - x0) * (b - x0) < 0.0:
            t = _weight(a, b, x0, logx)
            return y[i] + t * (y[i + 1] - y[i])
    return None


def _crossings(x: np.ndarray, y: np.ndarray, value: float,
               edge: Optional[str], td: float, logx: bool
               ) -> List[Tuple[float, int, float]]:
    """Every crossing of ``y == value``, as ``(x_cross, index, weight)``.

    *edge* filters the direction (``None`` accepts both).  *td* discards
    crossings at or before it, which is what ngspice's ``td=`` does; a ``td`` of
    exactly 0.0 means "no delay" and is NOT applied, because a dc abscissa
    (temperature, load current) legitimately runs negative.
    """
    out: List[Tuple[float, int, float]] = []
    for i in range(len(x) - 1):
        y0, y1 = float(np.real(y[i])), float(np.real(y[i + 1]))
        if edge == "rise" or edge is None:
            rising = y0 < value <= y1
        else:
            rising = False
        if edge == "fall" or edge is None:
            falling = y0 > value >= y1
        else:
            falling = False
        if not (rising or falling):
            continue
        t = 0.0 if y1 == y0 else (value - y0) / (y1 - y0)
        a, b = float(x[i]), float(x[i + 1])
        if logx and a > 0.0 and b > 0.0:
            x_c = 10.0 ** (math.log10(a) + t * (math.log10(b) - math.log10(a)))
        else:
            x_c = a + t * (b - a)
        if td != 0.0 and x_c <= td:
            continue
        out.append((x_c, i, t))
    return out


def _sample_at_weight(y: np.ndarray, index: int, t: float,
                      func: str) -> Optional[float]:
    """Sample *y* at the crossing found in *y_ref*, same bracketing weight.

    A phase is interpolated on the UNWRAPPED sequence and wrapped back into
    ``(-pi, pi]`` afterwards, so a branch cut falling between the two
    bracketing points cannot corrupt the reading -- this is what makes
    ``phase_in_rad`` agree with ``acstab``'s ``ph_xover``.
    """
    if func == "vp":
        unwrapped = np.unwrap(np.real(y))
        val = unwrapped[index] + t * (unwrapped[index + 1] - unwrapped[index])
        val = ((float(val) + math.pi) % (2.0 * math.pi)) - math.pi
        return val
    val = y[index] + t * (y[index + 1] - y[index])
    return _scalar(val)


def _windowed(x: np.ndarray, y: np.ndarray,
              window: Optional[Tuple[Optional[float], Optional[float]]]
              ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Ascending (abscissa, value) trace clipped to a closed window.

    Both endpoints are interpolated into the trace before any reduction runs,
    and the window is direction-independent: the amplifier temperature bench
    sweeps 125 -> -40 while its cards say ``from=-40 to=125``.  Endpoints are
    inclusive within ``1e-9*max(1,|x|)`` so a float endpoint like
    ``-39.99999999999747`` is not excluded.

    A window that overhangs the solved data is CLAMPED to it, not rejected --
    that is measured ngspice behaviour: on a sweep that stops at -39.9 the card
    ``from=-40 to=125`` still reports, over -39.9..125.  A clamp that collapses
    to a single point is left as one point; :func:`_reduce` then decides, as
    ngspice does (MAX/MIN report it, AVG fails).
    """
    if len(x) == 0:
        return None
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = np.asarray(y)[order]
    lo = float(xs[0]) if window is None or window[0] is None else float(window[0])
    hi = float(xs[-1]) if window is None or window[1] is None else float(window[1])
    if hi < lo:
        lo, hi = hi, lo
    lo = max(lo, float(xs[0]))
    hi = min(hi, float(xs[-1]))
    if hi < lo:
        return None
    tol_lo = 1e-9 * max(1.0, abs(lo))
    tol_hi = 1e-9 * max(1.0, abs(hi))
    v_lo = _interp_at(xs, ys, lo, False)
    v_hi = _interp_at(xs, ys, hi, False)
    if v_lo is None or v_hi is None:
        return None
    if hi - lo <= max(tol_lo, tol_hi):
        return np.array([lo]), np.asarray([v_lo])
    mask = (xs > lo + tol_lo) & (xs < hi - tol_hi)
    xw = np.concatenate(([lo], xs[mask], [hi]))
    yw = np.concatenate(([v_lo], ys[mask], [v_hi]))
    return xw, yw


def _reduce(kind: str, xw: np.ndarray, yw: np.ndarray) -> Optional[float]:
    vals = np.real(yw).astype(np.float64)
    if vals.size == 0:
        return None
    if kind == "max":
        return _scalar(np.max(vals))
    if kind == "min":
        return _scalar(np.min(vals))
    if kind == "pp":
        return _scalar(np.max(vals) - np.min(vals))
    if kind == "avg":
        span = float(xw[-1]) - float(xw[0])
        if vals.size < 2 or span <= 0.0:
            # An AVG needs an interval.  ngspice reports "failed" here and that
            # matters: the amplifier's hot/cold recovery passes each carry the
            # OTHER segment's window, and a degenerate value would overwrite the
            # real one when the two passes are merged (measured -- ngspice's cold
            # run prints max_hot/min_hot but no avg_hot).
            return None
        # Range-weighted (trapezoidal) mean -- ngspice's AVG, and the identity
        # finalize.py relies on when it recombines (100*avg_hot+65*avg_cold)/165.
        return _scalar(float(_trapezoid(vals, xw) / span))
    raise ValueError(f"not a reduction: {kind!r}")


# --------------------------------------------------------------------------
# param='<expr>'
# --------------------------------------------------------------------------

class _MissingDependency(Exception):
    """A name the expression needs is unmeasured, or measured as ``None``."""


_ALLOWED_CALLS: Dict[str, Any] = {"abs": abs, "max": max, "min": min}


def _eval_expr_node(node: ast.AST, env: Mapping[str, Optional[float]]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_expr_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)):
            raise ValueError(f"non-numeric literal {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key not in env or env[key] is None:
            raise _MissingDependency(key)
        return float(env[key])          # type: ignore[arg-type]
    if isinstance(node, ast.UnaryOp):
        val = _eval_expr_node(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval_expr_node(node.left, env)
        right = _eval_expr_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("unsupported binary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("unsupported call")
        fname = node.func.id.lower()
        if fname not in _ALLOWED_CALLS:
            raise ValueError(f"unsupported function {fname!r}")
        args = [_eval_expr_node(a, env) for a in node.args]
        return float(_ALLOWED_CALLS[fname](*args))
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def eval_param_expr(expr: str,
                    env: Mapping[str, Optional[float]]) -> Optional[float]:
    """Evaluate a ``.meas ... param='<expr>'`` body over *env*.

    The grammar is exactly what the corpus uses: ``+ - * /``, unary minus,
    parentheses, ``abs()`` and ``max()`` (``min()`` allowed for symmetry), plus
    numeric literals and names measured earlier or resolved from ``.param``.
    Name lookup is case-insensitive.  Any missing or ``None`` dependency, a
    division by zero, or a non-finite result yields ``None`` -- ngspice's
    "failed".  ``Parser._eval_expr`` is deliberately not reused: it rejects
    identifiers absent from its params dict and charset-checks the residue, so
    ``abs()``/``max()`` raise there.
    """
    lowered = {str(k).lower(): v for k, v in env.items()}
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse param expression {expr!r}") from exc
    try:
        value = _eval_expr_node(tree, lowered)
    except _MissingDependency:
        return None
    except ZeroDivisionError:
        return None
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------

def _eval_card(card: MeasCard, result: SweepResult,
               env: Mapping[str, Optional[float]]) -> Optional[float]:
    if card.kind == "param":
        assert card.expr is not None
        return eval_param_expr(card.expr, env)

    x_all = _abscissa(result)
    logx = _logx(result)

    if card.kind == "find_at":
        assert card.signal is not None and card.at is not None
        y = eval_signal(result, card.signal)
        if y is None:
            return None
        x, y = _finite_pair(x_all, y)
        return _scalar(_interp_at(x, y, float(card.at), logx))

    if card.kind in ("when", "find_when"):
        assert card.cond is not None
        ref = eval_signal(result, card.cond.signal)
        if ref is None:
            return None
        y: Optional[np.ndarray] = None
        if card.kind == "find_when":
            assert card.signal is not None
            y = eval_signal(result, card.signal)
            if y is None:
                return None
        # ONE mask over the abscissa, the reference and the sampled signal
        # together: filtering them separately would let a non-finite point in
        # one of them slide the other's indices by a sample.
        good = (_finite_mask(x_all, ref) if y is None
                else _finite_mask(x_all, ref, y))
        x = x_all[good]
        ref = np.asarray(ref)[good]
        hits = _crossings(x, ref, card.cond.value, card.cond.edge,
                          card.cond.td, logx)
        if len(hits) < card.cond.nth:
            return None
        x_c, index, t = hits[card.cond.nth - 1]
        if card.kind == "when":
            return _scalar(x_c)
        assert card.signal is not None and y is not None
        return _sample_at_weight(np.asarray(y)[good], index, t,
                                 card.signal.func)

    if card.kind in _REDUCTIONS:
        assert card.signal is not None
        y = eval_signal(result, card.signal)
        if y is None:
            return None
        x, y = _finite_pair(x_all, y)
        clipped = _windowed(x, y, card.window)
        if clipped is None:
            return None
        return _reduce(card.kind, clipped[0], clipped[1])

    raise ValueError(f"cannot evaluate .meas kind {card.kind!r}")


def measure(cards: Sequence[MeasCard], result: SweepResult,
            params: Optional[Dict[str, float]] = None,
            seed: Optional[MetricDict] = None) -> MetricDict:
    """Evaluate *cards* over one sweep, in deck order.

    Only cards whose analysis word matches ``result.kind`` are evaluated -- that
    is what ngspice does, and a deck's cards are all one analysis in this corpus
    anyway.  *params* is the deck's resolved ``.param`` environment (measured:
    ``supply_voltage`` DOES resolve inside ``Power``, contrary to a deck
    comment); *seed* pre-loads names produced by an earlier pass over the same
    deck (the LDO line segments, the amplifier hot/cold recovery).

    Storage policy mirrors ``tools/meas.py`` exactly so the PyCircuitSim, the
    ngspice-sweep and the ngspice-``.meas`` columns are comparable key for key:
    a numeric reading overwrites whatever was there, a ``None`` only fills a
    name that has none yet.  (No deck in the corpus names a measurement twice,
    so this only matters when a caller merges passes.)
    """
    out: MetricDict = dict(seed) if seed else {}
    env: Dict[str, Optional[float]] = {}
    if params:
        env.update({str(k).lower(): float(v) for k, v in params.items()})
    env.update(out)

    for card in cards:
        if card.analysis != result.kind:
            continue
        value = _eval_card(card, result, env)
        if value is None:
            out.setdefault(card.name, None)
        else:
            out[card.name] = value
        env[card.name] = out[card.name]
    return out


# --------------------------------------------------------------------------
# Derived layers the artifacts actually score
# --------------------------------------------------------------------------

def _unwrap_deg(phases: np.ndarray) -> np.ndarray:
    """Unwrap principal-value phases in DEGREES (acstab.unwrap_deg)."""
    if phases.size == 0:
        return phases
    out = np.empty_like(phases)
    out[0] = phases[0]
    for i in range(1, phases.size):
        d = phases[i] - out[i - 1]
        d -= 360.0 * round(d / 360.0)
        out[i] = out[i - 1] + d
    return out


def stability_metrics(result: SweepResult, node: str, gbw_key: str,
                      suffix: str) -> MetricDict:
    """Wrap-aware loop stability of one AC sweep -- a port of ``acstab``.

    ``vp()`` reports the PRINCIPAL value, so a loop whose true phase has fallen
    through -180 deg before crossover wraps back into (+90..+180] and looks
    lead-recovered.  ``.meas`` cannot unwrap, so the margin is computed here:
    unwrap in degrees from the lowest frequency up, anchor ``phase_ref`` at the
    nearest multiple of 180 deg of the low-frequency phase, take the first
    ``g[i] >= 0 > g[i+1]`` crossing, interpolate the crossover in ``log10(f)``
    and the unwrapped phase with the same weight, then
    ``pm_true = 180 - dev_c`` unless the unwrapped deviation reaches 180 deg at
    or below crossover, in which case the margin is ``180 - maxdev`` (negative:
    a dip through -180 that recovers is still a Nyquist crossing).

    ``gbw_key`` is emitted here and OVERWRITES the deck's ``.meas`` value -- one
    definition of the 0 dB crossover for both artifact paths.  Everything is
    ``None`` when the gain never crosses 0 dB, which is a real outcome.
    """
    keys = ("pm_true", "ph_xover", "phase_ref", "phase_maxdev")
    none: MetricDict = {gbw_key: None}
    none.update({f"{k}{suffix}": None for k in keys})
    if result.kind != "ac":
        return none
    raw = result.v.get(node)
    if raw is None:
        return none
    freqs = _abscissa(result)
    values = np.asarray(raw, dtype=np.complex128)
    good = np.isfinite(freqs) & np.isfinite(values)
    freqs = freqs[good]
    values = values[good]
    if freqs.size < 2:
        return none
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = 20.0 * np.log10(np.abs(values))
    phases = np.degrees(np.angle(values))
    good = np.isfinite(gains) & np.isfinite(phases)
    freqs, gains, phases = freqs[good], gains[good], phases[good]
    if freqs.size < 2:
        return none
    unwrapped = _unwrap_deg(phases)
    idx = next((i for i in range(gains.size - 1)
                if gains[i] >= 0.0 > gains[i + 1]), None)
    if idx is None:
        return none
    t = gains[idx] / (gains[idx] - gains[idx + 1])
    lf = (math.log10(freqs[idx])
          + t * (math.log10(freqs[idx + 1]) - math.log10(freqs[idx])))
    ph_c = unwrapped[idx] + t * (unwrapped[idx + 1] - unwrapped[idx])
    ref = 180.0 * round(unwrapped[0] / 180.0)
    maxdev = max([abs(u - ref) for u in unwrapped[:idx + 1]]
                 + [abs(ph_c - ref)])
    dev_c = abs(ph_c - ref)
    pm = 180.0 - dev_c if maxdev < 180.0 else 180.0 - maxdev
    principal = ((ph_c + 180.0) % 360.0) - 180.0
    return {
        gbw_key: 10.0 ** lf,
        f"pm_true{suffix}": float(pm),
        f"ph_xover{suffix}": float(principal),
        f"phase_ref{suffix}": float(ref),
        f"phase_maxdev{suffix}": float(maxdev),
    }


def sfe_sweep_metrics(result: SweepResult, node: str = "vout") -> MetricDict:
    """Full-sweep smoothness of a PTAT front end -- a port of ``sfe``.

    ``mono_violations`` counts non-increasing adjacent pairs in 0..100 C;
    ``min_slope_25_75c`` and ``max_step_frac_25_75c`` are the staircase gates.
    ``sfe_score`` charges a 25.0 penalty when any of the three is missing, so
    they are gating, not decorative, and they must come from the whole solved
    table rather than from ``.meas`` samples.  Adjacent pairs are taken in the
    SOLVED order, exactly as the reference does off its ``wrdata`` dump (the
    bench sweeps -20 -> 120 C, so that order is ascending).
    """
    out: MetricDict = {
        "sweep_points": None, "mono_violations": None,
        "min_slope_25_75c": None, "max_step_frac_25_75c": None,
    }
    raw = result.v.get(node)
    if raw is None:
        return out
    temps = _abscissa(result)
    vouts = np.real(np.asarray(raw)).astype(np.float64)
    good = np.isfinite(temps) & np.isfinite(vouts)
    temps, vouts = temps[good], vouts[good]
    if temps.size < 3:
        return out
    out["sweep_points"] = float(temps.size)
    window = [(t, v) for t, v in zip(temps, vouts)
              if -1e-9 <= t <= 100.0 + 1e-9]
    if len(window) >= 2:
        out["mono_violations"] = float(sum(
            1 for (_, v0), (_, v1) in zip(window, window[1:]) if v1 <= v0))
    seg = [(t, v) for t, v in zip(temps, vouts)
           if 25.0 - 1e-9 <= t <= 75.0 + 1e-9]
    if len(seg) >= 2:
        slopes = [(v1 - v0) / (t1 - t0)
                  for (t0, v0), (t1, v1) in zip(seg, seg[1:]) if t1 > t0]
        if slopes:
            out["min_slope_25_75c"] = float(min(slopes))
        rise = seg[-1][1] - seg[0][1]
        max_step = max(v1 - v0 for (_, v0), (_, v1) in zip(seg, seg[1:]))
        out["max_step_frac_25_75c"] = float(max_step / rise) if rise > 0 else None
    return out


def combine_line_segments(up: SweepResult, dn: SweepResult, node: str,
                          tag: str) -> MetricDict:
    """LDO line regulation from two outward dc segments -- ``line_control``.

    The tb_line decks carry ZERO ``.meas`` cards: the reference computes the
    numbers in an ngspice control block over the two ``dc V1`` segments, and
    this reproduces its vector algebra exactly.  ``seg_avg`` uses ngspice's
    ``mean()``, i.e. the plain arithmetic mean of the SOLVED POINTS of each
    segment -- NOT the trapezoidal ``AVG`` of a ``.meas`` card.
    """
    out: MetricDict = {f"lnr_avg{tag}": None, f"lnr_pp{tag}": None,
                       f"lnr{tag}": None}
    a = up.v.get(node)
    b = dn.v.get(node)
    if a is None or b is None:
        return out
    va = np.real(np.asarray(a)).astype(np.float64)
    vb = np.real(np.asarray(b)).astype(np.float64)
    va = va[np.isfinite(va)]
    vb = vb[np.isfinite(vb)]
    if va.size == 0 or vb.size == 0:
        return out
    seg_max = max(float(np.max(va)), float(np.max(vb)))
    seg_min = min(float(np.min(va)), float(np.min(vb)))
    seg_avg = (float(np.mean(va)) + float(np.mean(vb))) / 2.0
    seg_pp = seg_max - seg_min
    out[f"lnr_avg{tag}"] = seg_avg
    out[f"lnr_pp{tag}"] = seg_pp
    if seg_avg != 0.0:
        out[f"lnr{tag}"] = seg_pp / seg_avg / 0.2
    return out


_TEMP_SEGMENT_KEYS = ("max_hot", "min_hot", "avg_hot",
                      "max_cold", "min_cold", "avg_cold")
_TEMP_CARRY_KEYS = ("power", "vos25", "vout25", "ivdd25")


def recombine_temp_segments(hot: MetricDict, cold: MetricDict) -> MetricDict:
    """Rebuild the full -40..125 C range from two continuation segments.

    A port of ``finalize.py``'s recovery pass: some high-gain loops lose their
    Newton branch in one monolithic 165 C sweep, so the bench is swept outward
    from 25 C in both directions and recombined.  ``avgval`` is the
    range-weighted blend ``(100*avg_hot + 65*avg_cold)/165`` -- which is only
    the true full-sweep average because ``AVG`` is trapezoidal.  ``power`` /
    ``vos25`` / ``vout25`` / ``ivdd25`` are carried through, cold winning over
    hot exactly as ``finalize``'s successive ``update`` calls do.  The
    recombined keys are ``None`` unless all six segment measurements exist.

    KNOWN TRAP, reproduced deliberately because this is a port and the ``ng``
    column has to match the reference path: each segment's deck also carries the
    OTHER segment's window, whose clamp collapses to the single point at 25 C, and
    ngspice reports MAX/MIN there (only AVG fails).  So ``max_hot``/``min_hot``
    come from the COLD pass and vice versa, and ``maxval`` can land far below the
    monolithic sweep's -- measured on Alfio_RAFFC_Pin_3: hot segment max_hot
    0.3701908 V, merged max_hot 0.1933828 V, recombined maxval 0.1933828 V
    against the monolithic 0.3699788 V.  A caller that wants the honest range
    must take ``max_hot``/``min_hot`` from the hot metrics and ``max_cold`` /
    ``min_cold`` from the cold ones itself; this function stays bug-compatible
    with ``finalize.py`` so the reference numbers remain reproducible.
    """
    merged: MetricDict = {}
    for part in (hot, cold):
        merged.update({k: v for k, v in part.items() if v is not None})
    out: MetricDict = dict(merged)
    for key in ("maxval", "minval", "avgval", "ppavl", "tc"):
        out.setdefault(key, None)
    if not all(merged.get(k) is not None for k in _TEMP_SEGMENT_KEYS):
        for key in ("maxval", "minval", "avgval", "ppavl", "tc"):
            out[key] = None
        return out
    maxval = max(float(merged["max_hot"]), float(merged["max_cold"]))
    minval = min(float(merged["min_hot"]), float(merged["min_cold"]))
    avgval = (100.0 * float(merged["avg_hot"])
              + 65.0 * float(merged["avg_cold"])) / 165.0
    ppavl = maxval - minval
    out["maxval"] = maxval
    out["minval"] = minval
    out["avgval"] = avgval
    out["ppavl"] = ppavl
    out["tc"] = (ppavl / avgval / 165.0) if avgval != 0.0 else None
    for key in _TEMP_CARRY_KEYS:
        if merged.get(key) is not None:
            out[key] = merged[key]
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def compare_metrics(a: MetricDict, b: MetricDict, *, rtol: float = 0.02,
                    atol_by_key: Optional[Mapping[str, float]] = None
                    ) -> Dict[str, Dict[str, Any]]:
    """Key-by-key comparison of two metric columns.

    ``None`` against ``None`` is agreement (both simulators declined the same
    measurement); ``None`` against a number never is.  A key present in only one
    column is reported as missing rather than dropped.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(a) | set(b)):
        va = a.get(key)
        vb = b.get(key)
        entry: Dict[str, Any] = {"a": va, "b": vb, "rel": None, "abs": None,
                                 "ok": False, "reason": ""}
        if key not in a:
            entry["reason"] = "absent-in-a"
        elif key not in b:
            entry["reason"] = "absent-in-b"
        elif va is None and vb is None:
            entry["ok"] = True
            entry["reason"] = "both-none"
        elif va is None:
            entry["reason"] = "none-in-a"
        elif vb is None:
            entry["reason"] = "none-in-b"
        elif not (math.isfinite(va) and math.isfinite(vb)):
            entry["reason"] = "non-finite"
        else:
            diff = abs(va - vb)
            denom = max(abs(va), abs(vb))
            rel = diff / denom if denom > 0.0 else 0.0
            entry["abs"] = diff
            entry["rel"] = rel
            atol = atol_by_key.get(key) if atol_by_key else None
            entry["ok"] = rel <= rtol or (atol is not None and diff <= atol)
            entry["reason"] = "" if entry["ok"] else f"rel={rel:.3e}"
        out[key] = entry
    return out
