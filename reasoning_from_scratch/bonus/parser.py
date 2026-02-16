"""Bonus utilities for Chapter 3.

Hybrid LaTeX answer parser from issue #133.

Source discussion:
https://github.com/rasbt/reasoning-from-scratch/issues/133

Original implementation shared by @labdmitriy and added here as bonus material.
"""

import re

from sympy import Contains, FiniteSet, Interval, Matrix, Symbol, Union, oo, sympify
from sympy.parsing.latex import parse_latex as _parse_latex

RE_TEXT = re.compile(r"^\\text\s*\{([^}]*)\}$")
RE_MBOX = re.compile(r"^\\mbox\s*\{([^}]*)\}$")
RE_CURRENCY = re.compile(r"^\\\$\s*([0-9,.\s\\!]+)$")
RE_MATRIX = re.compile(
    r"\\begin\{[pbvBV]?matrix\}(.*)\\end\{[pbvBV]?matrix\}", re.DOTALL
)
RE_TUPLE_LIKE = re.compile(r"^[\(\[\{].*,.*[\)\]\}]$", re.DOTALL)
RE_INTERVAL_BOUNDS = re.compile(r"^([\(\[])(.+),(.*)([\)\]])$")
RE_LEADING_DOT = re.compile(r"^(\.)(\d+)$")
RE_INFTY = re.compile(r"[+-]?\\infty")
RE_MEMBERSHIP = re.compile(r"^(?P<lhs>.+?)\\in(?P<rhs>.+)$", re.DOTALL)
RE_PM = re.compile(r"^(?P<a>.+?)\\pm(?P<b>.+)$", re.DOTALL)
RE_UNION = re.compile(r"\\cup")
RE_BARE_COMMA_LIST = re.compile(r"^[^,()\[\]{}]+(?:,\s*[^,()\[\]{}]+)+$")
RE_THOUSANDS_SEP = re.compile(r",\d{3}(?:\D|$)")
RE_BASE_SUBSCRIPT = re.compile(
    r"^(?P<digits>[0-9A-Fa-f]+)_(?:\{)?(?P<base>[0-9]+)(?:\})?$"
)
RE_SYMPY_STYLE = re.compile(r"(?:sqrt|log|sin|cos|tan|exp)\s*\(|(?<!\^)\*\*")

SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁽": "(",
    "⁾": ")",
}
RE_SUPERSCRIPT_CHARS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")


def _convert_superscripts(text: str) -> str:
    if not text or not RE_SUPERSCRIPT_CHARS.search(text):
        return text

    def convert(s, base=None):
        converted = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
        return f"{base}**{converted}" if base else converted

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert(m.group(2), base=m.group(1)),
        text,
    )
    text = convert(text)
    return text


def _strip_left_right(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    return text


def _strip_text_mbox_and_exponents(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\text\s*\{[^}]*\}\s*\^\s*(\{[^}]*\}|\w+)", "", text)
    text = re.sub(r"\\mbox\s*\{[^}]*\}\s*\^\s*(\{[^}]*\}|\w+)", "", text)
    text = re.sub(r"\\text\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\mbox\s*\{[^}]*\}", "", text)
    return text


def _insert_implicit_multiplication(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\sqrt\s*([0-9]+)", r"\\sqrt{\1}", text)
    text = re.sub(r"(\d)\s*(\\sqrt)", r"\1\\cdot\2", text)
    text = re.sub(r"([\)\}])\s*(\\sqrt)", r"\1\\cdot\2", text)
    text = re.sub(r"(\d)\s*(\\pi)", r"\1\\cdot\2", text)
    text = re.sub(r"([\)\}])\s*(\\pi)", r"\1\\cdot\2", text)
    return text


def _preprocess(text: str) -> str:
    text = text.strip().strip("$")
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
    text = _convert_superscripts(text)
    m = RE_LEADING_DOT.match(text)
    if m:
        text = "0" + text
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    text = _strip_text_mbox_and_exponents(text)
    text = _insert_implicit_multiplication(text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = text.replace("°", "")
    return text


def _parse_single_expr(text: str):
    if not text:
        return None
    text = _preprocess(text)
    if not text:
        return None
    try:
        return _parse_latex(text)
    except Exception:
        return None


def _parse_text_answer(text: str):
    m = RE_TEXT.match(text.strip())
    if m:
        return m.group(1).strip()
    m = RE_MBOX.match(text.strip())
    if m:
        return m.group(1).strip()
    return None


def _parse_currency(text: str):
    m = RE_CURRENCY.match(text.strip())
    if m:
        num_str = re.sub(r"\\[,!]|\s|,", "", m.group(1))
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def _parse_interval_endpoint(text: str):
    text = text.strip()
    if text in (r"\infty", r"+\infty"):
        return oo
    if text == r"-\infty":
        return -oo
    return _parse_single_expr(text)


def _parse_interval(text: str):
    text = _strip_left_right(text.strip())
    m = RE_INTERVAL_BOUNDS.match(text)
    if not m:
        return None
    left_brace, left_val, right_val, right_brace = m.groups()
    a = _parse_interval_endpoint(left_val)
    b = _parse_interval_endpoint(right_val)
    if a is None or b is None:
        return None
    try:
        return Interval(
            a, b, left_open=(left_brace == "("), right_open=(right_brace == ")")
        )
    except Exception:
        return None


def _parse_union(text: str):
    parts = RE_UNION.split(text)
    if len(parts) < 2:
        return None
    intervals = []
    for p in parts:
        iv = _parse_interval(p.strip())
        if iv is None:
            return None
        intervals.append(iv)
    try:
        return Union(*intervals)
    except Exception:
        return None


def _parse_matrix(text: str):
    m = RE_MATRIX.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    rows_raw = re.split(r"\\\\", inner)
    rows = []
    for row in rows_raw:
        row = row.strip()
        if not row:
            continue
        cols = [c.strip() for c in row.split("&")]
        parsed_cols = []
        for c in cols:
            expr = _parse_single_expr(c)
            if expr is None:
                return None
            parsed_cols.append(expr)
        rows.append(parsed_cols)
    if not rows:
        return None
    try:
        return Matrix(rows)
    except Exception:
        return None


def _parse_tuple_or_list(text: str):
    text = _strip_left_right(text.strip())
    if not text or text[0] not in "([{" or text[-1] not in ")]}":
        return None
    inner = text[1:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) < 2:
        return None
    parsed = []
    for p in parts:
        expr = _parse_single_expr(p)
        if expr is None:
            return None
        parsed.append(expr)
    return tuple(parsed)


def _parse_set_braces(text: str):
    m = re.match(r"^\\\{(.+)\\\}$", text.strip())
    if not m:
        return None
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",")]
    parsed = []
    for p in parts:
        if r"\pm" in p:
            m_pm = re.match(r"^(?P<a>.*)\\pm(?P<b>.*)$", p)
            if not m_pm:
                return None
            a, b = m_pm.group("a").strip(), m_pm.group("b").strip()
            for op in ("+", "-"):
                expr = _parse_single_expr(f"{a} {op} {b}")
                if expr is None:
                    return None
                parsed.append(expr)
        else:
            expr = _parse_single_expr(p)
            if expr is None:
                return None
            parsed.append(expr)
    try:
        return FiniteSet(*parsed)
    except Exception:
        return None


def _parse_plus_minus(text: str):
    m = RE_PM.match(text.strip())
    if not m:
        return None
    a = _parse_single_expr(m.group("a").strip())
    b = _parse_single_expr(m.group("b").strip())
    if a is None or b is None:
        return None
    try:
        return FiniteSet(a + b, a - b)
    except Exception:
        return None


def _parse_base_subscript(text: str):
    m = RE_BASE_SUBSCRIPT.match(text.strip())
    if not m:
        return None
    digits, base_str = m.group("digits"), m.group("base")
    try:
        base = int(base_str)
        if base < 2 or base > 36:
            return None
        int(digits, base)
        return f"{digits}_{base}"
    except ValueError:
        return None


def _parse_bare_comma_list(text: str):
    text = text.strip()
    if not RE_BARE_COMMA_LIST.match(text):
        return None
    clean = re.sub(r"\\[,!]|\s", "", text)
    if RE_THOUSANDS_SEP.search(clean):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 2:
        return None
    parsed = []
    for p in parts:
        if not p:
            return None
        expr = _parse_single_expr(p)
        if expr is None:
            return None
        parsed.append(expr)
    return tuple(parsed)


def _parse_membership(text: str):
    m = RE_MEMBERSHIP.match(text.strip())
    if not m:
        return None
    lhs_raw, rhs_raw = m.group("lhs").strip(), m.group("rhs").strip()
    lhs = _parse_single_expr(lhs_raw) or (
        Symbol(lhs_raw) if re.fullmatch(r"[A-Za-z]", lhs_raw) else None
    )
    if lhs is None:
        return None
    rhs = (
        _parse_interval(rhs_raw)
        or _parse_set_braces(rhs_raw)
        or _parse_single_expr(rhs_raw)
    )
    if rhs is None:
        return None
    try:
        return Contains(lhs, rhs)
    except Exception:
        return None


def sympy_parser_hybrid(answer: str):
    raw = answer.strip()
    if not raw:
        return None
    text = _strip_left_right(raw)

    if r"\in" in text:
        result = _parse_membership(text)
        if result is not None:
            return result

    if r"\pm" in text:
        result = _parse_plus_minus(text)
        if result is not None:
            return result

    if RE_TEXT.match(raw) or RE_MBOX.match(raw):
        return _parse_text_answer(raw)

    if raw.startswith(r"\$"):
        result = _parse_currency(raw)
        if result is not None:
            return result

    if r"\begin{" in raw and "matrix" in raw:
        return _parse_matrix(raw)

    if r"\cup" in text:
        return _parse_union(text)

    if RE_INFTY.search(text) or (
        RE_TUPLE_LIKE.match(text) and ("]" in text or "[" in text)
    ):
        result = _parse_interval(text)
        if result is not None:
            return result

    if text.startswith(r"\{") and text.endswith(r"\}"):
        result = _parse_set_braces(text)
        if result is not None:
            return result

    if RE_TUPLE_LIKE.match(text):
        result = _parse_tuple_or_list(text)
        if result is not None:
            return result

    result = _parse_base_subscript(text)
    if result is not None:
        return result

    result = _parse_bare_comma_list(text)
    if result is not None:
        return result

    preprocessed = _preprocess(text)
    if not preprocessed:
        return None

    is_sympy_style = RE_SYMPY_STYLE.search(preprocessed) and "\\" not in preprocessed
    if is_sympy_style:
        try:
            return sympify(preprocessed)
        except Exception:
            pass

    try:
        return _parse_latex(preprocessed)
    except Exception:
        try:
            return sympify(preprocessed)
        except Exception:
            return None


def normalize_text_hybrid(answer: str) -> str:
    if not answer:
        return ""

    expr = sympy_parser_hybrid(answer)

    if expr is None:
        return answer.strip()

    if isinstance(expr, str):
        return expr

    return str(expr)
