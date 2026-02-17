# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import warnings

import pytest

import reasoning_from_scratch.bonus.parser as bonus_parser
import reasoning_from_scratch.ch03 as ch03

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    HAS_LATEX_BACKEND = bonus_parser.normalize_text_hybrid(r"\frac{1}{2}") == "1/2"


def current_parser_output(answer: str) -> str:
    normalized = ch03.normalize_text(answer)
    parsed = ch03.sympy_parser(normalized)
    if parsed is None:
        return normalized
    return str(parsed)


def test_bonus_module_direct_import():
    from reasoning_from_scratch.bonus.parser import (
        normalize_text_hybrid,
        sympy_parser_hybrid,
    )

    assert normalize_text_hybrid(r"52_8") == "52_8"
    assert sympy_parser_hybrid(r"52_8") == "52_8"


@pytest.mark.parametrize(
    "answer, expected",
    [
        (r"\text{Evelyn}", "Evelyn"),
        (r"52_8", "52_8"),
        (r"\$32,\!348", "32348.0"),
    ],
)
def test_hybrid_parser_issue_cases_without_latex_backend(answer, expected):
    assert bonus_parser.normalize_text_hybrid(answer) == expected


@pytest.mark.parametrize(
    "answer, current_expected, hybrid_expected",
    [
        (r"52_8", "528", "52_8"),
        (r"\text{(C)}", "c", "(C)"),
    ],
)
def test_current_vs_hybrid_divergent_cases_without_latex_backend(
    answer,
    current_expected,
    hybrid_expected,
):
    assert current_parser_output(answer) == current_expected
    assert bonus_parser.normalize_text_hybrid(answer) == hybrid_expected


@pytest.mark.skipif(
    not HAS_LATEX_BACKEND,
    reason="SymPy LaTeX parser backend is unavailable in this environment.",
)
@pytest.mark.parametrize(
    "answer, current_expected, hybrid_expected",
    [
        (r"11,\! 111,\! 111,\! 100", "(11, 111, 111, 100)", "11111111100"),
        (r"11\sqrt2", r"11\sqrt2", "11*sqrt(2)"),
        (
            r"(0,9) \cup (9,36)",
            r"(0,9) \cup (9,36)",
            "Union(Interval.open(0, 9), Interval.open(9, 36))",
        ),
        (r"x=5", "x=5", "Eq(x, 5)"),
        (r"\pi", r"\pi", "pi"),
        (r"\frac{14}{3}", "14/3", "14/3"),
    ],
)
def test_current_vs_hybrid_selected_issue_examples_with_latex_backend(
    answer,
    current_expected,
    hybrid_expected,
):
    assert current_parser_output(answer) == current_expected
    assert bonus_parser.normalize_text_hybrid(answer) == hybrid_expected
