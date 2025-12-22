# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path
import json
import sympy as sp
import torch
import reasoning_from_scratch.ch03 as ch03


class DummyTokenizer:
    eos_token_id = 0
    _map = {1: "A", 2: "B"}

    def encode(self, text):
        return [123]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._map.get(i, "?") for i in ids)


def test_load_math500_test_has_500_entries():
    repo_root = Path(__file__).resolve().parent.parent
    local_path = repo_root / "math500_test.json"

    data = ch03.load_math500_test(local_path=local_path, save_copy=False)

    assert len(data) == 500


def test_generate_text_stream_concat(monkeypatch):
    # Stub the underlying streaming generator to avoid any real compute
    def fake_stream(**kwargs):
        for t in (1, 2, 1):
            yield torch.tensor([t])  # matches squeeze(0) in the function

    monkeypatch.setattr(ch03, "generate_text_basic_stream_cache", fake_stream)

    tok = DummyTokenizer()
    out = ch03.generate_text_stream_concat(
        model=None,
        tokenizer=tok,
        prompt="ignored",
        device="cpu",
        max_new_tokens=3,
        verbose=False,
    )

    assert out == "ABA"


def test_get_last_boxed():

    cases = [
        (r"foo \boxed{42}", "42"),
        (r"\boxed{a} bla \boxed{b+c}", "b+c"),   # picks last
        (r"noise \boxed   {  x^2 } end", "  x^2 "),  # allows spaces
        (r"\boxed{outer {inner} ok}", "outer {inner} ok"),  # nesting
        (r"nothing here", None),  # missing boxed
        (r"\boxed  not_brace", None),  # no opening {
        (r"\boxed{unbalanced", None),  # unbalanced
    ]

    for i, (text, expected) in enumerate(cases, 1):
        got = ch03.get_last_boxed(text)
        assert got == expected, f"case {i} failed: {text!r} -> {got!r}, expected {expected!r}"


def test_extract_final_candidate():

    cases = [
        (r"Steps...\n\boxed{3/4}\nDone.", "3/4"),
        (r"Reasoning...\n\boxed{\frac{3}{4}}", r"\frac{3}{4}"),
        (r"Compute...\n\boxed{\sqrt{2}}", r"\sqrt{2}"),
        (r"Tuple case...\n\boxed{(1,2)}", "(1,2)"),
        (r"Earlier box \boxed{1/3}, later \boxed{2/3}", "2/3"),
        (r"Noisy \boxed   {  x^2 } trailing text", "x^2"),
        (r"Nested braces \boxed{outer {inner} ok}", "outer {inner} ok"),
        (r"In math mode: $ \boxed{ \dfrac{14}{3} } $", r"\dfrac{14}{3}"),

        # Fallbacks without \boxed{...}
        ("Some steps...\nFinal Answer: 14/3", "14/3"),
        ("All done. 1 \nFinal 2 answer: ", "2"),
    ]

    for i, (generated, expected) in enumerate(cases, 1):
        got = ch03.extract_final_candidate(generated)
        assert got == expected, f"case {i} failed: {generated!r} -> {got!r}, expected {expected!r}"


def test_normalize():
    cases = [
        # Basic whitespace trimming
        ("  3/4  ", "3/4"),
        ("\n\t  (1, 2)  \t", "(1, 2)"),

        # LaTeX math mode and spacing
        ("$2/3$", "2/3"),
        (r"\( 2/3 \)", "2/3"),
        (r"\left(1,\,2\right)", "(1,2)"),

        # Fractions
        (r"\frac{3}{4}", "(3)/(4)"),
        (r"\dfrac{14}{3}", "(14)/(3)"),
        (r"(3)/(4)", "(3)/(4)"),

        # Multiple choice labels
        ("c. 3", "3"),
        ("b: 2", "2"),

        # Roots (don’t simplify math here but just normalize text)
        (r"\sqrt{2}", "sqrt(2)"),

        # Braces removal
        (r"{x}", "x"),
        (r"{ (1, 2) }", "(1, 2)"),
    ]

    for i, (raw, expected) in enumerate(cases, 1):
        got = ch03.normalize_text(raw)
        assert got == expected, f"case {i} failed: {raw!r} -> {got!r}, expected {expected!r}"


def test_sympy_parser():

    x, y = sp.symbols("x y")

    success_cases = [
        ("2+3", sp.Integer(5)),
        ("14/3", sp.Rational(14, 3)),
        ("sqrt(8)/2", sp.sqrt(8) / 2),
        ("2*x + 3*x", 5*x),
        ("2x + 3x", 5*x),
        ("3(1+2)", sp.Integer(9)),
        ("x(1+y)", x*(1+y)),
        ("pi*x", sp.pi*x),
    ]

    for i, (expr, expected) in enumerate(success_cases, 1):
        got = ch03.sympy_parser(expr)
        assert got is not None, f"case {i} produced None for {expr!r}"
        # Numeric/symbolic equivalence
        assert sp.simplify(got - expected) == 0, f"{expr!r}: {got!r} != {expected!r}"

    long_expr = "1" * 2001
    failure_cases = [
        "sqrt(",  # unbalanced
        "??",  # invalid tokens
        "2**",  # incomplete operator
        None,  # guard against missing inputs
        long_expr,  # guard against overly long inputs
    ]
    for expr in failure_cases:
        assert ch03.sympy_parser(expr) is None, f"expected None for {expr!r}"


def test_equality_check():

    cases = [
        ("13/4.", r"(13)/(4)", True),
        ("0.5", r"(1)/(2)", True),
        ("14/3", "15/3", False),
        ("(14/3, 2/3)", "(14/3, 4/6)", False),
    ]

    for pred, truth, expected in cases:
        got = ch03.equality_check(ch03.normalize_text(pred), ch03.normalize_text(truth))
        assert got == expected, f"{pred!r} -> {got!r}, expected {expected!r}"


def test_split_into_parts():
    cases = [
        ("(a, b)", ["a", "b"]),
        ("[x,y,z]", ["x", "y", "z"]),
        ("( 1 ,  2 )", ["1", "2"]),
        ("hello", ["hello"]),
        ("(a)", ["(a)"]),
        ("(a, )", ["(a, )"]),
        ("", []),
    ]

    for raw, expected in cases:
        got = ch03.split_into_parts(raw)
        assert got == expected, f"{raw!r} -> {got!r}, expected {expected!r}"


def test_grade_answer():
    cases = [
        ("3/4", r"\frac{3}{4}", True),
        ("(3)/(4)", r"3/4", True),
        (r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
        (r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
        ("(1, 2)", r"(1,2)", True),
        ("(2, 1)", "(1, 2)", False),
        ("(1, 2, 3)", "(1, 2)", False),
        ("0.5", "1/2", True),
        ("0.3333333333", "1/3", False),
        ("1,234/2", "617", True),
        (r"\text{2/3}", "2/3", True),
        ("50%", "1/2", False),
        (r"2\cdot 3/4", "3/2", True),
        (r"90^\circ", "90", True),
        (r"\left(\frac{3}{4}\right)", "3/4", True),
        ("(1,2,3)", "(1,2,3)", True),
        ("(1,2,3)", "(1,2)", False),
        ("", "1/2", False),
        (None, "1/2", False),
        ("1/2", None, False),
    ]

    for i, (pred, gt, expected) in enumerate(cases, 1):
        got = ch03.grade_answer(pred, gt)
        assert got == expected, f"case {i} failed: pred={pred!r}, gt={gt!r} -> {got!r}, expected {expected!r}"


tests = [
    ("check_1", "3/4", r"\frac{3}{4}", True),
    ("check_2", "(3)/(4)", r"3/4", True),
    ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
    ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
    ("check_5", "(1, 2)", r"(1,2)", True),
    ("check_6", "(2, 1)", "(1, 2)", False),
    ("check_7", "(1, 2, 3)", "(1, 2)", False),
    ("check_8", "0.5", "1/2", True),
    ("check_9", "0.3333333333", "1/3", False),
    ("check_10", "1,234/2", "617", True),
    ("check_11", r"\text{2/3}", "2/3", True),
    ("check_12", "50%", "1/2", False),
    ("check_13", r"2\cdot 3/4", "3/2", True),
    ("check_14", r"90^\circ", "90", True),
    ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
]


def test_run_demos_table_pass_fail_labels(capsys):
    ch03.run_demos_table(tests)
    out = capsys.readouterr().out

    rows = {}
    for line in out.splitlines():
        if not line.strip() or line.startswith("Test"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        name, expect_str, got_str, status = parts
        rows[name] = (expect_str, got_str, status)

    for name, *_ in tests:
        assert name in rows, f"Row for {name} not found.\n{out}"
        expect_str, got_str, status = rows[name]
        should_pass = (expect_str == got_str)
        expected_label = "PASS" if should_pass else "FAIL"
        assert status == expected_label, (
            f"{name}: expected {expected_label} but saw {status}\n{out}"
        )


def test_render_prompt():
    cases = [
        "Compute 1/2 + 1/6.",
        "Find x.\nGiven y.",
        r"Simplify $\frac{1}{2}$.",
    ]

    for p in cases:
        got = ch03.render_prompt(p)
        expected = (
            "You are a helpful math assistant.\n"
            "Answer the question and write the final result on a new line as:\n"
            "\\boxed{ANSWER}\n\n"
            f"Question:\n{p}\n\nAnswer:"
        )
        assert got == expected, f"mismatch for prompt {p!r}"
        assert got.count("\\boxed{ANSWER}") == 1
        assert got.endswith("Answer:")


def test_evaluate_math500_stream(tmp_path, monkeypatch, qwen3_weights_path):

    outputs = iter([
        "Reasoning...\n\\boxed{A}",
        "Some steps...\nTherefore, \\boxed{X}",
    ])
    monkeypatch.setattr(
        ch03,
        "generate_text_stream_concat",
        lambda *a, **k: next(outputs),
    )

    math_data = [
        {"problem": "Compute #1", "answer": "A"},
        {"problem": "Compute #2", "answer": "B"},
    ]

    tokenizer = ch03.load_tokenizer_only(
        which_model="base",
        local_dir=qwen3_weights_path,
    )

    out_path = tmp_path / "math500-test.jsonl"
    num_correct, num_examples, acc = ch03.evaluate_math500_stream(
        model=None,
        tokenizer=tokenizer,
        device="cpu",
        math_data=math_data,
        out_path=out_path,
        max_new_tokens=8,
        verbose=False,
    )

    assert (num_correct, num_examples) == (1, 2)
    assert abs(acc - 0.5) < 1e-9

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])

    assert rec1["index"] == 1 and rec2["index"] == 2
    assert rec1["problem"] == "Compute #1"
    assert rec2["problem"] == "Compute #2"
    assert rec1["gtruth_answer"] == "A"
    assert rec2["gtruth_answer"] == "B"

    assert "\\boxed{A}" in rec1["generated_text"]
    assert "\\boxed{X}" in rec2["generated_text"]
    assert rec1["extracted"] == "A" and rec1["correct"] is True
    assert rec2["extracted"] == "X" and rec2["correct"] is False
