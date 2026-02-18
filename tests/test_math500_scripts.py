# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATHS = [
    Path("ch03/02_math500-verifier-scripts/evaluate_math500_batched.py"),
    Path("ch03/02_math500-verifier-scripts/evaluate_math500.py"),
    Path("ch03/02_math500-verifier-scripts/evaluate_json.py"),
    Path("ch04/02_math500-inference-scaling-scripts/self_consistency_math500.py"),
    Path("ch04/02_math500-inference-scaling-scripts/cot_prompting_math500.py"),
]


@pytest.mark.parametrize("script_path", SCRIPT_PATHS)
def test_script_help_runs_without_import_errors(script_path):

    repo_root = Path(__file__).resolve().parent.parent
    full_path = repo_root / script_path
    assert full_path.exists(), f"Expected script at {full_path}"

    # Run scripts with --help to make sure they work

    result = subprocess.run(
        [sys.executable, str(full_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_evaluate_json_script_jsonl(tmp_path):

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "ch03/02_math500-verifier-scripts/evaluate_json.py"
    assert script_path.exists(), f"Expected script at {script_path}"

    records_path = tmp_path / "records.jsonl"
    records = [
        {
            "gtruth_answer": "2/3",
            "generated_text": "Let's solve this.\n\\boxed{2/3}",
        },
        {
            "gtruth_answer": "5",
            "generated_text": "Final answer is \\boxed{4}",
        },
    ]
    records_path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script_path), "--json_path", str(records_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Accuracy: 50.0% (1/2)" in result.stdout


def test_evaluate_json_script_custom_keys(tmp_path):

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "ch03/02_math500-verifier-scripts/evaluate_json.py"
    assert script_path.exists(), f"Expected script at {script_path}"

    records_path = tmp_path / "records.json"
    records = [
        {"answer_key": "7", "model_output": "Answer: \\boxed{7}"},
        {"answer_key": "1/4", "model_output": "Answer: \\boxed{3/4}"},
    ]
    records_path.write_text(json.dumps(records), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--json_path",
            str(records_path),
            "--gtruth_answer",
            "answer_key",
            "--generated_text",
            "model_output",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Accuracy: 50.0% (1/2)" in result.stdout
