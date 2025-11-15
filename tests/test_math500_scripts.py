# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATHS = [
    Path("ch03/02_math500-verifier-scripts/evaluate_math500_batched.py"),
    Path("ch03/02_math500-verifier-scripts/evaluate_math500.py"),
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
