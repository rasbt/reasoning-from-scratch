# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATHS = [
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_3_plus_tracking.py"),
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_4_plus_clip_ratio.py"),
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_5_plus_kl.py"),
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_6_plus_format_reward.py"),
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/gdpo.py"),
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/deepseek_v32_style.py"),
    Path("ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/olmo3_style.py"),
]


@pytest.mark.parametrize("script_path", SCRIPT_PATHS)
def test_ch07_script_help_runs_without_import_errors(script_path):
    repo_root = Path(__file__).resolve().parent.parent
    full_path = repo_root / script_path
    assert full_path.exists(), f"Expected script at {full_path}"

    result = subprocess.run(
        [sys.executable, str(full_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
