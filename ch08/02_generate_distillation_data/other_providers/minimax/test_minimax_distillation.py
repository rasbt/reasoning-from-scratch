# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

"""Tests for the MiniMax distillation generation script."""

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib import error

import pytest


SCRIPT_PATH = Path(__file__).resolve().with_name("generate_with_minimax.py")
REPO_ROOT = SCRIPT_PATH.parents[4]


def load_minimax_module():
    """Import the MiniMax generation script as a module."""
    spec = importlib.util.spec_from_file_location("generate_with_minimax", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Prevent __main__ block from executing
    mod.__name__ = "generate_with_minimax"
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def minimax_mod():
    return load_minimax_module()


# ---------------------------------------------------------------------------
# Unit tests for clamp_temperature
# ---------------------------------------------------------------------------


class TestClampTemperature:
    def test_normal_value(self, minimax_mod):
        assert minimax_mod.clamp_temperature(0.7) == 0.7

    def test_zero_clamped(self, minimax_mod):
        assert minimax_mod.clamp_temperature(0.0) == 0.01

    def test_negative_clamped(self, minimax_mod):
        assert minimax_mod.clamp_temperature(-1.0) == 0.01

    def test_above_one_clamped(self, minimax_mod):
        assert minimax_mod.clamp_temperature(1.5) == 1.0

    def test_exactly_one(self, minimax_mod):
        assert minimax_mod.clamp_temperature(1.0) == 1.0

    def test_small_positive(self, minimax_mod):
        assert minimax_mod.clamp_temperature(0.01) == 0.01


# ---------------------------------------------------------------------------
# Unit tests for parse_minimax_response
# ---------------------------------------------------------------------------


class TestParseMiniMaxResponse:
    def test_standard_response(self, minimax_mod):
        decoded = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think about this...",
                    }
                }
            ]
        }
        result = minimax_mod.parse_minimax_response(decoded)
        assert result["message_content"] == "The answer is 42."
        assert result["message_thinking"] == "Let me think about this..."

    def test_no_reasoning_content(self, minimax_mod):
        decoded = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                    }
                }
            ]
        }
        result = minimax_mod.parse_minimax_response(decoded)
        assert result["message_content"] == "The answer is 42."
        assert result["message_thinking"] == ""

    def test_thinking_only_fallback(self, minimax_mod):
        decoded = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Thinking only response...",
                    }
                }
            ]
        }
        result = minimax_mod.parse_minimax_response(decoded)
        assert result["message_content"] == "Thinking only response..."
        assert result["message_thinking"] == "Thinking only response..."

    def test_missing_choices_raises(self, minimax_mod):
        with pytest.raises(RuntimeError, match="missing choices"):
            minimax_mod.parse_minimax_response({})

    def test_empty_choices_raises(self, minimax_mod):
        with pytest.raises(RuntimeError, match="missing choices"):
            minimax_mod.parse_minimax_response({"choices": []})

    def test_invalid_choice_format_raises(self, minimax_mod):
        with pytest.raises(RuntimeError, match="invalid choices format"):
            minimax_mod.parse_minimax_response({"choices": ["not a dict"]})

    def test_no_content_raises(self, minimax_mod):
        decoded = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                    }
                }
            ]
        }
        with pytest.raises(RuntimeError, match="did not contain"):
            minimax_mod.parse_minimax_response(decoded)

    def test_message_as_string(self, minimax_mod):
        decoded = {
            "choices": [
                {
                    "message": "Plain string message",
                }
            ]
        }
        result = minimax_mod.parse_minimax_response(decoded)
        assert result["message_content"] == "Plain string message"
        assert result["message_thinking"] == ""


# ---------------------------------------------------------------------------
# Unit tests for render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_default_prompt(self, minimax_mod):
        result = minimax_mod.render_prompt("What is 2+2?")
        assert "What is 2+2?" in result
        assert "\\boxed{ANSWER}" in result
        assert "short explanation" not in result

    def test_shorter_prompt(self, minimax_mod):
        result = minimax_mod.render_prompt("What is 2+2?", shorter_answers_prompt=True)
        assert "What is 2+2?" in result
        assert "short explanation" in result


# ---------------------------------------------------------------------------
# Unit tests for model_to_filename
# ---------------------------------------------------------------------------


class TestModelToFilename:
    def test_standard_model(self, minimax_mod):
        assert minimax_mod.model_to_filename("MiniMax-M2.7") == "math500_minimax_m2_7_full_answers.json"

    def test_highspeed_model(self, minimax_mod):
        result = minimax_mod.model_to_filename("MiniMax-M2.7-highspeed")
        assert result == "math500_minimax_m2_7_highspeed_full_answers.json"

    def test_empty_model(self, minimax_mod):
        assert minimax_mod.model_to_filename("") == "math500_model_full_answers.json"


# ---------------------------------------------------------------------------
# Unit tests for write_rows_json_incremental
# ---------------------------------------------------------------------------


class TestWriteRowsJsonIncremental:
    def test_write_and_read(self, minimax_mod, tmp_path):
        out_file = tmp_path / "output.json"
        rows = [{"problem": "1+1", "answer": "2"}]
        minimax_mod.write_rows_json_incremental(rows, out_file)
        assert out_file.exists()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded == rows

    def test_incremental_append(self, minimax_mod, tmp_path):
        out_file = tmp_path / "output.json"
        rows = [{"problem": "1+1"}]
        minimax_mod.write_rows_json_incremental(rows, out_file)
        rows.append({"problem": "2+2"})
        minimax_mod.write_rows_json_incremental(rows, out_file)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Unit tests for load_resume_rows
# ---------------------------------------------------------------------------


class TestLoadResumeRows:
    def test_load_list(self, minimax_mod, tmp_path):
        out_file = tmp_path / "resume.json"
        data = [{"problem": "x"}]
        out_file.write_text(json.dumps(data), encoding="utf-8")
        result = minimax_mod.load_resume_rows(out_file)
        assert result == data

    def test_load_records_dict(self, minimax_mod, tmp_path):
        out_file = tmp_path / "resume.json"
        data = {"records": [{"problem": "x"}]}
        out_file.write_text(json.dumps(data), encoding="utf-8")
        result = minimax_mod.load_resume_rows(out_file)
        assert result == [{"problem": "x"}]

    def test_invalid_format_raises(self, minimax_mod, tmp_path):
        out_file = tmp_path / "resume.json"
        out_file.write_text(json.dumps({"bad": "data"}), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            minimax_mod.load_resume_rows(out_file)


# ---------------------------------------------------------------------------
# Unit tests for validate_resume_rows
# ---------------------------------------------------------------------------


class TestValidateResumeRows:
    def test_valid_resume(self, minimax_mod):
        rows = [{"problem": "1+1"}]
        selected_data = [{"problem": "1+1"}, {"problem": "2+2"}]
        minimax_mod.validate_resume_rows(rows, selected_data)

    def test_too_many_rows(self, minimax_mod):
        rows = [{"problem": "1+1"}, {"problem": "2+2"}, {"problem": "3+3"}]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(ValueError, match="dataset has only"):
            minimax_mod.validate_resume_rows(rows, selected_data)

    def test_mismatch_problem(self, minimax_mod):
        rows = [{"problem": "wrong"}]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(ValueError, match="does not match"):
            minimax_mod.validate_resume_rows(rows, selected_data)

    def test_missing_problem_key(self, minimax_mod):
        rows = [{"answer": "2"}]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(KeyError, match="problem"):
            minimax_mod.validate_resume_rows(rows, selected_data)

    def test_non_dict_row(self, minimax_mod):
        rows = ["not a dict"]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(ValueError, match="not a JSON object"):
            minimax_mod.validate_resume_rows(rows, selected_data)


# ---------------------------------------------------------------------------
# Unit test for script --help
# ---------------------------------------------------------------------------


def test_script_help_runs_without_errors():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
    assert "MINIMAX_API_KEY" in result.stdout or "minimax" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Integration tests (mocked HTTP, no real API calls)
# ---------------------------------------------------------------------------


class TestQueryMiniMaxChat:
    def test_successful_query(self, minimax_mod):
        mock_response_body = json.dumps({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "OK",
                        "reasoning_content": "",
                    }
                }
            ]
        }).encode("utf-8")

        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_body
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.object(minimax_mod.request, "urlopen", return_value=mock_response):
            result = minimax_mod.query_minimax_chat(
                prompt="Reply with OK.",
                model="MiniMax-M2.7",
                api_key="test-key",
                max_new_tokens=8,
                temperature=0.7,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
            )
        assert result["message_content"] == "OK"

    def test_retries_on_http_error(self, minimax_mod):
        mock_exc = error.HTTPError(
            url=minimax_mod.MINIMAX_CHAT_URL,
            code=500,
            msg="Server Error",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b"error")),
        )

        with patch.object(minimax_mod.request, "urlopen", side_effect=mock_exc):
            with pytest.raises(RuntimeError, match="Failed to query MiniMax"):
                minimax_mod.query_minimax_chat(
                    prompt="test",
                    model="MiniMax-M2.7",
                    api_key="test-key",
                    max_new_tokens=8,
                    temperature=0.7,
                    top_p=1.0,
                    timeout=30,
                    max_retries=2,
                    retry_delay=0.0,
                )

    def test_temperature_clamping_in_query(self, minimax_mod):
        """Verify that temperature is clamped before being sent."""
        captured_payloads = []

        mock_response_body = json.dumps({
            "choices": [{"message": {"content": "OK"}}]
        }).encode("utf-8")

        def capture_urlopen(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            captured_payloads.append(payload)
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_response_body
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        with patch.object(minimax_mod.request, "urlopen", side_effect=capture_urlopen):
            minimax_mod.query_minimax_chat(
                prompt="test",
                model="MiniMax-M2.7",
                api_key="test-key",
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
            )

        assert len(captured_payloads) == 1
        assert captured_payloads[0]["temperature"] == 0.01


class TestGenerateRow:
    def test_generate_row_returns_expected_format(self, minimax_mod):
        mock_response_body = json.dumps({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "\\boxed{42}",
                        "reasoning_content": "Thinking...",
                    }
                }
            ]
        }).encode("utf-8")

        mock_response = MagicMock()
        mock_response.read.return_value = mock_response_body
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        row = {"problem": "What is 6*7?", "answer": "42"}

        with patch.object(minimax_mod.request, "urlopen", return_value=mock_response):
            result = minimax_mod.generate_row(
                row=row,
                shorter_answers_prompt=False,
                model="MiniMax-M2.7",
                api_key="test-key",
                max_new_tokens=2048,
                temperature=0.7,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
            )

        assert result["problem"] == "What is 6*7?"
        assert result["gtruth_answer"] == "42"
        assert result["message_content"] == "\\boxed{42}"
        assert result["message_thinking"] == "Thinking..."


# ---------------------------------------------------------------------------
# Integration test: real MiniMax API (skipped unless MINIMAX_API_KEY is set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not sys.modules.get("os", __import__("os")).environ.get("MINIMAX_API_KEY"),
    reason="Set MINIMAX_API_KEY to run real MiniMax API integration tests",
)
class TestRealMiniMaxAPI:
    def test_real_query(self, minimax_mod):
        import os
        api_key = os.environ["MINIMAX_API_KEY"]
        result = minimax_mod.query_minimax_chat(
            prompt="What is 2+2? Reply with just the number.",
            model="MiniMax-M2.7",
            api_key=api_key,
            max_new_tokens=64,
            temperature=0.7,
            top_p=1.0,
            timeout=60,
            max_retries=2,
            retry_delay=2.0,
        )
        assert result["message_content"]
        assert "4" in result["message_content"]

    def test_real_query_highspeed(self, minimax_mod):
        import os
        api_key = os.environ["MINIMAX_API_KEY"]
        result = minimax_mod.query_minimax_chat(
            prompt="What is 3+5? Reply with just the number.",
            model="MiniMax-M2.7-highspeed",
            api_key=api_key,
            max_new_tokens=64,
            temperature=0.7,
            top_p=1.0,
            timeout=60,
            max_retries=2,
            retry_delay=2.0,
        )
        assert result["message_content"]
        assert "8" in result["message_content"]

    def test_real_generate_row(self, minimax_mod):
        import os
        api_key = os.environ["MINIMAX_API_KEY"]
        row = {"problem": "What is 10+5?", "answer": "15"}
        result = minimax_mod.generate_row(
            row=row,
            shorter_answers_prompt=False,
            model="MiniMax-M2.7-highspeed",
            api_key=api_key,
            max_new_tokens=256,
            temperature=0.7,
            top_p=1.0,
            timeout=60,
            max_retries=2,
            retry_delay=2.0,
        )
        assert result["problem"] == "What is 10+5?"
        assert result["gtruth_answer"] == "15"
        assert result["message_content"]
