# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

"""Tests for the OrcaRouter distillation generation script."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib import error

import pytest


SCRIPT_PATH = Path(__file__).resolve().with_name("generate_with_orcarouter.py")
REPO_ROOT = SCRIPT_PATH.parents[4]


def load_orcarouter_module():
    """Import the OrcaRouter generation script as a module."""
    spec = importlib.util.spec_from_file_location("generate_with_orcarouter", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Prevent __main__ block from executing
    mod.__name__ = "generate_with_orcarouter"
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def orcarouter_mod():
    return load_orcarouter_module()


def make_mock_response(payload):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(payload).encode("utf-8")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


# ---------------------------------------------------------------------------
# Unit tests for extract_thinking
# ---------------------------------------------------------------------------


class TestExtractThinking:
    def test_reasoning_content_key(self, orcarouter_mod):
        # DeepSeek, Qwen, and MiniMax spelling
        message = {"content": "42", "reasoning_content": "thinking..."}
        assert orcarouter_mod.extract_thinking(message) == "thinking..."

    def test_reasoning_key(self, orcarouter_mod):
        # GLM spelling
        message = {"content": "42", "reasoning": "thinking..."}
        assert orcarouter_mod.extract_thinking(message) == "thinking..."

    def test_thinking_key(self, orcarouter_mod):
        message = {"content": "42", "thinking": "thinking..."}
        assert orcarouter_mod.extract_thinking(message) == "thinking..."

    def test_reasoning_content_wins_over_reasoning(self, orcarouter_mod):
        message = {"reasoning_content": "first", "reasoning": "second"}
        assert orcarouter_mod.extract_thinking(message) == "first"

    def test_empty_string_is_skipped(self, orcarouter_mod):
        message = {"reasoning_content": "", "reasoning": "second"}
        assert orcarouter_mod.extract_thinking(message) == "second"

    def test_no_thinking_key(self, orcarouter_mod):
        assert orcarouter_mod.extract_thinking({"content": "42"}) == ""

    def test_non_dict_message(self, orcarouter_mod):
        assert orcarouter_mod.extract_thinking("not a dict") == ""

    def test_non_string_value_is_ignored(self, orcarouter_mod):
        # Some upstreams also return a structured "reasoning_details" list;
        # only plain strings should be used as the thinking stream.
        message = {"reasoning_content": [{"text": "x"}], "reasoning": "second"}
        assert orcarouter_mod.extract_thinking(message) == "second"


# ---------------------------------------------------------------------------
# Unit tests for parse_orcarouter_response
# ---------------------------------------------------------------------------


class TestParseOrcaRouterResponse:
    def test_standard_response(self, orcarouter_mod):
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
        result = orcarouter_mod.parse_orcarouter_response(decoded)
        assert result["message_content"] == "The answer is 42."
        assert result["message_thinking"] == "Let me think about this..."

    def test_glm_style_response(self, orcarouter_mod):
        decoded = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning": "Let me think about this...",
                    }
                }
            ]
        }
        result = orcarouter_mod.parse_orcarouter_response(decoded)
        assert result["message_thinking"] == "Let me think about this..."

    def test_response_without_thinking(self, orcarouter_mod):
        decoded = {"choices": [{"message": {"content": "42"}}]}
        result = orcarouter_mod.parse_orcarouter_response(decoded)
        assert result["message_content"] == "42"
        assert result["message_thinking"] == ""

    def test_thinking_only_response_falls_back_to_content(self, orcarouter_mod):
        decoded = {"choices": [{"message": {"content": "", "reasoning_content": "only thinking"}}]}
        result = orcarouter_mod.parse_orcarouter_response(decoded)
        assert result["message_content"] == "only thinking"
        assert result["message_thinking"] == "only thinking"

    def test_string_message(self, orcarouter_mod):
        decoded = {"choices": [{"message": "plain text"}]}
        result = orcarouter_mod.parse_orcarouter_response(decoded)
        assert result["message_content"] == "plain text"

    def test_missing_choices_raises(self, orcarouter_mod):
        with pytest.raises(RuntimeError, match="missing choices"):
            orcarouter_mod.parse_orcarouter_response({})

    def test_empty_choices_raises(self, orcarouter_mod):
        with pytest.raises(RuntimeError, match="missing choices"):
            orcarouter_mod.parse_orcarouter_response({"choices": []})

    def test_invalid_choice_format_raises(self, orcarouter_mod):
        with pytest.raises(RuntimeError, match="invalid choices format"):
            orcarouter_mod.parse_orcarouter_response({"choices": ["not a dict"]})

    def test_unparseable_content_raises(self, orcarouter_mod):
        with pytest.raises(RuntimeError, match="did not contain parseable"):
            orcarouter_mod.parse_orcarouter_response({"choices": [{"message": {"role": "assistant"}}]})


# ---------------------------------------------------------------------------
# Unit tests for render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_default_prompt(self, orcarouter_mod):
        result = orcarouter_mod.render_prompt("What is 2+2?")
        assert "What is 2+2?" in result
        assert "\\boxed{ANSWER}" in result
        assert "short explanation" not in result

    def test_shorter_prompt(self, orcarouter_mod):
        result = orcarouter_mod.render_prompt("What is 2+2?", shorter_answers_prompt=True)
        assert "What is 2+2?" in result
        assert "short explanation" in result


# ---------------------------------------------------------------------------
# Unit tests for model_to_filename
# ---------------------------------------------------------------------------


class TestModelToFilename:
    def test_namespaced_model(self, orcarouter_mod):
        result = orcarouter_mod.model_to_filename("deepseek/deepseek-reasoner")
        assert result == "math500_deepseek_deepseek_reasoner_full_answers.json"

    def test_versioned_model(self, orcarouter_mod):
        result = orcarouter_mod.model_to_filename("qwen/qwen3.5-plus")
        assert result == "math500_qwen_qwen3_5_plus_full_answers.json"

    def test_router_alias(self, orcarouter_mod):
        result = orcarouter_mod.model_to_filename("orcarouter/auto")
        assert result == "math500_orcarouter_auto_full_answers.json"

    def test_empty_model(self, orcarouter_mod):
        assert orcarouter_mod.model_to_filename("") == "math500_model_full_answers.json"


# ---------------------------------------------------------------------------
# Unit tests for write_rows_json_incremental
# ---------------------------------------------------------------------------


class TestWriteRowsJsonIncremental:
    def test_write_and_read(self, orcarouter_mod, tmp_path):
        out_file = tmp_path / "output.json"
        rows = [{"problem": "1+1", "answer": "2"}]
        orcarouter_mod.write_rows_json_incremental(rows, out_file)
        assert out_file.exists()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded == rows

    def test_incremental_append(self, orcarouter_mod, tmp_path):
        out_file = tmp_path / "output.json"
        rows = [{"problem": "1+1"}]
        orcarouter_mod.write_rows_json_incremental(rows, out_file)
        rows.append({"problem": "2+2"})
        orcarouter_mod.write_rows_json_incremental(rows, out_file)
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Unit tests for load_resume_rows
# ---------------------------------------------------------------------------


class TestLoadResumeRows:
    def test_load_list(self, orcarouter_mod, tmp_path):
        out_file = tmp_path / "resume.json"
        data = [{"problem": "x"}]
        out_file.write_text(json.dumps(data), encoding="utf-8")
        result = orcarouter_mod.load_resume_rows(out_file)
        assert result == data

    def test_load_records_dict(self, orcarouter_mod, tmp_path):
        out_file = tmp_path / "resume.json"
        data = {"records": [{"problem": "x"}]}
        out_file.write_text(json.dumps(data), encoding="utf-8")
        result = orcarouter_mod.load_resume_rows(out_file)
        assert result == [{"problem": "x"}]

    def test_invalid_format_raises(self, orcarouter_mod, tmp_path):
        out_file = tmp_path / "resume.json"
        out_file.write_text(json.dumps({"bad": "data"}), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            orcarouter_mod.load_resume_rows(out_file)


# ---------------------------------------------------------------------------
# Unit tests for validate_resume_rows
# ---------------------------------------------------------------------------


class TestValidateResumeRows:
    def test_valid_resume(self, orcarouter_mod):
        rows = [{"problem": "1+1"}]
        selected_data = [{"problem": "1+1"}, {"problem": "2+2"}]
        orcarouter_mod.validate_resume_rows(rows, selected_data)

    def test_too_many_rows(self, orcarouter_mod):
        rows = [{"problem": "1+1"}, {"problem": "2+2"}, {"problem": "3+3"}]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(ValueError, match="dataset has only"):
            orcarouter_mod.validate_resume_rows(rows, selected_data)

    def test_mismatch_problem(self, orcarouter_mod):
        rows = [{"problem": "wrong"}]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(ValueError, match="does not match"):
            orcarouter_mod.validate_resume_rows(rows, selected_data)

    def test_missing_problem_key(self, orcarouter_mod):
        rows = [{"answer": "2"}]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(KeyError, match="problem"):
            orcarouter_mod.validate_resume_rows(rows, selected_data)

    def test_non_dict_row(self, orcarouter_mod):
        rows = ["not a dict"]
        selected_data = [{"problem": "1+1"}]
        with pytest.raises(ValueError, match="not a JSON object"):
            orcarouter_mod.validate_resume_rows(rows, selected_data)


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
    assert "orcarouter" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Integration tests (mocked HTTP, no real API calls)
# ---------------------------------------------------------------------------


class TestQueryOrcaRouterChat:
    def test_successful_query(self, orcarouter_mod):
        mock_response = make_mock_response({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "OK",
                        "reasoning_content": "",
                    }
                }
            ]
        })

        with patch.object(orcarouter_mod.request, "urlopen", return_value=mock_response):
            result = orcarouter_mod.query_orcarouter_chat(
                prompt="Reply with OK.",
                model="deepseek/deepseek-reasoner",
                api_key="test-key",
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
            )
        assert result["message_content"] == "OK"

    def test_retries_on_http_error(self, orcarouter_mod):
        mock_exc = error.HTTPError(
            url=orcarouter_mod.ORCAROUTER_CHAT_URL,
            code=500,
            msg="Server Error",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b"error")),
        )

        with patch.object(orcarouter_mod.request, "urlopen", side_effect=mock_exc):
            with pytest.raises(RuntimeError, match="Failed to query OrcaRouter"):
                orcarouter_mod.query_orcarouter_chat(
                    prompt="test",
                    model="deepseek/deepseek-reasoner",
                    api_key="test-key",
                    max_new_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                    timeout=30,
                    max_retries=2,
                    retry_delay=0.0,
                )

    def test_request_target_and_headers(self, orcarouter_mod):
        """Verify the endpoint, auth header, and attribution headers."""
        captured_requests = []

        def capture_urlopen(req, timeout=None):
            captured_requests.append(req)
            return make_mock_response({"choices": [{"message": {"content": "OK"}}]})

        with patch.object(orcarouter_mod.request, "urlopen", side_effect=capture_urlopen):
            orcarouter_mod.query_orcarouter_chat(
                prompt="test",
                model="deepseek/deepseek-reasoner",
                api_key="test-key",
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
            )

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.full_url == "https://api.orcarouter.ai/v1/chat/completions"
        # urllib normalizes header names to title case
        assert req.headers["Authorization"] == "Bearer test-key"
        assert req.headers["X-title"] == "reasoning-from-scratch"

    def test_reasoning_effort_omitted_by_default(self, orcarouter_mod):
        captured_payloads = []

        def capture_urlopen(req, timeout=None):
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return make_mock_response({"choices": [{"message": {"content": "OK"}}]})

        with patch.object(orcarouter_mod.request, "urlopen", side_effect=capture_urlopen):
            orcarouter_mod.query_orcarouter_chat(
                prompt="test",
                model="deepseek/deepseek-reasoner",
                api_key="test-key",
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
            )

        assert "reasoning_effort" not in captured_payloads[0]

    def test_reasoning_effort_sent_flat(self, orcarouter_mod):
        """OrcaRouter takes a top-level reasoning_effort, not a nested object."""
        captured_payloads = []

        def capture_urlopen(req, timeout=None):
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return make_mock_response({"choices": [{"message": {"content": "OK"}}]})

        with patch.object(orcarouter_mod.request, "urlopen", side_effect=capture_urlopen):
            orcarouter_mod.query_orcarouter_chat(
                prompt="test",
                model="deepseek/deepseek-reasoner",
                api_key="test-key",
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                timeout=30,
                max_retries=1,
                retry_delay=0.0,
                reasoning_effort="high",
            )

        payload = captured_payloads[0]
        assert payload["reasoning_effort"] == "high"
        assert "reasoning" not in payload


class TestGenerateRow:
    def test_generate_row_returns_expected_format(self, orcarouter_mod):
        mock_response = make_mock_response({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "\\boxed{42}",
                        "reasoning_content": "Thinking...",
                    }
                }
            ]
        })

        row = {"problem": "What is 6*7?", "answer": "42"}

        with patch.object(orcarouter_mod.request, "urlopen", return_value=mock_response):
            result = orcarouter_mod.generate_row(
                row=row,
                shorter_answers_prompt=False,
                model="deepseek/deepseek-reasoner",
                api_key="test-key",
                max_new_tokens=2048,
                temperature=0.0,
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
# Integration test: real OrcaRouter API
# (skipped unless ORCAROUTER_API_KEY is set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ORCAROUTER_API_KEY"),
    reason="Set ORCAROUTER_API_KEY to run real OrcaRouter API integration tests",
)
class TestRealOrcaRouterAPI:
    def test_real_query(self, orcarouter_mod):
        api_key = os.environ["ORCAROUTER_API_KEY"]
        result = orcarouter_mod.query_orcarouter_chat(
            prompt="What is 2+2? Reply with just the number.",
            model="deepseek/deepseek-reasoner",
            api_key=api_key,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            timeout=120,
            max_retries=2,
            retry_delay=2.0,
            reasoning_effort="none",
        )
        assert result["message_content"]
        assert "4" in result["message_content"]

    def test_real_query_returns_thinking(self, orcarouter_mod):
        api_key = os.environ["ORCAROUTER_API_KEY"]
        result = orcarouter_mod.query_orcarouter_chat(
            prompt="What is 17*23? Reply with just the number.",
            model="deepseek/deepseek-reasoner",
            api_key=api_key,
            max_new_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            timeout=120,
            max_retries=2,
            retry_delay=2.0,
        )
        assert "391" in result["message_content"]
        assert result["message_thinking"]

    def test_real_generate_row(self, orcarouter_mod):
        api_key = os.environ["ORCAROUTER_API_KEY"]
        row = {"problem": "What is 10+5?", "answer": "15"}
        result = orcarouter_mod.generate_row(
            row=row,
            shorter_answers_prompt=False,
            model="deepseek/deepseek-reasoner",
            api_key=api_key,
            max_new_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            timeout=120,
            max_retries=2,
            retry_delay=2.0,
        )
        assert result["problem"] == "What is 10+5?"
        assert result["gtruth_answer"] == "15"
        assert "15" in result["message_content"]

    def test_real_invalid_key_is_reported(self, orcarouter_mod):
        with pytest.raises(RuntimeError, match="Failed to query OrcaRouter"):
            orcarouter_mod.query_orcarouter_chat(
                prompt="Reply with OK.",
                model="deepseek/deepseek-reasoner",
                api_key="sk-orca-invalid-key",
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                timeout=60,
                max_retries=1,
                retry_delay=0.0,
            )
