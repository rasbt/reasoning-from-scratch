# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
import os
import re
import time
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib import error, request

import requests
from reasoning_from_scratch.ch03 import eta_progress_message


DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful math assistant.\n"
    "Answer the question and write the final result on a new line as:\n"
    "\\boxed{{ANSWER}}\n\n"
    "Question:\n{prompt}\n\n"
    "Answer:"
)

SHORTER_ANSWERS_PROMPT_TEMPLATE = (
    "You are a helpful math assistant.\n"
    "Provide a short explanation, and then write the "
    "final result on a new line as:\n"
    "\\boxed{{ANSWER}}\n\n"
    "Question:\n{prompt}\n\n"
    "Answer:"
)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=500,
        help="Number of MATH-500 examples to evaluate. Default: 500",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Optional single prompt mode. If set, skips MATH-500 and prints "
            "one JSON object to stdout."
        ),
    )
    parser.add_argument(
        "--math_json",
        type=str,
        default=None,
        help=(
            "Optional path to a MATH-500 JSON file. "
            "If omitted, load_math500_test() defaults are used."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-r1",
        help="OpenRouter model name. Default: deepseek/deepseek-r1",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation (maps to max_tokens). Default: 2048",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter. Default: 1.0",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds. Default: 600",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retries per sample on request failure. Default: 3",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=3.0,
        help="Base seconds between retries with exponential backoff. Default: 3.0",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help=(
            "Output JSON file path. "
            "If omitted, uses a model-based default filename."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full prediction for each sample.",
    )
    parser.add_argument(
        "--shorter_answers_prompt",
        action="store_true",
        help=(
            "Use a prompt that asks for shorter explanations while keeping the "
            "final boxed answer format."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing output JSON file by skipping already "
            "completed rows."
        ),
    )
    return parser.parse_args()


def render_prompt(prompt, shorter_answers_prompt=False):
    template = (
        SHORTER_ANSWERS_PROMPT_TEMPLATE
        if shorter_answers_prompt
        else DEFAULT_PROMPT_TEMPLATE
    )
    return template.format(prompt=prompt)


def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy:
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


def content_to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Prefer common text-bearing keys first.
        preferred_keys = (
            "content",
            "text",
            "value",
            "output_text",
            "response",
            "answer",
            "final",
            "refusal",
            "reasoning",
            "reasoning_content",
            "thinking",
            "message",
            "parts",
        )
        chunks = []
        for key in preferred_keys:
            if key in value:
                chunk = content_to_text(value[key])
                if chunk:
                    chunks.append(chunk)

        # If no preferred keys are present, recursively inspect nested values
        # but skip metadata keys that are unlikely to contain model text.
        if not chunks:
            metadata_keys = {
                "id",
                "type",
                "role",
                "index",
                "finish_reason",
                "logprobs",
                "usage",
            }
            for key, nested_value in value.items():
                if key in metadata_keys:
                    continue
                chunk = content_to_text(nested_value)
                if chunk:
                    chunks.append(chunk)

        return "".join(chunks)
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                chunks.append(str(item))
                continue
            text = item["text"] if "text" in item else None
            if text is None and "content" in item:
                text = item["content"]
            if text is None and "value" in item:
                text = item["value"]
            if text is None:
                continue
            chunks.append(str(text))
        return "".join(chunks)
    if value is None:
        return ""
    return str(value)


def parse_openrouter_response(decoded):
    choices = decoded["choices"] if "choices" in decoded else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter response missing choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("OpenRouter response has invalid choices format.")

    message = first_choice["message"] if "message" in first_choice else None

    content = ""
    if isinstance(message, dict):
        content = content_to_text(message["content"] if "content" in message else None)
    elif isinstance(message, str):
        content = content_to_text(message)

    if not content:
        content = content_to_text(first_choice["text"] if "text" in first_choice else None)
    if not content:
        content = content_to_text(first_choice["content"] if "content" in first_choice else None)
    if not content:
        content = content_to_text(first_choice["output_text"] if "output_text" in first_choice else None)
    if not content:
        content = content_to_text(first_choice["delta"] if "delta" in first_choice else None)
    if not content:
        content = content_to_text(decoded["response"] if "response" in decoded else None)
    if not content:
        content = content_to_text(
            decoded["output_text"] if "output_text" in decoded else None
        )
    if not content:
        content = content_to_text(decoded["output"] if "output" in decoded else None)
    if not content:
        content = content_to_text(decoded["message"] if "message" in decoded else None)
    if not content:
        content = content_to_text(decoded["messages"] if "messages" in decoded else None)
    if not content:
        content = content_to_text(decoded["completion"] if "completion" in decoded else None)

    thinking = ""
    if isinstance(message, dict):
        thinking = content_to_text(message["reasoning"] if "reasoning" in message else None)
    if not thinking:
        if isinstance(message, dict):
            thinking = content_to_text(
                message["reasoning_content"] if "reasoning_content" in message else None
            )
    if not thinking:
        if isinstance(message, dict):
            thinking = content_to_text(
                message["thinking"] if "thinking" in message else None
            )
    if not thinking:
        thinking = content_to_text(
            first_choice["reasoning"] if "reasoning" in first_choice else None
        )
    if not thinking:
        thinking = content_to_text(
            first_choice["reasoning_content"] if "reasoning_content" in first_choice else None
        )
    if not thinking:
        thinking = content_to_text(
            first_choice["thinking"] if "thinking" in first_choice else None
        )
    if not thinking:
        thinking = content_to_text(decoded["reasoning"] if "reasoning" in decoded else None)
    if not thinking:
        thinking = content_to_text(
            decoded["reasoning_content"] if "reasoning_content" in decoded else None
        )
    if not thinking:
        thinking = content_to_text(decoded["thinking"] if "thinking" in decoded else None)
    if not thinking:
        thinking = content_to_text(decoded["output"] if "output" in decoded else None)

    # Some models/providers only expose reasoning-like text.
    if not content and thinking:
        content = thinking

    if not content:
        choice_keys = sorted(first_choice.keys()) if isinstance(first_choice, dict) else []
        root_keys = sorted(decoded.keys()) if isinstance(decoded, dict) else []
        raise RuntimeError(
            "OpenRouter response did not contain parseable assistant content. "
            f"choice_keys={choice_keys}, root_keys={root_keys}"
        )

    return {
        "message_thinking": thinking,
        "message_content": content,
    }


def query_openrouter_chat(
    prompt,
    model,
    api_key,
    max_new_tokens,
    temperature,
    top_p,
    timeout,
    max_retries,
    retry_delay,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(
            url=OPENROUTER_CHAT_URL,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            decoded = json.loads(body)
            return parse_openrouter_response(decoded)

        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"HTTP {exc.code} from OpenRouter at {OPENROUTER_CHAT_URL}: {err_body}"
            )
        except (
            error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            IncompleteRead,
            RemoteDisconnected,
            ConnectionResetError,
        ) as exc:
            last_error = exc

        if attempt < max_retries:
            backoff_delay = min(retry_delay * (2 ** (attempt - 1)), 60.0)
            time.sleep(backoff_delay)

    raise RuntimeError(
        f"Failed to query OpenRouter after {max_retries} attempt(s). "
        f"Last error: {last_error}"
    )


def model_to_filename(model_name):
    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    if not safe_model:
        safe_model = "model"
    return f"math500_{safe_model}_full_answers.json"


def write_rows_json_incremental(rows, out_file):
    tmp_file = out_file.with_name(f"{out_file.name}.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_file.replace(out_file)


def load_resume_rows(out_file):
    with out_file.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if isinstance(parsed, list):
        return parsed
    if (
        isinstance(parsed, dict)
        and "records" in parsed
        and isinstance(parsed["records"], list)
    ):
        return parsed["records"]
    raise ValueError(
        f"Resume file must contain a JSON array. Got {type(parsed).__name__}."
    )


def validate_resume_rows(rows, selected_data):
    if len(rows) > len(selected_data):
        raise ValueError(
            f"Resume file has {len(rows)} rows, but dataset has only "
            f"{len(selected_data)} examples."
        )

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Resume row {idx} is not a JSON object: {type(row).__name__}."
            )
        if "problem" not in row:
            raise KeyError(f"Resume row {idx} is missing key: problem")

        expected_problem = selected_data[idx - 1]["problem"]
        if row["problem"] != expected_problem:
            raise ValueError(
                f"Resume row {idx} does not match the current dataset. "
                "Use a different output file or disable --resume."
            )


if __name__ == "__main__":
    args = parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"] if "OPENROUTER_API_KEY" in os.environ else None
    if not api_key:
        raise SystemExit(
            "OpenRouter API key missing. Set OPENROUTER_API_KEY."
        )

    if args.prompt is not None:
        response = query_openrouter_chat(
            prompt=args.prompt,
            model=args.model,
            api_key=api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        print(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "message_thinking": response["message_thinking"],
                    "message_content": response["message_content"],
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(0)

    if args.out_file:
        out_file = Path(args.out_file).expanduser().resolve()
    else:
        out_file = (Path.cwd() / model_to_filename(args.model)).resolve()

    if args.math_json:
        math_data = load_math500_test(
            local_path=args.math_json,
            save_copy=False,
        )
    else:
        math_data = load_math500_test()
    selected_data = math_data[: args.dataset_size]
    num_examples = len(selected_data)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start_idx = 0
    if args.resume and out_file.exists():
        rows = load_resume_rows(out_file)
        validate_resume_rows(rows, selected_data)
        start_idx = len(rows)
        print(f"Resume enabled: {start_idx}/{num_examples} rows already completed.")
    else:
        if args.resume:
            print(
                f"Resume enabled but output file does not exist yet: {out_file}"
            )
        write_rows_json_incremental(rows, out_file)

    if start_idx >= num_examples:
        print(f"All {num_examples} rows are already completed: {out_file}")
        raise SystemExit(0)

    remaining_data = selected_data[start_idx:]
    remaining_total = len(remaining_data)

    print(f"Loading model: {args.model}")
    print(f"Using OpenRouter API: {OPENROUTER_CHAT_URL}")

    query_openrouter_chat(
        prompt="Reply with OK.",
        model=args.model,
        api_key=api_key,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    print("Model ready")

    start_time = time.time()

    for offset, row in enumerate(remaining_data, start=1):
        idx = start_idx + offset
        prompt = render_prompt(
            row["problem"],
            shorter_answers_prompt=args.shorter_answers_prompt,
        )
        response = query_openrouter_chat(
            prompt=prompt,
            model=args.model,
            api_key=api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        rows.append(
            {
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "message_thinking": response["message_thinking"],
                "message_content": response["message_content"],
            }
        )
        write_rows_json_incremental(rows, out_file)

        progress_msg = eta_progress_message(
            processed=offset,
            total=remaining_total,
            start_time=start_time,
            show_eta=True,
            label="MATH-500",
        )

        if args.verbose:
            print(f"{progress_msg}")
            print(f"{idx}/{num_examples} -> {rows[-1]['message_content']}")
        else:
            print(f"{idx}/{num_examples} | {progress_msg}", end="\r", flush=True)

    write_rows_json_incremental(rows, out_file)

    seconds_elapsed = time.time() - start_time
    print(f"\nTotal time: {seconds_elapsed/60:.1f} min")
    print(f"\nWrote {len(rows)} rows to: {out_file}")
