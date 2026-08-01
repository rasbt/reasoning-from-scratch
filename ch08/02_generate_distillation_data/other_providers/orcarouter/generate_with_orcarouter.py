# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import concurrent.futures
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

ORCAROUTER_CHAT_URL = "https://api.orcarouter.ai/v1/chat/completions"

# OrcaRouter accepts the same optional attribution headers as other
# OpenAI-compatible routers, which is how a router reports which client
# sent the request.
ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/rasbt/reasoning-from-scratch",
    "X-Title": "reasoning-from-scratch",
}

# OrcaRouter takes the reasoning budget as a flat, top-level
# "reasoning_effort" string (the OpenAI spelling), not as a nested
# {"reasoning": {"effort": ...}} object.
REASONING_EFFORT_CHOICES = ("none", "low", "medium", "high", "xhigh")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=500,
        help="Number of MATH-500 examples to evaluate",
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
        default="deepseek/deepseek-reasoner",
        help=(
            "OrcaRouter model name. Model IDs keep the upstream namespace, "
            "for example deepseek/deepseek-reasoner or qwen/qwen3.5-plus."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation (maps to max_tokens)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=REASONING_EFFORT_CHOICES,
        help=(
            "Optional reasoning budget passed to OrcaRouter as the top-level "
            "reasoning_effort field. Omit to use the model default. Note that "
            "\"none\" turns the thinking stream off, which leaves "
            "message_thinking empty."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retries per sample on request failure",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=3.0,
        help="Base seconds between retries with exponential backoff",
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
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help=(
            "Number of parallel OrcaRouter requests. "
            "Use 1 for sequential generation"
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


def extract_thinking(message):
    """Return the thinking stream of an OrcaRouter chat message.

    OrcaRouter forwards the upstream field name unchanged, and the upstreams
    do not agree on one: DeepSeek, Qwen, and MiniMax return
    "reasoning_content", while Z.ai GLM returns "reasoning". Checking both
    (plus "thinking") keeps message_thinking populated across models.
    """
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def parse_orcarouter_response(decoded):
    """Parse an OrcaRouter API response in OpenAI-compatible format."""
    choices = decoded.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OrcaRouter response missing choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("OrcaRouter response has invalid choices format.")

    message = first_choice.get("message", {})

    content = ""
    thinking = ""

    if isinstance(message, dict):
        content = message.get("content", "") or ""
        thinking = extract_thinking(message)
    elif isinstance(message, str):
        content = message

    if not content and thinking:
        content = thinking

    if not content:
        choice_keys = sorted(first_choice.keys()) if isinstance(first_choice, dict) else []
        root_keys = sorted(decoded.keys()) if isinstance(decoded, dict) else []
        raise RuntimeError(
            "OrcaRouter response did not contain parseable assistant content. "
            f"choice_keys={choice_keys}, root_keys={root_keys}"
        )

    return {
        "message_thinking": thinking,
        "message_content": content,
    }


def query_orcarouter_chat(
    prompt,
    model,
    api_key,
    max_new_tokens,
    temperature,
    top_p,
    timeout,
    max_retries,
    retry_delay,
    reasoning_effort=None,
):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(ATTRIBUTION_HEADERS)

    last_error = None
    for attempt in range(1, max_retries + 1):
        req = request.Request(
            url=ORCAROUTER_CHAT_URL,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            decoded = json.loads(body)
            return parse_orcarouter_response(decoded)

        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"HTTP {exc.code} from OrcaRouter at {ORCAROUTER_CHAT_URL}: {err_body}"
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
        f"Failed to query OrcaRouter after {max_retries} attempt(s). "
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


def generate_row(
    row,
    shorter_answers_prompt,
    model,
    api_key,
    max_new_tokens,
    temperature,
    top_p,
    timeout,
    max_retries,
    retry_delay,
    reasoning_effort=None,
):
    prompt = render_prompt(
        row["problem"],
        shorter_answers_prompt=shorter_answers_prompt,
    )
    response = query_orcarouter_chat(
        prompt=prompt,
        model=model,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        reasoning_effort=reasoning_effort,
    )
    return {
        "problem": row["problem"],
        "gtruth_answer": row["answer"],
        "message_thinking": response["message_thinking"],
        "message_content": response["message_content"],
    }


if __name__ == "__main__":
    args = parse_args()

    api_key = os.environ.get("ORCAROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OrcaRouter API key missing. Set ORCAROUTER_API_KEY.\n"
            "Get your API key at: https://www.orcarouter.ai/console"
        )
    if args.num_processes < 1:
        raise SystemExit("--num_processes must be >= 1.")

    if args.prompt is not None:
        response = query_orcarouter_chat(
            prompt=args.prompt,
            model=args.model,
            api_key=api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            reasoning_effort=args.reasoning_effort,
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
    print(f"Using OrcaRouter API: {ORCAROUTER_CHAT_URL}")

    query_orcarouter_chat(
        prompt="Reply with OK.",
        model=args.model,
        api_key=api_key,
        max_new_tokens=8,
        temperature=args.temperature,
        top_p=1.0,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        reasoning_effort="none",
    )
    print("Model ready")

    start_time = time.time()

    if args.num_processes == 1:
        for offset, row in enumerate(remaining_data, start=1):
            idx = start_idx + offset
            generated_row = generate_row(
                row=row,
                shorter_answers_prompt=args.shorter_answers_prompt,
                model=args.model,
                api_key=api_key,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                reasoning_effort=args.reasoning_effort,
            )

            rows.append(generated_row)
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
    else:
        print(f"Parallel requests enabled: {args.num_processes}")
        next_submit = 0
        next_write = 0
        futures = {}
        completed_rows = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_processes
        ) as executor:
            while next_write < remaining_total:
                while (
                    next_submit < remaining_total
                    and len(futures) < args.num_processes
                ):
                    row = remaining_data[next_submit]
                    future = executor.submit(
                        generate_row,
                        row,
                        args.shorter_answers_prompt,
                        args.model,
                        api_key,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                        args.timeout,
                        args.max_retries,
                        args.retry_delay,
                        args.reasoning_effort,
                    )
                    futures[future] = next_submit
                    next_submit += 1

                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                failed_at = None
                failed_exc = None
                for future in done:
                    offset0 = futures.pop(future)
                    try:
                        completed_rows[offset0] = future.result()
                    except Exception as exc:
                        if failed_at is None:
                            failed_at = offset0
                            failed_exc = exc

                while next_write in completed_rows:
                    rows.append(completed_rows.pop(next_write))
                    write_rows_json_incremental(rows, out_file)

                    processed = next_write + 1
                    idx = start_idx + processed
                    progress_msg = eta_progress_message(
                        processed=processed,
                        total=remaining_total,
                        start_time=start_time,
                        show_eta=True,
                        label="MATH-500",
                    )

                    if args.verbose:
                        print(f"{progress_msg}")
                        print(f"{idx}/{num_examples} -> {rows[-1]['message_content']}")
                    else:
                        print(
                            f"{idx}/{num_examples} | {progress_msg}",
                            end="\r",
                            flush=True,
                        )
                    next_write += 1

                if failed_at is not None:
                    for pending_future in futures:
                        pending_future.cancel()
                    failing_idx = start_idx + failed_at + 1
                    raise RuntimeError(
                        f"Generation failed at dataset row {failing_idx}."
                    ) from failed_exc

    write_rows_json_incremental(rows, out_file)

    seconds_elapsed = time.time() - start_time
    print(f"\nTotal time: {seconds_elapsed/60:.1f} min")
    print(f"\nWrote {len(rows)} rows to: {out_file}")
