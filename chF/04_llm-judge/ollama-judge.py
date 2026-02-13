# Copyright (c) Sebastian Raschka under Apache License 2.0
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
import re
from pathlib import Path
import psutil
import requests

import torch
from reasoning_from_scratch.ch02 import get_device, generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import load_model_and_tokenizer


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


# Same as chapter 3
def get_data():
    local_path = Path("math500_test.json")
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            math_data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        math_data = r.json()

    return math_data


def prepare_math500_pairs(math_items):
    pairs = []
    for ex in math_items:
        problem = ex.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            continue
        ref = ex.get("answer")
        if not isinstance(ref, str) or not ref.strip():
            ref = ex.get("solution", "")
        if not isinstance(ref, str):
            ref = str(ref)
        pairs.append(
            {
                "instruction": problem.strip(),
                "reference_answer": ref.strip(),
            }
        )
    return pairs


def query_model(
    prompt,
    model="gpt-oss:20b",
    url="http://localhost:11434/api/chat",
    max_new_tokens=2048,
):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": int(max_new_tokens),
        }
    }

    # Send the POST request
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


def rubric_prompt(instruction, reference_answer, model_answer):
    rubric = (
        "You are a fair judge assistant. You will be given an instruction, "
        "a reference answer, and a candidate answer to evaluate, according "
        "to the following rubric:\n\n"
        "1: The response fails to address the instruction, providing "
        "irrelevant, incorrect, or excessively verbose content.\n"
        "2: The response partially addresses the instruction but contains "
        "major errors, omissions, or irrelevant details.\n"
        "3: The response addresses the instruction to some degree but is "
        "incomplete, partially correct, or unclear in places.\n"
        "4: The response mostly adheres to the instruction, with only minor "
        "errors, omissions, or lack of clarity.\n"
        "5: The response fully adheres to the instruction, providing a "
        "clear, accurate, and relevant answer in a concise and efficient "
        "manner.\n\n"
        "Now here is the instruction, the reference answer, and the "
        "response.\n"
    )

    prompt = (
        f"{rubric}\n"
        f"Instruction:\n{instruction}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer:\n{model_answer}\n\n"
        f"Evaluation: "
    )
    return prompt


def parse_score(judge_text, default=3):
    m = re.search(r"([1-5])(?:\D|$)", judge_text)
    return int(m.group(1)) if m else int(default)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device e.g., 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning"],
        help="Candidate variant to use. Defaults to 'base'.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="Number of MATH-500 examples to evaluate. Default: 10",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for candidate generation. Default: 2048",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434/api/chat",
        help="Ollama chat endpoint for the judge. Default: 'http://localhost:11434/api/chat'"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-oss:20b",
        help="Judge model name (Ollama). Used only for scoring. Default: 'gpt-oss:20b'",
    )
    return parser.parse_args()


def generate_with_qwen3(model, tokenizer, prompt, device, max_new_tokens):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
    ).unsqueeze(0)

    new_token_ids = []
    with torch.no_grad():
        for tok in generate_text_basic_stream_cache(
            model=model,
            token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id
        ):
            new_token_ids.append(int(tok.squeeze(0)))

    # Decode only the generated continuation (not the prompt)
    return tokenizer.decode(new_token_ids)


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")

    # Data and local Qwen3 model
    math_data = get_data()
    pairs = prepare_math500_pairs(math_data)
    if not pairs:
        raise SystemExit("No usable (instruction, reference_answer) pairs found.")
    pairs = pairs[:dataset_size]
    total = len(pairs)

    candidate_model, tokenizer = load_model_and_tokenizer(which_model, device, use_compile=False)
    candidate_model.eval()
    torch.set_float32_matmul_precision("high")

    # Evaluation loop
    results = []
    score_counts = {i: 0 for i in range(1, 6)}

    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError(
            "Ollama not running. "
            "Launch ollama before proceeding."
        )
    print("Ollama running:", check_if_running("ollama"))

    for idx, ex in enumerate(pairs, start=1):
        instruction = ex["instruction"]
        reference = ex["reference_answer"]

        # 1) Candidate answer from local Qwen3 (base or reasoning)
        answer = generate_with_qwen3(
            candidate_model, tokenizer, instruction, device, max_new_tokens
        ).strip()

        # 2) Judge with rubric using gpt-oss:20b (or --judge_model) via Ollama
        judge_in = rubric_prompt(instruction, reference, answer)
        judge_out = query_model(
            judge_in,
            model=args.judge_model,
            url=args.url,
            max_new_tokens=256,
        ).strip()
        score = parse_score(judge_out)

        score_counts[score] += 1
        results.append(
            {
                "idx": idx,
                "instruction": instruction,
                "reference_answer": reference,
                "model_answer": answer,
                "judge_text": judge_out,
                "score": score,
                "candidate_model": f"qwen3-{which_model}",
                "judge_model": args.judge_model,
            }
        )
        print(f"[{idx}/{total}] score={score}", flush=True)

    # Summary
    total_scored = sum(score_counts.values())
    avg = (sum(k * v for k, v in score_counts.items()) / total_scored) if total_scored else 0.0

    print("\nSummary")
    print("-------")
    print(f"Average score: {avg:.3f} over {total_scored} example(s)")
    print("Counts:", " ".join(f"{k}:{v}" for k, v in score_counts.items()))
