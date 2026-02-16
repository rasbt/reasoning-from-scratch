# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path
import time

import torch

import reasoning_from_scratch.bonus as bonus
from reasoning_from_scratch.qwen3 import (
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    eta_progress_message,
    extract_final_candidate,
    load_math500_test,
    evaluate_math500_stream,
    generate_text_stream_concat,
    load_model_and_tokenizer,
    load_tokenizer_only,
    render_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto' (default), or any torch device string like 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to load. Defaults to 'base'.",
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
        help="Max new tokens for generation. Default: 2048",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to load model weights from.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
    )
    parser.add_argument(
        "--hybrid_parser",
        action="store_true",
        help=(
            "Use the advanced hybrid parser for answer grading instead of the "
            "default chapter parser."
        ),
    )
    return parser.parse_args()


def evaluate_math500_stream_hybrid(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])
            gen_text = generate_text_stream_concat(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            total_len += len(tokenizer.encode(gen_text))

            extracted = extract_final_candidate(gen_text)
            pred = bonus.normalize_text_hybrid(extracted)
            gold = bonus.normalize_text_hybrid(row["answer"])
            is_correct = pred == gold
            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "pred_normalized_hybrid": pred,
                "gold_normalized_hybrid": gold,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    avg_len = total_len / num_examples
    print(f"Average response length: {avg_len:.2f} tokens")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


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

    math_data = load_math500_test()

    if args.which_model == "instruct":
        which_model = "reasoning"
    else:
        which_model = args.which_model

    if args.checkpoint_path:
        # To load the saved RL checkpoint files from chapter 6
        tokenizer = load_tokenizer_only(which_model=which_model)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        if args.compile:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            model = torch.compile(model)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model=which_model,
            device=device,
            use_compile=args.compile
        )

    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    model.eval()
    torch.set_float32_matmul_precision("high")

    if args.hybrid_parser:
        backend_ok = bonus.normalize_text_hybrid(r"\frac{1}{2}") == "1/2"
        print("LaTeX backend ready:", backend_ok)
        if not backend_ok:
            print('Suggestion: uv pip install "antlr4-python3-runtime==4.11.*"')

        num_correct, num_examples, acc = evaluate_math500_stream_hybrid(
            model=model,
            out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
            tokenizer=tokenizer,
            device=device,
            math_data=math_data[:dataset_size],
            max_new_tokens=max_new_tokens,
            verbose=args.verbose,
        )
    else:
        num_correct, num_examples, acc = evaluate_math500_stream(
            model=model,
            out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
            tokenizer=tokenizer,
            device=device,
            math_data=math_data[:dataset_size],
            max_new_tokens=max_new_tokens,
            verbose=args.verbose,
        )
