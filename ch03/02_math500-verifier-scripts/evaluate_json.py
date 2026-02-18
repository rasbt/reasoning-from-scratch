# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path

from reasoning_from_scratch.ch03 import (
    extract_final_candidate,
    grade_answer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the records file (.json or .jsonl).",
    )
    parser.add_argument(
        "--gtruth_answer",
        type=str,
        default="gtruth_answer",
        help="Key name for the ground-truth answer. Default: gtruth_answer",
    )
    parser.add_argument(
        "--generated_text",
        type=str,
        default="generated_text",
        help="Key name for generated model output. Default: generated_text",
    )
    return parser.parse_args()


def load_records(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            parsed = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            records = []
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}: {exc}"
                    ) from exc
            return records

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "records" in parsed and isinstance(parsed["records"], list):
            return parsed["records"]
        return [parsed]

    raise ValueError(
        f"Unsupported JSON root type in {path}: {type(parsed).__name__}"
    )


def evaluate_records(records, gtruth_key, generated_text_key):
    num_examples = len(records)
    num_correct = 0

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Record {idx} is not a JSON object: {type(record).__name__}"
            )

        if gtruth_key not in record:
            raise KeyError(f"Record {idx} is missing key: {gtruth_key}")
        if generated_text_key not in record:
            raise KeyError(f"Record {idx} is missing key: {generated_text_key}")

        extracted = extract_final_candidate(record[generated_text_key])
        is_correct = grade_answer(extracted, record[gtruth_key])
        num_correct += int(is_correct)

    acc = num_correct / num_examples if num_examples else 0.0
    return num_correct, num_examples, acc


if __name__ == "__main__":
    args = parse_args()
    records = load_records(args.json_path)
    num_correct, num_examples, acc = evaluate_records(
        records=records,
        gtruth_key=args.gtruth_answer,
        generated_text_key=args.generated_text,
    )
    print(f"Accuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
