# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from collections import defaultdict
from pathlib import Path

from reasoning_from_scratch.ch03 import load_tokenizer_only


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the records file (.json or .jsonl).",
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


def value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def compute_average_field_lengths(records, tokenizer):
    total_tokens_by_field = defaultdict(int)
    count_by_field = defaultdict(int)
    min_tokens_by_field = {}
    max_tokens_by_field = {}

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Record {idx} is not a JSON object: {type(record).__name__}"
            )

        for field, value in record.items():
            text_value = value_to_text(value)
            token_count = len(tokenizer.encode(text_value))
            total_tokens_by_field[field] += token_count
            count_by_field[field] += 1
            if (
                field not in min_tokens_by_field
                or token_count < min_tokens_by_field[field]
            ):
                min_tokens_by_field[field] = token_count
            if (
                field not in max_tokens_by_field
                or token_count > max_tokens_by_field[field]
            ):
                max_tokens_by_field[field] = token_count

    rows = []
    for field in sorted(total_tokens_by_field):
        total = total_tokens_by_field[field]
        count = count_by_field[field]
        avg = total / count if count else 0.0
        min_tokens = min_tokens_by_field[field]
        max_token = max_tokens_by_field[field]
        rows.append((field, avg, min_tokens, max_token, count))
    return rows


if __name__ == "__main__":
    args = parse_args()
    records = load_records(args.json_path)
    tokenizer = load_tokenizer_only(which_model="reasoning")
    averages = compute_average_field_lengths(records, tokenizer)

    print(f"Records: {len(records)}")
    print("Tokenizer: reasoning")

    header = ("Field", "AvgTokens", "MinTokens", "MaxToken", "Count")
    rows = [
        (field, f"{avg:.2f}", str(min_tokens), str(max_token), str(count))
        for field, avg, min_tokens, max_token, count in averages
    ]

    width_field = max([len(header[0])] + [len(row[0]) for row in rows])
    width_avg = max([len(header[1])] + [len(row[1]) for row in rows])
    width_min = max([len(header[2])] + [len(row[2]) for row in rows])
    width_max = max([len(header[3])] + [len(row[3]) for row in rows])
    width_count = max([len(header[4])] + [len(row[4]) for row in rows])

    print(
        f"{header[0]:<{width_field}}  "
        f"{header[1]:>{width_avg}}  "
        f"{header[2]:>{width_min}}  "
        f"{header[3]:>{width_max}}  "
        f"{header[4]:>{width_count}}"
    )
    for field, avg, min_tokens, max_token, count in rows:
        print(
            f"{field:<{width_field}}  "
            f"{avg:>{width_avg}}  "
            f"{min_tokens:>{width_min}}  "
            f"{max_token:>{width_max}}  "
            f"{count:>{width_count}}"
        )
