# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from reasoning_from_scratch.ch03 import render_prompt

from hf_wrapper import build_model_and_tokenizer, format_distilled_answer


class AnswerOnlyDataCollator:
    def __init__(self, tokenizer):
        # Keep this private so Trainer does not mistake the repo tokenizer for
        # a Hugging Face processing class during checkpoint saves
        self._tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self._tokenizer.pad_token_id

        input_ids = []
        labels = []
        attention_mask = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Expected a top-level JSON array.")

    return data


def split_data(data, validation_size=25, seed=123):
    data = list(data)
    rnd = random.Random(seed)
    rnd.shuffle(data)

    if len(data) < 2:
        raise ValueError("Need at least 2 examples to create train/validation splits.")
    if not (1 <= validation_size < len(data)):
        raise ValueError("--validation_size must be between 1 and dataset size - 1.")

    train_size = len(data) - validation_size
    return data[:train_size], data[train_size:]


def build_records(data, tokenizer, tokenizer_kind, max_seq_len):
    records = []
    skipped = 0
    use_think_tokens = tokenizer_kind == "reasoning"

    for entry in data:
        try:
            prompt = render_prompt(entry["problem"])
            answer = format_distilled_answer(
                entry,
                use_think_tokens=use_think_tokens,
            )

            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(answer, chat_wrapped=False)

            token_ids = prompt_ids + answer_ids
            if tokenizer.eos_token_id is not None:
                token_ids.append(tokenizer.eos_token_id)

            if len(token_ids) < 2 or len(token_ids) > max_seq_len:
                skipped += 1
                continue

            labels = list(token_ids)
            prompt_len = min(len(prompt_ids), len(labels) - 1)
            labels[:prompt_len] = [-100] * prompt_len

            records.append(
                {
                    "input_ids": token_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(token_ids),
                }
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1

    return records, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Continue training Qwen3 with transformers.Trainer via a local "
            "wrapper around the scratch Qwen3Model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer_kind",
        type=str,
        choices=("base", "reasoning"),
        default="base",
        help="Tokenizer family used for the raw .pth weights.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional local .pth file. If omitted, downloads the default model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer JSON file. If omitted, downloads the tokenizer.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="qwen3",
        help="Download/cache directory for the model and tokenizer.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to a chapter-8 distillation JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wrapper_trainer_output",
        help="Trainer output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=0,
        help="Use only the first N rows before splitting (0 uses the full file).",
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=25,
        help="Absolute number of validation examples.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Drop examples longer than this token length.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="How often Trainer prints training metrics.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="How often Trainer writes checkpoints.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0, override epochs and stop after this many optimizer steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(
        tokenizer_kind=args.tokenizer_kind,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        local_dir=args.local_dir,
    )
    model.config.use_cache = False

    data = load_json(args.data_path)
    if args.dataset_size > 0:
        data = data[: args.dataset_size]

    records, skipped = build_records(
        data=data,
        tokenizer=tokenizer,
        tokenizer_kind=args.tokenizer_kind,
        max_seq_len=args.max_seq_len,
    )
    train_records, val_records = split_data(
        records,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    collator = AnswerOnlyDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.logging_steps,
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        max_steps=args.max_steps,
        bf16=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_records,
        eval_dataset=val_records,
        data_collator=collator,
    )

    print("Tokenizer kind:", args.tokenizer_kind)
    print("Prepared examples:", len(records))
    print("Skipped rows:", skipped)
    print("Train examples:", len(train_records))
    print("Validation examples:", len(val_records))

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
