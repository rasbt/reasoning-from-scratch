# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
import random
import time
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    eta_progress_message,
    load_tokenizer_only,
    render_prompt,
)
from reasoning_from_scratch.qwen3_batched import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    load_model_and_tokenizer,
)

SCRIPT_NAME = Path(__file__).stem
CSV_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.csv"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / SCRIPT_NAME
IGNORE_INDEX = -100


def strip_think_tags(text):
    return text.replace("<think>", "").replace("</think>", "").strip()


def format_distilled_answer(entry, use_think_tokens=False):
    content = str(entry["message_content"]).strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")

    content = strip_think_tags(content)

    if "message_thinking" in entry:
        thinking = str(entry["message_thinking"]).strip()
    else:
        thinking = ""
    thinking = strip_think_tags(thinking)

    if use_think_tokens:
        return f"<think>{thinking}</think>\n\n{content}"

    if thinking:
        return f"{thinking}\n\n{content}"

    return content


def build_examples(data, tokenizer, use_think_tokens=False):
    examples = []
    skipped = 0

    for entry in data:
        try:
            prompt = render_prompt(entry["problem"])
            target_answer = format_distilled_answer(
                entry,
                use_think_tokens=use_think_tokens,
            )

            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(target_answer, chat_wrapped=False)

            token_ids = prompt_ids + answer_ids
            if tokenizer.eos_token_id is not None:
                token_ids += [tokenizer.eos_token_id]

            if len(token_ids) < 2:
                skipped += 1
                continue

            prompt_len = min(len(prompt_ids), len(token_ids) - 1)
            answer_token_count = len(token_ids) - prompt_len
            if answer_token_count <= 0:
                skipped += 1
                continue

            examples.append({"token_ids": token_ids, "prompt_len": prompt_len})
        except (KeyError, TypeError, ValueError):
            skipped += 1

    return examples, skipped


def filter_examples_by_max_len(examples, max_len=2048):
    filtered_examples = [
        example for example in examples if len(example["token_ids"]) <= max_len
    ]
    removed = len(examples) - len(filtered_examples)
    return filtered_examples, removed


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")

    return data


def split_data(data, validation_size=50, seed=123):
    data = list(data)
    rnd = random.Random(seed)
    rnd.shuffle(data)

    n_total = len(data)
    if n_total < 2:
        raise ValueError("Need at least 2 examples to create train/validation splits.")

    if not (1 <= validation_size < n_total):
        raise ValueError("--validation_size must be between 1 and dataset size - 1.")

    n_val = validation_size
    n_train = n_total - n_val

    train_data = data[:n_train]
    val_data = data[n_train:]
    return train_data, val_data


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    checkpoint_path = checkpoint_dir / f"qwen3-0.6B-distill-step{step:05d}{suffix}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def append_csv_metrics(
    csv_log_path,
    epoch_idx,
    total_steps,
    train_loss,
    val_loss,
):
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "epoch,total_steps,train_loss,val_loss\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{epoch_idx},{total_steps},{train_loss:.6f},"
            f"{val_loss:.6f}\n"
        )


def iter_batches(examples, batch_size):
    for start_idx in range(0, len(examples), batch_size):
        yield examples[start_idx:start_idx + batch_size]


def prepare_batch_tensors(batch_examples, pad_id, device):
    if pad_id is None:
        raise ValueError("Tokenizer is missing pad_token_id, which is required for batching.")

    max_input_len = max(len(example["token_ids"]) - 1 for example in batch_examples)
    batch_size = len(batch_examples)

    input_ids = torch.full(
        (batch_size, max_input_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    attn_mask = torch.zeros(
        (batch_size, max_input_len),
        dtype=torch.bool,
        device=device,
    )
    labels = torch.full(
        (batch_size, max_input_len),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )

    supervised_token_count = 0
    for row_idx, example in enumerate(batch_examples):
        token_ids = example["token_ids"]
        prompt_len = example["prompt_len"]

        input_seq = token_ids[:-1]
        target_seq = token_ids[1:]
        seq_len = len(input_seq)
        offset = max_input_len - seq_len

        input_ids[row_idx, offset:] = torch.tensor(
            input_seq, dtype=torch.long, device=device
        )
        attn_mask[row_idx, offset:] = True
        labels[row_idx, offset:] = torch.tensor(
            target_seq, dtype=torch.long, device=device
        )

        answer_start = max(prompt_len - 1, 0)
        if answer_start > 0:
            labels[row_idx, offset:offset + answer_start] = IGNORE_INDEX

        supervised_token_count += max(0, len(token_ids) - prompt_len)

    return input_ids, attn_mask, labels, supervised_token_count


def compute_batch_loss(model, batch_examples, pad_id, device):
    input_ids, attn_mask, labels, supervised_token_count = prepare_batch_tensors(
        batch_examples=batch_examples,
        pad_id=pad_id,
        device=device,
    )

    logits = model(input_ids, attn_mask=attn_mask)
    per_token_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2).float(),
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )

    active_mask = labels.ne(IGNORE_INDEX)
    token_count_per_example = active_mask.sum(dim=1)
    token_loss_sum_per_example = (per_token_loss * active_mask).sum(dim=1)
    per_example_loss = token_loss_sum_per_example / token_count_per_example.clamp(min=1)
    batch_loss = per_example_loss.mean()

    return batch_loss, supervised_token_count


@torch.no_grad()
def evaluate_examples_batched(model, examples, batch_size, pad_id, device):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch_examples in iter_batches(examples, batch_size):
        batch_loss, _ = compute_batch_loss(
            model=model,
            batch_examples=batch_examples,
            pad_id=pad_id,
            device=device,
        )
        batch_size_actual = len(batch_examples)
        total_loss += batch_loss.item() * batch_size_actual
        total_examples += batch_size_actual

    if was_training:
        model.train()

    return total_loss / total_examples


def train_distillation_batched(
    model,
    train_examples,
    val_examples,
    pad_id,
    device,
    batch_size=4,
    epochs=2,
    lr=5e-6,
    seed=42,
    log_every=50,
    grad_clip_norm=None,
    checkpoint_dir=CHECKPOINT_DIR,
    csv_log_path=CSV_LOG_PATH,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    if batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if log_every < 0:
        raise ValueError("--log_every must be >= 0.")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("--grad_clip_norm must be > 0 when provided.")

    steps_per_epoch = (len(train_examples) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    global_step = 0
    rng = random.Random(seed)
    start_time = time.time()
    csv_log_path = Path(csv_log_path)

    for epoch in range(1, epochs + 1):
        epoch_examples = list(train_examples)
        rng.shuffle(epoch_examples)

        epoch_train_loss = 0.0
        epoch_example_count = 0

        for batch_examples in iter_batches(epoch_examples, batch_size):
            global_step += 1
            step_start = time.time()
            optimizer.zero_grad()

            loss, supervised_tokens = compute_batch_loss(
                model=model,
                batch_examples=batch_examples,
                pad_id=pad_id,
                device=device,
            )

            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_size_actual = len(batch_examples)
            epoch_train_loss += loss.item() * batch_size_actual
            epoch_example_count += batch_size_actual

            step_time = time.time() - step_start
            step_tokens_per_sec = (
                supervised_tokens / step_time if step_time > 0 else 0.0
            )
            if log_every and global_step % log_every == 0:
                val_loss = evaluate_examples_batched(
                    model=model,
                    examples=val_examples,
                    batch_size=batch_size,
                    pad_id=pad_id,
                    device=device,
                )
                model.train()
                progress_msg = eta_progress_message(
                    processed=global_step,
                    total=total_steps,
                    start_time=start_time,
                    show_eta=True,
                    label="Progress",
                ).rstrip()
                if "| ETA:" in progress_msg:
                    eta_value = progress_msg.split("| ETA:", 1)[1].strip()
                else:
                    eta_value = "--"
                print(
                    f"[Epoch {epoch}/{epochs} Step {global_step}/{total_steps}] "
                    f"train_loss={loss.item():.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"tok/sec={step_tokens_per_sec:.1f} | "
                    f"ETA: {eta_value}",
                    flush=True,
                )
                append_csv_metrics(
                    csv_log_path=csv_log_path,
                    epoch_idx=epoch,
                    total_steps=global_step,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                )

        avg_train_loss = epoch_train_loss / epoch_example_count
        val_loss = evaluate_examples_batched(
            model=model,
            examples=val_examples,
            batch_size=batch_size,
            pad_id=pad_id,
            device=device,
        )
        append_csv_metrics(
            csv_log_path=csv_log_path,
            epoch_idx=epoch,
            total_steps=global_step,
            train_loss=avg_train_loss,
            val_loss=val_loss,
        )

        checkpoint_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=global_step,
            suffix=f"epoch{epoch}",
        )
        print(f"Saved checkpoint to {checkpoint_path}", flush=True)

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple batched distillation into Qwen3 0.6B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="distill_data.json",
        help="Path to distillation JSON data.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=0,
        help="Number of dataset examples to use before splitting (0 = all).",
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=25,
        help="Absolute number of validation examples.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of examples per optimization step.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help=(
            "Run validation every N global training steps for step logs. "
            "Use 0 to disable step-level validation."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum tokenized sequence length; longer examples are filtered out.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional .pth checkpoint to initialize model weights from before "
            "training. Optimizer and step state are not restored."
        ),
    )
    parser.add_argument(
        "--grad_clip_norm",
        "--grad_clip",
        dest="grad_clip_norm",
        type=float,
        default=None,
        help=(
            "Clip gradient norm to this value. "
            "Default is no gradient clipping."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--use_think_tokens",
        action="store_true",
        help=(
            "Wrap thinking in '<think>...</think>' and use the reasoning tokenizer. "
            "Default behavior concatenates thinking and answer without think tokens."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device()

    data = load_json(args.data_path)
    if args.dataset_size > 0:
        data = data[:args.dataset_size]
    checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print("Device:", device)
    print("Dataset size:", len(data))

    tokenizer_variant = "reasoning" if args.use_think_tokens else "base"
    tokenizer = load_tokenizer_only(which_model=tokenizer_variant)
    if checkpoint_path is not None:
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model, _ = load_model_and_tokenizer(
            which_model="base",
            device=device,
            use_compile=False,
        )

    print("Model variant: base")
    print("Tokenizer variant:", tokenizer_variant)
    print("Use think tokens:", args.use_think_tokens)
    print("Batch size:", args.batch_size)
    print("Grad clip norm:", args.grad_clip_norm)
    print("Checkpoint path:", checkpoint_path if checkpoint_path is not None else "--")

    raw_row_count = len(data)
    all_examples, skipped_rows = build_examples(
        data,
        tokenizer,
        use_think_tokens=args.use_think_tokens,
    )
    tokenized_example_count = len(all_examples)
    all_examples, length_filtered_rows = filter_examples_by_max_len(
        all_examples, max_len=args.max_seq_len
    )
    train_examples, val_examples = split_data(
        all_examples,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    print("Raw dataset rows:", raw_row_count)
    print(
        "Skipped rows during preprocessing (invalid, empty, or malformed rows):",
        skipped_rows,
    )
    print("Examples after tokenization:", tokenized_example_count)
    print(
        f"Examples filtered by max_seq_len={args.max_seq_len}:",
        length_filtered_rows,
    )
    print(
        "Prepared examples after preprocessing (total/train/val):",
        len(all_examples),
        len(train_examples),
        len(val_examples),
    )

    if len(train_examples) == 0:
        raise RuntimeError("No valid training examples after preprocessing.")
    if len(val_examples) == 0:
        raise RuntimeError("No valid validation examples after preprocessing.")

    start = time.perf_counter()
    train_distillation_batched(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        pad_id=tokenizer.pad_token_id,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        log_every=args.log_every,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_dir=CHECKPOINT_DIR,
        csv_log_path=CSV_LOG_PATH,
    )
    elapsed_minutes = (time.perf_counter() - start) / 60
    print(f"Training completed in {elapsed_minutes:.2f} minutes.")

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
