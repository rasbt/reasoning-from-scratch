# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from reasoning_from_scratch.ch03 import (
    render_prompt,
    extract_final_candidate,
    grade_answer,
    load_model_and_tokenizer,
    load_tokenizer_only,
)
from reasoning_from_scratch.ch04 import top_p_filter
from reasoning_from_scratch.ch06 import (
    load_math_train,
)
from reasoning_from_scratch.qwen3 import KVCache, Qwen3Model, QWEN_CONFIG_06_B

SCRIPT_NAME = Path(__file__).stem
LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_outputs.txt"
METRICS_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.txt"
CSV_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.csv"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / SCRIPT_NAME


def setup_distributed(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def shard_data(data, rank, world_size):
    if world_size <= 1:
        return data
    return data[rank::world_size]


def get_model_state_dict(model):
    if isinstance(model, FSDP):
        full_state_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_config
        ):
            return model.state_dict()
    return model.state_dict()


@torch.no_grad()
def sample_responses_batched(
    model,
    tokenizer,
    prompt,
    device,
    batch_size,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt), device=device, dtype=torch.long
    )
    prompt_len = prompt_ids.numel()
    input_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)

    base_model = model.module if isinstance(model, FSDP) else model
    cache = KVCache(n_layers=base_model.cfg["n_layers"])
    base_model.reset_kv_cache()
    logits = model(input_ids, cache=cache)[:, -1]

    eos_id = tokenizer.eos_token_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    generated_steps = []
    for _ in range(max_new_tokens):
        step_logits = logits
        if temperature and temperature != 1.0:
            step_logits = step_logits / temperature

        probas = torch.softmax(step_logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas, num_samples=1)

        eos_tok = next_token.new_full((batch_size, 1), eos_id)
        next_token = torch.where(
            finished.view(batch_size, 1), eos_tok, next_token
        )

        generated_steps.append(next_token)

        finished = finished | (next_token.squeeze(1) == eos_id)
        if torch.all(finished):
            break

        logits = model(next_token, cache=cache)[:, -1]

    if generated_steps:
        gen_tokens = torch.cat(generated_steps, dim=1)
    else:
        gen_tokens = torch.empty((batch_size, 0), dtype=input_ids.dtype, device=device)

    results = []
    for idx in range(batch_size):
        row_tokens = gen_tokens[idx]
        eos_pos = (row_tokens == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            row_tokens = row_tokens[: eos_pos[0]]

        full_token_ids = torch.cat([prompt_ids, row_tokens], dim=0)
        gen_text = tokenizer.decode(row_tokens.tolist())
        results.append((full_token_ids, prompt_len, gen_text))

    return results


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return selected[prompt_len - 1:].sum()


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # Require \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def compute_grpo_loss(
    model,
    tokenizer,
    example,
    device,
    num_rollouts=4,
    batch_size=None,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    if batch_size is None or batch_size <= 0:
        batch_size = num_rollouts

    remaining = num_rollouts
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch = sample_responses_batched(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            batch_size=current_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for token_ids, prompt_len, text in batch:
            logp = sequence_logprob(model, token_ids, prompt_len)
            reward = reward_rlvr(text, example["answer"])

            roll_logps.append(logp)
            roll_rewards.append(reward)
            samples.append(
                {
                    "text": text,
                    "reward": reward,
                    "gen_len": token_ids.numel() - prompt_len,
                }
            )
        remaining -= current_batch

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    logps = torch.stack(roll_logps)

    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
    }


def append_sample_logs(step_idx, samples, max_samples=3):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Step {step_idx}] sample outputs\n")
        for i, sample in enumerate(samples[:max_samples]):
            text = sample["text"].replace("\n", "\\n")
            f.write(
                f"  {i+1}) reward={sample['reward']:.3f} "
                f"len={sample['gen_len']}: {text}\n"
            )
        f.write("\n")


def append_step_metrics(
    step_idx,
    total_steps,
    loss,
    reward_avg,
    tokens_per_sec,
    avg_response_len,
    eval_acc=None,
):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Step {step_idx}/{total_steps}] "
            f"loss={loss:.4f} reward_avg={reward_avg:.3f} "
            f"tokens_per_sec={tokens_per_sec:.1f} "
            f"avg_response_len={avg_response_len:.1f}\n"
        )
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_LOG_PATH.exists():
        CSV_LOG_PATH.write_text(
            "step,total_steps,loss,reward_avg,tokens_per_sec,avg_response_len,eval_acc\n",
            encoding="utf-8",
        )
    with CSV_LOG_PATH.open("a", encoding="utf-8") as f:
        eval_acc_str = "" if eval_acc is None else f"{eval_acc:.6f}"
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{tokens_per_sec:.6f},{avg_response_len:.6f},{eval_acc_str}\n"
        )


def save_checkpoint(model, checkpoint_dir, step, suffix="", is_main=True):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = checkpoint_dir / f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    state_dict = get_model_state_dict(model)
    if not is_main:
        return None
    if not state_dict:
        return None
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def train_rlvr_grpo(
    model,
    tokenizer,
    math_data,
    device,
    steps=None,
    num_rollouts=9,
    batch_size=None,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=CHECKPOINT_DIR,
    is_main=True,
):
    if steps is None:
        steps = len(math_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    try:
        for step in range(steps):
            step_start = time.perf_counter()
            current_step = step + 1
            example = math_data[step % len(math_data)]
            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            optimizer.zero_grad()
            stats["loss_tensor"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_time = time.perf_counter() - step_start
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            tokens_per_sec = step_tokens / step_time if step_time > 0 else 0.0
            if is_main:
                append_step_metrics(
                    current_step,
                    steps,
                    stats["loss"],
                    reward_avg,
                    tokens_per_sec,
                    avg_response_len,
                )

                if current_step % 10 == 0:
                    append_sample_logs(current_step, stats["samples"])

            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                    is_main=is_main,
                )
                if is_main and ckpt_path is not None:
                    print(f"Saved checkpoint to {ckpt_path}")

            if is_main:
                print(
                    f"[Step {current_step}/{steps}] "
                    f"loss={stats['loss']:.4f} "
                    f"reward_avg={reward_avg:.3f} "
                    f"tok/sec={tokens_per_sec:.1f} "
                    f"avg_resp_len={avg_response_len:.1f}"
                )
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
            is_main=is_main,
        )
        if is_main and ckpt_path is not None:
            print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model


def main_worker(rank, world_size, args):
    is_main = rank == 0

    setup_distributed(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    if args.seed is not None and str(args.seed).strip().lower() != "none":
        torch.manual_seed(int(args.seed))

    try:
        math_data_full = load_math_train()
        total_steps = args.steps if args.steps is not None else len(math_data_full)
        math_data = shard_data(math_data_full, rank, world_size)
        if not math_data:
            raise ValueError("Sharded dataset is empty on this rank.")

        if args.checkpoint_path:
            tokenizer = load_tokenizer_only(which_model="base")
            model = Qwen3Model(QWEN_CONFIG_06_B)
            state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            target_dtype = model.cfg["dtype"]
            model = model.to(device=device, dtype=target_dtype)
        else:
            model, tokenizer = load_model_and_tokenizer(
                which_model="base", device=device, use_compile=False
            )
            target_dtype = model.cfg["dtype"]
            model = model.to(device=device, dtype=target_dtype)

        model = FSDP(model, device_id=device, use_orig_params=True)

        trained = train_rlvr_grpo(
            model=model,
            tokenizer=tokenizer,
            math_data=math_data,
            device=device,
            steps=total_steps,
            num_rollouts=args.num_rollouts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            is_main=is_main,
        )

        if is_main and torch.cuda.is_available():
            max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")

        final_state = get_model_state_dict(trained)
        if is_main and final_state:
            torch.save(final_state, CHECKPOINT_DIR / "qwen3-0.6B-rlvr-grpo.pth")

        dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RLVR GRPO on the MATH dataset."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps (default: len(math_data)).",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=8,
        help="Number of rollouts per step.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of rollouts to generate in a batch (default: num_rollouts).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per rollout.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling cutoff.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Random seed (int) or None to disable seeding.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to resume training from.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for FSDP (default: 1).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP requires CUDA.")
    if args.num_gpus < 2:
        raise ValueError("--num_gpus must be >= 1")
    if args.num_gpus > torch.cuda.device_count():
        raise ValueError("Requested --num_gpus exceeds available GPUs.")

    mp.spawn(main_worker, args=(args.num_gpus, args), nprocs=args.num_gpus)
