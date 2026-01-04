# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import copy
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    evaluate_math500_stream,
    render_prompt,
    extract_final_candidate,
    grade_answer,
    load_model_and_tokenizer,
    load_math500_test,
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
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / SCRIPT_NAME


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
        )

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas.cpu(), num_samples=1).to(device)

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break
        generated.append(next_token.item())
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


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


def grpo_step(
    model,
    ref_model,
    tokenizer,
    example,
    device,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    kl_coeff=0.02,
):
    roll_logps, roll_ref_logps, roll_rewards, samples = [], [], [], []
    prompt = render_prompt(example["problem"])

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        logp = sequence_logprob(model, token_ids, prompt_len)
        with torch.no_grad():
            ref_logp = sequence_logprob(ref_model, token_ids, prompt_len)
        reward = reward_rlvr(text, example["answer"])

        roll_logps.append(logp)
        roll_ref_logps.append(ref_logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    logps = torch.stack(roll_logps)
    ref_logps = torch.stack(roll_ref_logps).detach()

    pg_loss = -(advantages.detach() * logps).mean()
    kl_loss = kl_coeff * torch.mean(logps - ref_logps)
    loss = pg_loss + kl_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
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


def append_step_metrics(step_idx, total_steps, loss, reward_avg, kl_loss):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Step {step_idx}/{total_steps}] "
            f"loss={loss:.4f} reward_avg={reward_avg:.3f} kl={kl_loss:.4f}\n"
        )


def append_eval_metrics(step_idx, acc, correct, total):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Eval step {step_idx}] math500_acc={acc:.4f} "
            f"({correct}/{total})\n"
        )


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = checkpoint_dir / f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train_rlvr_grpo(
    model,
    ref_model,
    tokenizer,
    math_data,
    math500_eval_data,
    device,
    steps=None,
    num_rollouts=8,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    kl_coeff=0.02,
    lr=1e-5,
    accum_steps=1,
    checkpoint_every=50,
    checkpoint_dir=CHECKPOINT_DIR,
    eval_on_checkpoint=True,
    eval_max_items=20,
):
    if steps is None:
        steps = len(math_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    current_step = 0
    try:
        for step in range(steps):
            current_step = step + 1
            example = math_data[step % len(math_data)]
            stats = grpo_step(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                kl_coeff=kl_coeff,
            )
            (stats["loss_tensor"] / accum_steps).backward()

            if current_step % accum_steps == 0 or current_step == steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            append_step_metrics(
                current_step, steps, stats["loss"], reward_avg, stats["kl_loss"]
            )

            if current_step % 10 == 0:
                append_sample_logs(current_step, stats["samples"])

            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")
                if eval_on_checkpoint and math500_eval_data:
                    was_training = model.training
                    model.eval()
                    subset = (
                        math500_eval_data[:eval_max_items]
                        if eval_max_items
                        else math500_eval_data
                    )
                    out_path = (
                        Path(checkpoint_dir)
                        / f"{SCRIPT_NAME}-step{current_step:05d}-math500.jsonl"
                    )
                    num_correct, num_examples, acc = evaluate_math500_stream(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        math_data=subset,
                        out_path=out_path,
                        max_new_tokens=max_new_tokens,
                        verbose=False,
                    )
                    append_eval_metrics(current_step, acc, num_correct, num_examples)
                    print(
                        f"MATH-500 eval @ step {current_step}: "
                        f"acc={acc:.3f} ({num_correct}/{num_examples})"
                    )
                    if was_training:
                        model.train()

            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"kl={stats['kl_loss']:.4f}"
            )
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model


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
        default=9,
        help="Number of rollouts per step.",
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
        "--kl_coeff",
        type=float,
        default=0.02,
        help="KL penalty coefficient.",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to resume training from.",
    )
    args = parser.parse_args()

    device = get_device()

    math_data = load_math_train()
    if args.checkpoint_path:
        tokenizer = load_tokenizer_only(which_model="base")
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model="base", device=device, use_compile=False
        )

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    trained = train_rlvr_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        math_data=math_data,
        math500_eval_data=load_math500_test(),
        device=device,
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        kl_coeff=args.kl_coeff,
        accum_steps=args.accum_steps,
    )

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")

    torch.save(trained.state_dict(), CHECKPOINT_DIR/"qwen3-0.6B-rlvr-grpo.pth")
