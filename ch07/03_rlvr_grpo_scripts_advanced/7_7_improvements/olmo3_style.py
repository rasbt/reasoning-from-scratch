# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import copy
import time
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
CSV_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.csv"
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
        next_token = torch.multinomial(probas, num_samples=1)

        token_id = next_token.item()
        generated.append(token_id)

        if (
            tokenizer.eos_token_id is not None
            and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids,
         torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


# Renamed from sequence_logprob_and_entropy to reflect token-level log-probs (DAPO)
# entropy now optional for logging.
def answer_token_logprobs(model, token_ids, prompt_len, return_entropy=False):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # Log-prob of the generated answer tokens (sum over answer steps)
    selected_answer_logprobs = selected[prompt_len - 1:]
    if not return_entropy:
        return selected_answer_logprobs

    # Entropy over the full vocab distribution at each answer step
    all_answer_logprobs = logprobs[:-1][prompt_len - 1:]
    if all_answer_logprobs.numel() == 0:  # Safeguard if the model immediately emits EOS token
        entropy_all_steps = logprobs.new_tensor(0.0)
    else:
        all_answer_probs = torch.exp(all_answer_logprobs)
        plogp = all_answer_probs * all_answer_logprobs    # elementwise p * log p
        step_entropy = -torch.sum(plogp, dim=-1)          # sum over vocab -> entropy per step
        entropy_all_steps = torch.mean(step_entropy)      # average over answer steps

    return selected_answer_logprobs, entropy_all_steps


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None  # Require \boxed{}
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def compute_grpo_loss_plus_kl(
    model,
    old_model,
    ref_model,
    tokenizer,
    example,
    device,
    num_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    clip_eps_low=0.2,
    clip_eps_high=10.0,
    kl_coeff=0.0,
):
    if kl_coeff and ref_model is None:
        raise ValueError("ref_model must be provided when kl_coeff is non-zero.")
    if old_model is None:
        old_model = model
    roll_old_logps, roll_ref_logps, roll_rewards, roll_entropies, samples = [], [], [], [], []
    roll_token_ids, roll_prompt_lens = [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()
    old_model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=old_model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        with torch.no_grad():
            old_logp, entropy = answer_token_logprobs(
                old_model, token_ids, prompt_len, return_entropy=True
            )
            if kl_coeff:
                ref_logp = answer_token_logprobs(ref_model, token_ids, prompt_len).sum()
            else:
                ref_logp = None
        reward = reward_rlvr(text, example["answer"])

        roll_old_logps.append(old_logp)
        if kl_coeff:
            roll_ref_logps.append(ref_logp)
        roll_rewards.append(reward)
        roll_entropies.append(entropy.item())
        roll_token_ids.append(token_ids)
        roll_prompt_lens.append(prompt_len)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )

    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = rewards - rewards.mean()  # (7) No standard deviation normalization (Dr. GRPO)
    zero_grad = rewards.max().item() == rewards.min().item()  # (1) Zero gradient signal filtering (DAPO)
    adv = advantages.detach()

    obj_terms = []
    ratio_terms = []
    new_logps_sum = []
    for idx, (token_ids, prompt_len) in enumerate(zip(roll_token_ids, roll_prompt_lens)):
        # (3) Token-level loss (DAPO)
        new_logp = answer_token_logprobs(model, token_ids, prompt_len)
        old_logp = roll_old_logps[idx]
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, max=1.0 + clip_eps_high)  # (6) Truncated importance sampling (Yao et al., 2025)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high
        )  # (5) Clip higher (DAPO)

        adv_i = adv[idx]
        unclipped = ratio * adv_i
        clipped = clipped_ratio * adv_i
        token_obj = torch.where(
            adv_i >= 0,
            torch.minimum(unclipped, clipped),
            torch.maximum(unclipped, clipped),
        )
        new_logps_sum.append(new_logp.sum())
        if token_obj.numel() > 0:
            obj_terms.append(token_obj)
            ratio_terms.append(ratio)

    if obj_terms:
        obj = torch.cat(obj_terms).mean()
        policy_ratio = torch.cat(ratio_terms).mean().item()
    else:
        obj = advantages.new_tensor(0.0)
        policy_ratio = 1.0
    new_logps = torch.stack(new_logps_sum) if new_logps_sum else advantages.new_zeros(0)
    if not obj_terms:
        zero_grad = True

    pg_loss = -obj.mean()
    if kl_coeff and new_logps.numel() > 0:
        ref_logps = torch.stack(roll_ref_logps).detach()
        kl_loss = kl_coeff * torch.mean(new_logps - ref_logps)
    else:
        kl_loss = torch.tensor(0.0, device=new_logps.device)
    loss = pg_loss + kl_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "policy_ratio": policy_ratio,
        "rewards": roll_rewards,
        "entropies": roll_entropies,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss,
        "zero_grad": zero_grad,
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
    kl_loss,
    policy_ratio,
    reward_avg,
    tokens_per_sec,
    avg_response_len,
    adv_avg,
    adv_std,
    entropy_avg,
    eval_acc=None,
):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    policy_ratio_str = (
        "" if policy_ratio is None else f" policy_ratio={policy_ratio:.2f}"
    )
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Step {step_idx}/{total_steps}] "
            f"loss={loss:.2f} reward_avg={reward_avg:.3f} "
            f"kl={kl_loss:.2f} "
            f"tokens_per_sec={tokens_per_sec:.1f} "
            f"avg_response_len={avg_response_len:.1f}"
            f" adv_avg={adv_avg:.2f}"
            f" adv_std={adv_std:.2f}"
            f" entropy_avg={entropy_avg:.2f}"
            f"{policy_ratio_str}\n"
        )
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_LOG_PATH.exists():
        CSV_LOG_PATH.write_text(
            "step,total_steps,loss,kl_loss,policy_ratio,reward_avg,tokens_per_sec,avg_response_len,adv_avg,adv_std,entropy_avg,eval_acc\n",
            encoding="utf-8",
        )
    with CSV_LOG_PATH.open("a", encoding="utf-8") as f:
        eval_acc_str = "" if eval_acc is None else f"{eval_acc:.6f}"
        policy_ratio_str = "" if policy_ratio is None else f"{policy_ratio:.6f}"
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{kl_loss:.6f},{policy_ratio_str},{reward_avg:.6f},"
            f"{tokens_per_sec:.6f},{avg_response_len:.6f},"
            f"{adv_avg:.6f},{adv_std:.6f},{entropy_avg:.6f},{eval_acc_str}\n"
        )


def append_eval_metrics(step_idx, acc, correct, total):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Eval step {step_idx}] math500_acc={acc:.2f} "
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
    num_rollouts=9,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    clip_eps_low=0.2,
    clip_eps_high=10.0,
    inner_epochs=2,
    kl_coeff=0.0,  # (4) No KL loss (DAPO, Dr. GRPO)
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=CHECKPOINT_DIR,
    eval_max_items=0,
    active_sampling=True,
    max_active_sampling_tries=8,
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
            old_model = copy.deepcopy(model).to(device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False
            stats = None

            for _ in range(inner_epochs):
                for attempt in range(max_active_sampling_tries):
                    example = math_data[(step + attempt) % len(math_data)]
                    stats = compute_grpo_loss_plus_kl(
                        model=model,
                        old_model=old_model,
                        ref_model=ref_model,
                        tokenizer=tokenizer,
                        example=example,
                        device=device,
                        num_rollouts=num_rollouts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        clip_eps_low=clip_eps_low,
                        clip_eps_high=clip_eps_high,
                        kl_coeff=kl_coeff,
                    )
                    if not (active_sampling and stats["zero_grad"]):  # (2) Active sampling (DAPO)
                        break
                if stats is None or stats["zero_grad"]:
                    continue
                optimizer.zero_grad()
                stats["loss_tensor"].backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            entropy_avg = torch.tensor(stats["entropies"]).mean().item()
            advantage_tensor = torch.tensor(stats["advantages"])
            adv_avg = advantage_tensor.mean().item()
            adv_std = advantage_tensor.std().item()
            policy_ratio = stats.get("policy_ratio")
            step_time = time.perf_counter() - step_start
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            tokens_per_sec = step_tokens / step_time if step_time > 0 else 0.0
            if current_step % 10 == 0:
                append_sample_logs(current_step, stats["samples"])

            eval_acc = None
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")
                if eval_max_items and math500_eval_data:
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
                    eval_acc = acc
                    append_eval_metrics(current_step, acc, num_correct, num_examples)
                    print(
                        f"MATH-500 eval @ step {current_step}: "
                        f"acc={acc:.3f} ({num_correct}/{num_examples})"
                    )
                    if was_training:
                        model.train()

            append_step_metrics(
                current_step,
                steps,
                stats["loss"],
                stats["kl_loss"],
                policy_ratio,
                reward_avg,
                tokens_per_sec,
                avg_response_len,
                adv_avg=adv_avg,
                adv_std=adv_std,
                entropy_avg=entropy_avg,
                eval_acc=eval_acc,
            )

            policy_ratio_str = (
                "" if policy_ratio is None else f"policy_ratio={policy_ratio:.2f} "
            )
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.2f} "
                f"kl={stats['kl_loss']:.2f} "
                f"reward_avg={reward_avg:.3f} "
                f"tok/sec={tokens_per_sec:.1f} "
                f"avg_resp_len={avg_response_len:.1f} "
                f"adv_avg={adv_avg:.2f} "
                f"adv_std={adv_std:.2f} "
                f"entropy_avg={entropy_avg:.2f} "
                f"{policy_ratio_str}"
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
        default=8,
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
        "--clip_eps_low",
        type=float,
        default=0.2,
        help="Lower PPO clip epsilon (DAPO).",
    )
    parser.add_argument(
        "--clip_eps_high",
        type=float,
        default=10.0,
        help="Upper PPO clip epsilon (DAPO clip-higher).",
    )
    parser.add_argument(
        "--inner_epochs",
        type=int,
        default=2,
        help="Number of inner update iterations per step.",
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.0,
        help="KL penalty coefficient (default disabled).",
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
        "--eval_on_checkpoint",
        type=int,
        default=0,
        help=(
            "Number of MATH-500 examples to evaluate at checkpoints "
            "(0 disables)."
        ),
    )
    args = parser.parse_args()

    if args.seed is not None and str(args.seed).strip().lower() != "none":
        torch.manual_seed(int(args.seed))
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

    if args.kl_coeff:
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        ref_model = None

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
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
        inner_epochs=args.inner_epochs,
        kl_coeff=args.kl_coeff,
        eval_max_items=args.eval_on_checkpoint,
    )

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), CHECKPOINT_DIR/"qwen3-0.6B-rlvr-grpo.pth")
