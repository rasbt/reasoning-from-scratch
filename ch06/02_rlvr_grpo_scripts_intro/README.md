# Chapter 6: Training Reasoning Models with Reinforcement Learning

&nbsp;

&nbsp;
## Bonus materials

- [rlvr_grpo_original.py](rlvr_grpo_original.py): script that implements the original GRPO algorithm to train a reasoning model using reinforcement learning with verifiable rewards (RLVR)
  - The algorithm was used by [DeepSeek R1](https://arxiv.org/abs/2501.12948) and originally proposed in the [DeepSeekMath](https://arxiv.org/abs/2402.03300) paper
- [rlvr_grpo_original_no_kl.py](rlvr_grpo_original_no_kl.py): same as above but without the KL divergence term (as recommended in [DAPO](https://arxiv.org/abs/2503.14476), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [Olmo 3](https://arxiv.org/abs/2512.13961), and others)
  - The KL divergence term ensures that the trained model doesn't deviate too much from the original model, but it can hurt performance (especially on math tasks)
  - The same can be achieved by setting `--kl_coeff 0.0` in the `rlvr_grpo_original.py` script, but this script, without the KL term, is simpler and uses less memory
  - This script implements the same code as in chapter 6; chapter 7 introduces the KL term


The script imports some functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid code duplication. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.) However, in this case, the code also reimplements the core functions from the chapter itself to allow for easier inspection and modification.



<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---


&nbsp;

|      | Method                    | Step | Max tokens | Num rollouts | MATH-500 Acc | Avg # of tokens |
| ---- | ------------------------- | ---- | ---------- | ------------ | ------------ | --------------- |
| 1    | Base (chapter 3)          | -    |            |              | 15.2%        | 78.85           |
| 2    | Reasoning (chapter 3)     | -    |            |              | 48.2%        | 1369.79         |
| 3    | GRPO original             | 50   | 512        | 8            | 33.4%        | 910.33          |
| 4    | GRPO original             | 100  | 512        | 8            | 0.4%         | 1168.05         |
| 5    | GRPO original but no KL   | 50   | 512        | 8            | 47.4%        | 586.11          |
| 6    | GRPO original but no KL   | 100  | 512        | 8            | 44.0%        | 555.95          |
| 7    | GRPO (Olmo 3 mod.)        | 50   | 512        | 8            | 46.4%        | 601.61          |
| 8    | GRPO (Olmo 3 mod.)        | 100  | 512        | 8            | 45.4%        | 589.51          |
| 9    | GRPO (DeepSeek V3.2 mod.) | 50   | 512        | 8            | 44.2%        | 618.49          |
| 10   | GRPO (DeepSeek V3.2 mod.) | 100  | 512        | 8            | 45.2%        | 676.96          |

Checkpoints are saved every 50 steps. If you KeyboardInterrupt a script, it will also save the last step as a checkpoint.

Note that the training only allows up to 512 generated tokens (max tokens in the table above) to make it more accessible regarding required compute memory.

However, the evaluation script (same method as in chapter 3) allows up to 2048 generated tokens, and the "Avg # of tokens" column in the table above measures how many tokens are used on average over the MATH-500 test dataset. (The training is done on the 12,000 examples in the MATH dataset that don't overlap with the MATH-500 test set. See [https://github.com/rasbt/math_full_minus_math500](https://github.com/rasbt/math_full_minus_math500) for more details.)

**Row 1**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

**Row 2**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

**Rows 3 & 4**

```bash
uv run rlvr_grpo_original.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

Then, to evaluate the model, run the `evaluate_math500.py` script on the generated checkpoint. For instance,

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path checkpoints/rlvr_grpo_original/qwen3-0.6B-rlvr-grpo-step00050.pth
```

**Rows 5 & 6**

```bash
uv run rlvr_grpo_original_no_kl.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**Rows 7 & 8**

```bash
uv run ../../ch06/02_rlvr_grpo_scripts_original/rlvr_grpo_olmo3.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**Rows 9 & 10**

```bash
uv run ../../ch06/02_rlvr_grpo_scripts_original/rlvr_grpo_deepseek_v32.py \
--num_rollouts 8 \
--max_new_tokens 512 
```


<br>

If you are low on RAM, consider lowering the number of rollouts (`--num_rollouts`) or response length (`--max_new_tokens`). The table below lists some resource requirements for reference.



| num_rollouts | max_new_tokens | Required RAM (GB) |
| ------------ | -------------- | ----------------- |
| 8            | 1024           | 30.50 GB          |
| 8            | 512            | 20.31 GB          |
| 8            | 256            | 15.60 GB          |
| 4            | 1024           | 12.80 GB          |
| 4            | 512            | 14.60 GB          |
| 4            | 256            | 10.59 GB          |


Please note that lowering the number of tokens or rollouts will likely negatively affect the performance. If you are using a low rollout number, you can somewhat improve the training stability by increasing `--accum_steps` from 1 to 2 or 4 (gradient accumulation); however, this will require more compute time. 

Note that the original ("vanilla") GRPO method with these settings is not very stable for more than 50 steps, and you may want to consider the improved versions in chapter 7 if you want to train for more than 50 steps.


<br>

Note that the original GRPO algorithm can be improved in several ways to stabilize and improve the training, which is the topic of the [next chapter](../../ch07).
