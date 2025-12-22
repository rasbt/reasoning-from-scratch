# Chapter 6: Training Reasoning Models with Reinforcement Learning

&nbsp;

&nbsp;
## Bonus materials

- [rlvr_grpo_original.py](rlvr_grpo_original.py): script that implements the original GRPO algorithm to train a reasoning model using reinforcement learning with verifiable rewards (RLVR)

The script imports some functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid code duplication. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.) However, in this case, the code also reimplements the core functions from the chapter itself to allow for easier inspection and modification.



<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---


&nbsp;

|   | Method                    | Step | Max tokens | Num rollouts | MATH-500 Acc | Avg # of tokens |
|---|---------------------------|------|------------|--------------|--------------|-----------------|
| 1 | Base (chapter 3)          | -    |            |              | 15.2%        | 78.85           |
| 2 | Reasoning (chapter 3)     | -    |            |              | 48.2%        | 1369.79         |
| 3 | GRPO original             | 50   | 512        | 8            | 33.4%        | 910.33          |
| 4 | GRPO (Olmo 3 mod.)        | 50   | 512        | 8            | 46.4%        | 601.61          |
| 5 | GRPO (DeepSeek V3.2 mod.) | 50   | 512        | 8            | 44.2%        | 618.49          |

Checkpoints are saved every 50 steps. If you KeyboardInterrupt a script, it will also save the last step as a checkpoint.

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

**Row 3**

```bash
uv run rlvr_grpo_original.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**Row 4**

```bash
uv run ../../ch06/02_rlvr_grpo_scripts_original/rlvr_grpo_olmo3.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**Row 5**

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

Note that the original ("vanilla") GRPO method with these settings is not very stable for more than 50 steps, and you may want to consider the improved versions in chapter 6 if you want to train for more than 50 steps.



**Row 1**

<br>

Note that the original GRPO algorithm can be improved in several ways to stabilize and improve the training, which is the topic of the [next chapter](../ch07).