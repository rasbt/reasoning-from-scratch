# Chapter 4: Improving Reasoning with Inference-Time Scaling


&nbsp;
## Bonus materials

- [cot_prompting_math500.py](cot_prompting_math500.py): standalone script to evaluate models with chain-of-thought prompting on the MATH-500 dataset
- [self_consistency_math500.py](self_consistency_math500.py): standalone script to evaluate models with self-consistency sampling on the MATH-500 dataset

Both evaluation scripts import functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid code duplication. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.)



<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---



&nbsp;

## Chain-of-thought prompting

The [`cot_prompting_math500.py`](self_consistency_math500.py) script implements the chain-of-thought prompting method from chapter 4.

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F04_raschka.webp" width=600>

&nbsp;

The table below compares this approach (row 3) with the baselines from chapter 3:

|    | Method                                       | Model     | Accuracy | Time       |
|----|----------------------------------------------|-----------|----------|------------|
| 1  | Baseline (chapter 3), greedy decoding        | Base      | 15.2%    | 10.1 min   |
| 2  | Baseline (chapter 3), greedy decoding        | Reasoning | 48.2%    | 182.1 min  |
| 3  | Chain-of-thought prompting ("CoT")           | Base      | 40.6%    | 84.5 min   |

The accuracy values and runtimes shown in the table were computed on all 500 samples in the MATH-500 test set using a "cuda" GPU (DGX Spark).

To run the experiment in row one, use:

```bash
python cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

Or, with `uv:`


```bash
uv run cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

For additional options, use the `--help` flag.



&nbsp;
## Self-consistency sampling

The [`self_consistency_math500.py`](self_consistency_math500.py) script implements the sampling method from chapter 4.

(Optionally, there is a [`self_consistency_math500_batched.py`](self_consistency_math500_batched.py) variant, which executes all `--num_samples` as a batch for faster processing. Note that this requires more compute memory though.)

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F17_raschka.webp" width=600>

&nbsp;

The table below compares this approach (row 4-12) with the baselines from chapter 3 (rows 1-2):

|      | Method                                    | Model     | Accuracy | Time      |
| ---- | ----------------------------------------- | --------- | -------- | --------- |
| 1    | Baseline (chapter 3), greedy decoding     | Base      | 15.2%    | 10.1 min  |
| 2    | Baseline (chapter 3), greedy decoding     | Reasoning | 48.2%    | 182.1 min |
| 3    | Chain-of-thought prompting ("CoT")        | Base      | 40.6%    | 84.5 min  |
| 4    | Temperature and top-p ("Top-p")           | Base      | 17.8%    | 30.7 min  |
| 5    | "Top-p" + Self-consistency (n=3)          | Base      | 29.6%    | 97.6 min  |
| 6    | "Top-p" + Self-consistency (n=5)          | Base      | 27.8%    | 116.8 min |
| 7    | "Top-p" + Self-consistency (n=10)         | Base      | 31.6%    | 300.4 min |
| 8    | "Top-p" + "CoT"                           | Base      | 33.4%    | 129.2 min |
| 9    | Self-consistency (n=3) + "Top-p" + "CoT"  | Base      | 42.2%    | 211.6 min |
| 10   | Self-consistency (n=5) + "Top-p" + "CoT"  | Base      | 48.0%    | 452.9 min |
| 11   | Self-consistency (n=10) + "Top-p" + "CoT" | Base      | 52.0%    | 862.6 min |
| 12   | Self-consistency (n=3) + "Top-p" + "CoT"  | Reasoning | 55.2%    | 544.4 min |

The accuracy values and runtimes shown in the table were computed on all 500 samples in the MATH-500 test set using a "cuda" GPU (DGX Spark).

The following codes give instructions on how to run the self-consistency experiments in rows 4-12 (replace `uv run` with `python` if you are not a `uv` user).

**Row 4:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500
```

**Row 5:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500
```

**Row 6:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500
```

**Row 7:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500
```

**Row 8:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Row 9:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Row 10:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Row 11:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**Row 12:**

```bash
uv run self_consistency_math500.py \
    --which_model "reasoning" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```


For additional options, use the `--help` flag.

