# Chapter 5: Inference-Time Scaling via Self-Refinement


&nbsp;
## Bonus materials

- [self_refinement_math500.py](self_refinement_math500.py): standalone script to evaluate models with self-refinement on the MATH-500 dataset

The script impors functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid code duplication. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.)



<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---



&nbsp;

## Self-refinement

The [`self_refinement_math500.py`](self_refinement_math500.py) script implements the self-refinement method from chapter 5.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch05/CH05_F21_raschka.webp" width=600>

&nbsp;

The table below compares this approach with the baselines from chapter 3:

| #  | Method          | Scorer    | Iterations | Model     | Accuracy | Time      |
|----|-----------------|-----------|------------|-----------|----------|-----------|
| 1  | Baseline (ch03) | -         | -          | Base      | 15.2%    | 10.1 min  |
| 2  | Self-refinement | None      | 1          | Base      | 25.0%    | 84.8 min  |
| 3  | Self-refinement | None      | 2          | Base      | 22.0%    | 165.4 min |
|    |                 |           |            |           |          |           |
| 4  | Self-refinement | Heuristic | 1          | Base      | 21.6%    | 84.7 min  |
| 5  | Self-refinement | Heuristic | 2          | Base      | 20.8%    | 151.4 min |
|    |                 |           |            |           |          |           |
| 6  | Self-refinement | Logprob   | 1          | Base      | 21.4%    | 85.3 min  |
| 7  | Self-refinement | Logprob   | 2          | Base      | 22.0%    | 165.3 min |
|    |                 |           |            |           |          |           |
| 8  | Self-refinement | Logp-ex   | 1          | Base      | 20.4%    | 85.0 min  |
| 9  | Self-refinement | Logp-ex   | 2          | Base      | 21.2%    | 160.2 min |
|    |                 |           |            |           |          |           |
| 10 | Baseline (ch03) | -         | -          | Reasoning | 48.2%    | 182.1 min |
| 11 | Self-refinement | None      | 1          | Reasoning | 56.6%    | 498.8 min |
| 12 | Self-refinement | None      | 2          | Reasoning | 56.6%    | 713.9 min |
|    |                 |           |            |           |          |           |
| 13 | Self-refinement | Heuristic | 1          | Reasoning | 57.8%    | 498.6 min |
| 14 | Self-refinement | Heuristic | 2          | Reasoning | 57.8%    | 713.9 min |
|    |                 |           |            |           |          |           |
| 15 | Self-refinement | Logprob   | 1          | Reasoning | 48.4%    | 499.7 min |
| 16 | Self-refinement | Logprob   | 2          | Reasoning | 48.6%    | 753.0 min |

The accuracy values and runtimes shown in the table were computed on all 500 samples in the MATH-500 test set using a "cuda" GPU (DGX Spark).

The following codes give instructions on how to run the self-consistency experiments in rows 4-12 (replace `uv run` with `python` if you are not a `uv` user).

**Row 2:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**Row 3:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**Row 4:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**Row 5:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**Row 6:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**Row 7:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```

**Row 8:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob_extract"
```

**Row 9:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob_extract"
```

**Row 11:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**Row 12:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**Row 13:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**Row 14:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**Row 15:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**Row 16:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```