# Chapter 8 Bonus Material: Train with Distillation

This folder contains a simple distillation script for training the Qwen3 0.6B model on teacher-generated reasoning traces, as covered in chapter 8.

&nbsp;
## Files

- [distill.py](distill.py): Trains Qwen3 0.6B on JSON-formatted distillation data (more on the format in the next section).
  - By default, it trains the base model with the base tokenizer
  - If you pass `--use_think_tokens`, it uses the reasoning tokenizer and wraps the reasoning trace as `<think>...</think>` before the final answer similar to how it's done in chapter 8
  - After each epoch, it saves a checkpoint to `checkpoints/distill/` and appends training metrics to `logs/distill_metrics.csv`
  - If you initialize from `--checkpoint_path` (optional) instead of the base model, you can continue an already existing checkpoint
- [distill_batched.py](distill_batched.py): Batched version of the script above.
  - It uses the padding-aware batched Qwen3 implementation so examples of different lengths can be trained together
  - It adds a `--batch_size` argument to process multiple examples per optimization step
  - It saves checkpoints to `checkpoints/distill_batched/` and appends metrics to `logs/distill_batched_metrics.csv`
  - Of course, note that the batched variant uses much more GPU memory (depending on the batch size)

The script imports shared functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid duplicating the model-loading and prompt-formatting code. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.)


<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---


&nbsp;
## Input data format

The input is the JSON output produced by [`../02_generate_distillation_data`](../02_generate_distillation_data). Each row should look like this:

```json
{
  "problem": "Compute 1/2 + 1/6.",
  "gtruth_answer": "2/3",
  "message_thinking": "I will rewrite the fractions with a common denominator.",
  "message_content": "The final answer is \\boxed{\\tfrac{2}{3}}."
}
```

For training, only the following fields are used:

- `problem`: inserted into the same math prompt template used in chapter 3
- `message_content`: required; used as the supervised target answer
- `message_thinking`: optional; if present, it is prepended before `message_content`

Rows with missing or malformed fields are skipped automatically, and examples longer than `--max_seq_len` are filtered out before the train/validation split.


&nbsp;
## Example run

For a quick sanity check, you can train on a small sample generated in the previous folder:

```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1
```

This will:

- load the base Qwen3 0.6B weights
- tokenize the prompt/answer pairs
- reserve 1 example for validation
- save a checkpoint after each epoch in `checkpoints/distill/`
- write CSV metrics to `logs/distill_metrics.csv`

If you want to train with explicit reasoning tags and the reasoning tokenizer instead, add `--use_think_tokens`:

```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1 \
  --use_think_tokens
```

If you want to train in batches instead, run:

```bash
uv run distill_batched.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --batch_size 2 \
  --log_every 1
```


&nbsp;
## Useful options

```bash
uv run distill.py --help
```

Important arguments:

- `--data_path`: path to the distillation JSON file
- `--dataset_size`: truncate the dataset before splitting (`0` uses all rows)
- `--validation_size`: absolute number of validation examples
- `--epochs`: number of passes over the training split
- `--batch_size`: number of examples per optimization step in `distill_batched.py`
- `--lr`: AdamW learning rate
- `--max_seq_len`: drops examples whose prompt + answer sequence is longer than this limit
- `--checkpoint_path`: initialize from an earlier distillation checkpoint
- `--grad_clip_norm`: optional gradient clipping
- `--use_think_tokens`: switch to the reasoning tokenizer and `<think>...</think>` formatting

See the "Experiments" section below for hands-on examples.

&nbsp;

## Evaluating a distilled checkpoint

After training, you can evaluate a checkpoint on MATH-500 using the chapter 3 evaluation script.

If you trained without `--use_think_tokens`, evaluate it as a `base` model:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model base \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```

**Important:** If you trained with `--use_think_tokens`, evaluate it as a `reasoning` model so the reasoning tokenizer is used:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model reasoning \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```


&nbsp;
## Experiments

The distillation datasets used in chapter 8 are available from my Hugging Face repo at [rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill). In chapter 8, they are loaded via a helper that downloads partitions, e.g.,

````python
from reasoning_from_scratch.ch08 import load_distill_data

_ = load_distill_data(
    partition="deepseek-r1-math-train.json",
    local_path="deepseek-r1-math-train.json"
)
_ = load_distill_data(
    partition="qwen3-235b-a22b-math-train.json",
    local_path="qwen3-235b-a22b-math-train.json"
)
````



For the experiments below, I used the `deepseek-r1-math-train.json` and `qwen3-235b-a22b-math-train.json` files from that dataset collection.


&nbsp;

|      | Teacher data                         | Epoch | MATH-500 Acc | Final val loss |
| ---- | ------------------------------------ | ----- | ------------ | -------------- |
| 1    | Base (chapter 3)                     | -     | 15.2%        | -              |
| 2    | Reasoning (chapter 3)                | -     | 48.2%        | -              |
| 3    | DeepSeek R1 distillation data        | 1     | 30.6%        | 0.5436         |
| 4    | DeepSeek R1 distillation data        | 2     | 32.4%        | 0.5349         |
| 5    | DeepSeek R1 distillation data        | 3     | 33.6%        | 0.5343         |
| 6    | Qwen3 235B A22B distillation data    | 1     | 45.0%        | 0.4043         |
| 7    | Qwen3 235B A22B distillation data    | 2     | 43.8%        | 0.3963         |
| 8    | Qwen3 235B A22B distillation data    | 3     | 44.2%        | 0.3948         |

The training takes about 30 min on an H100 and about 3 hours on a DGX Spark and uses up to 15 GB RAM.

Below are the code snippets to reproduce the results reported in the table.

&nbsp;
**Row 1**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

&nbsp;
**Row 2**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

&nbsp;
**Rows 3, 4, & 5**

```bash
uv run distill.py \
--data_path deepseek-r1-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

Then, to evaluate the epoch checkpoints, run:

&nbsp;
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run-1/checkpoints/distill/qwen3-0.6B-distill-step06682-epoch1.pth
```

For row 4 and row 5, replace the checkpoint path with `...step13364-epoch2.pth` and `...step20046-epoch3.pth`, respectively.

&nbsp;
**Rows 6, 7, & 8**

```bash
uv run distill.py \
--data_path qwen3-235b-a22b-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

Then, to evaluate the epoch checkpoints, run:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run_11/checkpoints/distill/qwen3-0.6B-distill-step05746-epoch1.pth
```

For row 7 and row 8, replace the checkpoint path with `...step11492-epoch2.pth` and `...step17238-epoch3.pth`, respectively.
