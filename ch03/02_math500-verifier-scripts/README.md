# Chapter 3: Evaluating Reasoning Models

&nbsp;


&nbsp;
## Bonus materials

- [evaluate_math500.py](evaluate_math500.py): standalone script to evaluate models on the MATH-500 dataset
- [evaluate_math500_batched.py](evaluate_math500_batched.py): same as above, but processes multiple examples in parallel during generation (for higher throughput)

Both evaluation scripts import functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid code duplication. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.)



<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---



&nbsp;

## `evaluate_math500.py` usage

Run with:

```bash
python evaluate_math500.py
```

Or, with `uv:`


```bash
uv run evaluate_math500.py
```

Options:

```bash
uv run evaluate_math500.py --help

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to use: "auto" (default) or any torch device string
                        (e.g., "cpu", "cuda", "cuda:0", "mps").
  --which_model {base,reasoning}
                        Model variant to load (default: "base").
  --dataset_size DATASET_SIZE
                        Number of MATH-500 examples to evaluate (default: 10).
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens to generate (default: 2048).
  --compile             Enable torch.compile.
  --verbose             Print per-sample correctness while evaluating.
```

&nbsp;
## `evaluate_math500_batch.py` usage

This version extends batching to generation itself, enabling parallel decoding:

```bash
uv run evaluate_math500_batched.py --help
```

Extra options:

```bash
  --batch_size BATCH_SIZE
                        Number of examples to generate in parallel (default: 4).
  --disable_efficient_mode
                        Use a simpler batched inference method. Slower and more
                        memory-intensive, but easier to debug.
```

&nbsp;


**Implementation note:**
By default, batched generation halts for sequences that emit a stop token. With `--disable_efficient_mode`, all sequences continue until the longest finishes. This affects compute efficiency only, not qualitative results, since tokens after the stop token are discarded.

&nbsp;

**Tip (MPS devices):**
Run with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run evaluate_math500_batched.py
```

Some PyTorch ops used in efficient batched inference are not yet supported on MPS. As a fallback, you can also use `--disable_efficient_mode`.



&nbsp;

- `evaluate_math500.py --dataset_size 500`


| Device / Dataset size                       | Base model | Reasoning model |
| ------------------------------------------- | ---------- | --------------- |
| **Mac Mini M4 CPU** (500 examples, sequential | 43.6 min | Didn't run (too hot)           |
| **Mac Mini M4 GPU** (500 examples, sequential) | 37.5 min | Didn't run (too hot) |
| **DGX Spark** (500 examples, sequential) | 10.0 min  | 182.2 min      |
| **H100 GPU** (500 examples, sequential) | 13.3 min  | 185.4 min      |

<br>
<br>

- `evaluate_math500_batched.py --dataset_size 500 --batch_size 128`

| Device / Dataset size                                        | Base model | Reasoning model |
| ------------------------------------------------------------ | ---------- | --------------- |
| **Mac Mini M4 CPU** (500 examples, batched, `--batch_size 128`) | 167.2 min | Didn't run (too hot)           |
| **Mac Mini M4 GPU** (500 examples, batched, `--batch_size 128`) | Error*     | Error           |
| **DGX Spark** (500 examples, batched, `--batch_size 128`)    | 16.3 min  | 119.3 min      |
| **H100 GPU** (500 examples, batched, `--batch_size 128`)     | 3.3 min   | 14.6 min       |



- The accuracy of the base model  is 15.6% (78/500); the accuracy of the reasoning model is 50.8% (254/500).
