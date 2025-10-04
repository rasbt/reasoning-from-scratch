# Optimized Qwen3

The Qwen3 from-scratch implementation used in this book strikes a balance between being efficient (both on CPU and GPU) and lean while remaining easy to read by a human. 

As an alternative, you can use the optional `Qwen3Model` drop-in replacement, which is slightly more GPU-efficient. The optimized version in [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) (discussed further in Appendix C) differs from the baseline implementation in [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) in two key ways:

- It implements attention using PyTorch’s built-in `torch.nn.functional.scaled_dot_product` instead of a custom implementation.
- It introduces a modified `KVCache` that pre-allocates key/value tensors. This increases memory usage but avoids repeatedly allocating new storage during execution.


To explore the differences, I recommend opening [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) and [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py)  side by side and/or looking at a file-diff:

<br>

![](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/optimized-LLM/vscode.webp)

<br>

&nbsp;
## How to use

The optimized code can be used as drop-in replacement for the code used in the main chapters as shown below.

**Before:**

```python
from reasoning_from_scratch.qwen3 import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_cache
```


**After:**

```python
from reasoning_from_scratch.qwen3_optimized import Qwen3Model
from reasoning_from_scratch.qwen3_optimized import generate_text_basic_cache
```

&nbsp;
## How to run comparisons

To evaluate the performance on your system, you can use the [`compare_inference.py`](compare_inference.py) function contained in this folder:

```python
python compare_inference.py
```

or

```python
uv run compare_inference.py
```

Then, add the following flags:

- `--device`: Select the device, e.g., `cpu`, `mps`, or `cuda`
- `--cache`: Enables the KV cache
- `--compile`: Uses `torch.compile`
- `--reasoning`: Uses the Qwen3 reasoning variant instead of the base model. The base model generates approximately 50 tokens in response to the given prompt. The reasoning variant generates about 2000 tokens.
- `--optimize`: Uses the optimized model from `qwen3_optimized.py` instead of the standard model from `qwen3.py`.

<br>

&nbsp;
### Standard model



| Model    | Mode              | Command                         | Hardware        | Tokens/sec    | GPU Memory (VRAM) |
| -------- | ----------------- | ------------------------------- | --------------- | ------------- | ----------------- |
| qwen3.py | Regular           | --device cpu                    | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | Regular compiled  | --device cpu --compile          | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | KV cache          | --device cpu --cache            | Mac Mini M4 CPU | 28            | -                 |
| qwen3.py | KV cache compiled | --device cpu --compile --cache  | Mac Mini M4 CPU | 68            | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Regular           | --device mps                    | Mac Mini M4 GPU | 17            | -                 |
| qwen3.py | Regular compiled  | --device mps --compile          | Mac Mini M4 GPU | InductorError | -                 |
| qwen3.py | KV cache          | --device mps --cache            | Mac Mini M4 GPU | 18            | -                 |
| qwen3.py | KV cache compiled | --device mps --compile --cache  | Mac Mini M4 GPU | InductorError | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Regular           | --device cuda                   | NVIDIA H100 GPU | 51            | 1.55 GB           |
| qwen3.py | Regular compiled  | --device cuda --compile         | NVIDIA H100 GPU | 164           | 1.81 GB           |
| qwen3.py | KV cache          | --device cuda --cache           | NVIDIA H100 GPU | 48            | 1.52 GB           |
| qwen3.py | KV cache compiled | --device cuda --compile --cache | NVIDIA H100 GPU | 141           | 1.81 GB           |

<br>

&nbsp;
### Optimized model


| Model              | Mode              | Command                                     | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ------------------ | ----------------- | ------------------------------------------- | --------------- | ---------- | ----------------- |
| qwen3_optimized.py | Regular           | --optimized --device cpu                    | Mac Mini M4 CPU | 5          | -                 |
| qwen3_optimized.py | Regular compiled  | --optimized --device cpu --compile          | Mac Mini M4 CPU | 7          | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device cpu --cache            | Mac Mini M4 CPU | 49         | -                 |
| qwen3_optimized.py | KV cache compiled | --optimized --device cpu --compile --cache  | Mac Mini M4 CPU | 51         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Regular           | --optimized --device mps                    | Mac Mini M4 GPU | 21         | -                 |
| qwen3_optimized.py | Regular compiled  | --optimized --device mps --compile          | Mac Mini M4 GPU | NameError  | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device mps --cache            | Mac Mini M4 GPU | 29         | -                 |
| qwen3_optimized.py | KV cache compiled | --optimized --device mps --compile --cache  | Mac Mini M4 GPU | 38         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Regular           | --optimized --device cuda                   | NVIDIA H100 GPU | 55         | 1.50 GB           |
| qwen3_optimized.py | Regular compiled  | --optimized --device cuda --compile         | NVIDIA H100 GPU | 173        | 1.81 GB           |
| qwen3_optimized.py | KV cache          | --optimized --device cuda --cache           | NVIDIA H100 GPU | 56         | 5.85 GB           |
| qwen3_optimized.py | KV cache compiled | --optimized --device cuda --compile --cache | NVIDIA H100 GPU | 177        | 5.85 GB           |

<br>

Comparing the 2 tables above, we can see that the optimized variant is clearly faster in terms of tokens/second in most cases. 

However, note that the unoptimized version is faster (68 tok/sec) than the optimized version (51 tok/sec) when using the compiled version with KV cache.

The optimized version also uses more base RAM (5.85 GB with KV Cache) than the unoptimized version (1.5 GB). This is because it pre-allocates the tensors holding the KV values for the maximum supported context length. (So, when running the unoptimized version on a prompt with 41k context length, the RAM usage would be approximately similar.)

**Perhaps the best recommendation is to use the unoptimized version (with `--cache` and `--compile`) when using a CPU. When using a GPU, use the optimized version (with `--cache` and `--compile`).**