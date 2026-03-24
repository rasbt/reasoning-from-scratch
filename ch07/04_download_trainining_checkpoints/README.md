# Downloading and Using Training Checkpoints

This folder explains how to download and use the chapter 7 training checkpoints from Hugging Face at [https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints).

The checkpoints are plain PyTorch `state_dict` files for the `reasoning_from_scratch` package. They are not Hugging Face Transformers checkpoints.

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---

&nbsp;
## Available Checkpoint Folders

- `7_3_plus_tracking`: GRPO checkpoints with additional metric tracking
- `7_4_plus_clip_ratio`: GRPO checkpoints with clipped policy ratios
- `7_5_plus_kl`: GRPO checkpoints with a KL term
- `7_6_plus_format_reward`: GRPO checkpoints with an explicit format reward for `<think>` tags

The checkpoints are hosted in:

- [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints)

&nbsp;
## Downloading a Checkpoint

Use `download_qwen3_grpo_checkpoints(...)` from [`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py):

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir="qwen3",
)
```

&nbsp;
## Which Tokenizer to Use

Use the base tokenizer for:

- `7_3_plus_tracking`
- `7_4_plus_clip_ratio`
- `7_5_plus_kl`

Use the reasoning tokenizer for:

- `7_6_plus_format_reward`

The reason is that `7_6_plus_format_reward` was trained from the reasoning model and expects the reasoning chat formatting.

&nbsp;
## Usage Example

The example below downloads a checkpoint, downloads the matching tokenizer, loads the model, and generates text with `generate_text_basic_stream_cache` from chapter 2:

```python
from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_grpo_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir=local_dir,
)
download_qwen3_small(kind="base", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(tokenizer_file_path=local_dir / "tokenizer-base.json")
model = Qwen3Model(QWEN_CONFIG_06_B)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

prompt = render_prompt("Solve: If x + 7 = 19, what is x?")
input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
):
    token_id = token.squeeze(0).item()
    print(tokenizer.decode([token_id]), end="", flush=True)
```

&nbsp;
## Format-Reward Example

For `7_6_plus_format_reward`, switch to the reasoning tokenizer:

```python
from pathlib import Path

from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
)

local_dir = Path("qwen3")
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
```

&nbsp;
## Chapter 6 Example

The same helper also supports the original chapter 6 no-KL checkpoint:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050", out_dir="qwen3")
```

&nbsp;
## Available Checkpoints

Section mapping:

- `no_kl`: chapter 6 baseline from the original no-KL GRPO setup
- `tracking`: section 7.3 in the main chapter
- `clip_ratio`: section 7.4 in the main chapter
- `kl`: section 7.5 in the main chapter
- `format_reward`: section 7.6 in the main chapter

Available saved steps:

- `no_kl`: `00050`, `00100`, `00500`, `01000`, `01500`, `03000`, `05000`, `09000`
- `tracking`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `clip_ratio`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `kl`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `format_reward`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
