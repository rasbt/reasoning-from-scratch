# Downloading and Using Training Checkpoints

This folder explains how to download and use the chapter 8 distillation checkpoints from the Hugging Face model hub at [https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints).

The checkpoints are plain PyTorch `state_dict` files for the `reasoning_from_scratch` package. They are not Hugging Face Transformers checkpoints.

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---

&nbsp;
## Available Checkpoint Folders

- `ch08_distill_deepseek_r1`: the 3 DeepSeek-R1 distillation checkpoints used for rows 3-5 in [`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb)
- `ch08_distill_qwen3_235b_a22b`: the 3 Qwen3 235B A22B distillation checkpoints used for rows 6-8 in [`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb)

The checkpoints are hosted in:

- [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints)

&nbsp;
## Downloading a Checkpoint

Use `download_qwen3_distill_checkpoints(...)` from [`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py):

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir="qwen3",
)
```

&nbsp;
## Which Tokenizer to Use

Use the reasoning tokenizer for:

- `ch08_distill_deepseek_r1`
- `ch08_distill_qwen3_235b_a22b`

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
    download_qwen3_distill_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir=local_dir,
)
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
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
## Qwen3 Example

For `ch08_distill_qwen3_235b_a22b`, use the same helper with the other `distill_type`:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

download_qwen3_distill_checkpoints(
    distill_type="qwen3_235b_a22b",
    step="05746",
    out_dir="qwen3",
)
```

&nbsp;
## Available Steps

Available saved steps for `deepseek_r1`:

- `06682`
- `13364`
- `20046`

Available saved steps for `qwen3_235b_a22b`:

- `05746`
- `11492`
- `17238`
