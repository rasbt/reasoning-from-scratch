# Chapter 8 Bonus Material: Use Qwen3 From-Scratch Code via Hugging Face Transformers

This folder shows how to convert the scratch [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py) and any compatible `.pth` checkpoint created via chapters 6-8 into a Hugging Face Transformers-compatible folder, and how to run it with Hugging Face inference functions and the `Trainer`.

The export is implemented as a custom `transformers` architecture, so it works with the standard Hugging Face APIs such as `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `model.generate(...)`, and `Trainer`. But because it is custom code, load it with `trust_remote_code=True`.

&nbsp;
## Files

- [hf_export.py](hf_export.py): converts the scratch Qwen3 weights or a saved `.pth` checkpoint into a Hugging Face model folder
- [hf_inference.py](hf_inference.py): runs text generation with `AutoModelForCausalLM`
- [hf_trainer.py](hf_trainer.py): continues training an exported model with `transformers.Trainer` on the chapter 8 distillation JSON format
- [hf_qwen3.py](hf_qwen3.py): custom Hugging Face `PretrainedConfig` and `PreTrainedModel` implementation for the exported Qwen3 architecture

The export scripts keep the Hugging Face-specific model code locally in this folder and import shared utilities from the [`reasoning_from_scratch`](../../../reasoning_from_scratch) package for the chapter 3 prompt template, RoPE helpers, and Qwen3 download functions. (See [chapter 2 setup instructions](../../../ch02/02_setup-tips/python-instructions.md) for installation details.)

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---

&nbsp;
## Step 1: Install dependencies

This guide uses Hugging Face Transformers in addition to the repository dependencies. For `transformers.Trainer`, you also need `accelerate`.

```bash
pip install transformers accelerate
```

Or, if you are using `uv`:

```bash
uv add --dev transformers accelerate
```

&nbsp;
## Step 2: Export the vanilla Qwen3 model

To export the original base model as a Hugging Face folder, run:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind "base"  # or use "reasoning"
```

If you already have the raw `.pth` model and tokenizer locally, you can avoid a download:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind base \
  --model_path ../../../ch02/01_main-chapter-code/qwen3/qwen3-0.6B-base.pth \
  --tokenizer_path ../../../ch02/01_main-chapter-code/qwen3/tokenizer-base.json
```

The same also works with the chapter 6-8 checkpoint `.pth` files.

The exported folder will contain:

- `config.json`
- `generation_config.json`
- tokenizer files
- model weights (by default as `model.safetensors`)
- a copied custom Python module required by `trust_remote_code=True`

&nbsp;
### What the export code does

The exporter does not translate the model into the official Hugging Face Qwen implementation, and it does not modify the learned weights. What it does is the following:

1. it builds a custom Hugging Face `PretrainedConfig` and `PreTrainedModel` that reproduce the from-scratch `Qwen3Model` architecture
2. it loads the original `.pth` `state_dict` directly into that custom Hugging Face model without renaming or reshaping the trainable parameters
3. it saves the result using the standard Hugging Face folder format so `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `generate(...)`, and `Trainer` can load it

The main things that are added or wrapped are:

- a Hugging Face config file (`config.json`)
- a Hugging Face model class with a `forward(...)` signature compatible with `transformers`
- Hugging Face tokenizer files
- Hugging Face generation metadata (`generation_config.json`)
- a custom Python source file that `trust_remote_code=True` loads

There is one small extra detail during export. I.e., the from-scratch checkpoints only save the trainable weights, while the Hugging Face export also bundles the precomputed RoPE `cos` and `sin` buffers so reloading the exported model is numerically consistent.

For `--tokenizer_kind reasoning`, the exporter also attaches the reasoning chat template to the tokenizer, so inference scripts can automatically wrap prompts in the expected chat format. 

Because this custom Hugging Face module imports the installed [`reasoning_from_scratch`](../../../reasoning_from_scratch) package, the exported folder is compatible as long as that package is installed in the Python environment.

&nbsp;
## Step 3: Export a saved checkpoint

The same exporter also works for chapter 8 distillation checkpoints or any other compatible `.pth` file produced by this repo.

For example, if you trained a chapter 8 checkpoint with the reasoning tokenizer:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-distill \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --tokenizer_kind reasoning
```

Important notes:

- Use `--tokenizer_kind reasoning` for chapter 8 distillation checkpoints and other checkpoints trained with the reasoning tokenizer.
- Use `--tokenizer_kind base` for checkpoints trained with the base tokenizer.
- If the matching tokenizer JSON is already on disk, you can pass it via `--tokenizer_path` to avoid a download.

&nbsp;
## Step 4: Run Hugging Face inference

After export, run inference with `AutoTokenizer` and `AutoModelForCausalLM`:

```bash
uv run hf_inference.py \
  --model_dir hf-qwen3-base \
  --prompt "If x + 7 = 19, what is x?"
```

Internally, the script:

1. loads the exported model with `trust_remote_code=True`
2. formats the prompt with the same chapter 3 math prompt template
3. applies the reasoning chat wrapper automatically when the exported model uses the reasoning tokenizer
4. calls `model.generate(...)`

If you prefer the raw Hugging Face API directly, the equivalent pattern is:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-qwen3-base")
config = AutoConfig.from_pretrained("hf-qwen3-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "hf-qwen3-base",
    trust_remote_code=True,
)
```

&nbsp;
## Step 5: Continue training with `Trainer`

You can continue training an exported checkpoint with Hugging Face `Trainer` on the same JSON format used in [`../../04_train_with_distillation`](../../04_train_with_distillation).

Example:

```bash
uv run hf_trainer.py \
  --model_dir hf-qwen3-base \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1 \
  --save_steps 10
```

The script keeps the same answer-only training objective used in the scratch distillation code:

- prompt tokens are masked out of the loss
- only the distilled answer tokens contribute to the cross-entropy loss
- for reasoning exports, the script wraps teacher traces as `<think>...</think>` before the final answer



&nbsp;
## Loading the export elsewhere

Once exported, you can copy the folder to another machine or upload it to the Hugging Face Hub and load it there too, as long as `reasoning_from_scratch` is installed in that environment:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
```
