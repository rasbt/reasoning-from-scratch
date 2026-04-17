# Chapter 8 Bonus Material: Use Qwen3 via a Local Hugging Face Wrapper

This folder shows how to use the scratch [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py) with Hugging Face `transformers` library by wrapping it in a thin local `PreTrainedModel` class for compatibility.

This lets you use:

- `model.generate(...)`
- `transformers.Trainer`

directly with local `.pth` model files from this repository, including the base Qwen3 weights and compatible checkpoints from chapters 6-8.

&nbsp;
## Files

- [hf_wrapper.py](hf_wrapper.py): local `PreTrainedModel` wrapper around the from-scratch `Qwen3Model` we use throughout the book
- [hf_inference.py](hf_inference.py): text generation using the wrapper and the repo's tokenizer
- [hf_trainer.py](hf_trainer.py): `Trainer` example using the wrapper and the chapter 8 distillation JSON format

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---

&nbsp;
## What This Wrapper Does

The wrapper keeps the model local to this repository and adapts it to the Hugging Face API.

Concretely, it:

- loads a local `.pth` model file directly into `Qwen3Model`
- wraps that model in a `PreTrainedModel` interface
- exposes a `forward(...)` method compatible with `Trainer`
- enables `model.generate(...)`
- keeps using the repository's `Qwen3Tokenizer`

Why? There were some readers curious about exploring the models further in `transformers`, which has more bells and whistles than the from-scratch code in this repo.

&nbsp;
## Limitations

This is a small, local wrapper around the scratch model.

Important implications:

- it is meant for environments where `reasoning_from_scratch` is installed
- it does not provide an `AutoTokenizer.from_pretrained(...)` workflow
- it does not create a reusable model directory with `config.json` and tokenizer files
- generation is kept intentionally simple, so it recomputes the full prefix instead of adapting the scratch KV cache to Hugging Face cache classes; if you want full support, you'd need to switch to the [../export_approach](../export_approach)

Note that these constraints keep the code short and focused on local use inside this repository.

&nbsp;
## Step 1: Install dependencies

This guide uses Hugging Face Transformers in addition to the repository dependencies. 

```bash
pip install transformers accelerate
```

Or, if you are using `uv`:

```bash
uv add --dev transformers accelerate
```

&nbsp;
## Step 2: Run local wrapped inference

To run the base model through the wrapper, use:

```bash
  uv run hf_inference.py \
    --tokenizer_kind base \
    --prompt "If x + 7 = 19, what is x?"
```

To run the reasoning variant, use:

```bash
  uv run hf_inference.py \
    --tokenizer_kind reasoning \
    --prompt "If x + 7 = 19, what is x?"
```

To run a local checkpoint instead:

```bash
uv run hf_inference.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --prompt "If x + 7 = 19, what is x?"
```

If `--model_path` is omitted, the script downloads the default base or reasoning model for the selected `--tokenizer_kind`. If `--model_path` is provided, it can point to the base Qwen3 `.pth` file or to any compatible checkpoint produced in chapters 6-8.

Internally, the inference script:

1. builds the local wrapper model
2. loads the selected `.pth` model file into the wrapped `Qwen3Model`
3. tokenizes the prompt with the repo's tokenizer
4. calls `model.generate(...)`

&nbsp;
## Step 3: Continue training with `Trainer`

The same wrapper can also be used with `transformers.Trainer`:

```bash
uv run hf_trainer.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1
```

As with inference, `--model_path` can point to the base Qwen3 weights or to a compatible chapter 6-8 checkpoint.

The trainer keeps the same answer-only objective used elsewhere in chapter 8:

- prompt tokens are masked out
- only answer tokens contribute to the loss
- reasoning mode wraps teacher traces as `<think>...</think>`

The input JSON format matches the distillation data generated in [../../02_generate_distillation_data](../../02_generate_distillation_data).

&nbsp;
