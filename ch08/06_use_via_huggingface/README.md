# Chapter 8 Bonus Material: Use Qwen3 with Hugging Face

This folder contains two ways to use the scratch [`Qwen3Model`](../../reasoning_from_scratch/qwen3.py) and compatible `.pth` checkpoints from this repository with Hugging Face `transformers`.

Both approaches let you use Hugging Face-style inference and training. The difference is whether you want a reusable Hugging Face model directory or a lighter local wrapper around the existing PyTorch model.

&nbsp;
## Approaches


&nbsp;
### 1) `wrapper_approach`

The [./wrapper_approach](./wrapper_approach) keeps the model as a local `.pth` file and wraps `Qwen3Model` in a thin local `PreTrainedModel` so it can work with parts of the Hugging Face API.

Use this approach if you want:

- the smallest amount of extra code
- local experimentation inside this repository
- `model.generate(...)` and `transformers.Trainer` without an export step
- to load the base model or chapter 6-8 checkpoints directly from `.pth`


&nbsp;
### 2) `export_approach`

The [./export_approach](./export_approach) converts the scratch Qwen3 weights or a compatible checkpoint into a Hugging Face-compatible model folder.

Use this approach if you want:

- a saved model directory with `config.json`, tokenizer files, and weights
- `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM`
- a workflow that is closer to how Hugging Face models are usually packaged



&nbsp;
## Which To Use?

- Choose [wrapper_approach](wrapper_approach) for learning purposes and if the goal is a lighter local integration with `transformers`.
- Choose [export_approach](export_approach) if the goal is a creating a Hugging Face model package and optimizing computational performance.