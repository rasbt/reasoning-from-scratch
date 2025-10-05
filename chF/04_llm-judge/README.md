
# LLM-as-a-judge

This bonus material implements an LLM-as-a-judge approach, where gpt-oss:20b (via the open-source Ollama library) evaluates Qwen3 0.6B base and reasoning variants on MATH-500.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-f/Appendix_F_F06_raschka.webp" width="500px">





- Ollama is an open-source application to run LLMs efficiently
- It is a wrapper around llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)), which implements LLMs in pure C/C++ to maximize efficiency
- Note that it is a to ol for using LLMs to generate text (inference), not training or finetuning LLMs
- Before running the code below, install ollama by visiting [https://ollama.com](https://ollama.com) and following the instructions (for instance, clicking on the "Download" button and downloading the ollama application for your operating system)
- For macOS and Windows users, click on the ollama application you downloaded; if it prompts you to install the command line usage, say "yes"
- Linux users can use the installation command provided on the ollama website
- There are 3 ways we can run ollama on our computer:



**1. `ollama serve`**

- This runs the ollama backend as a server, usually on `http://localhost:11434`. It doesn't load a model until we call it through the API. This is what we want if we want to use ollama through Python.

**2. `ollama run gpt-oss:20b`**

- This is a convenience wrapper. If the server is not already running, it will start it, then download the model (the first time), and drop us into an interactive terminal where we can chat with the model. Behind the scenes, it uses the same server API.

**3. Ollama desktop app**

- This runs the same backend automatically and provides a GUI on top of it (as shown in the figure above).
It also applies defaults (system prompt, temperature, stop sequences), which can explain why answers look different from raw API usage.



## Usage



The options and defaults are shown below. 

<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---



```bash
uv run ollama-judge.py --help
usage: ollama-judge.py [-h] [--device DEVICE]
                       [--which_model {base,reasoning}]
                       [--dataset_size DATASET_SIZE]
                       [--max_new_tokens MAX_NEW_TOKENS]
                       [--url URL]
                       [--judge_model JUDGE_MODEL]

options:
  -h, --help            show this help message and
                        exit
  --device DEVICE       Device e.g., "cpu",
                        "cuda", "cuda:0", "mps".
  --which_model {base,reasoning}
                        Candidate variant to use.
                        Defaults to "base".
  --dataset_size DATASET_SIZE
                        Number of MATH-500
                        examples to evaluate.
                        Default: 10
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens for
                        candidate generation.
                        Default: 2048
  --url URL             Ollama chat endpoint for
                        the judge. Default: "http:
                        //localhost:11434/api/chat
                        "
  --judge_model JUDGE_MODEL
                        Judge model name (Ollama).
                        Used only for scoring.
                        Default: "gpt-oss:20b"
```



**Base model**

```bash
➜  uv run ollama-judge.py
Using Apple Silicon GPU (MPS)
Model: base
Device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=1
[3/10] score=5
[4/10] score=5
[5/10] score=3
[6/10] score=5
[7/10] score=5
[8/10] score=3
[9/10] score=5
[10/10] score=1

Summary
-------
Average score: 3.800 over 10 example(s)
Counts: 1:2 2:0 3:2 4:0 5:6
```

**Reasoning model**

```bash
➜  uv run ollama-judge.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Model: reasoning
Device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=5
[3/10] score=5
[4/10] score=5
[5/10] score=4
[6/10] score=5
[7/10] score=5
[8/10] score=1
[9/10] score=5
[10/10] score=3

Summary
-------
Average score: 4.300 over 10 example(s)
Counts: 1:1 2:0 3:1 4:1 5:7
```

