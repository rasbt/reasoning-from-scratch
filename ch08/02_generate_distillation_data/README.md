# Chapter 8 Bonus Material: Generate Distillation Data

This folder contains scripts to generate teacher outputs for math problems, which can be used as distillation data for training a smaller reasoning model, as covered in chapter 8.

&nbsp;
## Files

- `generate_with_ollama.py`: Uses the Ollama to generate the model answers for distillation. This script is recommended if  you want to distill from smaller models you can run locally, for example, Qwen3 4B, gpt-oss 20B, DeepSeek R1 32B, etc. 
- `generate_with_openrouter.py`: Uses models through the OpenRouter API to generate the model answers distillation. This is recommended when using larger models like DeepSeek R1 (671B) or Kimi K2.5 (1T) that are too large to be run locally.
- `math_train_sample.json`: Small sample dataset for quick sanity checks.

&nbsp;
## Input Data Format

Both scripts expect a JSON file via `--math_json`. At a minimum, each object should have:

- `problem` (string): The math question.
- `answer` (string): Ground-truth answer.

Extra keys such as `level`, `type`, and `unique_id` are ignored. You can look at the [math_train_sample.json](math_train_sample.json) file for an example structure, which is based on the [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json) we used in chapters 6 and 7. 

To apply it to the full 12,000 samples, simply download the [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json) and pass it into the scripts via `--math_json math_full_minus_math500.json`. Note that this will take a long time, so I recommend truncating the file to a few hundred or a thousand examples.


&nbsp;
## Output Format

Both scripts write a JSON array where each row looks like:

```
{
  "problem": "...",             # The original "problem"
  "gtruth_answer": "...",       # The original "answer"
  "message_thinking": "...",    # The model's thinking stream
  "message_content": "..."      # The model's final answer
}
```

Notes:

- The original `"answer"` from the input JSON file was renamed to `"gtruth_answer"` to avoid ambiguity (because "answer" is a general term that could also refer to the model's answer).
- Files are written incrementally after each sample, so it's possible to work with intermediate files or interrupt the run.
- The scripts have a `--resume` option to continue an interrupted run.

&nbsp;
## 1. Local generation with Ollama


- Ollama is an open-source application to run LLMs efficiently.
- It is a wrapper around llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)), which implements LLMs in pure C/C++ to maximize efficiency.
- Note that it is a tool for using LLMs to generate text (inference), not training or finetuning LLMs.

&nbsp;
### 1.1 Ollama setup


- Before running the code below, install ollama by visiting [https://ollama.com](https://ollama.com) and following the instructions (for instance, clicking on the "Download" button and downloading the ollama application for your operating system).
- For macOS and Windows users, click on the ollama application you downloaded; if it prompts you to install the command line usage, say "yes".
- Linux users can use the installation command provided on the ollama website.
- There are 3 ways we can run ollama on our computer:


&nbsp;
**1. `ollama serve`**

- This runs the ollama backend as a server, usually on `http://localhost:11434`. It doesn't load a model until we call it through the API. This is what we want if we want to use ollama through Python.

&nbsp;
**2. `ollama run deepseek-r1:8b`**

- This is a convenience wrapper. If the server is not already running, it will start it, then download the model (the first time), and drop us into an interactive terminal where we can chat with the model. Behind the scenes, it uses the same server API.
- The `deepseek-r1:8b` model will require approximately 30 GB of RAM with the `--max_new_tokens 8192` token setting.
  - If you have more RAM, I recommend trying larger models for higher-quality answers, for example, `deepseek-r1:32b` (it requires approximately 60 GB)
  - If you have less RAM, try selecting a smaller model; you can find a list of smaller R1 models [here](https://ollama.com/library/deepseek-r1). Also, instead of using a DeepSeek model, feel free to use the "Search model" field on the [Ollama website](https://ollama.com/) to select other models you might find interesting.
  - Alternatively, you could also reduce `--max_new_tokens 8192` to `--max_new_tokens 2048` to reduce RAM usage, but this might truncate some answers prematurely.

&nbsp;
**3. Ollama desktop app**

- This runs the same backend automatically and provides a GUI on top of it (as shown in the figure above).
  It also applies defaults (system prompt, temperature, stop sequences), which can explain why answers look different from raw API usage.

&nbsp;
### 1.2 Local data generation with Ollama

```bash
uv run generate_with_ollama.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file sample_ollama_outputs.json
```

If you are not a `uv` user, replace `uv run` with `python`.

The expected output should be as follows:

```
Loading model: deepseek-r1:8b
Using CUDA:0
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 3.2 min

Wrote 5 rows to: /home/rasbt/reasoning-from-scratch-codedev/ch08/sample_ollama_outputs.json
```

The entries in the resulting [sample_ollama_outputs.json](sample_ollama_outputs.json) file are as follows:

```json
  {
    "problem": "A rectangular band formation...",
    "gtruth_answer": "98",
    "message_thinking": "I need to find the largest number of...",
    "message_content": "The function is continuous..."
  },
```

The `"message_thinking"` field contains the chain-of-thought explanation, and `"message_content"` contains the final answer. For instance, these could be connected as

```python
complete_answer = f"<think>{data['message_thinking']}</think>\n\n{data['message_content']}"
```

That is,

```
"<think>I need to find the largest number of...</think>

The function is continuous..."
```

&nbsp;
### 1.3 Ollama troubleshooting

Below are some common issues when running the Ollama data generation script.

&nbsp;
#### 1.3.1 Ollama not running

If you see an error like 

```
Loading model: deepseek-r1:32b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt//reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: <urlopen error [Errno 111] Connection refused>
```

make sure `ollama serve` is running (in a different terminal tab).

&nbsp;
#### 1.3.2 Ollama model not downloaded

If you see the following error:

```
Loading model: deepseek-r1:8b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: HTTP 404 from Ollama at http://localhost:11434/api/chat: {"error":"model 'deepseek-r1:8b' not found"}
```

this means the model hasn't been downloaded yet. In this case, run `ollama run deepseek-r1:8b` in a separate terminal, which will download the model and start a chat. You can try the model in the chat and then exit via `\bye`.


&nbsp;
## 2. Hosted generation with OpenRouter

Ollama is convenient if you want to run models locally. However, there are several large models (such as the 671B-parameter DeepSeek R1 model) that are too large to be run locally on our hardware. For these cases, I recommend [OpenRouter](https://openrouter.ai), which lets us use a large variety of both open-weight and proprietary LLMs, hosted in the cloud, through a ChatGPT-like API. 

As of this writing, [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) costs \$0.70 per 1 million input tokens and \$2.50 per 1 million output tokens. Note that there are many cheaper (and faster) models on OpenRouter; even the newer [DeepSeek V3.2](https://openrouter.ai/deepseek/deepseek-v3.2) model only costs $0.40 per 1 million output tokens.

That being said, let's do a simple cost calculation. Given an average input prompt length of 11 tokens and an average response length of 1524 tokens, it costs about $3.82 to generate the answers to 1000 MATH questions.

Here is the breakdown:

- Total input tokens: 11 × 1000 = 11,000 
- Total output tokens: 1524 × 1000 = 1,524,000 
- Input cost: $\frac{11{,}000}{1{,}000{,}000} \times 0.70 = \$0.0077$
- Output cost: $\frac{1{,}524{,}000}{1{,}000{,}000} \times 2.50 = \$3.81$
- Total cost: $\$3.81 + \$0.0077 \approx \$3.82$

&nbsp;
### 2.1 OpenRouter setup

The setup is quite simple. All you have to do is create an account at [OpenRouter](https://openrouter.ai/), generate an API key under [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys), and save the API key in a secure location (e.g., a password manager).


&nbsp;
### 2.2 Data generation with OpenRouter

The OpenRouter script works similarly to the Ollama script, except that we prepend the API key as an environment variable:

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file sample_openrouter_outputs.json
```

If you are not a `uv` user, replace `uv run` with `python`.

The output looks as follows:

```
Loading model: deepseek/deepseek-r1
Using OpenRouter API: https://openrouter.ai/api/v1/chat/completions
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 2.2 min

Wrote 5 rows to: /Users/sebastian/Developer/reasoning-from-scratch/ch08/02_generate_distillation_data/sample_openrouter_outputs.json
```

The [sample_openrouter_outputs.json](sample_openrouter_outputs.json) output file has the same structure as the one generated by the Ollama script.

**Tip:** If you are generating a lot of data, running this sequential distillation process can be very slow (e.g., ~100 hours for 12,000 answers with DeepSeek R1). In this case, I recommend running multiple parallel data generation threads via `--num_processes`. For instance, using `--num_processes 50` with the DeepSeek R1 models cuts the runtime from 100 hours down to approximately 2 hours.

