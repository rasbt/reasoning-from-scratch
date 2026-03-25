# Appendix G: Building a Chat Interface



This folder contains code for running a ChatGPT-like user interface to interact with the LLMs used and/or developed in this book, as shown below.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



To implement this user interface, we use the open-source [Chainlit Python package](https://github.com/Chainlit/chainlit).

&nbsp;
## Step 1: Install dependencies

First, we install the `chainlit` package and dependency:

```bash
pip install chainlit
```

Or, if you are using `uv`:

```bash
uv add chainlit
```



&nbsp;

## Step 2: Run `app` code

This folder contains 2 files:

1. [`qwen3_chat_interface.py`](qwen3_chat_interface.py): This file loads and uses the Qwen3 0.6B model in thinking mode.
2. [`qwen3_chat_interface_multiturn.py`](qwen3_chat_interface_multiturn.py): The same as above, but configured to remember the message history.

(Open and inspect these files to learn more.)

Run one of the following commands from the terminal to start the UI server:

```bash
chainlit run qwen3_chat_interface.py
```

or, if you are using `uv`:

```bash
uv run chainlit run qwen3_chat_interface.py
```

Running one of the commands above should open a new browser tab where you can interact with the model. If the browser tab does not open automatically, inspect the terminal command and copy the local address into your browser address bar (usually, the address is `http://localhost:8000`).

## Using a custom checkpoint

Since `chainlit run ...` owns the command-line arguments, these scripts read a custom checkpoint path from the `CHECKPOINT_PATH` environment variable instead of `argparse`.

Terminal example:

```bash
CHECKPOINT_PATH=/absolute/path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run qwen3_chat_interface.py
```

Notes:

- Keep `WHICH_MODEL` in the script aligned with the tokenizer the checkpoint expects.
- The chapter 8 checkpoints from [`ch08/05_download_training_checkpoints`](../../ch08/05_download_training_checkpoints) use the reasoning tokenizer, so use `WHICH_MODEL = "reasoning"`.
- When `CHECKPOINT_PATH` is set, the script only downloads the tokenizer into `LOCAL_DIR`; it does not re-download the default model weights.
