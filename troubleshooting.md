# Troubleshooting Guide

This page collects common issues and setup tips encountered while working through the book.

&nbsp;
## JupyterLab scrolling bug

If you are viewing the notebook code in JupyterLab rather than VSCode, note that JupyterLab (in its default setting) has had scrolling bugs in recent versions. My recommendation is to go to Settings -> Settings Editor and change the "Windowing mode" to "none" (as illustrated below), which seems to address the issue.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)


&nbsp;
## Chapter 2

&nbsp;
### File Download Issues

Please use [this discussion page](https://github.com/rasbt/reasoning-from-scratch/discussions/145) if you have any issues with file downloads.

The code downloads from the following Hugging Face locations, which you can also open manually in a browser to check whether your machine or network is blocking them:

- Chapter 2 model and tokenizer files: [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch/tree/main)
- Base model file: [qwen3-0.6B-base.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-base.pth)
- Base tokenizer file: [tokenizer-base.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-base.json)
- Reasoning model file: [qwen3-0.6B-reasoning.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-reasoning.pth)
- Reasoning tokenizer file: [tokenizer-reasoning.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-reasoning.json)
- Chapter 7 GRPO checkpoints: [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main)
- Chapter 8 distillation checkpoints: [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints/tree/main)

&nbsp;
#### SSL / proxy / certificate errors

If a model download fails with errors mentioning `SSL`, `CERTIFICATE_VERIFY_FAILED`, or `ProxyError`, the issue is often environmental rather than a missing file.

This is uncommon overall, but it can happen on work or school machines where a VPN, proxy, firewall, or antivirus product intercepts HTTPS traffic. In that case, try the following:

- Check whether the relevant Hugging Face URL listed above opens in your browser.
- If the tokenizer downloads but the `.pth` model file does not, the proxy may be blocking larger files or the `.pth` extension.
- Ask your IT team to allow the download or to make the proxy certificate trusted by Python.
- On some managed machines, readers reported success with `pip install pip-system-certs`, which makes Python use the operating system certificate store.

&nbsp;
### `InductorError: CppCompileError`
If you are a Linux user and see an `InductorError: CppCompileError: C++ compile error` when executing `torch.compile` containing the following lines:

```python
Python.h: No such file or directory
81 | #include <Python.h>
| ^~~~~~~~~~
compilation terminated.
```

it indicates they your Python runtime may be lacking some C++ header files required for compiling the model for CPU usage.

You could for example check if the file exists: `ls -l /usr/include/python3.12/Python.h`.

If it doesn't exist, you could then try to install a different Python runtime via

```bash
sudo apt-get install -y python3.12-dev build-essential
````

Or you could disable the C++ requirements in PyTorch before calling `torch.compile`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = False

compiled_model = torch.compile(model)
```

Also see [#192](https://github.com/rasbt/reasoning-from-scratch/issues/192) for more context.



&nbsp;
## Chapter 6

&nbsp;
### Corrupted Checkpoints

In `train_rlvr_grpo` (Chapter 6), a `Ctrl+C` triggers the `KeyboardInterrupt` handler to save a `-interrupt` checkpoint. If you press `Ctrl+C` a second time before the save completes, it can interrupt `torch.save` mid-write and leave a truncated `.pth` file. Wait for the `-interrupt` checkpoint message before exiting.

Corrupted model checkpoints usually raise load errors or fail during evaluation; another telltale sign is that they are much smaller than the expected ~1.5 GB.

&nbsp;
## Other Issues

For other issues, please feel free to open a new GitHub [Issue](https://github.com/rasbt/reasoning-from-scratch/issues).
