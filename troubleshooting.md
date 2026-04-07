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

it indicates that your Python runtime may be lacking some C++ header files required for compiling the model for CPU usage.

You could for example check if the file exists: `ls -l /usr/include/python3.12/Python.h`.

If it doesn't exist, you could then try to install a different Python runtime via

```bash
sudo apt-get install -y python3.12-dev build-essential
```

Or you could disable the C++ requirements in PyTorch before calling `torch.compile`:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = False

compiled_model = torch.compile(model)
```

Also see [#192](https://github.com/rasbt/reasoning-from-scratch/issues/192) for more context.


&nbsp;
### Windows CPU: `fatal error C1083` with `algorithm` or `omp.h`

If you are on Windows and `torch.compile()` fails with

```text
fatal error C1083: Cannot open include file: 'algorithm': No such file or directory
```

or

```text
fatal error C1083: Cannot open include file: 'omp.h': No such file or directory
```

the problem is usually the local Windows compiler / OpenMP setup used by TorchInductor (rather than the code in this book / repository).

A reader reported the following tips on a CPU-only Intel system in the [forum](https://livebook.manning.com/forum?product=raschka2&comment=583365):

- Upgrading PyTorch resolved the missing `algorithm` header.
- But the missing `omp.h` header remained.
- Using a fallback backend such as `"eager"` or `"aot_eager"` allowed the code to run.

For example:

```python
compiled_model = torch.compile(model, backend="eager")

# or

compiled_model = torch.compile(model, backend="aot_eager")
```

Note that this is a workaround, not a full fix. It can help, but it does not use the full TorchInductor compilation path, so speedups may be smaller than with a fully working `torch.compile()`. 

**But also please keep in mind that torch.compile is not essential for this book and you can feel free to skip the section entirely.**

Anyways, if you are trying to make it work, before spending a lot of time debugging, it can be helpful to run a minimal sanity check first:

```python
import torch

device = "cpu"  # or "xpu"

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo = torch.compile(foo)
out = opt_foo(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device))
print(out.shape)
```

If this small example already fails, the issue is likely your PyTorch / compiler setup rather than the model code in this book / repository.

For additional setup tips, also see [Using `torch.compile()` on Windows](ch02/04_torch-compile-windows/README.md) and the PyTorch [Windows CPU/XPU guide](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html). And, as mentioned before, if `torch.compile()` remains unstable on your system, it is fine to skip it for the book examples.



&nbsp;
## Chapter 6

&nbsp;
### Corrupted Checkpoints

In `train_rlvr_grpo` (Chapter 6), a `Ctrl+C` triggers the `KeyboardInterrupt` handler to save a `-interrupt` checkpoint. If you press `Ctrl+C` a second time before the save completes, it can interrupt `torch.save` mid-write and leave a truncated `.pth` file. Wait for the `-interrupt` checkpoint message before exiting.

Corrupted model checkpoints usually raise load errors or fail during evaluation; another telltale sign is that they are much smaller than the expected ~1.5 GB.

&nbsp;
## Other Issues

For other issues, please feel free to open a new GitHub [Issue](https://github.com/rasbt/reasoning-from-scratch/issues).
