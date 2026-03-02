# Troubleshooting Guide

This is a placeholder for providing help and tips on common issues encountered in the book. Right now, this page is intentionally left blank as no known issues have been reported.

&nbsp;
## File Download Issues

Please use [this discussion page](https://github.com/rasbt/reasoning-from-scratch/discussions/145) if you have any issues with file downloads.

&nbsp;
## JupyterLab scrolling bug

If you are viewing the notebook code in JupyterLab rather than VSCode, note that JupyterLab (in its default setting) has had scrolling bugs in recent versions. My recommendation is to go to Settings -> Settings Editor and change the "Windowing mode" to "none" (as illustrated below), which seems to address the issue.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)


&nbsp;
## Chapter 2

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

Orm you could disable the C++ requirements in PyTorch before calling `torch.compile`:

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
