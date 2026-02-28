# Python Setup Recommendations

The code in this book is largely self-contained, and I have made an effort to minimize external dependencies. However, to keep the book accessible, readable, and well under 2000 pages, a few Python packages are necessary.

This section introduces two beginner-friendly methods for installing the required packages so you can run the code examples. 

There are, of course, many other ways to install and manage Python packages. If you are an experienced Python user and already have your own setup or preferences, feel free to skip this section.

If neither of the two options below works for you, please do not hesitate to reach out, for example, by opening a [Discussion](https://github.com/rasbt/reasoning-from-scratch/discussions).

&nbsp;
## Option 1: Using `pip` (built-in, works everywhere)

If you are using a recent version of Python already, you can install packages using the built-in `pip` installer. 

I used Python 3.12 for this book. However, older versions like 3.11 and 3.10 will also work fine. You can check your Python version by running:

```bash
python --version
```

If you are using Python 3.9 or older, consider installing the latest from [python.org](https://www.python.org/downloads/) or using a tool like [`pyenv`](https://github.com/pyenv/pyenv) to manage versions. However, if you are installing a new Python version, please make sure that it is supported by PyTorch by checking the recommendation on the [official PyTorch website](https://pytorch.org/get-started/locally/). PyTorch typically lags a few months behind the latest Python release, so newly released Python versions are not supported or recommended immediately.

To install new packages, as needed, (for example, PyTorch and Jupyter Lab), run:

```bash
pip install torch jupyterlab
```

Alternatively, you can install all required Python package used in this book all once via the [`requirements.txt`](https://github.com/rasbt/reasoning-from-scratch/blob/main/requirements.txt) file:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```


&nbsp;
## Option 2: Use `uv` (faster and widely recommended)

While `pip` remains the classic and official way to install Python packages, [`uv`](https://github.com/astral-sh/uv) is a modern and widely recommended Python package manager that automatically:

- Creates and manages a virtual environment
- Installs packages quickly
- Keeps a lockfile for reproducible installs
- Supports `pip`-like commands

&nbsp;
### Installing `uv` and Python packages

To install `uv`,  you can use the commands below (also see the official [Installation](https://docs.astral.sh/uv/getting-started/installation/) page for the latest recommendations).

&nbsp;
**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

&nbsp;
**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once installed, you can install new Python packages similar to how you would do it via `pip` as described in the previous section, except that you replace `pip` with `uv pip`. For example

```bash
uv pip install torch jupyterlab
```

However, if you are using `uv`, which I recommend and use myself, it's even better to use the native `uv` syntax instead of `uv pip`, as described below.

&nbsp;
### Recommended `uv` workflow

Instead of using `uv pip`, I recommend and use the native `uv` worklow.

First, clone the GitHub repository to your local machine:



```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
```

Next, navigate into this folder, e.g., on Linux and MacOS:

```bash
cd reasoning-from-scratch
```

Then, since this folder contains a `pyproject.toml` file, you are already good to go: `uv` will automatically create a (by default invisible) virtual environment folder (`.venv`) for this `reasoning-from-scratch` project into which it installs all the dependencies the first time you run a script or open Jupyter Lab.

You will probably not need it but in general, you can install additional packages, which are not already part of the requirements listed in `pyproject.toml`, via `uv add`:


```bash
uv add llms_from_scratch
```

The above command will then add the package to the virtual environment and `pyproject.toml` file.

&nbsp;
### Running code via `uv`

This section describes the `uv` commands to run Jupyter Lab and Python scripts.

To open Jupyter Lab, execute:

```python
uv run jupyter lab
```

Python scripts can be run via:

```bash
uv run python script.py
```




> **Advanced usage:** This section describes a simple way to use `uv` that looks familiar to `pip` users. If you are interested in more advanced usage, please see [this document](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences) for more explicit instructions on managing virtual environments in `uv`. 
> If you are a macOS or Linux user and prefer the native uv commands, please refer to [this tutorial](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md). I also recommend checking the [official uv documentation](https://docs.astral.sh/uv/) for additional information.



&nbsp;
### JupyterLab tips

If you are viewing the notebook code in JupyterLab rather than VSCode, note that JupyterLab (in its default setting) has had scrolling bugs in recent versions. My recommendation is to go to Settings -> Settings Editor and change the "Windowing mode" to "none" (as illustrated below), which seems to address the issue.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

&nbsp;
## Questions?

If you have any questions, please don't hesitate to reach out via the [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) forum in this GitHub repository.
