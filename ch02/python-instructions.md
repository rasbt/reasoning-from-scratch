# Python Setup Recommendations

The code in this book is largely self-contained, and I have made an effort to minimize external dependencies. However, to keep the book accessible, readable, and well under 2000 pages, a few Python packages are necessary.

This section introduces two beginner-friendly methods for installing the required packages so you can run the code examples. 

There are, of course, many other ways to install and manage Python packages. If you are an experienced Python user and already have your own setup or preferences, feel free to skip this section.

If neither of the two options below works for you, please do not hesitate to reach out, for example, by opening a [Discussion](https://gi
thub.com/rasbt/reasoning-from-scratch/discussions).

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

Once installed, you can install Python packages like this:

```bash
uv pip install torch jupyterlab
```

Alternatively, you can install all required Python package used in this book all once via the [`requirements.txt`](https://github.com/rasbt/reasoning-from-scratch/blob/main/requirements.txt) file:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```


&nbsp;
#### Using virtual environments in `uv`

`uv` automatically creates and uses a virtual environment in a central location, typically at `~/.venv` on macOS and Linux systems or `%LOCALAPPDATA%\uv\venv` on Windows. However, you can tell `uv` to create a virtual environment in your project folder by using the following command:

```bash
uv venv --project reasoning --python python3.12
```

Creating a separate virtual environment for this project  is useful when working with many different projects that require different Python packages. 

After creating your environment, you have activate it via the instructions shown after executing the previous command. This is typically `source .venv/bin/activate` on macOS and Linux systems and `.venv\Scripts\Activate.ps1` in Windows PowerShell.

&nbsp;
#### Running code using `uv`

To run code, simply use:

```bash
uv python script.py
```

Or start a JupyterLab session with:

```bash
uv run jupyterlab
```

&nbsp;

> **Advanced usage:** This section describes a simple way to use `uv` that looks familiar to `pip` users. If you are interested in more advanced usage, please see [this document](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences) for more explicit instructions on managing virtual environments in `uv`. 
> If you are a macOS or Linux user and prefer the native uv commands, please refer to [this tutorial](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md). I also recommend checking the [official uv documentation](https://docs.astral.sh/uv/) for additional information.



&nbsp;
## Questions?

If you have any questions, please don't hesitate to reach out via the [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) forum in this GitHub repository.
