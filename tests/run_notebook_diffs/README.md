# Notebook output comparison

These are convenience scripts that run notebooks with two different PyTorch versions and create a Markdown report of changed cell sources and outputs to investigate any discrepancies. This is basically for manual compatibility checks and are not run by pytest or GitHub CI.

&nbsp;
## Run notebooks with two PyTorch versions

From the repository root, run:

```bash
uv run python tests/run_notebook_diffs/run.py \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  ch02/01_main-chapter-code/ch02_main.ipynb
```

`uv` creates one isolated environment with the base dependencies installed for each version. 

One can use `--with` for additional notebook dependencies:

```bash
uv run python tests/run_notebook_diffs/run.py \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  --with transformers \
  --with datasets \
  ch08/01_main-chapter-code/ch08_main.ipynb
```

One can use `--python 3.11` if either PyTorch version does not provide a wheel for the default Python version.

Also note that multiple notebooks can be passed in one command. E.g.,

```bash
notebooks=(
  ch02/01_main-chapter-code/ch02_main.ipynb
  ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb

  ch03/01_main-chapter-code/ch03_main.ipynb
  ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb
  ch03/03_advanced-parser/compare_with_current_parser.ipynb

  ch04/01_main-chapter-code/ch04_main.ipynb
  ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb

  ch05/01_main-chapter-code/ch05_main.ipynb
  ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb

  ch06/01_main-chapter-code/ch06_main.ipynb
  ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb

  # ch07/01_main-chapter-code/ch07_main.ipynb
  # ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb

  ch08/01_main-chapter-code/ch08_main.ipynb
  ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb

  chC/01_main-chapter-code/chC_main.ipynb
  chD/chD_main.ipynb
  chE/chE_main.ipynb
  # chF/01_main-chapter-code/chF_main.ipynb
)

UV_PYTHON=3.13 uv run python tests/run_notebook_diffs/run.py \
  --allow-errors \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  "${notebooks[@]}"
```

Each notebook runs from its own directory, so relative paths behave as they do in Jupyter.

&nbsp;
## Results

By default, results are written below `tests/run_notebook_diffs/results/`:

```text
results/
└── ch02__01_main-chapter-code__ch02_main/
    ├── torch-2.7.1.ipynb
    ├── torch-2.13.0.ipynb
    └── comparison.md
```

The `comparison.md` is structured like this:

---

#### Cell 23

##### Outputs

```diff
--- left outputs
+++ right outputs
@@ -2,6 +2,6 @@
   {
     "name": "stdout",
     "output_type": "stream",
-    "text": "PyTorch version 2.7.1\nApple Silicon GPU\n"
+    "text": "PyTorch version 2.13.0\nApple Silicon GPU\n"
   }
 ]
```

#### Cell 87

##### Outputs

```diff
--- left outputs
+++ right outputs
@@ -207,6 +207,6 @@
   {
     "name": "stdout",
     "output_type": "stream",
-    "text": "\n\nTime: 8.20 sec\n5 tokens/sec\n"
+    "text": "\n\nTime: 8.16 sec\n5 tokens/sec\n"
   }
 ]
```

---



The `results/` directory is ignored by Git. Optionally you can use `--output-dir PATH` to write to somewhere else.

Execution stops on a notebook error by default. For some notebooks where errors are there on purpose for educational reasons, use `--allow-errors` to continue through the remaining cells. 

There is a default timeout of 1 hour per cell but you can override that with `--timeout SECONDS`.

&nbsp;
## Compare existing notebooks

If you already have to executed notebooks handy, the comparator can also be used independently:

```bash
uv run python tests/run_notebook_diffs/compare.py \
  first.ipynb second.ipynb --output comparison.md
```

