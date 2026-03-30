# Tests

This directory contains the repository's Python test suite.

## Local runs

Install the dev environment first:

```bash
uv sync --group dev
```

### 1. Normal suite, ignoring expensive tests  (recommended)

This is recommended for quick testing and development of new features.

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

Run a single test file:

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests/test_ch03.py
```


This is the closest local equivalent to the default GitHub test matrix.

### 2. Normal suite plus expensive tests

There are some codes that are ignored by default, because they are relatively expensive to run. I recommend running these tests if you are finished with the basic debugging.

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

Note that this runs tests guarded by `SKIP_EXPENSIVE` in the test files, but it still excludes the real network/download integration tests that download large model checkpoints.

### 3. Download tests only

There are some tests that check whether the model checkpoint files are available for download and the servers (still) work. It's not necessary to run these tests locally or a regular basis. This is more meant for occasional testing.

To run these download tests, use:

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests -k real_download
```

How this works:

- `pytest tests` collects tests from the `tests/` directory
- `-k real_download` keeps only tests whose names contain `real_download`

For a more targeted example, for example, to run the appendix D real snapshot test directly, use:

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests/test_appendix_d.py -k real_download_1_7b
```

The opt-in real-download tests currently cover:

- `tests/test_ch03.py`: real `math500_test.json` download and tokenizer downloads
- `tests/test_ch06.py`: real math training set download
- `tests/test_ch07.py`: real GitHub raw file download
- `tests/test_ch08.py`: real distillation dataset and tokenizer downloads
- `tests/test_appendix_d.py`: real `Qwen/Qwen3-1.7B-Base` snapshot download


### 4. Everything (not recommended)

This runs everything in the test suite. Note that this includes the computationally expensive tests (section 3) as well as the expenive download tests (section 4).

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests
```

This is not recommended for routine tests when making code changes, because the file downloads are very expensive and unnecessary to run on a regular basis.


## What runs in GitHub CI

The default GitHub test matrix runs the normal suite and omits heavier tests:

- `.github/workflows/tests-linux.yml`
- `.github/workflows/tests-macos.yml`
- `.github/workflows/tests-windows.yml`
- `.github/workflows/basic-tests-pip.yml`

These workflows set `SKIP_EXPENSIVE=1`, so expensive tests are skipped there. The reason is that the GitHub CI does not have the necessary computational resources (like a GPU) to run the expensive tests.

The real network/download integration tests run in a separate workflow:

- `.github/workflows/real-download-tests.yml`

That workflow sets `RUN_REAL_DOWNLOAD_TESTS=1` and runs only tests selected by `-k real_download`.
It is not part of the default PR/push matrix. It runs on a weekly schedule and can also be started manually via `workflow_dispatch`.
