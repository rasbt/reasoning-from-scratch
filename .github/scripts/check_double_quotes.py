# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Verify that Python source files and notebooks use double quotes for strings.

import io
import json
import sys
import tokenize
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

PREFIX_CHARS = {"r", "u", "f", "b"}
SINGLE_QUOTE = "'"
DOUBLE_QUOTE = "\""
TRIPLE_SINGLE = SINGLE_QUOTE * 3
TRIPLE_DOUBLE = DOUBLE_QUOTE * 3


def should_skip(path):
    parts = set(path.parts)
    return bool(EXCLUDED_DIRS & parts)


def check_quotes_in_source(source, path):
    violations = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok_type, tok_str, start, _, _ in tokens:
        if tok_type == tokenize.STRING:
            lowered = tok_str.lower()
            # ignore triple-quoted strings
            if lowered.startswith((TRIPLE_DOUBLE, TRIPLE_SINGLE)):
                continue

            # find the prefix and quote type
            # prefix = ""
            for c in PREFIX_CHARS:
                if lowered.startswith(c):
                    # prefix = c
                    lowered = lowered[1:]
                    break

            # report if not using double quotes
            if lowered.startswith(SINGLE_QUOTE):
                line, col = start
                violations.append(f"{path}:{line}:{col}: uses single quotes")
    return violations


def check_file(path):
    try:
        if path.suffix == ".ipynb":
            return check_notebook(path)
        else:
            text = path.read_text(encoding="utf-8")
            return check_quotes_in_source(text, path)
    except Exception as e:
        return [f"{path}: failed to check ({e})"]


def check_notebook(path):
    violations = []
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            violations.extend(check_quotes_in_source(src, path))
    return violations


def main():
    project_root = Path(".").resolve()
    py_files = sorted(project_root.rglob("*.py"))
    notebook_files = sorted(project_root.rglob("*.ipynb"))

    violations = []
    for path in py_files + notebook_files:
        if should_skip(path):
            continue
        violations.extend(check_file(path))

    if violations:
        print("\n".join(violations))
        print(f"\n{len(violations)} violations found.")
        return 1

    print("All files use double quotes correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
