# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Check if code in notebooks exceeds line length
# Usage: uv run .github/scripts/check_notebook_line_length.py --max-len 76 ch02/01_main-chapter-code/ch02_main.ipynb  ch03/01_main-chapter-code/ch03_main.ipynb

import argparse
import sys
from pathlib import Path
import nbformat as nbf


def parse_args():
    p = argparse.ArgumentParser(
        description="Check code-cell line lengths in specific .ipynb files."
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=76,
        help="Maximum allowed characters per line (default: 76)",
    )
    p.add_argument(
        "notebooks",
        nargs="+",
        help="Paths to .ipynb files to check.",
    )
    return p.parse_args()


def strip_inline_comment(line):
    """Return line with any trailing #... comment removed, but keep # inside quotes."""
    in_quote = None
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in ("'", '"'):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            continue
        if ch == "#" and in_quote is None:
            return line[:i].rstrip()
    return line.rstrip()


def main():
    args = parse_args()

    nb_paths = []
    for raw in args.notebooks:
        p = Path(raw).resolve()
        if not p.exists():
            print(f"::warning file={raw}::File not found, skipping.")
            continue
        if p.suffix != ".ipynb":
            print(f"::warning file={raw}::Not a .ipynb file, skipping.")
            continue
        nb_paths.append(p)

    if not nb_paths:
        print("No valid notebooks to check.")
        return 0

    violations = []

    for nb_path in nb_paths:
        try:
            nb = nbf.read(nb_path, as_version=4)
        except Exception as e:
            print(f"::warning file={nb_path}::Failed to read notebook: {e}")
            continue

        for ci, cell in enumerate(nb.cells, start=1):
            if cell.get("cell_type") != "code":
                continue

            src = cell.get("source", "")
            lines = src if isinstance(src, list) else src.splitlines()

            for li, line in enumerate(lines, start=1):
                code_part = strip_inline_comment(line)
                length = len(code_part)
                if length > args.max_len:
                    print(
                        f"::error file={nb_path}::Line length {length} exceeds "
                        f"{args.max_len} in code cell {ci}, line {li}"
                    )
                    snippet = code_part if len(code_part) <= 120 else code_part[:120] + "…"
                    violations.append((str(nb_path), ci, li, length, snippet))

    if violations:
        print("\nFound lines exceeding the limit:\n")
        for path, ci, li, length, snippet in violations:
            print(f"- {path} | cell {ci} line {li}: {length} chars\n    {snippet}")
        return 1

    print(f"All notebooks pass: no code-cell lines exceed {args.max_len} characters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
