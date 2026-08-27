#!/usr/bin/env python3
"""Execute notebooks with two PyTorch versions and compare their outputs."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
PROBE_TAG = "notebook-diff-torch-version-probe"
PROBE_PREFIX = "NOTEBOOK_DIFF_TORCH_VERSION="


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_")


def _notebook_name(path):
    try:
        relative = path.relative_to(REPO_ROOT)
        parts = relative.with_suffix("").parts
    except ValueError:
        parts = (path.stem,)
    return "__".join(_safe_name(part) for part in parts)


def _output_path(notebook, output_dir, torch_version):
    version_name = _safe_name(torch_version)
    return output_dir / _notebook_name(notebook) / f"torch-{version_name}.ipynb"


def _resolve_notebooks(paths):
    notebooks = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Notebook does not exist: {value}")
        if path.suffix != ".ipynb":
            raise ValueError(f"Expected an .ipynb file: {value}")
        notebooks.append(path)

    names = [_notebook_name(path) for path in notebooks]
    if len(names) != len(set(names)):
        raise ValueError("Notebook paths produce duplicate output names")
    return notebooks


def _base_version(version):
    return version.split("+", 1)[0]


def _probe_source(torch_version):
    return (
        "import torch as _notebook_diff_torch\n"
        "_notebook_diff_version = "
        "_notebook_diff_torch.__version__.split('+', 1)[0]\n"
        f"assert _notebook_diff_version == {torch_version!r}, (\n"
        f"    'Expected PyTorch {torch_version}, got ' "
        "+ _notebook_diff_torch.__version__\n"
        ")\n"
        f"print({PROBE_PREFIX!r} + _notebook_diff_version)\n"
    )


def _read_probe_version(cell):
    for output in cell.get("outputs", []):
        if output.get("output_type") != "stream":
            continue
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        for line in text.splitlines():
            if line.startswith(PROBE_PREFIX):
                return line.removeprefix(PROBE_PREFIX)
    return None


def _execute_notebook(
    notebook,
    output_path,
    torch_version,
    timeout,
    allow_errors,
    notebook_index,
    notebook_count,
):
    import nbformat
    from nbclient import NotebookClient

    document = nbformat.read(notebook, as_version=4)
    code_cell_count = sum(cell.cell_type == "code" for cell in document.cells)
    current_code_cell = 0

    def report_progress(cell, cell_index):
        nonlocal current_code_cell
        if PROBE_TAG in cell.metadata.get("tags", []):
            return
        current_code_cell += 1
        print(
            f"Executing Cell {current_code_cell}/{code_cell_count} in "
            f"Nb {notebook_index}/{notebook_count}: {notebook.name} "
            f"(PyTorch {torch_version})",
            flush=True,
        )

    probe = nbformat.v4.new_code_cell(
        source=_probe_source(torch_version),
        metadata={"tags": [PROBE_TAG]},
    )
    document.cells.insert(0, probe)

    execution_error = None
    kernel_version = None
    try:
        client = NotebookClient(
            document,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=allow_errors,
            resources={"metadata": {"path": str(notebook.parent)}},
            on_cell_execute=report_progress,
        )
        client.execute()
        kernel_version = _read_probe_version(document.cells[0])
        if kernel_version != torch_version:
            raise RuntimeError(
                "Notebook kernel did not confirm the requested PyTorch version: "
                f"expected {torch_version}, got {kernel_version or 'no probe output'}"
            )
    except Exception as error:
        execution_error = error
        kernel_version = _read_probe_version(document.cells[0])
    finally:
        if document.cells and PROBE_TAG in document.cells[0].metadata.get("tags", []):
            document.cells.pop(0)
        document.metadata["notebook_diff"] = {
            "requested_torch_version": torch_version,
            "kernel_torch_version": kernel_version,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(document, output_path)

    if execution_error is not None:
        raise execution_error


def _worker_main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("notebooks", nargs="+")
    args = parser.parse_args(argv)

    import torch

    installed_version = _base_version(torch.__version__)
    if installed_version != args.torch_version:
        parser.error(
            f"requested PyTorch {args.torch_version}, but uv installed {torch.__version__}"
        )

    failures = []
    notebooks = [Path(value).resolve() for value in args.notebooks]
    for notebook_index, notebook in enumerate(notebooks, start=1):
        output_path = _output_path(notebook, args.output_dir.resolve(), args.torch_version)
        print(f"Running {notebook} with PyTorch {args.torch_version}", flush=True)
        try:
            _execute_notebook(
                notebook,
                output_path,
                args.torch_version,
                args.timeout,
                args.allow_errors,
                notebook_index,
                len(notebooks),
            )
        except Exception as error:
            failures.append((notebook, error))
            print(f"Execution failed for {notebook}: {error}", file=sys.stderr)
        else:
            print(f"Wrote {output_path}", flush=True)

    if failures:
        return 1
    return 0


def _public_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run notebooks in two isolated PyTorch environments and compare outputs."
        )
    )
    parser.add_argument(
        "--torch-version",
        action="append",
        required=True,
        help="Exact PyTorch version. Pass this option twice.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--python",
        help="Python version or executable passed to uv, for example 3.11",
    )
    parser.add_argument(
        "--with",
        action="append",
        default=[],
        dest="extra_packages",
        metavar="PACKAGE",
        help="Additional package for the notebook environment. May be repeated.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Continue after errors in regular notebook cells.",
    )
    parser.add_argument("notebooks", nargs="+", help="Notebook paths to execute")
    return parser


def _run_main(argv):
    parser = _public_parser()
    args = parser.parse_args(argv)

    if len(args.torch_version) != 2:
        parser.error("pass --torch-version exactly twice")
    if len(set(args.torch_version)) != 2:
        parser.error("the two PyTorch versions must be different")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if shutil.which("uv") is None:
        parser.error("uv is required but was not found on PATH")

    try:
        notebooks = _resolve_notebooks(args.notebooks)
    except ValueError as error:
        parser.error(str(error))

    output_dir = args.output_dir.expanduser().resolve()
    for torch_version in args.torch_version:
        command = [
            "uv",
            "run",
            "--isolated",
            "--no-default-groups",
            "--with",
            f"torch=={torch_version}",
        ]
        for package in args.extra_packages:
            command.extend(["--with", package])
        if args.python:
            command.extend(["--python", args.python])
        command.extend(
            [
                "python",
                str(Path(__file__).resolve()),
                "_execute",
                "--torch-version",
                torch_version,
                "--output-dir",
                str(output_dir),
                "--timeout",
                str(args.timeout),
            ]
        )
        if args.allow_errors:
            command.append("--allow-errors")
        command.extend(str(notebook) for notebook in notebooks)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    from compare import write_report

    for notebook in notebooks:
        left = _output_path(notebook, output_dir, args.torch_version[0])
        right = _output_path(notebook, output_dir, args.torch_version[1])
        report = left.parent / "comparison.md"
        summary = write_report(left, right, report)
        print(
            f"Compared {notebook}: {summary['changed_cells']} changed cell(s)"
        )
        print(f"Wrote {report}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_execute":
        return _worker_main(argv[1:])
    return _run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
