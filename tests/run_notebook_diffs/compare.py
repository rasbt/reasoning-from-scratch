#!/usr/bin/env python3
"""Compare the sources and outputs of two executed notebooks."""

import argparse
import base64
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

import nbformat


ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
BINARY_MIME_TYPES = {
    "application/pdf",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_DIFF_LINES = 200
MAX_SOURCE_PREVIEW = 100


def _clean_text(value):
    if isinstance(value, list):
        value = "".join(value)
    return ANSI_ESCAPE.sub("", value.replace("\r\n", "\n").replace("\r", "\n"))


def _binary_summary(value):
    encoded = "".join(value) if isinstance(value, list) else value
    try:
        content = base64.b64decode(encoded)
    except (ValueError, TypeError):
        content = str(encoded).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    return {"bytes": len(content), "sha256": digest}


def _normalize_value(value):
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return _clean_text(value)
    return value


def _normalize_data(data):
    normalized = {}
    for mime_type in sorted(data):
        value = data[mime_type]
        if mime_type in BINARY_MIME_TYPES:
            normalized[mime_type] = _binary_summary(value)
        else:
            normalized[mime_type] = _normalize_value(value)
    return normalized


def _normalize_output(output):
    output_type = output.get("output_type")
    normalized = {"output_type": output_type}
    if output_type == "stream":
        normalized["name"] = output.get("name")
        normalized["text"] = _clean_text(output.get("text", ""))
    elif output_type == "error":
        normalized["ename"] = output.get("ename")
        normalized["evalue"] = _clean_text(output.get("evalue", ""))
        normalized["traceback"] = [
            _clean_text(line) for line in output.get("traceback", [])
        ]
    elif output_type in {"display_data", "execute_result"}:
        normalized["data"] = _normalize_data(output.get("data", {}))
    else:
        normalized.update(_normalize_value(dict(output)))
    return normalized


def _normalize_outputs(outputs):
    normalized = []
    for output in outputs:
        current = _normalize_output(output)
        if (
            current["output_type"] == "stream"
            and normalized
            and normalized[-1]["output_type"] == "stream"
            and normalized[-1]["name"] == current["name"]
        ):
            normalized[-1]["text"] += current["text"]
        else:
            normalized.append(current)
    return normalized


def _cell_view(cell):
    view = {
        "cell_type": cell.get("cell_type"),
        "source": _clean_text(cell.get("source", "")),
    }
    if cell.get("cell_type") == "code":
        view["outputs"] = _normalize_outputs(cell.get("outputs", []))
    return view


def _cell_heading(left_document, right_document, index):
    if index < len(left_document.cells):
        document = left_document
    else:
        document = right_document

    cell = document.cells[index]
    cell_type = cell.get("cell_type", "unknown")
    type_number = sum(
        item.get("cell_type") == cell_type for item in document.cells[: index + 1]
    )
    type_label = "Markdown" if cell_type == "markdown" else cell_type
    details = f"{type_label} cell {type_number}"
    if cell_type == "code" and cell.get("execution_count") is not None:
        details += f", execution [{cell['execution_count']}]"

    source = _clean_text(cell.get("source", ""))
    first_line = next((line.strip() for line in source.splitlines() if line.strip()), "")
    if len(first_line) > MAX_SOURCE_PREVIEW:
        first_line = first_line[: MAX_SOURCE_PREVIEW - 3] + "..."

    heading = f"## Notebook cell {index + 1}/{len(document.cells)} ({details})"
    preview = f"Source: `{first_line}`" if first_line else "Source: *(empty)*"
    return heading, preview


def _diff_lines(left, right, left_label, right_label):
    lines = list(
        difflib.unified_diff(
            left.splitlines(),
            right.splitlines(),
            fromfile=left_label,
            tofile=right_label,
            lineterm="",
        )
    )
    if len(lines) > MAX_DIFF_LINES:
        omitted = len(lines) - MAX_DIFF_LINES
        lines = lines[:MAX_DIFF_LINES]
        lines.append(f"... {omitted} additional diff lines omitted")
    return "\n".join(lines)


def _json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def compare_notebooks(left_path, right_path):
    left_path = Path(left_path).resolve()
    right_path = Path(right_path).resolve()
    left_document = nbformat.read(left_path, as_version=4)
    right_document = nbformat.read(right_path, as_version=4)
    left_cells = [_cell_view(cell) for cell in left_document.cells]
    right_cells = [_cell_view(cell) for cell in right_document.cells]

    changed = []
    source_changes = 0
    output_changes = 0
    type_changes = 0
    for index in range(max(len(left_cells), len(right_cells))):
        left = left_cells[index] if index < len(left_cells) else None
        right = right_cells[index] if index < len(right_cells) else None
        if left == right:
            continue
        if left is None or right is None:
            source_changes += 1
            type_changes += 1
        else:
            source_changes += left["source"] != right["source"]
            type_changes += left["cell_type"] != right["cell_type"]
            output_changes += left.get("outputs") != right.get("outputs")
        changed.append((index, left, right))

    summary = {
        "left_cells": len(left_cells),
        "right_cells": len(right_cells),
        "changed_cells": len(changed),
        "source_changes": source_changes,
        "output_changes": output_changes,
        "cell_type_changes": type_changes,
    }

    lines = [
        "# Notebook comparison",
        "",
        f"- Left: `{left_path}`",
        f"- Right: `{right_path}`",
        "",
        "## Summary",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
        f"| Left cells | {summary['left_cells']} |",
        f"| Right cells | {summary['right_cells']} |",
        f"| Changed cells | {summary['changed_cells']} |",
        f"| Source changes | {summary['source_changes']} |",
        f"| Output changes | {summary['output_changes']} |",
        f"| Cell type changes | {summary['cell_type_changes']} |",
        "",
    ]

    if not changed:
        lines.extend(["No source or output differences found.", ""])
        return "\n".join(lines), summary

    for index, left, right in changed:
        heading, preview = _cell_heading(left_document, right_document, index)
        lines.extend([heading, "", preview, ""])
        if left is None:
            lines.extend(["Cell exists only in the right notebook.", ""])
        elif right is None:
            lines.extend(["Cell exists only in the left notebook.", ""])
        else:
            if left["cell_type"] != right["cell_type"]:
                lines.extend(
                    [
                        f"Cell type changed from `{left['cell_type']}` to "
                        f"`{right['cell_type']}`.",
                        "",
                    ]
                )
            if left["source"] != right["source"]:
                source_diff = _diff_lines(
                    left["source"], right["source"], "left source", "right source"
                )
                lines.extend(["### Source", "", "```diff", source_diff, "```", ""])
            if left.get("outputs") != right.get("outputs"):
                output_diff = _diff_lines(
                    _json_text(left.get("outputs", [])),
                    _json_text(right.get("outputs", [])),
                    "left outputs",
                    "right outputs",
                )
                lines.extend(
                    ["### Outputs", "", "```diff", output_diff, "```", ""]
                )

        if left is None or right is None:
            existing = right if left is None else left
            lines.extend(
                [
                    "### Cell contents",
                    "",
                    "```json",
                    _json_text(existing),
                    "```",
                    "",
                ]
            )

    return "\n".join(lines), summary


def write_report(left_path, right_path, output_path):
    report, summary = compare_notebooks(left_path, right_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare cell sources and normalized outputs in two notebooks."
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, help="Write the Markdown report here")
    args = parser.parse_args(argv)

    report, summary = compare_notebooks(args.left, args.right)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(report)
    print(f"Changed cells: {summary['changed_cells']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
