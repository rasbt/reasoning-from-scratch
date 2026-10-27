# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERSION_CASES = [
    ("2.7.1", (2, 7)),
    ("2.8.0", (2, 8)),
    ("2.9.1", (2, 9)),
    ("2.10.0", (2, 10)),
    ("2.10.0+cpu", (2, 10)),
    ("2.10.0+cu128", (2, 10)),
    ("2.10.0rc1", (2, 10)),
    ("2.11.0.dev20261027", (2, 11)),
]


def read_code_sources(relative_path):
    path = ROOT / relative_path
    if path.suffix == ".py":
        return [path.read_text(encoding="utf-8")]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.fixture(scope="module", params=[
    pytest.param(("reasoning_from_scratch/ch02.py", 1), id="package-get-device"),
    pytest.param(
        ("ch02/01_main-chapter-code/ch02_main.ipynb", 2),
        id="main-notebook-get-device-and-compile",
    ),
    pytest.param(
        ("ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb", 1),
        id="exercise-notebook",
    ),
    pytest.param(
        ("ch02/03_optimized-LLM/compare_inference.py", 1),
        id="optimized-inference",
    ),
])
def version_assignments(request):
    relative_path, expected_count = request.param
    assignments = []
    for source in read_code_sources(relative_path):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Tuple):
                continue
            names = [getattr(element, "id", None) for element in target.elts]
            if names == ["major", "minor"]:
                # Execute the actual assignment without running model setup.
                module = ast.Module(body=[node], type_ignores=[])
                assignments.append(compile(module, str(ROOT / relative_path), "exec"))
    # Fail if a source stops being covered instead of silently testing nothing.
    assert len(assignments) == expected_count
    return assignments


@pytest.mark.parametrize("version, expected_version", VERSION_CASES)
def test_torch_version_parsing(version_assignments, version, expected_version):
    for assignment in version_assignments:
        namespace = {"torch": SimpleNamespace(__version__=version)}
        exec(assignment, namespace)
        assert (namespace["major"], namespace["minor"]) == expected_version


@pytest.fixture(scope="module")
def compile_cell_source():
    sources = read_code_sources(
        "ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb"
    )
    compile_cells = [
        source for source in sources
        if "model_compiled = torch.compile(model)" in source
    ]
    assert len(compile_cells) == 1
    return compile_cells[0]


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("version, expected_version", VERSION_CASES)
def test_compile_cell_handles_torch_versions(
    compile_cell_source, device_type, version, expected_version
):
    model = object()
    compiled_model = object()
    config = SimpleNamespace(allow_unspec_int_on_nn_module=False)
    fake_torch = SimpleNamespace(
        __version__=version,
        compile=Mock(return_value=compiled_model),
        _dynamo=SimpleNamespace(config=config),
    )
    namespace = {
        "torch": fake_torch,
        "device": SimpleNamespace(type=device_type),
        "model": model,
    }

    # Execute the notebook's actual cell so a regression there fails this test.
    exec(compile_cell_source, namespace)

    assert (namespace["major"], namespace["minor"]) == expected_version
    if device_type == "mps" and expected_version < (2, 10):
        fake_torch.compile.assert_not_called()
        assert namespace["model_compiled"] is model
        assert config.allow_unspec_int_on_nn_module is False
    else:
        fake_torch.compile.assert_called_once_with(model)
        assert namespace["model_compiled"] is compiled_model
        assert config.allow_unspec_int_on_nn_module is (expected_version >= (2, 8))
