# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys
import pytest

from reasoning_from_scratch.ch02 import (
    generate_text_basic,
    generate_text_basic_cache,
)
# Local imports
from test_qwen3 import test_model
from conftest import import_definitions_from_notebook


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


nb_path = ROOT / "chC" / "01_main-chapter-code" / "chC_main.ipynb"
mod = import_definitions_from_notebook(nb_path, "chC_chC_main_defs")
Qwen3Model = getattr(mod, "Qwen3Model")


@pytest.mark.parametrize("ModelClass", [Qwen3Model])
@pytest.mark.parametrize("generate_fn", [generate_text_basic, generate_text_basic_cache])
def test_model_here_too(ModelClass, qwen3_weights_path, generate_fn):
    test_model(ModelClass, qwen3_weights_path, generate_fn)
