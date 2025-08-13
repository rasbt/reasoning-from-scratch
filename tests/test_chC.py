# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path
import sys

import pytest
from importnb import Notebook

from reasoning_from_scratch.ch02 import (
    generate_text_basic,
    generate_text_basic_cache,
)

from test_qwen3 import test_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with Notebook():
    from chC.chC_main import Qwen3Model


@pytest.mark.parametrize("ModelClass", [Qwen3Model])
@pytest.mark.parametrize("generate_fn", [generate_text_basic, generate_text_basic_cache])
def test_model_here_too(ModelClass, qwen3_weights_path, generate_fn):
    test_model(ModelClass, qwen3_weights_path, generate_fn)
