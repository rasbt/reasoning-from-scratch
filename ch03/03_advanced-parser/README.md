# Chapter 3: Advanced Parser (Bonus Material)

This folder contains the parser experiment from [issue #133](https://github.com/rasbt/reasoning-from-scratch/issues/133), where a hybrid LaTeX parser was proposed to handle edge cases that the current chapter parser may miss.



&nbsp;

## Files

- [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb): notebook with usage examples
- [math500_llm_answer.json](math500_llm_answer.json): MATH-500 examples with LLM answers, used to for a section in the notebook above
- [evaluate_math500_advanced.py](evaluate_math500_advanced.py): Same as the chapter 3 LLM evaluation script [evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py) but supports `--hybrid_parser` as an additional argument to use the alternative hybrid parser





&nbsp;
## How This Differs From The Chapter 3 Parser
evaluate_math500_advanced.py
The chapter parser in [reasoning_from_scratch/ch03.py](../../reasoning_from_scratch/ch03.py) is designed to stay compact and teachable:

- It focuses on lightweight normalization plus symbolic equivalence checks
- It mainly treats answers as arithmetic/symbolic expressions

The hybrid parser in this folder (`latex_normalizer_hybrid.py`) is pattern-first and broader:

- It recognizes answer formats before fallback parsing.
- It adds support for intervals, unions, equations, matrices, set notation, membership (`\\in`), and `\\pm`
- It preserves important edge cases better, such as base-subscript answers (`52_8`) and text casing (`\\text{Evelyn}`)

Examples where behavior differs:

- `52_8` -> chapter path often resolves to `528`; hybrid keeps `52_8`
- `11,\\! 111,\\! 111,\\! 100` -> chapter path can become a tuple; hybrid normalizes to `11111111100`
- `(0,9) \\cup (9,36)` -> chapter path usually remains text; hybrid returns a symbolic union

Tradeoffs:

- Chapter parser: simpler, faster, and easier to interpret
- Hybrid parser: better coverage on LaTeX edge cases, but more rules and complexity; also adds SymPy LaTeX backend dependencies

&nbsp;
## Usage

You can import the hybrid parser directly from the package:

```python
from reasoning_from_scratch.bonus.parser import normalize_text_hybrid, sympy_parser_hybrid
```

See [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb) for more detailed usage examples.
