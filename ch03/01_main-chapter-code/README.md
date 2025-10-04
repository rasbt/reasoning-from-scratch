# Chapter 3: Evaluating Reasoning Models

&nbsp;
## Main chapter code

- [ch03_main.ipynb](ch03_main.ipynb): main chapter code
- [ch03_exercise-solutions.ipynb](ch03_exercise-solutions.ipynb): exercise solutions


&nbsp;
## Bonus materials

- [../02_math500-verifier-scripts/evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py): standalone script to evaluate models on the MATH-500 dataset
- [../02_math500-verifier-scripts/evaluate_math500_batched.py](../02_math500-verifier-scripts/evaluate_math500_batched.py): same as above, but processes multiple examples in parallel during generation (for higher throughput)

Both evaluation scripts import functionality from the [`reasoning_from_scratch`](../../reasoning_from_scratch) package to avoid code duplication. (See [chapter 2 setup instructions](../../ch02/02_setup-tips/python-instructions.md) for installation details.)
