
# MMLU Benchmarking

This bonus material implements three different methods for evaluating models on MMLU. 
- Method 1 is meant as an intuitive introduction
- Method 2 is the most widely used method in practice
- Method 3 is a more robust method that is better suited for reasoning models

- Please note that the code loads the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu) from the Hugging Face model hub. So, you need to install the `datasets` Python library before running the code:

```python
pip install datasets
```

or

```python
uv add datasets
```

- In the following sections, we apply the MMLU evaluation methods to  (`"high_school_mathematics"`)

- Note that there are many other interesting subsets; this one is chosen for simplicity and efficiency; you can use, for example

  - Use `--subsets list` to list other available subsets 

  - Use, for example, `--subsets "astronomy,high_school_mathematics"` to select multiple subsets

  - Use `--subsets "all"` to evaluate on all subsets

(Not that for simplicity and code readability, we focus on a zero-shot, as opposed to a 5-shot, setting.)

<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---

&nbsp;

## Method 1: MMLU letter matching

- We let the model generate the answer
- We extract the first generated A/B/C/D letter and compare it to the correct answer
- This is the most intuitive method, but the downside is that the model may not respond with a letter A/B/C/D

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_1.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model base     
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.200 [high_school_mathematics]
MMLU 150 acc=0.193 [high_school_mathematics]
MMLU 200 acc=0.235 [high_school_mathematics]
MMLU 250 acc=0.224 [high_school_mathematics]

MMLU letter accuracy: 58/270 = 21.48% in 69.1s
{'accuracy': 0.21481481481481482, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
qwen3-0.6B-reasoning.pth: 100% (1433 MiB / 1433 MiB)
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy: 57/270 = 21.11% in 43.6s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## Method 2: Log-probability scoring

- We run the prompt through the model and get log-probabilities (log-probs) for the next token (see chapter 4 for log-probs discussion)
- For each letter choice, we then compute which token ID would appear first if we appended that letter
- Then, we compare those four log-probs and pick the highest one (max)

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_2.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.420 [high_school_mathematics]
MMLU 150 acc=0.400 [high_school_mathematics]
MMLU 200 acc=0.370 [high_school_mathematics]
MMLU 250 acc=0.344 [high_school_mathematics]

MMLU letter accuracy (log-prob): 93/270 = 34.44% in 22.5s
{'accuracy': 0.34444444444444444, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy (log-prob): 57/270 = 21.11% in 22.4s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## Method 3: Teacher forcing

- Instead of looking up the log-prob of each of the letters A/B/C/D, a more robust scoring (specifically for reasoning models), is to feed the letter along with the complete answer string
- For our example, the answer strings are "A. 7", "B. 11", "C. 16", "D. 8"
- This method is known by the unfortunate term "teacher forcing"
- This method is the most reliable, but the caveat is that it takes 4x longer than the log-probability approach in method 2 (since we feed the model all 4 answer variants)

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_3.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.310 [high_school_mathematics]
MMLU 150 acc=0.307 [high_school_mathematics]
MMLU 200 acc=0.315 [high_school_mathematics]
MMLU 250 acc=0.312 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 86/270 = 31.85% in 67.9s
{'accuracy': 0.31851851851851853, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.250 [high_school_mathematics]
MMLU 150 acc=0.267 [high_school_mathematics]
MMLU 200 acc=0.255 [high_school_mathematics]
MMLU 250 acc=0.280 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 78/270 = 28.89% in 68.8s
{'accuracy': 0.28888888888888886, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



## Random guessing baseline

- This random guessing baseline is just to put the numbers above into perspective
  
- A model that guesses randomly with uniform (equal) probability across all answers is expected to achieve $25\%$ accuracy
  
- However, for a random guesser, we can expect deviations from the $25\%$ (depending on the sample size)

- For instance, we can model one evaluation run as a binomial with $K$ correct out of $n$ questions:

  - $K \sim \mathrm{Binomial}(n,p)$ with $p=\tfrac14$ and $n=$ number of questions.  
  - Accuracy $A = K/n$.

- Let's walk through this for the *high_school_mathematics* subset with $n=270$

- In general, the properties of the binomial are:

  - Mean: $\mathbb{E}[K] = np$
  - SD: $\sigma_K = \sqrt{np(1-p)}$

- For accuracy $A=K/n$:

  - Mean: $\mathbb{E}[A] = p = 0.25$
  - SD: $\sigma_A = \sqrt{\tfrac{p(1-p)}{n}}$

- Plugging in $n=270$:

  - $\mathbb{E}[A] = 25\%$  
  - $\sigma_A = \sqrt{\tfrac{0.25\cdot 0.75}{270}} \approx 2.64\%$

- Convert the one standard deviation ($\pm 1\sigma$) accuracy bounds to counts:

  - Lower: $K \le \lfloor 270\,(0.25-0.02636)\rfloor = 60$
  - Upper: $K \ge \lceil 270\,(0.25+0.02636)\rceil = 75$
  - (Inside the band is $K=61,\dots,74$; equivalently $A\in[22.36\%,\,27.64\%]$)

- So, the probability of falling outside this bound is:

  $$
  z = \pm\,\frac{75-67.5}{\sqrt{270\cdot 0.25\cdot 0.75}} \approx \pm 1.054, \qquad
  \Pr(|A-0.25|>0.02636) \approx 2\bigl(1-\Phi(1.054)\bigr) \approx 0.292.
  $$

  So about 29.2% of random-guess runs are below 22.36% or above 27.64%

- This means in about $29.2\%$ of cases where the model is random guessing (assuming uniformly), we get an accuracy below $22.36\%$ or above $27.64\%$
- Below is an empirical look:


```bash
➜  02_mmlu git:(main) ✗ uv run 0_random_guessing_baseline.py --subset "high_school_mathematics"
Subset: high_school_mathematics | split: test | n=270
Gold distribution provided in the dataset:
  A: 57 (21.11%)
  B: 71 (26.30%)
  C: 71 (26.30%)
  D: 71 (26.30%)

Random guessing over 10,000 trials (uniform A/B/C/D, seed=42):
  Mean accuracy: 24.98%
  Std dev across trials: 2.65%

Selected quantiles of accuracy:
  1% quantile: 18.889%
  5% quantile: 20.741%
  25% quantile: 23.333%
  50% quantile: 24.815%
  75% quantile: 26.667%
  95% quantile: 29.259%
  99% quantile: 31.111%

Full frequency table of accuracies (rounded):
  0.160: 1 times (0.01%)
  0.170: 11 times (0.11%)
  0.180: 38 times (0.38%)
  0.190: 124 times (1.24%)
  0.200: 302 times (3.02%)
  0.210: 562 times (5.62%)
  0.220: 612 times (6.12%)
  0.230: 1254 times (12.54%)
  0.240: 1619 times (16.19%)
  0.250: 1096 times (10.96%)
  0.260: 1525 times (15.25%)
  0.270: 1248 times (12.48%)
  0.280: 572 times (5.72%)
  0.290: 565 times (5.65%)
  0.300: 281 times (2.81%)
  0.310: 132 times (1.32%)
  0.320: 28 times (0.28%)
  0.330: 24 times (0.24%)
  0.340: 5 times (0.05%)
  0.360: 1 times (0.01%)
```
