
# Leaderboard Rankings

This bonus material implements two different ways to construct LM Arena (formerly Chatbot Arena) style leaderboards from pairwise comparisons.

Both implementations take in a list of pairwise preferences (left: winner, right: loser) from a json file via the `--path` argument. Here's an excerpt of the provided [votes.json](votes.json) file:

```json
[
  ["GPT-5", "Claude-3"],
  ["GPT-5", "Llama-4"],
  ["Claude-3", "Llama-3"],
  ["Llama-4", "Llama-3"],
  ...
]
```



<br>

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---

&nbsp;
## Method 1: Elo ratings

- Implements the popular Elo rating method (inspired by chess rankings) that was originally used by LM Arena
- See the [main notebook](../01_main-chapter-code/chF_main.ipynb) for details

```bash
➜  03_leaderboards git:(main) ✗ uv run 1_elo_leaderboard.py --path votes.json

Leaderboard (Elo) 
-----------------------
 1. GPT-5       1095.9
 2. Claude-3    1058.7
 3. Llama-4      958.2
 4. Llama-3      887.2
```






&nbsp;
## Method 2: Bradley-Terry model

- Implements a [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley–Terry_model), similar to the new LM Arena leaderboard as described in the official paper ([Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132))
- Like on the LM Arena leaderboard, the scores are re-scaled to be similar to the original Elo scores
- The code here uses the Adam optimizer from PyTorch to fit the model (for better code familiarity and readability)



```bash
➜  03_leaderboards git:(main) ✗ uv run 2_bradley_terry_leaderboard.py --path votes.json 

Leaderboard (Bradley-Terry)
-----------------------------
 1. GPT-5       1140.6
 2. Claude-3    1058.7
 3. Llama-4      950.3
 4. Llama-3      850.4
```

