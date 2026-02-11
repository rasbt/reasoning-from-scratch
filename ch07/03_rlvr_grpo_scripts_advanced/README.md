# Chapter 7: Improving Policy Optimization in Reinforcement Learning

&nbsp;

In progress ...


&nbsp;
## Scripts Overview

&nbsp;
### Main Scripts

- `ch06_ref.py` (*7.2 Tracking GRPO performance metrics*): The code from chapter 6 included for reference; this is the same code that is used in section 
- `7_3_plus_tracking.py` (*7.3 Tracking more advanced GRPO performance metrics*): Tracks additional performance metrics (advantage statistics and entropy)
- `7_4_plus_clip_ratio.py` (*7.4 Stabilizing sequence-level GRPO using clipped policy ratios*): Like above but computes policy gradient loss with clipped policy ratios
- `7_5_plus_kl.py` (*7.5 Controlling how much the model changes with a KL term*): Like above but adds a KL loss term
- `7_6_plus_format_reward.py` (*7.6 Adding an explicit format reward*): Like above but adds an additional format reward for `<think>` tokens (a key difference to the other scripts is that this is applied to the reasoning instead of base model since it is already familiar with these tokens as discussed in the main chapter)

<br>

&nbsp;
### GRPO Tips & Tricks Bonus Scripts

Since GRPO was first published in April 2024 ([DeepSeekMath](https://arxiv.org/abs/2402.03300)) and became popular in January 2025 ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)), many improvements have been suggested in the literature. Some for the most notable ones are listed below:

1. Zero gradient signal filtering ([DAPO by Yu et al., 2025](https://arxiv.org/abs/2503.14476))
2. Active sampling (DAPO)
3. Token-level loss (DAPO)
4. No KL loss (DAPO and [Dr. GRPO by Liu et al., 2025](https://arxiv.org/abs/2503.20783))
5. Clip higher (DAPO)
6. Truncated importance sampling ([Yao et al., 2025](https://fengyao.notion.site/off-policy-rl))
7. No standard deviation normalization (Dr. GRPO)
8. KL tuning with domain-specific KL strengths; zero for math ([DeepSeek V3.2](https://arxiv.org/abs/2512.02556)
9. Reweighted KL (DeepSeek V3.2)
10. Off-policy sequence masking (DeepSeek V3.2)
11. Keep sampling mask for top-p / top-k (DeepSeek V3.2)
12. Keep original GRPO advantage normalization (DeepSeek V3.2)
13. Per-reward group-wise normalization before aggregation ([GDPO by Liu et al., 2026](https://arxiv.org/abs/2601.05242))
14. Sequence-level importance sampling and clipping ([GSPO by Zheng et al., 2025](https://arxiv.org/abs/2507.18071))
15. Clip importance-sampling weights rather than token updates ([CISPO by MiniMax et al., 2025](https://arxiv.org/abs/2506.13585))

(I am planning to do a more detailed write-up one day after finishing the main contents.)

<br>

The following scripts implement some of these improvements:

- `7_7_improvements/olmo3_style.py`: This script implements improvements 1-7 similar to [Olmo 3](https://arxiv.org/abs/2512.13961) on top of [7_5_plus_kl.py](7_5_plus_kl.py)

- `7_7_improvements/deepseek_v32_style.py`: This script implements improvements 8-12 similar to [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) on top of [7_5_plus_kl.py](7_5_plus_kl.py)

- `7_7_improvements/gdpo.py`: Implements [GDPO](https://arxiv.org/abs/2601.05242) on top of [7_6_plus_format_reward.py](7_6_plus_format_reward.py) (since GDPO is a tweak for multiple rewards)

---

**Note**: If you are not a `uv` user, replace `uv run ...py` with `python ...py` in the examples below.

---


&nbsp;
