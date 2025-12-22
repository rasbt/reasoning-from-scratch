# Build A Reasoning Model (From Scratch)

This repository contains the code for developing an LLM reasoning model and is the official code repository for the book [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B).


<br>
<br>

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="250px"></a>

(Printed in color.)

<br>

In [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B), you will learn and understand how a reasoning large language model (LLM) works.

Reasoning is one of the most exciting and important recent advances in improving LLMs, but it’s also one of the easiest to misunderstand if you only hear the term reasoning and read about it in theory. This is why this book takes a hands-on approach. We will start with a pre-trained base LLM and then add reasoning capabilities ourselves, step by step in code, so you can see exactly how it works.

The methods described in this book walk you through the process of developing your own small-but-functional reasoning model for educational purposes. It mirrors the approaches used in creating large-scale reasoning models such as DeepSeek R1, GPT-5 Thinking, and others. In addition, this book includes code for loading the weights of existing, pretrained models.

- Link to the official [source code repository](https://github.com/rasbt/reasoning-from-scratch)
- Link to the [book at Manning](https://mng.bz/lZ5B) (the publisher's website)
- Link to the book page on Amazon.com (TBD)
- ISBN 9781633434677



<br>
<br>

To download a copy of this repository, click on the [Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip) button or execute the following command in your terminal:

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>


> **Tip:**
> Chapter 2 provides additional tips on installing Python, managing Python packages, and setting up your coding environment.

<br>
<br>

## Table of Contents (In Progress)

[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

| Chapter Title                                                | Main Code                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ch 1: Understanding reasoning Models                         | No code                                                      |
| Ch 2: Generating Text with a Pre-trained LLM                 | - [ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| Ch 3: Evaluating Reasoning Models                            | - [ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br/>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| Ch 4: Improving Reasoning with Inference-Time Scaling        | - [ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br/>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| Ch 5: Inference-Time Scaling via Self-Refinement             | - [ch05_main.ipynb](ch05/01_main-chapter-code/ch05_main.ipynb)<br/>- [ch05_exercise-solutions.ipynb](ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) |
| Ch 6: Training Reasoning Models with Reinforcement Learning  | TBA                                                          |
| Ch 7: Improving Policy Optimization in Reinforcement Learning | TBA                                                          |
| Ch 8: Distilling Reasoning Models for Efficient Reasoning    | TBA                                                          |
| Appendix A: References and Further Reading                   | No code                                                      |
| Appendix B: Exercise Solutions                               | Code and solutions are in each chapter's subfolder           |
| Appendix C: Qwen3 LLM Source Code                            | - [chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb)  |
| Appendix D                                                   | TBA                                                          |
| Appendix E                                                   | TBA                                                          |
| Appendix F: Common Approaches to LLM Evaluation              | - [chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb)  |

<br>
&nbsp;

The mental model below summarizes the main techniques covered in this book.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">



<br>



&nbsp;
## Companion Book

Please note that *Build A Reasoning Model (From Scratch)* is a standalone book focused on methods to improve LLM reasoning.

In this book, we work with a pre-trained open-source base LLM (Qwen3) on top of which we code apply reasoning methods from scratch. This includes inference-time scaling, reinforcement learning, and distillation.

However, if you are interested in understanding how a conventional base LLM is implemented, you may like my previous book, [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D).

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="120px"></a>

- [Amazon link](https://amzn.to/4fqvn0D)
- [Manning link](http://mng.bz/orYv)
- [GitHub repository](https://github.com/rasbt/LLMs-from-scratch)


<br>
&nbsp;

## Hardware Requirements

The code in the main chapters of this book is designed to mostly run on consumer hardware within a reasonable timeframe and does not require specialized server hardware. This approach ensures that a wide audience can engage with the material. Additionally, the code automatically utilizes GPUs if they are available. That being said, chapters 2-4 will work well on CPUs and GPUs. For chapters 5 and 6, it is recommended to use a GPU if you want to replicate the results in the chapter.


(Please see the [setup_tips](ch02/https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/01_main-chapter-code/python-instructions.md) doc for additional recommendations.)

&nbsp;
## Exercises

Each chapter of the book includes several exercises. The solutions are summarized in Appendix B, and the corresponding code notebooks are available in the main chapter folders of this repository (for example,  [`ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb`](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb)).


&nbsp;
## Bonus Material

Several folders contain optional materials as a bonus for interested readers:

- **Chapter 2: Generating Text with a Pre-trained LLM**
  - [Optional Python Setup and Cloud GPU Recommendations](ch02/02_setup-tips)
  - [Using a GPU-optimized version of the LLM](ch02/03_optimized-LLM)
  - [Using `torch.compile()` on Windows](ch02/04_torch-compile-windows)
  - [Run inference and chat with the model](ch02/05_use_model)
- **Chapter 3: Evaluating LLMs**
  - [MATH-500 Verifier Scripts](ch03/02_math500-verifier-scripts)

- **Chapter 4: Improving Reasoning with Inference-Time Scaling**
  - [Inference Scaling on MATH-500 ](ch04/02_math500-inference-scaling-scripts)

- **Appendix F: Common Approaches to LLM Evaluation**
  - [MMLU Evaluation Methods](chF/02_mmlu)
  - [LLM leaderboards](chF/03_leaderboards)
  - [LLM-as-a-judge](chF/04_llm-judge)


&nbsp;
## Questions, Feedback, and Contributing to This Repository


I welcome all sorts of feedback, best shared via the [Manning Discussion Forum](https://livebook.manning.com/forum?product=raschka2&page=1) or [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions). Likewise, if you have any questions or just want to bounce ideas off others, please don't hesitate to post these in the forum as well.

Please note that since this repository contains the code corresponding to a print book, I currently cannot accept contributions that would extend the contents of the main chapter code, as it would introduce deviations from the physical book. Keeping it consistent helps ensure a smooth experience for everyone.

&nbsp;
## Citation

If you find this book or code useful for your research, please consider citing it.

Chicago-style citation:

> Raschka, Sebastian. *Build A Reasoning Model (From Scratch)*. Manning, 2025. ISBN: 9781633434677.

BibTeX entry:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Reasoning Model (From Scratch)},
  publisher    = {Manning},
  year         = {2025},
  isbn         = {9781633434677},
  url          = {https://mng.bz/lZ5B},
  github       = {https://github.com/rasbt/reasoning-from-scratch}
}
```
