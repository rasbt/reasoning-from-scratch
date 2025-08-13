# Build A Reasoning Model (From Scratch)

This repository contains the code for developing an LLM reasoning model and is the official code repository for the book *Build A Reasoning Model (From Scratch)*.

&nbsp;

[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

&nbsp;

## Table of Contents (In Progress)




| Chapter Title                                                | Main Code                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ch 1: Understanding reasoning models                         | No code                                                      |
| Ch 2: Generating text with a pre-trained LLM                 | - [ch02_main.ipynb](ch02/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/ch02_exercise-solutions.ipynb) |
| Ch 3: Evaluating reasoning models                            | TBA                                                          |
| Ch 4: Improving reasoning with inference-time scaling        | TBA                                                          |
| Ch 5: Training reasoning models with reinforcement learning  | TBA                                                          |
| Ch 6: Distilling reasoning models for efficient reasoning    | TBA                                                          |
| Ch 7: Improving the reasoning pipeline and future directions | TBA                                                          |
| Appendix A: References and further reading                   | No code                                                      |
| Appendix B: Exercise solutions                               | No code; solutions are in each chapter's subfolder           |
| Appendix C: Qwen3 LLM source code                            | - [chC_main.ipynb](chC/chC_main.ipynb)                       |

<br>
&nbsp;

The mental model below summarizes the main techniques covered in this book (the reasoning techniques 1-3) and how they relate to conventional LLMs.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">



<br>
<br>

To download a copy of this code repository, click on the [Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip) button or execute the following command in your terminal:

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>

(If you downloaded the code bundle from the Manning website, please consider visiting the official code repository on GitHub at [https://github.com/rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) for the latest updates.)

<br>
<br>



&nbsp;
## Companion Book

Please note that *Build A Reasoning Model (From Scratch)* is a standalone book focused on methods to improve LLM reasoning. From chapter 1, the definition of reasoning is:

> Reasoning, in the context of LLMs, refers to the model's ability to produce intermediate steps before providing a final answer. This is a process that is often described as chain-of-thought (CoT) reasoning. In CoT reasoning, the LLM explicitly generates a structured sequence of statements or computations that illustrate how it arrives at its conclusion.

In this book, we work with a pre-trained open-source base LLM (Qwen3) on top of which we code apply reasoning methods from scratch. If you are interested in understanding how a conventional base LLM is implemented, you may like my previous book, [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D).

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="150px"></a>

- [Amazon link](https://amzn.to/4fqvn0D)
- [Manning link](http://mng.bz/orYv)
- [GitHub repository](https://github.com/rasbt/LLMs-from-scratch)


<br>
&nbsp;

## Hardware Requirements

The code in the main chapters of this book is designed to mostly run on consumer hardware within a reasonable timeframe and does not require specialized hardware. This approach ensures that a wide audience can engage with the material. Additionally, the code automatically utilizes GPUs if they are available. That being said, chapters 2-4 will work well on CPUs and GPUs. For chapters 5 and 6, it is recommended to use a GPU if you want to replicate the results in the chapter.


(Please see the [setup_tips](ch02/https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/python-instructions.md) doc for additional recommendations.)

&nbsp;
## Exercises

Each chapter of the book includes several exercises. The solutions are summarized in Appendix B, and the corresponding code notebooks are available in the main chapter folders of this repository (for example,  `ch02/01_main-chapter-code/exercise-solutions.ipynb`).

&nbsp;
## Questions, Feedback, and Contributing to This Repository


I welcome all sorts of feedback, best shared via the Manning Forum (URL TBD) or [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions). Likewise, if you have any questions or just want to bounce ideas off others, please don't hesitate to post these in the forum as well.

Please note that since this repository contains the code corresponding to a print book, I currently cannot accept contributions that would extend the contents of the main chapter code, as it would introduce deviations from the physical book. Keeping it consistent helps ensure a smooth experience for everyone.

&nbsp;
## Citation

If you find this book or code useful for your research, please consider citing it.

Chicago-style citation:

> Raschka, Sebastian. *Build A Reasoning Model (From Scratch)*. Manning, 2025. ISBN: TBD.

BibTeX entry:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Reasoning Model (From Scratch)},
  publisher    = {Manning},
  year         = {2025},
  isbn         = {TBD},
  url          = {TBD},
  github       = {https://github.com/rasbt/reasoning-from-scratch}
}
```