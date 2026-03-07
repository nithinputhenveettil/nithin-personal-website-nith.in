---
title: "I Trained a Local LLM to Write Git Commit Messages"
date: 2025-02-01
description: Training a small local LLM using MLX and LoRA to generate conventional commit messages from git diffs.
hero: /images/posts/local-llm-commit/hero.svg
menu:
  sidebar:
    name: Finetuning LLM locally
    identifier: blog-local-llm-commit
    weight: 10
tags: ["llm", "machine-learning", "mlx", "apple-silicon", "qwen", "lora", "conventional-commits", "git", "eveloper-tools", "blog"]
categories: ["Machine Learning", "Developer Tools"]
---

> Writing good commit messages is a small but constant friction in daily development.

Most of us end up writing things like:
- fix stuff
- update code
- minor change


But well-structured commit messages are extremely useful. They improve:

- Code history readability
- Automated changelogs
- Release notes
- Collaboration across teams

The **Conventional Commits** format solves this nicely:

```
feat(auth): add JWT validation
fix(api): handle invalid token error
refactor(core): simplify request pipeline
```

However, writing them manually every time can still feel repetitive.

So I wondered:

> Can I train a small **local LLM** to generate conventional commit messages directly from git diffs?

In this post, I will walk through how I built a **commit message generator** by fine-tuning a small open model locally using **MLX** and **LoRA**.

The entire experiment runs **on a MacBook**, without requiring any cloud GPUs.

---

## Overview

The full pipeline looks like this:

```
Git Repositories
        ↓
Extract commits and diffs
        ↓
Clean and balance dataset
        ↓
Fine-tune model with LoRA (MLX)
        ↓
Generate commit messages
```

The goal is to train a model that learns the mapping:

```
git diff → conventional commit message
```

---

## Choosing the Model

For this experiment I used:

```
Qwen3.5-2B
```

Reasons for choosing it:

- Small enough to run locally
- Good base instruction capability
- Widely supported in tooling

To train locally on Apple Silicon, I used **MLX**.

MLX is Apple’s machine learning framework optimised for **Apple GPUs using Metal**.

For fine-tuning, I used **LoRA (Low Rank Adaptation)**.

Instead of retraining the entire model, LoRA only updates a very small subset of parameters.

Example training output:

```
Trainable parameters: 0.298% (5.6M / 1.88B)
```

So we are training **less than 1% of the full model**, which makes the process very efficient.

---

## Building the Dataset

To train the model we need examples of:

```
Diff → Commit message
```

I collected commits from several popular repositories that generally follow conventional commit style.

Repositories used:

```
angular
next.js
nestjs
semantic-release
commitizen
```

From each repository we extract:

- commit message
- git diff

Example training pair:

```
Diff:
+ add JWT validation
+ reject expired tokens

Commit:
feat(auth): add JWT validation and reject expired tokens
```

---

## Cleaning the Dataset

Raw commit history contains a lot of noise:

- Merge commits
- Extremely large diffs
- Auto-generated commits
- Inconsistent formatting

So several filters were applied.

### Remove large diffs

Large diffs do not fit within the model context window.

```
if diff.count("\n") > 200:
    skip
```

### Enforce conventional commit format

We only keep commits that match these types:

```
feat:
fix:
refactor:
perf:
```

### Balance commit types

If one commit type dominates the dataset, the model may develop a bias.

Final dataset distribution:

```
refactor 400
feat     400
fix      400
perf     200
```

Total dataset size:

```
~1400 samples
```

---

## Training Format

Each training sample looks like this:

```json
{
 "content": "Write a conventional commit message.

Diff:
+ add JWT validation
+ reject expired tokens

Commit:
feat(auth): add JWT validation and reject expired tokens"
}
```

The model learns to complete the **Commit** section based on the diff.

---

## Training the Model

MLX provides a straightforward command for LoRA training.

```
mlx_lm.lora \
  --model ./Qwen3.5-2B-Base \
  --train \
  --data ./dataset \
  --batch-size 1 \
  --iters 1000 \
  --learning-rate 5e-5 \
  --max-seq-length 256 \
  --adapter-path commit-lora
```

Important parameters:

| Parameter | Value |
|----------|------|
| Iterations | 1000 |
| Batch size | 1 |
| Sequence length | 256 |
| Learning rate | 5e-5 |

Training runs entirely on **Apple GPU through MLX**.

---

## Training Progress

Training logs showed gradual improvement.

```
Iter 1:   Val loss 1.66
Iter 200: Val loss 1.30
Iter 400: Val loss 1.25
Iter 1000: Val loss 1.18
```

Final metrics:

```
Train loss: 1.13
Val loss:   1.18
```

The validation loss steadily decreased, indicating that the model learned the mapping without significant overfitting.

---

## Generating Commit Messages

After training, the LoRA adapter can be loaded together with the base model.

Example prompt:

```
Generate a conventional commit message.

Diff:
+ add JWT validation
+ reject expired tokens

Commit:
```

Inference command:

```
mlx_lm.generate \
  --model ./Qwen3.5-2B-Base \
  --adapter-path ./commit-lora \
  --temp 0 \
  --max-tokens 20 \
  --prompt "<prompt>"
```

---

## Results

Let’s compare the **base model vs the fine-tuned model**.

### Base Model

```
add JWT validation
reject expired tokens
```

The base model simply repeats the diff and does **not generate a structured commit message**.

### Fine-Tuned Model

```
feat(core): add JWT validation and reject expired tokens
```

The fine-tuned model:

- Uses **conventional commit format**
- Summarises the change
- Sometimes adds a **scope**

The behaviour clearly reflects patterns learned from the dataset.

---

## Observations

A few interesting patterns appeared during testing.

### Learned commit structure

The model learned the typical format:

```
type(scope): description
```

### Scope generation

Sometimes the model generates scopes like:

```
feat(core):
fix(api):
```

This likely comes from projects like Angular.

### GitHub style commits

Occasionally the model adds PR numbers:

```
feat(core): add JWT validation (#12345)
```

This happens because some training repositories include PR references in commit messages.

---

## Limitations

This experiment is relatively small in scale.

### Small dataset

Only:

```
~1400 commits
```

Production systems usually train on **tens of thousands** of examples.

### Diff truncation

Diffs longer than 256 tokens are truncated during training.

### Occasional hallucinations

The model sometimes generates PR numbers even when none exist.

---

## Future Improvements

There are several ways this model could be improved.

### Larger dataset

Expanding to around **30k commits** from repositories like:

```
react
kubernetes
docker
vite
grafana
prometheus
terraform
```

would likely improve performance significantly.

### Better scope detection

Scopes could be inferred from changed directories.

Example:

```
auth/
api/
core/
```

### Longer context window

Increasing sequence length to **384–512 tokens** would allow larger diffs.

---

## Why Local Fine-Tuning is Interesting

One of the most interesting aspects of this experiment is that **everything runs locally**.

No cloud GPUs. No expensive infrastructure.

Just:

```
MacBook + MLX + LoRA
```

This makes experimenting with domain-specific LLMs very accessible for individual developers.

---

## Final Thoughts

This experiment shows that even a **small dataset and lightweight fine-tuning** can teach a model useful developer workflows.

The fine-tuned model successfully learned to:

- Interpret git diffs
- Generate conventional commit messages
- Follow commit style patterns

And the entire setup runs locally on a laptop.

As local LLM tooling improves, we will likely see more specialised models trained for developer tasks such as:

- Commit generation
- PR summaries
- Code review suggestions
- Changelog generation

Small, focused models like this can be surprisingly effective.

---

## References

MLX  
https://github.com/ml-explore/mlx

Qwen  
https://github.com/QwenLM

Conventional Commits  
https://www.conventionalcommits.org/
