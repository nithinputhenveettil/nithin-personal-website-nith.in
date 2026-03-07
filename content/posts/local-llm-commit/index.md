---
title: "I Trained a Local LLM to Write Git Commit Messages Because I Am Lazy"
date: 2025-02-01
description: Training a small local LLM using MLX and LoRA to do the needful and generate conventional commit messages from git diffs.
hero: /images/posts/local-llm-commit/hero.svg
menu:
  sidebar:
    name: Finetuning LLM locally
    identifier: blog-local-llm-commit
    weight: 10
tags: ["llm", "machine-learning", "mlx", "apple-silicon", "qwen", "lora", "conventional-commits", "git", "developer-tools", "blog"]
categories: ["Machine Learning", "Developer Tools"]
---

## Introduction

So, this is the very first time I am actually trying to train an LLM locally. I have been using these LLMs for a while now, and honestly, I am quite impressed with their capabilities. So, I thought, why not try some *jugaad* and train one locally on my own machine? 

The setup I used:

- Apple MacBook Pro - M4 Pro (EMI status: ongoing)
- MLX
- LoRA
- Qwen3.5-2B
- LM Studio
- Python

I decided to train the Qwen 3.5 2B Base model to kindly generate human-like commit messages in the conventional commit format, so I can simply stop typing them myself.

## The Trouble with Commit Messages

> Writing good commit messages is a small but constant headache in our daily development lives, I am telling you.

Most of us end up writing absolute nonsense on a Friday evening like:
- `fix stuff`
- `update code`
- `minor change`
- `asdfghjkl`

But well-structured commit messages are extremely useful. They improve:

- Code history readability
- Automated changelogs
- Release notes
- Preventing your tech lead from shouting at you

The **Conventional Commits** format solves this nicely:

```text
feat(auth): add JWT validation
fix(api): handle invalid token error
refactor(core): simplify request pipeline
```

However, typing them out manually every single time? Very repetitive, boss.

So I simply wondered:

> Can I just train a small **local LLM** to look at my git diffs and automatically do the needful by generating conventional commit messages?

In this post, I will walk you through how I built a **commit message generator** by fine-tuning a small open-source model locally using **MLX** and **LoRA**.

The whole drama runs **strictly on my MacBook**. No cloud GPUs, no AWS billing shocks.

---

## Overview

The grand architecture looks somewhat like this:

```text
Git Repositories
        ↓
Extract commits and diffs
        ↓
Clean and balance dataset
        ↓
Fine-tune model with LoRA (MLX)
        ↓
Generate commit messages like a boss
```

The sole purpose is to make the model learn this magic trick:

```text
git diff → conventional commit message
```

---

## Choosing the Model

For this experiment, my weapon of choice was:

```text
Qwen3.5-2B
```

Reasons for choosing it:

- Small enough to run locally without my Mac taking flight like a Boeing.
- Good base instruction-following capability.

Now, to train locally on Apple Silicon, I used **MLX**.

For the uninitiated, MLX is Apple’s own machine learning framework optimized for **Apple GPUs using Metal**. 

For the fine-tuning part, I went with **LoRA (Low-Rank Adaptation)**.

Instead of retraining the whole massive model from scratch and blowing up my laptop, LoRA only updates a very tiny subset of parameters.

Example training output:

```text
Trainable parameters: 0.298% (5.6M / 1.88B)
```

See? We are training **less than 1% of the full model**, which makes the whole process faster than ordering a masala dosa on Swiggy.

---

## Building the Dataset

To train the model, we obviously need examples of:

```text
Diff → Commit message
```

I collected commits from a bunch of popular repositories that generally follow the conventional commit style (and yes, of course, ChatGPT kindly helped me copy-paste the scripts for this process, don't judge me :P).

Repositories used:

```text
angular
cz-cli
nest
next.js
semantic-release
```

From each repository, we extract:

- The commit message
- The git diff

Example training pair so you get the exact idea:

```text
Diff:
# The diff goes here

Commit:
feat(auth): add JWT validation and reject expired tokens
```

---

## Cleaning the Dataset

Now, raw commit history contains a ridiculous amount of garbage:

- Merge commits
- Unnecessarily large diffs
- Auto-generated bot commits
- People typing whatever they want

So, several filters were forcefully applied to clean this mess (Again, ChatGPT was my saviour here :P).

### 1. Remove large diffs

Huge diffs strictly do not fit within the model's context window. 

```python
if diff.count("\n") > 200:
    skip() # Please go away
```

### 2. Enforce the conventional format

We only kept top-tier commits that matched perfectly with these prefixes:

```text
feat:
fix:
refactor:
perf:
```

### 3. Balance the commit types

If one commit type dominates our precious dataset, the model will just rote-memorize it like an engineering student the night before the exam. We need balance.

Final dataset distribution:

```text
refactor 400
feat     400
fix      400
perf     200
```

Total dataset size:

```text
~1400 samples
```

---

## Training Format

Each training sample we prepared exactly looks like this (yes, actual JSONL format, not some fake string):

```jsonl
{"messages": [{"role": "user", "content": "Generate a conventional commit message for this diff:\n\n+import { DestroyRef } from '@angular/core';\n+import { MonoTypeOperatorFunction } from 'rxjs';\n+// @public\n+export function takeUntilDestroyed<T>(destroyRef?: DestroyRef): MonoTypeOperatorFunction<T>;\n+\n+export {takeUntilDestroyed} from './take_until_destroyed';\n+/**\n+ * @license\n+ * Copyright Google LLC All Rights Reserved.\n+ *\n+ * Use of this source code is governed by an MIT-style license that can be\n+ * found in the LICENSE file at https://angular.io/license\n+ */\n+\n+import {assertInInjectionContext, DestroyRef, inject} from '@angular/core';\n+import {MonoTypeOperatorFunction, Observable} from 'rxjs';\n+import {takeUntil} from 'rxjs/operators';\n+\n+/**\n+ * Operator which completes the Observable when the calling context (component, directive, service,\n+ * etc) is destroyed.\n+ *\n+ * @param destroyRef optionally, the `DestroyRef` representing the current context. This can be\n+ *     passed explicitly to use `takeUntilDestroyed` outside of an injection context. Otherwise, the\n+ * current `DestroyRef` is injected.\n+ *\n+ * @developerPreview\n+ */\n+exp"}, {"role": "assistant", "content": "feat(core): implement `takeUntilDestroyed` in rxjs-interop (#49154)"}]}
```

The model essentially learns to play the role of a helpful "assistant" and reply with a properly formatted conventional commit message after analyzing the user's diff.

---

## Training the Model

MLX provides a very straightforward command to kick off the LoRA training. Just fire this in your terminal and relax:

```bash
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

Important parameters to note:

| Parameter | Value |
|----------|------|
| Iterations | 1000 |
| Batch size | 1 |
| Sequence length | 256 |
| Learning rate | 5e-5 |

The entire training runs peacefully on the **Apple GPU via MLX**.

Note: Please be informed that I had to do a lot of trial-and-error with these parameters for quite some time to get decent results. 

---

## Training Progress

The training logs showed a nice, gradual improvement over time.

```text
Iter 1:   Val loss 1.66
Iter 200: Val loss 1.30
Iter 400: Val loss 1.25
Iter 1000: Val loss 1.18
```

Final report card:

```text
Train loss: 1.13
Val loss:   1.18
```

The validation loss steadily decreased, meaning our model actually learned the mapping instead of just mugging up the answers (no overfitting, yay!).

---

## Generating Commit Messages

After the training was successfully completed, the LoRA adapter could be loaded hand-in-hand with the base model.

Example prompt to test it out:

```text
Generate a conventional commit message.

Diff:
+ add JWT validation
+ reject expired tokens

Commit:
```

The magical inference command:

```bash
mlx_lm.generate \
  --model ./Qwen3.5-2B-Base \
  --adapter-path ./commit-lora \
  --temp 0 \
  --max-tokens 20 \
  --prompt "<prompt>"
```

---

## Results

Let’s do a face-off. **Base model vs Fine-Tuned model**. Who will win?

### The Base Model Response

```text
add JWT validation
reject expired tokens
```

The base model simply repeats whatever is in the diff. Absolutely zero effort, and it basically **fails completely to generate a structured commit message**.

### The Fine-Tuned Model Response

```text
feat(core): add JWT validation and reject expired tokens
```

Oh, just look at that beauty! The fine-tuned model:

- Uses the proper **conventional commit format**
- Summarizes the changes nicely
- Even throws in a **scope** for good measure!

The behaviour clearly reflects the good manners it learned from our carefully crafted dataset.

---

## Observations

During testing, a few hilarious and interesting patterns popped up.

### 1. Learned commit structure

The model completely by-hearted the typical format:

```text
type(scope): description
```

### 2. Scope generation

Sometimes the model confidently generates scopes like:

```text
feat(core):
fix(api):
```

It most likely picked up this habit from highly structured projects like Angular.

### 3. Fake GitHub PRs

Occasionally, the model starts hallucinating heavily and drops PR numbers from thin air (wow, that's what I call a pro-level improvement, honestly :P):

```text
feat(core): add JWT validation (#12345)
```

This happens because some of our training repositories actually include PR references in their commit messages, and the model simply thought, "Why not fake it till you make it?"

---

## Limitations

As with all great initial *jugaads*, this experiment has a few limitations.

### Small dataset

We only used:

```text
~1400 commits
```

Actual production systems usually train on **tens of thousands** of examples to get things right.

### Diff truncation

If your diff is excessively long (longer than 256 tokens), it just gets brutally truncated during training.

### Occasional hallucinations

As mentioned, the model sometimes generates PR numbers out of nowhere, acting strictly like a developer who just wants to close random JIRA tickets to look busy.

---

## Future Improvements

There is definitely plenty of scope for improvement here (pun strictly intended).

### 1. Larger dataset

We could expand the dataset to around **30k commits** using heavyweights like:

```text
react
kubernetes
docker
vite
grafana
prometheus
terraform
```

Doing this would definitely boost the overall performance.

### 2. Better scope detection

We could kindly teach the model to infer scopes directly from the changed directories.

For example:

```text
auth/
api/
core/
```

### 3. Longer context window

We could supposedly increase the max sequence length to **384–512 tokens** to handle larger diffs. (But listen yaar, as powerful as my personal machine is, it simply can't handle that much heat right now :P)

---

## Why Local Fine-Tuning is Awesome

One of the most satisfying things about this whole experiment is that **everything runs locally**.

No renting cloud GPUs. No asking your manager for AWS credits. No expensive infrastructure whatsoever.

Just you, your cutting chai, and:

```text
MacBook Pro M4 Pro + MLX + LoRA
```

This makes exploring and building domain-specific LLMs highly accessible for hard-working developers like us.

---

## Final Thoughts

This experiment practically proves that even with a **small dataset and some lightweight tuning**, you can happily teach an AI some very useful developer workflows.

Our fine-tuned buddy successfully learned to:

- Interpret git diffs
- Generate proper conventional commit messages
- Follow specific commit style guidelines

And the entire setup happily runs locally on a laptop without catching fire.

As local LLM tooling becomes even better, we will definitely see more specialized AI models taking over our daily chores, like:

- Commit generation
- PR summaries
- Code review suggestions
- Changelog generation

These small, highly focused models can be surprisingly effective. 

And honestly, it's quite fascinating. I literally can't wait to see what the future holds for us developers!

---

## References

- **MLX Framework:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Qwen Models:** [QwenLM](https://github.com/QwenLM)
- **Conventional Commits:** [conventionalcommits.org](https://www.conventionalcommits.org/)

