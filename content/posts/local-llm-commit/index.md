---
title: "I Trained a Local LLM to Write Git Commit Messages and here is the result"
date: 2026-03-07
description: Training a small local LLM using MLX and LoRA to do the needful and generate conventional commit messages from git diffs.
hero: /images/posts/local-llm-commit/hero.png
menu:
  sidebar:
    name: Finetuning LLM locally
    identifier: blog-local-llm-commit
    weight: 10
tags: ["llm", "machine-learning", "mlx", "apple-silicon", "qwen", "lora", "conventional-commits", "git", "developer-tools", "blog"]
categories: ["Machine Learning", "Developer Tools"]
---

## Introduction

Machine Learning was never a new thing for me, it fascinated me every single time since I first started learning it. Back in 2014, while checking out the latest research papers for my academic project, the sheer capabilities of ML completely blew my mind. 

Instead of writing endless `if-else` conditions, we could actually let the model learn the logic and do the needful. How fascinating is that?

Back then, I didn't even have a proper internet connection at home. All I had was a Linux PC with an Intel Dual-Core CPU, a whopping 2GB of RAM, and an Idea 2G data card. The speed was worse than a classic dial-up connection, and highly unstable too.

But I still remember the thrill. I literally had to kill the X-Server (the Linux GUI) just to squeeze out some extra processing juice to train the model. 

That college project was a massive success (or at least I like to say that it was). I managed to train a model to learn the specific writing styles of various authors and predict who wrote which book. You can check out that vintage code right here: [Authorship Predictor](https://github.com/nithinputhenveettil/authorship-predictor)

Fast forward 12 years to yesterday, and I decided to train another model. This time, the mission was to fine-tune a local LLM to generate conventional git commit messages straight from diffs. And thankfully, computing power is no longer an issue. My current personal machine is a MacBook Pro with an M4 Pro chip and 24GB of RAM, better in absolutely every sense.

After using all these huge LLMs for a while and being thoroughly impressed, I just thought: why not try a custom setup and cook one up locally on my own machine? 

The joy and happiness remained exactly the same as 12 years ago. The computing power, the tools, and the entire process have changed drastically, but the core thrill of training a model? Still intact. 

So grab your coffee, because in this blog, I am taking you along for the ride. Let's get started.

---

## The Setup

Here is the exact setup I used:

- Apple MacBook Pro (M4 Pro, 24GB RAM)
- MLX Framework
- LoRA (Low-Rank Adaptation)
- Model: `Qwen3.5-2B-Base`
- LM Studio (for testing)
- Python

I decided to train this 2B parameter Qwen Base model to kindly act like a human and generate conventional commit messages for me, simply so I can stop typing them myself. Being lazy is the true mother of invention, right?

---

## The Trouble with Commit Messages

> Writing good commit messages is a small but constant headache in our daily development lives, I am telling you.

Most of us end up writing absolute nonsense on a Friday evening, like:
- `fix stuff`
- `update code`
- `minor change`
- `asdfghjkl`

But well-structured commit messages are extremely useful. They drastically improve:
- Code history readability
- Automated changelogs
- Release notes
- Preventing your tech lead from shouting at you on Monday morning

The **Conventional Commits** format solves this issue beautifully:

```text
feat(auth): add JWT validation
fix(api): handle invalid token error
refactor(core): simplify request pipeline
```

But typing them out manually every single time? Very repetitive, right?

So, I simply wondered:

> Can I just train a small **local LLM** to look at my git diffs and automatically do the needful by generating conventional commit messages?

In this post, I will walk you through exactly how I built a **commit message generator** by fine-tuning a small open-source model locally. 

And the best part? The whole drama runs **strictly on my MacBook**. No cloud GPUs, no AWS billing shocks.

---

## Overview

The architecture looks somewhat like this:

![Architecture Overview](/images/posts/local-llm-commit/overview.svg)

```text
Some Good Open Source Git Repositories
        ↓
Extract commits and their diffs
        ↓
Clean and balance the dataset
        ↓
Fine-tune the model with LoRA (using MLX)
        ↓
Generate commit messages like a boss
```

The sole purpose of this entire exercise is to make the model learn this one neat magic trick:

```text
git diff → conventional commit message
```

---

## Choosing the Model

For this experiment, my choice was:

```text
Qwen3.5-2B-Base
```

The reasons for choosing this specific model:
- It's small enough to run locally without my Mac's temper boiling over.
- It has excellent base instruction-following capabilities.

Now, to train locally on Apple Silicon, I used **MLX**, which is Apple’s very own machine learning framework optimized specifically for Apple GPUs using Metal. 

For the fine-tuning process, I went with **LoRA** (Low-Rank Adaptation).

Instead of retraining the whole massive model from scratch (and blowing up my laptop in the process), LoRA only updates a very tiny subset of parameters. 

Take a look at the training output:

```text
Trainable parameters: 0.298% (5.6M / 1.88B)
```

See? We are training **less than 1%** of the full model, which makes the whole process blazingly fast.

---

## Building the Dataset

To train the model, we obviously need a solid set of examples showing:

```text
Diff → Commit message
```

I collected commits from a bunch of popular repositories that generally follow the conventional commit style (and yes, ChatGPT kindly helped me write the boilerplate scripts for this process :P).

The repositories I used:

```text
angular
cz-cli
nest
next.js
semantic-release
```

From each repository, we extract:
- The git diff
- The corresponding commit message

![Generating the dataset](/images/posts/local-llm-commit/Training_Dataset_Generation.png)

Here is a quick example of a training pair so you get the exact idea:

```text
Diff:
# The raw code diff goes here

Commit:
feat(auth): add JWT validation and reject expired tokens
```

---

### Cleaning the Dataset

Now, raw commit history contains a ridiculous amount of garbage:
- Merge commits
- Unnecessarily massive diffs
- Auto-generated bot commits
- People typing whatever they want

So, several filters were forcefully applied to clean up this mess.

#### 1. Remove large diffs

Huge diffs strictly do not fit within the model's context window.

```python
if diff.count("\n") > 200:
    skip() # Please go away
```

#### 2. Enforce the conventional format

We only kept top-tier commits that matched perfectly with these prefixes:

```text
feat:
fix:
refactor:
perf:
```

#### 3. Balance the commit types

If one commit type dominates our precious dataset, the model will just rote-memorize it like an engineering student the night before the final exams. 

Believe me, I learned this the hard way. During my first few tries, the model was just spewing out `docs:` for absolutely everything because it was the most common type in the raw data. We need balance, yaar!

Final balanced dataset distribution:

```text
refactor: 400
feat:     400
fix:      400
perf:     200
```

Total dataset size:

```text
~1400 samples
```

### The Dataset Extraction Script

In case you want to replicate this setup yourself, here is the exact Python script I used to scrape, filter, and build the dataset directly from local git repositories. It handles all the heavy lifting: extracting diffs, ditching garbage commits, and ensuring our final sample sizes are perfectly balanced.

Just clone whatever repositories you want to use into a `repos` directory and run this script. It will magically create a `dataset` folder for you.

```python
import subprocess
import json
import random
import re
from pathlib import Path
from collections import defaultdict

# Config

REPOS_DIR = "repos"
OUTPUT_DIR = "dataset"

MAX_DIFF_CHARS = 2000
MAX_DIFF_LINES = 200
MIN_DIFF_LINES = 3

VALID_SPLIT = 0.1

TARGET_PER_TYPE = {
    "feat": 400,
    "fix": 400,
    "refactor": 400,
    "perf": 200
}

TYPE_REGEX = re.compile(r"^(feat|fix|refactor|perf)(\(.+\))?:", re.IGNORECASE)

# GIT HELPERS

def get_commits(repo_path):

    commits = subprocess.check_output(
        ["git", "-C", repo_path, "log", "--pretty=%H|%s"],
        text=True
    ).splitlines()

    for line in commits:

        try:
            sha, msg = line.split("|", 1)
        except ValueError:
            continue

        yield sha, msg.strip()

def get_diff(repo_path, sha):

    try:
        diff = subprocess.check_output(
            ["git", "-C", repo_path, "show", sha, "--pretty=", "--unified=0"],
            text=True,
            stderr=subprocess.DEVNULL
        )
    except:
        return None

    if not diff:
        return None

    # skip very large diffs
    if diff.count("\n") > MAX_DIFF_LINES:
        return None

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS]

    # keep only actual code changes
    diff_lines = [
        line for line in diff.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++")
        and not line.startswith("---")
    ]

    if len(diff_lines) < MIN_DIFF_LINES:
        return None

    diff = "\n".join(diff_lines)

    return diff

# CLEANERS

def clean_message(msg):

    msg = msg.replace("<|im_end|>", "")
    msg = msg.strip()

    return msg

# DATASET FORMATTER

def build_example(diff, message):

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Generate a conventional commit message for this diff:\n\n{diff}"
            },
            {
                "role": "assistant",
                "content": message
            }
        ]
    }

# MAIN DATASET BUILDER

def main():

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    buckets = defaultdict(list)

    for repo in Path(REPOS_DIR).iterdir():

        if not repo.is_dir():
            continue

        print("Scanning repo:", repo.name)

        for sha, msg in get_commits(str(repo)):

            msg = clean_message(msg)

            match = TYPE_REGEX.match(msg)

            if not match:
                continue

            commit_type = match.group(1).lower()

            if commit_type not in TARGET_PER_TYPE:
                continue

            if len(buckets[commit_type]) >= TARGET_PER_TYPE[commit_type]:
                continue

            diff = get_diff(str(repo), sha)

            if not diff:
                continue

            example = build_example(diff, msg)

            buckets[commit_type].append(example)

    # combine dataset
    dataset = []

    for k, v in buckets.items():
        dataset.extend(v)

    random.shuffle(dataset)

    # split train / valid
    split_index = int(len(dataset) * (1 - VALID_SPLIT))

    train = dataset[:split_index]
    valid = dataset[split_index:]

    # write train set
    with open(f"{OUTPUT_DIR}/train.jsonl", "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")

    # write validation set
    with open(f"{OUTPUT_DIR}/valid.jsonl", "w") as f:
        for row in valid:
            f.write(json.dumps(row) + "\n")

    # stats
    print("\nDataset distribution:")

    for k, v in buckets.items():
        print(k, len(v))

    print("\nTrain:", len(train))
    print("Valid:", len(valid))

    print("\nDataset written to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
```

---

## Training Format

Each training sample we prepared looks exactly like this. Yes, we used proper JSONL formatting, not some fake string arrays:

```json
{"messages": [{"role": "user", "content": "Generate a conventional commit message for this diff:\n\n+import { DestroyRef } from '@angular/core';\n+import { MonoTypeOperatorFunction } from 'rxjs';\n+// @public\n+export function takeUntilDestroyed<T>(destroyRef?: DestroyRef): MonoTypeOperatorFunction<T>;\n+\n+export {takeUntilDestroyed} from './take_until_destroyed';\n+/**\n+ * @license\n+ * Copyright Google LLC All Rights Reserved.\n+ *\n+ * Use of this source code is governed by an MIT-style license that can be\n+ * found in the LICENSE file at https://angular.io/license\n+ */\n+\n+import {assertInInjectionContext, DestroyRef, inject} from '@angular/core';\n+import {MonoTypeOperatorFunction, Observable} from 'rxjs';\n+import {takeUntil} from 'rxjs/operators';\n+\n+/**\n+ * Operator which completes the Observable when the calling context (component, directive, service,\n+ * etc) is destroyed.\n+ *\n+ * @param destroyRef optionally, the `DestroyRef` representing the current context. This can be\n+ *     passed explicitly to use `takeUntilDestroyed` outside of an injection context. Otherwise, the\n+ * current `DestroyRef` is injected.\n+ *\n+ * @developerPreview\n+ */\n+exp"}, {"role": "assistant", "content": "feat(core): implement `takeUntilDestroyed` in rxjs-interop (#49154)"}]}
```

The model essentially learns to nicely play the role of an "assistant". It looks at the user's diff, and replies with a properly formatted conventional commit message.

---

## Training the Model

MLX provides a very straightforward command to kick off the LoRA training. Just fire this command in your terminal, sit back, and relax:

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

A few important parameters to note:

| Parameter | Value |
|----------|------|
| Iterations | 1000 |
| Batch size | 1 |
| Sequence length | 256 |
| Learning rate | 5e-5 |

The entire training runs peacefully on the **Apple GPU via MLX**. 

![Starting the LoRA Finetuning process](/images/posts/local-llm-commit/finetuning-start.png)

*Note: Please be informed that I had to do quite a bit of trial-and-error with these parameters to actually get decent results.*

Overall, the training only took around **20 - 25 minutes** to complete!

---

## Training Progress

The training logs showed a lovely, gradual improvement over time:

```text
Iter 1:   Val loss 1.66
Iter 200: Val loss 1.30
Iter 400: Val loss 1.25
Iter 1000: Val loss 1.18
```

And finally, our report card:

```text
Train loss: 1.13
Val loss:   1.18
```

![Finetuning completed successfully](/images/posts/local-llm-commit/finetuning-end.png)

The validation loss steadily decreased, meaning our model actually learned the mapping instead of just mugging up the answers (no overfitting, yay!).

---

## Testing (Generating Commit Messages)

After the training was successfully completed, the LoRA adapter could be loaded hand-in-hand with the base model. Of course, we can use MLX for the inference, or load it up in LM Studio or any other tool of your choice.

Here are the freshly baked adapter files sitting nicely in my directory:

![LoRA adapter files generated after finetuning](/images/posts/local-llm-commit/Finetuning_Files.png)

Let's use an example prompt to test it out:

```text
Generate a conventional commit message.

Diff:
+ add JWT validation
+ reject expired tokens

Commit:
```

And here is the magical inference command:

```bash
mlx_lm.generate \
  --model ./Qwen3.5-2B-Base \
  --adapter-path ./commit-lora-new \
  --temp 0 \
  --max-tokens 20 \
  --prompt "<prompt>"
```

---

## Results

Let’s do a face-off. **Base model vs. Fine-Tuned model**. Who will win?

### The Base Model Response

```text
add JWT validation
reject expired tokens
```

![Testing the base model response](/images/posts/local-llm-commit/Test_Result_Base_Model.png)

The base model simply repeats whatever is in the diff. Absolutely zero effort, and it basically **fails completely** to generate a structured commit message.

### The Fine-Tuned Model Response

```text
feat(docs-infra): add readme for explanation (#63444)
```

![Testing the fine-tuned model response](/images/posts/local-llm-commit/Test_Result_After_Finetuning.png)

Oh, just look at that beauty! The fine-tuned model:
- Uses the proper **conventional commit format**
- Summarizes the changes nicely
- Even throws in a **scope** for good measure!
- And of course, it hallucinates a totally fake PR number :P 

The behaviour clearly reflects the good manners it learned from our carefully crafted dataset.

The model hallucinated the PR number because most of the commit messages in our training set actually had PR numbers glued to them. But honestly, that's not a big deal. For my use case, all I wanted was for the model to learn the conventional commit structure, and this proves that it absolutely did. In a way, even this hallucination is a strong sign of success!

---

## Observations

During testing, a few hilarious and interesting patterns popped up.

### 1. Learned commit structure

The model completely by-hearted the typical format:

```text
type(scope): description
```

### 2. Scope generation

Sometimes the model confidently generates specific scopes like:

```text
feat(core):
fix(api):
```

It most likely picked up this habit from highly structured projects we used for training, like Angular.

### 3. Fake GitHub PRs

Sometimes, the model hallucinates heavily and drops PR numbers out of thin air (wow, that's what I call a pro-level developer improvement, honestly :P):

```text
feat(core): add JWT validation (#12345)
```

As mentioned earlier, it just simply thought, "Why not fake it till you make it?"

---

## Limitations

As with all great initial setups, this experiment has a few limitations.

### Small dataset

We only used:

```text
~1400 commits
```

Actual production systems usually train on **tens of thousands** of examples to get things exactly right.

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

Doing this would definitely boost the overall performance. But it will obviously take a lot of time to train, and my MacBook will absolutely curse me for sure :P

### 2. Better scope detection

We could kindly teach the model to infer scopes directly from the changed directories. For example:

```text
auth/
api/
core/
```

### 3. Longer context window

We could theoretically increase the max sequence length to **384–512 tokens** to handle much larger diffs. 

---

## Why Local Fine-Tuning is Awesome

One of the most satisfying things about this whole experiment is that **everything runs locally**.

No renting cloud GPUs. No asking your manager for AWS credits. No expensive infrastructure whatsoever.

Just you, your coffee, and:

```text
MacBook Pro M4 Pro + MLX + LoRA
```

This makes exploring and building domain-specific LLMs highly accessible for developers like us.

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

These small, highly focused models can be surprisingly effective. And honestly, it's quite fascinating. I literally can't wait to see what the future holds for us developers!

---

## References

- **MLX Framework:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Qwen Models:** [QwenLM](https://github.com/QwenLM)
- **Conventional Commits:** [conventionalcommits.org](https://www.conventionalcommits.org/)
