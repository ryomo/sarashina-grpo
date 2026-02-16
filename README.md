# Sarashina-GRPO

## What is this?

This repository demonstrates how to fine-tune Sarashina, a Japanese LLM, using GRPO.

Based on: Unsloth's [Tutorial: Train your own Reasoning model with GRPO](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)

日本語の詳細な解説: [日本語LLMのSarashinaをGRPOでトレーニングしてReasoning Modelを作る | ryomo’s tech blog](https://ryomo.github.io/notes/sarashina-grpo)

## Requirements

* NVIDIA GPU
* CUDA
* uv: <https://docs.astral.sh/uv/getting-started/installation/>

## Installation

```bash
uv sync --frozen
```

Optional: Install nbstripout to automatically remove output cells from Jupyter notebooks before committing them to Git.

```bash
uv run nbstripout --install
```

## Usage

Choose either **marimo** (`notebooks/marimo/`) or **Jupyter Notebook** (`notebooks/jupyter/`) for the following steps.

### 1. Inference before fine-tuning

Open `inference.py` (marimo) or `inference.ipynb` (Jupyter) and "Run All".
This uses the original model, which is NOT fine-tuned.

### 2. Fine-tuning

Open `finetuning.py` (marimo) or `finetuning.ipynb` (Jupyter) and "Run All".

### 3. Inference with fine-tuned model

Reopen the inference notebook and set the `MODEL_PATH` variable to the fine-tuned checkpoint path (e.g., `./artifact/outputs/checkpoint-500`).

### TensorBoard

Monitor training progress:

```bash
uv run tensorboard --logdir=./artifact/logs
```
