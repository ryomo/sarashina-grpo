# Sarashina-GRPO

## What is this?

This repository demonstrates how to fine-tune Sarashina, a Japanese LLM, using GRPO.

Based on: Unsloth's [Tutorial: Train your own Reasoning model with GRPO](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)

日本語の詳細な解説: [日本語LLMのSarashinaをGRPOでトレーニングしてReasoning Modelを作る | ryomo’s tech blog](https://ryomo.github.io/notes/sarashina-grpo)

## Requirements

* NVIDIA GPU
* CUDA
* uv: https://docs.astral.sh/uv/getting-started/installation/

## Installation

```bash
uv sync --frozen
```

Optional: Install nbstripout to automatically remove output cells from Jupyter notebooks before committing them to Git.

```bash
uv run nbstripout --install
```

## Usage

### Inference

Open `notebooks/inference.ipynb` in Jupyter Notebook and "Run All".
You can use the original model, which is not fine-tuned.

### Fine-tuning

Open `notebooks/finetuning.ipynb` in Jupyter Notebook and "Run All".

#### TensorBoard

Monitor the training progress using TensorBoard.

```bash
uv run tensorboard --logdir=./artifact/logs
```

#### Inference by Fine-tuned Model

After training, you can reopen the inference notebook to test the fine-tuned model.
Set the `MODEL_PATH` variable to the path of the fine-tuned checkpoint.
