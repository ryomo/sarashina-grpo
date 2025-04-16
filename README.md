# Sarashina-GRPO

## Requirements

* NVIDIA GPU
* CUDA
* uv: https://docs.astral.sh/uv/getting-started/installation/

## Installation

```bash
uv sync --frozen
```

Optional: Install nbstriopout to remove output from Jupyter notebooks.

```bash
uv run nbstripout --install
```

## Usage

### TensorBoard

Check the training progress with TensorBoard.

```bash
uv run tensorboard --logdir=./artifact/logs
```
