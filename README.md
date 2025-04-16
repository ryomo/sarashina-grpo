# Sarashina-GRPO

## Requirements

* NVIDIA GPU
* CUDA
* uv: https://docs.astral.sh/uv/getting-started/installation/

## Installation

```bash
uv sync --frozen
```

Optional: Install nbstripout to remove output from Jupyter notebooks.

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
