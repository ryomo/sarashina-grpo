from datasets import Dataset, load_dataset
from sarashina_grpo.config import PROJECT_ROOT, SYSTEM_PROMPT


def get_dataset() -> Dataset:
    train_csv = f"{PROJECT_ROOT}/datasets/smalltalk.csv"

    # Load the dataset from a CSV file
    # https://huggingface.co/docs/datasets/v3.5.0/en/package_reference/loading_methods#datasets.load_dataset
    # NOTE: If `split` is not specified, `load_dataset` will return a dict with the splits("train" or "test") as keys.
    dataset = load_dataset("csv", data_files=train_csv, split="train")

    # Remap the dataset
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["prompt"]},
            ],
            "lang": x["lang"],
        }
    )

    # Filter the dataset to include only Japanese responses
    dataset = dataset.filter(lambda x: x["lang"] == "ja")

    return dataset


def main():
    dataset = get_dataset()
    print(f"dataset: {dataset}")
    print(f"dataset[0]: {dataset[0]}")


if __name__ == "__main__":
    main()
