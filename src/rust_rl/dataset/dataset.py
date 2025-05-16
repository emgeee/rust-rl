from datasets import Dataset, load_dataset


def create_dataset(path, system_prompt) -> Dataset:
    """
    Create a dataset for training from a parquet file.

    Args:
        path: Path to the parquet file containing Rust code examples
        system_prompt: System prompt to use for the model

    Returns:
        Dataset object with prompt and test_list fields
    """
    data = load_dataset("parquet", data_files={"train": path})["train"]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["rust_prompt"]},
            ],
            "test_list": x["rust_test_list"],
        }
    )
    return data

