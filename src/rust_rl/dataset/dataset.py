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
            "task_id": x.get("task_id", "unknown"),  # Add task_id field
        }
    )
    return data


def prepare_hf_dataset(dataset, system_prompt) -> Dataset:
    """
    Prepare a HuggingFace dataset for training.

    Args:
        dataset: HuggingFace dataset (should have rust_prompt and rust_test_list fields)
        system_prompt: System prompt to use for the model

    Returns:
        Dataset object with prompt and test_list fields
    """
    # Transform the dataset to the format expected by the trainer
    prepared_dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["rust_prompt"]},
            ],
            "test_list": x["rust_test_list"],
            "task_id": x.get("task_id", str(x.get("id", "unknown"))),  # Add task_id field
        }
    )
    return prepared_dataset