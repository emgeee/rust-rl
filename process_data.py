import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium")


@app.cell
def _(os):
    # Setup data paths
    repo_name = "qwen3-rust-finetune"
    
    # Create directories if they don't exist
    if not os.path.exists(repo_name):
        os.makedirs(repo_name, exist_ok=True)
    
    # Create results directory if it doesn't exist
    results_dir = f"{repo_name}/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    return


@app.cell
def _(pl):
    _df = pl.read_parquet("qwen3-rust-finetune/results/Qwen3-4B/predictions_code_and_tests.parquet")
    _df
    return


@app.cell
def _(pl):
    eval_df = pl.read_parquet("qwen3-rust-finetune/cargo_test_passed_eval.parquet")
    eval_df
    return


@app.cell
def _(pl):
    train_df = pl.read_parquet("qwen3-rust-finetune/cargo_test_passed_train.parquet")
    train_df
    return


@app.cell
def _():
    import marimo as mo
    import os
    import json
    import re

    from datasets import load_dataset, Dataset

    import polars as pl
    return os, pl


if __name__ == "__main__":
    app.run()