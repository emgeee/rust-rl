import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _(RemoteRepo, os, oxen):

    # Connect your client
    ox_repo = RemoteRepo("ox/Rust")


    repo_name = "qwen3-rust-finetune"
    if os.path.exists(repo_name):
      # if you already have a local copy of the repository, you can load it
      repo = oxen.Repo(repo_name)
    else:
      # if you don't have a local copy of the repository, you can clone it
      repo = oxen.clone(f"mgreen/{repo_name}")

    # Pull the latest changes from the remote repository
    repo.pull()
    return


@app.cell
def _(pl):
    _df = pl.read_parquet("qwen3-rust-finetune/results/Qwen3-4B/predictions_code_and_tests.parquet")
    _df
    return


@app.cell
def _(pl):
    _df = pl.read_parquet("qwen3-rust-finetune/cargo_test_passed_eval.parquet")
    _df
    return


@app.cell
def _():
    import marimo as mo
    import os

    import oxen
    from oxen import RemoteRepo
    from oxen import Repo
    from oxen.remote_repo import create_repo

    import polars as pl
    return RemoteRepo, os, oxen, pl


if __name__ == "__main__":
    app.run()
