import marimo

__generated_with = "0.13.9"
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
    eval_df = pl.read_parquet("qwen3-rust-finetune/cargo_test_passed_eval.parquet")
    eval_df
    return


@app.cell
def _(pl):
    train_df = pl.read_parquet("qwen3-rust-finetune/cargo_test_passed_train.parquet")
    train_df
    return


@app.cell
def _(re):
    def extract_regex(text: str, pattern: str) -> str | None:
        # Use re.DOTALL to make '.' match newlines as well
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1)
        else:
            return None


    def extract_code_regex():
        return r"```rust\n(.*?)\n```"


    def extract_test_regex():
        return r"(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\})"


    def extract_rust_code(response: str) -> str:
        code = extract_regex(response, extract_code_regex())
        if code:
            return code
        else:
            return response


    def extract_test_code(response: str) -> str:
        return extract_regex(response, extract_test_regex())


    def response_contains_one_code_block(response: str) -> float:
        # It has to have a ```rust``` block and a fn
        if extract_rust_code(response) and "fn " in response:
            return 0.5
        else:
            return 0.0


    def response_contains_one_test_block(response: str) -> float:
        if extract_test_code(response):
            return 0.5
        else:
            return 0.0


    def response_contains_asserts(response: str) -> float:
        test_code = extract_test_code(response)
        if not test_code:
            return 0.0

        unique_asserts = set()
        for line in test_code.split("\n"):
            line = line.strip()
            if line.startswith("assert!(") or line.startswith("assert_eq!("):
                unique_asserts.add(line)
        if len(unique_asserts) >= 4:
            return 1.0
        return 0.25 * len(unique_asserts)


    def response_contains_more_than_non_empty_line(response: str) -> float:
        if not (
            response_contains_one_code_block(response)
            and response_contains_one_test_block(response)
        ):
            return 0.0

        code = extract_rust_code(response)
        num_non_empty = 0
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("//"):
                continue
            if len(line) < 2:
                continue
            num_non_empty += 1
        return 1.0 if num_non_empty >= 3 else 0.0


    def template_rs_file():
        return """
    #![allow(dead_code)]
    // {code}

    // Need basic main function for the code to compile
    fn main() {
      println!("Hello World");
    }
    """


    def cargo_toml_file():
        return """
    [package]
    name = "rust-program"
    version = "0.1.0"
    edition = "2021"

    [dependencies]
    """
    return


app._unparsable_cell(
    r"""
    results = []
    with open(\"qwen3-rust-finetune/outputs/GRPO_17_2025-05-15_00-23-30_Qwen2.5-Coder-1.5B-Instruct/non_empty_rewards.jsonl\") as f:
        for line in f:
            results.append(json.loads(line))

    results = [r for r in results if r['score'] == 0.0]

    # for r in results:
    r = results[]
    completion = r.get('completion')
    print(f\"score: {r.get('score')}, func: {response_contains_more_than_non_empty_line(completion)}\")
    print(completion)
        # print(r.get('completion'))
    
    """,
    name="_"
)


@app.cell
def _():
    import marimo as mo
    import os
    import json
    import re

    import oxen
    from oxen import RemoteRepo
    from oxen import Repo
    from oxen.remote_repo import create_repo

    import polars as pl
    return RemoteRepo, os, oxen, pl, re


if __name__ == "__main__":
    app.run()
