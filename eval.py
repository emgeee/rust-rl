import marimo

__generated_with = "0.13.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from uuid import uuid4

    # Import from our project modules
    from rust_rl.reward_functions.utils import RustTool
    from rust_rl.evaluation import evaluate_solutions, plot_results

    return RustTool, evaluate_solutions, mo, pd, plot_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Rust Eval 🦀

    This is an example of writing custom eval code that pulls a data frame from an oxen repo, then running code on it. In this case we have a data frame with columns `rust_code`, `rust_test_list` and run `cargo build`, `cargo clippy`, and `cargo test` then save the results.

    The predictions file is generated from the `inference.py` script
    """
    )
    return


@app.cell
def _(RustTool, mo):
    tools = [RustTool("build"), RustTool("clippy"), RustTool("test")]
    should_add_tests = True
    num_rows = -1  # -1 == all

    model_name = "Qwen3-4B"

    file_path_text = mo.ui.text(
        value=f"./qwen3-rust-finetune/results/{model_name}/predictions_code_and_tests.parquet",
        full_width=True,
    )
    output_path_text = mo.ui.text(
        value=f"./qwen3-rust-finetune/results/{model_name}/results_code_and_tests.parquet",
        full_width=True,
    )

    run_form = (
        mo.md(
            """
        Enter the local path to your data frame:
        {file_path_text}
        Output file path
        {output_path_text}
        """
        )
        .batch(
            file_path_text=file_path_text,
            output_path_text=output_path_text,
        )
        .form(
            submit_button_label="Run Eval",
            bordered=False,
            show_clear_button=True,
            clear_button_label="Reset",
        )
    )

    run_form
    return file_path_text, num_rows, output_path_text, run_form, tools


@app.cell
def _(
    evaluate_solutions,
    file_path_text,
    mo,
    num_rows,
    output_path_text,
    pd,
    run_form,
    tools,
):
    # If the button is not pressed, stop execution
    mo.stop(run_form.value is None)

    # Read in df from oxen
    df = pd.read_parquet(file_path_text.value)

    # Use the module's evaluate_solutions function with the progress bar
    results = evaluate_solutions(
        df, tools, output_path_text.value, 
        progress_bar=mo.status.progress_bar,
        max_rows=num_rows
    )
    return (results,)


@app.cell
def _(mo, plot_results, results):
    # Use the module's plot_results function with marimo's as_html
    display = plot_results(results, as_html_func=mo.as_html)
    display
    return


@app.cell
def _(output_path_text, results):
    results.to_parquet(output_path_text.value)
    results
    return


if __name__ == "__main__":
    app.run()
