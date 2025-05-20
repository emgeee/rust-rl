import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
    # 🚀 Model Inference

    Run inference over a dataset from a trained model.
    """
    )
    return


@app.cell
def _(mo):
    model_name = mo.ui.text(value="mgreen/QWEN-2.5-1.5B-instruct-rust-ft-8800", full_width=True)
    oxen_repo_name = mo.ui.text(value="mgreen/qwen3-rust-finetune", full_width=True)
    oxen_dataset_name = mo.ui.text(value="qwen3-rust-finetune/cargo_test_passed_eval.parquet", full_width=True)

    run_form = mo.md(
        """
        Model Name
        {model_name}
        Oxen Repo Name
        {oxen_repo_name}
        Dataset Name
        {oxen_dataset_name}
        """
    ).batch(
        oxen_repo_name=oxen_repo_name,
        oxen_dataset_name=oxen_dataset_name,
        model_name=model_name,
    ).form(
        submit_button_label="Predict",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )
    run_form
    return model_name, oxen_dataset_name, oxen_repo_name, run_form


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    from oxen import RemoteRepo, Workspace

    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from transformers import pipeline

    # Import from our module
    from rust_rl.prompts import RUST_SYSTEM_PROMPT

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Path,
        RUST_SYSTEM_PROMPT,
        RemoteRepo,
        TextStreamer,
        Workspace,
        mo,
        os,
        pd,
    )


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    RUST_SYSTEM_PROMPT,
    RemoteRepo,
    TextStreamer,
    mo,
    model_name,
    os,
    oxen_dataset_name,
    oxen_repo_name,
    pd,
    run_form,
    save_results_to_oxen,
):
    # If the button is not pressed, stop execution
    mo.stop(run_form.value is None)

    repo = RemoteRepo(oxen_repo_name.value)
    path = oxen_dataset_name.value

    if not os.path.exists(path):
        print(f"Downloading {path}")
        repo.download(path)
    else:
        print(f"Already have {path}")

    device = "mps" # Fastest on Mac right now
    # device = "cuda"

    model_name_str = model_name.value
    print(f"Downloading {model_name_str}")

    output_model_name = model_name_str.split("/")[1]
    output_path = f"qwen3-rust-finetune/results/{output_model_name}/temp_predictions_code_and_tests.parquet"

    tokenizer = AutoTokenizer.from_pretrained(model_name_str)
    model = AutoModelForCausalLM.from_pretrained(model_name_str).to(device)
    streamer = TextStreamer(tokenizer)

    save_every = 10
    df = pd.read_parquet(path)
    results = []
    with mo.status.progress_bar(total=len(df)) as bar:
        for index, row in df.iterrows():
            print(row)
            messages = [
                {"role": "system", "content": RUST_SYSTEM_PROMPT},
                # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                # {"role": "user", "content": f"{SYSTEM_PROMPT}{row['rust_prompt']}"}
                {"role": "user", "content": f"{row['rust_prompt']}\n"},
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)
            input_text = input_text + "<|im_start|>assistant\n"
            print("=" * 80)
            print(input_text)
            print("=" * 80)
            inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                streamer=streamer,
            )
            full_response = tokenizer.decode(outputs[0])
            response = tokenizer.decode(
                outputs[0][len(inputs[0]) :], skip_special_tokens=True
            )
            print(response)
            results.append(
                {
                    "task_id": row["task_id"],
                    "prompt": row["rust_prompt"],
                    "test_list": row["rust_test_list"],
                    "input": input_text,
                    "full_response": full_response,
                    "response": response,
                }
            )
            if index % save_every == 0:
                save_results_to_oxen(repo, results, output_path, model_name_str)
            bar.update()
    save_results_to_oxen(repo, results, output_path, model_name_str)
    return


@app.cell
def _(Path, RemoteRepo, Workspace, os, pd):
    def save_results_to_oxen(repo: RemoteRepo, results, filepath, model_name):
        path = Path(filepath)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)

        result_df = pd.DataFrame(results)
        print(f"Saving to {filepath}")
        result_df.to_parquet(filepath)
        if False:
            cleaned_model_name = model_name.split("/")[-1].replace('.', '-')
            branch_name = f"results-{cleaned_model_name}"
            # TODO: Need a branch_exists method on RemoteRepo
            branches = [b.name for b in repo.branches()]
            print("Got branches")
            print(branches)
            if not branch_name in branches:
                print(f"Create branch: {branch_name}")
                repo.create_checkout_branch(branch_name)
            workspace = Workspace(repo, branch_name)
            workspace.add(filepath, dst=str(path.parent))
            try:
                workspace.commit(f"Saving {len(results)} results to {filepath}")
            except Exception as e:
                print(f"Did not commit: {e}")
            return result_df
    return (save_results_to_oxen,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
