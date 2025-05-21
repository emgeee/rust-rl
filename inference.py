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
    model_name = mo.ui.text(value="Qwen/Qwen2.5-Coder-1.5B-Instruct", full_width=True)
    dataset_path = mo.ui.text(value="qwen3-rust-finetune/cargo_test_passed_eval.parquet", full_width=True)
    
    run_form = mo.md(
        """
        Model Name
        {model_name}
        Dataset Path
        {dataset_path}
        """
    ).batch(
        dataset_path=dataset_path,
        model_name=model_name,
    ).form(
        submit_button_label="Predict",
        bordered=False,
        show_clear_button=True,
        clear_button_label="Reset"
    )
    run_form
    return model_name, dataset_path, run_form


@app.cell
def _():
    import os
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from transformers import pipeline

    # Import from our module
    from rust_rl.prompts import RUST_SYSTEM_PROMPT

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Path,
        RUST_SYSTEM_PROMPT,
        TextStreamer,
        mo,
        os,
        pd,
    )


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    RUST_SYSTEM_PROMPT,
    TextStreamer,
    mo,
    model_name,
    os,
    dataset_path,
    pd,
    run_form,
    save_results,
):
    # If the button is not pressed, stop execution
    mo.stop(run_form.value is None)

    path = dataset_path.value

    if not os.path.exists(path):
        print(f"File not found: {path}")
        mo.stop()
    else:
        print(f"Using dataset: {path}")

    device = "mps" # Fastest on Mac right now
    # device = "cuda"

    model_name_str = model_name.value
    print(f"Loading model: {model_name_str}")

    output_model_name = model_name_str.split("/")[-1]
    output_path = f"qwen3-rust-finetune/results/{output_model_name}/predictions_code_and_tests.parquet"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
                save_results(results, output_path)
            bar.update()
    save_results(results, output_path)
    return


@app.cell
def _(Path, os, pd):
    def save_results(results, filepath):
        path = Path(filepath)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent, exist_ok=True)

        result_df = pd.DataFrame(results)
        print(f"Saving to {filepath}")
        result_df.to_parquet(filepath)
        return result_df
    return (save_results,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
