import marimo

__generated_with = "0.13.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🦀 Cargo Is All You Need

    ## Training a 1.5B parameter Rust coder with GRPO

    This is an example of how to use reinforcement learning to train a model to code in Rust using cargo as feedback.
    """
    )
    return


@app.cell
def _(os):
    # Parameters
    local_oxen_path = "qwen3-rust-finetune"

    oxen_repo_name_value = "ox/Rust"
    output_oxen_repo_name_value = "mgreen/rust-rl"

    output_dir = f"{local_oxen_path}/outputs"

    # model_name = "Qwen/Qwen3-4B"
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    train_dataset_file = f"{local_oxen_path}/cargo_test_passed_train.parquet"

    # repo = RemoteRepo(oxen_repo_name_value)
    # if not os.path.exists(train_dataset_file):
    #     repo.download(train_dataset_file)

    save_every_value = 100
    commit_every_value = 100
    num_generations_value = 4
    use_peft = True
    use_gpu = True

    os.environ["WANDB_PROJECT"]="rust-rl"
    os.environ["WANDB_WATCH"]="false"
    return (
        commit_every_value,
        model_name,
        num_generations_value,
        output_dir,
        output_oxen_repo_name_value,
        save_every_value,
        train_dataset_file,
        use_gpu,
    )


@app.cell
def _(
    RemoteRepo,
    model_name,
    output_dir,
    output_oxen_repo_name_value,
    train_dataset_file,
):
    # Import OxenExperiment and related modules
    from rust_rl.oxen_utils import OxenExperiment
    from rust_rl.dataset import create_dataset
    from rust_rl.prompts import RUST_SYSTEM_PROMPT

    # Setup the Experiment
    output_repo = RemoteRepo(output_oxen_repo_name_value)
    experiment = OxenExperiment(output_repo, model_name, output_dir)
    train_dataset = create_dataset(train_dataset_file, RUST_SYSTEM_PROMPT)
    # train_dataset = train_dataset.select(range(10))
    # print(f"Running experiment in dir: {experiment.dir}")
    return experiment, train_dataset


@app.cell
def _(experiment):
    # Import reward functions from the package
    from rust_rl.reward_functions.functions import (
        non_empty_reward_func, 
        tests_have_asserts_reward_func,
        test_block_count_reward_func,
        code_block_count_reward_func,
        cargo_build_reward_func,
        cargo_clippy_reward_func,
        cargo_test_reward_func,
        test_reward_func
    )
    from rust_rl.reward_functions.utils import RustTool

    # Add experiment logging wrappers
    non_empty_reward_func = experiment.log(f"non_empty_rewards.jsonl")(non_empty_reward_func)
    tests_have_asserts_reward_func = experiment.log(f"tests_have_asserts_rewards.jsonl")(tests_have_asserts_reward_func)
    test_block_count_reward_func = experiment.log(f"test_block_count_rewards.jsonl")(test_block_count_reward_func)
    code_block_count_reward_func = experiment.log(f"code_block_count_rewards.jsonl")(code_block_count_reward_func)
    cargo_build_reward_func = experiment.log(f"cargo_build_rewards.jsonl")(cargo_build_reward_func)
    cargo_clippy_reward_func = experiment.log(f"cargo_clippy_rewards.jsonl")(cargo_clippy_reward_func)
    cargo_test_reward_func = experiment.log(f"cargo_test_rewards.jsonl")(cargo_test_reward_func)

    return (
        cargo_build_reward_func,
        cargo_clippy_reward_func,
        cargo_test_reward_func,
        non_empty_reward_func,
        test_block_count_reward_func,
        tests_have_asserts_reward_func,
    )


@app.cell
def training(
    AutoModelForCausalLM,
    AutoTokenizer,
    GRPOConfig,
    GRPOTrainer,
    LoraConfig,
    cargo_build_reward_func,
    cargo_clippy_reward_func,
    cargo_test_reward_func,
    commit_every_value,
    experiment,
    gc,
    mo,
    model_name,
    non_empty_reward_func,
    num_generations_value,
    os,
    save_every_value,
    test_block_count_reward_func,
    tests_have_asserts_reward_func,
    torch,
    train_dataset,
    use_gpu,
):
    # Training Code
    print("Starting training...")
    gc.collect()

    device = "cpu"
    device_map = None
    attn_implementation = None
    if use_gpu:
        device = "cuda"
        device_map = "auto"
        attn_implementation = "flash_attention_2"
    print(f"Using device: {device} -> '{use_gpu}'")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation=attn_implementation,
        device_map=device_map,
    ).to(device)

    # peft_config = None
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        # Trying to get working with 16GB of VRAM
        target_modules="all-linear",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    model.enable_input_require_grads()
    model.train()
    print(f"model in training mode: {model.training}")

    with open(os.path.join(experiment.dir, "peft.txt"), "w") as f:
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                f.write(f"{name}\t{param.numel()}\n")
        param_str = f"trainable params: {trainable_params} || all params: {all_param} || trainable %: {100 * trainable_params / all_param}"
        print(param_str)
        f.write(param_str)

    batch_size = 1
    num_examples = len(train_dataset)
    num_batches = num_examples / batch_size
    print(f"num_examples: {num_examples}, num_batches: {num_batches}")

    with mo.status.progress_bar(total=num_batches) as bar:
        training_args = GRPOConfig(
            output_dir=experiment.dir,
            report_to="wandb",
            learning_rate=5e-6,
            # learning_rate=5e-5,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_generations=num_generations_value,
            max_prompt_length=256,
            max_completion_length=786,
            num_train_epochs=1,
            save_steps=save_every_value,
            save_total_limit=1,
            max_grad_norm=0.1,
            log_on_each_node=False,
            optim="adamw_torch",
            label_names=[],
        )
    
        # Import the callback
        from rust_rl.oxen_utils import OxenTrainerCallback
    
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                # test_reward_func,
                cargo_build_reward_func,  # 1.0 if passes cargo build else 0.0
                cargo_clippy_reward_func,  # 1.0 if passes cargo clippy else 0.0
                cargo_test_reward_func,  # 2.0 if passes cargo test else 0.0
                non_empty_reward_func,  # 1.0 if the code is not empty else 0.0
                test_block_count_reward_func,  # 1.0 if there is a test block else 0.0
                tests_have_asserts_reward_func,  # 1.0 if there are assert statements in the test else 0.0
            ],
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            callbacks=[
                OxenTrainerCallback(
                    experiment, bar, commit_every=commit_every_value
                )
            ],
        )
        trainer.train()
    return


@app.cell
def _():
    # def transform_dataset(system_prompt, dataset, tokenizer):
    #     data = dataset.map(lambda x: {
    #         'messages': [
    #             {'role': 'system', 'content': system_prompt},
    #             {'role': 'user', 'content': x['rust_prompt']},
    #             {'role': 'assistant', 'content': x['rust_code']}
    #         ]
    #     })
    #     tokenized_dataset = data.map(
    #         lambda x: tokenizer([tokenizer.apply_chat_template(conv, tokenize=False) for conv in x['messages']], truncation=True, padding=True),
    #         batched=True,
    #         remove_columns=data.column_names
    #     )
    #     return tokenized_dataset
    return


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model, PeftModel
    from oxen import RemoteRepo, Workspace
    import os
    import numpy as np
    import json
    import torch
    from datetime import datetime
    import gc
    from pathlib import Path

    import wandb
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        GRPOConfig,
        GRPOTrainer,
        LoraConfig,
        RemoteRepo,
        gc,
        mo,
        os,
        torch,
    )


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
