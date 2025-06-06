#!/usr/bin/env python3
"""
Standalone training script for Rust-RL project.

This script provides functionality to train language models for generating Rust code
using GRPO (Generative Reinforcement Learning from Policy Optimization).
"""

# Set tokenizers parallelism before importing transformers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import functools
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from rust_rl.dataset import prepare_hf_dataset
# Import from our project modules
from rust_rl.prompts import RUST_SYSTEM_PROMPT
from rust_rl.reward_functions import (
    cargo_build_reward_func,
    cargo_clippy_reward_func,
    cargo_test_reward_func,
    code_block_count_reward_func,
    non_empty_reward_func,
    test_block_count_reward_func,
    tests_have_asserts_reward_func,
    safe_reward_func,
    create_reward_logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a language model to generate Rust code using GRPO"
    )

    # Core parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Model name/path",
    )
    parser.add_argument(
        "--run-name", type=str, help="Custom run name for wandb and experiment tracking"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mgreen/rust-ft",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Number of samples to use from dataset (-1 for all)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base directory to save outputs",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--resume-wandb-id",
        type=str,
        default=None,
        help="WandB run ID to resume tracking from",
    )

    # Training configuration
    parser.add_argument(
        "--save-every", type=int, default=100, help="Save model every N steps"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    # Lora parameters
    parser.add_argument(
        "--use-peft",
        action="store_true",
        default=True,
        help="Use PEFT (LoRA) for training",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA r parameter")
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout value"
    )

    # Hardware configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, mps, or None for auto)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Precision for training",
    )

    # WandB configuration
    parser.add_argument(
        "--wandb-project", type=str, default="rust-rl", help="WandB project name"
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="WandB group name"
    )
    parser.add_argument(
        "--wandb-tags", type=str, nargs="+", default=None, help="WandB tags"
    )

    return parser.parse_args()


# Define a custom TQDM callback for training progress
class TQDMTrainerCallback(TrainerCallback):
    """Custom callback using tqdm for progress tracking."""
    
    def __init__(self, output_dir):
        """Initialize callback with output directory."""
        super().__init__()
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_logs.jsonl")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.pbar = None
        self.current_step = 0
        self.total_steps = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        if hasattr(args, "max_steps") and args.max_steps > 0:
            self.total_steps = args.max_steps
        else:
            # Estimate total steps from epochs and dataset size
            train_dl = kwargs.get("train_dataloader", None)
            if train_dl:
                batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
                steps_per_epoch = len(train_dl) // batch_size
                self.total_steps = steps_per_epoch * args.num_train_epochs
            else:
                self.total_steps = 1000  # Arbitrary default
        
        self.pbar = tqdm(total=self.total_steps, desc="Training", dynamic_ncols=True)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f"Step {state.global_step}")
            self.current_step = state.global_step
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are about to be sent to logger."""
        if logs:
            # Add timestamp to logs
            logs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save logs to file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(logs) + "\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def setup_wandb(args):
    """Set up Weights & Biases configuration."""
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_WATCH"] = "false"

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    # Use existing run ID if resuming
    if args.resume_wandb_id:
        print(f"Resuming WandB run: {args.resume_wandb_id}")
        os.environ["WANDB_RUN_ID"] = args.resume_wandb_id
        # Keep the original run name
        if args.run_name:
            os.environ["WANDB_NAME"] = args.run_name
    # Otherwise use run_name for a new wandb run
    elif args.run_name:
        run_id = args.run_name.replace(" ", "_").lower()
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_NAME"] = args.run_name
        print(f"Starting new WandB run: {run_id}")


def setup_device(args):
    """Configure the device for training."""
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Configure device map and attention implementation
    device_map = None
    attn_implementation = None

    if device == "cuda":
        device_map = "auto"
        attn_implementation = "flash_attention_2"

    print(f"Using device: {device}")

    # Determine precision
    if args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    return device, device_map, attn_implementation, dtype


def setup_model(args, device, device_map, attn_implementation, dtype):
    """Set up the model for training."""
    print(f"Loading model: {args.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Configure LoRA if using PEFT
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=args.lora_dropout,
        )
    else:
        peft_config = None

    # Check if resuming from checkpoint
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Load the model from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            args.resume_from_checkpoint,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).to(device)
    else:
        # Load model from original source
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).to(device)

    # Prepare model for training
    model.enable_input_require_grads()
    model.train()
    print(f"Model in training mode: {model.training}")

    return tokenizer, model, peft_config


def log_trainable_parameters(model, output_dir):
    """Log trainable parameters to a file."""
    param_file = os.path.join(output_dir, "peft_parameters.txt")
    os.makedirs(os.path.dirname(param_file), exist_ok=True)
    
    with open(param_file, "w") as f:
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


def main():
    """Main training function."""
    args = parse_args()

    # Set up run name if not provided
    if not args.run_name and not args.resume_from_checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model_name.split("/")[-1]
        args.run_name = f"rust-rl_{model_short_name}_{timestamp}"
    elif args.resume_from_checkpoint and not args.run_name:
        # Extract run name from checkpoint path if possible
        checkpoint_dir = Path(args.resume_from_checkpoint)
        if checkpoint_dir.parent.name.startswith("rust-rl_"):
            args.run_name = checkpoint_dir.parent.name
        else:
            # Create a name indicating it's a resumed run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short_name = args.model_name.split("/")[-1]
            args.run_name = f"rust-rl_{model_short_name}_resumed_{timestamp}"

    # Setup WandB
    setup_wandb(args)

    # Setup output directory
    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup device and model
    device, device_map, attn_implementation, dtype = setup_device(args)
    tokenizer, model, peft_config = setup_model(
        args, device, device_map, attn_implementation, dtype
    )

    # If resuming from checkpoint, use that checkpoint's parent directory
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint).parent
        if checkpoint_dir.exists():
            print(f"Using checkpoint's parent directory: {checkpoint_dir}")

    # Load dataset from HuggingFace
    raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Prepare dataset for training
    train_dataset = prepare_hf_dataset(raw_dataset, RUST_SYSTEM_PROMPT)

    # Limit dataset size if specified
    if args.sample_size > 0:
        train_dataset = train_dataset.select(
            range(min(args.sample_size, len(train_dataset)))
        )

    print(f"Dataset loaded with {len(train_dataset)} examples")

    # Log trainable parameters
    log_trainable_parameters(model, output_dir)

    # Free up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate number of batches
    num_examples = len(train_dataset)
    num_batches = num_examples / args.batch_size
    print(f"Number of examples: {num_examples}, Number of batches: {num_batches}")

    # Configure training arguments
    training_args = GRPOConfig(
        output_dir=output_dir,
        report_to="wandb",
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_every,
        save_total_limit=3,
        max_grad_norm=0.1,
        log_on_each_node=False,
        optim="adamw_torch",
        label_names=[],
        # Add run_name to training args for better tracking
        run_name=args.run_name,
    )

    # Create a reward logger
    reward_log_dir = os.path.join(output_dir, "reward_logs")
    os.makedirs(reward_log_dir, exist_ok=True)
    logger = create_reward_logger(reward_log_dir)
    
    # Apply decorators properly - first create the logging decorators
    log_non_empty = logger("non_empty")
    log_tests_have_asserts = logger("tests_have_asserts")
    log_test_block_count = logger("test_block_count")
    log_code_block_count = logger("code_block_count")
    log_cargo_build = logger("cargo_build")
    log_cargo_clippy = logger("cargo_clippy")
    log_cargo_test = logger("cargo_test")
    
    # Then wrap each reward function with logging and safety
    wrapped_reward_funcs = [
        safe_reward_func(log_non_empty(non_empty_reward_func)),
        safe_reward_func(log_tests_have_asserts(tests_have_asserts_reward_func)),
        safe_reward_func(log_test_block_count(test_block_count_reward_func)),
        safe_reward_func(log_code_block_count(code_block_count_reward_func)),
        safe_reward_func(log_cargo_build(cargo_build_reward_func)),
        safe_reward_func(log_cargo_clippy(cargo_clippy_reward_func)),
        safe_reward_func(log_cargo_test(cargo_test_reward_func)),
    ]
    
    # Create a simple progress bar callback
    progress_callback = TQDMTrainerCallback(output_dir=output_dir)
    
    # Set up trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=wrapped_reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        callbacks=[progress_callback],
    )

    # Train the model - determine if resuming
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_path = args.resume_from_checkpoint

        # Verify the checkpoint exists
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")

            # Resume training from the checkpoint
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            print(
                f"Warning: Checkpoint path {checkpoint_path} not found. Starting from scratch."
            )
            trainer.train()
    else:
        print(f"Starting new training run: {args.run_name}")
        trainer.train()

    # Save training metadata
    metadata = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "training_completed": True,
        "completion_time": datetime.now().isoformat(),
        "num_examples": num_examples,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training completed successfully")


if __name__ == "__main__":
    main()
