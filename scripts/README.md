# Training Scripts

This directory contains helper scripts for running training with different configurations.

## Usage

### Training Qwen2.5-Coder-1.5B-Instruct

```bash
./scripts/train_1_5b.sh
```

### Training Qwen3-4B

```bash
./scripts/train_4b.sh
```

### Resuming Training from a Checkpoint

If your training gets interrupted or you want to continue from a previous checkpoint:

```bash
./scripts/resume_training.sh /path/to/checkpoint/directory
```

The script will automatically try to find the associated WandB ID to continue tracking in the same run. You can also provide additional arguments:

```bash
./scripts/resume_training.sh /path/to/checkpoint --learning-rate 1e-6 --num-epochs 2
```

### Running Multiple Training Jobs in Parallel

You can run multiple training jobs in parallel by specifying different run names and output directories:

```bash
# First training job
./scripts/train_1_5b.sh --run-name "rust_1_5b_lora16" --output-dir "outputs/rust_lora16"

# Second training job (in another terminal)
./scripts/train_1_5b.sh --run-name "rust_1_5b_lora32" --output-dir "outputs/rust_lora32" --lora-r 32
```

## Command Line Arguments

The training script supports numerous command line arguments:

```
usage: train_script.py [-h] [--model-name MODEL_NAME] [--run-name RUN_NAME] [--output-dir OUTPUT_DIR]
                       [--local-oxen-path LOCAL_OXEN_PATH] 
                       [--resume-from-checkpoint RESUME_FROM_CHECKPOINT]
                       [--resume-wandb-id RESUME_WANDB_ID]
                       [--train-dataset-file TRAIN_DATASET_FILE]
                       [--oxen-repo-name OXEN_REPO_NAME] [--output-oxen-repo-name OUTPUT_OXEN_REPO_NAME]
                       [--sample-size SAMPLE_SIZE] [--save-every SAVE_EVERY] [--commit-every COMMIT_EVERY]
                       [--num-generations NUM_GENERATIONS] [--batch-size BATCH_SIZE]
                       [--learning-rate LEARNING_RATE] [--num-epochs NUM_EPOCHS]
                       [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--use-peft]
                       [--lora-r LORA_R] [--lora-alpha LORA_ALPHA] [--lora-dropout LORA_DROPOUT]
                       [--device DEVICE] [--precision {bf16,fp16,fp32}] [--wandb-project WANDB_PROJECT]
                       [--wandb-entity WANDB_ENTITY] [--wandb-group WANDB_GROUP]
                       [--wandb-tags WANDB_TAGS [WANDB_TAGS ...]]
```

## Examples

### Train with different LoRA settings

```bash
./scripts/train_1_5b.sh --run-name "rust_lora_r32" --lora-r 32 --lora-alpha 128
```

### Train on a subset of data

```bash
./scripts/train_1_5b.sh --run-name "rust_small_sample" --sample-size 100
```

### Train with a different learning rate

```bash
./scripts/train_1_5b.sh --run-name "rust_lr_1e-5" --learning-rate 1e-5
```

### Use a different device

```bash
./scripts/train_1_5b.sh --device mps  # For Apple Silicon
```

### Resume training after interruption

If training was interrupted or you want to continue from a checkpoint:

```bash
# Find your checkpoint directory
ls -la qwen3-rust-finetune/outputs/rust-rl_*/checkpoint-*

# Resume from the checkpoint
./scripts/resume_training.sh qwen3-rust-finetune/outputs/rust-rl_Qwen2.5-Coder-1.5B-Instruct_*/checkpoint-500

# You can also modify parameters when resuming
./scripts/resume_training.sh path/to/checkpoint --learning-rate 1e-6 --num-epochs 2
```

The script will automatically:
1. Check if the checkpoint path exists
2. Look for WandB run IDs in the parent directory to continue the same experiment tracking
3. Resume training from exactly where it left off

## WandB Integration

Training runs are automatically logged to Weights & Biases. You can specify custom project, entity, group, and tags:

```bash
./scripts/train_1_5b.sh --wandb-project "rust-rl-experiments" --wandb-entity "your-team" --wandb-group "lora-tuning"
```