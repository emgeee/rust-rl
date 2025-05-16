#!/usr/bin/env python3
"""
Standalone evaluation script for Rust-RL project.

This script provides functionality to evaluate generated Rust code
against cargo build, cargo clippy, and cargo test requirements.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from oxen import RemoteRepo, Repo, auth, Workspace

# Import from our project modules
from rust_rl.reward_functions.utils import RustTool
from rust_rl.evaluation.evaluator import evaluate_solutions
from rust_rl.evaluation.visualize import plot_results


class SimpleProgressBar:
    """A simple progress bar for tracking evaluation."""
    
    def __init__(self, total):
        self.total = total
        self.pbar = tqdm(total=total, dynamic_ncols=True)
        
    def update(self, title=None):
        if title:
            self.pbar.set_description(title)
        self.pbar.update(1)
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        self.pbar.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated Rust code against cargo requirements"
    )
    
    # Input parameters
    parser.add_argument("--model-name", type=str, 
                      default="Qwen3-4B",
                      help="Model name used for file paths")
    parser.add_argument("--run-name", type=str,
                      default=None,
                      help="Name for this evaluation run (defaults to 'eval_{model_name}')")
    
    # Oxen config parameters
    parser.add_argument("--oxen-repo", type=str, 
                      default="mgreen/rust-rl",
                      help="Oxen repository (single repo for both input and output)")
    parser.add_argument("--oxen-key", type=str, 
                      default=None,
                      help="Optional Oxen API key. If not provided, uses env var OXEN_KEY")
    parser.add_argument("--predictions-path", type=str, 
                      default="results/{model_name}/predictions_code_and_tests.parquet",
                      help="Path within Oxen repo to predictions file (supports {model_name} placeholder)")
    parser.add_argument("--results-path", type=str, 
                      default="results/{model_name}/results_code_and_tests.parquet",
                      help="Path within Oxen repo to save results (supports {model_name} placeholder)")
    parser.add_argument("--results-plot", type=str, 
                      default="results/{model_name}/results_plot.png",
                      help="Path within Oxen repo to save results plot (supports {model_name} placeholder)")
    
    # Dataset parameters
    parser.add_argument("--local-cache-dir", type=str, 
                      default="cache",
                      help="Local directory to cache Oxen dataset files")
    parser.add_argument("--sample-size", type=int, default=-1,
                      help="Number of samples to evaluate (-1 for all)")
    
    # Evaluation configuration
    parser.add_argument("--run-build", action="store_true", default=True,
                      help="Run cargo build evaluation")
    parser.add_argument("--run-clippy", action="store_true", default=True,
                      help="Run cargo clippy evaluation")
    parser.add_argument("--run-test", action="store_true", default=True,
                      help="Run cargo test evaluation")
    parser.add_argument("--save-every", type=int, default=100,
                      help="Save results every N samples")
    parser.add_argument("--commit-results", action="store_true", default=True,
                      help="Commit results to Oxen repo")
    
    return parser.parse_args()


def sanitize_branch_name(name):
    """
    Ensure the branch name is valid for git.
    Remove problematic characters and truncate if too long.
    """
    # Replace invalid characters
    invalid_chars = [':', ' ', '\\', '~', '^', ':', '?', '*', '[', ']', '{', '}']
    for char in invalid_chars:
        name = name.replace(char, '_')
    
    # Remove consecutive underscores
    while '__' in name:
        name = name.replace('__', '_')
    
    # Truncate if too long (git typically has a limit around 250 chars)
    max_length = 100
    if len(name) > max_length:
        name = name[:max_length]
    
    return name


def setup_oxen(args):
    """Set up Oxen repository and authenticate if needed."""
    print(f"Setting up Oxen repository: {args.oxen_repo}")
    
    # Set up authentication if provided
    if args.oxen_key:
        print("Using provided Oxen API key")
        auth.set_key(args.oxen_key)
    elif "OXEN_KEY" in os.environ:
        print("Using Oxen API key from environment")
    else:
        print("Warning: No Oxen API key found. Assuming you're already authenticated.")
    
    # Create local cache directory if it doesn't exist
    if not os.path.exists(args.local_cache_dir):
        os.makedirs(args.local_cache_dir, exist_ok=True)
    
    # Get the Oxen repo
    repo_path = os.path.join(args.local_cache_dir, args.oxen_repo.replace("/", "_"))
    
    # Check if repo exists locally
    if os.path.exists(repo_path):
        print(f"Using existing local repository at: {repo_path}")
        repo = Repo(repo_path)
        # Pull latest changes
        print("Pulling latest changes...")
        repo.pull()
    else:
        print(f"Cloning repository to: {repo_path}")
        repo = Repo.clone(args.oxen_repo, repo_path)
    
    # Create remote repo reference
    remote_repo = RemoteRepo(args.oxen_repo)
    
    return repo, remote_repo, repo_path


def format_path_with_model_name(path_template, model_name):
    """Format a path template with model name."""
    return path_template.format(model_name=model_name)


def save_results_plot(results_df, output_path):
    """Save visualization of evaluation results."""
    print(f"Saving results plot to: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Plot the results
    plots = plot_results(results_df)
    
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Copy the content from each figure to the new subplots
    for ax, (name, fig_obj) in zip(axes, plots.items()):
        # Get the data from the original figure
        data = fig_obj.axes[0].containers[0].datavalues
        labels = ["Passed", "Failed"]
        colors = ["#6fcb9f", "#fb2e01"]
        
        # Add title
        title = fig_obj.axes[0].get_title()
        ax.set_title(title)
        
        # Plot the data
        ax.bar(labels, data, color=colors)
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(output_path)
    
    # Close all figures to free memory
    plt.close('all')
    
    return output_path


def setup_branch(repo, branch_name):
    """Set up a branch for the evaluation results."""
    # Check if branch exists
    existing_branches = [branch.name for branch in repo.branches()]
    
    if branch_name in existing_branches:
        print(f"Using existing branch: {branch_name}")
        repo.checkout(branch_name)
    else:
        print(f"Creating new branch: {branch_name}")
        repo.create_checkout_branch(branch_name)
    
    return branch_name


def save_metadata(output_dir, metadata):
    """Save metadata about the evaluation run."""
    metadata_file = os.path.join(output_dir, "evaluation_metadata.json")
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return metadata_file


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set default run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = args.model_name.split("/")[-1].replace(".", "_")
        args.run_name = f"eval_{model_short_name}_{timestamp}"
    
    # The branch name is simply the sanitized run_name
    branch_name = sanitize_branch_name(args.run_name)
    
    # Format paths with model name
    predictions_path = format_path_with_model_name(args.predictions_path, args.model_name)
    results_path = format_path_with_model_name(args.results_path, args.model_name)
    results_plot_path = format_path_with_model_name(args.results_plot, args.model_name)
    
    # Setup Oxen repos and ensure data is available
    repo, remote_repo, repo_path = setup_oxen(args)
    
    # Set up branch for results
    setup_branch(repo, branch_name)
    
    # Get full paths
    full_predictions_path = os.path.join(repo_path, predictions_path)
    full_results_path = os.path.join(repo_path, results_path)
    full_results_plot_path = os.path.join(repo_path, results_plot_path)
    
    # Ensure predictions file exists
    if not os.path.exists(full_predictions_path):
        print(f"Downloading predictions file: {predictions_path}")
        remote_repo.download(predictions_path, target_path=full_predictions_path)
    else:
        print(f"Using existing predictions file: {full_predictions_path}")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(full_results_path), exist_ok=True)
    
    # Read in predictions
    print(f"Reading predictions from: {full_predictions_path}")
    df = pd.read_parquet(full_predictions_path)
    print(f"Loaded {len(df)} predictions")
    
    # Limit sample size if specified
    if args.sample_size > 0:
        print(f"Limiting to {args.sample_size} samples")
        df = df.head(args.sample_size)
    
    # Configure tools to run
    tools = []
    if args.run_build:
        tools.append(RustTool("build"))
    if args.run_clippy:
        tools.append(RustTool("clippy"))
    if args.run_test:
        tools.append(RustTool("test"))
    
    # Save run metadata
    metadata = {
        "run_name": args.run_name,
        "branch_name": branch_name,
        "model_name": args.model_name,
        "tools": [tool.name for tool in tools],
        "sample_size": args.sample_size if args.sample_size > 0 else len(df),
        "timestamp": datetime.now().isoformat(),
        "predictions_path": predictions_path,
        "results_path": results_path,
        "results_plot_path": results_plot_path
    }
    metadata_file = save_metadata(os.path.dirname(full_results_path), metadata)
    
    print(f"Running evaluation with tools: {[tool.name for tool in tools]}")
    
    # Setup progress bar
    progress_bar = SimpleProgressBar
    
    # Create workspace for committing results
    workspace = Workspace(
        repo,
        branch=branch_name,
        workspace_name=f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run evaluation
    print(f"Running evaluation, saving results to: {full_results_path}")
    results = evaluate_solutions(
        df, 
        tools, 
        full_results_path, 
        progress_bar=progress_bar,
        max_rows=args.sample_size
    )
    
    # Save results
    print(f"Saving final results to: {full_results_path}")
    results.to_parquet(full_results_path)
    
    # Save visualization
    full_results_plot_path = save_results_plot(results, full_results_plot_path)
    
    # Commit results if requested
    if args.commit_results:
        commit_message = f"Evaluation results for {args.model_name}\n\nRun: {args.run_name}"
        
        # Add files to commit
        files_to_commit = [full_results_path, full_results_plot_path, metadata_file]
        for file_path in files_to_commit:
            if os.path.exists(file_path):
                print(f"Adding file: {file_path}")
                workspace.add(file_path, dst=os.path.dirname(results_path))
        
        # Commit and push
        try:
            workspace.commit(commit_message)
            print(f"Committed results to branch: {branch_name}")
        except Exception as e:
            print(f"Error committing results: {e}")
    
    print(f"Evaluation complete. Results saved to: {full_results_path}")
    print(f"Results plot saved to: {full_results_plot_path}")
    print(f"Branch: {branch_name}")


if __name__ == "__main__":
    main()