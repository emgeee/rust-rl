from pathlib import Path
from datetime import datetime
import functools
import json
import os
import time
from typing import Any, Callable

class OxenExperiment:
    """
    An experiment that logs results to an Oxen repository.
    
    Creates a branch based on the run_name and outputs all results to that branch.
    This allows for easy tracking and comparison of different runs.
    """
    def __init__(self, repo, model_name, output_dir, run_name="GRPO"):
        self.repo = repo
        self.output_dir = output_dir
        
        # Clean the run_name to be a valid branch name
        self.branch_name = self._sanitize_branch_name(run_name.replace(" ", "_").lower())
        
        # Set up experiment metadata
        self.experiment_number = 0  # For compatibility with existing code
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Set name and directory (name is the original run_name)
        self.name = run_name
        self.dir = Path(self.output_dir) / self.name
        
        # Create the directory
        self.dir.mkdir(parents=True, exist_ok=True)
        
        # Check if branch exists
        existing_branches = [branch.name for branch in repo.branches()]
        
        if self.branch_name in existing_branches:
            print(f"Using existing branch: {self.branch_name}")
            # Switch to the branch
            repo.checkout(self.branch_name)
        else:
            print(f"Creating new experiment branch: {self.branch_name}")
            # Create and checkout a new branch
            repo.create_checkout_branch(self.branch_name)
        
        # Save experiment metadata
        self._save_metadata(model_name, timestamp)

    def _sanitize_branch_name(self, name):
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
    
    def _save_metadata(self, model_name, timestamp):
        """Save metadata about the experiment to a JSON file."""
        metadata = {
            "model_name": model_name,
            "branch_name": self.branch_name,
            "run_name": self.name,
            "timestamp": timestamp,
            "output_directory": str(self.dir)
        }
        
        metadata_file = self.dir / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log(self, filename: str) -> Callable:
        """
        Create a decorator for a specific log file.

        Args:
            filename (str): Name of the log file to write to
        """
        log_path = self.dir / filename

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Log the timestamp and function name
                timestamp = datetime.now().isoformat()
                func_name = func.__name__
                start_time = time.time()
                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Record one row for each of the results
                    for i, r in enumerate(result):
                        log_entry = {
                            "timestamp": timestamp,
                            "function": func_name,
                            "score": r,
                            "task_id": kwargs['task_id'][i],
                            "rust_prompt": kwargs['rust_prompt'][i],
                            "completion": kwargs['completions'][i][0]['content'],
                            "func_execution_time": time.time() - start_time
                        }

                        # Write to log file
                        with open(log_path, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')

                except Exception as e:
                    print(f"Could not run func {func_name}: {e}")

                return result

            return wrapper

        return decorator
