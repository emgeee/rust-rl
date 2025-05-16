from pathlib import Path
from datetime import datetime
import functools
import json
import time
from typing import Any, Callable

class OxenExperiment:
    """
    An experiment helps log the experiment to an oxen repository,
    keeps track of the name and creates a corresponding branch to save results to
    """
    def __init__(self, repo, model_name, output_dir, experiment_type="GRPO"):
        self.repo = repo
        self.output_dir = output_dir

        experiment_number = 0
        # branches = repo.branches()
        # for branch in branches:
        #     if branch.name.startswith(f"{experiment_type}_"):
        #         experiment_number += 1
        self.experiment_number = experiment_number
        short_model_name = model_name.split('/')[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.name = f"{experiment_type}_{experiment_number}_{timestamp}_{short_model_name}"

        self.name = f"{experiment_type}_{short_model_name}"
        self.dir = Path(self.output_dir) / self.name

        # if self.dir.exists():
            # shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating experiment branch {self.name}")
        repo.create_checkout_branch(self.name)

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