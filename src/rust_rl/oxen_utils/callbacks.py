import os
import json
from datetime import datetime
from transformers import TrainerCallback

from .experiment import OxenExperiment

class OxenTrainerCallback(TrainerCallback):
    """
    TrainerCallback for logging experiment progress to Oxen
    
    Handles periodic commits to the experiment branch and logs
    training metrics.
    """
    def __init__(self, experiment: OxenExperiment, progress_bar, commit_every):
        self.experiment = experiment
        self.bar = progress_bar
        self.commit_every = commit_every
        self.log_file_name = "logs.jsonl"
        self.log_file = os.path.join(self.experiment.dir, self.log_file_name)
        self.dst_dir = os.path.dirname(self.log_file)
        self.workspace = self._create_workspace()
        super().__init__()
    
    def _create_workspace(self):
        # Import here to avoid circular imports
        from oxen import Workspace
        return Workspace(
            self.experiment.repo,
            branch=self.experiment.branch_name,
            workspace_name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        # add timestamp to logs
        logs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # add branch name to logs for tracking
        logs['branch_name'] = self.experiment.branch_name

        # save logs to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(logs) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"on_step_end {state.global_step}")
        self.bar.update()

        if state.global_step % self.commit_every == 0:
            try:
                # Create a more descriptive commit message
                commit_message = (
                    f"Step {state.global_step}: Training update for {self.experiment.name}\n\n"
                    f"Model updates at step {state.global_step} of training"
                )
                
                # Add all files in the experiment directory that need to be tracked
                for dir_path, _, files in os.walk(self.experiment.dir):
                    for file_name in files:
                        path = os.path.join(dir_path, file_name)
                        # Add all JSON, JSONL, and checkpoint files
                        if path.endswith(("jsonl", "json", "pt", "bin")) or "/checkpoint-" in path:
                            self.workspace.add(path, dst=str(self.experiment.dir))
                
                # Commit changes
                self.workspace.commit(commit_message)
                print(f"Committed changes to branch: {self.experiment.branch_name}")
            except Exception as e:
                print(f"Error committing to Oxen: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Commit final state at the end of training."""
        try:
            # Create a final commit message
            commit_message = (
                f"Training complete for {self.experiment.name}\n\n"
                f"Final model state after {state.global_step} steps"
            )
            
            # Add all files in the experiment directory
            for dir_path, _, files in os.walk(self.experiment.dir):
                for file_name in files:
                    path = os.path.join(dir_path, file_name)
                    # Add all relevant files
                    if path.endswith(("jsonl", "json", "pt", "bin", "txt")) or "/checkpoint-" in path:
                        self.workspace.add(path, dst=str(self.experiment.dir))
            
            # Commit final changes
            self.workspace.commit(commit_message)
            print(f"Training complete. Final state committed to branch: {self.experiment.branch_name}")
        except Exception as e:
            print(f"Error committing final state to Oxen: {e}")