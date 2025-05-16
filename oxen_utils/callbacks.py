import os
import json
from datetime import datetime
from transformers import TrainerCallback

from .experiment import OxenExperiment

class OxenTrainerCallback(TrainerCallback):
    """
    TrainerCallback for logging experiment progress to Oxen
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
            branch=f"{self.experiment.name}",
            workspace_name=f"training_run_{self.experiment.experiment_number}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        # add timestamp to logs
        logs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # save logs to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(logs) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"on_step_end {state.global_step}")
        self.bar.update()

        if state.global_step % self.commit_every == 0:
            try:
                for dir_path, _, files in os.walk(self.experiment.dir):
                    for file_name in files:
                        path = os.path.join(dir_path, file_name)
                        if path.endswith("jsonl"):
                            self.workspace.add(path, dst=str(self.experiment.dir))
                self.workspace.commit(f"step {state.global_step} end GRPO")
            except Exception as e:
                print(e)