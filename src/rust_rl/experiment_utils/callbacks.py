import os
import json
from datetime import datetime
from transformers import TrainerCallback

from .experiment import Experiment

class TrainingCallback(TrainerCallback):
    """
    TrainerCallback for logging experiment progress.
    
    Handles periodic saving of training metrics.
    """
    def __init__(self, experiment: Experiment, progress_bar, save_every):
        self.experiment = experiment
        self.bar = progress_bar
        self.save_every = save_every
        self.log_file_name = "logs.jsonl"
        self.log_file = os.path.join(self.experiment.dir, self.log_file_name)
        self.dst_dir = os.path.dirname(self.log_file)
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # add timestamp to logs
        logs['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # add run name to logs for tracking
        logs['run_name'] = self.experiment.name

        # save logs to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(logs) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        print(f"on_step_end {state.global_step}")
        self.bar.update()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log final state at the end of training."""
        try:
            print(f"Training complete for {self.experiment.name}")
        except Exception as e:
            print(f"Error logging final state: {e}")