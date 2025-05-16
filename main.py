import os
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rust_rl.dataset import create_dataset
from rust_rl.oxen_utils import OxenExperiment
from rust_rl.prompts import RUST_SYSTEM_PROMPT
from rust_rl.reward_functions import RustTool


def main():
    print("hello")


if __name__ == "__main__":
    main()

