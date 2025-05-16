import os
import sys

def check_file_exists(path):
    """Check if a file exists and print status."""
    exists = os.path.exists(path)
    print(f"{'✅' if exists else '❌'} {path} - {'Exists' if exists else 'Missing'}")
    return exists

def main():
    # Add the src directory to the Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    # Check main package structure
    package_paths = [
        "src/rust_rl/__init__.py",
        "src/rust_rl/reward_functions/__init__.py",
        "src/rust_rl/reward_functions/utils.py",
        "src/rust_rl/reward_functions/functions.py",
        "src/rust_rl/oxen_utils/__init__.py",
        "src/rust_rl/oxen_utils/experiment.py",
        "src/rust_rl/oxen_utils/callbacks.py",
        "src/rust_rl/dataset/__init__.py",
        "src/rust_rl/dataset/dataset.py",
        "src/rust_rl/prompts/__init__.py",
        "src/rust_rl/prompts/system_prompts.py",
        "src/rust_rl/evaluation/__init__.py",
        "src/rust_rl/evaluation/evaluator.py",
        "src/rust_rl/evaluation/visualize.py",
    ]
    
    print("Checking package structure...")
    all_exist = all(check_file_exists(p) for p in package_paths)
    
    if all_exist:
        print("\nAll required files exist.")
        
        # Try to import modules
        print("\nTrying to import modules...")
        try:
            import rust_rl
            print(f"✅ Successfully imported rust_rl package")
            
            import rust_rl.reward_functions
            print(f"✅ Successfully imported rust_rl.reward_functions")
            
            import rust_rl.oxen_utils
            print(f"✅ Successfully imported rust_rl.oxen_utils")
            
            import rust_rl.dataset
            print(f"✅ Successfully imported rust_rl.dataset")
            
            import rust_rl.prompts
            print(f"✅ Successfully imported rust_rl.prompts")
            
            import rust_rl.evaluation
            print(f"✅ Successfully imported rust_rl.evaluation")
            
            # Check specific imports
            from rust_rl.reward_functions import RustTool
            print(f"✅ Successfully imported RustTool")
            
            from rust_rl.oxen_utils import OxenExperiment
            print(f"✅ Successfully imported OxenExperiment")
            
            from rust_rl.dataset import create_dataset
            print(f"✅ Successfully imported create_dataset")
            
            from rust_rl.prompts import RUST_SYSTEM_PROMPT
            print(f"✅ Successfully imported RUST_SYSTEM_PROMPT")
            
            from rust_rl.evaluation import evaluate_solutions, plot_results
            print(f"✅ Successfully imported evaluation functions")
            
            print("\nAll imports successful!")
        except ImportError as e:
            print(f"❌ Import error: {e}")
    else:
        print("\nSome required files are missing. Please check the structure.")
        
if __name__ == "__main__":
    main()