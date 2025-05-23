"""
Evaluation utilities for Rust code
"""

import pandas as pd
from pathlib import Path
import shutil
from uuid import uuid4

from ..reward_functions.utils import RustTool, template_rs_file, cargo_toml_file


def extract_rust_code(rust_code: str) -> str:
    """
    Extract the Rust code from a markdown code block.
    
    Args:
        rust_code: String containing Rust code, possibly within markdown code blocks
        
    Returns:
        Extracted Rust code
    """
    if "```rust" in rust_code:
        code = rust_code.split("```rust")[-1]
        code = code.split("```")[0]
        return code.strip()
    else:
        return rust_code


def setup_and_test_rust_project(row, tools):
    """
    Sets up a Rust project from template and runs tests for a single row of data.
    
    Args:
        row: Dictionary containing response with Rust code
        tools: List of RustTool objects to run on the project
        
    Returns:
        Dictionary with test results
    """
    # Create temporary project directory
    project_dir = Path("outputs") / Path("tests") / Path(f"rust_project_{uuid4()}")
    project_dir_src = project_dir / Path("src")

    # mkdirs if they don't exist
    project_dir_src.mkdir(parents=True, exist_ok=True)

    # Read template
    template = template_rs_file()

    # Replace placeholders
    rust_code = extract_rust_code(row["response"])
    template = template.replace("// {code}", rust_code)

    print(template)

    # Write the cargo project files
    main_rs_path = project_dir_src / Path("main.rs")
    with open(main_rs_path, "w") as f:
        f.write(template)

    cargo_file_path = project_dir / Path("Cargo.toml")
    with open(cargo_file_path, "w") as f:
        f.write(cargo_toml_file())

    results = {"template": template}

    for tool in tools:
        results = tool.run(results, project_dir)

    # Clean up
    shutil.rmtree(project_dir)

    return results


def evaluate_solutions(df, tools, output_file, progress_bar=None, max_rows=-1):
    """
    Evaluates all solutions in the dataframe.
    
    Args:
        df: Pandas DataFrame with solutions to evaluate
        tools: List of RustTool objects to run on each solution
        output_file: Path to write the results to
        progress_bar: Optional progress bar object
        max_rows: Maximum number of rows to evaluate (-1 for all)
        
    Returns:
        DataFrame with added tool results columns
    """
    results = []

    total_passed = 0
    total_failed = 0
    num_rows = len(df) if max_rows < 0 else max_rows
    
    progress_context = progress_bar(total=num_rows) if progress_bar else None
    bar = progress_context.__enter__() if progress_context else None
    
    try:
        for idx, row in df.iterrows():
            if max_rows > 0 and idx >= max_rows:
                break

            test_results = setup_and_test_rust_project(row, tools)
            test_results["idx"] = idx
            # merge the row with the test results
            row_dict = row.to_dict()
            row_dict.update(test_results)
            results.append(row_dict)

            num_tools = len(tools)
            num_passed = 0
            for tool in tools:
                passed = test_results[f"{tool.name}_passed"]
                if passed:
                    num_passed += 1
            all_passed = num_passed == num_tools
            print(f"Row {idx}: {num_passed}/{num_tools} passed")
            if all_passed:
                total_passed += 1
            else:
                total_failed += 1
            print(f"Total passed: {total_passed}, Total failed: {total_failed}")
            
            # print percentage
            accuracy = total_passed / (idx + 1) * 100
            percent_passed_str = f"Percentage passed {total_passed}/{idx + 1} = {accuracy:.1f}%"
            print(percent_passed_str)
            
            if bar:
                bar.update(title=percent_passed_str)

            if idx % 100 == 0:
                results_df = pd.DataFrame(results).set_index("idx")
                results_df.to_parquet(output_file)

        # Convert results to dataframe and merge with original
        results_df = pd.DataFrame(results).set_index("idx")
        return results_df
    
    finally:
        if progress_context:
            progress_context.__exit__(None, None, None)
