"""
Evaluation utilities for Rust code
"""

import pandas as pd
from pathlib import Path
import shutil
from uuid import uuid4

from ..reward_functions.utils import RustTool, template_rs_file, cargo_toml_file, extract_rust_code, setup_and_test_rust_project as _setup_and_test_rust_project
from ..common.utils import save_dataframe


def setup_and_test_rust_project(row, tools):
    """
    Sets up a Rust project from template and runs tests for a single row of data.
    
    Args:
        row: Dictionary containing response with Rust code (expects "response" field)
        tools: List of RustTool objects to run on the project
        
    Returns:
        Dictionary with test results
    """
    # Use the utils version directly - it now handles both "response" and "rust_code" fields
    results = _setup_and_test_rust_project(row, tools)
    
    # Add template to results for compatibility with existing code that expects it
    if "template" not in results:
        code_content = row.get('rust_code') or row.get('response', '')
        rust_code = extract_rust_code(code_content)
        template = template_rs_file()
        template = template.replace("// {code}", rust_code)
        results["template"] = template
        
        # Add debug print for compatibility
        print(template)
    
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
                save_dataframe(results_df, output_file)

        # Convert results to dataframe and merge with original
        results_df = pd.DataFrame(results).set_index("idx")
        return results_df
    
    finally:
        if progress_context:
            progress_context.__exit__(None, None, None)
