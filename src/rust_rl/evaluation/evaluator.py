"""
Evaluation utilities for Rust code
"""

import pandas as pd
from pathlib import Path
import shutil
from uuid import uuid4

from ..reward_functions.utils import RustTool, template_rs_file, cargo_toml_file, extract_rust_code, setup_and_test_rust_project as _setup_and_test_rust_project
from ..common.utils import save_dataframe, load_dataframe


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


def print_evaluation_summary(results_df, tools):
    """
    Print comprehensive evaluation statistics from results dataframe.
    
    Args:
        results_df: DataFrame with evaluation results
        tools: List of RustTool objects used in evaluation
    """
    total_evaluated = len(results_df)
    if total_evaluated == 0:
        print("No evaluation results found.")
        return
    
    # Calculate overall pass rate (all tools must pass)
    total_passed = 0
    tool_stats = {tool.name: {"passed": 0, "failed": 0} for tool in tools}
    
    for idx, row in results_df.iterrows():
        num_tools = len(tools)
        num_passed = 0
        for tool in tools:
            passed_col = f"{tool.name}_passed"
            if passed_col in row and row[passed_col]:
                num_passed += 1
                tool_stats[tool.name]["passed"] += 1
            else:
                tool_stats[tool.name]["failed"] += 1
        
        if num_passed == num_tools:
            total_passed += 1
    
    total_failed = total_evaluated - total_passed
    overall_accuracy = total_passed / total_evaluated * 100 if total_evaluated > 0 else 0
    
    print("\n" + "="*60)
    print("🦀 EVALUATION SUMMARY (FROM EXISTING RESULTS)")
    print("="*60)
    
    print(f"📊 Overall Results:")
    print(f"   Total evaluated: {total_evaluated}")
    print(f"   All tests passed: {total_passed} ({overall_accuracy:.1f}%)")
    print(f"   Some tests failed: {total_failed} ({100-overall_accuracy:.1f}%)")
    
    print(f"\n📋 Individual Test Results:")
    for tool in tools:
        passed = tool_stats[tool.name]["passed"]
        failed = tool_stats[tool.name]["failed"]
        tool_accuracy = passed / total_evaluated * 100 if total_evaluated > 0 else 0
        print(f"   {tool.name}: {passed}/{total_evaluated} passed ({tool_accuracy:.1f}%)")
    
    print("="*60)


def evaluate_solutions(df, tools, output_file, progress_bar=None, max_rows=-1, save_every=100):
    """
    Evaluates all solutions in the dataframe.
    
    Args:
        df: Pandas DataFrame with solutions to evaluate
        tools: List of RustTool objects to run on each solution
        output_file: Path to write the results to
        progress_bar: Optional progress bar object
        max_rows: Maximum number of rows to evaluate (-1 for all)
        save_every: Save results every N evaluations (default 100)
        
    Returns:
        DataFrame with added tool results columns
    """
    # Check if results already exist
    output_path = Path(output_file)
    if output_path.exists():
        print(f"📁 Found existing results at {output_file}")
        try:
            existing_results = load_dataframe(output_file)
            print(f"✅ Loaded {len(existing_results)} existing evaluation results")
            print_evaluation_summary(existing_results, tools)
            return existing_results
        except Exception as e:
            print(f"⚠️  Failed to load existing results: {e}")
            print("🔄 Proceeding with fresh evaluation...")
    
    results = []

    total_passed = 0
    total_failed = 0
    num_rows = len(df) if max_rows < 0 else max_rows
    
    # Track per-tool statistics
    tool_stats = {tool.name: {"passed": 0, "failed": 0} for tool in tools}
    
    progress_context = progress_bar(total=num_rows) if progress_bar else None
    bar = progress_context.__enter__() if progress_context else None
    
    try:
        for row_count, (idx, row) in enumerate(df.iterrows()):
            if max_rows > 0 and row_count >= max_rows:
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
                    tool_stats[tool.name]["passed"] += 1
                else:
                    tool_stats[tool.name]["failed"] += 1
                    
            all_passed = num_passed == num_tools
            print(f"Row {idx}: {num_passed}/{num_tools} passed")
            if all_passed:
                total_passed += 1
            else:
                total_failed += 1
            print(f"Total passed: {total_passed}, Total failed: {total_failed}")
            
            # print percentage
            accuracy = total_passed / (row_count + 1) * 100
            percent_passed_str = f"Percentage passed {total_passed}/{row_count + 1} = {accuracy:.1f}%"
            print(percent_passed_str)
            
            if bar:
                bar.update(1)
                if hasattr(bar, 'set_description'):
                    bar.set_description(percent_passed_str)

            if save_every > 0 and (row_count + 1) % save_every == 0:
                results_df = pd.DataFrame(results).set_index("idx")
                save_dataframe(results_df, output_file)

        # Convert results to dataframe and merge with original
        results_df = pd.DataFrame(results).set_index("idx")
        
        # Save final results to disk
        save_dataframe(results_df, output_file)
        
        # Print final comprehensive statistics
        print("\n" + "="*60)
        print("🦀 EVALUATION SUMMARY")
        print("="*60)
        
        total_evaluated = len(results)
        overall_accuracy = total_passed / total_evaluated * 100 if total_evaluated > 0 else 0
        
        print(f"📊 Overall Results:")
        print(f"   Total evaluated: {total_evaluated}")
        print(f"   All tests passed: {total_passed} ({overall_accuracy:.1f}%)")
        print(f"   Some tests failed: {total_failed} ({100-overall_accuracy:.1f}%)")
        
        print(f"\n📋 Individual Test Results:")
        for tool in tools:
            passed = tool_stats[tool.name]["passed"]
            failed = tool_stats[tool.name]["failed"]
            tool_accuracy = passed / total_evaluated * 100 if total_evaluated > 0 else 0
            print(f"   {tool.name}: {passed}/{total_evaluated} passed ({tool_accuracy:.1f}%)")
        
        print("="*60)
        
        return results_df
    
    finally:
        if progress_context:
            progress_context.__exit__(None, None, None)
