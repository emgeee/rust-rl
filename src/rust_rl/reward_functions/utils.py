import re
import subprocess
from pathlib import Path
import shutil
from uuid import uuid4

from ..common.utils import ensure_dir

class RustTool:
    """
    Tool for running Rust cargo commands on a project
    """
    def __init__(self, name):
        self.name = name

    def run(self, results, project_dir):
        try:
            result = subprocess.run(
                ["cargo", self.name, "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            results[f'{self.name}_passed'] = result.returncode == 0
            results[f'{self.name}_stderr'] = str(result.stderr)
        except Exception as e:
            results[f'{self.name}_passed'] = False
            results[f'{self.name}_stderr'] = f"{e}"
        return results

def extract_regex(text: str, pattern: str) -> str | None:
    """
    Extract content from text using a regex pattern with DOTALL flag
    """
    # Use re.DOTALL to make '.' match newlines as well
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)
    else:
        return None

def extract_code_regex():
    """
    Returns regex pattern for extracting Rust code blocks
    """
    return r"```rust\n(.*?)\n```"

def extract_test_regex():
    """
    Returns regex pattern for extracting Rust test modules
    """
    return r"(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\})"

def extract_rust_code(response: str) -> str:
    """
    Extract Rust code from a markdown response
    """
    code = extract_regex(response, extract_code_regex())
    if code:
        return code
    else:
        return response

def extract_test_code(response: str) -> str:
    """
    Extract Rust test module from code
    """
    return extract_regex(response, extract_test_regex())

def response_contains_one_code_block(response: str) -> float:
    """
    Check if response contains a valid Rust code block with a function
    """
    # It has to have a ```rust``` block and a fn
    if extract_rust_code(response) and "fn " in response:
        return 0.5
    else:
        return 0.0

def response_contains_one_test_block(response: str) -> float:
    """
    Check if response contains a test module
    """
    if extract_test_code(response):
        return 0.5
    else:
        return 0.0

def response_contains_asserts(response: str) -> float:
    """
    Check if test code contains assert statements and give a score based on the count
    """
    test_code = extract_test_code(response)
    if not test_code:
        return 0.0

    unique_asserts = set()
    for line in test_code.split("\n"):
        line = line.strip()
        if line.startswith("assert!(") or line.startswith("assert_eq!("):
            unique_asserts.add(line)
    if len(unique_asserts) >= 4:
        return 1.0
    return 0.25 * len(unique_asserts)

def response_contains_more_than_non_empty_line(response: str) -> float:
    """
    Check if generated code has enough non-empty, non-comment lines
    """
    if not (
        response_contains_one_code_block(response)
        and response_contains_one_test_block(response)
    ):
        return 0.0

    code = extract_rust_code(response)
    num_non_empty = 0
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("//"):
            continue
        if len(line) < 2:
            continue
        num_non_empty += 1
    return 1.0 if num_non_empty >= 3 else 0.0

def template_rs_file():
    """
    Rust file template for testing
    """
    return """
#![allow(dead_code)]
// {code}

// Need basic main function for the code to compile
fn main() {
  println!("Hello World");
}
"""

def cargo_toml_file():
    """
    Cargo.toml template for Rust projects
    """
    return """
[package]
name = "rust-program"
version = "0.1.0"
edition = "2021"

[dependencies]
"""

def setup_and_test_rust_project(row, tools):
    """
    Sets up a Rust project from template and runs tests for a single row of data
    
    Args:
        row: Dictionary containing Rust code in either 'rust_code' or 'response' field
        tools: List of RustTool objects to run on the project
        
    Returns:
        Dictionary with test results from running tools
    """
    # Create temporary project directory
    project_dir = Path("outputs") / Path("tests") / Path(f"rust_project_{uuid4()}")
    project_dir_src = project_dir / Path("src")

    # mkdirs if they don't exist
    ensure_dir(project_dir_src)

    # Read template
    template = template_rs_file()

    # Replace placeholders - handle both 'rust_code' and 'response' field names
    code_content = row.get('rust_code') or row.get('response', '')
    rust_code = extract_rust_code(code_content)
    template = template.replace("// {code}", rust_code)

    # Write the cargo project files
    main_rs_path = project_dir_src / Path("main.rs")
    with open(main_rs_path, "w") as f:
        f.write(template)

    cargo_file_path = project_dir / Path("Cargo.toml")
    with open(cargo_file_path, "w") as f:
        f.write(cargo_toml_file())

    results = {}
    for tool in tools:
        results = tool.run(results, project_dir)

    # Print debug information if needed
    # print("----> Tool Results")
    # for k,v in results.items():
    #     print("")
    #     print(k)
    #     print(v)
    # print("="*80)

    # Clean up
    shutil.rmtree(project_dir)

    return results
