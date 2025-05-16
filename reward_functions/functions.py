from .utils import (
    RustTool,
    extract_rust_code,
    extract_test_code,
    response_contains_more_than_non_empty_line,
    response_contains_asserts,
    response_contains_one_test_block,
    response_contains_one_code_block,
    setup_and_test_rust_project,
)

def non_empty_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if response contains more than a minimum number of non-empty lines
    
    Returns:
        1.0 if the code has at least 3 non-empty, non-comment lines, 0.0 otherwise
    """
    contents = [completion[0]["content"] for completion in completions]
    return [response_contains_more_than_non_empty_line(c) for c in contents]

def tests_have_asserts_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if test code contains assert statements
    
    Returns:
        1.0 if there are 4+ unique asserts
        0.25 * (number of asserts) for fewer asserts
        0.0 if no test code or no asserts
    """
    contents = [completion[0]["content"] for completion in completions]
    return [response_contains_asserts(c) for c in contents]

def test_block_count_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if response contains a test module
    
    Returns:
        0.5 if a test module is present, 0.0 otherwise
    """
    contents = [completion[0]["content"] for completion in completions]
    return [response_contains_one_test_block(c) for c in contents]

def code_block_count_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if response contains a Rust code block with a function
    
    Returns:
        0.5 if a valid Rust code block with fn is present, 0.0 otherwise
    """
    contents = [completion[0]["content"] for completion in completions]
    return [response_contains_one_code_block(c) for c in contents]

def cargo_build_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if code compiles with cargo build
    
    Returns:
        1.0 if cargo build passes, 0.0 otherwise
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_answers = [extract_rust_code(r) for r in responses]
    results = []
    for i, answer in enumerate(extracted_answers):
        data = {"rust_code": answer}
        tools = [RustTool("build")]
        cargo_results = setup_and_test_rust_project(data, tools)
        score = 1.0 if cargo_results["build_passed"] else 0.0
        results.append(score)
    return results

def cargo_clippy_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if code passes cargo clippy
    
    Returns:
        1.0 if cargo clippy passes, 0.0 otherwise
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_answers = [extract_rust_code(r) for r in responses]
    results = []
    for i, answer in enumerate(extracted_answers):
        data = {"rust_code": answer}
        tools = [RustTool("clippy")]
        cargo_results = setup_and_test_rust_project(data, tools)
        score = 1.0 if cargo_results["clippy_passed"] else 0.0
        results.append(score)
    return results

def cargo_test_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Reward function that checks if code passes cargo test
    
    Returns:
        2.0 if cargo test passes, 0.0 otherwise
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_codes = [extract_rust_code(r) for r in responses]
    extracted_tests = [extract_test_code(c) for c in extracted_codes]
    results = []
    for i, answer in enumerate(extracted_codes):
        score = 0.0
        if extracted_tests[i]:
            data = {"rust_code": answer}
            tools = [RustTool("test")]
            cargo_results = setup_and_test_rust_project(data, tools)
            score = 2.0 if cargo_results["test_passed"] else 0.0
        results.append(score)
    return results

def test_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Simple reward function that always returns 1.0 (for testing)
    """
    return [1.0]