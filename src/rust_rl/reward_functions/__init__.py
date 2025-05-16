"""
Reward functions module for evaluating Rust code quality in RL training.
"""

from .functions import (
    non_empty_reward_func,
    tests_have_asserts_reward_func,
    test_block_count_reward_func,
    code_block_count_reward_func,
    cargo_build_reward_func,
    cargo_clippy_reward_func,
    cargo_test_reward_func,
    test_reward_func,
)

from .wrapper import (
    safe_reward_func,
    create_reward_logger,
)

from .utils import (
    RustTool,
    extract_regex,
    extract_code_regex,
    extract_test_regex,
    extract_rust_code,
    extract_test_code,
    response_contains_one_code_block,
    response_contains_one_test_block,
    response_contains_asserts,
    response_contains_more_than_non_empty_line,
    template_rs_file,
    cargo_toml_file,
    setup_and_test_rust_project,
)

__all__ = [
    # Utility functions
    'RustTool',
    'extract_regex',
    'extract_code_regex',
    'extract_test_regex',
    'extract_rust_code',
    'extract_test_code',
    'response_contains_one_code_block',
    'response_contains_one_test_block',
    'response_contains_asserts',
    'response_contains_more_than_non_empty_line',
    'template_rs_file',
    'cargo_toml_file',
    'setup_and_test_rust_project',
    
    # Reward functions
    'non_empty_reward_func',
    'tests_have_asserts_reward_func',
    'test_block_count_reward_func',
    'code_block_count_reward_func',
    'cargo_build_reward_func',
    'cargo_clippy_reward_func',
    'cargo_test_reward_func',
    'test_reward_func',
    
    # Wrappers
    'safe_reward_func',
    'create_reward_logger',
]