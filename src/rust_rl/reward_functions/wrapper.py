"""
Wrapper for reward functions to handle errors gracefully.
"""

import functools
import json
import os
import traceback
from datetime import datetime
from typing import Any, Callable, List

def safe_reward_func(func: Callable) -> Callable:
    """
    Wrap a reward function to handle errors gracefully.
    
    Args:
        func: The reward function to wrap
        
    Returns:
        A wrapped function that catches exceptions and returns a default score
    """
    @functools.wraps(func)
    def wrapper(prompts, completions, **kwargs) -> List[float]:
        try:
            # Execute the function
            return func(prompts, completions, **kwargs)
        except Exception as e:
            # Log the error
            print(f"Error in reward function {func.__name__}: {e}")
            traceback.print_exc()
            
            # Return default scores (0.0 for each completion)
            return [0.0] * len(completions)
    
    return wrapper


def create_reward_logger(log_dir: str):
    """
    Create a reward logger that saves results to files.
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        A decorator factory that creates decorators for logging reward functions
    """
    os.makedirs(log_dir, exist_ok=True)
    
    def create_decorator(func_name: str):
        """Create a decorator for a specific reward function."""
        log_path = os.path.join(log_dir, f"{func_name}_rewards.jsonl")
        
        def decorator(func: Callable) -> Callable:
            """Decorator that logs the results of a reward function."""
            @functools.wraps(func)
            def wrapper(prompts, completions, **kwargs) -> List[float]:
                # Call the original function
                scores = func(prompts, completions, **kwargs)
                
                # Log the results
                timestamp = datetime.now().isoformat()
                
                # Create logs for each score
                try:
                    for i, score in enumerate(scores):
                        # Extract data from kwargs for logging
                        task_id = kwargs.get('task_id', [f"task_{i}"])[i] if 'task_id' in kwargs else f"task_{i}"
                        
                        completion_content = "unknown"
                        if completions and i < len(completions) and completions[i] and len(completions[i]) > 0:
                            if isinstance(completions[i][0], dict) and "content" in completions[i][0]:
                                completion_content = completions[i][0]["content"]
                        
                        prompt_content = "unknown"
                        if prompts and i < len(prompts):
                            if isinstance(prompts[i], str):
                                prompt_content = prompts[i]
                            elif isinstance(prompts[i], list) and prompts[i] and isinstance(prompts[i][-1], dict) and "content" in prompts[i][-1]:
                                prompt_content = prompts[i][-1]["content"]
                        
                        # Create log entry
                        log_entry = {
                            "timestamp": timestamp,
                            "function": func.__name__,
                            "function_alias": func_name,
                            "score": float(score),
                            "task_id": task_id,
                            "prompt": prompt_content,
                            "completion": completion_content,
                        }
                        
                        # Write to log file
                        with open(log_path, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                except Exception as e:
                    print(f"Error logging reward function results: {e}")
                    traceback.print_exc()
                
                return scores
            
            return wrapper
        
        return decorator
    
    return create_decorator
