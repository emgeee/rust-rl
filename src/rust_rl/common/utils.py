"""
Shared utility functions for the rust-rl project
"""

import pandas as pd
from pathlib import Path
from typing import Any, Optional, Union
from tqdm import tqdm


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, output_path: Union[str, Path], format: str = "parquet") -> None:
    """
    Save DataFrame to file with automatic directory creation.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        format: Format to save in ("parquet", "csv", "json")
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    if format == "parquet":
        df.to_parquet(output_path)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(input_path: Union[str, Path], format: Optional[str] = None) -> pd.DataFrame:
    """
    Load DataFrame from file with automatic format detection.
    
    Args:
        input_path: Input file path
        format: Format to load ("parquet", "csv", "json"). Auto-detected if None.
        
    Returns:
        Loaded DataFrame
    """
    input_path = Path(input_path)
    
    if format is None:
        # Auto-detect format from extension
        format = input_path.suffix.lower().lstrip('.')
        if format == "pq":
            format = "parquet"
    
    if format == "parquet":
        return pd.read_parquet(input_path)
    elif format == "csv":
        return pd.read_csv(input_path)
    elif format == "json":
        return pd.read_json(input_path, orient="records")
    else:
        raise ValueError(f"Unsupported format: {format}")


class ProgressTracker:
    """
    Unified progress tracking utility for consistent progress reporting.
    """
    
    def __init__(self, total: int, desc: str = "Processing", unit: str = "item"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            desc: Description for progress bar
            unit: Unit name for progress display
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.pbar = None
    
    def __enter__(self):
        """Start progress tracking context."""
        self.pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End progress tracking context."""
        if self.pbar:
            self.pbar.close()
    
    def update(self, n: int = 1, title: Optional[str] = None):
        """
        Update progress.
        
        Args:
            n: Number of items completed
            title: Optional title update for progress bar
        """
        if self.pbar:
            self.pbar.update(n)
            if title:
                self.pbar.set_description(f"{self.desc}: {title}")
        self.current += n
    
    def set_postfix(self, **kwargs):
        """Set postfix text for progress bar."""
        if self.pbar:
            self.pbar.set_postfix(**kwargs)