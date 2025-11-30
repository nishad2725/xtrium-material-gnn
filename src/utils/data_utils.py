"""Data loading and preprocessing utilities"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def find_repo_root(start: Optional[str] = None, 
                   markers: Tuple[str, ...] = ('.git', 'environment.yml', 'requirements.txt', 'README.md')) -> Optional[str]:
    """
    Walk upward from `start` (or cwd) until a marker file/dir is found.
    
    Args:
        start: Starting directory path. If None, uses current working directory.
        markers: Tuple of marker file/directory names to look for.
    
    Returns:
        Absolute path to repository root or None if not found.
    """
    if start is None:
        start = os.getcwd()
    cur = os.path.abspath(start)
    while True:
        for m in markers:
            if os.path.exists(os.path.join(cur, m)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def load_materials_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load materials dataset from CSV file.
    
    Args:
        data_path: Path to materials.csv file. If None, auto-detects from repo root.
    
    Returns:
        DataFrame containing materials data.
    
    Raises:
        FileNotFoundError: If materials.csv is not found.
    """
    if data_path is None:
        repo_root = find_repo_root()
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        data_path = os.path.join(repo_root, 'data', 'raw', 'materials.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"materials.csv not found at {data_path}. "
            f"Checked repo root: {repo_root if 'repo_root' in locals() else 'unknown'}"
        )
    
    df = pd.read_csv(data_path)
    return df


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Categorize columns into numeric, categorical, and text columns.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        Dictionary with keys 'numeric', 'categorical', 'text', 'property_value'.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Further categorize text vs categorical
    text_cols = [c for c in categorical_cols if df[c].nunique() > 20 and df[c].notna().sum() > 0]
    cat_cols = [c for c in categorical_cols if df[c].nunique() <= 20 and df[c].notna().sum() > 0]
    
    # Extract property value columns
    property_value_cols = [col for col in df.columns 
                          if col.endswith('_Value') and col in numeric_cols]
    
    return {
        'numeric': numeric_cols,
        'categorical': cat_cols,
        'text': text_cols,
        'property_value': property_value_cols
    }

