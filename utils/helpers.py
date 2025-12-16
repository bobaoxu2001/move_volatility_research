"""
Utility functions for MOVE volatility research.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import json
from datetime import datetime


def format_pvalue(p: float) -> str:
    """Format p-value with significance stars."""
    if np.isnan(p):
        return "NA"
    elif p < 0.01:
        return f"{p:.4f}***"
    elif p < 0.05:
        return f"{p:.4f}**"
    elif p < 0.10:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


def format_percentage(x: float) -> str:
    """Format as percentage."""
    if np.isnan(x):
        return "NA"
    return f"{x:.1%}"


def format_number(x: float, decimals: int = 4) -> str:
    """Format number with specified decimals."""
    if np.isnan(x):
        return "NA"
    return f"{x:.{decimals}f}"


def safe_divide(a: float, b: float) -> float:
    """Safe division handling zero denominator."""
    if b and abs(b) > 1e-12:
        return a / b
    return np.nan


def print_section_header(title: str, char: str = "=", width: int = 80):
    """Print formatted section header."""
    print("\n" + char * width)
    print(title)
    print(char * width)


def print_results_table(data: Dict[str, Dict], columns: list = None):
    """Print formatted results table."""
    if not data:
        print("No data to display")
        return
    
    if columns is None:
        columns = list(list(data.values())[0].keys())
    
    # Header
    header = f"{'Model':<15}"
    for col in columns:
        header += f"{col:>12}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for name, values in data.items():
        row = f"{name:<15}"
        for col in columns:
            val = values.get(col, np.nan)
            if isinstance(val, float):
                row += f"{format_number(val):>12}"
            else:
                row += f"{str(val):>12}"
        print(row)


def generate_summary_dict(df: pd.DataFrame, 
                          ic_stats: Dict,
                          oos_metrics: Dict,
                          model_results: Dict,
                          granger_results: Dict) -> Dict[str, Any]:
    """
    Generate summary dictionary for reporting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed data
    ic_stats : dict
        IC statistics
    oos_metrics : dict
        OOS evaluation metrics
    model_results : dict
        Model comparison results
    granger_results : dict
        Granger causality results
        
    Returns
    -------
    dict
        Summary dictionary
    """
    return {
        'data': {
            'n_observations': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d'),
                'end': df.index.max().strftime('%Y-%m-%d'),
            },
            'high_vol_pct': (df['regime'] == 'High Vol').mean(),
        },
        'ic': ic_stats,
        'oos': oos_metrics,
        'models': {
            'best_model': model_results.get('best_model'),
            'ols_r2': model_results['models']['ols']['oos_r2'],
            'lasso_r2': model_results['models']['lasso']['oos_r2'],
            'ridge_r2': model_results['models']['ridge']['oos_r2'],
        },
        'granger': {
            'min_pvalue': granger_results.get('min_pvalue'),
            'significant': granger_results.get('significant'),
        },
        'generated_at': datetime.now().isoformat(),
    }


def save_results_json(results: Dict, filepath: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
