"""
Data loading and processing module.
"""

from .loader import (
    fetch_data_with_fallback,
    create_synthetic_volatility_data,
    load_all_data,
)
from .processor import (
    compute_zscore,
    compute_percentile,
    compute_spread_volatility,
    process_data,
    train_test_split,
)

__all__ = [
    'fetch_data_with_fallback',
    'create_synthetic_volatility_data',
    'load_all_data',
    'compute_zscore',
    'compute_percentile',
    'compute_spread_volatility',
    'process_data',
    'train_test_split',
]
