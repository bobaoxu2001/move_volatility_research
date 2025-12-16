"""
Configuration parameters for MOVE Volatility Research Project.
"""

import numpy as np
from datetime import datetime

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
DATA_CONFIG = {
    'start_date': '2018-01-01',
    'end_date': '2025-12-15',
    'tickers': {
        'move': '^MOVE',
        'vix': '^VIX',
        'vxn': '^VXN',
        'ief': 'IEF',
        'tlt': 'TLT',
    },
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
FEATURE_CONFIG = {
    'volatility_window': 21,      # ~1 month for volatility calculation
    'zscore_window': 60,          # ~3 months for z-score normalization
    'signal_short_window': 10,    # Short-term MA for signal
    'signal_long_window': 60,     # Long-term MA for signal
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    'regression_window': 60,      # Rolling regression window
    'high_vol_threshold': 1.0,    # Z-score > 1 = High vol regime
    'low_vol_threshold': -1.0,    # Z-score < -1 = Low vol regime
    
    # Regularization
    'lasso_alphas': np.logspace(-4, 1, 50),
    'ridge_alphas': np.logspace(-4, 4, 50),
    'cv_splits': 5,
    
    # Time Series
    'arima_order': (2, 0, 2),     # (p, d, q)
    'garch_order': (1, 1),        # (p, q)
}

# =============================================================================
# BACKTESTING CONFIGURATION
# =============================================================================
BACKTEST_CONFIG = {
    'test_ratio': 0.3,
    'purge_gap': 60,              # Days to purge between train/test
    'min_train_samples': 252,     # Minimum training observations
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_CONFIG = {
    'figure_dpi': 150,
    'figure_format': 'png',
    'output_dir': 'reports/figures',
    'data_dir': 'data',
}

# =============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# =============================================================================
SEEDS = {
    'move': 42,
    'vix': 43,
    'vxn': 44,
    'spread': 45,
}
