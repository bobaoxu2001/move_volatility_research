"""
Configuration parameters for MOVE Volatility Research Project.
"""

import numpy as np
from datetime import datetime

CONFIG = {
    # Data parameters
    'start_date': '2018-01-01',
    'end_date': '2025-12-15',
    
    # Rolling window parameters
    'volatility_window': 21,      # ~1 month for volatility calculation
    'zscore_window': 60,          # ~3 months for z-score normalization
    'regression_window': 60,      # Rolling regression window
    
    # Signal parameters
    'signal_short_window': 10,    # Short-term MA for signal
    'signal_long_window': 60,     # Long-term MA for signal
    'high_vol_threshold': 1.0,    # Z-score > 1 = High vol regime
    'low_vol_threshold': -1.0,    # Z-score < -1 = Low vol regime
    
    # Backtesting parameters
    'test_ratio': 0.3,
    'purge_gap': 60,              # Gap between train/test to prevent leakage
    'min_train_obs': 252,         # Minimum training observations
    
    # Regularization parameters
    'lasso_alphas': np.logspace(-4, 1, 50),
    'ridge_alphas': np.logspace(-4, 4, 50),
    'cv_splits': 5,
    
    # Time series model parameters
    'arima_order': (2, 0, 2),     # (p, d, q)
    'garch_order': (1, 1),        # (p, q)
    
    # Granger causality lags
    'granger_lags': [1, 2, 3, 5],
    
    # Output paths
    'output_dir': 'outputs/',
    'figure_dpi': 150,
}

# Ticker symbols
TICKERS = {
    'move': '^MOVE',
    'vix': '^VIX',
    'vxn': '^VXN',
    'ief': 'IEF',
    'tlt': 'TLT',
}

# Feature columns for regularized models
FEATURE_COLS = ['move_zscore', 'vix_zscore', 'move_zscore_sq', 'move_vix_interaction']
