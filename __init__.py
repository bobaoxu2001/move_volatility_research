"""
MOVE Index Cross-Asset Volatility Research Project
===================================================

A buy-side quant research study examining whether Treasury Implied 
Volatility (MOVE Index) predicts:
1. Treasury term-spread volatility (TLT/IEF spread volatility proxy)
2. Technology equity implied volatility (VXN/VIX)

Author: Allen Xu
Style: Cubist / Point72 Buy-Side Workflow

Modules:
--------
- data: Data loading and processing
- models: OLS, regularized, and time series models
- analysis: Regime analysis, IC computation, Granger causality
- visualization: Chart generation
- utils: Helper functions

Usage:
------
    from move_volatility_research import run_full_analysis
    results = run_full_analysis()
"""

__version__ = '1.0.0'
__author__ = 'Allen Xu'

from .config import CONFIG, TICKERS, FEATURE_COLS
from .main import run_full_analysis

__all__ = [
    'CONFIG',
    'TICKERS', 
    'FEATURE_COLS',
    'run_full_analysis',
]
