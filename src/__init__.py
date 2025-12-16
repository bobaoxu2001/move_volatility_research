"""
MOVE Volatility Research Project

A buy-side quantitative research study analyzing the predictive power of
Treasury implied volatility (MOVE Index) for cross-asset volatility forecasting.

Author: Allen Xu
"""

from .config import DATA_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, BACKTEST_CONFIG
from .data_loader import get_data, load_all_data, preprocess_data
from .features import engineer_all_features, get_feature_matrix
from .models import (
    fit_ols_regression,
    fit_rolling_ols,
    fit_lasso_cv,
    fit_ridge_cv,
    fit_arima,
    fit_garch,
    granger_causality_test,
    compute_information_coefficient,
    compute_rolling_ic,
)
from .evaluation import (
    train_test_split_ts,
    compute_hit_rate,
    compute_regime_metrics,
    run_oos_backtest,
    regime_separation_test,
)
from .visualization import (
    plot_time_series,
    plot_zscore_with_regimes,
    plot_scatter_with_fit,
    plot_rolling_ic,
    plot_regime_comparison,
    plot_model_comparison,
    plot_feature_importance,
    plot_key_chart,
    create_eda_dashboard,
)

__version__ = '1.0.0'
__author__ = 'Allen Xu'
