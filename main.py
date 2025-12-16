"""
Main execution script for MOVE Volatility Research Project.

Usage:
    python main.py [--output-dir DIR] [--no-plots]

Author: Allen Xu
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    get_data, engineer_all_features, get_feature_matrix,
    fit_ols_regression, fit_rolling_ols, fit_lasso_cv, fit_ridge_cv,
    fit_arima, granger_causality_test, compute_rolling_ic,
    run_oos_backtest, compute_regime_metrics, regime_separation_test,
    create_eda_dashboard, plot_rolling_ic, plot_regime_comparison,
    plot_model_comparison, plot_feature_importance, plot_key_chart,
    MODEL_CONFIG,
)


def main():
    """Main execution function."""
    print("=" * 80)
    print("MOVE INDEX CROSS-ASSET VOLATILITY RESEARCH")
    print("Author: Allen Xu")
    print("=" * 80)
    
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    print("\n[1/6] Loading and processing data...")
    data = get_data()
    data = engineer_all_features(data)
    data.to_csv('data/processed_data.csv')
    
    # =========================================================================
    # 2. BASE MODELS
    # =========================================================================
    print("\n[2/6] Fitting base models...")
    
    y = data['spread_vol_zscore'].shift(-1)
    x = data['move_zscore']
    reg_df = pd.DataFrame({'y': y, 'x': x}).dropna()
    
    ols_result = fit_ols_regression(reg_df[['x']], reg_df['y'])
    gc_results = granger_causality_test(data['spread_vol_zscore'], data['move_zscore'])
    rolling_ic = compute_rolling_ic(data['signal_zscore'], data['spread_vol_zscore'])
    
    print(f"  OLS R²: {ols_result.metrics['r_squared']:.4f}")
    print(f"  Granger p: {gc_results['min_pvalue']:.4f}")
    print(f"  Mean IC: {rolling_ic.mean():.4f}")
    
    # =========================================================================
    # 3. REGIME ANALYSIS
    # =========================================================================
    print("\n[3/6] Analyzing regime effects...")
    regime_metrics = compute_regime_metrics(data, 'move_zscore', 'fwd_spread_vol', 'regime')
    print(regime_metrics.to_string(index=False))
    
    # =========================================================================
    # 4. REGULARIZED MODELS
    # =========================================================================
    print("\n[4/6] Fitting regularized models...")
    
    X, y_feat = get_feature_matrix(data, target_col='fwd_spread_vol')
    X_arr, y_arr = X.values, y_feat.values
    
    lasso_result, lasso_scaler = fit_lasso_cv(X_arr, y_arr)
    ridge_result, ridge_scaler = fit_ridge_cv(X_arr, y_arr)
    
    print(f"  LASSO R²: {lasso_result.metrics['r_squared_cv']:.4f}")
    print(f"  Ridge R²: {ridge_result.metrics['r_squared_cv']:.4f}")
    
    # =========================================================================
    # 5. BACKTESTING
    # =========================================================================
    print("\n[5/6] Running backtest...")
    backtest = run_oos_backtest(data)
    
    print(f"  IS R²: {backtest.is_r2:.4f}, OOS R²: {backtest.oos_r2:.4f}")
    print(f"  OOS IC: {backtest.oos_ic:.4f}, Hit Rate: {backtest.oos_hit_rate:.1%}")
    
    # =========================================================================
    # 6. VISUALIZATION
    # =========================================================================
    print("\n[6/6] Generating figures...")
    
    create_eda_dashboard(data, save_path='reports/figures/eda_dashboard.png')
    plot_rolling_ic(rolling_ic, save_path='reports/figures/rolling_ic.png')
    plot_regime_comparison(regime_metrics, save_path='reports/figures/regime_ic.png')
    
    metrics = {
        'mean_ic': rolling_ic.mean(),
        'oos_ic': backtest.oos_ic,
        'best_model': 'LASSO',
        'granger_p': f"{gc_results['min_pvalue']:.4f}"
    }
    plot_key_chart(data, metrics, save_path='reports/figures/key_chart.png')
    plt.close('all')
    
    print("\n" + "=" * 80)
    print("COMPLETE - See reports/figures/ for outputs")
    print("=" * 80)


if __name__ == '__main__':
    main()
