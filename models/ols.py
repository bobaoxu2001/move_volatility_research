"""
OLS regression models for volatility prediction.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from typing import Dict, Tuple, Optional
from ..config import CONFIG


def fit_ols_regression(y: pd.Series, X: pd.DataFrame, 
                       cov_type: str = 'HC1') -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS regression with robust standard errors.
    
    Parameters
    ----------
    y : pd.Series
        Dependent variable
    X : pd.DataFrame
        Independent variables (without constant)
    cov_type : str
        Covariance type for robust SE
        
    Returns
    -------
    RegressionResultsWrapper
        Fitted model results
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(cov_type=cov_type)
    return model


def fit_predictive_regression(df: pd.DataFrame, 
                              signal_col: str = 'move_zscore',
                              target_col: str = 'spread_vol_zscore') -> Dict:
    """
    Fit predictive regression: signal(t) -> target(t+1).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with signal and target columns
    signal_col : str
        Signal column name
    target_col : str
        Target column name
        
    Returns
    -------
    dict
        Dictionary with model results and statistics
    """
    y = df[target_col].shift(-1)  # Forward target
    x = df[signal_col]
    
    reg_df = pd.DataFrame({'y': y, 'x': x}).dropna()
    model = fit_ols_regression(reg_df['y'], reg_df[['x']])
    
    return {
        'model': model,
        'beta': model.params.iloc[1],
        't_stat': model.tvalues.iloc[1],
        'p_value': model.pvalues.iloc[1],
        'r_squared': model.rsquared,
        'n_obs': len(reg_df),
    }


def fit_diff_regression(df: pd.DataFrame,
                        signal_col: str = 'move_zscore',
                        target_col: str = 'spread_vol_zscore') -> Dict:
    """
    Fit regression on first differences (robustness check).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with signal and target columns
    signal_col : str
        Signal column name
    target_col : str
        Target column name
        
    Returns
    -------
    dict
        Dictionary with model results
    """
    y_diff = df[target_col].diff().shift(-1)
    x_diff = df[signal_col].diff()
    
    reg_df = pd.DataFrame({'y': y_diff, 'x': x_diff}).dropna()
    model = fit_ols_regression(reg_df['y'], reg_df[['x']])
    
    return {
        'model': model,
        'beta': model.params.iloc[1],
        'p_value': model.pvalues.iloc[1],
        'r_squared': model.rsquared,
    }


def fit_nonlinear_regression(df: pd.DataFrame) -> Dict:
    """
    Fit quadratic and interaction models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with all required columns
        
    Returns
    -------
    dict
        Dictionary with quadratic and interaction model results
    """
    results = {}
    
    # Quadratic model
    y = df['spread_vol_zscore'].shift(-1)
    X_quad = df[['move_zscore', 'move_zscore_sq']]
    quad_df = pd.DataFrame({'y': y}).join(X_quad).dropna()
    
    quad_model = fit_ols_regression(quad_df['y'], quad_df[['move_zscore', 'move_zscore_sq']])
    results['quadratic'] = {
        'model': quad_model,
        'beta_linear': quad_model.params['move_zscore'],
        'beta_quadratic': quad_model.params['move_zscore_sq'],
        'p_linear': quad_model.pvalues['move_zscore'],
        'p_quadratic': quad_model.pvalues['move_zscore_sq'],
        'r_squared': quad_model.rsquared,
    }
    
    # Interaction model
    X_int = df[['move_zscore', 'vix_zscore', 'move_vix_interaction']]
    int_df = pd.DataFrame({'y': y}).join(X_int).dropna()
    
    int_model = fit_ols_regression(int_df['y'], int_df[['move_zscore', 'vix_zscore', 'move_vix_interaction']])
    results['interaction'] = {
        'model': int_model,
        'beta_move': int_model.params['move_zscore'],
        'beta_vix': int_model.params['vix_zscore'],
        'beta_interaction': int_model.params['move_vix_interaction'],
        'p_interaction': int_model.pvalues['move_vix_interaction'],
        'r_squared': int_model.rsquared,
    }
    
    return results


def fit_rolling_regression(df: pd.DataFrame, window: int = 60,
                           signal_col: str = 'move_zscore',
                           target_col: str = 'spread_vol_zscore') -> pd.DataFrame:
    """
    Fit rolling OLS regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    window : int
        Rolling window size
    signal_col : str
        Signal column
    target_col : str
        Target column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling beta, R², and p-values
    """
    y = df[target_col].shift(-1)
    x = df[signal_col]
    
    reg_df = pd.DataFrame({'y': y, 'x': x}).dropna()
    X = sm.add_constant(reg_df['x'])
    
    rolling_model = RollingOLS(reg_df['y'], X, window=window)
    rolling_results = rolling_model.fit()
    
    results = pd.DataFrame(index=reg_df.index)
    
    if hasattr(rolling_results.params, 'iloc'):
        results['beta'] = rolling_results.params.iloc[:, 1]
    else:
        results['beta'] = rolling_results.params[:, 1]
    
    results['r_squared'] = rolling_results.rsquared
    
    if hasattr(rolling_results.pvalues, 'iloc'):
        results['pvalue'] = rolling_results.pvalues.iloc[:, 1]
    else:
        results['pvalue'] = rolling_results.pvalues[:, 1]
    
    return results
