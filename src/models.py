"""
Model implementations for volatility prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from .config import MODEL_CONFIG


@dataclass
class ModelResult:
    """Container for model results."""
    name: str
    params: Dict[str, float]
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    model_object: Optional[Any] = None


# =============================================================================
# OLS REGRESSION MODELS
# =============================================================================

def fit_ols_regression(
    X: pd.DataFrame,
    y: pd.Series,
    add_constant: bool = True,
    robust_cov: bool = True
) -> ModelResult:
    """
    Fit OLS regression model.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    add_constant : bool
        Whether to add constant term
    robust_cov : bool
        Whether to use heteroskedasticity-robust standard errors
        
    Returns
    -------
    ModelResult
        Fitted model results
    """
    X_fit = sm.add_constant(X) if add_constant else X
    cov_type = 'HC1' if robust_cov else 'nonrobust'
    
    model = sm.OLS(y, X_fit).fit(cov_type=cov_type)
    
    params = dict(model.params)
    metrics = {
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
    }
    
    # Add coefficient-specific metrics
    for col in X.columns:
        metrics[f'beta_{col}'] = model.params.get(col, model.params.iloc[1] if len(model.params) > 1 else np.nan)
        metrics[f'pvalue_{col}'] = model.pvalues.get(col, model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan)
        metrics[f'tstat_{col}'] = model.tvalues.get(col, model.tvalues.iloc[1] if len(model.tvalues) > 1 else np.nan)
    
    return ModelResult(
        name='OLS',
        params=params,
        metrics=metrics,
        model_object=model
    )


def fit_rolling_ols(
    X: pd.Series,
    y: pd.Series,
    window: int = None
) -> pd.DataFrame:
    """
    Fit rolling OLS regression.
    
    Parameters
    ----------
    X : pd.Series
        Feature series
    y : pd.Series
        Target series
    window : int, optional
        Rolling window size (default from config)
        
    Returns
    -------
    pd.DataFrame
        Rolling coefficients, R², and p-values
    """
    window = window or MODEL_CONFIG['regression_window']
    
    # Align data
    df = pd.DataFrame({'y': y, 'x': X}).dropna()
    X_roll = sm.add_constant(df['x'])
    
    rolling_model = RollingOLS(df['y'], X_roll, window=window)
    rolling_results = rolling_model.fit()
    
    results_df = pd.DataFrame(index=df.index)
    
    if hasattr(rolling_results.params, 'iloc'):
        results_df['beta'] = rolling_results.params.iloc[:, 1]
    else:
        results_df['beta'] = rolling_results.params[:, 1]
    
    results_df['r_squared'] = rolling_results.rsquared
    
    if hasattr(rolling_results.pvalues, 'iloc'):
        results_df['pvalue'] = rolling_results.pvalues.iloc[:, 1]
    else:
        results_df['pvalue'] = rolling_results.pvalues[:, 1]
    
    return results_df


# =============================================================================
# REGULARIZED REGRESSION MODELS
# =============================================================================

def fit_lasso_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray = None,
    cv_splits: int = None,
    scale_features: bool = True
) -> Tuple[ModelResult, StandardScaler]:
    """
    Fit LASSO regression with cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    alphas : np.ndarray, optional
        Alpha values to try
    cv_splits : int, optional
        Number of CV splits
    scale_features : bool
        Whether to standardize features
        
    Returns
    -------
    tuple of (ModelResult, StandardScaler)
        Fitted model results and scaler
    """
    alphas = alphas if alphas is not None else MODEL_CONFIG['lasso_alphas']
    cv_splits = cv_splits or MODEL_CONFIG['cv_splits']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if scale_features else X
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)
    model.fit(X_scaled, y)
    
    metrics = {
        'alpha': model.alpha_,
        'r_squared_cv': model.score(X_scaled, y),
        'n_nonzero': np.sum(np.abs(model.coef_) > 1e-6),
    }
    
    params = {'intercept': model.intercept_}
    for i, coef in enumerate(model.coef_):
        params[f'coef_{i}'] = coef
    
    return ModelResult(
        name='LASSO',
        params=params,
        metrics=metrics,
        model_object=model
    ), scaler


def fit_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray = None,
    cv_splits: int = None,
    scale_features: bool = True
) -> Tuple[ModelResult, StandardScaler]:
    """
    Fit Ridge regression with cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    alphas : np.ndarray, optional
        Alpha values to try
    cv_splits : int, optional
        Number of CV splits
    scale_features : bool
        Whether to standardize features
        
    Returns
    -------
    tuple of (ModelResult, StandardScaler)
        Fitted model results and scaler
    """
    alphas = alphas if alphas is not None else MODEL_CONFIG['ridge_alphas']
    cv_splits = cv_splits or MODEL_CONFIG['cv_splits']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) if scale_features else X
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model = RidgeCV(alphas=alphas, cv=tscv)
    model.fit(X_scaled, y)
    
    metrics = {
        'alpha': model.alpha_,
        'r_squared_cv': model.score(X_scaled, y),
    }
    
    params = {'intercept': model.intercept_}
    for i, coef in enumerate(model.coef_):
        params[f'coef_{i}'] = coef
    
    return ModelResult(
        name='Ridge',
        params=params,
        metrics=metrics,
        model_object=model
    ), scaler


# =============================================================================
# TIME SERIES MODELS
# =============================================================================

def fit_arima(
    y: pd.Series,
    order: Tuple[int, int, int] = None,
    exog: pd.Series = None
) -> ModelResult:
    """
    Fit ARIMA or ARIMA-X model.
    
    Parameters
    ----------
    y : pd.Series
        Target time series
    order : tuple of (p, d, q), optional
        ARIMA order (default from config)
    exog : pd.Series, optional
        Exogenous variable (for ARIMA-X)
        
    Returns
    -------
    ModelResult
        Fitted model results
    """
    order = order or MODEL_CONFIG['arima_order']
    model_name = 'ARIMA-X' if exog is not None else 'ARIMA'
    
    try:
        model = ARIMA(y, exog=exog, order=order)
        fitted = model.fit()
        
        metrics = {
            'aic': fitted.aic,
            'bic': fitted.bic,
        }
        
        params = dict(fitted.params)
        
        return ModelResult(
            name=model_name,
            params=params,
            metrics=metrics,
            model_object=fitted
        )
    except Exception as e:
        print(f"  ARIMA fitting failed: {e}")
        return ModelResult(
            name=model_name,
            params={},
            metrics={'error': str(e)}
        )


def fit_garch(
    y: pd.Series,
    p: int = None,
    q: int = None
) -> ModelResult:
    """
    Fit GARCH model for volatility clustering.
    
    Parameters
    ----------
    y : pd.Series
        Return series
    p : int, optional
        GARCH p order
    q : int, optional
        GARCH q order
        
    Returns
    -------
    ModelResult
        Fitted model results
    """
    p = p or MODEL_CONFIG['garch_order'][0]
    q = q or MODEL_CONFIG['garch_order'][1]
    
    try:
        from arch import arch_model
        
        # Scale for numerical stability
        returns = y.diff().dropna() * 100
        
        model = arch_model(returns, vol='Garch', p=p, q=q)
        fitted = model.fit(disp='off')
        
        params = dict(fitted.params)
        metrics = {
            'aic': fitted.aic,
            'persistence': fitted.params['alpha[1]'] + fitted.params['beta[1]'],
        }
        
        return ModelResult(
            name='GARCH',
            params=params,
            metrics=metrics,
            model_object=fitted
        )
    except ImportError:
        return ModelResult(
            name='GARCH',
            params={},
            metrics={'error': 'arch package not available'}
        )
    except Exception as e:
        return ModelResult(
            name='GARCH',
            params={},
            metrics={'error': str(e)}
        )


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def granger_causality_test(
    y: pd.Series,
    x: pd.Series,
    lags: List[int] = [1, 2, 3, 5]
) -> Dict[str, float]:
    """
    Perform Granger causality test.
    
    Parameters
    ----------
    y : pd.Series
        Dependent variable
    x : pd.Series
        Independent variable (potential cause)
    lags : list of int
        Lags to test
        
    Returns
    -------
    dict
        Test results for each lag
    """
    df = pd.DataFrame({'y': y, 'x': x}).dropna()
    
    results = {}
    min_pval = 1.0
    
    for lag in lags:
        try:
            gc_test = grangercausalitytests(df[['y', 'x']], maxlag=lag, verbose=False)
            f_stat = gc_test[lag][0]['ssr_ftest'][0]
            p_val = gc_test[lag][0]['ssr_ftest'][1]
            results[f'lag_{lag}_f'] = f_stat
            results[f'lag_{lag}_p'] = p_val
            min_pval = min(min_pval, p_val)
        except Exception as e:
            results[f'lag_{lag}_error'] = str(e)
    
    results['min_pvalue'] = min_pval
    
    return results


def compute_information_coefficient(
    signal: pd.Series,
    target: pd.Series,
    forward: bool = True
) -> float:
    """
    Compute Spearman IC between signal and target.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    target : pd.Series
        Target values
    forward : bool
        Whether target is forward-looking
        
    Returns
    -------
    float
        Spearman correlation (IC)
    """
    if forward:
        target = target.shift(-1)
    
    df = pd.DataFrame({'signal': signal, 'target': target}).dropna()
    return spearmanr(df['signal'], df['target']).correlation


def compute_rolling_ic(
    signal: pd.Series,
    target: pd.Series,
    window: int = 60,
    forward: bool = True
) -> pd.Series:
    """
    Compute rolling Spearman IC.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values
    target : pd.Series
        Target values
    window : int
        Rolling window size
    forward : bool
        Whether target is forward-looking
        
    Returns
    -------
    pd.Series
        Rolling IC values
    """
    if forward:
        target = target.shift(-1)
    
    return signal.rank().rolling(window).corr(target.rank())
