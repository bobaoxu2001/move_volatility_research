"""
Regularized regression models (LASSO, Ridge) for volatility prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple
from ..config import CONFIG, FEATURE_COLS


def fit_lasso_cv(X: np.ndarray, y: np.ndarray, 
                 alphas: np.ndarray = None,
                 cv_splits: int = 5) -> Tuple[LassoCV, Dict]:
    """
    Fit LASSO with cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (should be standardized)
    y : np.ndarray
        Target array
    alphas : np.ndarray
        Alpha values to search
    cv_splits : int
        Number of CV splits
        
    Returns
    -------
    tuple
        (fitted model, results dict)
    """
    if alphas is None:
        alphas = CONFIG['lasso_alphas']
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    lasso_cv = LassoCV(alphas=alphas, cv=tscv, max_iter=10000)
    lasso_cv.fit(X, y)
    
    results = {
        'optimal_alpha': lasso_cv.alpha_,
        'r_squared_cv': lasso_cv.score(X, y),
        'coefficients': lasso_cv.coef_,
        'n_nonzero': np.sum(np.abs(lasso_cv.coef_) > 1e-6),
    }
    
    return lasso_cv, results


def fit_ridge_cv(X: np.ndarray, y: np.ndarray,
                 alphas: np.ndarray = None,
                 cv_splits: int = 5) -> Tuple[RidgeCV, Dict]:
    """
    Fit Ridge with cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (should be standardized)
    y : np.ndarray
        Target array
    alphas : np.ndarray
        Alpha values to search
    cv_splits : int
        Number of CV splits
        
    Returns
    -------
    tuple
        (fitted model, results dict)
    """
    if alphas is None:
        alphas = CONFIG['ridge_alphas']
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
    ridge_cv.fit(X, y)
    
    results = {
        'optimal_alpha': ridge_cv.alpha_,
        'r_squared_cv': ridge_cv.score(X, y),
        'coefficients': ridge_cv.coef_,
    }
    
    return ridge_cv, results


def compare_regularized_models(df: pd.DataFrame, 
                               feature_cols: List[str] = None,
                               target_col: str = 'spread_vol_zscore',
                               test_ratio: float = 0.3) -> Dict:
    """
    Compare OLS, LASSO, and Ridge models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_cols : list
        Feature column names
    target_col : str
        Target column name
    test_ratio : float
        Test set proportion
        
    Returns
    -------
    dict
        Comparison results for all models
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].shift(-1)
    
    full_df = pd.DataFrame({'y': y}).join(X).dropna()
    X_features = full_df[feature_cols].values
    y_target = full_df['y'].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Train-test split
    split_idx = int(len(X_scaled) * (1 - test_ratio))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_target[:split_idx], y_target[split_idx:]
    
    results = {
        'feature_cols': feature_cols,
        'scaler': scaler,
        'models': {},
    }
    
    # LASSO
    lasso_cv, lasso_results = fit_lasso_cv(X_scaled, y_target)
    lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    lasso_final.fit(X_train, y_train)
    lasso_pred = lasso_final.predict(X_test)
    
    results['models']['lasso'] = {
        'model': lasso_final,
        'optimal_alpha': lasso_cv.alpha_,
        'coefficients': dict(zip(feature_cols, lasso_cv.coef_)),
        'n_nonzero': lasso_results['n_nonzero'],
        'is_r2': lasso_final.score(X_train, y_train),
        'oos_r2': max(0, r2_score(y_test, lasso_pred)),
        'oos_rmse': np.sqrt(mean_squared_error(y_test, lasso_pred)),
    }
    
    # Ridge
    ridge_cv, ridge_results = fit_ridge_cv(X_scaled, y_target)
    ridge_final = Ridge(alpha=ridge_cv.alpha_)
    ridge_final.fit(X_train, y_train)
    ridge_pred = ridge_final.predict(X_test)
    
    results['models']['ridge'] = {
        'model': ridge_final,
        'optimal_alpha': ridge_cv.alpha_,
        'coefficients': dict(zip(feature_cols, ridge_cv.coef_)),
        'is_r2': ridge_final.score(X_train, y_train),
        'oos_r2': max(0, r2_score(y_test, ridge_pred)),
        'oos_rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
    }
    
    # OLS (for comparison)
    import statsmodels.api as sm
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    ols_pred = ols_model.predict(X_test_const)
    
    results['models']['ols'] = {
        'model': ols_model,
        'is_r2': ols_model.rsquared,
        'oos_r2': max(0, r2_score(y_test, ols_pred)),
        'oos_rmse': np.sqrt(mean_squared_error(y_test, ols_pred)),
    }
    
    # Best model
    oos_r2s = {k: v['oos_r2'] for k, v in results['models'].items()}
    results['best_model'] = max(oos_r2s, key=oos_r2s.get)
    
    return results
