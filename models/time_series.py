"""
Time series models (ARIMA, GARCH) for volatility forecasting.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
from ..config import CONFIG

# Try to import arch for GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def fit_arima(y_train: pd.Series, order: Tuple[int, int, int] = None,
              forecast_steps: int = None) -> Dict:
    """
    Fit ARIMA model.
    
    Parameters
    ----------
    y_train : pd.Series
        Training series
    order : tuple
        (p, d, q) order
    forecast_steps : int
        Number of steps to forecast
        
    Returns
    -------
    dict
        Model results and forecasts
    """
    if order is None:
        order = CONFIG['arima_order']
    
    try:
        model = ARIMA(y_train, order=order)
        fit = model.fit()
        
        results = {
            'model': fit,
            'aic': fit.aic,
            'bic': fit.bic,
            'params': fit.params.to_dict(),
            'success': True,
        }
        
        if forecast_steps:
            results['forecast'] = fit.forecast(steps=forecast_steps)
        
        return results
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def fit_arimax(y_train: pd.Series, exog_train: pd.Series,
               order: Tuple[int, int, int] = None,
               exog_test: pd.Series = None) -> Dict:
    """
    Fit ARIMA-X model with exogenous variable.
    
    Parameters
    ----------
    y_train : pd.Series
        Training target series
    exog_train : pd.Series
        Training exogenous variable
    order : tuple
        (p, d, q) order
    exog_test : pd.Series
        Test exogenous variable for forecasting
        
    Returns
    -------
    dict
        Model results and forecasts
    """
    if order is None:
        order = CONFIG['arima_order']
    
    try:
        model = ARIMA(y_train, exog=exog_train, order=order)
        fit = model.fit()
        
        results = {
            'model': fit,
            'aic': fit.aic,
            'bic': fit.bic,
            'params': fit.params.to_dict(),
            'success': True,
        }
        
        if exog_test is not None:
            results['forecast'] = fit.forecast(steps=len(exog_test), exog=exog_test)
        
        return results
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> Dict:
    """
    Fit GARCH model for volatility clustering.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    p : int
        GARCH p order
    q : int
        GARCH q order
        
    Returns
    -------
    dict
        Model results
    """
    if not ARCH_AVAILABLE:
        return {'success': False, 'error': 'arch package not available'}
    
    try:
        # Scale returns for numerical stability
        scaled_returns = returns * 100
        
        model = arch_model(scaled_returns, vol='Garch', p=p, q=q)
        fit = model.fit(disp='off')
        
        results = {
            'model': fit,
            'omega': fit.params['omega'],
            'alpha': fit.params[f'alpha[{p}]'],
            'beta': fit.params[f'beta[{q}]'],
            'persistence': fit.params[f'alpha[{p}]'] + fit.params[f'beta[{q}]'],
            'aic': fit.aic,
            'success': True,
        }
        
        return results
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def compare_time_series_models(df: pd.DataFrame, 
                               target_col: str = 'spread_vol_zscore',
                               exog_col: str = 'move_zscore',
                               test_ratio: float = 0.3) -> Dict:
    """
    Compare ARIMA, ARIMA-X, and GARCH models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target column
    exog_col : str
        Exogenous variable column
    test_ratio : float
        Test set proportion
        
    Returns
    -------
    dict
        Comparison results
    """
    ts_data = df[[target_col, exog_col]].dropna()
    
    split_idx = int(len(ts_data) * (1 - test_ratio))
    train = ts_data.iloc[:split_idx]
    test = ts_data.iloc[split_idx:]
    
    results = {'models': {}}
    
    # ARIMA
    arima_results = fit_arima(
        train[target_col], 
        forecast_steps=len(test)
    )
    
    if arima_results['success']:
        arima_r2 = r2_score(test[target_col], arima_results['forecast'])
        arima_rmse = np.sqrt(mean_squared_error(test[target_col], arima_results['forecast']))
        results['models']['arima'] = {
            'aic': arima_results['aic'],
            'bic': arima_results['bic'],
            'oos_r2': max(0, arima_r2),
            'oos_rmse': arima_rmse,
            'success': True,
        }
    else:
        results['models']['arima'] = {'success': False, 'error': arima_results.get('error')}
    
    # ARIMA-X
    arimax_results = fit_arimax(
        train[target_col],
        train[exog_col],
        exog_test=test[exog_col]
    )
    
    if arimax_results['success']:
        arimax_r2 = r2_score(test[target_col], arimax_results['forecast'])
        arimax_rmse = np.sqrt(mean_squared_error(test[target_col], arimax_results['forecast']))
        results['models']['arimax'] = {
            'aic': arimax_results['aic'],
            'bic': arimax_results['bic'],
            'oos_r2': max(0, arimax_r2),
            'oos_rmse': arimax_rmse,
            'exog_coef': arimax_results['params'].get(exog_col, arimax_results['params'].get('x1')),
            'success': True,
        }
    else:
        results['models']['arimax'] = {'success': False, 'error': arimax_results.get('error')}
    
    # GARCH
    returns = train[target_col].diff().dropna()
    garch_results = fit_garch(returns, p=CONFIG['garch_order'][0], q=CONFIG['garch_order'][1])
    
    if garch_results['success']:
        results['models']['garch'] = {
            'omega': garch_results['omega'],
            'alpha': garch_results['alpha'],
            'beta': garch_results['beta'],
            'persistence': garch_results['persistence'],
            'aic': garch_results['aic'],
            'success': True,
        }
    else:
        results['models']['garch'] = {'success': False, 'error': garch_results.get('error')}
    
    return results
