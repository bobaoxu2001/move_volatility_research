"""
Granger causality testing utilities.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Tuple
from ..config import CONFIG


def test_granger_causality(df: pd.DataFrame,
                           target_col: str,
                           predictor_col: str,
                           lags: List[int] = None,
                           verbose: bool = False) -> Dict:
    """
    Test Granger causality from predictor to target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target variable column
    predictor_col : str
        Predictor variable column
    lags : list
        Lags to test
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Test results by lag
    """
    if lags is None:
        lags = CONFIG['granger_lags']
    
    gc_df = df[[target_col, predictor_col]].dropna()
    
    results = {'by_lag': {}, 'min_pvalue': np.nan}
    pvalues = []
    
    for lag in lags:
        try:
            gc_test = grangercausalitytests(gc_df, maxlag=lag, verbose=False)
            
            f_stat = gc_test[lag][0]['ssr_ftest'][0]
            p_val = gc_test[lag][0]['ssr_ftest'][1]
            
            results['by_lag'][lag] = {
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant_01': p_val < 0.01,
                'significant_05': p_val < 0.05,
                'significant_10': p_val < 0.10,
            }
            
            pvalues.append(p_val)
            
            if verbose:
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  Lag {lag}: F={f_stat:.2f}, p={p_val:.4f} {sig}")
                
        except Exception as e:
            results['by_lag'][lag] = {'error': str(e)}
    
    if pvalues:
        results['min_pvalue'] = min(pvalues)
        results['significant'] = results['min_pvalue'] < 0.05
    
    return results


def test_granger_both_directions(df: pd.DataFrame,
                                 col1: str,
                                 col2: str,
                                 lags: List[int] = None) -> Dict:
    """
    Test Granger causality in both directions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    col1 : str
        First variable column
    col2 : str
        Second variable column
    lags : list
        Lags to test
        
    Returns
    -------
    dict
        Bidirectional test results
    """
    results = {}
    
    # col1 -> col2
    results[f'{col1}_to_{col2}'] = test_granger_causality(
        df, target_col=col2, predictor_col=col1, lags=lags
    )
    
    # col2 -> col1
    results[f'{col2}_to_{col1}'] = test_granger_causality(
        df, target_col=col1, predictor_col=col2, lags=lags
    )
    
    return results


def test_granger_on_differences(df: pd.DataFrame,
                                target_col: str,
                                predictor_col: str,
                                lags: List[int] = None) -> Dict:
    """
    Test Granger causality on first differences (for stationarity).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target_col : str
        Target variable column
    predictor_col : str
        Predictor variable column
    lags : list
        Lags to test
        
    Returns
    -------
    dict
        Test results on differences
    """
    diff_df = pd.DataFrame({
        target_col: df[target_col].diff(),
        predictor_col: df[predictor_col].diff()
    }).dropna()
    
    return test_granger_causality(diff_df, target_col, predictor_col, lags)
