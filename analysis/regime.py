"""
Regime-conditional analysis utilities.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, ttest_ind
from typing import Dict, List


def compute_regime_statistics(df: pd.DataFrame,
                              regime_col: str = 'regime',
                              signal_col: str = 'move_zscore',
                              target_col: str = 'fwd_spread_vol') -> pd.DataFrame:
    """
    Compute statistics for each regime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    regime_col : str
        Regime column name
    signal_col : str
        Signal column name
    target_col : str
        Target column name
        
    Returns
    -------
    pd.DataFrame
        Statistics by regime
    """
    results = []
    
    for regime in df[regime_col].unique():
        if pd.isna(regime):
            continue
            
        subset = df[df[regime_col] == regime].dropna(subset=[signal_col, target_col])
        
        if len(subset) < 30:
            continue
        
        ic = spearmanr(subset[signal_col], subset[target_col]).correlation
        hit_rate = (np.sign(subset[signal_col]) == np.sign(subset[target_col])).mean()
        avg_target = subset[target_col].mean()
        std_target = subset[target_col].std()
        
        results.append({
            'regime': regime,
            'n_obs': len(subset),
            'ic': ic,
            'hit_rate': hit_rate,
            'avg_target': avg_target,
            'std_target': std_target,
        })
    
    return pd.DataFrame(results)


def compute_cross_regime_statistics(df: pd.DataFrame,
                                    conditioning_col: str = 'vix_regime',
                                    signal_col: str = 'move_zscore',
                                    target_col: str = 'fwd_spread_vol') -> pd.DataFrame:
    """
    Compute statistics conditioned on another regime variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    conditioning_col : str
        Conditioning regime column
    signal_col : str
        Signal column name
    target_col : str
        Target column name
        
    Returns
    -------
    pd.DataFrame
        Cross-regime statistics
    """
    import statsmodels.api as sm
    
    results = []
    
    for regime in df[conditioning_col].unique():
        if pd.isna(regime):
            continue
            
        subset = df[df[conditioning_col] == regime].dropna(subset=[signal_col, target_col])
        
        if len(subset) < 30:
            continue
        
        ic = spearmanr(subset[signal_col], subset[target_col]).correlation
        hit_rate = (np.sign(subset[signal_col]) == np.sign(subset[target_col])).mean()
        
        # Regime-specific regression
        X = sm.add_constant(subset[signal_col])
        model = sm.OLS(subset[target_col], X).fit()
        
        results.append({
            'regime': regime,
            'n_obs': len(subset),
            'ic': ic,
            'hit_rate': hit_rate,
            'beta': model.params.iloc[1],
            'p_value': model.pvalues.iloc[1],
        })
    
    return pd.DataFrame(results)


def test_regime_separation(df: pd.DataFrame,
                           regime_col: str = 'regime',
                           target_col: str = 'fwd_spread_vol') -> Dict:
    """
    Test statistical significance of regime separation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    regime_col : str
        Regime column name
    target_col : str
        Target column name
        
    Returns
    -------
    dict
        Test results
    """
    results = {}
    
    # Get regime means
    regime_stats = df.groupby(regime_col)[target_col].agg(['mean', 'std', 'count'])
    results['regime_stats'] = regime_stats
    
    # T-test between high and low regimes
    high_vol = df[df[regime_col] == 'High Vol'][target_col].dropna()
    low_vol = df[df[regime_col] == 'Low Vol'][target_col].dropna()
    
    if len(high_vol) > 10 and len(low_vol) > 10:
        t_stat, p_val = ttest_ind(high_vol, low_vol)
        results['ttest'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
        }
    
    return results
