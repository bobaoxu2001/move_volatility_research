"""
Information Coefficient and signal evaluation utilities.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Tuple


def compute_rolling_ic(signal: pd.Series, target: pd.Series, 
                       window: int = 60, forward: bool = True) -> pd.Series:
    """
    Compute rolling Spearman IC.
    
    Parameters
    ----------
    signal : pd.Series
        Signal series
    target : pd.Series
        Target series
    window : int
        Rolling window
    forward : bool
        If True, use forward target (shift -1)
        
    Returns
    -------
    pd.Series
        Rolling IC series
    """
    if forward:
        t = target.shift(-1)
    else:
        t = target
    
    # Spearman = Pearson on ranks
    return signal.rank().rolling(window).corr(t.rank())


def compute_ic_statistics(ic_series: pd.Series) -> Dict:
    """
    Compute IC statistics.
    
    Parameters
    ----------
    ic_series : pd.Series
        IC time series
        
    Returns
    -------
    dict
        IC statistics
    """
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    
    if ic_std and ic_std > 1e-12:
        ir = ic_mean / ic_std
    else:
        ir = np.nan
    
    return {
        'mean_ic': ic_mean,
        'std_ic': ic_std,
        'info_ratio': ir,
        'pct_positive': (ic_series > 0).mean(),
        'max_ic': ic_series.max(),
        'min_ic': ic_series.min(),
    }


def compute_hit_rate(signal: pd.Series, target: pd.Series, 
                     forward: bool = True) -> Dict:
    """
    Compute hit rate (directional accuracy).
    
    Parameters
    ----------
    signal : pd.Series
        Signal series
    target : pd.Series
        Target series
    forward : bool
        If True, use forward target
        
    Returns
    -------
    dict
        Hit rate statistics
    """
    if forward:
        t = target.shift(-1)
    else:
        t = target
    
    signal_dir = np.sign(signal)
    target_dir = np.sign(t)
    
    # Overall hit rate
    valid = ~(signal_dir.isna() | target_dir.isna())
    overall_hit = (signal_dir[valid] == target_dir[valid]).mean()
    
    # Conditional hit rates
    high_signal = signal > 1
    low_signal = signal < -1
    
    if high_signal.sum() > 0:
        hr_high = ((signal_dir == target_dir) & high_signal).sum() / high_signal.sum()
    else:
        hr_high = np.nan
    
    if low_signal.sum() > 0:
        hr_low = ((signal_dir == target_dir) & low_signal).sum() / low_signal.sum()
    else:
        hr_low = np.nan
    
    return {
        'overall': overall_hit,
        'high_regime': hr_high,
        'low_regime': hr_low,
    }


def compute_oos_metrics(train_signal: pd.Series, train_target: pd.Series,
                        test_signal: pd.Series, test_target: pd.Series) -> Dict:
    """
    Compute out-of-sample evaluation metrics.
    
    Parameters
    ----------
    train_signal : pd.Series
        Training signal
    train_target : pd.Series
        Training target
    test_signal : pd.Series
        Test signal
    test_target : pd.Series
        Test target
        
    Returns
    -------
    dict
        OOS metrics
    """
    # OOS IC
    oos_ic = spearmanr(test_signal, test_target).correlation
    
    # OOS hit rate
    oos_hit = (np.sign(test_signal) == np.sign(test_target)).mean()
    
    # OOS by regime
    high_vol_mask = test_signal > 1
    if high_vol_mask.sum() > 10:
        oos_ic_high = spearmanr(
            test_signal[high_vol_mask], 
            test_target[high_vol_mask]
        ).correlation
        oos_hit_high = (
            np.sign(test_signal[high_vol_mask]) == np.sign(test_target[high_vol_mask])
        ).mean()
    else:
        oos_ic_high = np.nan
        oos_hit_high = np.nan
    
    return {
        'oos_ic': oos_ic,
        'oos_hit_rate': oos_hit,
        'oos_ic_high_vol': oos_ic_high,
        'oos_hit_rate_high_vol': oos_hit_high,
        'n_test': len(test_signal),
        'n_high_vol': high_vol_mask.sum(),
    }
