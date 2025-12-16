"""
Backtesting and evaluation module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.stats import spearmanr, ttest_ind
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

from .config import BACKTEST_CONFIG, MODEL_CONFIG


@dataclass
class BacktestResult:
    """Container for backtest results."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_r2: float
    oos_r2: float
    oos_ic: float
    oos_hit_rate: float
    metrics: Dict[str, float]


def train_test_split_ts(
    data: pd.DataFrame,
    test_ratio: float = None,
    purge_gap: int = None,
    min_train: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time series train/test split with purged gap.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to split
    test_ratio : float, optional
        Fraction of data for testing
    purge_gap : int, optional
        Days to purge between train and test
    min_train : int, optional
        Minimum training samples
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Train and test dataframes
    """
    test_ratio = test_ratio or BACKTEST_CONFIG['test_ratio']
    purge_gap = purge_gap or BACKTEST_CONFIG['purge_gap']
    min_train = min_train or BACKTEST_CONFIG['min_train_samples']
    
    split_idx = int(len(data) * (1 - test_ratio))
    
    # Apply purge gap with safety check
    cut = split_idx - purge_gap
    if cut < min_train:
        print(f"  Warning: Adjusting cut from {cut} to {min_train}")
        cut = min_train
    
    train = data.iloc[:cut]
    test = data.iloc[split_idx:]
    
    return train, test


def compute_hit_rate(
    signal: pd.Series,
    target: pd.Series,
    forward: bool = True
) -> float:
    """
    Compute hit rate (directional accuracy).
    
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
        Hit rate (fraction of correct sign predictions)
    """
    if forward:
        target = target.shift(-1)
    
    df = pd.DataFrame({'signal': signal, 'target': target}).dropna()
    signal_dir = np.sign(df['signal'])
    target_dir = np.sign(df['target'])
    
    return (signal_dir == target_dir).mean()


def compute_regime_metrics(
    data: pd.DataFrame,
    signal_col: str = 'signal_zscore',
    target_col: str = 'fwd_spread_vol',
    regime_col: str = 'regime'
) -> pd.DataFrame:
    """
    Compute metrics by regime.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with signal, target, and regime columns
    signal_col : str
        Signal column name
    target_col : str
        Target column name
    regime_col : str
        Regime column name
        
    Returns
    -------
    pd.DataFrame
        Metrics for each regime
    """
    results = []
    
    for regime in data[regime_col].dropna().unique():
        subset = data[data[regime_col] == regime].dropna(subset=[signal_col, target_col])
        
        if len(subset) > 30:
            ic = spearmanr(subset[signal_col], subset[target_col]).correlation
            hit = (np.sign(subset[signal_col]) == np.sign(subset[target_col])).mean()
            avg_target = subset[target_col].mean()
            
            results.append({
                'regime': regime,
                'n_obs': len(subset),
                'ic': ic,
                'hit_rate': hit,
                'avg_target': avg_target,
            })
    
    return pd.DataFrame(results)


def run_oos_backtest(
    data: pd.DataFrame,
    signal_col: str = 'signal_zscore',
    target_col: str = 'spread_vol_zscore'
) -> BacktestResult:
    """
    Run out-of-sample backtest.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with signal and target columns
    signal_col : str
        Signal column name
    target_col : str
        Target column name
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    # Prepare data with forward target
    df = data[[signal_col, target_col]].copy()
    df['y_fwd'] = df[target_col].shift(-1)
    df['x_lag'] = df[signal_col]
    df = df.dropna(subset=['x_lag', 'y_fwd'])
    
    # Split
    train, test = train_test_split_ts(df)
    
    # Fit in-sample model
    X_train = sm.add_constant(train['x_lag'])
    model = sm.OLS(train['y_fwd'], X_train).fit()
    
    # Predict out-of-sample
    X_test = sm.add_constant(test['x_lag'])
    pred = model.predict(X_test)
    
    # Calculate metrics
    oos_r2 = r2_score(test['y_fwd'], pred)
    oos_ic = spearmanr(test['x_lag'], test['y_fwd']).correlation
    oos_hit = (np.sign(test['x_lag']) == np.sign(test['y_fwd'])).mean()
    
    # Additional metrics
    metrics = {
        'oos_rmse': np.sqrt(mean_squared_error(test['y_fwd'], pred)),
        'is_beta': model.params.iloc[1],
        'is_pvalue': model.pvalues.iloc[1],
        'degradation': model.rsquared - max(0, oos_r2),
    }
    
    return BacktestResult(
        train_start=train.index.min().strftime('%Y-%m-%d'),
        train_end=train.index.max().strftime('%Y-%m-%d'),
        test_start=test.index.min().strftime('%Y-%m-%d'),
        test_end=test.index.max().strftime('%Y-%m-%d'),
        is_r2=model.rsquared,
        oos_r2=max(0, oos_r2),
        oos_ic=oos_ic,
        oos_hit_rate=oos_hit,
        metrics=metrics
    )


def regime_separation_test(
    data: pd.DataFrame,
    target_col: str = 'fwd_spread_vol',
    regime_col: str = 'regime_zscore'
) -> Dict[str, float]:
    """
    Test for statistically significant regime separation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with target and regime columns
    target_col : str
        Target column name
    regime_col : str
        Regime column name
        
    Returns
    -------
    dict
        T-test statistics and means by regime
    """
    high_outcomes = data[data[regime_col] == 'High Vol'][target_col].dropna()
    low_outcomes = data[data[regime_col] == 'Low Vol'][target_col].dropna()
    
    results = {
        'high_vol_mean': high_outcomes.mean(),
        'high_vol_std': high_outcomes.std(),
        'high_vol_n': len(high_outcomes),
        'low_vol_mean': low_outcomes.mean(),
        'low_vol_std': low_outcomes.std(),
        'low_vol_n': len(low_outcomes),
    }
    
    if len(high_outcomes) > 10 and len(low_outcomes) > 10:
        t_stat, p_val = ttest_ind(high_outcomes, low_outcomes)
        results['t_statistic'] = t_stat
        results['p_value'] = p_val
    
    return results


def evaluate_model_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, object]
) -> pd.DataFrame:
    """
    Compare multiple models on same train/test split.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    models : dict
        Dictionary of model name -> fitted model
        
    Returns
    -------
    pd.DataFrame
        Comparison metrics for each model
    """
    results = []
    
    for name, model in models.items():
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        results.append({
            'model': name,
            'is_r2': r2_score(y_train, pred_train),
            'oos_r2': max(0, r2_score(y_test, pred_test)),
            'oos_rmse': np.sqrt(mean_squared_error(y_test, pred_test)),
        })
    
    return pd.DataFrame(results)


def generate_summary_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Generate summary statistics for the analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Fully processed data
        
    Returns
    -------
    dict
        Summary statistics
    """
    stats = {}
    
    # Data coverage
    stats['n_observations'] = len(data)
    stats['start_date'] = data.index.min().strftime('%Y-%m-%d')
    stats['end_date'] = data.index.max().strftime('%Y-%m-%d')
    
    # Signal statistics
    if 'signal_zscore' in data.columns:
        stats['signal_mean'] = data['signal_zscore'].mean()
        stats['signal_std'] = data['signal_zscore'].std()
        stats['signal_skew'] = data['signal_zscore'].skew()
        stats['signal_kurt'] = data['signal_zscore'].kurtosis()
    
    # Regime distribution
    if 'regime' in data.columns:
        regime_dist = data['regime'].value_counts(normalize=True)
        for regime, pct in regime_dist.items():
            stats[f'regime_{regime}_pct'] = pct
    
    # Correlations
    if 'move_zscore' in data.columns and 'spread_vol_zscore' in data.columns:
        stats['corr_move_spread'] = data['move_zscore'].corr(data['spread_vol_zscore'])
    
    if 'move_zscore' in data.columns and 'vxn_zscore' in data.columns:
        stats['corr_move_vxn'] = data['move_zscore'].corr(data['vxn_zscore'])
    
    return stats
