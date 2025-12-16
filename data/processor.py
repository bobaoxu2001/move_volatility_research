"""
Data processing and feature engineering utilities.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
from ..config import CONFIG


def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std


def compute_percentile(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank."""
    return series.rolling(window=window).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
    )


def compute_spread_volatility(tlt: pd.Series, ief: pd.Series, 
                              window: int = 21) -> pd.Series:
    """
    Compute Treasury term-spread volatility proxy.
    
    Parameters
    ----------
    tlt : pd.Series
        TLT (20+ year Treasury) prices
    ief : pd.Series
        IEF (7-10 year Treasury) prices
    window : int
        Rolling window for volatility calculation
        
    Returns
    -------
    pd.Series
        Annualized spread volatility
    """
    term_spread = np.log(tlt / ief)
    spread_change = term_spread.diff()
    return spread_change.rolling(window=window).std() * np.sqrt(252)


def process_data(raw_data: Dict[str, pd.Series], config: Dict = None) -> pd.DataFrame:
    """
    Process raw data into analysis-ready DataFrame.
    
    Parameters
    ----------
    raw_data : dict
        Dictionary of raw data series
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame with all features
    """
    if config is None:
        config = CONFIG
    
    zscore_window = config['zscore_window']
    vol_window = config['volatility_window']
    
    # Create master dataframe
    df = pd.DataFrame()
    df['move'] = raw_data['move']
    df['vix'] = raw_data['vix']
    df['vxn'] = raw_data['vxn']
    
    # Compute spread volatility
    if raw_data['ief'] is not None and raw_data['tlt'] is not None:
        df['spread_volatility'] = compute_spread_volatility(
            raw_data['tlt'], raw_data['ief'], vol_window
        )
    else:
        # Use synthetic proxy
        from .loader import create_synthetic_volatility_data
        spread_vol = create_synthetic_volatility_data(
            config['start_date'], config['end_date'],
            base_level=0.05, vol=0.01, name="spread_volatility", seed=45
        )
        spread_vol = spread_vol.reindex(df.index)
        df['spread_volatility'] = 0.3 * (df['move'] / df['move'].mean() * 0.05) + \
                                   0.7 * spread_vol.fillna(method='ffill')
    
    # MOVE features
    df['move_return'] = np.log(df['move']).diff()
    df['move_zscore'] = compute_zscore(df['move'], zscore_window)
    df['move_percentile'] = compute_percentile(df['move'], zscore_window)
    
    # VIX features
    df['vix_return'] = np.log(df['vix']).diff()
    df['vix_zscore'] = compute_zscore(df['vix'], zscore_window)
    
    # VXN features
    df['vxn_return'] = np.log(df['vxn']).diff()
    df['vxn_zscore'] = compute_zscore(df['vxn'], zscore_window)
    
    # Spread volatility features
    df['spread_vol_zscore'] = compute_zscore(df['spread_volatility'], zscore_window)
    
    # Additional features for advanced models
    df['move_zscore_sq'] = df['move_zscore'] ** 2
    df['move_vix_interaction'] = df['move_zscore'] * df['vix_zscore']
    
    # Clean data
    df = df.ffill(limit=3).dropna()
    
    # Regime classification
    df['regime'] = pd.cut(
        df['move_zscore'],
        bins=[-np.inf, config['low_vol_threshold'], config['high_vol_threshold'], np.inf],
        labels=['Low Vol', 'Normal', 'High Vol']
    )
    
    df['vix_regime'] = pd.cut(
        df['vix_zscore'],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=['Low VIX', 'Normal VIX', 'High VIX']
    )
    
    # Forward targets
    df['fwd_spread_vol'] = df['spread_vol_zscore'].shift(-1)
    df['fwd_vxn'] = df['vxn_zscore'].shift(-1)
    
    return df


def train_test_split(df: pd.DataFrame, test_ratio: float = 0.3, 
                     purge_gap: int = 60, min_train: int = 252) -> tuple:
    """
    Time series train-test split with purge gap.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    test_ratio : float
        Proportion of data for testing
    purge_gap : int
        Number of observations to purge between train/test
    min_train : int
        Minimum training observations
        
    Returns
    -------
    tuple
        (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_ratio))
    cut = split_idx - purge_gap
    
    if cut < min_train:
        cut = min_train
    
    train = df.iloc[:cut]
    test = df.iloc[split_idx:]
    
    return train, test
