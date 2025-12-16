"""
Feature engineering module for signal construction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .config import FEATURE_CONFIG, MODEL_CONFIG


def create_regime_labels(
    data: pd.DataFrame,
    column: str = 'move_zscore',
    high_threshold: float = None,
    low_threshold: float = None
) -> pd.Series:
    """
    Create regime labels based on z-score thresholds.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing the column to classify
    column : str
        Column name to use for regime classification
    high_threshold : float, optional
        Threshold for high vol regime (default from config)
    low_threshold : float, optional
        Threshold for low vol regime (default from config)
        
    Returns
    -------
    pd.Series
        Categorical series with regime labels
    """
    high_th = high_threshold or MODEL_CONFIG['high_vol_threshold']
    low_th = low_threshold or MODEL_CONFIG['low_vol_threshold']
    
    return pd.cut(
        data[column],
        bins=[-np.inf, low_th, high_th, np.inf],
        labels=['Low Vol', 'Normal', 'High Vol']
    )


def create_signal_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create z-score based signal.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data
        
    Returns
    -------
    pd.DataFrame
        Data with signal columns added
    """
    df = data.copy()
    df['signal_zscore'] = df['move_zscore']
    df['regime_zscore'] = create_regime_labels(df, 'move_zscore')
    return df


def create_signal_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create momentum-based signal (short MA - long MA).
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data
        
    Returns
    -------
    pd.DataFrame
        Data with momentum signal columns added
    """
    df = data.copy()
    short_window = FEATURE_CONFIG['signal_short_window']
    long_window = FEATURE_CONFIG['signal_long_window']
    
    short_ma = df['move'].rolling(window=short_window).mean()
    long_ma = df['move'].rolling(window=long_window).mean()
    
    df['signal_momentum'] = short_ma - long_ma
    roll_std = df['signal_momentum'].rolling(window=long_window).std()
    df['signal_momentum_zscore'] = df['signal_momentum'] / roll_std
    
    df['regime_momentum'] = np.where(
        df['signal_momentum_zscore'] > 0.5, 'Rising Vol',
        np.where(df['signal_momentum_zscore'] < -0.5, 'Falling Vol', 'Stable')
    )
    
    return df


def create_signal_divergence(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create divergence signal (MOVE z-score - VXN z-score).
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data
        
    Returns
    -------
    pd.DataFrame
        Data with divergence signal columns added
    """
    df = data.copy()
    df['signal_divergence'] = df['move_zscore'] - df['vxn_zscore']
    
    df['regime_divergence'] = np.where(
        df['signal_divergence'] > 1, 'MOVE High / VXN Low',
        np.where(df['signal_divergence'] < -1, 'VXN High / MOVE Low', 'Aligned')
    )
    
    return df


def create_nonlinear_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create non-linear features (quadratic, interaction terms).
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data
        
    Returns
    -------
    pd.DataFrame
        Data with non-linear features added
    """
    df = data.copy()
    
    # Quadratic term
    df['move_zscore_sq'] = df['move_zscore'] ** 2
    
    # Interaction term
    df['move_vix_interaction'] = df['move_zscore'] * df['vix_zscore']
    
    return df


def create_forward_targets(data: pd.DataFrame, horizons: List[int] = [1]) -> pd.DataFrame:
    """
    Create forward-looking target variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data
    horizons : list of int
        Forecast horizons in days
        
    Returns
    -------
    pd.DataFrame
        Data with forward target columns added
    """
    df = data.copy()
    
    for h in horizons:
        suffix = '' if h == 1 else f'_{h}d'
        df[f'fwd_spread_vol{suffix}'] = df['spread_vol_zscore'].shift(-h)
        df[f'fwd_vxn{suffix}'] = df['vxn_zscore'].shift(-h)
    
    return df


def create_vix_regime(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create VIX-based regime labels for cross-asset conditioning.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data
        
    Returns
    -------
    pd.DataFrame
        Data with VIX regime labels added
    """
    df = data.copy()
    df['vix_regime'] = pd.cut(
        df['vix_zscore'],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=['Low VIX', 'Normal VIX', 'High VIX']
    )
    return df


def engineer_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed data from data_loader
        
    Returns
    -------
    pd.DataFrame
        Data with all engineered features
    """
    df = data.copy()
    
    # Apply all feature engineering
    df = create_signal_zscore(df)
    df = create_signal_momentum(df)
    df = create_signal_divergence(df)
    df = create_nonlinear_features(df)
    df = create_forward_targets(df, horizons=[1, 5])
    df = create_vix_regime(df)
    
    # Add MOVE regime
    df['regime'] = create_regime_labels(df, 'move_zscore')
    
    print(f"✓ Feature engineering complete: {len(df.columns)} columns")
    
    return df


def get_feature_matrix(
    data: pd.DataFrame,
    feature_cols: List[str] = None,
    target_col: str = 'fwd_spread_vol'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix and target vector for modeling.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with engineered features
    feature_cols : list of str, optional
        Feature column names (default: standard features)
    target_col : str
        Target column name
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        Feature matrix X and target vector y
    """
    if feature_cols is None:
        feature_cols = ['move_zscore', 'vix_zscore', 'move_zscore_sq', 'move_vix_interaction']
    
    df = data[feature_cols + [target_col]].dropna()
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y
