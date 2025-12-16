"""
Data acquisition and preprocessing module.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Tuple
from scipy import stats

from .config import DATA_CONFIG, FEATURE_CONFIG, SEEDS


def fetch_ticker_data(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """
    Fetch ticker data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD)
        
    Returns
    -------
    pd.Series or None
        Close prices or None if fetch fails
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) > 0:
            print(f"  ✓ {ticker}: {len(data)} observations fetched")
            return data['Close'].squeeze()
        else:
            raise ValueError(f"No data for {ticker}")
    except Exception as e:
        print(f"  ⚠ {ticker} failed: {e}")
        return None


def create_synthetic_data(
    start: str, 
    end: str, 
    base_level: float = 100, 
    vol: float = 10, 
    seed: int = 42
) -> pd.Series:
    """
    Create synthetic mean-reverting volatility data for demonstration.
    
    Parameters
    ----------
    start : str
        Start date
    end : str
        End date
    base_level : float
        Mean level to revert to
    vol : float
        Volatility of shocks
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Synthetic volatility series
    """
    dates = pd.date_range(start, end, freq='B')
    np.random.seed(seed)
    n = len(dates)
    
    values = [base_level]
    regime = 0
    
    for i in range(1, n):
        # Occasional regime shifts
        if np.random.random() < 0.02:
            regime = 1 - regime
        
        shock = np.random.normal(0, vol * (1 + regime * 0.5))
        mean_reversion = 0.03 * (base_level * (1 + regime * 0.3) - values[-1])
        values.append(max(base_level * 0.5, values[-1] + mean_reversion + shock))
    
    return pd.Series(values, index=dates)


def load_all_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_synthetic_fallback: bool = True
) -> pd.DataFrame:
    """
    Load all required data sources.
    
    Parameters
    ----------
    start_date : str, optional
        Override config start date
    end_date : str, optional
        Override config end date
    use_synthetic_fallback : bool
        Whether to use synthetic data if fetch fails
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all data sources
    """
    start = start_date or DATA_CONFIG['start_date']
    end = end_date or DATA_CONFIG['end_date']
    
    print("--- Fetching Data Sources ---")
    
    # MOVE Index
    move_data = fetch_ticker_data(DATA_CONFIG['tickers']['move'], start, end)
    if move_data is None and use_synthetic_fallback:
        print("    → Creating synthetic MOVE data")
        move_data = create_synthetic_data(start, end, 100, 8, SEEDS['move'])
    
    # VIX
    vix_data = fetch_ticker_data(DATA_CONFIG['tickers']['vix'], start, end)
    if vix_data is None and use_synthetic_fallback:
        print("    → Creating synthetic VIX data")
        vix_data = create_synthetic_data(start, end, 18, 4, SEEDS['vix'])
    
    # VXN
    vxn_data = fetch_ticker_data(DATA_CONFIG['tickers']['vxn'], start, end)
    if vxn_data is None and use_synthetic_fallback:
        print("    → Creating synthetic VXN data")
        np.random.seed(SEEDS['vxn'])
        vxn_data = vix_data * 1.15 + np.random.normal(0, 2, len(vix_data))
        vxn_data = pd.Series(vxn_data.values, index=vix_data.index)
    
    # Treasury ETFs
    ief_data = fetch_ticker_data(DATA_CONFIG['tickers']['ief'], start, end)
    tlt_data = fetch_ticker_data(DATA_CONFIG['tickers']['tlt'], start, end)
    
    # Calculate spread volatility
    if ief_data is not None and tlt_data is not None:
        term_spread = np.log(tlt_data / ief_data)
        spread_change = term_spread.diff()
        spread_vol = spread_change.rolling(
            window=FEATURE_CONFIG['volatility_window']
        ).std() * np.sqrt(252)
    else:
        print("    → Creating synthetic spread volatility")
        spread_vol = create_synthetic_data(start, end, 0.05, 0.01, SEEDS['spread'])
        spread_vol = spread_vol.reindex(move_data.index)
        spread_vol = 0.3 * (move_data / move_data.mean() * 0.05) + \
                     0.7 * spread_vol.fillna(method='ffill')
    
    # Combine into DataFrame
    data = pd.DataFrame({
        'move': move_data,
        'vix': vix_data,
        'vxn': vxn_data,
        'spread_volatility': spread_vol,
    })
    
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw data: compute returns, z-scores, and clean.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw data from load_all_data()
        
    Returns
    -------
    pd.DataFrame
        Processed data with derived features
    """
    df = data.copy()
    window = FEATURE_CONFIG['zscore_window']
    
    # MOVE features
    df['move_return'] = np.log(df['move']).diff()
    roll_mean = df['move'].rolling(window=window).mean()
    roll_std = df['move'].rolling(window=window).std()
    df['move_zscore'] = (df['move'] - roll_mean) / roll_std
    df['move_percentile'] = df['move'].rolling(window=window).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
    )
    
    # VIX features
    df['vix_return'] = np.log(df['vix']).diff()
    roll_mean_vix = df['vix'].rolling(window=window).mean()
    roll_std_vix = df['vix'].rolling(window=window).std()
    df['vix_zscore'] = (df['vix'] - roll_mean_vix) / roll_std_vix
    
    # VXN features
    df['vxn_return'] = np.log(df['vxn']).diff()
    roll_mean_vxn = df['vxn'].rolling(window=window).mean()
    roll_std_vxn = df['vxn'].rolling(window=window).std()
    df['vxn_zscore'] = (df['vxn'] - roll_mean_vxn) / roll_std_vxn
    
    # Spread volatility features
    roll_mean_sv = df['spread_volatility'].rolling(window=window).mean()
    roll_std_sv = df['spread_volatility'].rolling(window=window).std()
    df['spread_vol_zscore'] = (df['spread_volatility'] - roll_mean_sv) / roll_std_sv
    
    # Clean data
    df = df.ffill(limit=3).dropna()
    
    print(f"\n--- Data Quality Report ---")
    print(f"  Total Observations: {len(df)}")
    print(f"  Date Range: {df.index.min().strftime('%Y-%m-%d')} to "
          f"{df.index.max().strftime('%Y-%m-%d')}")
    
    return df


def get_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Main entry point: load and preprocess all data.
    
    Parameters
    ----------
    start_date : str, optional
        Override config start date
    end_date : str, optional
        Override config end date
        
    Returns
    -------
    pd.DataFrame
        Clean, processed data ready for analysis
    """
    raw_data = load_all_data(start_date, end_date)
    processed_data = preprocess_data(raw_data)
    return processed_data
