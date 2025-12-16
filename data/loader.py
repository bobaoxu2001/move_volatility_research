"""
Data loading and synthetic data generation utilities.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict
from ..config import CONFIG, TICKERS


def fetch_data_with_fallback(ticker: str, start: str, end: str, 
                             fallback_name: str = "data") -> Optional[pd.Series]:
    """
    Fetch data from Yahoo Finance with synthetic fallback.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    start : str
        Start date (YYYY-MM-DD)
    end : str
        End date (YYYY-MM-DD)
    fallback_name : str
        Name for synthetic data if fetch fails
        
    Returns
    -------
    pd.Series or None
        Price series or None if fetch fails
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
        print(f"    → Creating synthetic {fallback_name} for demonstration")
        return None


def create_synthetic_volatility_data(start: str, end: str, 
                                     base_level: float = 100, 
                                     vol: float = 10, 
                                     name: str = "MOVE", 
                                     seed: int = 42) -> pd.Series:
    """
    Create realistic synthetic volatility data with mean-reverting process.
    
    Parameters
    ----------
    start : str
        Start date
    end : str
        End date
    base_level : float
        Mean level for the process
    vol : float
        Volatility parameter
    name : str
        Series name
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
    
    return pd.Series(values, index=dates, name=name)


def load_all_data(config: Dict = None) -> Dict[str, pd.Series]:
    """
    Load all required data sources.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    dict
        Dictionary of data series
    """
    if config is None:
        config = CONFIG
    
    start = config['start_date']
    end = config['end_date']
    
    print("\n--- Fetching Data Sources ---")
    
    data = {}
    
    # MOVE Index
    data['move'] = fetch_data_with_fallback(TICKERS['move'], start, end, "MOVE")
    if data['move'] is None:
        data['move'] = create_synthetic_volatility_data(
            start, end, base_level=100, vol=8, name="MOVE", seed=42
        )
    
    # VIX
    data['vix'] = fetch_data_with_fallback(TICKERS['vix'], start, end, "VIX")
    if data['vix'] is None:
        data['vix'] = create_synthetic_volatility_data(
            start, end, base_level=18, vol=4, name="VIX", seed=43
        )
    
    # VXN
    data['vxn'] = fetch_data_with_fallback(TICKERS['vxn'], start, end, "VXN")
    if data['vxn'] is None:
        np.random.seed(44)
        data['vxn'] = data['vix'] * 1.15 + np.random.normal(0, 2, len(data['vix']))
        data['vxn'] = pd.Series(data['vxn'].values, index=data['vix'].index, name="VXN")
    
    # Treasury ETFs
    data['ief'] = fetch_data_with_fallback(TICKERS['ief'], start, end, "IEF")
    data['tlt'] = fetch_data_with_fallback(TICKERS['tlt'], start, end, "TLT")
    
    return data
