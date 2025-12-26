"""
Data ingestion and preprocessing module for GME trading strategy.
Handles loading BTC/USD data and computing log returns.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import config


def load_btc_data(
    filepath: Optional[str] = None,
    ticker: str = config.TICKER,
    interval: str = config.INTERVAL,
    lookback_days: int = config.LOOKBACK_DAYS
) -> pd.DataFrame:
    """
    Load hourly BTC/USD historical data.
    
    Args:
        filepath: Optional path to CSV file. If None, downloads from yfinance.
        ticker: Ticker symbol (default: BTC-USD)
        interval: Data interval (default: 1h)
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with OHLCV data indexed by datetime
    """
    if filepath is not None:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    
    # Ensure datetime index and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Remove any duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from price series.
    
    Log returns: r_t = ln(P_t / P_{t-1})
    
    Advantages:
    - Time-additive: r_{t1,t2} = r_{t1} + r_{t2}
    - More suitable for statistical modeling
    - Approximately normally distributed
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of log returns (first value is NaN)
    """
    return np.log(prices / prices.shift(1))


def prepare_dataset(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Prepare dataset for feature engineering.
    
    Steps:
    1. Extract price column
    2. Compute log returns
    3. Drop NaN values
    4. Add returns as column
    
    Args:
        df: OHLCV DataFrame
        price_col: Column to use for price (default: Close)
        
    Returns:
        DataFrame with 'price' and 'log_return' columns
    """
    result = pd.DataFrame(index=df.index)
    result['price'] = df[price_col]
    result['log_return'] = compute_log_returns(result['price'])
    
    # Drop rows with NaN (first row after log return)
    result = result.dropna()
    
    return result


if __name__ == "__main__":
    # Test data loading
    print("Loading BTC/USD hourly data...")
    df = load_btc_data()
    print(f"Loaded {len(df)} hourly records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Prepare dataset
    data = prepare_dataset(df)
    print(f"\nPrepared dataset shape: {data.shape}")
    print(f"Log return stats:")
    print(data['log_return'].describe())
