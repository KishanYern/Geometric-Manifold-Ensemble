"""
Binance data loader using CCXT for Open Interest and Funding Rates.

CCXT provides free access to Binance public data without API keys:
- OHLCV candles
- Open Interest (futures only)
- Funding Rates (perpetual futures)

These "Z-axis" dimensions capture market microstructure that price alone
cannot reveal - liquidation cascades, leverage buildup, and sentiment.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
import config

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not installed. Run: pip install ccxt")


def get_binance_exchange(testnet: bool = False) -> 'ccxt.Exchange':
    """
    Initialize Binance exchange connection.
    
    Args:
        testnet: Whether to use testnet (for testing)
        
    Returns:
        CCXT Binance exchange instance
    """
    if not CCXT_AVAILABLE:
        raise ImportError("ccxt is required. Install with: pip install ccxt")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Use futures for OI and funding
        }
    })
    
    if testnet:
        exchange.set_sandbox_mode(True)
    
    return exchange


def load_ohlcv(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1d',
    limit: int = 1000,
    since: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe ('1h', '1d', etc.)
        limit: Number of candles to fetch (max 1000 per request)
        since: Start datetime (None = most recent)
        
    Returns:
        DataFrame with OHLCV columns
    """
    exchange = get_binance_exchange()
    
    since_ts = None
    if since:
        since_ts = int(since.timestamp() * 1000)
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
    except Exception as e:
        print(f"Error fetching OHLCV: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df


def load_funding_rates(
    symbol: str = 'BTC/USDT',
    limit: int = 500,
    since: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load historical funding rates from Binance Futures.
    
    Funding rates indicate market sentiment:
    - Positive: Longs pay shorts (bullish overcrowding)
    - Negative: Shorts pay longs (bearish overcrowding)
    
    Args:
        symbol: Trading pair
        limit: Number of funding periods to fetch
        since: Start datetime
        
    Returns:
        DataFrame with funding rate history
    """
    exchange = get_binance_exchange()
    
    since_ts = None
    if since:
        since_ts = int(since.timestamp() * 1000)
    
    try:
        # Fetch funding rate history
        funding = exchange.fetch_funding_rate_history(symbol, since=since_ts, limit=limit)
        
        if not funding:
            return pd.DataFrame()
        
        df = pd.DataFrame(funding)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        
        return df
    except Exception as e:
        print(f"Warning: Could not fetch funding rates: {e}")
        return pd.DataFrame()


def load_open_interest(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1d',
    limit: int = 500
) -> pd.DataFrame:
    """
    Load Open Interest history from Binance Futures.
    
    Open Interest tracks total outstanding contracts:
    - Rising OI + Rising Price = New longs entering (trend strength)
    - Rising OI + Falling Price = New shorts entering
    - Falling OI = Positions closing (trend exhaustion)
    
    Args:
        symbol: Trading pair
        timeframe: Timeframe for OI aggregation
        limit: Number of periods to fetch
        
    Returns:
        DataFrame with open interest history
    """
    exchange = get_binance_exchange()
    
    # Convert symbol format for OI endpoint
    market = exchange.market(symbol)
    binance_symbol = market['id']  # e.g., 'BTCUSDT'
    
    try:
        # Use Binance's specific OI history endpoint
        oi_data = exchange.fapiPublic_get_openinteresthist({
            'symbol': binance_symbol,
            'period': timeframe,
            'limit': limit
        })
        
        df = pd.DataFrame(oi_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['open_interest'] = df['sumOpenInterest'].astype(float)
        df['oi_value'] = df['sumOpenInterestValue'].astype(float)
        
        return df[['open_interest', 'oi_value']]
        
    except Exception as e:
        print(f"Warning: Could not fetch OI history: {e}")
        return pd.DataFrame()


def compute_cvd(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Compute Cumulative Volume Delta from OHLCV data.
    
    Approximates buy/sell volume using candle structure:
    - Green candle (close > open): Assume more buying
    - Red candle (close < open): Assume more selling
    
    CVD = cumsum((close - open) / (high - low) * volume)
    
    Args:
        ohlcv: DataFrame with OHLCV columns
        
    Returns:
        CVD series
    """
    # Avoid division by zero
    range_pct = (ohlcv['high'] - ohlcv['low']).replace(0, np.nan)
    close_position = (ohlcv['close'] - ohlcv['open']) / range_pct
    
    # Fill NaN with 0 (doji candles)
    close_position = close_position.fillna(0)
    
    # Delta per candle
    volume_delta = close_position * ohlcv['volume']
    
    # Cumulative
    cvd = volume_delta.cumsum()
    
    return cvd


def load_binance_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1d',
    lookback_days: int = 365,
    include_oi: bool = True,
    include_funding: bool = True
) -> pd.DataFrame:
    """
    Load comprehensive Binance data including OI and Funding Rates.
    
    This provides the "Z-axis" data needed for multi-dimensional paths.
    
    Args:
        symbol: Trading pair
        timeframe: Candle timeframe
        lookback_days: Days of history to fetch
        include_oi: Whether to include Open Interest
        include_funding: Whether to include Funding Rates
        
    Returns:
        DataFrame with all available columns merged
    """
    # Calculate how many candles we need
    if timeframe == '1d':
        limit = lookback_days
    elif timeframe == '1h':
        limit = min(lookback_days * 24, 1000)
    else:
        limit = 1000
    
    since = datetime.now() - timedelta(days=lookback_days)
    
    # Load OHLCV
    print(f"Loading OHLCV for {symbol}...")
    df = load_ohlcv(symbol, timeframe, limit, since)
    
    if df.empty:
        raise ValueError("No OHLCV data returned from Binance")
    
    # Compute CVD
    df['cvd'] = compute_cvd(df)
    df['cvd_normalized'] = (df['cvd'] - df['cvd'].mean()) / df['cvd'].std()
    
    # Load Open Interest
    if include_oi:
        print("Loading Open Interest...")
        try:
            oi_df = load_open_interest(symbol, timeframe, limit)
            if not oi_df.empty:
                df = df.join(oi_df, how='left')
                
                if 'open_interest' in df.columns:
                    # Check if empty or entirely nan
                    if df['open_interest'].isna().all():
                        print("Warning: Retrieved OI data is empty. Using Hybrid Mode fallback (No OI).")
                        df.drop(columns=['open_interest', 'oi_value'], errors='ignore', inplace=True)
                    else:
                        # Forward fill small gaps
                        df['open_interest'] = df['open_interest'].ffill()
                        
                        # Fill initial NaNs with mean
                        oi_mean = df['open_interest'].mean()
                        df['open_interest'] = df['open_interest'].fillna(oi_mean)
                        
                        df['oi_normalized'] = (df['open_interest'] - df['open_interest'].mean()) / df['open_interest'].std()
            else:
                 print("Warning: No Open Interest data returned. Using Hybrid Mode fallback.")
        except Exception as e:
            print(f"Error loading Open Interest: {e}. Using Hybrid Mode fallback.")
    
    # Load Funding Rates
    if include_funding:
        print("Loading Funding Rates...")
        try:
            funding_df = load_funding_rates(symbol, limit)
            if not funding_df.empty:
                # Resample funding to match OHLCV timeframe (funding is every 8h)
                funding_resampled = funding_df.resample(timeframe).last()
                df = df.join(funding_resampled, how='left')
                
                if 'funding_rate' in df.columns:
                    if df['funding_rate'].isna().all():
                        print("Warning: Retrieved Funding data is empty. Using Hybrid Mode fallback.")
                        df.drop(columns=['funding_rate'], errors='ignore', inplace=True)
                    else:
                        df['funding_rate'] = df['funding_rate'].ffill()
                        # Fill initial NaNs with 0
                        df['funding_rate'] = df['funding_rate'].fillna(0.0)
            else:
                print("Warning: No Funding Rate data returned. Using Hybrid Mode fallback.")
        except Exception as e:
             print(f"Error loading Funding Rates: {e}. Using Hybrid Mode fallback.")
    
    # Forward fill any remaining missing values in OHLCV
    df = df.ffill()
    
    return df


if __name__ == "__main__":
    if not CCXT_AVAILABLE:
        print("CCXT not available. Install with: pip install ccxt")
    else:
        print("Testing Binance Data Loader")
        print("=" * 50)
        
        # Load sample data
        try:
            df = load_binance_data(
                symbol='BTC/USDT',
                timeframe='1d',
                lookback_days=30,
                include_oi=True,
                include_funding=True
            )
            
            print(f"\nLoaded {len(df)} records")
            print(f"Columns: {list(df.columns)}")
            
            if 'open_interest' not in df.columns:
                 print("Result: Hybrid Mode Active (OI missing)")
            else:
                 print("Result: Full Dimensionality (OI present)")
            
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
