"""
Configuration constants for Geometric Manifold Ensemble trading strategy.
Updated for classification-based approach with multi-dimensional paths.
"""

# Data parameters
TICKER = "BTC-USD"
SYMBOL = "BTC/USDT"  # For Binance/CCXT

# Timeframe options: "1h" (hourly) or "1d" (daily)
TIMEFRAME = "1d"

# Lookback period (adaptive based on timeframe)
# Daily: 3 years = ~1,095 candles
LOOKBACK_DAYS = 365 * 5

# For backward compatibility
INTERVAL = TIMEFRAME

# ==================================
# Fractional Differentiation
# ==================================
FRACDIFF_D_RANGE = (0.3, 0.7)  # Search range for optimal d
FRACDIFF_ADF_PVALUE = 0.05     # Target p-value for stationarity
FRACDIFF_THRESHOLD = 1e-4      # FFD weight threshold

# ==================================
# Feature engineering parameters
# ==================================
# Window size in CANDLES
# For hourly: 24 candles = 1 day of lookback
# For daily: 14 candles = ~2 weeks of lookback
WINDOW_SIZE = 14 if TIMEFRAME == "1d" else 24

# Multi-Scale Signature Windows (Fractal Feature Set)
# Captures market geometry at multiple timescales
MULTI_SCALE_WINDOWS = [6, 14, 28]  # Short, Medium, Long windows

SIGNATURE_DEGREE = 4  # Truncation level for path signature

# Path dimensions (updated for 5D paths with liquidations)
# 1D: Price only
# 4D: (Price, CVD, OI, FundingRate) - Standard approach
# 5D: (Price, CVD, OI, FundingRate, NetLiquidations) - Full microstructure
PATH_DIMENSIONS = 5

# Signature dimensions calculation
# For D-dimensional path with lead-lag doubling (2D dimensions) at degree 4:
# 4D: 8 + 64 + 512 + 4096 = 4680
# 5D: 10 + 100 + 1000 + 10000 = 11110
def calculate_sig_dim(d: int, degree: int, use_lead_lag: bool = True) -> int:
    """Calculate signature dimension for d-dimensional path at given degree.
    
    Args:
        d: Number of input dimensions
        degree: Signature truncation degree
        use_lead_lag: Whether lead-lag transform is applied (doubles dimensions)
    
    Returns:
        Total signature dimension
    """
    if use_lead_lag:
        d = d * 2  # Lead-lag doubles path dimension
    return sum(d**k for k in range(1, degree + 1))

# Single-scale signature dimension (for reference)
SIGNATURE_DIMENSIONS = calculate_sig_dim(PATH_DIMENSIONS, SIGNATURE_DEGREE)

# Multi-scale total dimension (before PCA)
MULTI_SCALE_SIG_DIMENSIONS = SIGNATURE_DIMENSIONS * len(MULTI_SCALE_WINDOWS)

# ==================================
# Triple Barrier Labeling (Tuned Parameters)
# ==================================
BARRIER_UPPER_MULT = 1.5   # Upper barrier: 1.5x volatility (optimized from tuning)
BARRIER_LOWER_MULT = 1.0   # Lower barrier: 1x volatility (stop loss)
BARRIER_TIME_DAYS = 3      # Time barrier: 3 days max holding (optimized from tuning)
VOLATILITY_WINDOW = 20     # Rolling window for volatility calculation

# ==================================
# Regime Detection (Hurst Exponent)
# ==================================
HURST_WINDOW = 100                 # Rolling window for Hurst calculation (needs >50 for DFA)
HURST_TRENDING_THRESHOLD = 0.55      # H above this = trending (trade allowed)
HURST_MEAN_REVERT_THRESHOLD = 0.45   # H below this = mean-reverting (no trade)
DFA_ALPHA_THRESHOLD = 1.45    # DFA Alpha threshold (optimized)

# ==================================
# Model parameters
# ==================================
# Logistic Regression (classifier)
LOGISTIC_C = 1.0

# LightGBM Classifier
LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42
}

# Extra Trees Classifier
EXTRA_TREES_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "n_jobs": -1,
    "random_state": 42
}

# XGBoost Meta-Labeler
META_LABELER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42
}

# ==================================
# Validation parameters
# ==================================
N_SPLITS = 5
PURGE_WINDOW = WINDOW_SIZE  # Must match signature window to avoid leakage

# ==================================
# Trading parameters (DEPRECATED - now using Hurst filter)
# ==================================
CONFIDENCE_THRESHOLD_PERCENTILE = 70  # Kept for backward compatibility

# ==================================
# Position Sizing (Kelly-Scale)
# ==================================
KELLY_FRACTION = 0.5  # Half-Kelly for conservative sizing

# ==================================
# Structural Persistence (Barrier Extension)
# ==================================
PERSISTENCE_ALPHA_THRESHOLD = 1.45   # Extend barriers when Alpha > this
PERSISTENCE_EXTENSION_CANDLES = 5    # Max additional candles to extend (up to another max_holding_period)

# ==================================
# Random seed for reproducibility
# ==================================
RANDOM_SEED = 42
