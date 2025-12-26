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
LOOKBACK_DAYS = 365 * 3

# For backward compatibility
INTERVAL = TIMEFRAME

# ==================================
# Fractional Differentiation
# ==================================
FRACDIFF_D_RANGE = (0.3, 0.7)  # Search range for optimal d
FRACDIFF_ADF_PVALUE = 0.05     # Target p-value for stationarity
FRACDIFF_THRESHOLD = 1e-5      # FFD weight threshold

# ==================================
# Feature engineering parameters
# ==================================
# Window size in CANDLES
# For hourly: 24 candles = 1 day of lookback
# For daily: 20 candles = ~1 month of lookback
WINDOW_SIZE = 24 if TIMEFRAME == "1h" else 20

SIGNATURE_DEGREE = 4  # Truncation level for path signature

# Path dimensions (updated for multi-dimensional paths)
# 2D: (Price, Time) - Original approach
# 3D: (Price, CVD, Time)
# 4D: (Price, CVD, OI, FundingRate) - Full approach
PATH_DIMENSIONS = 4

# Signature dimensions for different path dimensions at degree 4:
# 2D: 2 + 4 + 8 + 16 = 30
# 3D: 3 + 9 + 27 + 81 = 120
# 4D: 4 + 16 + 64 + 256 = 340
def calculate_sig_dim(d: int, degree: int) -> int:
    """Calculate signature dimension for d-dimensional path at given degree."""
    return sum(d**k for k in range(1, degree + 1))

SIGNATURE_DIMENSIONS = calculate_sig_dim(PATH_DIMENSIONS, SIGNATURE_DEGREE)

# ==================================
# Triple Barrier Labeling
# ==================================
BARRIER_UPPER_MULT = 2.0   # Upper barrier: 2x volatility (profit take)
BARRIER_LOWER_MULT = 1.0   # Lower barrier: 1x volatility (stop loss)
BARRIER_TIME_DAYS = 5      # Time barrier: 5 days max holding
VOLATILITY_WINDOW = 20     # Rolling window for volatility calculation

# ==================================
# Regime Detection (Hurst Exponent)
# ==================================
HURST_WINDOW = 30                    # Rolling window for Hurst calculation
HURST_TRENDING_THRESHOLD = 0.55      # H above this = trending (trade allowed)
HURST_MEAN_REVERT_THRESHOLD = 0.45   # H below this = mean-reverting (no trade)

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
# Random seed for reproducibility
# ==================================
RANDOM_SEED = 42
