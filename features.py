"""
Feature engineering module for GME trading strategy.
Implements multi-dimensional Lead-Lag transformation, path signature computation,
and RobustScaler + PCA preprocessing.

Updated for:
- 4D paths: (Price_fracdiff, CVD, OI, FundingRate)
- Classification targets via Triple Barrier labeling
- Fractional differentiation for memory-preserving stationarity
- Robust Scaling and PCA
"""

import numpy as np
import pandas as pd
import iisignature
from typing import Tuple, Optional, Any
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import config


class SignaturePreprocessor:
    """
    Preprocessor for signature features involving Robust Scaling and PCA.
    
    Ensures that scaling and dimensionality reduction are fitted 
    strictly on training data to prevent lookahead bias.
    """
    
    def __init__(self, variance_threshold: float = 0.95):
        """
        Initialize preprocessor.
        
        Args:
            variance_threshold: PCA explained variance ratio threshold.
        """
        self.variance_threshold = variance_threshold
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=variance_threshold)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'SignaturePreprocessor':
        """
        Fit scaler and PCA on training data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Optional target (unused)
            
        Returns:
            self
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        # Check if n_samples < n_features, PCA might complain if we want high variance
        # but have few samples.
        n_samples, n_features = X.shape
        if n_samples > n_features:
             self.pca.fit(X_scaled)
        else:
            # Fallback for very small datasets (like unit tests)
            # Use min(n_samples, n_features) components or simple reduction
            n_components = min(n_samples, n_features)
            if n_components > 1:
                self.pca = PCA(n_components=n_components - 1)
            else:
                self.pca = PCA(n_components=1)
            self.pca.fit(X_scaled)
            
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scaling and PCA transformation.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Transform D-dimensional time series into 2D Lead-Lag path.
    
    For multi-dimensional input, we apply lead-lag to each dimension,
    creating a higher-dimensional path that captures cross-correlations.
    
    For a D-dimensional path of length n:
    - Creates a (2n-1, 2D) path
    - Each dimension gets its own lead and lag component
    
    Args:
        path: D-dimensional numpy array of shape (n,) or (n, D)
        
    Returns:
        2D numpy array of shape (2n-1, 2D) representing the lead-lag path
    """
    # Handle 1D input
    if path.ndim == 1:
        path = path.reshape(-1, 1)
    
    n, d = path.shape
    if n < 2:
        raise ValueError("Path must have at least 2 points")
    
    # Create lead-lag path with shape (2n-1, 2*d)
    lead_lag = np.zeros((2 * n - 1, 2 * d))
    
    for i in range(n):
        # Lead point: (x_i, x_{i-1}) for each dimension
        lead_lag[2 * i, :d] = path[i]  # Lead component
        lead_lag[2 * i, d:] = path[i] if i == 0 else path[i - 1]  # Lag component
        
        # Lag point: (x_i, x_i) - only for i < n-1
        if i < n - 1:
            lead_lag[2 * i + 1, :d] = path[i]
            lead_lag[2 * i + 1, d:] = path[i]
    
    return lead_lag


def compute_signature(path_nd: np.ndarray, degree: int = config.SIGNATURE_DEGREE) -> np.ndarray:
    """
    Compute truncated path signature using iisignature.
    
    The path signature is a sequence of iterated integrals that provides
    a faithful representation of the path. For D-dimensional path at degree d,
    the signature has dimension sum_{k=1}^{d} D^k.
    
    Args:
        path_nd: N-dimensional path of shape (n, N)
        degree: Truncation level for signature
        
    Returns:
        1D numpy array of signature features
    """
    # Prepare path for iisignature (needs to be contiguous)
    path = np.ascontiguousarray(path_nd, dtype=np.float64)
    
    # Compute signature
    sig = iisignature.sig(path, degree)
    
    return sig


def generate_signature_features(
    price_fracdiff: np.ndarray,
    cvd: Optional[np.ndarray] = None,
    oi: Optional[np.ndarray] = None,
    funding_rate: Optional[np.ndarray] = None,
    window_size: int = config.WINDOW_SIZE,
    degree: int = config.SIGNATURE_DEGREE,
    use_lead_lag: bool = True
) -> np.ndarray:
    """
    Generate signature features using sliding window approach.
    
    CRITICAL: Uses only past data [t-window:t] to predict t+1.
    This ensures ZERO lookahead bias.
    
    Args:
        price_fracdiff: Fractionally differentiated price (main signal)
        cvd: Cumulative Volume Delta (optional, 2nd dimension)
        oi: Open Interest normalized (optional, 3rd dimension)
        funding_rate: Funding rate (optional, 4th dimension)
        window_size: Size of sliding window
        degree: Signature truncation degree
        use_lead_lag: Whether to apply lead-lag transform
        
    Returns:
        2D array of shape (n_samples, signature_dim)
        Note: These are RAW features. Scaling and PCA should be applied downstream.
    """
    n = len(price_fracdiff)
    if n <= window_size:
        raise ValueError(f"Not enough data: {n} points, need > {window_size}")
    
    # Build multi-dimensional path
    dims = [np.array(price_fracdiff)]
    
    if cvd is not None:
        dims.append(np.array(cvd))
    if oi is not None:
        dims.append(np.array(oi))
    if funding_rate is not None:
        dims.append(np.array(funding_rate))
    
    # Stack dimensions
    multi_dim_data = np.column_stack(dims)  # (n, num_dims)
    num_dims = multi_dim_data.shape[1]
    
    # Calculate expected signature dimension
    if use_lead_lag:
        path_dim = 2 * num_dims  # Lead-lag doubles dimensions
    else:
        path_dim = num_dims
    sig_dim = sum(path_dim**k for k in range(1, degree + 1))
    
    # Number of samples we can generate
    n_samples = n - window_size
    
    # Pre-allocate arrays
    features = np.zeros((n_samples, sig_dim))
    
    for i in range(n_samples):
        # Window: [i, i + window_size) - strictly past data
        window_data = multi_dim_data[i:i + window_size]
        
        # Convert to cumulative path for each dimension
        path = np.cumsum(window_data, axis=0)
        
        # Apply Lead-Lag transformation
        if use_lead_lag:
            path = lead_lag_transform(path)
        
        # Compute signature
        sig = compute_signature(path, degree)
        
        features[i] = sig
    
    return features


def generate_classification_features(
    df: pd.DataFrame,
    price_col: str = 'price_fracdiff',
    cvd_col: str = 'cvd_normalized',
    oi_col: str = 'oi_normalized',
    funding_col: str = 'funding_rate',
    label_col: str = 'label',
    window_size: int = config.WINDOW_SIZE,
    degree: int = config.SIGNATURE_DEGREE
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Generate signature features and triple barrier labels for classification.
    
    This is the main entry point for the classification pipeline.
    
    Args:
        df: DataFrame with all required columns
        price_col: Column name for fractionally differentiated price
        cvd_col: Column name for CVD (optional)
        oi_col: Column name for OI (optional)
        funding_col: Column name for funding rate (optional)
        label_col: Column name for triple barrier labels
        window_size: Sliding window size
        degree: Signature truncation degree
        
    Returns:
        Tuple of (features, labels, timestamps)
    """
    # Extract available dimensions
    price = df[price_col].values if price_col in df.columns else None
    cvd = df[cvd_col].values if cvd_col in df.columns else None
    oi = df[oi_col].values if oi_col in df.columns else None
    funding = df[funding_col].values if funding_col in df.columns else None
    
    if price is None:
        raise ValueError(f"Required column '{price_col}' not found")
    
    # Generate signature features
    features = generate_signature_features(
        price_fracdiff=price,
        cvd=cvd,
        oi=oi,
        funding_rate=funding,
        window_size=window_size,
        degree=degree
    )
    
    # Align labels with features
    # Features[i] uses data [i:i+window_size], target is label at i+window_size
    if label_col in df.columns:
        labels = df[label_col].values[window_size:window_size + len(features)]
    else:
        labels = np.zeros(len(features))
    
    # Aligned timestamps
    timestamps = df.index[window_size:window_size + len(features)]
    
    return features, labels, timestamps


def get_signature_feature_names(
    num_dims: int = config.PATH_DIMENSIONS,
    degree: int = config.SIGNATURE_DEGREE,
    use_lead_lag: bool = True
) -> list:
    """
    Generate human-readable names for signature components.
    
    Args:
        num_dims: Number of input dimensions
        degree: Signature truncation degree
        use_lead_lag: Whether lead-lag is applied
        
    Returns:
        List of feature names
    """
    if use_lead_lag:
        num_dims *= 2
    
    # Create dimension labels
    dim_labels = [f"d{i}" for i in range(num_dims)]
    
    names = []
    
    # Generate names for each level
    def generate_level(level, prefix=""):
        if level == 0:
            names.append(f"sig_{prefix}") if prefix else None
            return
        for d in dim_labels:
            new_prefix = f"{prefix}_{d}" if prefix else d
            if level == 1:
                names.append(f"sig_{new_prefix}")
            else:
                generate_level(level - 1, new_prefix)
    
    for level in range(1, degree + 1):
        generate_level(level)
    
    return names


if __name__ == "__main__":
    # Test feature engineering
    np.random.seed(42)
    
    print("Testing Multi-Dimensional Feature Engineering & Preprocessing")
    print("=" * 60)
    
    # Simulate 4D data
    n_points = 200
    
    # Fractionally differentiated price (stationary)
    price_fracdiff = np.random.randn(n_points) * 0.01
    
    # CVD (cumulative volume delta)
    cvd = np.cumsum(np.random.randn(n_points) * 100)
    cvd_norm = (cvd - cvd.mean()) / cvd.std()
    
    # Open Interest (normalized)
    oi = np.cumsum(np.random.randn(n_points) * 50) + 1000
    oi_norm = (oi - oi.mean()) / oi.std()
    
    # Funding rate
    funding = np.random.randn(n_points) * 0.0001
    
    print(f"Input dimensions: 4 (price, CVD, OI, funding)")
    print(f"Window size: {config.WINDOW_SIZE}")
    print(f"Signature degree: {config.SIGNATURE_DEGREE}")
    
    # Test signature generation
    print("\nTesting signature computation (4D path)...")
    features = generate_signature_features(
        price_fracdiff=price_fracdiff,
        cvd=cvd_norm,
        oi=oi_norm,
        funding_rate=funding,
        window_size=config.WINDOW_SIZE,
        degree=config.SIGNATURE_DEGREE
    )
    
    print(f"  Features shape: {features.shape}")
    
    # Test Preprocessing (Scaler + PCA)
    print("\nTesting SignaturePreprocessor...")
    preprocessor = SignaturePreprocessor(variance_threshold=0.95)
    
    # Split train/test
    split = int(len(features) * 0.7)
    X_train = features[:split]
    X_test = features[split:]
    
    # Fit
    print(f"  Fitting on {len(X_train)} samples...")
    preprocessor.fit(X_train)
    
    # Transform
    X_train_pca = preprocessor.transform(X_train)
    X_test_pca = preprocessor.transform(X_test)
    
    print(f"  Original Dim: {X_train.shape[1]}")
    print(f"  Reduced Dim: {X_train_pca.shape[1]}")
    print(f"  PCA Components: {preprocessor.pca.n_components_}")
    
    # Verify no NaN/Inf
    print(f"\nData quality:")
    print(f"  NaN features: {np.isnan(X_test_pca).sum()}")
    print(f"  Inf features: {np.isinf(X_test_pca).sum()}")
    
    print("Done.")
