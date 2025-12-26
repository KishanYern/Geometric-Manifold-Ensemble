# Geometric Manifold Ensemble (GME)

A sophisticated institutional-grade trading strategy that leverages path signatures from Rough Path Theory to capture the geometric properties of market dynamics. This project implements a comprehensive pipeline including fractional differentiation, multi-dimensional feature engineering, ensemble classification, and a robust validation framework ensuring statistical significance.

## 🚀 Key Features

### 1. Advanced Feature Engineering
- **Path Signatures**: Uses 4th-degree truncated path signatures to capture non-linear dependencies in 4D market paths (Price, CVD, Open Interest, Funding Rate).
- **Lead-Lag Transformation**: Augments time-series data to capture causality structure.
- **Fractional Differentiation**: Preserves memory while achieving stationarity (unlike standard differencing).

### 2. Ensemble Strategy
- **Base Classifier Ensemble**: Combines Logistic Regression (linear), LightGBM (gradient boosting), and Extra Trees (bagging) to predict event directions (Long/Short/Flat).
- **Triple Barrier Labeling**: Labels events based on profit-taking, stop-loss, and time limits.

### 3. Risk Management & Filtering
- **Hurst Exponent Regime Filter**: Only trades in trending regimes ($H > 0.55$) to avoid "chop".
- **Meta-Labeling**: A secondary XGBoost model that learns *when* the primary model is likely to be wrong, acting as a smart trade gate.

### 4. Robust Validation Suite
- **Combinatorial Purged Cross-Validation (CPCV)**: Simulates thousands of alternative historical paths to prevent overfitting.
- **Deflated Sharpe Ratio (DSR)**: Adjusts performance metrics for the number of trials attempted (multiple testing correction).
- **Synthetic Stress Testing**: Tests strategy resilience on 10,000 generated paths (GBM and OU processes).
- **Sensitivity Analysis**: "Slippage-Sensitivity Sweep" to determine the strategy's capacity threshold.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/KishanYern/Geometric-Manifold-Ensemble.git
cd Geometric-Manifold-Ensemble

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Ensure scipy and xgboost are installed
pip install scipy xgboost
```

## 📉 Usage

### Running Validation
Execute the comprehensive validation suite to verify strategy robustness:
```bash
python validation.py
```

### Stress Testing
Run synthetic stress tests and sensitivity analysis:
```bash
python stress_test.py
```

### Backtesting
Run the historical backtest logic:
```bash
python backtest.py
```

## 📚 References
- *Advances in Financial Machine Learning*, Marcos Lopez de Prado.
- *Machine Learning for Asset Managers*, Marcos Lopez de Prado.
