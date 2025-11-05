# TIMARAImputer: sklearn-style Wrapper for MISSNET/TIMARA

This document describes the sklearn-style wrapper that has been added to `missnet_imputer.py` to provide a familiar scikit-learn API for the TIMARA imputation algorithm.

## Overview

The `TIMARAImputer` class provides a scikit-learn compatible interface for the MISSNET/TIMARA imputation algorithm, enabling:

- **Drop-in compatibility** with pandas and polars pipelines
- **Automatic hyperparameter tuning** based on data characteristics
- **Walk-forward (rolling) imputation** for time series backtesting
- **Container type preservation** (numpy, pandas, polars)
- **Familiar sklearn API** (fit, transform, fit_transform)

## Quick Start

```python
from missnet_imputer import TIMARAImputer
import pandas as pd
import numpy as np

# Create sample data with missing values
data = np.random.randn(100, 5)
data[np.random.random(data.shape) < 0.2] = np.nan  # 20% missing

# Basic sklearn-style usage
imputer = TIMARAImputer(verbose=True)
imputed = imputer.fit_transform(data)

# With pandas DataFrame (preserves index and columns)
df = pd.DataFrame(data, index=pd.date_range('2023-01-01', periods=100))
imputer = TIMARAImputer(output="pandas")
df_imputed = imputer.fit_transform(df)
```

## Key Features

### 1. Automatic Hyperparameter Tuning

The wrapper automatically analyzes your data and selects optimal parameters:

```python
imputer = TIMARAImputer(auto_tune=True, verbose=True)
imputer.fit(data_with_missing)

# Access the learned parameters
print(f"Alpha: {imputer.alpha:.3f}")  # Network/temporal trade-off
print(f"Beta: {imputer.beta:.4f}")    # Sparsity regularization
print(f"L: {imputer.L}")              # Latent dimension
print(f"n_cl: {imputer.n_cl}")        # Number of regimes
```

### 2. Container Type Support

Seamlessly works with NumPy, pandas, and polars:

```python
# NumPy arrays (default)
imputer = TIMARAImputer(output="array")
numpy_result = imputer.fit_transform(data)

# pandas DataFrames (preserves index/columns)
imputer = TIMARAImputer(output="pandas")
pandas_result = imputer.fit_transform(df)

# polars DataFrames (preserves schema)
import polars as pl
imputer = TIMARAImputer(output="polars")
polars_result = imputer.fit_transform(pl_df)
```

### 3. Walk-Forward (Rolling) Imputation

Perfect for time series backtesting and cross-validation:

```python
# Row-based windows
imputer = TIMARAImputer()
walk_result = imputer.walk_transform(
    data, 
    window=30,      # 30-row window
    step=7,         # Step forward by 7 rows
    only_last=True  # Only impute the last row of each window
)

# Time-based windows (requires datetime index or time column)
imputer = TIMARAImputer(time_col="timestamp")
walk_result = imputer.walk_transform(
    df, 
    window="7D",    # 7-day window
    step="1D",     # 1-day step
    only_last=True
)
```

### 4. Sklearn Pipeline Compatibility

Works seamlessly with sklearn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('imputer', TIMARAImputer(verbose=False)),
    ('scaler', StandardScaler())
])

X_processed = pipeline.fit_transform(data_with_missing)
```

## API Reference

### TIMARAImputer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_tune` | bool | True | Automatically select optimal hyperparameters |
| `alpha` | float | None | Network/temporal trade-off (auto-tuned if None) |
| `beta` | float | None | Sparsity regularization (auto-tuned if None) |
| `L` | int | None | Latent dimension (auto-tuned if None) |
| `n_cl` | int | None | Number of regimes (auto-tuned if None) |
| `max_iteration` | int | None | Max iterations (auto-tuned if None) |
| `verbose` | bool | False | Print progress information |
| `time_col` | str | None | Name of datetime column (for time-based windows) |
| `output` | str | "like" | Output format: "like", "array", "pandas", "polars" |

### Methods

#### `fit(X, y=None)`
Auto-tune hyperparameters and learn from data.

**Parameters:**
- `X`: Input data (numpy array, pandas DataFrame, or polars DataFrame)
- `y`: Ignored (included for sklearn compatibility)

**Returns:** Self (fitted imputer)

#### `transform(X)`
Impute missing values using learned hyperparameters.

**Parameters:**
- `X`: Input data to impute

**Returns:** Imputed data in specified output format

#### `fit_transform(X, y=None)`
Convenience method combining fit and transform.

#### `walk_transform(X, window, step=None, tune_each_window=False, only_last=True, aggregate="last")`
Perform rolling/walk-forward imputation.

**Parameters:**
- `X`: Input data
- `window`: Window size (int rows or time string like '7D')
- `step`: Step size (defaults to window)
- `tune_each_window`: Whether to re-tune per window
- `only_last`: Only impute last row of each window
- `aggregate`: How to combine overlapping windows ('last', 'mean', 'median')

## Advanced Usage

### Custom Hyperparameters

Override auto-tuning for specific parameters:

```python
imputer = TIMARAImputer(
    alpha=0.5,        # Fixed alpha
    beta=None,        # Auto-tune beta
    L=10,             # Fixed latent dimension
    auto_tune=True    # Still auto-tune other params
)
```

### Time-Based Windows

For time series with datetime index:

```python
# pandas DataFrame with DatetimeIndex
df.index = pd.date_range('2023-01-01', periods=len(df))

imputer = TIMARAImputer()
result = imputer.walk_transform(df, window="30D", step="7D")
```

### Performance Considerations

- Use `tune_each_window=False` for faster walk-forward imputation
- Smaller windows are faster but may be less accurate
- Consider `max_iteration` for faster convergence during testing

## Examples

### Basic Imputation

```python
from missnet_imputer import TIMARAImputer
import numpy as np

# Create data with missing values
X = np.random.randn(200, 8)
X[np.random.random(X.shape) < 0.25] = np.nan

# Impute with auto-tuning
imputer = TIMARAImputer(verbose=True)
X_imputed = imputer.fit_transform(X)

print(f"Original missing: {np.isnan(X).sum()}")
print(f"After imputation: {np.isnan(X_imputed).sum()}")
```

### Time Series Backtesting

```python
import pandas as pd

# Create time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = np.random.randn(365, 5).cumsum(axis=0)
data[np.random.random(data.shape) < 0.2] = np.nan

df = pd.DataFrame(data, index=dates)

# Walk-forward imputation for backtesting
imputer = TIMARAImputer(verbose=False)
backtest_results = imputer.walk_transform(
    df, 
    window="30D",     # Monthly windows
    step="7D",        # Weekly steps
    only_last=True    # Only predict the last day
)
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Create imputation + modeling pipeline
pipeline = Pipeline([
    ('imputer', TIMARAImputer(verbose=False)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

# Fit on data with missing values
X = data_with_missing[:, :-1]  # Features
y = data_with_missing[:, -1]   # Target

pipeline.fit(X, y)
predictions = pipeline.predict(X_new)
```

## Comparison with Original Function

The sklearn wrapper provides the same core functionality as `missnet_impute()` but with additional conveniences:

```python
# Original function
from missnet_imputer import missnet_impute
result1 = missnet_impute(data, verbose=True)

# sklearn wrapper (equivalent)
imputer = TIMARAImputer(verbose=True, output="array")
result2 = imputer.fit_transform(data)

# Results should be nearly identical
print(f"Difference: {np.max(np.abs(result1 - result2)):.2e}")
```

## Troubleshooting

### Common Issues

1. **Convergence warnings**: Small windows may not converge - try larger windows or reduce `max_iteration`
2. **Memory issues**: Large datasets may require reducing `L` or `n_cl`
3. **Slow performance**: Disable `auto_tune` for repeated transforms or use smaller windows

### Performance Tips

- Use `verbose=False` in production
- For repeated imputation, fit once and call `transform()` multiple times
- Consider smaller windows for walk-forward imputation with large datasets

## Dependencies

The wrapper gracefully handles missing dependencies:

- **pandas**: Optional (for DataFrame support)
- **polars**: Optional (for polars DataFrame support)  
- **sklearn**: Optional (for BaseEstimator/TransformerMixin)

If unavailable, the wrapper provides minimal shims for basic functionality.
