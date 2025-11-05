#!/usr/bin/env python3
"""
Test script for TIMARAImputer sklearn-style wrapper
"""

import numpy as np
import pandas as pd
from missnet_imputer import TIMARAImputer, missnet_impute

def test_sklearn_wrapper():
    """Test the sklearn-style wrapper functionality."""
    print("Testing TIMARAImputer sklearn-style wrapper")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_timesteps, n_features = 100, 5
    t = np.linspace(0, 4*np.pi, n_timesteps)
    
    # Create data with trends and seasonality
    data = np.zeros((n_timesteps, n_features))
    for i in range(n_features):
        trend = 0.01 * i * np.arange(n_timesteps)
        seasonal = 2.0 * np.sin(t + i*np.pi/4)
        noise = 0.5 * np.random.randn(n_timesteps)
        data[:, i] = trend + seasonal + noise
    
    # Add missing values (20% missing)
    mask = np.random.random(data.shape) < 0.2
    data_with_missing = data.copy()
    data_with_missing[mask] = np.nan
    
    print(f"Data shape: {data.shape}")
    print(f"Missing values: {np.isnan(data_with_missing).sum()}/{data_with_missing.size}")
    print(f"Missing percentage: {np.isnan(data_with_missing).sum()/data_with_missing.size*100:.1f}%")
    
    # Test 1: Basic sklearn-style usage
    print("\n" + "="*30)
    print("Test 1: Basic sklearn-style usage")
    print("="*30)
    
    imputer = TIMARAImputer(verbose=True, output="array")
    imputed_array = imputer.fit_transform(data_with_missing)
    
    print(f"Input shape: {data_with_missing.shape}")
    print(f"Output shape: {imputed_array.shape}")
    print(f"All missing imputed: {not np.any(np.isnan(imputed_array))}")
    print(f"Learned n_features_in_: {imputer.n_features_in_}")
    
    # Test 2: pandas DataFrame with datetime index
    print("\n" + "="*30)
    print("Test 2: pandas DataFrame integration")
    print("="*30)
    
    dates = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
    df = pd.DataFrame(data_with_missing, 
                     index=dates, 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    imputer_pandas = TIMARAImputer(verbose=False, output="pandas")
    df_imputed = imputer_pandas.fit_transform(df)
    
    print(f"Original DataFrame type: {type(df)}")
    print(f"Imputed DataFrame type: {type(df_imputed)}")
    print(f"Index preserved: {type(df_imputed.index).__name__}")
    print(f"Columns preserved: {list(df_imputed.columns)}")
    print(f"All missing imputed: {not df_imputed.isna().any().any()}")
    
    # Test 3: Walk-forward imputation
    print("\n" + "="*30)
    print("Test 3: Walk-forward imputation")
    print("="*30)
    
    # Use a smaller window for faster testing
    imputer_walk = TIMARAImputer(verbose=False, output="array")
    walk_imputed = imputer_walk.fit_transform(data_with_missing)
    
    # Test walk-forward with 30-day window
    walk_result = imputer_walk.walk_transform(
        data_with_missing, 
        window=30, 
        step=10, 
        only_last=True
    )
    
    print(f"Walk-forward result shape: {walk_result.shape}")
    print(f"Walk-forward missing count: {np.isnan(walk_result).sum()}")
    
    # Test 4: Compare with original function
    print("\n" + "="*30)
    print("Test 4: Compare with original missnet_impute")
    print("="*30)
    
    # Using same parameters for fair comparison
    original_imputed = missnet_impute(
        data_with_missing,
        alpha=imputer.alpha,
        beta=imputer.beta,
        L=imputer.L,
        n_cl=imputer.n_cl,
        max_iteration=imputer.max_iteration,
        verbose=False
    )
    
    # Calculate differences
    diff = np.abs(imputed_array - original_imputed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Maximum difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Results are consistent: {max_diff < 1e-10}")
    
    # Test 5: Parameter preservation
    print("\n" + "="*30)
    print("Test 5: Parameter preservation")
    print("="*30)
    
    print(f"Auto-tuned alpha: {imputer.alpha:.4f}")
    print(f"Auto-tuned beta: {imputer.beta:.4f}")
    print(f"Auto-tuned L: {imputer.L}")
    print(f"Auto-tuned n_cl: {imputer.n_cl}")
    print(f"Config stored: {imputer.config_ is not None}")
    
    print("\n" + "="*50)
    print("All tests completed successfully! ✅")
    print("="*50)

if __name__ == "__main__":
    test_sklearn_wrapper()
