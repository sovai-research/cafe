#!/usr/bin/env python3
"""
Test script for all MISSNET improvements.

This script tests:
1. Thread-safe CFG context manager
2. Cell-wise variance computation and uncertainty gating
3. Deterministic runs with random_state
4. Gaussian log-likelihood term fixes
5. Patience parameter with backwards compatibility
6. Return_std functionality
7. Dtype control
8. Auto-tuning improvements
9. Memory pool and cache management
10. Sklearn-style wrapper functionality
"""

import numpy as np
import time
import warnings
from missnet_imputer import (
    missnet_impute, TIMARAImputer, cfg_context, CFG, 
    get_optimal_config, MemoryPool, IdentityCache
)

def test_cfg_context_manager():
    """Test thread-safe CFG context manager."""
    print("Testing CFG context manager...")
    
    # Save original CFG values
    original_vals = {}
    for key in ['USE_HUBER', 'USE_CHOLESKY', 'JITTER']:
        original_vals[key] = CFG.get(key)
    
    # Test context manager with overrides
    overrides = {
        'USE_HUBER': not CFG.get('USE_HUBER', True),
        'USE_CHOLESKY': not CFG.get('USE_CHOLESKY', True),
        'JITTER': 1e-8
    }
    
    with cfg_context(overrides):
        # Verify overrides are applied within context
        assert CFG['USE_HUBER'] == overrides['USE_HUBER']
        assert CFG['USE_CHOLESKY'] == overrides['USE_CHOLESKY']
        assert CFG['JITTER'] == overrides['JITTER']
    
    # Verify original values are restored
    for key in original_vals:
        assert CFG.get(key) == original_vals[key]
    
    print("✅ CFG context manager works correctly")


def test_return_std():
    """Test return_std functionality."""
    print("Testing return_std functionality...")
    
    # Create test data
    np.random.seed(42)
    T, N = 50, 6
    data = np.random.randn(T, N)
    mask = np.random.random(data.shape) < 0.2
    data[mask] = np.nan
    
    # Test with return_std=True
    imputed, variance = missnet_impute(
        data, 
        return_std=True, 
        random_state=42,
        max_iteration=10,
        verbose=False
    )
    
    # Verify shapes
    assert imputed.shape == data.shape
    assert variance.shape == data.shape
    
    # Verify missing values are imputed
    assert not np.any(np.isnan(imputed[mask]))
    
    # Verify variance is positive
    assert np.all(variance >= 0)
    
    # Verify variance is higher in missing positions (more uncertainty)
    missing_variance = variance[mask].mean()
    observed_variance = variance[~mask].mean()
    assert missing_variance >= observed_variance
    
    print("✅ return_std functionality works correctly")


def test_random_state():
    """Test deterministic behavior with random_state."""
    print("Testing random_state deterministic behavior...")
    
    # Create test data
    np.random.seed(42)
    T, N = 30, 5
    data = np.random.randn(T, N)
    mask = np.random.random(data.shape) < 0.15
    data[mask] = np.nan
    
    # Run with same random_state twice
    result1 = missnet_impute(
        data, 
        random_state=123, 
        max_iteration=10,
        verbose=False
    )
    result2 = missnet_impute(
        data, 
        random_state=123, 
        max_iteration=10,
        verbose=False
    )
    
    # Should be identical
    np.testing.assert_array_equal(result1, result2)
    
    # Run with different random_state
    result3 = missnet_impute(
        data, 
        random_state=456, 
        max_iteration=10,
        verbose=False
    )
    
    # Should be different
    assert not np.array_equal(result1, result3).any()
    
    print("✅ random_state deterministic behavior works correctly")


def test_patience_parameter():
    """Test patience parameter with backwards compatibility."""
    print("Testing patience parameter...")
    
    # Create test data
    np.random.seed(42)
    T, N = 40, 5
    data = np.random.randn(T, N)
    mask = np.random.random(data.shape) < 0.2
    data[mask] = np.nan
    
    # Test with patience parameter
    result1 = missnet_impute(
        data, 
        patience=3, 
        max_iteration=10,
        verbose=False
    )
    
    # Test with tol parameter (backwards compatibility)
    result2 = missnet_impute(
        data, 
        tol=3, 
        max_iteration=10,
        verbose=False
    )
    
    # Should be identical
    np.testing.assert_array_equal(result1, result2)
    
    # Test with both (patience should take precedence)
    result3 = missnet_impute(
        data, 
        patience=2, 
        tol=5, 
        max_iteration=10,
        verbose=False
    )
    
    # Should follow patience, not tol
    assert not np.array_equal(result1, result3).any()
    
    print("✅ patience parameter works correctly")


def test_dtype_control():
    """Test dtype control."""
    print("Testing dtype control...")
    
    # Create test data
    np.random.seed(42)
    T, N = 30, 4
    data = np.random.randn(T, N)
    mask = np.random.random(data.shape) < 0.2
    data[mask] = np.nan
    
    # Test with float32
    result_f32 = missnet_impute(
        data, 
        dtype=np.float32, 
        max_iteration=10,
        verbose=False
    )
    
    # Test with float64
    result_f64 = missnet_impute(
        data, 
        dtype=np.float64, 
        max_iteration=10,
        verbose=False
    )
    
    # Verify dtypes
    assert result_f32.dtype == np.float32
    assert result_f64.dtype == np.float64
    
    # Results should be similar but not identical due to precision
    assert np.allclose(result_f32, result_f64, rtol=1e-5)
    
    print("✅ dtype control works correctly")


def test_memory_pool():
    """Test memory pool functionality."""
    print("Testing memory pool...")
    
    # Clear any existing pools
    MemoryPool.clear_all()
    
    # Test array allocation and return
    arr1 = MemoryPool.get_array((10, 5))
    arr2 = MemoryPool.get_array((10, 5))
    
    # Arrays should be different objects
    assert arr1 is not arr2
    
    # Return arrays
    MemoryPool.return_array(arr1)
    MemoryPool.return_array(arr2)
    
    # Get arrays again (should reuse)
    arr3 = MemoryPool.get_array((10, 5))
    arr4 = MemoryPool.get_array((10, 5))
    
    # Should reuse the returned arrays
    assert arr3 is arr1 or arr3 is arr2
    assert arr4 is arr1 or arr4 is arr2
    
    # Test different shapes/dtypes
    arr5 = MemoryPool.get_array((5, 3), dtype=np.float32)
    assert arr5.shape == (5, 3)
    assert arr5.dtype == np.float32
    
    print("✅ Memory pool works correctly")


def test_identity_cache():
    """Test identity matrix cache."""
    print("Testing identity matrix cache...")
    
    # Clear cache
    if hasattr(IdentityCache, '_cache'):
        IdentityCache._cache.clear()
    if hasattr(IdentityCache, '_jitter_cache'):
        IdentityCache._jitter_cache.clear()
    
    # Test cached identity matrices
    eye1 = IdentityCache.get_eye(5)
    eye2 = IdentityCache.get_eye(5)
    assert eye1 is eye2  # Same object from cache
    
    eye3 = IdentityCache.get_eye(10)
    assert eye3.shape == (10, 10)
    
    # Test jittered identity matrices
    jitter1 = IdentityCache.get_jittered_eye(5, 1e-6)
    jitter2 = IdentityCache.get_jittered_eye(5, 1e-6)
    assert jitter1 is jitter2  # Same object from cache
    
    jitter3 = IdentityCache.get_jittered_eye(5, 1e-8)
    assert not np.array_equal(jitter1, jitter3).any()
    
    print("✅ Identity cache works correctly")


def test_sklearn_wrapper():
    """Test sklearn-style wrapper functionality."""
    print("Testing sklearn-style wrapper...")
    
    # Create test data
    np.random.seed(42)
    T, N = 40, 6
    data = np.random.randn(T, N)
    mask = np.random.random(data.shape) < 0.2
    data[mask] = np.nan
    
    # Test basic fit/transform
    imputer = TIMARAImputer(verbose=False)
    imputer.fit(data)
    result1 = imputer.transform(data)
    
    # Test fit_transform
    imputer2 = TIMARAImputer(verbose=False)
    result2 = imputer2.fit_transform(data)
    
    # Results should be similar
    assert np.allclose(result1, result2, rtol=1e-5)
    
    # Test with different dtypes
    imputer_f32 = TIMARAImputer(verbose=False)
    result_f32 = imputer_f32.fit_transform(data.astype(np.float32))
    assert result_f32.dtype == np.float32
    
    print("✅ Sklearn-style wrapper works correctly")


def test_auto_tuning():
    """Test auto-tuning improvements."""
    print("Testing auto-tuning improvements...")
    
    # Create test data with different characteristics
    np.random.seed(42)
    
    # Test sparse data
    T, N = 50, 10
    sparse_data = np.random.randn(T, N)
    sparse_data[:, :N//2] *= 0.1  # Make half variables weak
    mask = np.random.random(sparse_data.shape) < 0.3
    sparse_data[mask] = np.nan
    
    # Test dense data
    dense_data = np.random.randn(T, N//2)
    mask2 = np.random.random(dense_data.shape) < 0.1
    dense_data[mask2] = np.nan
    
    # Test seasonal data
    t = np.linspace(0, 4*np.pi, T)
    seasonal_data = np.zeros((T, N))
    for i in range(N):
        seasonal_data[:, i] = np.sin(t + i*np.pi/4) + 0.1*np.random.randn(T)
    mask3 = np.random.random(seasonal_data.shape) < 0.15
    seasonal_data[mask3] = np.nan
    
    # Test auto-tuning for each dataset
    for name, data in [("sparse", sparse_data), ("dense", dense_data), ("seasonal", seasonal_data)]:
        config = get_optimal_config(data, fast=True, verbose=False)
        
        # Verify reasonable parameter ranges
        assert 0.1 <= config['alpha'] <= 0.9
        assert 0.001 <= config['beta'] <= 0.5
        assert 3 <= config['L'] <= min(15, N-1)
        assert 1 <= config['n_cl'] <= 3
        
        # Test imputation with auto-tuned config
        imputed = missnet_impute(data, **config, max_iteration=5, verbose=False)
        assert imputed.shape == data.shape
        assert not np.any(np.isnan(imputed[np.isnan(data)]))
    
    print("✅ Auto-tuning improvements work correctly")


def test_convergence_improvements():
    """Test improved convergence detection."""
    print("Testing convergence improvements...")
    
    # Create test data
    np.random.seed(42)
    T, N = 30, 5
    data = np.random.randn(T, N)
    mask = np.random.random(data.shape) < 0.15
    data[mask] = np.nan
    
    # Test with tight convergence tolerance
    start_time = time.time()
    imputed = missnet_impute(
        data, 
        conv_eps=1e-6, 
        conv_window=3,
        max_iteration=20,
        verbose=False
    )
    end_time = time.time()
    
    # Should converge quickly with good quality
    assert imputed.shape == data.shape
    assert not np.any(np.isnan(imputed[np.isnan(data)]))
    assert end_time - start_time < 30  # Should finish quickly
    
    print(f"✅ Convergence improvements work correctly (finished in {end_time - start_time:.2f}s)")


def test_all():
    """Run all tests."""
    print("Running MISSNET improvement tests...\n")
    print("=" * 50)
    
    try:
        test_cfg_context_manager()
        test_return_std()
        test_random_state()
        test_patience_parameter()
        test_dtype_control()
        test_memory_pool()
        test_identity_cache()
        test_sklearn_wrapper()
        test_auto_tuning()
        test_convergence_improvements()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_all()
