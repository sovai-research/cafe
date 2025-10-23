#!/usr/bin/env python3
"""
Comprehensive Test Suite for MISSNET Imputer on Synthetic Missing Noise Data

This script creates various synthetic datasets with different characteristics:
- Different missing data patterns (MCAR, MAR, MNAR)
- Different noise levels and distributions
- Different temporal patterns (seasonality, trends, regime changes)
- Different data dimensions and sparsity levels

Performance Metrics:
- Reconstruction accuracy (MSE, MAE, RMSE)
- Computational efficiency (time, memory)
- Robustness to noise and missing patterns
- Convergence behavior
Enhanced Features:
- Evaluation against clean AND noisy targets
- Autotuner verification and testing
- Baseline comparisons (mean, forward/backward fill)
- Ablation studies for feature importance
- Pairwise fair comparisons with identical masks
- Deterministic behavior and reproducibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import random
import argparse
from typing import Dict, List, Tuple, Optional
import gc
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import the missnet imputer and autotuner
from missnet_imputer import missnet_impute, get_optimal_config

# --- Baseline extras (all optional; guarded) ---
try:
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import savgol_filter
    _SCIPY_OK = True
except Exception:
    UnivariateSpline = None
    savgol_filter = None
    _SCIPY_OK = False

try:
    from sklearn.impute import KNNImputer
    _SKLEARN_IMPUTE_OK = True
except Exception:
    KNNImputer = None
    _SKLEARN_IMPUTE_OK = False

# Set deterministic behavior
random.seed(42)
np.random.seed(42)

# Print library availability
print("🔍 Library Availability Check:")
try:
    import scipy
    print(f"   ✅ SciPy: {scipy.__version__}")
except ImportError:
    print("   ❌ SciPy: Not available")

try:
    import numba
    print(f"   ✅ Numba: {numba.__version__}")
except ImportError:
    print("   ❌ Numba: Not available")

try:
    import joblib
    print(f"   ✅ Joblib: {joblib.__version__}")
except ImportError:
    print("   ❌ Joblib: Not available")

print()

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class SyntheticDataGenerator:
    """Generate various types of synthetic time series data with missing patterns."""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        
    def generate_base_signal(self, T: int, N: int, signal_type: str = 'mixed') -> np.ndarray:
        """Generate base time series signal."""
        t = np.arange(T)
        data = np.zeros((T, N))
        
        if signal_type == 'trend':
            # Linear trend with noise
            for i in range(N):
                trend = 0.01 * i * t + np.random.randn() * 0.1
                data[:, i] = trend + 0.1 * np.random.randn(T)
                
        elif signal_type == 'seasonal':
            # Seasonal patterns
            for i in range(N):
                freq1 = 2 * np.pi * (0.1 + 0.05 * i)
                freq2 = 2 * np.pi * (0.05 + 0.02 * i)
                seasonal = np.sin(freq1 * t) + 0.5 * np.cos(freq2 * t)
                data[:, i] = seasonal + 0.2 * np.random.randn(T)
                
        elif signal_type == 'regime':
            # Regime switching
            regimes = 3
            regime_length = T // regimes
            for r in range(regimes):
                start_idx = r * regime_length
                end_idx = (r + 1) * regime_length if r < regimes - 1 else T
                
                # Different parameters for each regime
                regime_mean = np.random.randn(N) * (r + 1)
                regime_cov = np.random.randn(N, N) * 0.1
                regime_cov = regime_cov @ regime_cov.T + np.eye(N) * 0.5
                
                regime_data = np.random.multivariate_normal(
                    regime_mean, regime_cov, end_idx - start_idx)
                data[start_idx:end_idx, :] = regime_data
                
        else:  # mixed
            # Combination of trends, seasonality, and noise
            for i in range(N):
                trend = 0.005 * i * t
                seasonal1 = 0.5 * np.sin(2 * np.pi * 0.1 * t + i * 0.2)
                seasonal2 = 0.3 * np.cos(2 * np.pi * 0.05 * t + i * 0.5)
                noise = 0.2 * np.random.randn(T)
                
                data[:, i] = trend + seasonal1 + seasonal2 + noise
                
        return data
    
    def add_correlation(self, data: np.ndarray, rho: float = 0.5) -> np.ndarray:
        """Apply AR(1)-style cross-feature correlation with parameter rho in [0,1)."""
        T, N = data.shape
        idx = np.arange(N)
        cov = rho ** np.abs(idx[:,None] - idx[None,:])
        L = np.linalg.cholesky(cov + 1e-8*np.eye(N))
        return data @ L.T
    
    def _match_missing_rate(self, miss_mask: np.ndarray, target_rate: float) -> np.ndarray:
        """Helper to match exact missing rate after masking."""
        cur = miss_mask.mean()
        if cur == 0 or abs(cur - target_rate) < 1e-3: 
            return miss_mask
        flat = miss_mask.ravel()
        k = int(round(target_rate * flat.size))
        # keep top-k random positions among current positives + negatives
        rng = np.random.default_rng(123)
        idx = rng.permutation(flat.size)
        flat[:] = False
        flat[idx[:k]] = True
        return miss_mask.reshape(miss_mask.shape)
    
    def introduce_missing_mcar(self, data: np.ndarray, missing_rate: float) -> np.ndarray:
        """Missing Completely At Random - uniform random missingness."""
        data_missing = data.copy()
        mask = np.random.random(data.shape) < missing_rate
        data_missing[mask] = np.nan
        return data_missing
    
    def introduce_missing_mar(self, data: np.ndarray, missing_rate: float) -> np.ndarray:
        """Missing At Random - missingness depends on observed values."""
        data_missing = data.copy()
        T, N = data.shape
        
        # Create missingness based on values of other variables
        for t in range(T):
            for i in range(N):
                # Probability of missing depends on average of other variables at same time
                other_vars = [j for j in range(N) if j != i]
                if other_vars:
                    other_mean = np.nanmean(data[t, other_vars])
                    missing_prob = missing_rate * (1 + 0.5 * np.tanh(other_mean))
                    if np.random.random() < missing_prob:
                        data_missing[t, i] = np.nan
                        
        return data_missing
    
    def introduce_missing_mnar(self, data: np.ndarray, missing_rate: float) -> np.ndarray:
        """Missing Not At Random - missingness depends on the missing values themselves."""
        data_missing = data.copy()
        T, N = data.shape
        
        # Create missingness based on the value itself
        for i in range(N):
            # Higher values more likely to be missing
            threshold = np.percentile(data[:, i], 100 * (1 - missing_rate * 1.5))
            missing_mask = data[:, i] > threshold
            # Add some randomness
            missing_mask &= (np.random.random(T) < missing_rate)
            data_missing[missing_mask, i] = np.nan
            
        return data_missing
    
    def add_noise(self, data: np.ndarray, noise_type: str = 'gaussian', 
                  noise_level: float = 0.1) -> np.ndarray:
        """Add different types of noise to the data."""
        noisy_data = data.copy()
        T, N = data.shape
        
        if noise_type == 'gaussian':
            noise = np.random.randn(T, N) * noise_level
            
        elif noise_type == 'uniform':
            noise = (np.random.random(T, N) - 0.5) * 2 * noise_level
            
        elif noise_type == 'outlier':
            # Add occasional outliers
            noise = np.random.randn(T, N) * noise_level * 0.1
            # Add outliers to 5% of data points
            outlier_mask = np.random.random((T, N)) < 0.05
            noise[outlier_mask] += np.random.choice([-1, 1], size=outlier_mask.sum()) * noise_level * 10
            
        elif noise_type == 'heteroscedastic':
            # Noise level depends on the signal magnitude
            noise = np.random.randn(T, N) * noise_level * (1 + np.abs(data))
            
        else:
            noise = np.random.randn(T, N) * noise_level
            
        return noisy_data + noise


class MissNetTester:
    """Comprehensive testing framework for MISSNET imputer."""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.generator = SyntheticDataGenerator(random_seed)
        self.results = []
        
    def calculate_metrics(self, original: np.ndarray, imputed: np.ndarray, 
                         missing_mask: np.ndarray, *, target: str = "noisy", 
                         debug: bool = False) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics with NaN-safe evaluation.
        
        target="noisy": evaluate against noisy ground truth (imputation focus)
        target="clean": evaluate against clean base signal (denoising+imputation)
        """
        if target not in ("noisy", "clean"):
            target = "noisy"
        
        # NaN-safe evaluation: only evaluate where both sides are finite
        eval_mask = missing_mask & np.isfinite(imputed) & np.isfinite(original)
        
        if debug:
            print(f"   Debug metrics:")
            print(f"      heldout_missing={missing_mask.sum()}")
            print(f"      finite_preds={np.isfinite(imputed[missing_mask]).sum()}")
            print(f"      finite_truth={np.isfinite(original[missing_mask]).sum()}")
            print(f"      evaluable={eval_mask.sum()}")
        
        # If no evaluable points, return NaN metrics
        if eval_mask.sum() == 0:
            if debug:
                print(f"      ⚠️  No evaluable points - returning NaN metrics")
            return {
                'mse': float('nan'),
                'mae': float('nan'),
                'rmse': float('nan'),
                'mape': float('nan'),
                'correlation': 0.0,
                'residual_std': float('nan'),
                'residual_mean': float('nan'),
                'coverage_95': float('nan'),
                'evaluable_points': 0,
            }
        
        # Only evaluate on evaluable points
        ref = original[eval_mask]
        est = imputed[eval_mask]
        
        # Guard correlation on degenerate cases
        if ref.size < 3 or np.std(ref) == 0 or np.std(est) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(ref, est)[0, 1])
        
        residuals = ref - est
        return {
            'mse': float(np.mean((ref - est)**2)),
            'mae': float(np.mean(np.abs(ref - est))),
            'rmse': float(np.sqrt(np.mean((ref - est)**2))),
            'mape': float(np.mean(np.abs((ref - est) / (np.abs(ref) + 1e-8))) * 100),
            'correlation': corr,
            'residual_std': float(np.std(residuals)),
            'residual_mean': float(np.mean(residuals)),
            'coverage_95': float(np.mean(np.abs(residuals) < 1.96 * (np.std(residuals) + 1e-12))),
            'evaluable_points': int(eval_mask.sum()),
        }
    
    def run_single_test(self, test_name: str, data: np.ndarray, 
                       missing_pattern: str = 'mcar', missing_rate: float = 0.2,
                       noise_type: str = 'gaussian', noise_level: float = 0.1,
                       **missnet_params) -> Dict:
        """Run a single test case."""
        print(f"\n🧪 Running test: {test_name}")
        print(f"   Data shape: {data.shape}")
        print(f"   Missing pattern: {missing_pattern}, rate: {missing_rate:.1%}")
        print(f"   Noise: {noise_type}, level: {noise_level}")
        
        # Build both references
        clean_ref = data                    # before noise
        noisy_ref = self.generator.add_noise(data, noise_type, noise_level)  # after noise, before masking
        
        # Introduce missing values
        if missing_pattern == 'mcar':
            data_missing = self.generator.introduce_missing_mcar(noisy_ref, missing_rate)
        elif missing_pattern == 'mar':
            data_missing = self.generator.introduce_missing_mar(noisy_ref, missing_rate)
        elif missing_pattern == 'mnar':
            data_missing = self.generator.introduce_missing_mnar(noisy_ref, missing_rate)
        else:
            raise ValueError(f"Unknown missing pattern: {missing_pattern}")
        
        missing_mask = np.isnan(data_missing)
        actual_missing_rate = missing_mask.sum() / missing_mask.size
        print(f"   Actual missing rate: {actual_missing_rate:.1%}")
        
        # Run MISSNET imputation
        start_time = time.time()
        
        try:
            imputed_data = missnet_impute(
                data_missing,
                verbose=False,
                **missnet_params
            )
            
            execution_time = time.time() - start_time
            
            # Calculate metrics for both targets
            metrics_noisy = self.calculate_metrics(noisy_ref, imputed_data, missing_mask, target="noisy")
            metrics_clean = self.calculate_metrics(clean_ref, imputed_data, missing_mask, target="clean")
            
            # Add throughput KPI
            throughput = missing_mask.size / execution_time
            
            # Add metadata
            result = {
                'test_name': test_name,
                'data_shape': data.shape,
                'missing_pattern': missing_pattern,
                'missing_rate': actual_missing_rate,
                'noise_type': noise_type,
                'noise_level': noise_level,
                'execution_time': execution_time,
                'cells_per_sec': float(throughput),
                'success': True,
                'error': None,
                # noisy-target metrics (imputation)
                **{f'noisy_{k}': v for k, v in metrics_noisy.items()},
                # clean-target metrics (imputation+denoising robustness)
                **{f'clean_{k}': v for k, v in metrics_clean.items()},
                **missnet_params
            }
            
            print(f"   ✅ Success in {execution_time:.2f}s | MSE(noisy): {metrics_noisy['mse']:.6f}  | MSE(clean): {metrics_clean['mse']:.6f}")
            print(f"   📊 Throughput: {throughput:.0f} cells/sec")
            
        except Exception as e:
            result = {
                'test_name': test_name,
                'data_shape': data.shape,
                'missing_pattern': missing_pattern,
                'missing_rate': actual_missing_rate,
                'noise_type': noise_type,
                'noise_level': noise_level,
                'execution_time': time.time() - start_time,
                'cells_per_sec': 0.0,
                'success': False,
                'error': str(e),
                **missnet_params
            }
            
            print(f"   ❌ Failed: {str(e)}")
        
        self.results.append(result)
        return result
    
    def run_comprehensive_tests(self):
        """Run a comprehensive suite of tests."""
        print("🚀 Starting Comprehensive MISSNET Testing Suite")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            # Basic functionality tests
            {
                'name': 'Basic Small Data',
                'T': 50, 'N': 5, 'signal_type': 'mixed',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            {
                'name': 'Basic Medium Data',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            {
                'name': 'Basic Large Data',
                'T': 500, 'N': 20, 'signal_type': 'mixed',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            
            # Missing pattern tests
            {
                'name': 'MCAR Missing',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_pattern': 'mcar', 'missing_rate': 0.3
            },
            {
                'name': 'MAR Missing',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_pattern': 'mar', 'missing_rate': 0.3
            },
            {
                'name': 'MNAR Missing',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_pattern': 'mnar', 'missing_rate': 0.3
            },
            
            # Missing rate tests
            {
                'name': 'Low Missing Rate',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_rate': 0.1
            },
            {
                'name': 'High Missing Rate',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_rate': 0.5
            },
            
            # Noise type tests
            {
                'name': 'Gaussian Noise',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'noise_type': 'gaussian', 'noise_level': 0.2
            },
            {
                'name': 'Outlier Noise',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'noise_type': 'outlier', 'noise_level': 0.2
            },
            {
                'name': 'Heteroscedastic Noise',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'noise_type': 'heteroscedastic', 'noise_level': 0.1
            },
            
            # Signal type tests
            {
                'name': 'Trend Signal',
                'T': 200, 'N': 10, 'signal_type': 'trend'
            },
            {
                'name': 'Seasonal Signal',
                'T': 200, 'N': 10, 'signal_type': 'seasonal'
            },
            {
                'name': 'Regime Signal',
                'T': 200, 'N': 10, 'signal_type': 'regime'
            },
            
            # Correlation tests
            {
                'name': 'Low Correlation',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'correlation_strength': 0.1
            },
            {
                'name': 'High Correlation',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'correlation_strength': 0.8
            },
            
            # Multi-regime tests
            {
                'name': 'Multi-Regime with Clusters',
                'T': 300, 'N': 10, 'signal_type': 'regime',
                'n_cl': 3, 'max_iteration': 30
            },
        ]
        
        # Run all tests
        for config in test_configs:
            # Extract parameters
            test_name = config.pop('name')
            T = config.pop('T', 200)
            N = config.pop('N', 10)
            signal_type = config.pop('signal_type', 'mixed')
            correlation_strength = config.pop('correlation_strength', 0.5)
            
            # Generate data
            data = self.generator.generate_base_signal(T, N, signal_type)
            data = self.generator.add_correlation(data, correlation_strength)
            
            # Set default parameters with autotune enabled
            default_params = {
                'missing_pattern': 'mcar',
                'missing_rate': 0.2,
                'noise_type': 'gaussian',
                'noise_level': 0.1,
                'auto_tune': True  # Enable autotuning by default
            }
            
            # Override with test-specific parameters
            params = {**default_params, **config}
            
            # Run the test
            self.run_single_test(test_name, data, **params)
            
            # Clean up memory
            gc.collect()
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        if not self.results:
            return "No test results available."
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Summary statistics
        successful_tests = df[df['success']]
        failed_tests = df[~df['success']]
        
        report = []
        report.append("📊 MISSNET Comprehensive Test Report")
        report.append("=" * 50)
        report.append(f"Total tests: {len(self.results)}")
        report.append(f"Successful: {len(successful_tests)}")
        report.append(f"Failed: {len(failed_tests)}")
        report.append(f"Success rate: {len(successful_tests)/len(self.results)*100:.1f}%")
        report.append("")
        
        if len(successful_tests) > 0:
            report.append("🎯 Performance Summary (Successful Tests)")
            report.append("-" * 40)
            
            for metric in ['mse', 'mae', 'rmse', 'correlation', 'execution_time']:
                if metric in successful_tests.columns:
                    values = successful_tests[metric]
                    report.append(f"{metric.upper():12}: "
                                f"Mean={values.mean():.6f}, "
                                f"Std={values.std():.6f}, "
                                f"Min={values.min():.6f}, "
                                f"Max={values.max():.6f}")
            report.append("")
            
            # Best and worst performing tests - use noisy_mse for baseline comparisons
            mse_col = 'mse' if 'mse' in successful_tests.columns else 'noisy_mse'
            corr_col = 'correlation' if 'correlation' in successful_tests.columns else 'noisy_correlation'
            
            best_mse = successful_tests.loc[successful_tests[mse_col].idxmin()]
            worst_mse = successful_tests.loc[successful_tests[mse_col].idxmax()]
            
            report.append("🏆 Best Performance (Lowest MSE)")
            report.append(f"   Test: {best_mse['test_name']}")
            report.append(f"   MSE: {best_mse[mse_col]:.6f}")
            report.append(f"   Correlation: {best_mse[corr_col]:.4f}")
            report.append("")
            
            report.append("⚠️  Worst Performance (Highest MSE)")
            report.append(f"   Test: {worst_mse['test_name']}")
            report.append(f"   MSE: {worst_mse[mse_col]:.6f}")
            report.append(f"   Correlation: {worst_mse[corr_col]:.4f}")
            report.append("")
        
        if len(failed_tests) > 0:
            report.append("❌ Failed Tests")
            report.append("-" * 20)
            for _, test in failed_tests.iterrows():
                report.append(f"   {test['test_name']}: {test['error']}")
            report.append("")
        
        # Performance by missing pattern
        if len(successful_tests) > 0 and 'missing_pattern' in successful_tests.columns:
            report.append("📈 Performance by Missing Pattern")
            report.append("-" * 35)
            for pattern in successful_tests['missing_pattern'].unique():
                pattern_data = successful_tests[successful_tests['missing_pattern'] == pattern]
                mse_col = 'mse' if 'mse' in pattern_data.columns else 'noisy_mse'
                corr_col = 'correlation' if 'correlation' in pattern_data.columns else 'noisy_correlation'
                report.append(f"{pattern.upper():8}: "
                            f"MSE={pattern_data[mse_col].mean():.6f}, "
                            f"Corr={pattern_data[corr_col].mean():.4f}, "
                            f"Tests={len(pattern_data)}")
            report.append("")
        
        # Performance by noise type
        if len(successful_tests) > 0 and 'noise_type' in successful_tests.columns:
            report.append("🔊 Performance by Noise Type")
            report.append("-" * 30)
            for noise_type in successful_tests['noise_type'].unique():
                noise_data = successful_tests[successful_tests['noise_type'] == noise_type]
                mse_col = 'mse' if 'mse' in noise_data.columns else 'noisy_mse'
                corr_col = 'correlation' if 'correlation' in noise_data.columns else 'noisy_correlation'
                report.append(f"{noise_type.upper():12}: "
                            f"MSE={noise_data[mse_col].mean():.6f}, "
                            f"Corr={noise_data[corr_col].mean():.4f}, "
                            f"Tests={len(noise_data)}")
            report.append("")
        
        # Scalability analysis
        if len(successful_tests) > 0 and 'data_shape' in successful_tests.columns:
            report.append("⚡ Scalability Analysis")
            report.append("-" * 25)
            
            # Group by data size
            successful_tests['data_size'] = successful_tests['data_shape'].apply(
                lambda x: x[0] * x[1])
            size_groups = successful_tests.groupby(
                pd.cut(successful_tests['data_size'], 
                      bins=[0, 500, 2000, 10000], 
                      labels=['Small', 'Medium', 'Large']))
            
            for size_name, group in size_groups:
                if len(group) > 0:
                    mse_col = 'mse' if 'mse' in group.columns else 'noisy_mse'
                    report.append(f"{size_name:8}: "
                                f"Avg Time={group['execution_time'].mean():.2f}s, "
                                f"Avg MSE={group[mse_col].mean():.6f}, "
                                f"Tests={len(group)}")
        
        return "\n".join(report)
    
    def create_visualizations(self, save_path: str = None):
        """Create comprehensive visualization plots."""
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        successful_tests = df[df['success']]
        
        if len(successful_tests) == 0:
            print("No successful tests to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MISSNET Performance Analysis', fontsize=16, fontweight='bold')
        
        # Use the correct column names for baseline comparisons
        mse_col = 'mse' if 'mse' in successful_tests.columns else 'noisy_mse'
        corr_col = 'correlation' if 'correlation' in successful_tests.columns else 'noisy_correlation'
        
        # 1. MSE distribution
        axes[0, 0].hist(successful_tests[mse_col], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('MSE Distribution')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Correlation vs MSE
        if corr_col in successful_tests.columns:
            axes[0, 1].scatter(successful_tests[corr_col], successful_tests[mse_col], 
                             alpha=0.6, s=50)
            axes[0, 1].set_xlabel('Correlation')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_title('Correlation vs MSE')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Execution time vs data size
        if 'data_shape' in successful_tests.columns:
            successful_tests['data_size'] = successful_tests['data_shape'].apply(
                lambda x: x[0] * x[1])
            axes[0, 2].scatter(successful_tests['data_size'], 
                             successful_tests['execution_time'], 
                             alpha=0.6, s=50)
            axes[0, 2].set_xlabel('Data Size (T × N)')
            axes[0, 2].set_ylabel('Execution Time (s)')
            axes[0, 2].set_title('Scalability: Time vs Data Size')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance by missing pattern
        if 'missing_pattern' in successful_tests.columns:
            pattern_perf = successful_tests.groupby('missing_pattern')[mse_col].mean()
            axes[1, 0].bar(pattern_perf.index, pattern_perf.values, alpha=0.7)
            axes[1, 0].set_xlabel('Missing Pattern')
            axes[1, 0].set_ylabel('Average MSE')
            axes[1, 0].set_title('Performance by Missing Pattern')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Performance by noise type
        if 'noise_type' in successful_tests.columns:
            noise_perf = successful_tests.groupby('noise_type')[mse_col].mean()
            axes[1, 1].bar(noise_perf.index, noise_perf.values, alpha=0.7)
            axes[1, 1].set_xlabel('Noise Type')
            axes[1, 1].set_ylabel('Average MSE')
            axes[1, 1].set_title('Performance by Noise Type')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Missing rate vs MSE
        if 'missing_rate' in successful_tests.columns:
            axes[1, 2].scatter(successful_tests['missing_rate'] * 100, 
                             successful_tests[mse_col], 
                             alpha=0.6, s=50)
            axes[1, 2].set_xlabel('Missing Rate (%)')
            axes[1, 2].set_ylabel('MSE')
            axes[1, 2].set_title('Missing Rate vs MSE')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def run_autotune_comparison_tests(self):
        """Run comprehensive tests comparing autotune vs manual parameters."""
        print("🔧 Starting MISSNET Autotune Comparison Tests")
        print("=" * 60)
        
        # Test configurations specifically for autotune evaluation
        autotune_test_configs = [
            # Small datasets
            {
                'name': 'Autotune Small Dense',
                'T': 50, 'N': 5, 'signal_type': 'mixed',
                'correlation_strength': 0.8, 'missing_rate': 0.2
            },
            {
                'name': 'Autotune Small Sparse',
                'T': 50, 'N': 5, 'signal_type': 'mixed',
                'correlation_strength': 0.1, 'missing_rate': 0.3
            },
            
            # Medium datasets with different characteristics
            {
                'name': 'Autotune Medium Trend',
                'T': 200, 'N': 10, 'signal_type': 'trend',
                'missing_rate': 0.25, 'noise_level': 0.15
            },
            {
                'name': 'Autotune Medium Seasonal',
                'T': 200, 'N': 10, 'signal_type': 'seasonal',
                'missing_rate': 0.25, 'noise_level': 0.15
            },
            {
                'name': 'Autotune Medium Regime',
                'T': 200, 'N': 10, 'signal_type': 'regime',
                'missing_rate': 0.25, 'noise_level': 0.15
            },
            
            # Large datasets
            {
                'name': 'Autotune Large High Missing',
                'T': 500, 'N': 20, 'signal_type': 'mixed',
                'missing_rate': 0.4, 'noise_level': 0.1
            },
            {
                'name': 'Autotune Large Low Missing',
                'T': 500, 'N': 20, 'signal_type': 'mixed',
                'missing_rate': 0.1, 'noise_level': 0.1
            },
            
            # Challenging scenarios
            {
                'name': 'Autotune High Correlation',
                'T': 300, 'N': 15, 'signal_type': 'mixed',
                'correlation_strength': 0.9, 'missing_rate': 0.3
            },
            {
                'name': 'Autotune Outlier Heavy',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'noise_type': 'outlier', 'noise_level': 0.2, 'missing_rate': 0.2
            },
            {
                'name': 'Autotune Heteroscedastic',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'noise_type': 'heteroscedastic', 'noise_level': 0.15, 'missing_rate': 0.25
            },
            
            # Missing pattern variations
            {
                'name': 'Autotune MCAR Pattern',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_pattern': 'mcar', 'missing_rate': 0.3
            },
            {
                'name': 'Autotune MAR Pattern',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_pattern': 'mar', 'missing_rate': 0.3
            },
            {
                'name': 'Autotune MNAR Pattern',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_pattern': 'mnar', 'missing_rate': 0.3
            },
            
            # Multi-regime scenarios
            {
                'name': 'Autotune Multi-Regime',
                'T': 400, 'N': 12, 'signal_type': 'regime',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
        ]
        
        # For each test configuration, compare autotune vs manual parameters
        for config in autotune_test_configs:
            test_name = config.pop('name')
            T = config.pop('T', 200)
            N = config.pop('N', 10)
            signal_type = config.pop('signal_type', 'mixed')
            correlation_strength = config.pop('correlation_strength', 0.5)
            
            print(f"\n🔄 Running comparison test: {test_name}")
            print(f"   Data shape: {T} × {N}, Signal: {signal_type}")
            
            # Generate data
            data = self.generator.generate_base_signal(T, N, signal_type)
            data = self.generator.add_correlation(data, correlation_strength)
            
            # Set base parameters
            base_params = {
                'missing_pattern': 'mcar',
                'missing_rate': 0.2,
                'noise_type': 'gaussian',
                'noise_level': 0.1,
                **config
            }
            
            # Test 1: With autotuning
            print("   🤖 Testing with AUTOTUNE=True...")
            autotune_params = {**base_params, 'auto_tune': True}
            autotune_result = self.run_single_test(
                f"{test_name} (AUTOTUNE)", data, **autotune_params
            )
            
            # Test 2: With fixed default parameters
            print("   🎛️  Testing with FIXED parameters...")
            fixed_params = {**base_params, 'auto_tune': False, 'alpha': 0.5, 'beta': 0.1, 'L': 10, 'n_cl': 1}
            fixed_result = self.run_single_test(
                f"{test_name} (FIXED)", data, **fixed_params
            )
            
            # Test 3: With conservative parameters
            print("   🔒 Testing with CONSERVATIVE parameters...")
            conservative_params = {**base_params, 'auto_tune': False, 'alpha': 0.3, 'beta': 0.2, 'L': 5, 'n_cl': 1}
            conservative_result = self.run_single_test(
                f"{test_name} (CONSERVATIVE)", data, **conservative_params
            )
            
        # Helper function to pick MSE from different key formats
        def _pick_mse(d):
            return d.get('mse', d.get('noisy_mse', d.get('mse_clean', d.get('mse_noisy'))))

        # Calculate improvement metrics
        if autotune_result['success'] and fixed_result['success']:
            autotune_mse = _pick_mse(autotune_result)
            fixed_mse = _pick_mse(fixed_result)
            mse_improvement = (fixed_mse - autotune_mse) / fixed_mse * 100
            
            # Handle correlation similarly
            autotune_corr = autotune_result.get('correlation', autotune_result.get('noisy_correlation', 0.0))
            fixed_corr = fixed_result.get('correlation', fixed_result.get('noisy_correlation', 0.0))
            if abs(fixed_corr) > 1e-12:
                corr_improvement = (autotune_corr - fixed_corr) / abs(fixed_corr) * 100
            else:
                corr_improvement = 0.0
            
            time_diff = autotune_result['execution_time'] - fixed_result['execution_time']
            
            print(f"   📊 Autotune vs Fixed:")
            print(f"      MSE improvement: {mse_improvement:+.2f}%")
            print(f"      Correlation improvement: {corr_improvement:+.2f}%")
            print(f"      Time difference: {time_diff:+.2f}s")
            
            # Clean up memory
            gc.collect()
    
    def test_autotune_parameter_selection(self):
        """Test autotune parameter selection on diverse data characteristics."""
        print("\n🎯 Testing Autotune Parameter Selection Logic")
        print("=" * 55)
        
        # Define specific data scenarios to test parameter selection
        scenarios = [
            {
                'name': 'High Correlation Scenario',
                'T': 150, 'N': 8, 'correlation_strength': 0.9,
                'missing_rate': 0.2, 'signal_type': 'mixed'
            },
            {
                'name': 'Low Correlation Scenario', 
                'T': 150, 'N': 8, 'correlation_strength': 0.1,
                'missing_rate': 0.2, 'signal_type': 'mixed'
            },
            {
                'name': 'High Missing Rate Scenario',
                'T': 150, 'N': 8, 'correlation_strength': 0.5,
                'missing_rate': 0.6, 'signal_type': 'mixed'
            },
            {
                'name': 'Large Dimensional Scenario',
                'T': 100, 'N': 30, 'correlation_strength': 0.5,
                'missing_rate': 0.2, 'signal_type': 'mixed'
            },
            {
                'name': 'Small Sample Scenario',
                'T': 50, 'N': 15, 'correlation_strength': 0.5,
                'missing_rate': 0.2, 'signal_type': 'mixed'
            },
            {
                'name': 'Seasonal Pattern Scenario',
                'T': 200, 'N': 10, 'correlation_strength': 0.5,
                'missing_rate': 0.2, 'signal_type': 'seasonal'
            },
            {
                'name': 'Regime Switching Scenario',
                'T': 300, 'N': 10, 'correlation_strength': 0.5,
                'missing_rate': 0.2, 'signal_type': 'regime'
            },
            {
                'name': 'Outlier Heavy Scenario',
                'T': 150, 'N': 10, 'correlation_strength': 0.5,
                'missing_rate': 0.2, 'signal_type': 'mixed', 'noise_type': 'outlier'
            }
        ]
        
        from missnet_imputer import get_optimal_config
        
        for scenario in scenarios:
            name = scenario.pop('name')
            T = scenario.pop('T')
            N = scenario.pop('N')
            correlation_strength = scenario.pop('correlation_strength', 0.5)
            
            print(f"\n📋 Scenario: {name}")
            
            # Generate data
            signal_type = scenario.pop('signal_type', 'mixed')
            data = self.generator.generate_base_signal(T, N, signal_type)
            data = self.generator.add_correlation(data, correlation_strength)
            
            # Add noise and missing values
            noise_type = scenario.pop('noise_type', 'gaussian')
            noise_level = scenario.pop('noise_level', 0.1)
            missing_rate = scenario.pop('missing_rate', 0.2)
            
            noisy_data = self.generator.add_noise(data, noise_type, noise_level)
            data_missing = self.generator.introduce_missing_mcar(noisy_data, missing_rate)
            
            # Get optimal configuration
            try:
                config = get_optimal_config(data_missing, fast=True, verbose=False)
                
                print(f"   📊 Data characteristics:")
                print(f"      Shape: {data_missing.shape}")
                print(f"      Missing rate: {missing_rate:.1%}")
                print(f"      Correlation strength: {correlation_strength:.2f}")
                print(f"      Signal type: {signal_type}")
                
                print(f"   ⚙️  Selected parameters:")
                print(f"      alpha: {config['alpha']:.3f} (network vs temporal balance)")
                print(f"      beta: {config['beta']:.4f} (sparsity regularization)")
                print(f"      L: {config['L']} (latent dimension)")
                print(f"      n_cl: {config['n_cl']} (regimes)")
                print(f"      max_iteration: {config['max_iteration']}")
                print(f"      use_robust_loss: {config['use_robust_loss']}")
                print(f"      use_skeleton: {config['use_skeleton']}")
                print(f"      use_cholesky: {config['use_cholesky']}")
                
                # Test the configuration
                result = self.run_single_test(
                    f"Param Selection: {name}", data_missing, 
                    auto_tune=False, verbose=False, **config
                )
                
            except Exception as e:
                print(f"   ❌ Failed to get optimal config: {str(e)}")
    
    def autotune_smoke(self):
        """Test that autotuner flips the right switches for curated scenarios."""
        print("\n🧭 AUTOTUNE SMOKE TEST")
        print("=" * 40)
        
        cases = [
            ("Seasonal", 200, 10, 'seasonal', dict()),
            ("Outliers", 200, 10, 'mixed', dict(noise_type='outlier', noise_level=0.3)),
            ("SparseHiMiss", 200, 64, 'mixed', dict(missing_rate=0.6, correlation_strength=0.1)),
            ("DenseLowMiss", 200, 32, 'mixed', dict(missing_rate=0.1, correlation_strength=0.9)),
        ]
        
        for name, T, N, sig, opts in cases:
            X = self.generator.generate_base_signal(T, N, sig)
            X = self.generator.add_correlation(X, opts.get('correlation_strength', 0.5))
            X = self.generator.add_noise(X, opts.get('noise_type', 'gaussian'), opts.get('noise_level', 0.1))
            X = self.generator.introduce_missing_mcar(X, opts.get('missing_rate', 0.2))
            cfg = get_optimal_config(X, fast=True, verbose=False)

            print(f"\n🧭 AUTOTUNE {name}: "
                  f"use_spectral={cfg['cfg_overrides'].get('USE_SPECTRAL')}, "
                  f"use_huber={cfg['cfg_overrides'].get('USE_HUBER')}, "
                  f"use_candidate_gl={cfg['cfg_overrides'].get('USE_CANDIDATE_GL')}, "
                  f"L={cfg['L']}, n_cl={cfg['n_cl']}")

            if name == "Seasonal":
                if not cfg['cfg_overrides'].get('USE_SPECTRAL', False):
                    print(f"   ⚠️  Warning: Seasonal case did not enable spectral")
            if name == "Outliers":
                if not cfg['cfg_overrides'].get('USE_HUBER', False):
                    print(f"   ⚠️  Warning: Outlier case did not enable Huber")
    
    def run_pair_autotune_vs_fixed(self, name, data, **base):
        """Pairwise autotune vs fixed comparison on same mask for fair comparison."""
        noisy = self.generator.add_noise(data, base.get('noise_type','gaussian'), base.get('noise_level',0.1))
        mask_fn = dict(mcar=self.generator.introduce_missing_mcar,
                       mar=self.generator.introduce_missing_mar,
                       mnar=self.generator.introduce_missing_mnar)[base.get('missing_pattern','mcar')]
        X_missing = mask_fn(noisy, base.get('missing_rate',0.2))
        M = np.isnan(X_missing)

        # autotune
        imp_auto  = missnet_impute(X_missing, verbose=False, auto_tune=True)
        # fixed
        imp_fixed = missnet_impute(X_missing, verbose=False, auto_tune=False, alpha=0.5, beta=0.1, L=10, n_cl=1)

        m_auto_noisy  = self.calculate_metrics(noisy, imp_auto,  M, target="noisy")
        m_fixed_noisy = self.calculate_metrics(noisy, imp_fixed, M, target="noisy")

        print(f"   ΔMSE(noisy) = {m_fixed_noisy['mse'] - m_auto_noisy['mse']:+.6f} "
              f"({(m_fixed_noisy['mse']-m_auto_noisy['mse'])/max(m_fixed_noisy['mse'],1e-12)*100:+.2f}%)")
        
        return m_auto_noisy, m_fixed_noisy

    def run_fourier_comparison(self, name, data, **base):
        """Compare performance with and without Fourier transform on same mask."""
        noisy = self.generator.add_noise(data, base.get('noise_type','gaussian'), base.get('noise_level',0.1))
        mask_fn = dict(mcar=self.generator.introduce_missing_mcar,
                       mar=self.generator.introduce_missing_mar,
                       mnar=self.generator.introduce_missing_mnar)[base.get('missing_pattern','mcar')]
        X_missing = mask_fn(noisy, base.get('missing_rate',0.2))
        M = np.isnan(X_missing)

        print(f"\n📊 Fourier Comparison: {name}")
        print(f"   Data shape: {data.shape}, Missing rate: {base.get('missing_rate',0.2):.1%}")
        
        # Run without Fourier (use_spectral=False)
        print("   🔄 Running WITHOUT Fourier transform...")
        start_time = time.time()
        try:
            imp_no_fourier = missnet_impute(X_missing, verbose=False, auto_tune=True, use_spectral=False)
            time_no_fourier = time.time() - start_time
            m_no_fourier = self.calculate_metrics(noisy, imp_no_fourier, M, target="noisy")
            success_no_fourier = True
            print(f"      ✅ Completed in {time_no_fourier:.2f}s | MSE: {m_no_fourier['mse']:.6f}")
        except Exception as e:
            print(f"      ❌ Failed: {str(e)}")
            imp_no_fourier = None
            time_no_fourier = float('inf')
            m_no_fourier = None
            success_no_fourier = False

        # Run with Fourier (use_spectral=True)
        print("   🌊 Running WITH Fourier transform...")
        start_time = time.time()
        try:
            imp_with_fourier = missnet_impute(X_missing, verbose=False, auto_tune=True, use_spectral=True)
            time_with_fourier = time.time() - start_time
            m_with_fourier = self.calculate_metrics(noisy, imp_with_fourier, M, target="noisy")
            success_with_fourier = True
            print(f"      ✅ Completed in {time_with_fourier:.2f}s | MSE: {m_with_fourier['mse']:.6f}")
        except Exception as e:
            print(f"      ❌ Failed: {str(e)}")
            imp_with_fourier = None
            time_with_fourier = float('inf')
            m_with_fourier = None
            success_with_fourier = False

        # Compare results if both succeeded
        if success_no_fourier and success_with_fourier:
            mse_diff = m_no_fourier['mse'] - m_with_fourier['mse']
            mse_improvement = mse_diff / m_no_fourier['mse'] * 100
            time_diff = time_with_fourier - time_no_fourier
            time_overhead = time_diff / time_no_fourier * 100
            
            print(f"\n   📈 Performance Comparison:")
            print(f"      MSE (No Fourier): {m_no_fourier['mse']:.6f}")
            print(f"      MSE (With Fourier): {m_with_fourier['mse']:.6f}")
            print(f"      MSE Improvement: {mse_improvement:+.2f}% ({'better' if mse_improvement > 0 else 'worse'})")
            print(f"      Time (No Fourier): {time_no_fourier:.2f}s")
            print(f"      Time (With Fourier): {time_with_fourier:.2f}s")
            print(f"      Time Overhead: {time_overhead:+.1f}% ({'slower' if time_overhead > 0 else 'faster'})")
            
            # Store results for analysis
            result_no_fourier = {
                'test_name': f"{name} [No Fourier]",
                'data_shape': data.shape,
                'missing_pattern': base.get('missing_pattern', 'mcar'),
                'missing_rate': float(M.mean()),
                'noise_type': base.get('noise_type', 'gaussian'),
                'noise_level': base.get('noise_level', 0.1),
                'execution_time': time_no_fourier,
                'cells_per_sec': float(M.size / max(time_no_fourier, 1e-9)),
                'success': True,
                'error': None,
                'use_fourier': False,
                **{f'noisy_{k}': v for k, v in m_no_fourier.items()},
            }
            
            result_with_fourier = {
                'test_name': f"{name} [With Fourier]",
                'data_shape': data.shape,
                'missing_pattern': base.get('missing_pattern', 'mcar'),
                'missing_rate': float(M.mean()),
                'noise_type': base.get('noise_type', 'gaussian'),
                'noise_level': base.get('noise_level', 0.1),
                'execution_time': time_with_fourier,
                'cells_per_sec': float(M.size / max(time_with_fourier, 1e-9)),
                'success': True,
                'error': None,
                'use_fourier': True,
                **{f'noisy_{k}': v for k, v in m_with_fourier.items()},
            }
            
            self.results.extend([result_no_fourier, result_with_fourier])
            
            return m_no_fourier, m_with_fourier, time_no_fourier, time_with_fourier
        
        return None, None, None, None
    
    def mean_impute(self, X):
        """Simple mean imputation baseline."""
        col_means = np.nanmean(X, axis=0)
        return np.where(np.isnan(X), col_means, X)

    def ffill_bfill_impute(self, X):
        """Forward-fill then backward-fill baseline."""
        df = pd.DataFrame(X).ffill().bfill()
        return df.values

    # ---------- Baseline family (fast) ----------

    def linear_interp_impute(self, X: np.ndarray) -> np.ndarray:
        """Per-column linear interpolation; two-sided (uses future). O(TN)."""
        df = pd.DataFrame(X).interpolate(method='linear', axis=0, limit_direction='both')
        return df.values

    def spline_impute(self, X: np.ndarray, order: int = 3, smooth: Optional[float] = None) -> np.ndarray:
        """
        Per-column cubic spline (global, uses future); falls back to polyfit if SciPy missing.
        Good for smooth series / seasonality. O(TN).
        """
        T, N = X.shape
        t = np.arange(T, dtype=float)
        out = X.copy()
        for i in range(N):
            col = X[:, i]
            obs = ~np.isnan(col)
            if obs.sum() < max(4, order + 1):
                # fallback: mean
                m = np.nanmean(col)
                out[~obs, i] = m
                continue
            if _SCIPY_OK and UnivariateSpline is not None:
                s = smooth if smooth is not None else max(1e-3 * obs.sum(), 1e-6)
                spl = UnivariateSpline(t[obs], col[obs], k=min(order, 5), s=s)
                pred = spl(t)
            else:
                deg = min(order, int(obs.sum()) - 1)
                # polyfit fallback
                coef = np.polyfit(t[obs], col[obs], deg=deg)
                pred = np.polyval(coef, t)
            out[~obs, i] = pred[~obs]
        return out

    def savgol_impute(self, X: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
        """
        Savitzky–Golay smoothing on ffilled/bfilled series; uses centered window (acausal).
        Falls back to symmetric moving average if SciPy missing. O(TN).
        """
        T, N = X.shape
        # ensure odd window and <= T
        window = max(3, min(window | 1, T if T % 2 == 1 else T - 1))
        filled = pd.DataFrame(X).ffill().bfill().values
        if _SCIPY_OK and savgol_filter is not None and window > polyorder:
            sm = savgol_filter(filled, window_length=window, polyorder=polyorder, axis=0, mode='interp')
        else:
            # symmetric moving average fallback
            k = window
            sm = np.empty_like(filled)
            pad = k // 2
            for j in range(N):
                a = np.pad(filled[:, j], (pad, pad), mode='edge')
                c = np.convolve(a, np.ones(k)/k, mode='valid')
                sm[:, j] = c
        return np.where(np.isnan(X), sm, X)

    def fourier_global_impute(self, X: np.ndarray, K: int = 6, lam: float = 1e-3) -> np.ndarray:
        """
        Global Fourier ridge fit per column (uses whole series); fast and strong for seasonal data.
        """
        T, N = X.shape
        t = np.arange(T, dtype=float)
        cols = [np.ones(T)]
        for k in range(1, K + 1):
            w = 2.0 * np.pi * k * t / float(T)
            cols += [np.sin(w), np.cos(w)]
        Phi = np.column_stack(cols)  # [T, 1+2K]
        D = Phi.shape[1]
        I = np.eye(D)
        out = X.copy()
        for i in range(N):
            col = X[:, i]
            obs = ~np.isnan(col)
            if obs.sum() < D:
                # fall back to linear interpolation
                continue
            Po = Phi[obs]
            yo = col[obs]
            coef = np.linalg.pinv(Po.T @ Po + lam * I) @ (Po.T @ yo)
            pred = Phi @ coef
            out[~obs, i] = pred[~obs]
        # any columns that fell back will be handled later if needed
        return out

    def knn_impute_fast(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """
        KNN imputer across features (uses correlations); quick strong baseline for structured data.
        """
        if not _SKLEARN_IMPUTE_OK:
            # graceful fallback: mean+linear
            return self.linear_interp_impute(self.mean_impute(X))
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
        return imputer.fit_transform(X)

    def iterative_svd_impute(self, X: np.ndarray, rank: int = 5, iters: int = 8) -> np.ndarray:
        """
        Lightweight low-rank matrix completion (Iterative SVD).
        Uses entire series (future-aware), often very competitive while fast for moderate sizes.
        """
        X_in = X.copy()
        M = np.isnan(X_in)
        # init with column means
        X_fill = self.mean_impute(X_in)
        for _ in range(max(1, iters)):
            # rank-r reconstruction
            U, s, Vt = np.linalg.svd(X_fill, full_matrices=False)
            r = min(rank, min(U.shape[1], Vt.shape[0]))
            X_hat = (U[:, :r] * s[:r]) @ Vt[:r, :]
            # keep observed values fixed, update missing
            X_fill[M] = X_hat[M]
        return X_fill

    def centered_window_mean_impute(self, X: np.ndarray, window: int = 9) -> np.ndarray:
        """
        Simple centered moving-average imputation (uses future via symmetric window).
        """
        T, N = X.shape
        window = max(3, window | 1)
        out = X.copy()
        pad = window // 2
        for j in range(N):
            col = X[:, j]
            obs = ~np.isnan(col)
            if obs.sum() == 0:
                out[:, j] = 0.0
                continue
            base = pd.Series(col).ffill().bfill().values
            a = np.pad(base, (pad, pad), mode='edge')
            mv = np.convolve(a, np.ones(window)/window, mode='valid')
            out[~obs, j] = mv[~obs]
        return out

    def introduce_block_missing(self, X, block_len=40, n_blocks=3, seed=0):
        """
        Introduce block missing patterns (long gaps) to stress test multivariate methods.
        Four per-column methods struggle; multivariate shines.
        
        Edge case handling: avoid edges to prevent interpolation issues.
        """
        X = X.copy()
        T, N = X.shape
        rng = np.random.default_rng(seed)
        for _ in range(n_blocks):
            j = rng.integers(0, N)
            # Avoid edges: ensure block doesn't start at 0 or end at T
            start = rng.integers(1, max(2, T - block_len - 1))
            end = min(start + block_len, T - 1)
            X[start:end, j] = np.nan
        return X

    def print_cross_feature_correlation(self, X, title="Data"):
        """
        Print cross-feature correlation analysis to verify multivariate structure.
        """
        # Use non-NaN values for correlation calculation
        X_clean = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        C = np.corrcoef(X_clean, rowvar=False)
        np.fill_diagonal(C, 0)
        mean_corr = np.nanmean(np.abs(C))
        max_corr = np.nanmax(np.abs(C))
        
        print(f"   📊 {title} cross-feature correlation:")
        print(f"      Mean |off-diagonal corr|: {mean_corr:.4f}")
        print(f"      Max |off-diagonal corr|: {max_corr:.4f}")
        
        if mean_corr < 0.1:
            print(f"      ⚠️  Low correlation - multivariate advantage may not be visible")
        elif mean_corr > 0.5:
            print(f"      ✅ High correlation - multivariate methods should excel")
        
        return mean_corr, max_corr

    def run_with_baselines(self, test_name, data, **cfg):
        """
        Compare MISSNET to a suite of fast baselines using the SAME mask for fairness.
        """
        print(f"\n📊 Baseline pack: {test_name}")
        noisy = self.generator.add_noise(
            data, cfg.get('noise_type','gaussian'), cfg.get('noise_level',0.1))
        mask_fn = dict(mcar=self.generator.introduce_missing_mcar,
                       mar=self.generator.introduce_missing_mar,
                       mnar=self.generator.introduce_missing_mnar)[cfg.get('missing_pattern','mcar')]
        Xmiss = mask_fn(noisy, cfg.get('missing_rate',0.2))
        M = np.isnan(Xmiss)

        baselines = [
            ("MISSNET (autotune)",  lambda X: missnet_impute(X, verbose=False, auto_tune=True)),
            ("Mean",                self.mean_impute),
            ("FFillBFill",          self.ffill_bfill_impute),
            ("LinearInterp",        self.linear_interp_impute),
            ("Spline(k=3)",         lambda X: self.spline_impute(X, order=3)),
            ("SavGol(11,3)",        lambda X: self.savgol_impute(X, window=11, polyorder=3)),
            ("Fourier(K=6)",        lambda X: self.fourier_global_impute(X, K=6, lam=1e-3)),
            ("KNN(k=5)",            lambda X: self.knn_impute_fast(X, n_neighbors=5)),
            ("IterSVD(rank=5)",     lambda X: self.iterative_svd_impute(X, rank=5, iters=8)),
            ("CenteredMA(w=9)",     lambda X: self.centered_window_mean_impute(X, window=9)),
        ]

        for label, fn in baselines:
            t0 = time.time()
            try:
                imp = fn(Xmiss.copy())
                dt = time.time() - t0
                m = self.calculate_metrics(noisy, imp, M, target="noisy")
                print(f"   {label:18} | MSE={m['mse']:.6f}  MAE={m['mae']:.6f}  ({dt:.2f}s)")

                # stash comparable record
                rec = dict(
                    test_name=f"{test_name} [{label}]",
                    data_shape=data.shape,
                    missing_pattern=cfg.get('missing_pattern','mcar'),
                    missing_rate=float(M.mean()),
                    noise_type=cfg.get('noise_type','gaussian'),
                    noise_level=cfg.get('noise_level',0.1),
                    execution_time=dt,
                    cells_per_sec=float(M.size / max(dt, 1e-9)),
                    success=True,
                    error=None,
                    **{f'noisy_{k}': v for k, v in m.items()},
                )
                self.results.append(rec)
            except Exception as e:
                print(f"   {label:18} | ❌ {e}")
                self.results.append(dict(
                    test_name=f"{test_name} [{label}]",
                    data_shape=data.shape,
                    missing_pattern=cfg.get('missing_pattern','mcar'),
                    missing_rate=float(M.mean()),
                    noise_type=cfg.get('noise_type','gaussian'),
                    noise_level=cfg.get('noise_level',0.1),
                    execution_time=time.time()-t0,
                    cells_per_sec=0.0,
                    success=False,
                    error=str(e),
                ))

    def run_multivariate_advantage_tests(self):
        """
        Test the multivariate advantage of MISSNET with strong cross-feature correlation.
        These tests demonstrate when MISSNET should outperform per-column methods.
        """
        print("\n🔗 Testing Multivariate Advantage")
        print("=" * 50)
        
        # Test 1: Strong cross-feature correlation with MCAR
        print("\n📊 Test 1: Strong Cross-Feature Correlation (MCAR)")
        X = self.generator.generate_base_signal(200, 10, 'mixed')
        X = self.generator.add_correlation(X, rho=0.8)   # <-- key line: strong correlation
        self.print_cross_feature_correlation(X, "Correlated Data")
        self.run_with_baselines("Multivariate Correlated", X, missing_rate=0.2)
        
        # Test 2: Strong correlation with MAR pattern
        print("\n📊 Test 2: Strong Correlation with MAR Pattern")
        self.run_with_baselines("Correlated MAR", X,
                               missing_pattern='mar', missing_rate=0.3)
        
        # Test 3: Strong correlation with MNAR pattern
        print("\n📊 Test 3: Strong Correlation with MNAR Pattern")
        self.run_with_baselines("Correlated MNAR", X,
                               missing_pattern='mnar', missing_rate=0.3)
        
        # Test 4: Block missing patterns (long gaps)
        print("\n📊 Test 4: Long Gaps (Block Missing) - Multivariate should shine")
        Xb = self.introduce_block_missing(X, block_len=40, n_blocks=4)
        actual_missing_rate = np.isnan(Xb).sum() / Xb.size
        print(f"   Block missing created: {actual_missing_rate:.1%} overall missing rate")
        self.run_with_baselines("Correlated Long Gaps", Xb, missing_rate=0.0)  # mask already applied
        
        # Test 5: Low correlation control case
        print("\n📊 Test 5: Low Correlation Control (per-column methods should compete)")
        X_low = self.generator.generate_base_signal(200, 10, 'mixed')
        X_low = self.generator.add_correlation(X_low, rho=0.1)   # <-- low correlation
        self.print_cross_feature_correlation(X_low, "Low Correlation Data")
        self.run_with_baselines("Low Correlation Control", X_low, missing_rate=0.2)
        
        # Test 6: Medium correlation
        print("\n📊 Test 6: Medium Correlation")
        X_med = self.generator.generate_base_signal(200, 10, 'mixed')
        X_med = self.generator.add_correlation(X_med, rho=0.5)   # <-- medium correlation
        self.print_cross_feature_correlation(X_med, "Medium Correlation Data")
        self.run_with_baselines("Medium Correlation", X_med, missing_rate=0.2)

    def run_network_effect_isolation_tests(self):
        """
        Isolate the network effect with ablation studies on the same mask.
        Tests whether the multivariate network component actually contributes.
        """
        print("\n🔬 Network Effect Isolation Tests")
        print("=" * 45)
        
        # Create strongly correlated data
        X = self.generator.generate_base_signal(200, 10, 'mixed')
        X = self.generator.add_correlation(X, rho=0.8)
        self.print_cross_feature_correlation(X, "Network Test Data")
        
        # Ablation study on correlated data
        print("\n🔬 Ablation Study (Correlated Data)")
        self.ablation_study(X, title="Ablation (correlated)",
                           base_kwargs={'missing_rate':0.3, 'noise_level':0.1})
        
        # Additional ablation with different configurations
        print("\n🔬 Extended Ablation Study")
        noisy = self.generator.add_noise(X, 'gaussian', 0.1)
        Xmiss = self.generator.introduce_missing_mcar(noisy, 0.3)
        M = np.isnan(Xmiss)
        
        extended_configs = [
            ("Default AUTOTUNE",       dict(auto_tune=True)),
            ("No spectral",            dict(auto_tune=True, use_spectral=False)),
            ("No robust",              dict(auto_tune=True, use_robust_loss=False)),
            ("No skeleton",            dict(auto_tune=True, use_skeleton=False)),
            ("No network (spectral only)", dict(auto_tune=True, use_spectral=True, alpha=0.0)),  # pure spectral
            ("Network only",           dict(auto_tune=True, use_spectral=False, alpha=1.0)),  # pure network
            ("Balanced",               dict(auto_tune=True, alpha=0.5, beta=0.1)),
        ]
        
        print(f"   Extended configurations on same mask:")
        for name, kw in extended_configs:
            try:
                imp = missnet_impute(Xmiss, verbose=False, **kw)
                m = self.calculate_metrics(noisy, imp, M, target="noisy")
                print(f"   {name:25} MSE={m['mse']:.6f}  Corr={m['correlation']:.4f}")
            except Exception as e:
                print(f"   {name:25} ❌ {str(e)}")

    def run_mar_mnar_stress_tests(self):
        """
        Stress test with MAR and MNAR patterns where multivariate methods should excel.
        Other variables help decide/impute the missing ones.
        """
        print("\n🎯 MAR/MNAR Stress Tests")
        print("=" * 35)
        
        # Create different correlation strengths for MAR/MNAR testing
        correlation_levels = [0.3, 0.6, 0.9]
        
        for rho in correlation_levels:
            print(f"\n📊 Testing with correlation strength rho={rho}")
            
            # Generate correlated data
            X = self.generator.generate_base_signal(150, 8, 'mixed')
            X = self.generator.add_correlation(X, rho=rho)
            self.print_cross_feature_correlation(X, f"rho={rho} Data")
            
            # Test MAR pattern
            print(f"   MAR Pattern (missing depends on other variables):")
            self.run_with_baselines(f"MAR rho={rho}", X,
                                   missing_pattern='mar', missing_rate=0.25)
            
            # Test MNAR pattern  
            print(f"   MNAR Pattern (missing depends on own values):")
            self.run_with_baselines(f"MNAR rho={rho}", X,
                                   missing_pattern='mnar', missing_rate=0.25)

    def run_block_missing_stress_tests(self):
        """
        Comprehensive block missing stress tests with various configurations.
        Fourier per-column struggles; multivariate shines on long gaps.
        """
        print("\n🕳️  Block Missing Stress Tests")
        print("=" * 40)
        
        # Test different block configurations
        block_configs = [
            {"block_len": 20, "n_blocks": 3, "name": "Short Blocks"},
            {"block_len": 40, "n_blocks": 4, "name": "Medium Blocks"}, 
            {"block_len": 60, "n_blocks": 2, "name": "Long Blocks"},
            {"block_len": 30, "n_blocks": 6, "name": "Many Short Blocks"},
        ]
        
        # Generate strongly correlated data for all tests
        X = self.generator.generate_base_signal(300, 12, 'mixed')
        X = self.generator.add_correlation(X, rho=0.8)
        self.print_cross_feature_correlation(X, "Block Test Data")
        
        for config in block_configs:
            print(f"\n📊 Testing {config['name']} (len={config['block_len']}, n={config['n_blocks']})")
            
            # Create block missing pattern
            Xb = self.introduce_block_missing(X, 
                                            block_len=config['block_len'], 
                                            n_blocks=config['n_blocks'],
                                            seed=42)
            actual_missing_rate = np.isnan(Xb).sum() / Xb.size
            print(f"   Created {actual_missing_rate:.1%} missing rate")
            
            # Run baselines
            self.run_with_baselines(f"Block {config['name']}", Xb, missing_rate=0.0)
            
            # Also test with some additional random missingness
            Xb_extra = Xb.copy()
            extra_mask = np.random.random(Xb.shape) < 0.1  # 10% additional random missing
            Xb_extra[extra_mask] = np.nan
            total_missing = np.isnan(Xb_extra).sum() / Xb_extra.size
            print(f"   With extra random missing: {total_missing:.1%}")
            self.run_with_baselines(f"Block {config['name']} + Random", Xb_extra, missing_rate=0.0)
    
    def quick_ablation_study(self, data, title="Quick Ablation", base_kwargs=None):
        """
        Comprehensive quick ablation study testing multiple configurations.
        Tests individual feature contributions and combinations on the same mask.
        """
        base = dict(missing_pattern='mcar', missing_rate=0.25, noise_type='gaussian', noise_level=0.15)
        base.update(base_kwargs or {})
        print(f"\n� {title}: Comprehensive Feature Ablation (same mask)")
        print(f"   Data shape: {data.shape}")
        print(f"   Missing rate: {base['missing_rate']:.1%}, Noise level: {base['noise_level']}")
        
        # Create the test data (same mask for all configurations)
        noisy = self.generator.add_noise(data, base['noise_type'], base['noise_level'])
        Xmiss = self.generator.introduce_missing_mcar(noisy, base['missing_rate'])
        M = np.isnan(Xmiss)
        
        print(f"   Actual missing rate: {M.mean():.1%}")
        print(f"   Total missing cells: {M.sum()}")
        
        # Comprehensive configuration set for ablation
        ablation_configs = [
            # Baseline configurations
            ("🤖 Default AUTOTUNE", dict(auto_tune=True)),
            ("🎛️  Fixed Default", dict(auto_tune=False, alpha=0.5, beta=0.1, L=10, n_cl=1)),
            
            # Individual feature toggles
            ("🚫 No Spectral", dict(auto_tune=True, use_spectral=False)),
            ("🛡️  No Robust Loss", dict(auto_tune=True, use_robust_loss=False)),
            ("🦴 No Skeleton", dict(auto_tune=True, use_skeleton=False)),
            ("🔧 No Cholesky", dict(auto_tune=True, use_cholesky=False)),
            
            # Pure component tests
            ("🌊 Pure Spectral", dict(auto_tune=True, use_spectral=True, alpha=0.0)),
            ("🕸️  Pure Network", dict(auto_tune=True, use_spectral=False, alpha=1.0)),
            
            # Alpha (network vs spectral) balance tests
            ("⚖️  Alpha 0.25", dict(auto_tune=True, alpha=0.25)),
            ("⚖️  Alpha 0.75", dict(auto_tune=True, alpha=0.75)),
            
            # Beta (sparsity) tests
            ("🎯 Beta 0.05", dict(auto_tune=True, beta=0.05)),
            ("🎯 Beta 0.2", dict(auto_tune=True, beta=0.2)),
            
            # Latent dimension tests
            ("📏 L=5", dict(auto_tune=True, L=5)),
            ("📏 L=20", dict(auto_tune=True, L=20)),
            
            # Regime/cluster tests
            ("🏗️  n_cl=1", dict(auto_tune=True, n_cl=1)),
            ("🏗️  n_cl=5", dict(auto_tune=True, n_cl=5)),
            
            # Robust loss specific tests
            ("🔒 Huber Only", dict(auto_tune=True, use_robust_loss=True, use_spectral=False, alpha=0.0)),
            ("🔒 L1 Only", dict(auto_tune=True, use_robust_loss=True, use_spectral=False, alpha=0.0, robust_type='l1')),
            
            # Conservative vs aggressive
            ("🐌 Conservative", dict(auto_tune=True, alpha=0.3, beta=0.2, L=5, max_iteration=50)),
            ("🚀 Aggressive", dict(auto_tune=True, alpha=0.7, beta=0.05, L=15, max_iteration=200)),
        ]
        
        print(f"\n   Testing {len(ablation_configs)} configurations...")
        print(f"   {'Config':25} | {'MSE':>10} | {'MAE':>10} | {'Corr':>8} | {'Time':>8} | {'ΔMSE':>8}")
        print(f"   {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        results = []
        baseline_mse = None
        
        for name, config in ablation_configs:
            try:
                start_time = time.time()
                imp = missnet_impute(Xmiss, verbose=False, **config)
                execution_time = time.time() - start_time
                
                m = self.calculate_metrics(noisy, imp, M, target="noisy")
                
                # Calculate relative improvement
                if baseline_mse is None:
                    baseline_mse = m['mse']
                    delta_mse = 0.0
                    delta_str = "baseline"
                else:
                    delta_mse = (baseline_mse - m['mse']) / baseline_mse * 100
                    delta_str = f"{delta_mse:+.1f}%"
                
                results.append({
                    'name': name,
                    'config': config,
                    'mse': m['mse'],
                    'mae': m['mae'],
                    'correlation': m['correlation'],
                    'time': execution_time,
                    'delta_mse': delta_mse
                })
                
                print(f"   {name:25} | {m['mse']:10.6f} | {m['mae']:10.6f} | {m['correlation']:8.4f} | {execution_time:8.2f}s | {delta_str:>8}")
                
            except Exception as e:
                print(f"   {name:25} | {'FAILED':>10} | {'ERROR':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
                results.append({
                    'name': name,
                    'config': config,
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'correlation': 0.0,
                    'time': float('inf'),
                    'delta_mse': -100.0,
                    'error': str(e)
                })
        
        # Summary analysis
        print(f"\n   📊 Quick Ablation Summary:")
        successful_results = [r for r in results if r['mse'] != float('inf')]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['mse'])
            worst_result = max(successful_results, key=lambda x: x['mse'])
            
            print(f"      🏆 Best: {best_result['name']} (MSE={best_result['mse']:.6f}, Δ={best_result['delta_mse']:+.1f}%)")
            print(f"      ⚠️  Worst: {worst_result['name']} (MSE={worst_result['mse']:.6f}, Δ={worst_result['delta_mse']:+.1f}%)")
            
            # Feature impact analysis
            feature_impacts = []
            
            # Spectral impact
            spectral_default = next((r for r in successful_results if 'Default AUTOTUNE' in r['name']), None)
            spectral_off = next((r for r in successful_results if 'No Spectral' in r['name']), None)
            if spectral_default and spectral_off:
                impact = (spectral_off['mse'] - spectral_default['mse']) / spectral_default['mse'] * 100
                feature_impacts.append(("Spectral Transform", impact))
            
            # Robust loss impact
            robust_default = next((r for r in successful_results if 'Default AUTOTUNE' in r['name']), None)
            robust_off = next((r for r in successful_results if 'No Robust Loss' in r['name']), None)
            if robust_default and robust_off:
                impact = (robust_off['mse'] - robust_default['mse']) / robust_default['mse'] * 100
                feature_impacts.append(("Robust Loss", impact))
            
            # Skeleton impact
            skeleton_default = next((r for r in successful_results if 'Default AUTOTUNE' in r['name']), None)
            skeleton_off = next((r for r in successful_results if 'No Skeleton' in r['name']), None)
            if skeleton_default and skeleton_off:
                impact = (skeleton_off['mse'] - skeleton_default['mse']) / skeleton_default['mse'] * 100
                feature_impacts.append(("Skeleton", impact))
            
            if feature_impacts:
                print(f"      🔬 Feature Impacts (degradation when disabled):")
                for feature, impact in sorted(feature_impacts, key=lambda x: x[1], reverse=True):
                    print(f"         • {feature}: {impact:+.1f}%")
            
            # Performance vs speed tradeoff
            fastest = min(successful_results, key=lambda x: x['time'])
            print(f"      ⚡ Fastest: {fastest['name']} ({fastest['time']:.2f}s, MSE={fastest['mse']:.6f})")
        
        return results

    def ablation_study(self, data, title="Ablation", base_kwargs=None):
        """Ablation study toggling one flag at a time on the same mask."""
        base = dict(missing_pattern='mcar', missing_rate=0.3, noise_type='gaussian', noise_level=0.2)
        base.update(base_kwargs or {})
        print(f"\n🔬 {title}: toggling features (same mask)")
        noisy = self.generator.add_noise(data, base['noise_type'], base['noise_level'])
        Xmiss = self.generator.introduce_missing_mcar(noisy, base['missing_rate'])
        M = np.isnan(Xmiss)

        configs = [
            ("Default AUTOTUNE", dict(auto_tune=True)),
            ("No spectral",      dict(auto_tune=True, use_spectral=False)),
            ("No robust",        dict(auto_tune=True, use_robust_loss=False)),
            ("No skeleton",      dict(auto_tune=True, use_skeleton=False)),
        ]
        for name, kw in configs:
            imp = missnet_impute(Xmiss, verbose=False, **kw)
            m = self.calculate_metrics(noisy, imp, M, target="noisy")
            print(f"   {name:18} MSE={m['mse']:.6f}  Corr={m['correlation']:.4f}")
    
    def run_fourier_comparison_tests(self):
        """Run comprehensive Fourier transform comparison tests."""
        print("🌊 Starting Fourier Transform Comparison Tests")
        print("=" * 60)
        
        # Test configurations for Fourier comparison
        fourier_test_configs = [
            # Different signal types
            {
                'name': 'Fourier Mixed Signal',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            {
                'name': 'Fourier Seasonal Signal',
                'T': 200, 'N': 10, 'signal_type': 'seasonal',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            {
                'name': 'Fourier Trend Signal',
                'T': 200, 'N': 10, 'signal_type': 'trend',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            {
                'name': 'Fourier Regime Signal',
                'T': 200, 'N': 10, 'signal_type': 'regime',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            
            # Different missing rates
            {
                'name': 'Fourier Low Missing',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_rate': 0.1, 'noise_level': 0.1
            },
            {
                'name': 'Fourier High Missing',
                'T': 200, 'N': 10, 'signal_type': 'mixed',
                'missing_rate': 0.4, 'noise_level': 0.1
            },
            
            # Different noise types
            {
                'name': 'Fourier Gaussian Noise',
                'T': 200, 'N': 10, 'signal_type': 'seasonal',
                'missing_rate': 0.2, 'noise_type': 'gaussian', 'noise_level': 0.15
            },
            {
                'name': 'Fourier Outlier Noise',
                'T': 200, 'N': 10, 'signal_type': 'seasonal',
                'missing_rate': 0.2, 'noise_type': 'outlier', 'noise_level': 0.15
            },
            
            # Different data sizes
            {
                'name': 'Fourier Small Data',
                'T': 100, 'N': 5, 'signal_type': 'seasonal',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            {
                'name': 'Fourier Large Data',
                'T': 400, 'N': 20, 'signal_type': 'seasonal',
                'missing_rate': 0.2, 'noise_level': 0.1
            },
            
            # Different correlation strengths
            {
                'name': 'Fourier Low Correlation',
                'T': 200, 'N': 10, 'signal_type': 'seasonal',
                'correlation_strength': 0.1, 'missing_rate': 0.2
            },
            {
                'name': 'Fourier High Correlation',
                'T': 200, 'N': 10, 'signal_type': 'seasonal',
                'correlation_strength': 0.8, 'missing_rate': 0.2
            },
        ]
        
        # Run all Fourier comparison tests
        for config in fourier_test_configs:
            test_name = config.pop('name')
            T = config.pop('T', 200)
            N = config.pop('N', 10)
            signal_type = config.pop('signal_type', 'mixed')
            correlation_strength = config.pop('correlation_strength', 0.5)
            
            print(f"\n🔄 Running Fourier test: {test_name}")
            print(f"   Data shape: {T} × {N}, Signal: {signal_type}")
            
            # Generate data
            data = self.generator.generate_base_signal(T, N, signal_type)
            data = self.generator.add_correlation(data, correlation_strength)
            
            # Set base parameters
            base_params = {
                'missing_pattern': 'mcar',
                'missing_rate': 0.2,
                'noise_type': 'gaussian',
                'noise_level': 0.1,
                **config
            }
            
            # Run Fourier comparison
            self.run_fourier_comparison(test_name, data, **base_params)
            
            # Clean up memory
            gc.collect()
        
        # Generate Fourier-specific summary
        self.generate_fourier_summary()
    
    def generate_fourier_summary(self):
        """Generate a summary of Fourier comparison results."""
        df = pd.DataFrame(self.results)
        
        # Filter for Fourier comparison results
        fourier_results = df[df['test_name'].str.contains('[No Fourier|With Fourier]', na=False)]
        
        if len(fourier_results) == 0:
            print("   No Fourier comparison results found")
            return
        
        print("\n📊 Fourier Transform Performance Summary")
        print("=" * 50)
        
        # Group by base test name (remove [No Fourier] and [With Fourier] suffixes)
        fourier_results['base_test'] = fourier_results['test_name'].str.replace(r' \[.*\]$', '', regex=True)
        
        summary_stats = []
        for base_test in fourier_results['base_test'].unique():
            test_group = fourier_results[fourier_results['base_test'] == base_test]
            
            no_fourier = test_group[test_group['test_name'].str.contains('No Fourier')]
            with_fourier = test_group[test_group['test_name'].str.contains('With Fourier')]
            
            if len(no_fourier) > 0 and len(with_fourier) > 0:
                mse_no = no_fourier['noisy_mse'].iloc[0]
                mse_with = with_fourier['noisy_mse'].iloc[0]
                time_no = no_fourier['execution_time'].iloc[0]
                time_with = with_fourier['execution_time'].iloc[0]
                
                mse_improvement = (mse_no - mse_with) / mse_no * 100
                time_overhead = (time_with - time_no) / time_no * 100
                
                summary_stats.append({
                    'test': base_test,
                    'mse_improvement': mse_improvement,
                    'time_overhead': time_overhead,
                    'better_mse': mse_improvement > 0,
                    'faster_time': time_overhead < 0
                })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            
            print(f"Total comparisons: {len(summary_df)}")
            print(f"MSE improvements: {summary_df['better_mse'].sum()}/{len(summary_df)} ({summary_df['better_mse'].mean()*100:.1f}%)")
            print(f"Time improvements: {summary_df['faster_time'].sum()}/{len(summary_df)} ({summary_df['faster_time'].mean()*100:.1f}%)")
            print(f"Average MSE improvement: {summary_df['mse_improvement'].mean():+.2f}%")
            print(f"Average time overhead: {summary_df['time_overhead'].mean():+.1f}%")
            
            # Best and worst cases
            best_mse = summary_df.loc[summary_df['mse_improvement'].idxmax()]
            worst_mse = summary_df.loc[summary_df['mse_improvement'].idxmin()]
            
            print(f"\n🏆 Best MSE improvement: {best_mse['test']} ({best_mse['mse_improvement']:+.2f}%)")
            print(f"⚠️  Worst MSE performance: {worst_mse['test']} ({worst_mse['mse_improvement']:+.2f}%)")
            
            # Cases where Fourier is beneficial
            beneficial = summary_df[summary_df['mse_improvement'] > 5]  # >5% improvement
            if len(beneficial) > 0:
                print(f"\n✅ Fourier beneficial (>5% improvement) in {len(beneficial)} cases:")
                for _, row in beneficial.iterrows():
                    print(f"   - {row['test']}: {row['mse_improvement']:+.2f}% MSE, {row['time_overhead']:+.1f}% time")
            
            # Cases where Fourier is harmful
            harmful = summary_df[summary_df['mse_improvement'] < -5]  # >5% degradation
            if len(harmful) > 0:
                print(f"\n❌ Fourier harmful (>5% degradation) in {len(harmful)} cases:")
                for _, row in harmful.iterrows():
                    print(f"   - {row['test']}: {row['mse_improvement']:+.2f}% MSE, {row['time_overhead']:+.1f}% time")

    def save_results(self, filepath: str):
        """Save test results to CSV file."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filepath, index=False)
            print(f"💾 Results saved to {filepath}")
        else:
            print("No results to save")


def main():
    """Main testing function with CLI argument support."""
    parser = argparse.ArgumentParser(description='MISSNET Synthetic Data Testing Suite')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick mode (3 small tests + autotune smoke)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--autotune-only', action='store_true',
                       help='Run only autotune-specific tests')
    parser.add_argument('--fourier-only', action='store_true',
                       help='Run only Fourier transform comparison tests')
    parser.add_argument('--multivariate-only', action='store_true',
                       help='Run only multivariate advantage tests')
    parser.add_argument('--network-only', action='store_true',
                       help='Run only network effect isolation tests')
    parser.add_argument('--mar-mnar-only', action='store_true',
                       help='Run only MAR/MNAR stress tests')
    parser.add_argument('--block-only', action='store_true',
                       help='Run only block missing stress tests')
    
    args = parser.parse_args()
    
    print("🧪 MISSNET Synthetic Data Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = MissNetTester(random_seed=42)
    
    if args.fourier_only:
        # Run only Fourier comparison tests
        print("\n🌊 Running Fourier Transform Comparison Tests Only")
        tester.run_fourier_comparison_tests()
        
    elif args.autotune_only:
        # Run only autotune-specific tests
        print("\n🔧 Running Autotune-Specific Tests Only")
        tester.autotune_smoke()
        tester.run_autotune_comparison_tests()
        tester.test_autotune_parameter_selection()
        
    elif args.multivariate_only:
        # Run only multivariate advantage tests
        print("\n🔗 Running Multivariate Advantage Tests Only")
        tester.run_multivariate_advantage_tests()
        
    elif args.network_only:
        # Run only network effect isolation tests
        print("\n🔬 Running Network Effect Isolation Tests Only")
        tester.run_network_effect_isolation_tests()
        
    elif args.mar_mnar_only:
        # Run only MAR/MNAR stress tests
        print("\n🎯 Running MAR/MNAR Stress Tests Only")
        tester.run_mar_mnar_stress_tests()
        
    elif args.block_only:
        # Run only block missing stress tests
        print("\n🕳️ Running Block Missing Stress Tests Only")
        tester.run_block_missing_stress_tests()
        
    elif args.quick:
        # Quick mode: run smoke test + enhanced multivariate tests
        print("\n⚡ QUICK MODE: Running essential tests only")
        
        # Run autotune smoke test first
        tester.autotune_smoke()
        
        # Run enhanced quick baseline comparisons with correlation
        print("\n📊 Quick Baseline Comparisons (with correlation)")
        data_small = tester.generator.generate_base_signal(50, 5, 'mixed')
        data_small = tester.generator.add_correlation(data_small, rho=0.8)  # Add correlation!
        tester.print_cross_feature_correlation(data_small, "Quick Test Data")
        tester.run_with_baselines("Quick Baseline", data_small, missing_rate=0.2)
        
        # Quick multivariate advantage test
        print("\n🔗 Quick Multivariate Advantage Test")
        tester.run_multivariate_advantage_tests()
        
        # Quick comprehensive ablation study
        tester.quick_ablation_study(data_small, "Quick Comprehensive Ablation")
        
        # One pairwise comparison
        print("\n🔄 Quick Pairwise Comparison")
        tester.run_pair_autotune_vs_fixed("Quick Pair", data_small, missing_rate=0.2)
        
        # Quick Fourier comparison
        print("\n🌊 Quick Fourier Comparison")
        tester.run_fourier_comparison("Quick Fourier", data_small, missing_rate=0.2)
        
    else:
        # Full testing suite with enhanced multivariate tests
        # Run autotune smoke test first
        print("\n🧭 Phase 0: Autotuner Smoke Test")
        tester.autotune_smoke()
        
        # NEW: Run multivariate advantage tests first
        print("\n🔗 Phase 1: Multivariate Advantage Tests")
        tester.run_multivariate_advantage_tests()
        
        # Run comprehensive tests with autotune enabled
        print("\n🚀 Phase 2: Comprehensive Tests with Autotune")
        tester.run_comprehensive_tests()
        
        # Run baseline comparisons for key scenarios
        print("\n📊 Phase 3: Baseline Comparisons")
        test_data = tester.generator.generate_base_signal(200, 10, 'mixed')
        test_data = tester.generator.add_correlation(test_data, rho=0.8)  # Add correlation!
        tester.print_cross_feature_correlation(test_data, "Baseline Test Data")
        tester.run_with_baselines("Baseline Mixed", test_data, missing_rate=0.2)
        
        seasonal_data = tester.generator.generate_base_signal(200, 10, 'seasonal')
        seasonal_data = tester.generator.add_correlation(seasonal_data, rho=0.6)
        tester.print_cross_feature_correlation(seasonal_data, "Baseline Seasonal Data")
        tester.run_with_baselines("Baseline Seasonal", seasonal_data, missing_rate=0.2)
        
        # Run pairwise autotune vs fixed comparisons
        print("\n🔄 Phase 4: Pairwise Autotune vs Fixed Comparisons")
        tester.run_pair_autotune_vs_fixed("Pairwise Mixed", test_data, missing_rate=0.2)
        tester.run_pair_autotune_vs_fixed("Pairwise Seasonal", seasonal_data, missing_rate=0.2)
        
        # NEW: Run network effect isolation tests
        print("\n🔬 Phase 5: Network Effect Isolation Tests")
        tester.run_network_effect_isolation_tests()
        
        # NEW: Run MAR/MNAR stress tests
        print("\n🎯 Phase 6: MAR/MNAR Stress Tests")
        tester.run_mar_mnar_stress_tests()
        
        # NEW: Run block missing stress tests
        print("\n🕳️ Phase 7: Block Missing Stress Tests")
        tester.run_block_missing_stress_tests()
        
        # Run Fourier comparison tests
        print("\n🌊 Phase 8: Fourier Transform Performance Comparison")
        tester.run_fourier_comparison_tests()
        
        # Run ablation studies
        print("\n🔬 Phase 9: Ablation Studies")
        tester.ablation_study(test_data, "Ablation Mixed")
        tester.ablation_study(seasonal_data, "Ablation Seasonal")
        
        # Run autotune comparison tests
        print("\n🔧 Phase 10: Detailed Autotune vs Manual Parameter Comparison")
        tester.run_autotune_comparison_tests()
        
        # Test autotune parameter selection logic
        print("\n🎯 Phase 11: Autotune Parameter Selection Analysis")
        tester.test_autotune_parameter_selection()
    
    # Generate and print report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save results
    if args.fourier_only:
        tester.save_results('missnet_fourier_results.csv')
        viz_file = 'missnet_fourier_analysis.png'
    elif args.autotune_only:
        tester.save_results('missnet_autotune_results.csv')
        viz_file = 'missnet_autotune_analysis.png'
    elif args.quick:
        tester.save_results('missnet_quick_results.csv')
        viz_file = 'missnet_quick_analysis.png'
    else:
        tester.save_results('missnet_test_results.csv')
        viz_file = 'missnet_performance_analysis.png'
    
    # Create visualizations
    if not args.no_visualizations:
        tester.create_visualizations(viz_file)
    
    # Print completion message
    if args.fourier_only:
        print("\n🎉 Fourier testing completed!")
        print(f"📄 Results saved to: missnet_fourier_results.csv")
        if not args.no_visualizations:
            print(f"📊 Visualizations saved to: {viz_file}")
    elif args.autotune_only:
        print("\n🎉 Autotune testing completed!")
        print(f"📄 Results saved to: missnet_autotune_results.csv")
        if not args.no_visualizations:
            print(f"📊 Visualizations saved to: {viz_file}")
    elif args.quick:
        print("\n⚡ Quick testing completed!")
        print(f"📄 Results saved to: missnet_quick_results.csv")
        if not args.no_visualizations:
            print(f"📊 Visualizations saved to: {viz_file}")
    else:
        print("\n🎉 Full testing completed!")
        print(f"📄 Results saved to: missnet_test_results.csv")
        if not args.no_visualizations:
            print(f"📊 Visualizations saved to: {viz_file}")


def run_autotune_only():
    """Run only autotune-specific tests."""
    print("🔧 MISSNET Autotune-Specific Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = MissNetTester(random_seed=42)
    
    # Run autotune comparison tests
    print("\n🔧 Phase 1: Autotune vs Manual Parameter Comparison")
    tester.run_autotune_comparison_tests()
    
    # Test autotune parameter selection logic
    print("\n🎯 Phase 2: Autotune Parameter Selection Analysis")
    tester.test_autotune_parameter_selection()
    
    # Generate and print report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save results
    tester.save_results('missnet_autotune_results.csv')
    
    # Create visualizations
    tester.create_visualizations('missnet_autotune_analysis.png')
    
    print("\n🎉 Autotune testing completed!")
    print("📄 Results saved to: missnet_autotune_results.csv")
    print("📊 Visualizations saved to: missnet_autotune_analysis.png")


if __name__ == "__main__":
    main()
