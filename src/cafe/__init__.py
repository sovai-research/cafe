"""CAFE -- Causal Adaptive Factor Estimation.

A zero-config, CPU-first, strictly point-in-time (no look-ahead) imputation model
for time-series and panel data. One causal estimator that also yields per-cell
uncertainty, latent factors, anomaly scores, an additive decomposition, a
dependency network, and forecasts -- from the same forward pass.

    import cafe
    filled = cafe.impute(df)            # numpy / pandas / polars, 1D or 2D
    res    = cafe.CAFE().run(df)        # rich result
    res.uncertainty; res.factors(); res.anomaly_scores(); res.forecast  # ...
"""
from .benchmark import benchmark
from .conformal import ConformalCalibrator, conformal_multipliers
from .model import CAFE, CafeResult, impute

__all__ = ["CAFE", "CafeResult", "impute", "benchmark",
           "ConformalCalibrator", "conformal_multipliers", "__version__"]
__version__ = "0.1.0"
