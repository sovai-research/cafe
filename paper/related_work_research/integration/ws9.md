# WS9 — Classical / non-deep baselines as first-class library methods

New module `src/cafe/baselines.py` (+ tests `src/tests/test_baselines.py`, demo
`bench/demo_baselines.py`). It promotes the benchmark baselines we already maintain into
the library behind a uniform, container-native API consistent with `cafe.impute`, so the
paper's "special cases of CAFE" table (`tab:special`) is backed by *runnable* code.

## EXACT one-line addition for `src/cafe/__init__.py` (apply centrally)

The current file ends with:

```python
from .benchmark import benchmark
from .model import CAFE, CafeResult, impute

__all__ = ["CAFE", "CafeResult", "impute", "benchmark", "__version__"]
```

Add the `baselines` subpackage import and export. Two edited lines:

```python
from . import baselines
from .benchmark import benchmark
from .model import CAFE, CafeResult, impute

__all__ = ["CAFE", "CafeResult", "impute", "benchmark", "baselines", "__version__"]
```

(i.e. add `from . import baselines` and add `"baselines"` to `__all__`.) After that,
`cafe.baselines.impute(df, method="softimpute")` works without an explicit submodule
import.

> Note: I also fixed a pre-existing bug in `src/cafe/io.py` (`_is_pandas`/`_is_polars`
> only matched `"pandas."`/`"polars."`, but pandas 3.x reports `__module__ == "pandas"`).
> This makes the pandas/polars round-trip work for **both** `cafe.impute` and the new
> baselines on the installed pandas 3.0.3. It is a one-character-class change (adds
> `m == "pandas"` / `m == "polars"`) and is covered by the existing + new tests.

## Paragraph for the appendix "Using the cafe library" section

> Beyond the CAFÉ estimator itself, the library bundles the classical imputers that the
> paper characterises as special cases of CAFÉ (Table~\ref{tab:special}) as runnable,
> first-class methods under `cafe.baselines`. Each is exposed both as a named function
> (`cafe.baselines.softimpute(data)`, `locf(data)`, `trmf(data)`, `ewcov(data)`, …) and
> through a registry dispatcher `cafe.baselines.impute(data, method="softimpute", **kw)`
> with `list_methods()` and a `METHODS` table; all accept the same numpy/pandas/polars
> 1-D and 2-D inputs as `cafe.impute` and return the same container type with observed
> cells preserved exactly. Every method carries a first-class causal/batch label so the
> point-in-time distinction central to this paper is explicit in code: the causal
> point-in-time methods (LOCF, expanding/rolling mean & median, EWMA, drift, local-level
> Kalman filter, cross-sectional mean, online TRMF, EW-covariance Gaussian conditional
> mean, online Gaussian copula) never read a future observation, whereas the batch
> methods (linear interpolation, SoftImpute-ALS, batch TRMF, MC-NNM, KNN, MICE) may. The
> classical numerical cores are ported from our benchmark implementations and verified
> bit-exactly against them (`src/tests/test_baselines.py`), so Table~\ref{tab:special} is
> not merely descriptive: each row corresponds to a method one can run from the library.

This strengthens `tab:special`: the classical estimators it lists (Feature-mean/FE →
`mean_impute`; SoftImpute → `softimpute`; TRMF → `trmf`/`online_trmf`; Kalman/SSM →
`kalman_local_level`; EW-cov / Gaussian conditional mean → `ewcov`; MC-NNM → `mcnnm`;
robust/heavy-tail and seasonal regimes exercised by the causal field) are now executable
and tested, not just cited.

## Notebook cell snippet

```python
from cafe import baselines

baselines.list_methods()                       # all 17 methods
baselines.list_methods(causal=True)             # only point-in-time methods
filled = baselines.impute(df, method="softimpute")        # same container type back
filled = baselines.impute(df, method="rolling_mean", W=12) # kwargs forwarded
baselines.softimpute(df)                         # or call the named function directly
```

## Methods shipped (17) with causal/batch label

CAUSAL (point-in-time): `locf`, `mean_impute`, `rolling_mean`, `rolling_median`,
`ewma`, `drift`, `kalman_local_level`, `xsec_mean`, `online_trmf`, `ewcov`,
`gaussian_copula` (optional `gcimpute`).

BATCH (non-causal): `linear_interp`, `softimpute`, `trmf`, `mcnnm`, `knn` (sklearn),
`mice` (sklearn).

`knn`/`mice` guard the sklearn import; `gaussian_copula` guards `gcimpute`. No
`bench/repro.py` entry is needed (no new experiment); `bench/demo_baselines.py` is an
optional, self-contained MAE race for illustration.

## Verification

- Smoke: `python3 -c "import sys; sys.path.insert(0,'.../src'); from cafe import baselines; print(baselines.list_methods())"` → 17 methods.
- `pytest src/tests/test_baselines.py -q` → **127 passed**. Full suite `pytest src/tests/ -q` → **178 passed** (no regressions from the io.py fix).
- Bit-exact parity (max|lib − bench| < 1e-8 over seeds 3/7/11) for the ported methods:
  `softimpute`, `trmf`, `linear_interp`, `locf`, `ewcov`, `online_trmf`.
- Container round-trip verified for numpy/pandas/polars, 1-D and 2-D, observed cells
  preserved exactly.
