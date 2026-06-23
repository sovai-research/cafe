# Real-world imputation datasets for CAFE

This document records the diverse, **real**, **small/fast** datasets added so the
CAFE paper has genuine real-world evidence (beyond synthetic masks and the two
existing panels). All loaders live in `bench/datasets_real.py` and cache
z-scored, finite `(time x features)` matrices to `data/<name>_clean.npy`.

Selection criteria: REAL (not synthetic); SMALL/fast (roughly <= 12k rows and
<= 150 cols); DIVERSE domains (macro / finance / environment / energy / traffic);
genuine multivariate **panel** structure; and ideally **native** missingness.
All sources are freely downloadable without authentication.

---

## Rank-ordered candidate table (~10 considered)

| rank | dataset | domain | approx shape (raw) | native missing? | why it stresses imputation | source | license | fetched? |
|---|---|---|---|---|---|---|---|---|
| 1 | **FRED-MD** | macro | ~728 x 128 | **yes** (ragged edge, ~1%) | THE canonical dynamic-factor-model panel; strong low-rank + AR structure; real publication-vintage missingness; CAFE's home turf | St. Louis Fed (McCracken & Ng) | public domain (US gov data) | **YES** |
| 2 | **AirQualityUCI** (de Vito) | environment | ~9357 x 13 | **yes, heavy** (-200 sentinel, ~7-90% per col) | hardware-sensor dropouts, one column ~90% missing, strong diurnal + cross-sensor correlation | UCI ML Repo #360/#387 | CC BY 4.0 | **YES** |
| 3 | **Exchange-Rate** | finance | 7588 x 8 | no | smooth, highly cross-correlated, near-low-rank FX series; classic LSTNet benchmark; tiny + fast | Lai et al. (LSTNet) GitHub | research/MIT-style (LSTNet repo) | **YES** |
| 4 | **Appliances Energy** | energy | 19735 x 29 -> sliced | no | dense indoor T/RH + weather + load; mixed scales; high-frequency 10-min cadence | UCI ML Repo #374 (Candanedo) | CC BY 4.0 | **YES** (4x row-subsampled, rv1/rv2 dropped) |
| 5 | **Traffic (PEMS)** | traffic | 17544 x 862 -> sliced | no | spatial road-occupancy panel; strongly correlated neighbouring sensors; ideal for low-rank + block-missing stress | Lai et al. (LSTNet) GitHub / Caltrans PEMS | research (LSTNet repo) / Caltrans public | **YES** (first 100 sensors, 5x row-subsampled) |
| 6 | FRED-QD | macro | ~260 x 245 | yes | quarterly macro companion to FRED-MD | St. Louis Fed | public domain | no — only `.rda`/`.RData` mirrors found (not Python-parseable without R); FRED-MD already covers macro |
| 7 | Solar-Energy | energy | 52560 x 137 | no | many correlated PV plants | LSTNet GitHub | research | no — redundant energy domain; very long |
| 8 | Electricity (LD2011) | energy | 26304 x 321 | no | client load curves | LSTNet GitHub / UCI | research / CC BY 4.0 | no — too wide; energy already covered by Appliances |
| 9 | METR-LA | traffic | 34272 x 207 | block | spatial traffic speed | DCRNN GitHub (.h5) | research | no — traffic already covered by PEMS slice; .h5 dependency |
| 10 | PhysioNet/MIMIC vitals | health | irregular | yes, extreme | clinical irregular sampling | PhysioNet | requires credentialed access / DUA | no — auth/DUA gate violates "no-auth" constraint |

**Final chosen 5 (one per domain, all fetched):** FRED-MD (macro), Exchange-Rate
(finance), AirQualityUCI (environment), Appliances Energy (energy), Traffic-PEMS
(traffic). Health was intentionally dropped: every high-quality clinical panel
(PhysioNet/MIMIC) is behind a credentialed data-use agreement, violating the
freely-downloadable-without-auth constraint.

---

## Final obtained shapes & missingness (verified by `python3 bench/datasets_real.py`)

| name | domain | clean shape (saved `_clean.npy`) | clean missing | native view | native missing | notes |
|---|---|---|---|---|---|---|
| `fredmd` | macro | **712 x 123** | 0.0% | `fredmd_native.npy` (729 x 128) + `fredmd_obsmask.npy` | 1.0% | clean = cols >=95% coverage, then dropna rows |
| `exchange` | finance | **7588 x 8** | 0.0% | — | — | complete; kept in full |
| `airquality` | environment | **6941 x 12** | 0.0% | `airquality_native.npy` (9326 x 12) + `airquality_obsmask.npy` | 7.0% | -200 sentinel masked; ~90%-empty NMHC(GT) column dropped from both views; clean = dropna block |
| `appliances` | energy | **4934 x 26** | 0.0% | — | — | every 4th row (40-min cadence); dropped date, rv1, rv2 |
| `traffic` | traffic | **3509 x 100** | 0.0% | — | — | first 100 of 862 sensors; every 5th hour |

All clean matrices are z-scored per column (mean 0, std 1), `float64`, and fully
finite (asserted in each loader). The two datasets with genuine native
missingness (`fredmd`, `airquality`) additionally expose a z-scored native matrix
with NaNs preserved plus a boolean observed-mask, so real-missingness imputation
(not just synthetic masks) can be evaluated later.

---

## Provenance & download details

- **FRED-MD** — `data/fredmd_current.csv`. The official St. Louis Fed
  `current.csv` endpoint
  (`https://www.stlouisfed.org/-/media/.../fred-md/monthly/current.csv`) was
  **unreachable from this environment** (HTTP/2 `INTERNAL_ERROR`, then repeated
  timeouts/`AccessDenied` on every host variant). Obtained instead from the
  GitHub mirror that bundles the verbatim file:
  `https://raw.githubusercontent.com/geoluna/FactorModels/master/data/current.csv`.
  Format verified: header row of series names, a `Transform:` code row (dropped),
  `sasdate` first column, 128 series, 1959-01 to 2019-09 (~729 months). This is a
  2019 publication vintage of the McCracken & Ng panel. Underlying data are US
  government series (public domain).

- **AirQualityUCI** — `data/airquality_uci.zip` from
  `https://archive.ics.uci.edu/static/public/360/air+quality.zip`. de Vito et al.
  2008; CC BY 4.0. European CSV (`;` separator, `,` decimals, `-200` = missing,
  two trailing blank columns). 11 retained numeric pollutant/met series.

- **Exchange-Rate** — `data/exchange_rate.txt.gz` from
  `https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz`.
  Lai et al. (LSTNet, SIGIR 2018). Daily rates of 8 countries, 1990-2016.

- **Appliances Energy** — `data/appliances_energy.zip` from
  `https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip`.
  Candanedo, Feldheim & Deramaix 2017; CC BY 4.0. 10-min cadence, ~4.5 months.

- **Traffic (PEMS)** — `data/traffic.txt.gz` from
  `https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz`.
  Lai et al. (LSTNet); underlying Caltrans PEMS freeway occupancy (public).
  Hourly, 862 sensors over 2 years.

### Failed / skipped sources
- St. Louis Fed direct `current.csv` (all host variants) — network unreachable
  from sandbox (HTTP/2 reset, timeouts, S3 `AccessDenied`). Mirror used instead.
- **FRED-QD** — no auth-free **Python-parseable** mirror found; the GitHub
  copies (`nk027/bvar`, `cykbennie/fbi`) ship it only as R `.rda`/`.RData`.
  Skipped (macro domain already covered by FRED-MD).
- **PhysioNet/MIMIC** (health) — credentialed data-use agreement required;
  violates the no-auth constraint.

---

## Reproduce

```bash
python3 bench/datasets_real.py
```

Prints each dataset's shape, clean and native missing fractions, and a one-line
description, and (re)writes the `data/*_clean.npy` (+ `*_native.npy`,
`*_obsmask.npy`) caches. Loaders are exposed as the `DATASETS_REAL` dict
`{name: loader}` for wiring into the harness/arena.
