"""Loader for Ken French daily portfolio CSVs (real equity returns).

Parses the first 'Average Value Weighted Returns -- Daily' block, maps the -99.99 /
-999 missing codes to NaN, converts percent -> decimal, and returns
(dates [T], names [N], R [T,N]). Caches a dense modern window to data/<tag>_real.npz.
"""
import os, re
import numpy as np


def load_french_daily(path):
    names, dates, rows = None, [], []
    started = False
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")
            if not started:
                if s.strip().startswith(",") and started is False and names is None \
                        and len(s) > 5 and not s[1:2].isdigit():
                    # header row of the first block: ",Agric,Food,..."
                    names = [c.strip() for c in s.split(",")[1:] if c.strip()]
                    started = True
                continue
            m = re.match(r"^\s*(\d{8})\s*,(.*)$", s)
            if m:
                dates.append(int(m.group(1)))
                vals = [float(x) if x.strip() not in ("", ) else np.nan
                        for x in m.group(2).split(",")]
                rows.append(vals[:len(names)])
            else:
                if rows:           # reached the end of the daily VW block
                    break
    R = np.array(rows, dtype=float)
    R[(R <= -99.99) | (R <= -999)] = np.nan
    R = R / 100.0                  # percent -> decimal
    return np.array(dates), names, R


def dense_window(dates, R, start=19900101, min_cov=0.999):
    """Modern sub-sample (>= start) restricted to columns that are essentially fully
    observed there, then drop any remaining rows with a NaN -> a clean dense panel."""
    mask = dates >= start
    d, X = dates[mask], R[mask]
    col_cov = np.mean(np.isfinite(X), axis=0)
    keep = col_cov >= min_cov
    X = X[:, keep]
    row_ok = np.all(np.isfinite(X), axis=1)
    return d[row_ok], X[row_ok], keep


if __name__ == "__main__":
    base = "/tmp/ffdata"
    OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    for tag, fn in [("ff49", "49_Industry_Portfolios_Daily.csv"),
                    ("ff100", "100_Portfolios_10x10_Daily.csv")]:
        dates, names, R = load_french_daily(os.path.join(base, fn))
        d, X, keep = dense_window(dates, R)
        print(f"{tag}: raw {R.shape}  -> dense modern {X.shape}  "
              f"({d[0]}..{d[-1]})  daily ret mean={X.mean()*100:.3f}% "
              f"std={X.std()*100:.2f}%  kept {keep.sum()}/{len(names)} cols")
        np.savez(os.path.join(OUT, f"{tag}_real.npz"), dates=d, R=X,
                 names=np.array([n for n, k in zip(names, keep) if k]))
        print(f"   cached -> data/{tag}_real.npz")
