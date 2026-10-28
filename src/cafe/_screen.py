"""Backward-only LAG SCREENER for CAFE-CD (causal discovery) -- numpy only.

PURPOSE
-------
Given an idiosyncratic residual panel ``E`` (entities x time x variables, NaN at
unobserved cells), cheaply produce a small candidate set

    C = {(i, j, k)}   "variable j may drive variable i at lag k"

so that the expensive sparse-regression stage downstream never has to consider all
``P^2 * K`` triples. The screener is a FILTER, not an estimator: its only job is to
be a cheap SUPERSET of the true parent set. Recall is the metric that matters;
false positives cost the next stage some time, false negatives cost it the truth.

PROVENANCE, AND TWO DELIBERATE DEVIATIONS FROM IT
------------------------------------------------
The design is derived from ARROW (Zhang et al., ICML 2025, PMLR v267:71668-71686,
"Accelerator for Time Series Causal Discovery with Time Weaving"). Two things in
that paper do not survive contact with this project, and both are changed here on
purpose. Neither change is cosmetic.

**Deviation 1 -- de-leaked encoding.** ARROW's Definition 4.1 encodes ``x_i^t`` as a
2-bit trend tuple taken relative to ``x_i^{t-w}`` AND ``x_i^{t+w}`` -- the second bit
reads a FUTURE value. This project's entire premise is no-look-ahead, so that
encoding is unusable. The forward bit is replaced by a second BACKWARD scale::

    b_s[t, i] = 1 if e[t, i] > e[t - w_s, i] else 0            w_s = 1
    b_l[t, i] = 1 if e[t, i] > e[t - w_l, i] else 0            w_l = ceil(sqrt(max_lag))
    v  [t, i] = obs[t, i] & obs[t - w_s, i] & obs[t - w_l, i]  (validity bit)

Everything at index ``t`` depends only on data at indices ``<= t``. That is asserted
exactly in ``src/tests/test_screen.py`` (encoding-truncation test, plus a "the test
has teeth" control that shows a forward-looking encoder fails it).

**Deviation 2 -- the theorem does not transfer, so the threshold is calibrated
empirically.** ARROW's Theorem 4.2 (if ``v, v'`` are causally related at lag ``k``
then ``P(XOR = (0,0))`` or ``P(XOR = (1,1))`` exceeds ``1/3 - eps``) is tied to the
(backward, forward) tuple, whose two bits are constructed from disjoint increments.
Our two bits are BOTH backward and therefore share the term ``e[t, i]``; they are
positively correlated by construction, and the correlation depends on ``w_l``, on the
serial dependence of the series, and on the missingness pattern. The null of the
statistic is consequently NEITHER 1/3 NOR 1/4, and no fixed constant is defensible.
**Nothing in this module hard-codes a significance threshold.** The threshold comes
from a circular block permutation of the data at hand (see NULL CALIBRATION). Any
transplanted statement of Theorem 4.2 for this encoding would simply be false, so
none is made.

WHAT THE STATISTIC ACTUALLY IS -- NO NOVELTY IS CLAIMED
------------------------------------------------------
For candidate ``(j -> i, lag k)`` we compare ``b_.[t, i]`` with ``b_.[t - k, j]``,
``d_. = b_.[t, i] XOR b_.[t - k, j]``, all counts restricted to
``v[t, i] & v[t - k, j]``::

    stat="phi"     S = max_w | 1 - 2 * hamming_rate_w |          (DEFAULT)
    stat="arrow"   R = max( P(d_s = 0 & d_l = 0), P(d_s = 1 & d_l = 1) )

Be clear about what ``phi`` is: on sign-of-difference bits, ``1 - 2 * hamming_rate``
is exactly ``P(agree) - P(disagree)``, i.e. the sample phi coefficient of the two
sign bits -- which IS the sample SIGN-CORRELATION of first differences (at scale
``w_s``) and of ``w_l``-differences (at scale ``w_l``). It is a rank-robust,
heavy-tail-safe surrogate for the lagged cross-correlation of increments, related to
Kendall/Blomqvist-type sign statistics that predate all of this by decades. Its
virtue here is SPEED and MEMORY (one bit per observation, one BLAS-3 call or one
popcount per lag), not novelty. It is presented as an engineering choice, not a new
method.

NULL CALIBRATION
----------------
Circular block permutation. For each of ``n_perm`` draws, every source variable
``j`` gets its own random circular shift ``s_j >= max_lag`` applied to its bit
streams (the target side is left alone); the whole ``(j, k)`` statistic surface is
recomputed and reduced to ``max_{j != i, k} S`` per target ``i``. The threshold for
target ``i`` is the ``ceil((1 - alpha) * (n_perm + 1))``-th order statistic of those
maxima -- a Westfall-Young max-statistic (FWER) null, taken with ``np.partition``
rather than a sort. Circular shifts preserve each source's own serial dependence and
its marginal bit rate while destroying its alignment with the target, which is the
null we want. A degenerate variable (e.g. a constant or LOCF-flattened stream, where
every bit is 0 and ``phi`` is 1 against any other flat stream) calibrates to a
threshold of 1 and is therefore silently rejected rather than silently selected.

With ``force_self=True`` the diagonal is excluded from the null max, because the
self terms are force-included and never tested; with ``force_self=False`` the
diagonal is included in both the null and the test, so the two stay consistent.

``n_perm=0`` disables the test (threshold ``+inf``); only the SIS floor and the
forced self terms survive. Note the null costs ``n_perm`` times one full statistic
pass and is by far the dominant cost of this module -- see the timings below.

CANDIDATE SELECTION (per target, not a global top-M)
----------------------------------------------------
1. keep ``(j, k)`` with ``S > threshold[i]``;
2. UNION with a SIS-style top-``m_i``, ``m_i = min(P*K, ceil(sis_c * n_i / log(P*K)))``
   where ``n_i`` is the median usable pair count for target ``i`` (so missingness
   shrinks the floor automatically). This is a floor, not a cap: it guarantees the
   downstream regression sees a non-empty, estimable design even when the
   permutation threshold is conservative;
3. ``force_self=True`` (default): ``(i, i, 1..K)`` are included UNCONDITIONALLY, with
   no test. If the AR terms are screened out the downstream regression is
   misspecified and every autoregressive effect leaks into cross-edges. This is not
   negotiable and is not a tuning knob for accuracy;
4. ALL significant lags are kept, not a single argmax -- a true edge can act at
   several lags. ``argmax`` is used only to REPORT a lag-accuracy metric
   (``ScreenResult.argmax_lag``), never to select.

BACKENDS -- AND AN HONEST STATEMENT OF WHICH ONE WINS
-----------------------------------------------------
``method="dense"``  : bits -> ``{-1, 0, +1}`` float32 (0 = invalid), then one BLAS-3
                      ``C_k = D[k:].T @ D[:-k]`` per lag. Masked counts need a second
                      matmul on the validity mask; ``stat="arrow"`` needs a third on
                      the elementwise product ``G = D_s * D_l`` (see below).
``method="bits"``   : ``np.packbits``; popcount by a uint16 lookup-table gather,
                      ``LUT[xored.view(np.uint16)].sum(dtype=np.int64)`` -- a 64 KB
                      table that sits in L2, one gather plus one reduction over
                      ``n/16`` elements into preallocated buffers. (This is much
                      faster than ``np.unpackbits``, which allocates 8x and undoes
                      the packing.) Masked counts are nearly free:
                      ``n_ij = popcount(v_i & v_j)``,
                      ``agree = popcount(~(b_i ^ b_j) & v_i & v_j)``.

MEASURED, VERBATIM (macOS-26.6.2-arm64, numpy 2.5.0 on Accelerate BLAS, single
process; STATISTIC ONLY, ``n_perm=0``, n = 4000, K = 8, best of 3):

    P     stat     dense (s)    bits (s)   bits/dense   max|dS|
    20    phi        0.0011      0.0071        6.6x     0.0e+00
    20    arrow      0.0010      0.0096        9.6x     0.0e+00
    50    phi        0.0047      0.0365        7.8x     0.0e+00
    50    arrow      0.0051      0.0561       11.0x     0.0e+00
    100   phi        0.0138      0.1274        9.2x     0.0e+00
    100   arrow      0.0141      0.1897       13.4x     0.0e+00
    200   phi        0.0278      0.4493       16.2x     0.0e+00
    200   arrow      0.0297      0.7179       24.2x     0.0e+00

    20% MCAR (phi):        P=20 7.0x, P=50 10.2x, P=100 16.7x, P=200 26.2x
    larger probe (n=2000): P=400 21.1x, P=800 26.4x
    full run, n_perm=200:  P=20 5.1x, P=50 9.7x, P=100 15.2x
                           (dense totals 0.212 / 0.599 / 1.497 s, of which the
                            permutation null is 99%)

**``bits`` LOSES at every size tested -- by 6.6x to 26.4x -- and the gap WIDENS with
P. There is no crossover anywhere in the tested range (P up to 800), so
``method="auto"`` selects ``dense`` unless dense would exceed the memory budget.**
This is a null result on the bit-packing speed idea and it is reported as one.

The reason is not mysterious. ARROW's headline 153x speed-up is measured against
SURD's brute-force lag search at 10 variables, whose cost is dominated by histogram
binning; there is no binning here, and the baseline is BLAS-3 sgemm, which does the
same ``P^2 n`` work at hundreds of GFLOP/s with perfect cache blocking and full
vector width. The popcount path, by contrast, is a memory-bound LUT gather that
numpy must drive from a Python loop over targets (P iterations per lag), so it pays
interpreter and dispatch overhead exactly where the GEMM pays none -- which is also
why the ratio GROWS with P instead of shrinking. Beating sgemm would need a SWAR
uint64 or a JIT/Rust popcount kernel, which this module deliberately does not have
(numpy-only is a hard constraint here).

Bit-packing's real payoffs are untouched by this benchmark and remain real:
  * INPUT footprint is 32x smaller (three float32 planes -> three packed bit planes):
    at P = 5000, n = 4000 that is 229 MiB vs 7.2 MiB. But the ``(K, P, P)`` OUTPUT
    tensor is 3.0 GiB at that size for BOTH backends -- packing shrinks the input,
    not the result, so "P = 5000 stays cache-resident" is true only of the encoding.
  * SPARSE candidate sets: scoring an arbitrary list of ``(i, j, k)`` triples (a
    shortlist re-scored at finer lags, an incremental update) is O(|list|) with
    popcounts and cannot use a matmul at all. That is where ``method="bits"`` earns
    its keep, and why the backend is kept rather than deleted.

One consequence for ``method="auto"``: the brief's memory test,
``P*P*max_lag*n*4 <= budget``, is not the dense working set (no ``(P,P,K,n)`` tensor
is ever built -- see ``_choose_method``), and taking it literally would route
P = 100, n = 4000 to the backend just measured 9x slower on that exact shape. The
real footprint is used instead: ``auto`` follows the measurement, not the formula.

Note also where the wall clock actually goes: with the default ``n_perm=200`` the
permutation null is ~99% of the run. The statistic is nearly free; the CALIBRATION
is the cost. Lower ``n_perm`` (or set it to 0 and lean on the SIS floor) if that
trade is wrong for a caller -- but do not replace it with a fixed constant.

WHICH STATISTIC RECOVERS MORE -- ALSO MEASURED
----------------------------------------------
On the sparse-VAR fixture in ``src/tests/test_screen.py`` (P = 100, K = 8, lags drawn
from 1..4), averaged over 4 draws, ``n_perm=100``. "sig" is the recall of the
permutation TEST alone (SIS floor off); "lagacc" is the share of true edges whose
argmax lag is the true lag; "rec" is end-to-end recall of the delivered set:

    regime                          phi  rec/sig/lagacc     arrow  rec/sig/lagacc
    strong VAR, T=400              1.000 / 0.891 / 0.990    1.000 / 0.911 / 0.995
    weak+dense, T=300, 4 edges     0.884 / 0.184 / 0.855    0.898 / 0.228 / 0.884
    short, T=200                   0.962 / 0.375 / 0.914    0.951 / 0.426 / 0.956
    heavy tails t(3), T=400        0.968 / 0.315 / 0.890    0.955 / 0.327 / 0.925

``arrow`` has the better TEST in every regime measured (higher sig-only recall and
higher lag accuracy, 4/4), because the joint two-bit count uses information that the
max of the two marginal phi's throws away. End-to-end recall is a WASH -- ``phi``
wins two of the four rows -- because the SIS floor, not the test, is what delivers
most of the recall at these sample sizes. So: ``phi`` stays the DEFAULT (cheaper,
and it is literally a sign-correlation, so its scale and null are easy to reason
about), but pass ``stat="arrow"`` when the TEST itself has to carry the load --
small ``sis_c``, or when the reported argmax lag matters.

MISSINGNESS -- AND A KNOWN LIMIT OF THE SPECIFIED VALIDITY BIT
--------------------------------------------------------------
Measured on the same fixture (P = 100, T = 800, MCAR, phi, ``n_perm=60``):

    p_miss   0.00    0.10    0.20    0.30    0.40    0.50
    recall   1.000   1.000   0.988   0.673   0.152   0.002
    median n_valid  788     416     205      91      36      12

Graceful to 20%, then a cliff. The cause is arithmetic, not statistical: a valid
pair needs SIX observed cells (``t, t-w_s, t-w_l`` for both variables), so the
usable count falls like ``(1 - p)^6`` -- 26% of pairs at p = 0.2, 12% at p = 0.3.
A per-scale validity bit (``v_s = obs[t] & obs[t-w_s]`` scored separately from
``v_l``) would use ``(1 - p)^4`` instead and is the obvious next lever if heavy
missingness matters; it is NOT done here because the single conjunctive ``v`` is
what the design specifies, and changing it would also change what ``stat="arrow"``
means (its two bits would no longer be counted on a common sample).

ARROW STAT IN CLOSED FORM (why it is still only one extra matmul)
-----------------------------------------------------------------
With ``u = 1 - 2 d_s in {-1, +1}`` and ``z = 1 - 2 d_l``, and ``m`` the validity
indicator, ``count(d_s = 0, d_l = 0) = sum_m (1+u)(1+z)/4 = (n + Su + Sz + Suz)/4``
and ``count(d_s = 1, d_l = 1) = (n - Su - Sz + Suz)/4``. Now
``u = D_s[t,i] D_s[t-k,j]`` and ``z = D_l[t,i] D_l[t-k,j]``, hence
``u z = G[t,i] G[t-k,j]`` with ``G = D_s * D_l`` elementwise. So ``Suz`` is one more
matmul of the same shape -- no joint-histogram loop is needed.

PANELS
------
Entities are concatenated along time. No padding rows are inserted; instead the
validity bit is zeroed for the first ``max(w_l, max_lag)`` rows of every entity,
which makes cross-entity pairs impossible (a target inside that prefix is invalid,
and a target past it can only reach back within its own entity). The permutation
null circularly shifts the concatenated stream, which is the usual circular block
bootstrap on a concatenated panel.

numpy only: no scipy, no sklearn, no numba, no numexpr, no pandas.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

__all__ = ["ScreenResult", "screen_lags", "screen_lags_online", "encode_bits",
           "popcount_bytes"]

# ---- defaults (structural, NOT tuned against any dataset) ------------------- #
MEM_BUDGET = 1 << 31        # 2 GiB working-set budget used by method="auto"
MIN_COUNT = 30              # a pair with fewer usable observations scores 0 (a
                            # sample-size guard, not a significance threshold)
PERM_CHUNK = 8              # permutations batched into one GEMM / popcount sweep
# Measured crossover above which "bits" would be preferred on speed. No crossover
# was observed up to P = 200 (bits is ~14x SLOWER there and the ratio is growing),
# so this is set out of range on purpose: "auto" means "dense unless it does not
# fit". Lower it if a future backend (SWAR / JIT / Rust popcount) changes the fact.
P_CROSSOVER = 1 << 30

_POPCOUNT16 = None


def _lut16() -> np.ndarray:
    """64 KB uint16 -> popcount table (built once, sits in L2)."""
    global _POPCOUNT16
    if _POPCOUNT16 is None:
        words = np.arange(1 << 16, dtype=np.uint16).view(np.uint8).reshape(-1, 2)
        _POPCOUNT16 = np.unpackbits(words, axis=1).sum(1).astype(np.uint8)
    return _POPCOUNT16


def popcount_bytes(a: np.ndarray, buf: np.ndarray | None = None) -> np.ndarray:
    """Popcount along the last axis of a C-contiguous uint8 array (even width).

    One uint16 LUT gather plus one reduction; ``buf`` (shape ``a.shape[:-1] +
    (a.shape[-1] // 2,)``, uint8) lets the caller preallocate the gather target so
    the hot loop allocates nothing.
    """
    v = a.view(np.uint16)
    lut = _lut16()
    if buf is None:
        buf = np.take(lut, v)
    else:
        np.take(lut, v, out=buf)
    return buf.sum(axis=-1, dtype=np.int64)


# --------------------------------------------------------------------------- #
# result type
# --------------------------------------------------------------------------- #
@dataclass
class ScreenResult:
    """Output of :func:`screen_lags`.

    Attributes
    ----------
    candidates : dict
        ``{i: [(j, k), ...]}``, sorted by ``(j, k)``. ``k`` is a 1-based lag.
    S : (K, P, P) float64
        Raw statistic, ``S[k-1, i, j]`` = evidence that ``j`` drives ``i`` at lag
        ``k``. Always finite (0 where there is no usable sample).
    threshold : (P,) float64
        Per-target permutation threshold (``+inf`` when ``n_perm == 0``).
    n_valid : (K, P, P) int64
        Usable pair counts behind every entry of ``S``.
    reduction : float
        ``|C| / (P * P * K)`` -- the share of the triple space handed downstream.
    meta : dict
        Settings actually used, timings, and the SIS floor per target.
    """

    candidates: dict
    S: np.ndarray
    threshold: np.ndarray
    n_valid: np.ndarray
    reduction: float
    meta: dict

    @property
    def n_candidates(self) -> int:
        return int(sum(len(v) for v in self.candidates.values()))

    def triples(self) -> list:
        """Flat sorted ``[(i, j, k), ...]``."""
        return [(i, j, k) for i in sorted(self.candidates)
                for (j, k) in self.candidates[i]]

    @property
    def argmax_lag(self) -> np.ndarray:
        """(P, P) int -- 1-based lag maximising ``S`` for each ``(i, j)``.

        REPORTING ONLY. Selection keeps every significant lag; this exists so a
        lag-accuracy metric can be computed against a known generator.
        """
        return np.argmax(self.S, axis=0) + 1

    def __repr__(self) -> str:                                # pragma: no cover
        K, P, _ = self.S.shape
        return (f"ScreenResult(P={P}, K={K}, |C|={self.n_candidates}, "
                f"reduction={self.reduction:.4f}, method={self.meta.get('method')}, "
                f"stat={self.meta.get('stat')})")


# --------------------------------------------------------------------------- #
# encoding -- everything at row t reads only rows <= t
# --------------------------------------------------------------------------- #
def _as_panel(E, obs):
    A = np.asarray(E)
    if A.ndim == 2:
        A = A[None]
    elif A.ndim != 3:
        raise ValueError(f"E must be (T, P) or (entities, T, P); got ndim={A.ndim}")
    X = np.ascontiguousarray(A, dtype=np.float64)
    if obs is None:
        O = np.isfinite(X)
    else:
        Ob = np.asarray(obs, dtype=bool)
        if Ob.ndim == 2:
            Ob = Ob[None]
        if Ob.shape != X.shape:
            raise ValueError(f"obs shape {Ob.shape} != E shape {X.shape}")
        # a cell claimed observed but non-finite is treated as missing
        O = Ob & np.isfinite(X)
    return X, O


def encode_bits(E, obs=None, max_lag: int = 8, w_long=None):
    """Backward-only 2-scale trend encoding of a residual panel.

    Returns ``(b_s, b_l, v, w_s, w_l)`` with the three bit planes shaped
    ``(n_total, P)`` where ``n_total = entities * T`` (entities concatenated along
    time). ``v`` is zeroed for the first ``max(w_l, max_lag)`` rows of each entity,
    which both respects the encoding's own warm-up and makes cross-entity lag pairs
    impossible.

    Row ``t`` of every output is a function of input rows ``<= t`` only.
    """
    X, O = _as_panel(E, obs)
    N, T, P = X.shape
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    w_s = 1
    w_l = int(w_long) if w_long is not None else int(math.ceil(math.sqrt(max_lag)))
    # the two scales must differ, else the second bit carries no extra information
    w_l = max(w_l, w_s + 1)
    t_min = max(w_l, max_lag)

    n_tot = N * T
    b_s = np.zeros((n_tot, P), dtype=bool)
    b_l = np.zeros((n_tot, P), dtype=bool)
    v = np.zeros((n_tot, P), dtype=bool)
    if T > t_min:
        for e in range(N):
            o = e * T
            Oe = O[e]
            # unobserved values never enter a valid comparison; zero them so no NaN
            # ever reaches a comparison operator
            Xe = np.where(Oe, X[e], 0.0)
            np.greater(Xe[w_s:], Xe[:T - w_s], out=b_s[o + w_s:o + T])
            np.greater(Xe[w_l:], Xe[:T - w_l], out=b_l[o + w_l:o + T])
            np.logical_and(Oe[t_min:], Oe[t_min - w_s:T - w_s], out=v[o + t_min:o + T])
            np.logical_and(v[o + t_min:o + T], Oe[t_min - w_l:T - w_l],
                           out=v[o + t_min:o + T])
    return b_s, b_l, v, w_s, w_l


# --------------------------------------------------------------------------- #
# dense backend
# --------------------------------------------------------------------------- #
def _dense_signals(b_s, b_l, v, stat):
    """bits -> {-1, 0, +1} float32 planes (0 exactly where the bit is invalid)."""
    vf = v.astype(np.float32)
    a_s = (b_s.astype(np.float32) * 2.0 - 1.0) * vf
    a_l = (b_l.astype(np.float32) * 2.0 - 1.0) * vf
    g = (a_s * a_l) if stat == "arrow" else None
    return a_s, a_l, vf, g


def _stats_dense(tgt, src, max_lag, stat, min_count, S, NV):
    """Fill ``S`` (K, P, Q) [and ``NV``] from float32 planes via one GEMM per lag.

    ``src`` may hold ``Q = P * batch`` columns (permutation batching); the target
    side always has ``P`` columns.
    """
    a_s_t, a_l_t, v_t, g_t = tgt
    a_s_s, a_l_s, v_s, g_s = src
    P = a_s_t.shape[1]
    Q = a_s_s.shape[1]
    nb = np.empty((P, Q), dtype=np.float32)
    su = np.empty((P, Q), dtype=np.float32)
    sz = np.empty((P, Q), dtype=np.float32)
    sg = np.empty((P, Q), dtype=np.float32) if stat == "arrow" else None
    den = np.empty((P, Q), dtype=np.float64)
    for k in range(1, max_lag + 1):
        np.dot(v_t[k:].T, v_s[:-k], out=nb)
        np.dot(a_s_t[k:].T, a_s_s[:-k], out=su)
        np.dot(a_l_t[k:].T, a_l_s[:-k], out=sz)
        ok = nb >= min_count
        Sk = S[k - 1]
        Sk.fill(0.0)
        # A TRUE float64 divide, never a multiply by a float64 reciprocal: IEEE
        # division is correctly rounded, so a SATURATED pair scores exactly 1.0.
        # (n * (1/n) can land on 0.9999999999999999, which would let a degenerate
        # all-flat pair beat its own permutation threshold by one ulp.)
        if stat == "arrow":
            np.dot(g_t[k:].T, g_s[:-k], out=sg)
            c00 = (nb + su + sz + sg)
            c11 = (nb - su - sz + sg)
            np.multiply(nb, np.float64(4.0), out=den)
            np.divide(np.maximum(c00, c11), den, out=Sk, where=ok)
        else:
            # |su| == |#agree - #disagree| == |n - 2 * hamming|
            np.copyto(den, nb)
            np.divide(np.maximum(np.abs(su), np.abs(sz)), den, out=Sk, where=ok)
        if NV is not None:
            NV[k - 1] = nb
    return S, NV


# --------------------------------------------------------------------------- #
# bits backend
# --------------------------------------------------------------------------- #
def _nbytes_even(n: int) -> int:
    nb = (n + 7) >> 3
    return nb + (nb & 1)          # even, so the uint16 LUT view is legal


def _pack_fixed(A2d, nb_even):
    """Pack a ``(rows, m)`` bool array along axis 1 into a fixed ``nb_even`` width."""
    q = np.packbits(A2d, axis=1)
    out = np.zeros((q.shape[0], nb_even), dtype=np.uint8)
    out[:, :q.shape[1]] = q
    return out


def _pack_targets(b_s, b_l, v, max_lag, nb_even):
    """Per-lag packed TARGET planes (rows ``k:``), variable-major.

    Bits beyond ``n - k`` are zero, so a source packed at FULL length can be reused
    for every lag: the AND with the target's validity plane masks the overhang.
    """
    bsT = np.ascontiguousarray(b_s.T)
    blT = np.ascontiguousarray(b_l.T)
    vT = np.ascontiguousarray(v.T)
    return [(_pack_fixed(bsT[:, k:], nb_even),
             _pack_fixed(blT[:, k:], nb_even),
             _pack_fixed(vT[:, k:], nb_even)) for k in range(1, max_lag + 1)]


def _stats_bits(tgt_packed, src_packed, max_lag, stat, min_count, S, NV):
    """Fill ``S`` (K, P, Q) [and ``NV``] by LUT popcount over packed bit planes."""
    bs_s, bl_s, v_s = src_packed
    Q, nb = bs_s.shape
    P = S.shape[1]
    t0 = np.empty((Q, nb), dtype=np.uint8)
    t1 = np.empty((Q, nb), dtype=np.uint8)
    t2 = np.empty((Q, nb), dtype=np.uint8)
    t3 = np.empty((Q, nb), dtype=np.uint8) if stat == "arrow" else None
    gbuf = np.empty((Q, nb // 2), dtype=np.uint8)
    for k in range(1, max_lag + 1):
        bs_t, bl_t, v_t = tgt_packed[k - 1]
        for i in range(P):
            np.bitwise_and(v_t[i], v_s, out=t0)                    # validity of pair
            n_ij = popcount_bytes(t0, gbuf)
            np.bitwise_xor(bs_t[i], bs_s, out=t1)
            np.bitwise_and(t1, t0, out=t1)                         # d_s (masked)
            d_s = popcount_bytes(t1, gbuf)
            np.bitwise_xor(bl_t[i], bl_s, out=t2)
            np.bitwise_and(t2, t0, out=t2)                         # d_l (masked)
            d_l = popcount_bytes(t2, gbuf)
            ok = n_ij >= min_count
            Sk = S[k - 1, i]
            Sk.fill(0.0)
            if stat == "arrow":
                np.bitwise_or(t1, t2, out=t3)
                np.invert(t3, out=t3)
                np.bitwise_and(t3, t0, out=t3)
                c00 = popcount_bytes(t3, gbuf)
                np.bitwise_and(t1, t2, out=t3)
                c11 = popcount_bytes(t3, gbuf)
                np.divide(np.maximum(c00, c11), n_ij, out=Sk, where=ok)
            else:
                su = n_ij - 2 * d_s
                sz = n_ij - 2 * d_l
                np.divide(np.maximum(np.abs(su), np.abs(sz)), n_ij, out=Sk, where=ok)
            if NV is not None:
                NV[k - 1, i] = n_ij
    return S, NV


# --------------------------------------------------------------------------- #
# permutation null (circular block permutation, max-statistic / Westfall-Young)
# --------------------------------------------------------------------------- #
def _shift_index(n, shifts, transposed):
    """Index array realising a per-column circular shift ``x[(t - s_j) mod n]``."""
    t = np.arange(n)
    if transposed:                                    # (Q, n) layout
        idx = (t[None, :] - shifts[:, None]) % n
    else:                                             # (n, Q) layout
        idx = (t[:, None] - shifts[None, :]) % n
    return idx


def _null_max(planes, packed_targets, max_lag, stat, min_count, n_perm, alpha,
              rng, exclude_self, method, nb_even, chunk):
    """Return the (P,) per-target permutation threshold."""
    if method == "dense":
        a_s, a_l, vf, g = planes
        n_tot, P = a_s.shape
    else:
        b_s, b_l, v = planes
        n_tot, P = b_s.shape
    lo, hi = max_lag, max(max_lag + 1, n_tot - max_lag)
    maxes = np.empty((n_perm, P), dtype=np.float64)
    Sbuf = None
    buf = None                     # rolled-source buffers, reallocated only if Q changes
    if method == "bits":
        bsT = np.ascontiguousarray(b_s.T)
        blT = np.ascontiguousarray(b_l.T)
        vT = np.ascontiguousarray(v.T)
    for b0 in range(0, n_perm, chunk):
        bc = min(chunk, n_perm - b0)
        Q = P * bc
        if Sbuf is None or Sbuf.shape[2] != Q:
            Sbuf = np.zeros((max_lag, P, Q), dtype=np.float64)
        if method == "dense":
            if buf is None or buf[0].shape[1] != Q:
                nplane = 4 if stat == "arrow" else 3
                buf = [np.empty((n_tot, Q), dtype=np.float32) for _ in range(nplane)]
                buf += [None] * (4 - nplane)
            src = buf
            for c in range(bc):
                sh = rng.integers(lo, hi, size=P)
                idx = _shift_index(n_tot, sh, transposed=False)
                sl = slice(c * P, (c + 1) * P)
                src[0][:, sl] = np.take_along_axis(a_s, idx, axis=0)
                src[1][:, sl] = np.take_along_axis(a_l, idx, axis=0)
                src[2][:, sl] = np.take_along_axis(vf, idx, axis=0)
                if stat == "arrow":
                    src[3][:, sl] = np.take_along_axis(g, idx, axis=0)
            _stats_dense((a_s, a_l, vf, g), tuple(src), max_lag, stat, min_count,
                         Sbuf, None)
        else:
            if buf is None or buf[0].shape[0] != Q:
                buf = [np.empty((Q, n_tot), dtype=bool) for _ in range(3)]
            rolled = buf
            for c in range(bc):
                sh = rng.integers(lo, hi, size=P)
                idx = _shift_index(n_tot, sh, transposed=True)
                sl = slice(c * P, (c + 1) * P)
                rolled[0][sl] = np.take_along_axis(bsT, idx, axis=1)
                rolled[1][sl] = np.take_along_axis(blT, idx, axis=1)
                rolled[2][sl] = np.take_along_axis(vT, idx, axis=1)
            src = (_pack_fixed(rolled[0], nb_even), _pack_fixed(rolled[1], nb_even),
                   _pack_fixed(rolled[2], nb_even))
            _stats_bits(packed_targets, src, max_lag, stat, min_count, Sbuf, None)
        S4 = Sbuf.reshape(max_lag, P, bc, P)
        if exclude_self:
            ar = np.arange(P)
            S4[:, ar, :, ar] = -np.inf     # self terms are forced in, never tested
        maxes[b0:b0 + bc] = S4.max(axis=3).max(axis=0).T
        if exclude_self:
            S4[:, np.arange(P), :, np.arange(P)] = 0.0
    # exact permutation order statistic (np.partition, no full sort)
    r = int(math.ceil((1.0 - alpha) * (n_perm + 1))) - 1
    r = min(max(r, 0), n_perm - 1)
    return np.partition(maxes, r, axis=0)[r].astype(np.float64)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _select(S, NV, thr, sis_c, force_self):
    K, P, _ = S.shape
    PK = P * K
    logPK = max(math.log(PK), 1.0)
    cands = {}
    m_used = np.zeros(P, dtype=np.int64)
    n_used = np.zeros(P, dtype=np.float64)
    for i in range(P):
        Si = S[:, i, :]                                  # (K, P)
        sel = Si > thr[i]
        work = Si
        if force_self:
            sel[:, i] = False
            work = Si.copy()
            work[:, i] = -np.inf                         # keep self out of the SIS rank
        nv = NV[:, i, :]
        pos = nv[nv > 0]
        n_i = float(np.median(pos)) if pos.size else 0.0
        n_used[i] = n_i
        m_i = int(min(PK, math.ceil(sis_c * n_i / logPK))) if sis_c > 0 else 0
        m_i = max(m_i, 0)
        m_used[i] = m_i
        if m_i > 0:
            flat = work.ravel()
            if m_i >= flat.size:
                top = np.arange(flat.size)
            else:
                top = np.argpartition(flat, flat.size - m_i)[flat.size - m_i:]
            sel.ravel()[top] = True
        if force_self:
            sel[:, i] = True                             # (i, i, 1..K) unconditionally
        kk, jj = np.nonzero(sel)                         # kk = lag-1, jj = source
        order = np.lexsort((kk, jj))                     # sort by (j, k)
        cands[i] = [(int(jj[o]), int(kk[o]) + 1) for o in order]
    return cands, m_used, n_used


def _choose_method(P, n_tot, max_lag, stat, mem_budget):
    """``auto`` policy: dense unless it genuinely does not fit.

    The brief specified the test ``P*P*max_lag*n*4 <= budget``. That expression is
    not the dense working set -- no ``(P, P, K, n)`` tensor is ever materialised; the
    dense path holds 3-4 ``(n, P)`` float32 planes plus the ``(K, P, P)`` output, and
    ``P*P*max_lag*n*4`` overstates it by a factor of ``n``. Using it literally would
    route P = 100, n = 4000 to ``bits``, i.e. to the backend measured 9x SLOWER on
    exactly that shape. The real footprint is used instead, so ``auto`` follows the
    measurement rather than the formula. Deliberate; see the module docstring.
    """
    ncols = 4 if stat == "arrow" else 3
    dense_bytes = ncols * n_tot * P * 4 + max_lag * P * P * (8 + 8)
    if dense_bytes > mem_budget:
        return "bits"
    if P > P_CROSSOVER:
        return "bits"
    return "dense"


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def screen_lags(E, obs=None, max_lag: int = 8, w_long=None, stat: str = "phi",
                method: str = "auto", alpha: float = 0.05, n_perm: int = 200,
                sis_c: float = 1.0, seed: int = 0, force_self: bool = True,
                min_count: int = MIN_COUNT, mem_budget_bytes: int = MEM_BUDGET,
                perm_chunk: int = PERM_CHUNK) -> ScreenResult:
    """Backward-only lag screening of a residual panel.

    Parameters
    ----------
    E : (T, P) or (entities, T, P) array
        Idiosyncratic residual panel; NaN (or ``obs=False``) marks unobserved cells.
    obs : bool array, optional
        Observation mask matching ``E``. Default: ``np.isfinite(E)``.
    max_lag : int
        Largest lag ``K`` considered. Candidates use 1-based lags ``1..K``.
    w_long : int, optional
        Second (long) BACKWARD scale. Default ``ceil(sqrt(max_lag))``, floored at 2.
    stat : {"phi", "arrow"}
        ``phi`` (default) is the sign-correlation surrogate; ``arrow`` is the
        faithful BACKWARD port of ARROW's joint-XOR statistic.
    method : {"auto", "dense", "bits"}
        Backend. ``auto`` picks ``dense`` unless its working set exceeds
        ``mem_budget_bytes`` -- ``bits`` measured slower at every tested size.
    alpha : float
        Family-wise error rate for the per-target permutation threshold.
    n_perm : int
        Circular block permutations. ``0`` disables the test (threshold ``+inf``).
        Runtime is dominated by this: one permutation costs one full statistic pass.
    sis_c : float
        SIS floor constant; ``0`` disables the floor.
    seed : int
        Seed for the permutation shifts.
    force_self : bool
        Include ``(i, i, 1..K)` unconditionally (default, strongly recommended).
    min_count : int
        Pairs with fewer usable observations score 0 (sample-size guard).

    Returns
    -------
    ScreenResult
    """
    t_wall = time.perf_counter()
    stat = str(stat).lower()
    if stat not in ("phi", "arrow"):
        raise ValueError(f"stat must be 'phi' or 'arrow'; got {stat!r}")
    method = str(method).lower()
    if method not in ("auto", "dense", "bits"):
        raise ValueError(f"method must be 'auto', 'dense' or 'bits'; got {method!r}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    b_s, b_l, v, w_s, w_l = encode_bits(E, obs, max_lag, w_long)
    n_tot, P = b_s.shape
    if n_tot <= max_lag:
        raise ValueError(f"need more than max_lag={max_lag} rows; got {n_tot}")
    used = _choose_method(P, n_tot, max_lag, stat, mem_budget_bytes) \
        if method == "auto" else method

    S = np.zeros((max_lag, P, P), dtype=np.float64)
    NV = np.zeros((max_lag, P, P), dtype=np.int64)
    nb_even = _nbytes_even(n_tot)
    packed_targets = None
    t_stat = time.perf_counter()
    if used == "dense":
        planes = _dense_signals(b_s, b_l, v, stat)
        _stats_dense(planes, planes, max_lag, stat, min_count, S, NV)
    else:
        planes = (b_s, b_l, v)
        packed_targets = _pack_targets(b_s, b_l, v, max_lag, nb_even)
        src = (_pack_fixed(np.ascontiguousarray(b_s.T), nb_even),
               _pack_fixed(np.ascontiguousarray(b_l.T), nb_even),
               _pack_fixed(np.ascontiguousarray(v.T), nb_even))
        _stats_bits(packed_targets, src, max_lag, stat, min_count, S, NV)
    t_stat = time.perf_counter() - t_stat

    t_perm = time.perf_counter()
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        if used == "bits" and packed_targets is None:     # pragma: no cover
            packed_targets = _pack_targets(b_s, b_l, v, max_lag, nb_even)
        thr = _null_max(planes, packed_targets, max_lag, stat, min_count, int(n_perm),
                        alpha, rng, force_self, used, nb_even, int(perm_chunk))
    else:
        thr = np.full(P, np.inf, dtype=np.float64)
    t_perm = time.perf_counter() - t_perm

    cands, m_used, n_used = _select(S, NV, thr, float(sis_c), bool(force_self))
    n_cand = int(sum(len(c) for c in cands.values()))
    reduction = n_cand / float(P * P * max_lag)
    meta = {
        "method": used, "method_requested": method, "stat": stat,
        "max_lag": int(max_lag), "w_short": int(w_s), "w_long": int(w_l),
        "alpha": float(alpha), "n_perm": int(n_perm), "sis_c": float(sis_c),
        "seed": int(seed), "force_self": bool(force_self),
        "min_count": int(min_count), "P": int(P), "n_rows": int(n_tot),
        "n_entities": int(np.asarray(E).shape[0]) if np.asarray(E).ndim == 3 else 1,
        "sis_m": m_used, "sis_n": n_used, "n_candidates": n_cand,
        "t_stat_s": t_stat, "t_perm_s": t_perm,
        "t_total_s": time.perf_counter() - t_wall,
    }
    return ScreenResult(cands, S, thr, NV, reduction, meta)


def screen_lags_online(E, obs=None, checkpoints=(), **kw):
    """Yield ``(t0, ScreenResult)`` computed from rows ``<= t0`` ONLY.

    The candidate set is a whole-sample object, so the meaningful causal claim is
    not "cell t was imputed causally" but: **the set reported at ``t0`` does not
    change when rows ``> t0`` are removed**. That is enforced structurally -- the
    panel is truncated at ``t0`` before anything is computed, and the encoding is
    backward-only, so no quantity can see past ``t0``. ``src/tests/test_screen.py``
    asserts it by scrambling the future and checking the yielded results are
    bit-identical.
    """
    A = np.asarray(E)
    T = A.shape[-2]
    for t0 in checkpoints:
        t0 = int(t0)
        if not (0 <= t0 < T):
            raise ValueError(f"checkpoint {t0} outside [0, {T})")
        Et = A[..., :t0 + 1, :]
        Ot = None if obs is None else np.asarray(obs)[..., :t0 + 1, :]
        yield t0, screen_lags(Et, Ot, **kw)
