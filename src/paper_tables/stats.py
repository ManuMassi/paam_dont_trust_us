"""Statistical tests used by the paper-table scripts.

Conventions as in common.py: labels 1 = malware, 0 = benign; a decision is
1 (malware), 0 (benign) or -1 (deferred to stage 2).

All tests are on the *test* set: thresholds and configurations are fixed on
validation beforehand, so the only randomness measured here is the sampling of
test APKs (threshold_bootstrap.py also adds the choice of the thresholds).

  bootstrap()          paired bootstrap of every metric (and of the difference
                       with a reference method): CIs and two-sided p-values
  mcnemar_correct()    exact McNemar on correctness, samples decided by both
  mcnemar_deferral()   exact McNemar on the deferral indicator: does one method
                       send fewer samples to stage 2 than the other?
  overlap_test()       hypergeometric test of the overlap of two deferred sets
  bootstrap_jaccard()  CI of the Jaccard index of two deferred sets
  wilson(), fisher()   proportions: CI and comparison of two proportions
  holm()               Holm-Bonferroni correction of a family of p-values
"""

import numpy as np
from scipy.stats import binomtest, fisher_exact, hypergeom

B_DEFAULT = 10_000
SEED = 0
CHUNK = 500  # bootstrap replicates drawn at a time (memory: CHUNK x n indices)
ALPHA = 0.05
METRICS = ["accuracy", "mal_precision", "mal_recall", "mal_f1", "ben_f1", "deferred_pct"]


# ---------------------------------------------------------------- bootstrap


def _indices(n, n_boot, seed, chunk):
    """Bootstrap indices in chunks. Same seed -> same resamples for every call,
    so methods bootstrapped in separate calls are still paired."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        yield rng.integers(0, n, (k, n))
        done += k


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def _metrics_rows(yy, pp):
    """Metrics of every bootstrap replicate (rows of yy, pp), as in common.metrics."""
    tp = ((pp == 1) & (yy == 1)).sum(1).astype(float)
    fp = ((pp == 1) & (yy == 0)).sum(1).astype(float)
    fn = ((pp == 0) & (yy == 1)).sum(1).astype(float)
    tn = ((pp == 0) & (yy == 0)).sum(1).astype(float)
    decided = tp + fp + fn + tn
    return {
        "accuracy": _div(tp + tn, decided),
        "mal_precision": _div(tp, tp + fp),
        "mal_recall": _div(tp, tp + fn),
        "mal_f1": _div(2 * tp, 2 * tp + fp + fn),
        "ben_f1": _div(2 * tn, 2 * tn + fn + fp),
        "deferred_pct": 100.0 * (1.0 - decided / yy.shape[1]),
    }


def p_two_sided(d):
    """Two-sided bootstrap p-value of H0: difference = 0 (with +1 correction)."""
    d = np.asarray(d)
    n = len(d)
    low = (np.sum(d <= 0) + 1) / (n + 1)
    high = (np.sum(d >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(low, high)))


def bootstrap(y, preds, ref=None, n_boot=B_DEFAULT, seed=SEED, chunk=CHUNK, alpha=ALPHA):
    """Paired bootstrap of every metric of every method.

    preds: {name: decisions on test}. ref: name of the reference method (e.g.
    "Drebin"); if given, also the difference method - ref of every metric.

    Returns {name: {metric: (low, high), f"d_{metric}": (mean, low, high, p)}}.
    """
    y = np.asarray(y)
    names = list(preds)
    reps = {m: {k: [] for k in METRICS} for m in names}
    for idx in _indices(len(y), n_boot, seed, chunk):
        yy = y[idx]
        for m in names:
            for k, v in _metrics_rows(yy, np.asarray(preds[m])[idx]).items():
                reps[m][k].append(v)
    reps = {m: {k: np.concatenate(v) for k, v in r.items()} for m, r in reps.items()}

    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out = {}
    for m in names:
        out[m] = {k: tuple(np.percentile(v, q)) for k, v in reps[m].items()}
        if ref is not None and m != ref:
            for k in METRICS:
                d = reps[m][k] - reps[ref][k]
                lo, hi = np.percentile(d, q)
                out[m][f"d_{k}"] = (float(d.mean()), float(lo), float(hi), p_two_sided(d))
    return out


def bootstrap_jaccard(def_a, def_b, n_boot=B_DEFAULT, seed=SEED, chunk=CHUNK, alpha=ALPHA):
    """CI of the Jaccard index |A & B| / |A | B| of two deferred sets."""
    a, b = np.asarray(def_a, bool), np.asarray(def_b, bool)
    vals = []
    for idx in _indices(len(a), n_boot, seed, chunk):
        inter = (a[idx] & b[idx]).sum(1)
        union = (a[idx] | b[idx]).sum(1)
        vals.append(_div(inter.astype(float), union.astype(float)))
    vals = np.concatenate(vals)
    return tuple(np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# ------------------------------------------------------------------ McNemar


def _exact_binom(k, n):
    return float(binomtest(int(k), int(n), 0.5).pvalue) if n > 0 else 1.0


def mcnemar_correct(y, a, b):
    """Exact McNemar on correctness, on the samples that both a and b decide.

    Returns (n_both, only_a_right, only_b_right, p)."""
    y, a, b = map(np.asarray, (y, a, b))
    both = (a != -1) & (b != -1)
    ca, cb = a[both] == y[both], b[both] == y[both]
    n10, n01 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    return int(both.sum()), n10, n01, _exact_binom(min(n10, n01), n10 + n01)


def mcnemar_deferral(a, b):
    """Exact McNemar on the deferral indicator (paired: same test APKs).

    Returns (only_a_defers, only_b_defers, p). only_b_defers < only_a_defers
    with small p means b sends significantly fewer samples to stage 2."""
    da, db = np.asarray(a) == -1, np.asarray(b) == -1
    n_a, n_b = int(np.sum(da & ~db)), int(np.sum(~da & db))
    return n_a, n_b, _exact_binom(min(n_a, n_b), n_a + n_b)


# -------------------------------------------------------------- proportions


def wilson(k, n, alpha=ALPHA):
    """Wilson score interval of the proportion k / n."""
    if n == 0:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact test of k1/n1 vs k2/n2."""
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


# ------------------------------------------------------------------ overlap


def overlap_test(n, n_a, n_b, k):
    """Overlap k of two deferred sets of sizes n_a, n_b out of n samples.

    expected   overlap if the two sets were drawn independently (n_a * n_b / n)
    p_more     P(overlap >= k) under independence (sets overlap more than chance)
    p_less     P(overlap <= k) under independence (sets overlap less than chance)
    jaccard    k / |A | B|
    """
    return {
        "expected": n_a * n_b / n,
        "p_more": float(hypergeom.sf(k - 1, n, n_a, n_b)),
        "p_less": float(hypergeom.cdf(k, n, n_a, n_b)),
        "jaccard": k / (n_a + n_b - k) if (n_a + n_b - k) > 0 else np.nan,
    }


# --------------------------------------------------------- multiple testing


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN entries are ignored and kept)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def noninferior(ci_low, margin):
    """True if the lower CI bound of (method - reference) is above -margin."""
    return bool(ci_low > -margin)


def fmt_ci(point, low, high, digits=3):
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
