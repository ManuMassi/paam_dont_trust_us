"""Code shared by the paper-table scripts.

Conventions:
  - labels: 1 = malware, 0 = benign;
  - a decision is 1 (malware), 0 (benign) or -1 (deferred to stage 2);
  - thresholds, weights and calibrations are always fitted on validation.

A *configuration* is a small dict that says how a method turns scores into
decisions (which score, how the thresholds are chosen, which voting rule).
decide() applies a configuration; sweep.py and the table scripts both use it.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "Ensemble" / "tables"
SWEEP_FILE = OUT_DIR / "sweep_configs.csv.gz"

DETECTORS = ["Drebin", "MalScan", "RAMDA"]
SHORT = {"Drebin": "D", "MalScan": "M", "RAMDA": "R"}

# Most samples the dual-threshold rows may defer on validation: 6.71%, i.e.
# what Drebin defers on validation with the paper's procedure (F1 floors 0.85).
BUDGET = 0.0671

# Grid explored for the dual thresholds (see DualSweep.select).
REL_DELTAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
FIXED_FLOORS = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]
GAPS = [0.0, 0.05, 0.10, 0.20]
BAND_BUDGETS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


# ----------------------------------------------------------------- data


def load_scores(split):
    """Scores of the three detectors on one split, joined on the APK hash."""
    def read(name, renames):
        df = pd.read_csv(ROOT / "results" / name / f"{split}_scores.csv")
        return df.rename(columns=renames)

    df = (
        read("Drebin", {"score": "drebin", "true_label": "y"})
        .merge(read("MalScan", {"score": "malscan", "true_label": "y_malscan"}), on="sha")
        .merge(read("RAMDA", {"vae_score": "ramda_vae", "mlp_score": "ramda_mlp", "true_label": "y_ramda"}), on="sha")
    )
    if not ((df["y"] == df["y_malscan"]) & (df["y"] == df["y_ramda"])).all():
        raise RuntimeError(f"{split}: labels differ across detectors")
    df["y"] = df["y"].astype(int)
    return df[["sha", "y", "drebin", "malscan", "ramda_vae", "ramda_mlp"]]


# ---------------------------------------------------------------- metrics


def metrics(y, pred):
    """Metrics on the decided samples (pred == -1 is deferred)."""
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    decided = tp + fp + fn + tn

    def div(a, b):
        return a / b if b > 0 else 0.0

    mal_f1, ben_f1 = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fn + fp)
    return {
        "accuracy": div(tp + tn, decided),
        "mal_precision": div(tp, tp + fp),
        "mal_recall": div(tp, tp + fn),
        "mal_f1": mal_f1,
        "ben_f1": ben_f1,
        "deferred_pct": 100.0 * (1.0 - decided / len(y)),
        "n": len(y),
        "decided": int(decided),
        "deferred": int(len(y) - decided),
    }


def bootstrap_f1_diff(y, a, b, n_boot=2000, seed=0):
    """Paired bootstrap of malware F1(a) - F1(b): mean and 95% CI."""
    idx = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))

    def f1(pred):
        yy, pp = y[idx], pred[idx]
        tp = ((pp == 1) & (yy == 1)).sum(1)
        fp = ((pp == 1) & (yy == 0)).sum(1)
        fn = ((pp == 0) & (yy == 1)).sum(1)
        return 2 * tp / (2 * tp + fp + fn)

    d = f1(a) - f1(b)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------- thresholds


def _counts_ge(sorted_scores, t):
    return len(sorted_scores) - np.searchsorted(sorted_scores, t, side="left")


def _counts_le(sorted_scores, t):
    return np.searchsorted(sorted_scores, t, side="right")


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def best_single_threshold(y, s):
    """Threshold t (malware if s >= t) with the highest malware F1."""
    mal, ben = np.sort(s[y == 1]), np.sort(s[y == 0])
    t = np.unique(s)
    tp, fp = _counts_ge(mal, t), _counts_ge(ben, t)
    f1 = _f1(tp, fp, len(mal) - tp)
    best = int(np.argmax(f1))
    return float(t[best]), float(f1[best])


def ramda_single_thresholds(y, vae, mlp):
    """RAMDA's own rule: malware if vae > t_vae OR mlp > t_mlp. Same grid search
    as RAMDA's classification pipeline (50 VAE x 201 MLP thresholds)."""
    vae_grid = np.linspace(vae.min(), vae.max(), 50)
    mlp_grid = np.linspace(0, 1, 201)
    mlp_fires = mlp[None, :] > mlp_grid[:, None]
    n_mal = int(np.sum(y == 1))
    best = (-1.0, None, None)
    for t_vae in vae_grid:
        pred = mlp_fires | (vae > t_vae)[None, :]
        tp = np.sum(pred & (y == 1), axis=1)
        fp = np.sum(pred & (y == 0), axis=1)
        f1 = _f1(tp, fp, n_mal - tp)
        i = int(np.argmax(f1))
        if f1[i] > best[0]:
            best = (float(f1[i]), float(t_vae), float(mlp_grid[i]))
    return best


def floor_settings():
    """(kind, malware floor, benign floor, gap) explored by the dual thresholds.
    'relative' floors are deltas below the score's best validation F1."""
    rel = [("relative", m, b, g) for m, b in itertools.product(REL_DELTAS, REL_DELTAS) for g in GAPS]
    fixed = [("fixed", m, b, g) for m, b in itertools.product(FIXED_FLOORS, FIXED_FLOORS) for g in GAPS]
    return rel + fixed


class DualSweep:
    """Dual-threshold selection of the paper, for any F1 floors and gap.

    thr_malware = highest threshold whose malware F1 is >= the malware floor;
    thr_benign  = lowest threshold <= thr_malware - gap whose benign F1 is >= the
    benign floor. If a floor cannot be met, the F1 argmax is used instead.
    The per-class F1 curves are computed once and reused for every setting.
    """

    def __init__(self, y, s):
        self.t = np.unique(s)
        mal, ben = np.sort(s[y == 1]), np.sort(s[y == 0])
        tp, fp = _counts_ge(mal, self.t), _counts_ge(ben, self.t)
        self.mal_f1 = _f1(tp, fp, len(mal) - tp)
        tn, fn = _counts_le(ben, self.t), _counts_le(mal, self.t)
        self.ben_f1 = _f1(tn, fn, len(ben) - tn)

    def select(self, setting):
        kind, p_mal, p_ben, gap = setting
        if kind == "relative":
            mal_floor, ben_floor = self.mal_f1.max() - p_mal, self.ben_f1.max() - p_ben
        else:
            mal_floor, ben_floor = p_mal, p_ben
        ok = np.flatnonzero(self.mal_f1 >= mal_floor)
        mi = ok[-1] if len(ok) else int(np.argmax(self.mal_f1))
        thr_malware = self.t[mi]
        valid = self.t <= thr_malware - gap
        ok = np.flatnonzero((self.ben_f1 >= ben_floor) & valid)
        if len(ok):
            bi = ok[0]
        else:
            vi = np.flatnonzero(valid)
            bi = vi[np.argmax(self.ben_f1[vi])] if len(vi) else int(np.argmax(self.ben_f1))
        return float(self.t[bi]), float(thr_malware)


def best_band(y, s, budget, grid=400):
    """Thresholds with the highest malware F1 on the decided samples, deferring
    at most `budget` of the samples. Budget 0 means a single threshold."""
    if budget == 0:
        t, _ = best_single_threshold(y, s)
        return float(np.nextafter(t, -np.inf)), t

    qs = np.unique(np.quantile(s, np.linspace(0, 1, grid)))
    mal, ben = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp, fp = _counts_ge(mal, qs)[None, :], _counts_ge(ben, qs)[None, :]  # columns: thr_malware
    fn, tn = _counts_le(mal, qs)[:, None], _counts_le(ben, qs)[:, None]  # rows: thr_benign
    deferral = 1.0 - (tp + fp + fn + tn) / len(s)
    f1 = _f1(tp, fp, fn)
    score = np.where((qs[:, None] < qs[None, :]) & (deferral <= budget + 1e-12), f1, -1.0)
    best = np.flatnonzero(score == score.max())
    i, j = np.unravel_index(best[np.argmin(deferral.ravel()[best])], score.shape)
    return float(qs[i]), float(qs[j])


def min_deferral_band(y, s, required, grid=400):
    """Thresholds deferring as few samples as possible while malware F1 and
    malware recall on the decided samples stay >= `required`. None if impossible."""
    qs = np.unique(np.quantile(s, np.linspace(0, 1, grid)))
    mal, ben = np.sort(s[y == 1]), np.sort(s[y == 0])

    # a single threshold (no deferral) is the best case
    tp, fp = _counts_ge(mal, qs), _counts_ge(ben, qs)
    f1, rec = _f1(tp, fp, len(mal) - tp), tp / len(mal)
    ok = np.flatnonzero((f1 >= required) & (rec >= required))
    if len(ok):
        t = qs[ok[np.argmax(f1[ok])]]
        return float(np.nextafter(t, -np.inf)), float(t)

    tp, fp = tp[None, :], fp[None, :]
    fn, tn = _counts_le(mal, qs)[:, None], _counts_le(ben, qs)[:, None]
    deferral = 1.0 - (tp + fp + fn + tn) / len(s)
    f1 = _f1(tp, fp, fn)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(f1), where=(tp + fn) > 0)
    valid = (qs[:, None] < qs[None, :]) & (f1 >= required) & (rec >= required)
    if not valid.any():
        return None
    cost = np.where(valid, deferral, np.inf)
    best = np.flatnonzero(cost == cost.min())
    i, j = np.unravel_index(best[np.argmax(f1.ravel()[best])], cost.shape)
    return float(qs[i]), float(qs[j])


def tri(s, thr_benign, thr_malware):
    """1 if s >= thr_malware, 0 if s <= thr_benign, -1 (deferred) in between."""
    return np.where(s >= thr_malware, 1, np.where(s <= thr_benign, 0, -1)).astype(np.int8)


# ------------------------------------------------------------ calibration


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _platt_input(col, scores):
    """Put raw scores on an unbounded scale before fitting the sigmoid."""
    scores = np.asarray(scores, dtype=float)
    if col == "drebin":  # SVM margin
        return scores
    if col == "ramda_vae":  # reconstruction distance, positive and heavy-tailed
        return np.log1p(np.maximum(scores, 0.0))
    return _logit(scores)  # probabilities


def fit_platt(x, y, n_iter=100):
    """Platt scaling p = sigmoid(a*x + b), fitted with Newton's method."""
    mu, sd = float(np.mean(x)), float(np.std(x)) or 1.0
    z = (x - mu) / sd
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        p = _sigmoid(a * z + b)
        grad = np.array([np.sum((p - y) * z), np.sum(p - y)])
        w = p * (1 - p)
        hess = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return a / sd, b - a * mu / sd


def calibrate(val, test):
    """Malware probability of every score, Platt-scaled on validation.

    RAMDA keeps the OR of its two heads: its probability is the max of the two
    calibrated heads, recalibrated on validation.
    """
    y = val["y"].to_numpy()
    cal = {"val": {}, "test": {}}
    for col in ["drebin", "malscan", "ramda_vae", "ramda_mlp"]:
        xv, xt = _platt_input(col, val[col]), _platt_input(col, test[col])
        a, b = fit_platt(xv, y)
        cal["val"][col], cal["test"][col] = _sigmoid(a * xv + b), _sigmoid(a * xt + b)
    x_or = {s: _logit(np.maximum(cal[s]["ramda_vae"], cal[s]["ramda_mlp"])) for s in cal}
    a, b = fit_platt(x_or["val"], y)
    for s in cal:
        cal[s]["ramda"] = _sigmoid(a * x_or[s] + b)
    return cal


def fit_logreg(X, y, l2=1e-3, n_iter=100):
    """Logistic regression with intercept (Newton's method, small L2), for stacking."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    reg = l2 * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(n_iter):
        p = _sigmoid(Z @ w)
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + reg, Z.T @ (p - y) + reg @ w)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w[:-1] / sd, float(w[-1] - np.sum(w[:-1] * mu / sd))


# ------------------------------------------------------------------- data


class Data:
    """Scores, calibrated scores and the combined scores used by the tables."""

    COLS = {"Drebin": "drebin", "MalScan": "malscan", "RAMDA": "ramda"}

    def __init__(self):
        self.val, self.test = load_scores("val"), load_scores("test")
        self.y_val, self.y_test = self.val["y"].to_numpy(), self.test["y"].to_numpy()
        self.cal = calibrate(self.val, self.test)

        # validation F1 of each detector alone (RAMDA with its OR rule): the weights
        t_d, f_d = best_single_threshold(self.y_val, self.val["drebin"].to_numpy())
        t_m, f_m = best_single_threshold(self.y_val, self.val["malscan"].to_numpy())
        f_r, t_vae, t_mlp = ramda_single_thresholds(
            self.y_val, self.val["ramda_vae"].to_numpy(), self.val["ramda_mlp"].to_numpy())
        self.f1 = {"Drebin": f_d, "MalScan": f_m, "RAMDA": f_r}
        self.single_thresholds = {"drebin": t_d, "malscan": t_m, "ramda_vae": t_vae, "ramda_mlp": t_mlp}
        prevalence = float(np.mean(self.y_val))
        self.trivial_f1 = 2 * prevalence / (1 + prevalence)  # F1 of "everything is malware"

        self.scores = self._combined_scores()
        self._sweeps = {}

    def _combined_scores(self):
        """name -> {"val", "test", "weights", "fusion"}; weights is None when the
        score is not a weighted average (single detectors, stacking)."""
        out = {}
        for d in DETECTORS:
            out[SHORT[d]] = {s: self.cal[s][self.COLS[d]] for s in ["val", "test"]} | {"weights": None, "fusion": None}

        def weighted(subset, w, fusion):
            w = {d: w[d] / sum(w.values()) for d in subset}
            if fusion == "mean prob":
                score = {s: sum(w[d] * self.cal[s][self.COLS[d]] for d in subset) for s in ["val", "test"]}
            else:  # mean of logits, mapped back to a probability
                score = {s: _sigmoid(sum(w[d] * _logit(self.cal[s][self.COLS[d]]) for d in subset)) for s in ["val", "test"]}
            return score | {"weights": w, "fusion": fusion}

        # Test 2 as specified: Drebin+MalScan+RAMDA, weights = validation F1
        out["D+M+R | F1 | mean prob"] = weighted(DETECTORS, dict(self.f1), "mean prob")

        # Test 2 without RAMDA: several weightings, then stacking
        pair = ["Drebin", "MalScan"]
        schemes = {f"F1^{p}": {d: self.f1[d] ** p for d in pair} for p in [1, 3, 10]}
        schemes["uniform"] = {d: 1.0 for d in pair}
        schemes["F1-trivial"] = {d: self.f1[d] - self.trivial_f1 for d in pair}
        for name, w in schemes.items():
            for fusion in ["mean prob", "mean logit"]:
                out[f"D+M | {name} | {fusion}"] = weighted(pair, w, fusion)
        feats = {s: np.column_stack([_logit(self.cal[s][self.COLS[d]]) for d in pair]) for s in ["val", "test"]}
        coef, intercept = fit_logreg(feats["val"], self.y_val)
        out["D+M | stacking"] = {s: _sigmoid(feats[s] @ coef + intercept) for s in ["val", "test"]} | {"weights": None, "fusion": None}
        return out

    def sweep_of(self, key):
        """DualSweep of a validation score (cached). key: score name or calibrated column."""
        if key not in self._sweeps:
            s = self.scores[key]["val"] if key in self.scores else self.cal["val"][key]
            self._sweeps[key] = DualSweep(self.y_val, s)
        return self._sweeps[key]


# ------------------------------------------------------- configurations


def decide(config, data):
    """Decisions (val, test) of one configuration.

    config keys: method, score, thresholds, floor_kind, floor_malware,
    floor_benign, gap, budget (unused keys are NaN or "-").
    """
    if config["method"] in ("Test 1", "Test 3"):
        return _vote(config, data)

    score = data.scores[config["score"]]
    setting = (config["floor_kind"], config["floor_malware"], config["floor_benign"], config["gap"])
    if config["thresholds"] == "paper floors":
        thr_benign, thr_malware = data.sweep_of(config["score"]).select(setting)
    elif config["thresholds"] == "averaged detector thresholds":
        # each detector's own dual thresholds, averaged with the score's weights
        w = score["weights"]
        thr = {d: data.sweep_of(Data.COLS[d]).select(setting) for d in w}
        if score["fusion"] == "mean prob":
            thr_benign, thr_malware = (sum(w[d] * thr[d][i] for d in w) for i in (0, 1))
        else:
            thr_benign, thr_malware = (float(_sigmoid(sum(w[d] * _logit(thr[d][i]) for d in w))) for i in (0, 1))
    elif config["thresholds"] == "best band":
        thr_benign, thr_malware = best_band(data.y_val, score["val"], config["budget"])
    else:
        raise ValueError(config["thresholds"])
    return tri(score["val"], thr_benign, thr_malware), tri(score["test"], thr_benign, thr_malware)


def _vote(config, data):
    """Test 1 (majority) and Test 3 (F1-weighted) votes of Drebin, MalScan, RAMDA."""
    out = []
    for split, df in [("val", data.val), ("test", data.test)]:
        if config["thresholds"] == "single threshold":
            # no abstention: each detector with its best single threshold (raw scores)
            t = data.single_thresholds
            votes = {
                "Drebin": (df["drebin"].to_numpy() >= t["drebin"]).astype(np.int8),
                "MalScan": (df["malscan"].to_numpy() >= t["malscan"]).astype(np.int8),
                "RAMDA": ((df["ramda_vae"].to_numpy() > t["ramda_vae"])
                          | (df["ramda_mlp"].to_numpy() > t["ramda_mlp"])).astype(np.int8),
            }
        else:
            setting = (config["floor_kind"], config["floor_malware"], config["floor_benign"], config["gap"])
            v = {c: tri(data.cal[split][c], *data.sweep_of(c).select(setting))
                 for c in ["drebin", "malscan", "ramda_vae", "ramda_mlp"]}
            # RAMDA: malware if a head says malware, benign only if both say benign
            ramda = np.full(len(df), -1, dtype=np.int8)
            ramda[(v["ramda_vae"] == 1) | (v["ramda_mlp"] == 1)] = 1
            ramda[(v["ramda_vae"] == 0) & (v["ramda_mlp"] == 0)] = 0
            votes = {"Drebin": v["drebin"], "MalScan": v["malscan"], "RAMDA": ramda}

        if config["method"] == "Test 1":  # at least 2 of 3 votes, otherwise deferred
            n_mal = sum((votes[d] == 1).astype(int) for d in DETECTORS)
            n_ben = sum((votes[d] == 0).astype(int) for d in DETECTORS)
            out.append(np.where(n_mal >= 2, 1, np.where(n_ben >= 2, 0, -1)).astype(np.int8))
        else:  # votes weighted by validation F1: a class needs more than half the weight
            total = sum(data.f1.values())
            w_mal = sum((data.f1[d] / total) * (votes[d] == 1) for d in DETECTORS)
            w_ben = sum((data.f1[d] / total) * (votes[d] == 0) for d in DETECTORS)
            out.append(np.where(w_mal > 0.5, 1, np.where(w_ben > 0.5, 0, -1)).astype(np.int8))
    return out[0], out[1]


def describe(config):
    """One-line description of a configuration."""
    if config["method"] in ("Test 1", "Test 3"):
        what = "majority of D, M, R" if config["method"] == "Test 1" else "F1-weighted vote of D, M, R"
        what += " (RAMDA = OR of its heads)"
    else:
        what = f"score {config['score']}"
    how = config["thresholds"]
    if config["floor_kind"] == "relative":
        how += f", F1 floors = best - ({config['floor_malware']:g}, {config['floor_benign']:g}), gap {config['gap']:g}"
    elif config["floor_kind"] == "fixed":
        how += f", F1 floors = ({config['floor_malware']:g}, {config['floor_benign']:g}), gap {config['gap']:g}"
    elif how == "best band":
        how += f", validation budget {100 * config['budget']:g}%"
    return f"{what}; {how}"


def load_sweep():
    if not SWEEP_FILE.exists():
        raise FileNotFoundError(f"{SWEEP_FILE} not found: run sweep.py first")
    return pd.read_csv(SWEEP_FILE)


def pick(rows, budget):
    """Configuration with the best validation F1 among those deferring at most
    `budget` of validation (ties: less deferral, then first in the sweep)."""
    rows = rows[rows["val_deferred_pct"] <= 100 * budget + 1e-9]
    if rows.empty:
        return None
    order = np.lexsort((np.arange(len(rows)), rows["val_deferred_pct"].to_numpy(), -rows["val_mal_f1"].to_numpy()))
    return rows.iloc[order[0]].to_dict()
