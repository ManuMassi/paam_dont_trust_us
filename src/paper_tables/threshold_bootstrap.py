"""Variability of Table 2 including the choice of the thresholds.

table2_min_deferral.py bootstraps the test set only, with thresholds fixed on
validation. But the deferral depends mostly on *where* the two thresholds end
up, which is itself random: a different validation sample would give different
thresholds. Here every bootstrap replicate
  1. resamples validation and re-selects the thresholds (min_deferral_band),
  2. resamples test and measures deferral, F1 and recall.
The same validation and test resamples are used for every method, so the
comparison with Drebin stays paired.

The score of every method is kept fixed (for "Test 2 without RAMDA", the one
chosen on the full validation set by table2_min_deferral.py): only the
thresholds are re-selected. The votes (Tests 1 and 3) are not included, since
their thresholds come from the sweep and not from a single score.

Output: results/Ensemble/tables/threshold_bootstrap.csv
The smallest bootstrap p-value is 2 / (n_boot + 1): with 16 comparisons Holm
needs n_boot >= 2000 or so to reach p < 0.05 even for clear differences.

Run:    python3 threshold_bootstrap.py [--n-boot 2000] [--seed 0]
"""

import argparse
import time

import numpy as np
import pandas as pd

import common as C
import stats as S
from table2_min_deferral import FIXED_SCORES, FUSION, REF, REQUIRED, choose_fusion_score

OUT_FILE = C.OUT_DIR / "threshold_bootstrap.csv"
KEYS = ["deferred_pct", "mal_f1", "mal_recall"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    data = C.Data()
    yv, yt = data.y_val, data.y_test
    q = [100 * S.ALPHA / 2, 100 * (1 - S.ALPHA / 2)]

    rows = []
    for r in REQUIRED:
        methods = dict(FIXED_SCORES)
        methods[FUSION] = choose_fusion_score(data, r)[0]
        rng = np.random.default_rng(args.seed)
        reps = {m: {k: np.full(args.n_boot, np.nan) for k in KEYS} for m in methods}
        for b in range(args.n_boot):
            iv = rng.integers(0, len(yv), len(yv))
            it = rng.integers(0, len(yt), len(yt))
            for name, score in methods.items():
                s = data.scores[score]
                thr = C.min_deferral_band(yv[iv], s["val"][iv], r)
                if thr is None:  # requirement not reachable on this validation resample
                    continue
                m = C.metrics(yt[it], C.tri(s["test"][it], *thr))
                for k in KEYS:
                    reps[name][k][b] = m[k]

        ref = reps[REF]["deferred_pct"]
        for name, score in methods.items():
            v = reps[name]
            ok = ~np.isnan(v["deferred_pct"])
            row = {"required_f1_and_recall": r, "method": name, "score": score,
                   "n_boot": args.n_boot, "reachable_share": float(ok.mean())}
            for k in KEYS:
                row[f"{k}_mean"] = float(np.nanmean(v[k]))
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = np.nanpercentile(v[k], q)
            meets = (v["mal_f1"] >= r) & (v["mal_recall"] >= r)
            row["meets_r_share"] = float(meets[ok].mean()) if ok.any() else np.nan
            if name != REF:
                both = ok & ~np.isnan(ref)
                d = v["deferred_pct"][both] - ref[both]
                row["dDeferred_pp_mean"] = float(d.mean())
                row["dDeferred_ci_low"], row["dDeferred_ci_high"] = np.percentile(d, q)
                row["p_dDeferred"] = S.p_two_sided(d)
                row["dDeferred_ci_excludes_0"] = bool(row["dDeferred_ci_high"] < 0 or row["dDeferred_ci_low"] > 0)
                rel = d / np.where(ref[both] > 0, ref[both], np.nan) * 100
                row["dDeferred_rel_pct_median"] = float(np.nanmedian(rel))
                df1 = v["mal_f1"][both] - reps[REF]["mal_f1"][both]
                row["dF1_mean"] = float(df1.mean())
                row["dF1_ci_low"], row["dF1_ci_high"] = np.percentile(df1, q)
            rows.append(row)
        print(f"  r = {r} done ({time.time() - t0:.0f}s)")

    table = pd.DataFrame(rows)
    table["p_dDeferred_holm"] = S.holm(table["p_dDeferred"].to_numpy())
    table.to_csv(OUT_FILE, index=False)

    view = pd.DataFrame({
        "r": table["required_f1_and_recall"], "method": table["method"],
        "Def% [CI]": [S.fmt_ci(t.deferred_pct_mean, t.deferred_pct_ci_low, t.deferred_pct_ci_high, 1)
                      for t in table.itertuples()],
        "F1 [CI]": [S.fmt_ci(t.mal_f1_mean, t.mal_f1_ci_low, t.mal_f1_ci_high) for t in table.itertuples()],
        "dDef pp [CI]": [S.fmt_ci(t.dDeferred_pp_mean, t.dDeferred_ci_low, t.dDeferred_ci_high, 1)
                         if t.method != REF else "-" for t in table.itertuples()],
        "p Holm": table["p_dDeferred_holm"], "meets r": table["meets_r_share"],
    })
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print(f"\nSaved {OUT_FILE} ({time.time() - t0:.0f}s)")
