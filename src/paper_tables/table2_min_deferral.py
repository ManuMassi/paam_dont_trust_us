"""Table 2: fewest deferred samples for a required quality, with statistical tests.

The other way round with respect to Table 1: fix the quality and minimise the
deferral. For each required value r and method, choose on validation the
thresholds that defer as few samples as possible while malware F1 >= r and
malware recall >= r on the decided samples; then measure deferral, F1 and
recall on test (they can end up slightly below r).

Score-based methods search the two thresholds directly on their score. For
"Test 2 without RAMDA" the score itself (weighting, fusion, stacking) is also
chosen on validation, separately for every r: the one deferring the fewest
validation samples (ties: higher validation F1). The votes (Tests 1 and 3, which
take the same decisions) pick among their configurations in the sweep.

Statistics (test set, every method compared with Drebin at the same r):
  - 95% CI of deferral, F1 and recall (paired bootstrap, B = 10,000);
  - dDeferred in percentage points and relative to Drebin, with CI;
  - exact McNemar on the deferral indicator (the main test: does the method
    send fewer samples to stage 2?), Holm-adjusted over all (r, method) pairs;
  - dF1 with CI and non-inferiority (margin NI_MARGIN);
  - whether the test F1 and recall are not significantly below r (upper CI >= r).
This measures test sampling only; threshold_bootstrap.py adds the variability
of the thresholds chosen on validation.

Input:  results/Ensemble/tables/sweep_configs.csv.gz (run sweep.py first)
Output, in results/Ensemble/tables/:
  table2_min_deferral.csv            the table with all statistics
  table2_decisions_per_sample.csv    test decisions of every row (1, 0, -1)
"""

import numpy as np
import pandas as pd

import common as C
import stats as S

OUT_FILE = C.OUT_DIR / "table2_min_deferral.csv"
DECISIONS_FILE = C.OUT_DIR / "table2_decisions_per_sample.csv"
REQUIRED = [0.87, 0.88, 0.89, 0.90]
REF = "Drebin"
FIXED_SCORES = [  # (name in the table, score)
    ("Drebin", "D"),
    ("MalScan", "M"),
    ("RAMDA", "R"),
    ("Test 2 (weighted avg.)", "D+M+R | F1 | mean prob"),
]
FUSION = "Test 2 without RAMDA"
VOTING = "Tests 1 and 3 (voting)"
ORDER = ["Drebin", "MalScan", "RAMDA", VOTING, "Test 2 (weighted avg.)", FUSION]
METRICS = ["deferred_pct", "accuracy", "mal_precision", "mal_recall", "mal_f1"]
NI_MARGIN = 0.01


def tests_1_and_3_identical(sweep):
    cols = ["floor_kind", "floor_malware", "floor_benign", "gap", "thresholds"] + \
           [c for c in sweep.columns if c.startswith(("val_", "test_"))]
    t1 = sweep[sweep["method"] == "Test 1"][cols].reset_index(drop=True)
    t3 = sweep[sweep["method"] == "Test 3"][cols].reset_index(drop=True)
    return t1.equals(t3)


def min_deferral(data, score, r):
    """Thresholds of `score` for requirement r, with validation metrics (None if unreachable)."""
    s = data.scores[score]
    thr = C.min_deferral_band(data.y_val, s["val"], r)
    if thr is None:
        return None
    return thr, C.metrics(data.y_val, C.tri(s["val"], *thr))


def choose_fusion_score(data, r):
    """Among the Drebin+MalScan scores, the one deferring least on validation."""
    best = None
    for score in [k for k in data.scores if k.startswith("D+M |")]:
        res = min_deferral(data, score, r)
        if res is None:
            continue
        key = (res[1]["deferred_pct"], -res[1]["mal_f1"])
        if best is None or key < best[0]:
            best = (key, score, res)
    assert best is not None, f"{FUSION}: {r} not reachable on validation"
    return best[1], best[2]


if __name__ == "__main__":
    data, sweep = C.Data(), C.load_sweep()
    y = data.y_test
    assert tests_1_and_3_identical(sweep), "Tests 1 and 3 do not coincide"
    votes = sweep[sweep["method"] == "Test 1"]

    rows, per_sample = [], {"sha": data.test["sha"], "true_label": y}
    for r in REQUIRED:
        decisions, info = {}, {}
        for name, score in FIXED_SCORES:
            res = min_deferral(data, score, r)
            if res is None:
                print(f"  {name}: r = {r} not reachable on validation, skipped")
                continue
            thr, m_val = res
            decisions[name] = C.tri(data.scores[score]["test"], *thr)
            info[name] = {"m_val": m_val, "thr_benign": thr[0], "thr_malware": thr[1], "config": f"score {score}"}

        score, (thr, m_val) = choose_fusion_score(data, r)
        decisions[FUSION] = C.tri(data.scores[score]["test"], *thr)
        info[FUSION] = {"m_val": m_val, "thr_benign": thr[0], "thr_malware": thr[1], "config": f"score {score}"}

        # voting: least validation deferral among the configurations meeting r (ties: higher F1)
        ok = votes[(votes["val_mal_f1"] >= r) & (votes["val_mal_recall"] >= r)]
        if len(ok):
            best = ok.iloc[np.lexsort((np.arange(len(ok)), -ok["val_mal_f1"].to_numpy(),
                                       ok["val_deferred_pct"].to_numpy()))[0]]
            decisions[VOTING] = C.decide(best.to_dict(), data)[1]
            info[VOTING] = {"m_val": {k: best[f"val_{k}"] for k in METRICS}, "config": C.describe(best)}
        else:
            print(f"  {VOTING}: r = {r} not reachable on validation, skipped")

        assert REF in decisions, f"{REF}: {r} not reachable on validation"
        boot = S.bootstrap(y, decisions, ref=REF)
        for name in [n for n in ORDER if n in decisions]:
            d, b = decisions[name], boot[name]
            m_test = C.metrics(y, d)
            row = {"required_f1_and_recall": r, "method": name,
                   **{f"val_{k}": info[name]["m_val"][k] for k in METRICS},
                   **{f"test_{k}": m_test[k] for k in METRICS}}
            for k in ["deferred_pct", "mal_f1", "mal_recall"]:
                row[f"test_{k}_ci_low"], row[f"test_{k}_ci_high"] = b[k]
            row["test_submissions_per_1000"] = 10 * m_test["deferred_pct"]
            row["test_meets_r"] = m_test["mal_f1"] >= r and m_test["mal_recall"] >= r
            row["test_not_signif_below_r"] = b["mal_f1"][1] >= r and b["mal_recall"][1] >= r
            if name != REF:
                dD, lo, hi, p = b["d_deferred_pct"]
                ref_def = C.metrics(y, decisions[REF])["deferred_pct"]
                rel = 100 * (m_test["deferred_pct"] - ref_def) / ref_def if ref_def > 0 else np.nan
                row.update(dDeferred_pp=dD, dDeferred_ci_low=lo, dDeferred_ci_high=hi, p_dDeferred=p,
                           dDeferred_rel_pct=rel)
                only_ref, only_m, p = S.mcnemar_deferral(decisions[REF], d)
                row.update(only_drebin_defers=only_ref, only_method_defers=only_m, p_mcnemar_deferral=p)
                dF1, lo, hi, p = b["d_mal_f1"]
                row.update(dF1=dF1, dF1_ci_low=lo, dF1_ci_high=hi, p_dF1=p,
                           f1_noninferior=S.noninferior(lo, NI_MARGIN))
            row.update({k: info[name].get(k, np.nan) for k in ["thr_benign", "thr_malware"]})
            row["config"] = info[name]["config"]
            rows.append(row)
            per_sample[f"r={r} | {name}"] = d

    table = pd.DataFrame(rows)
    # Holm over all (r, method) comparisons with Drebin
    for col in ["p_mcnemar_deferral", "p_dDeferred", "p_dF1"]:
        table[f"{col}_holm"] = S.holm(table[col].to_numpy())
    table["fewer_deferrals_significant"] = (table["p_mcnemar_deferral_holm"] < S.ALPHA) & (table["dDeferred_pp"] < 0)
    table.to_csv(OUT_FILE, index=False)
    pd.DataFrame(per_sample).to_csv(DECISIONS_FILE, index=False)

    view = pd.DataFrame({
        "r": table["required_f1_and_recall"], "method": table["method"],
        "Def% test [CI]": [S.fmt_ci(t.test_deferred_pct, t.test_deferred_pct_ci_low, t.test_deferred_pct_ci_high, 1)
                           for t in table.itertuples()],
        "F1 test [CI]": [S.fmt_ci(t.test_mal_f1, t.test_mal_f1_ci_low, t.test_mal_f1_ci_high)
                         for t in table.itertuples()],
        "dDef pp": table["dDeferred_pp"], "dDef %": table["dDeferred_rel_pct"],
        "p McNemar def Holm": table["p_mcnemar_deferral_holm"],
        "F1 non-inf.": table["f1_noninferior"], "meets r": table["test_meets_r"],
        "not < r": table["test_not_signif_below_r"],
    })
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print("\nFusion score chosen on validation per r:")
    for t in table[table["method"] == FUSION].itertuples():
        print(f"  r = {t.required_f1_and_recall}: {t.config}")
    print(f"\nSaved {OUT_FILE}\nSaved {DECISIONS_FILE}")
