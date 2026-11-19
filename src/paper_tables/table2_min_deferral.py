"""Table 2: fewest deferred samples for a required quality.

The other way round with respect to Table 1: fix the quality and minimise the
deferral. For each required value r and method, choose on validation the
thresholds that defer as few samples as possible while malware F1 >= r and
malware recall >= r on the decided samples; then measure deferral, F1 and
recall on test (they can end up slightly below r).

Score-based methods search the two thresholds directly on their score; the
votes (Tests 1 and 3) pick among their configurations in the sweep. Tests 1
and 3 share a row: the script checks that they take the same decisions.

Input:  results/Ensemble/tables/sweep_configs.csv.gz (run sweep.py first)
Output: results/Ensemble/tables/table2_min_deferral.csv
"""

import numpy as np
import pandas as pd

import common as C

OUT_FILE = C.OUT_DIR / "table2_min_deferral.csv"
REQUIRED = [0.87, 0.88, 0.89, 0.90]
SCORE_METHODS = [  # (name in the table, score)
    ("Drebin", "D"),
    ("Test 2 (weighted avg.)", "D+M+R | F1 | mean prob"),
    ("Test 2 without RAMDA", "D+M | F1^10 | mean prob"),
]
VOTING = "Tests 1 and 3 (voting)"
ORDER = ["Drebin", VOTING, "Test 2 (weighted avg.)", "Test 2 without RAMDA"]
METRICS = ["deferred_pct", "accuracy", "mal_precision", "mal_recall", "mal_f1"]


def tests_1_and_3_identical(sweep):
    cols = ["floor_kind", "floor_malware", "floor_benign", "gap", "thresholds"] + \
           [c for c in sweep.columns if c.startswith(("val_", "test_"))]
    t1 = sweep[sweep["method"] == "Test 1"][cols].reset_index(drop=True)
    t3 = sweep[sweep["method"] == "Test 3"][cols].reset_index(drop=True)
    return t1.equals(t3)


if __name__ == "__main__":
    data, sweep = C.Data(), C.load_sweep()
    assert tests_1_and_3_identical(sweep), "Tests 1 and 3 do not coincide"
    votes = sweep[sweep["method"] == "Test 1"]

    rows = []
    for r in REQUIRED:
        for name, score in SCORE_METHODS:
            s = data.scores[score]
            thr = C.min_deferral_band(data.y_val, s["val"], r)
            assert thr is not None, f"{name}: {r} not reachable on validation"
            m_val = C.metrics(data.y_val, C.tri(s["val"], *thr))
            m_test = C.metrics(data.y_test, C.tri(s["test"], *thr))
            rows.append({"required_f1_and_recall": r, "method": name,
                         **{f"val_{k}": m_val[k] for k in METRICS}, **{f"test_{k}": m_test[k] for k in METRICS},
                         "thr_benign": thr[0], "thr_malware": thr[1], "config": f"score {score}"})

        # voting: least validation deferral among the configurations meeting r (ties: higher F1)
        ok = votes[(votes["val_mal_f1"] >= r) & (votes["val_mal_recall"] >= r)]
        assert len(ok), f"voting: {r} not reachable on validation"
        best = ok.iloc[np.lexsort((np.arange(len(ok)), -ok["val_mal_f1"].to_numpy(), ok["val_deferred_pct"].to_numpy()))[0]]
        rows.append({"required_f1_and_recall": r, "method": VOTING,
                     **{f"val_{k}": best[f"val_{k}"] for k in METRICS}, **{f"test_{k}": best[f"test_{k}"] for k in METRICS},
                     "config": C.describe(best)})

    table = pd.DataFrame(rows)
    table["order"] = table["method"].map(ORDER.index)
    table = table.sort_values(["required_f1_and_recall", "order"]).drop(columns="order")
    table.to_csv(OUT_FILE, index=False)
    cols = ["required_f1_and_recall", "method", "val_deferred_pct", "test_deferred_pct", "test_mal_f1", "test_mal_recall"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {OUT_FILE}")
