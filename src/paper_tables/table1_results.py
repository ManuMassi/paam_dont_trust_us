"""Table 1: accuracy, precision, recall and F1 of Tests 1-3 on the test set.

Two settings:
  single threshold  every sample is decided (no deferral);
  dual threshold    samples between the two thresholds are deferred to stage 2,
                    at most 6.71% of validation (what Drebin defers with the
                    paper's procedure).
For each method and setting we take from the sweep the configuration with the
best validation F1 within that deferral, and measure it on test. Metrics are on
the decided samples. F1 is compared with Drebin by paired bootstrap.

Input:  results/Ensemble/tables/sweep_configs.csv.gz (run sweep.py first)
Output: results/Ensemble/tables/table1_results.csv
"""

import pandas as pd

import common as C

OUT_FILE = C.OUT_DIR / "table1_results.csv"
SETTINGS = [("Single threshold", 0.0), ("Dual threshold", C.BUDGET)]
METHODS = [  # (method in the sweep, name in the table)
    ("Drebin", "Drebin"),
    ("Test 1", "Test 1 (majority vote)"),
    ("Test 3", "Test 3 (weighted vote)"),
    ("Test 2", "Test 2 (weighted avg.)"),
    ("Test 2 without RAMDA", "Test 2 without RAMDA"),
]

if __name__ == "__main__":
    data, sweep = C.Data(), C.load_sweep()
    rows = []
    for setting, budget in SETTINGS:
        decisions, configs = {}, {}
        for method, _ in METHODS:
            configs[method] = C.pick(sweep[sweep["method"] == method], budget)
            decisions[method] = C.decide(configs[method], data)[1]  # test decisions

        for method, name in METHODS:
            m = C.metrics(data.y_test, decisions[method])
            d_f1, low, high = C.bootstrap_f1_diff(data.y_test, decisions[method], decisions["Drebin"])
            rows.append({
                "setting": setting, "method": name,
                **{k: m[k] for k in ["accuracy", "mal_precision", "mal_recall", "mal_f1", "ben_f1",
                                     "deferred_pct", "n", "decided", "deferred"]},
                "dF1_vs_drebin": d_f1, "dF1_ci_low": low, "dF1_ci_high": high,
                "significant": method != "Drebin" and low > 0,  # 95% CI above zero
                "selected_config": C.describe(configs[method]),
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT_FILE, index=False)
    cols = ["setting", "method", "accuracy", "mal_precision", "mal_recall", "mal_f1", "deferred_pct", "significant"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {OUT_FILE}")
