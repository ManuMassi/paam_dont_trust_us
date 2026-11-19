"""Table 3: which test APKs each classifier defers to stage 2.

Each classifier uses its dual-threshold configuration of Table 1 (best
validation F1, at most 6.71% of validation deferred). For the samples deferred
by one classifier we count how many the others defer too ("shared with") and
how often the others are right on them when they decide ("accuracy of").

Input:  results/Ensemble/tables/sweep_configs.csv.gz (run sweep.py first)
Output, in results/Ensemble/tables/:
  table3_deferred_sets.csv          the table; last row = accuracy on all decided samples
  table3_summary.csv                the counts quoted in the caption
  table3_decisions_per_sample.csv   hash, label and decision of every classifier
                                    (1 malware, 0 benign, -1 deferred)
"""

import numpy as np
import pandas as pd

import common as C

SINGLES = ["Drebin", "MalScan", "RAMDA"]
FUSION = "Test 2 without RAMDA"

if __name__ == "__main__":
    data, sweep = C.Data(), C.load_sweep()
    y = data.y_test

    decisions, configs = {}, {}
    for method in SINGLES + [FUSION]:
        configs[method] = C.pick(sweep[sweep["method"] == method], C.BUDGET)
        decisions[method] = C.decide(configs[method], data)[1]
    deferred = {k: d == -1 for k, d in decisions.items()}

    rows = []
    for name, d in deferred.items():
        row = {"classifier": name, "deferred_n": int(d.sum()), "deferred_pct": 100 * d.sum() / len(y),
               "malware": int(y[d].sum()), "benign": int((1 - y[d]).sum())}
        for other in SINGLES:
            row[f"shared_with_{other}"] = np.nan if other == name else int((d & deferred[other]).sum())
        for other in SINGLES:
            decided = d & ~deferred[other]
            row[f"accuracy_of_{other}"] = np.nan if other == name else float(np.mean(decisions[other][decided] == y[decided]))
        row["selected_config"] = C.describe(configs[name])
        rows.append(row)
    rows.append({"classifier": "All decided samples",
                 **{f"accuracy_of_{o}": float(np.mean(decisions[o][~deferred[o]] == y[~deferred[o]])) for o in SINGLES}})
    table = pd.DataFrame(rows)
    counts = ["deferred_n", "malware", "benign"] + [f"shared_with_{o}" for o in SINGLES]
    table[counts] = table[counts].astype("Int64")

    any_single = deferred["Drebin"] | deferred["MalScan"] | deferred["RAMDA"]
    all_singles = deferred["Drebin"] & deferred["MalScan"] & deferred["RAMDA"]
    both_decide = deferred[FUSION] & ~deferred["Drebin"] & ~deferred["MalScan"]
    contradict = both_decide & (decisions["Drebin"] != decisions["MalScan"])
    summary = pd.DataFrame([
        ("deferred_by_at_least_one_single", int(any_single.sum())),
        ("deferred_by_at_least_one_single_pct", 100 * float(any_single.mean())),
        ("deferred_by_all_three_singles", int(all_singles.sum())),
        ("deferred_by_fusion", int(deferred[FUSION].sum())),
        ("fusion_deferred_where_drebin_and_malscan_both_decide", int(both_decide.sum())),
        ("fusion_deferred_where_drebin_and_malscan_contradict", int(contradict.sum())),
    ], columns=["metric", "value"], dtype=object)

    table.to_csv(C.OUT_DIR / "table3_deferred_sets.csv", index=False)
    summary.to_csv(C.OUT_DIR / "table3_summary.csv", index=False)
    pd.DataFrame({"sha": data.test["sha"], "true_label": y, **decisions}).to_csv(
        C.OUT_DIR / "table3_decisions_per_sample.csv", index=False)

    print(table.drop(columns="selected_config").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(summary.to_string(index=False))
    print(f"\nSaved table3_deferred_sets.csv, table3_summary.csv, table3_decisions_per_sample.csv in {C.OUT_DIR}")
