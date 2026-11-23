"""Table 3: which test APKs each classifier defers to stage 2, with statistical tests.

Each classifier uses its dual-threshold configuration of Table 1 (best
validation F1, at most 6.71% of validation deferred). For the samples deferred
by one classifier we count how many the others defer too ("shared with") and
how often the others are right on them when they decide ("accuracy of").

Statistics:
  - overlap of every pair of deferred sets against independence
    (hypergeometric test): expected overlap, p-values for "more than chance"
    and "less than chance", Jaccard index with bootstrap CI;
  - "accuracy of": Wilson 95% CI, and Fisher's exact test against the same
    classifier's accuracy on the samples the row classifier does NOT defer
    (are the samples deferred by one detector also harder for the others?).
Holm's correction is applied within each family of tests.

Input:  results/Ensemble/tables/sweep_configs.csv.gz (run sweep.py first)
Output, in results/Ensemble/tables/:
  table3_deferred_sets.csv          the table; last row = accuracy on all decided samples
  table3_overlap_tests.csv          one row per pair of classifiers: overlap tests
  table3_summary.csv                the counts quoted in the caption
  table3_decisions_per_sample.csv   hash, label and decision of every classifier
                                    (1 malware, 0 benign, -1 deferred)
"""

import itertools

import numpy as np
import pandas as pd

import common as C
import stats as S

SINGLES = ["Drebin", "MalScan", "RAMDA"]
FUSION = "Test 2 without RAMDA"

if __name__ == "__main__":
    data, sweep = C.Data(), C.load_sweep()
    y = data.y_test
    n = len(y)

    decisions, configs = {}, {}
    for method in SINGLES + [FUSION]:
        configs[method] = C.pick(sweep[sweep["method"] == method], C.BUDGET)
        decisions[method] = C.decide(configs[method], data)[1]
    deferred = {k: d == -1 for k, d in decisions.items()}

    # ---- main table, with CI and test on "accuracy of"
    rows = []
    for name, d in deferred.items():
        row = {"classifier": name, "deferred_n": int(d.sum()), "deferred_pct": 100 * d.sum() / n,
               "malware": int(y[d].sum()), "benign": int((1 - y[d]).sum())}
        for other in SINGLES:
            row[f"shared_with_{other}"] = np.nan if other == name else int((d & deferred[other]).sum())
        for other in SINGLES:
            if other == name:
                for suffix in ["", "_ci_low", "_ci_high", "_on_rest", "_p_fisher"]:
                    row[f"accuracy_of_{other}{suffix}"] = np.nan
                continue
            on_def = d & ~deferred[other]          # deferred by the row, decided by `other`
            on_rest = ~d & ~deferred[other]        # decided by both
            k1, n1 = int(np.sum(decisions[other][on_def] == y[on_def])), int(on_def.sum())
            k2, n2 = int(np.sum(decisions[other][on_rest] == y[on_rest])), int(on_rest.sum())
            row[f"accuracy_of_{other}"] = k1 / n1 if n1 else np.nan
            row[f"accuracy_of_{other}_ci_low"], row[f"accuracy_of_{other}_ci_high"] = S.wilson(k1, n1)
            row[f"accuracy_of_{other}_on_rest"] = k2 / n2 if n2 else np.nan
            row[f"accuracy_of_{other}_p_fisher"] = S.fisher(k1, n1, k2, n2)
        row["selected_config"] = C.describe(configs[name])
        rows.append(row)
    rows.append({"classifier": "All decided samples",
                 **{f"accuracy_of_{o}": float(np.mean(decisions[o][~deferred[o]] == y[~deferred[o]])) for o in SINGLES}})
    table = pd.DataFrame(rows)
    pcols = [c for c in table.columns if c.endswith("_p_fisher")]
    flat = table[pcols].to_numpy().ravel()
    table[[c + "_holm" for c in pcols]] = S.holm(flat).reshape(len(table), len(pcols))
    counts = ["deferred_n", "malware", "benign"] + [f"shared_with_{o}" for o in SINGLES]
    table[counts] = table[counts].astype("Int64")

    # ---- overlap of every pair of deferred sets
    pairs = []
    for a, b in itertools.combinations(SINGLES + [FUSION], 2):
        k = int((deferred[a] & deferred[b]).sum())
        t = S.overlap_test(n, int(deferred[a].sum()), int(deferred[b].sum()), k)
        lo, hi = S.bootstrap_jaccard(deferred[a], deferred[b])
        pairs.append({"a": a, "b": b, "family": "singles" if FUSION not in (a, b) else "fusion",
                      "n_a": int(deferred[a].sum()), "n_b": int(deferred[b].sum()), "overlap": k,
                      "expected_if_independent": t["expected"], "ratio_to_expected": k / t["expected"],
                      "p_more_than_chance": t["p_more"], "p_less_than_chance": t["p_less"],
                      "jaccard": t["jaccard"], "jaccard_ci_low": lo, "jaccard_ci_high": hi})
    overlap = pd.DataFrame(pairs)
    for fam in overlap["family"].unique():
        m = overlap["family"] == fam
        for col in ["p_more_than_chance", "p_less_than_chance"]:
            overlap.loc[m, f"{col}_holm"] = S.holm(overlap.loc[m, col].to_numpy())

    # ---- caption counts
    any_single = deferred["Drebin"] | deferred["MalScan"] | deferred["RAMDA"]
    all_singles = deferred["Drebin"] & deferred["MalScan"] & deferred["RAMDA"]
    both_decide = deferred[FUSION] & ~deferred["Drebin"] & ~deferred["MalScan"]
    contradict = both_decide & (decisions["Drebin"] != decisions["MalScan"])
    summary = pd.DataFrame([
        ("deferred_by_at_least_one_single", int(any_single.sum())),
        ("deferred_by_at_least_one_single_pct", 100 * float(any_single.mean())),
        ("deferred_by_all_three_singles", int(all_singles.sum())),
        ("deferred_by_all_three_expected_if_independent",
         float(np.prod([deferred[s].sum() for s in SINGLES]) / n ** 2)),
        ("deferred_by_fusion", int(deferred[FUSION].sum())),
        ("fusion_deferred_where_drebin_and_malscan_both_decide", int(both_decide.sum())),
        ("fusion_deferred_where_drebin_and_malscan_contradict", int(contradict.sum())),
    ], columns=["metric", "value"], dtype=object)

    table.to_csv(C.OUT_DIR / "table3_deferred_sets.csv", index=False)
    overlap.to_csv(C.OUT_DIR / "table3_overlap_tests.csv", index=False)
    summary.to_csv(C.OUT_DIR / "table3_summary.csv", index=False)
    pd.DataFrame({"sha": data.test["sha"], "true_label": y, **decisions}).to_csv(
        C.OUT_DIR / "table3_decisions_per_sample.csv", index=False)

    main = ["classifier", "deferred_n", "deferred_pct", "malware", "benign"] + \
           [f"shared_with_{o}" for o in SINGLES] + [f"accuracy_of_{o}" for o in SINGLES]
    print(table[main].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print("\nOverlap of deferred sets vs independence:")
    cols = ["a", "b", "overlap", "expected_if_independent", "ratio_to_expected",
            "p_more_than_chance_holm", "jaccard", "jaccard_ci_low", "jaccard_ci_high"]
    print(overlap[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(summary.to_string(index=False))
    print(f"\nSaved table3_deferred_sets.csv, table3_overlap_tests.csv, table3_summary.csv, "
          f"table3_decisions_per_sample.csv in {C.OUT_DIR}")
