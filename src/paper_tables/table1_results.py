"""Table 1: detection quality at equal deferral budget, with statistical tests.

Two settings:
  single threshold  every sample is decided (no deferral);
  dual threshold    samples between the two thresholds are deferred to stage 2,
                    at most 6.71% of validation (what Drebin defers with the
                    paper's procedure).
For each method and setting we take from the sweep the configuration with the
best validation F1 within that deferral, and measure it on test. Metrics are on
the decided samples.

Statistics (test set, every method compared with Drebin):
  - 95% CI of every metric (paired bootstrap, B = 10,000);
  - dF1 = F1(method) - F1(Drebin): CI, bootstrap p-value, Holm-adjusted p;
  - non-inferiority of F1 (margin NI_MARGIN);
  - exact McNemar on correctness, on the samples both decide;
  - dDeferred (percentage points) and exact McNemar on the deferral indicator:
    the budget is equal on validation, but test deferral can differ.
Holm's correction is applied separately per setting and per test, over the
methods compared with Drebin.

Tests 1 and 3 take the same decisions (with three F1 weights close to 1/3 a
class needs at least 2 of 3 votes either way): the script checks it and keeps
one row, "Tests 1 and 3 (voting)".

Input:  results/Ensemble/tables/sweep_configs.csv.gz (run sweep.py first)
Output, in results/Ensemble/tables/:
  table1_results.csv                 the table with all statistics
  table1_decisions_per_sample.csv    test decisions of every row (1, 0, -1)
"""

import numpy as np
import pandas as pd

import common as C
import stats as S

OUT_FILE = C.OUT_DIR / "table1_results.csv"
DECISIONS_FILE = C.OUT_DIR / "table1_decisions_per_sample.csv"
SETTINGS = [("Single threshold", 0.0), ("Dual threshold", C.BUDGET)]
REF = "Drebin"
VOTING = "Tests 1 and 3 (voting)"
METHODS = [  # (method in the sweep, name in the table)
    ("Drebin", "Drebin"),
    ("MalScan", "MalScan"),
    ("RAMDA", "RAMDA"),
    ("Test 1", VOTING),
    ("Test 2", "Test 2 (weighted avg.)"),
    ("Test 2 without RAMDA", "Test 2 without RAMDA"),
]
NI_MARGIN = 0.01  # non-inferiority margin on F1


if __name__ == "__main__":
    data, sweep = C.Data(), C.load_sweep()
    y = data.y_test
    rows, per_sample = [], {"sha": data.test["sha"], "true_label": y}

    for setting, budget in SETTINGS:
        decisions, configs = {}, {}
        for method, name in METHODS:
            configs[name] = C.pick(sweep[sweep["method"] == method], budget)
            decisions[name] = C.decide(configs[name], data)[1]  # test decisions
        t3 = C.decide(C.pick(sweep[sweep["method"] == "Test 3"], budget), data)[1]
        assert np.array_equal(t3, decisions[VOTING]), f"{setting}: Tests 1 and 3 differ"

        boot = S.bootstrap(y, decisions, ref=REF)
        setting_rows = []
        for _, name in METHODS:
            d = decisions[name]
            m = C.metrics(y, d)
            b = boot[name]
            row = {"setting": setting, "method": name,
                   **{k: m[k] for k in ["accuracy", "mal_precision", "mal_recall", "mal_f1", "ben_f1",
                                        "deferred_pct", "n", "decided", "deferred"]}}
            for k in S.METRICS:
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = b[k]
            if name != REF:
                dF1, lo, hi, p = b["d_mal_f1"]
                row.update(dF1=dF1, dF1_ci_low=lo, dF1_ci_high=hi, p_dF1=p,
                           f1_noninferior=S.noninferior(lo, NI_MARGIN))
                dD, lo, hi, _ = b["d_deferred_pct"]
                row.update(dDeferred_pp=dD, dDeferred_ci_low=lo, dDeferred_ci_high=hi)
                n_both, only_m, only_ref, p = S.mcnemar_correct(y, d, decisions[REF])
                row.update(mcnemar_n_both=n_both, only_method_right=only_m, only_drebin_right=only_ref,
                           p_mcnemar_acc=p)
                if setting == "Dual threshold":
                    only_ref_def, only_m_def, p = S.mcnemar_deferral(decisions[REF], d)
                    row.update(only_drebin_defers=only_ref_def, only_method_defers=only_m_def,
                               p_mcnemar_deferral=p)
            row["selected_config"] = C.describe(configs[name])
            setting_rows.append(row)
            per_sample[f"{setting} | {name}"] = d

        # Holm per setting and per test, over the methods compared with Drebin
        df = pd.DataFrame(setting_rows)
        for col in ["p_dF1", "p_mcnemar_acc", "p_mcnemar_deferral"]:
            if col in df:
                df[f"{col}_holm"] = S.holm(df[col].to_numpy())
        df["significant_F1"] = df["p_dF1_holm"] < S.ALPHA
        rows.append(df)

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT_FILE, index=False)
    pd.DataFrame(per_sample).to_csv(DECISIONS_FILE, index=False)

    # compact view
    view = pd.DataFrame({
        "setting": table["setting"], "method": table["method"],
        "F1 [95% CI]": [S.fmt_ci(r.mal_f1, r.mal_f1_ci_low, r.mal_f1_ci_high) for r in table.itertuples()],
        "Def% [95% CI]": [S.fmt_ci(r.deferred_pct, r.deferred_pct_ci_low, r.deferred_pct_ci_high, 1)
                          for r in table.itertuples()],
        "dF1": table["dF1"], "p(dF1) Holm": table["p_dF1_holm"],
        "p McNemar acc Holm": table["p_mcnemar_acc_holm"],
        "dDef pp": table["dDeferred_pp"],
        "p McNemar def Holm": table.get("p_mcnemar_deferral_holm"),
        "F1 non-inf.": table["f1_noninferior"],
    })
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print(f"\nSaved {OUT_FILE}\nSaved {DECISIONS_FILE}")
