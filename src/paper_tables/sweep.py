"""Enumerate the configurations the paper tables choose from.

For every method in the tables, try all the ways of turning its scores into
decisions (F1 floors and gap of the dual thresholds, deferral budget, ...) and
store validation and test metrics. The table scripts then pick, for each
method, the configuration with the best *validation* F1 within a deferral
budget: test metrics are kept only to report them, never to choose.

Methods:
  Drebin, MalScan, RAMDA   one detector alone (calibrated score)
  Test 1                   majority vote of Drebin, MalScan and RAMDA
  Test 3                   votes weighted by validation F1
  Test 2                   F1-weighted average of the three calibrated scores
  Test 2 without RAMDA     weighted average (or stacking) of Drebin and MalScan

Output: results/Ensemble/tables/sweep_configs.csv.gz (one row per configuration)
Run:    python3 sweep.py
"""

import time

import numpy as np
import pandas as pd

import common as C

METRICS = ["deferred_pct", "accuracy", "mal_precision", "mal_recall", "mal_f1", "ben_f1"]


def configurations(data):
    def with_floors(method, score, thresholds):
        return [dict(method=method, score=score, thresholds=thresholds, floor_kind=kind,
                     floor_malware=f_mal, floor_benign=f_ben, gap=gap, budget=np.nan)
                for kind, f_mal, f_ben, gap in C.floor_settings()]

    def with_bands(method, score, budgets=C.BAND_BUDGETS):
        return [dict(method=method, score=score, thresholds="best band", floor_kind="-",
                     floor_malware=np.nan, floor_benign=np.nan, gap=np.nan, budget=b)
                for b in budgets]

    configs = []
    for method, score in [("Drebin", "D"), ("MalScan", "M"), ("RAMDA", "R")]:
        configs += with_floors(method, score, "paper floors") + with_bands(method, score)

    # every detector votes with its own dual thresholds (same floors for all),
    # or with its best single threshold (no abstention)
    for method in ["Test 1", "Test 3"]:
        configs += with_floors(method, "-", "dual thresholds")
        configs.append(dict(method=method, score="-", thresholds="single threshold", floor_kind="-",
                            floor_malware=np.nan, floor_benign=np.nan, gap=np.nan, budget=np.nan))

    # Test 2 as specified: thresholds = weighted average of the detectors' thresholds;
    # budget 0 gives its single-threshold version
    test2 = "D+M+R | F1 | mean prob"
    configs += with_floors("Test 2", test2, "averaged detector thresholds") + with_bands("Test 2", test2, [0.0])

    for score, s in data.scores.items():
        if score.startswith("D+M |"):
            configs += with_floors("Test 2 without RAMDA", score, "paper floors")
            if s["weights"] is not None:
                configs += with_floors("Test 2 without RAMDA", score, "averaged detector thresholds")
            configs += with_bands("Test 2 without RAMDA", score)
    return configs


if __name__ == "__main__":
    t0 = time.time()
    data = C.Data()
    rows = []
    for config in configurations(data):
        pred_val, pred_test = C.decide(config, data)
        m_val, m_test = C.metrics(data.y_val, pred_val), C.metrics(data.y_test, pred_test)
        rows.append({**config, **{f"val_{k}": m_val[k] for k in METRICS}, **{f"test_{k}": m_test[k] for k in METRICS}})

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(C.SWEEP_FILE, index=False)
    print(sweep.groupby("method", sort=False).size().to_string())
    print(f"{len(sweep)} configurations saved to {C.SWEEP_FILE} ({time.time() - t0:.0f}s)")
