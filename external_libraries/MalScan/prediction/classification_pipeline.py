import csv
from itertools import islice
import pandas as pd
from pathlib import Path
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import time
import numpy as np


DUAL_THR_MAL_F1_FLOOR = 0.85
DUAL_THR_BEN_F1_FLOOR = 0.85
DUAL_THR_MIN_GAP = 0.10


def split_train_val_test(samples_df: pd.DataFrame, samples_hash: list):

    samples_df = samples_df[samples_df["sha256"].isin(samples_hash)]
    samples_df["year_quarter"] = (
        pd.to_datetime(samples_df["first_seen"]).dt.to_period("Q").astype(str)
    )

    train_samples = samples_df[samples_df["year_quarter"].between("2024Q1", "2025Q2")]
    validation_samples = samples_df[samples_df["year_quarter"] == "2025Q3"]
    test_samples = samples_df[samples_df["year_quarter"] == "2025Q4"]

    return train_samples, validation_samples, test_samples


def load_features(df, samples):
    vectors = []
    labels = []
    years = []
    sha256s = []
    for sha256, label, year in zip(*samples):
        row = df[df["SHA256"] == sha256]
        if not row.empty:
            vector = row.iloc[0, 2:].values.tolist()
            vectors.append(vector)
            labels.append(label)
            years.append(year)
            sha256s.append(sha256)

    return vectors, labels, years, sha256s


def _class_metrics_sweep(y, scores, thresholds, positive_label: int):
    n_pos = int(np.sum(y == positive_label))
    precision_list, recall_list, f1_list = [], [], []

    for t in thresholds:
        if positive_label == 1:
            tp = int(np.sum((scores >= t) & (y == 1)))
            fp = int(np.sum((scores >= t) & (y == 0)))
        else:
            tp = int(np.sum((scores <= t) & (y == 0)))
            fp = int(np.sum((scores <= t) & (y == 1)))

        r = tp / n_pos if n_pos > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)

    return np.array(precision_list), np.array(recall_list), np.array(f1_list)


def choose_dual_thresholds_on_val(y_val, val_scores):
    """Select two thresholds from validation scores."""
    val_scores = np.asarray(val_scores).ravel()
    y_val = np.asarray(y_val)
    n_total = len(y_val)

    thresholds = np.sort(np.unique(val_scores))
    mal_prec, mal_rec, mal_f1 = _class_metrics_sweep(y_val, val_scores, thresholds, 1)
    ben_prec, ben_rec, ben_f1 = _class_metrics_sweep(y_val, val_scores, thresholds, 0)

    mal_ok = np.where(mal_f1 >= DUAL_THR_MAL_F1_FLOOR)[0]
    if len(mal_ok) > 0:
        mal_best_idx = int(mal_ok[-1])
        mal_status = "floor_met"
    else:
        mal_best_idx = int(np.argmax(mal_f1))
        mal_status = "floor_unmet_argmax_fallback"
    thr_malware = float(thresholds[mal_best_idx])

    valid_mask = thresholds <= (thr_malware - DUAL_THR_MIN_GAP)
    ben_ok = np.where((ben_f1 >= DUAL_THR_BEN_F1_FLOOR) & valid_mask)[0]
    if len(ben_ok) > 0:
        ben_best_idx = int(ben_ok[0])
        ben_status = "floor_met"
    else:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            ben_best_idx = int(valid_indices[np.argmax(ben_f1[valid_indices])])
        else:
            ben_best_idx = int(np.argmax(ben_f1))
        ben_status = "floor_unmet_argmax_fallback"
    thr_benign = float(thresholds[ben_best_idx])

    deferred_mask = (val_scores > thr_benign) & (val_scores < thr_malware)
    n_deferred = int(np.sum(deferred_mask))
    deferral_rate = n_deferred / n_total if n_total else 0.0

    mal_below_rate = np.array([np.sum(val_scores < t) / n_total for t in thresholds])
    ben_above_rate = np.array([np.sum(val_scores > t) / n_total for t in thresholds])

    print("\n=== Dual Threshold Selection on VAL ===")
    print(
        f"thr_malware = {thr_malware:.6f}  [{mal_status}]  "
        f"P={mal_prec[mal_best_idx]:.3f}  R={mal_rec[mal_best_idx]:.3f}  F1={mal_f1[mal_best_idx]:.3f}"
    )
    print(
        f"thr_benign  = {thr_benign:.6f}  [{ben_status}]  "
        f"P={ben_prec[ben_best_idx]:.3f}  R={ben_rec[ben_best_idx]:.3f}  F1={ben_f1[ben_best_idx]:.3f}"
    )
    print(
        f"Deferral on VAL: {n_deferred}/{n_total} = {deferral_rate:.3f}  "
        f"(gap: {thr_malware - thr_benign:.6f}, min_gap={DUAL_THR_MIN_GAP:.6f})"
    )

    return thr_benign, thr_malware


def stage2(sample, log_path: str = "stage2_samples.txt"):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(sample + "\n")
    return None


def xgboost(X_train, y_train, X_val, y_val, X_test, y_test, sha_test, type, threshold):

    clf = XGBClassifier(max_depth=64, random_state=0)
    start_time = time.time()
    clf.fit(X_train, y_train)
    print(f"Training time for {type}: {time.time() - start_time:.2f} seconds")
    if threshold == 0.5:
        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        report_val = classification_report(y_val, y_pred_val)
        clf.fit(X_train + X_val, y_train + y_val)
        report_test = classification_report(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)
    elif threshold == None:
        y_scores_val = clf.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0, 1, num=1001)
        best_threshold = 0.5
        best_f1 = -1

        for t in thresholds:
            y_pred_val_temp = (y_scores_val >= t).astype(int)
            f1 = f1_score(y_val, y_pred_val_temp)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        threshold = best_threshold
        print(f"Best threshold: {threshold:.3f} (Validation F1 = {best_f1:.4f})")

        y_scores = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (y_scores >= threshold).astype(int)
        y_scores_val = clf.predict_proba(X_val)[:, 1]
        y_pred_val = (y_scores_val >= threshold).astype(int)
        report_val = classification_report(y_val, y_pred_val)
        cm_val = confusion_matrix(y_val, y_pred_val)
        clf.fit(X_train + X_val, y_train + y_val)
        y_scores_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (y_scores_test >= threshold).astype(int)
        report_test = classification_report(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)
    elif threshold == "dual":
        y_scores_val = clf.predict_proba(X_val)[:, 1]
        thr_benign, thr_malware = choose_dual_thresholds_on_val(y_val, y_scores_val)

        y_pred_val = np.where(
            y_scores_val >= thr_malware,
            1,
            np.where(y_scores_val <= thr_benign, 0, -1),
        )
        decided = y_pred_val != -1
        # if decided.sum() > 0:
        #     eval_block(
        #         "VAL (stage-1 decided only)",
        #         np.array(y_val)[decided],
        #         y_val_pred[decided],
        #         y_val_pred[decided],
        #     )

        report_val = classification_report(
            np.array(y_val)[decided], y_pred_val[decided]
        )
        cm_val = confusion_matrix(y_val, y_pred_val)

        X_train = X_train + X_val
        y_train = y_train + y_val
        clf.fit(X_train, y_train)

        y_test_decided = []
        y_pred_decided = []
        scores_decided = []
        deferred_indices = []
        stage2_count = 0
        y_scores_test = clf.predict_proba(X_test)[:, 1]
        for i, score in enumerate(y_scores_test):
            if score >= thr_malware:
                y_test_decided.append(y_test[i])
                y_pred_decided.append(1)
                scores_decided.append(score)
            elif score <= thr_benign:
                y_test_decided.append(y_test[i])
                y_pred_decided.append(0)
                scores_decided.append(score)
            else:
                stage2(
                    sha_test[i],
                    Path(__file__).parent.parent.parent.parent
                    / "results"
                    / "MalScan"
                    / f"stage2_samples_{type}.txt",
                )
                deferred_indices.append(i)
                stage2_count += 1

        report_test = classification_report(y_test_decided, y_pred_decided)
        cm_test = confusion_matrix(y_test_decided, y_pred_decided)

    # save report to file
    out_dir = Path(__file__).parent.parent.parent.parent / "results" / "MalScan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"malscan_{type}_xgboost_classification_report_{threshold}.txt"
    with open(out_file, "w") as f:
        f.write("VALIDATION REPORT\n")
        f.write(report_val)
        f.write("\nTEST REPORT\n")
        f.write(report_test)

    out_file_cm = out_dir / f"malscan_{type}_xgboost_confusion_matrix_{threshold}.csv"
    pd.DataFrame(
        cm_test,
        index=["goodware", "malware"],
        columns=["pred_goodware", "pred_malware"],
    ).to_csv(out_file_cm)
    print(f"Saved confusion matrix to {out_file_cm}")


if __name__ == "__main__":
    samples = Path(__file__).parent.parent.parent.parent / "samples_hash.txt"
    empty = Path(__file__).parent.parent.parent.parent / "empty_hash.txt"
    samples_hash = set([line.strip() for line in islice(open(samples), 1, None)])
    empty_hash = set([line.strip().lower() for line in islice(open(empty), 1, None)])
    samples_hash = samples_hash - empty_hash
    settings_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/settings"
    samples_df = pd.read_csv(Path(settings_dir) / "labels_vt.csv")

    train_samples, validation_samples, test_samples = split_train_val_test(
        samples_df, samples_hash
    )

    train_samples = (
        train_samples["sha256"].values.tolist(),
        train_samples["label"].values.tolist(),
        train_samples["year_quarter"].values.tolist(),
    )

    val_samples = (
        validation_samples["sha256"].values.tolist(),
        validation_samples["label"].values.tolist(),
        validation_samples["year_quarter"].values.tolist(),
    )

    test_samples = (
        test_samples["sha256"].values.tolist(),
        test_samples["label"].values.tolist(),
        test_samples["year_quarter"].values.tolist(),
    )

    print(
        f"Found {len(train_samples[0])} training samples, {len(val_samples[0])} validation samples, and {len(test_samples[0])} testing samples"
    )
    types = ["pagerank"]
    feature_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/features"

    for typ in types:
        df = pd.read_csv(Path(feature_dir) / f"{typ}_features.csv")
        X_train, y_train, year_train, sha_train = load_features(df, train_samples)
        print(f"Loaded {len(X_train)} training features")
        X_val, y_val, year_val, sha_val = load_features(df, val_samples)
        print(f"Loaded {len(X_val)} validation features")
        X_test, y_test, year_test, sha_test = load_features(df, test_samples)
        print(f"Loaded {len(X_test)} testing features")
        # default threshold of 0.5
        xgboost(
            X_train, y_train, X_val, y_val, X_test, y_test, sha_test, typ, threshold=0.5
        )
        # threshold selected based on validation set f1 score
        xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            sha_test,
            typ,
            threshold=None,
        )
        # two thresholds selected based on validation set f1 score for both classes
        xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            sha_test,
            typ,
            threshold="dual",
        )
