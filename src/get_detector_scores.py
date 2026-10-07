import torch
from argparse import ArgumentParser
import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost.sklearn import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))

from external_libraries import DREBIN
from utils import load_drebin_features, load_malscan_features


def get_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--feature_path",
        type=str,
        required=True,
        help="Path to the features directory depending on the extractor.",
    )
    parser.add_argument(
        "--detector_name",
        type=str,
        required=True,
        choices=["Drebin", "RAMDA", "MalScan"],
        help="Detector type (Drebin, RAMDA, MalScan).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output scores and metrics.",
    )
    return parser.parse_args()


def save_scores(path, sha, scores, labels):
    pd.DataFrame(
        {
            "sha": sha,
            "score": scores,
            "true_label": labels,
        }
    ).to_csv(path, index=False)


if __name__ == "__main__":
    RANDOM_SEED = 42
    TR_YQ = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2"]
    VAL_YQ = ["2025Q3"]
    TS_YQ = ["2025Q4"]

    args = get_args()
    feature_path = args.feature_path
    detector_name = args.detector_name
    output_path = args.output_path
    label_path = Path(__file__).parent / "resources" / "common_samples.csv"
    labels = pd.read_csv(label_path)
    labels["year_quarter"] = (
        pd.to_datetime(labels["first_seen"]).dt.to_period("Q").astype(str)
    )

    if detector_name == "Drebin":
        train_data, val_data, test_data = load_drebin_features(
            feature_path,
            labels,
            TR_YQ,
            VAL_YQ,
            TS_YQ,
        )

        X_train, y_train, yq_train, sha_train = train_data
        X_val, y_val, yq_val, sha_val = val_data
        X_test, y_test, yq_test, sha_test = test_data
        print(
            "Loaded Drebin features and labels for training, validation, and testing."
        )
        detector = DREBIN()
        detector.fit(X_train, y_train)
        _, val_scores = detector.predict(X_val)

        detector.fit(X_train + X_val, y_train + y_val)
        _, test_scores = detector.predict(X_test)
    elif detector_name == "MalScan":
        typ = "pagerank"
        train_data, val_data, test_data = load_malscan_features(
            feature_path,
            typ,
            labels,
            TR_YQ,
            VAL_YQ,
            TS_YQ,
        )

        X_train, y_train, yq_train, sha_train = train_data
        X_val, y_val, yq_val, sha_val = val_data
        X_test, y_test, yq_test, sha_test = test_data
        print(
            "Loaded MalScan features and labels for training, validation, and testing."
        )

        detector = XGBClassifier(max_depth=64, random_state=0)
        detector.fit(X_train, y_train)
        val_scores = detector.predict_proba(X_val)[:, 1]

        detector.fit(X_train + X_val, y_train + y_val)
        test_scores = detector.predict_proba(X_test)[:, 1]
    elif detector_name == "RAMDA":
        pass

    results_path = Path(__file__).parent.parent / "results" / detector_name
    results_path.mkdir(parents=True, exist_ok=True)
    save_scores(
        results_path / "val_scores.csv",
        sha_val,
        val_scores,
        y_val,
    )

    save_scores(
        results_path / "test_scores.csv",
        sha_test,
        test_scores,
        y_test,
    )
