import torch
from argparse import ArgumentParser
import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from external_libraries import DREBIN


def get_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--label_path", type=str, required=True, help="Path to the labels csv."
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        required=True,
        help="Path to the features directory.",
    )
    parser.add_argument(
        "--detectpr_name",
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


if __name__ == "__main__":
    RANDOM_SEED = 42

    args = get_args()
    feature_path = args.feature_path
    detector_name = args.detectpr_name
    output_path = args.output_path
    label_path = args.label_path

    labels = pd.read_csv(label_path)

    if detector_name == "drebin":
        detector = DREBIN(
            C=0.03,
            max_iter=10000,
            random_state=RANDOM_SEED,
            dual="auto",
            min_df=1,
            max_df=0.9,
            max_features="None",
        )
        detector.fit(X_train, y_train)
        _, val_scores = detector.predict(X_val)

        detector.fit(X_train + X_val, y_train + y_val)
        _, test_scores = detector.predict(X_test)
