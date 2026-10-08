import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from external_libraries import RAMDADataset


def load_drebin_features(feature_path, labels, tr_yq, val_yq, ts_yq):
    feature_path = Path(feature_path)

    def load_split(year_quarters):
        split = labels[labels["year_quarter"].isin(year_quarters)]

        apk_sha = split["sha256"].to_list()
        y = split["label"].to_list()
        yq = split["year_quarter"].to_list()

        X = []

        for sha, quarter in zip(apk_sha, yq):
            feature_file = (
                feature_path
                / quarter
                / "DREBINFeatureExtractor"
                / f"{sha.upper()}.json"
            )

            with feature_file.open("r") as fp:
                js = json.load(fp)

            features = [f"{k}::{v}" for k, values in js.items() for v in values]

            X.append(features)

        return X, y, yq, apk_sha

    return (
        load_split(tr_yq),
        load_split(val_yq),
        load_split(ts_yq),
    )


def load_malscan_features(feature_path, typ, labels, tr_yq, val_yq, ts_yq):

    df = pd.read_csv(Path(feature_path) / f"{typ}_features.csv")

    def load_split(year_quarters):
        split = labels[labels["year_quarter"].isin(year_quarters)]

        apk_sha = split["sha256"].to_list()
        y = split["label"].to_list()
        yq = split["year_quarter"].to_list()

        X = []
        y_filtered = []
        yq_filtered = []
        sha_filtered = []

        for sha, label, quarter in zip(apk_sha, y, yq):
            row = df[df["SHA256"] == sha]

            if not row.empty:
                features = row.iloc[0, 2:].values.tolist()
                X.append(features)
                y_filtered.append(label)
                yq_filtered.append(quarter)
                sha_filtered.append(sha)

        return X, y_filtered, yq_filtered, sha_filtered

    return (
        load_split(tr_yq),
        load_split(val_yq),
        load_split(ts_yq),
    )


def load_ramda_features(feature_path, labels, tr_yq, val_yq, ts_yq):

    feature_dirs = [feature_path]

    def get_samples(year_quarters):
        split = labels[
            labels["year_quarter"].isin(year_quarters)
        ][["sha256", "label", "year_quarter"]].reset_index(drop=True)

        return split

    train_samples = get_samples(tr_yq)
    validation_samples = get_samples(val_yq)
    test_samples = get_samples(ts_yq)

    train_val_samples = pd.concat(
        [train_samples, validation_samples],
        ignore_index=True,
    )

    train_dataset = RAMDADataset(
        "train",
        feature_dirs,
        (train_samples, test_samples),
        shuffle=False,
        sampling_mode="oversampling",
    )

    validation_dataset = RAMDADataset(
        "test",
        feature_dirs,
        (train_samples, validation_samples),
        shuffle=False,
        sampling_mode="none",
    )

    test_dataset = RAMDADataset(
        "test",
        feature_dirs,
        (train_samples, test_samples),
        shuffle=False,
        sampling_mode="none",
    )

    train_val_dataset = RAMDADataset(
        "train",
        feature_dirs,
        (train_val_samples, test_samples),
        shuffle=False,
        sampling_mode="oversampling",
    )

    val_info = (
        validation_samples["label"].to_list(),
        validation_samples["year_quarter"].to_list(),
        validation_samples["sha256"].to_list(),
    )

    test_info = (
        test_samples["label"].to_list(),
        test_samples["year_quarter"].to_list(),
        test_samples["sha256"].to_list(),
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_val_dataset,
        val_info,
        test_info
    )

def scores_to_numpy(scores):
    return np.concatenate(
        [
            score.detach().cpu().numpy()
            if torch.is_tensor(score)
            else np.asarray(score)
            for score in scores
        ]
    )