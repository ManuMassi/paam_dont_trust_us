from pathlib import Path
import json
import pandas as pd


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
