import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from itertools import islice


def get_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--label_path", type=str, required=True, help="Path to the labels csv."
    )
    parser.add_argument(
        "--feature_path_drebin",
        type=str,
        required=True,
        help="Path to the features directory of drebin.",
    )
    parser.add_argument(
        "--feature_path_ramda",
        type=str,
        required=True,
        help="Path to the features directory of ramda.",
    )
    parser.add_argument(
        "--feature_path_malscan",
        type=str,
        required=True,
        help="Path to the features directory of malscan.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    label_path = args.label_path
    feature_path_drebin = args.feature_path_drebin
    feature_path_ramda = args.feature_path_ramda
    feature_path_malscan = args.feature_path_malscan
    output_path = Path(__file__).parent / "resources" / "common_samples.csv"
    empty_file = Path(__file__).parent / "resources" / "empty_hash.txt"
    empty_hash = {line.strip().lower() for line in islice(open(empty_file), 1, None)}

    labels = pd.read_csv(label_path)

    apk_sha = labels["sha256"].tolist()
    print(f"Total number of samples: {len(apk_sha)}")

    ramda_features = pd.read_csv(feature_path_ramda)
    malscan_features = pd.read_csv(feature_path_malscan)

    ramda_sha = ramda_features["apk_name"].tolist()
    malscan_sha = malscan_features["SHA256"].tolist()
    drebin_sha = Path(feature_path_drebin).glob(
        "202[4-5]Q*/DREBINFeatureExtractor/*.json"
    )
    drebin_sha = {path.stem.lower() for path in drebin_sha} - empty_hash

    print(f"Number of samples in Drebin: {len(drebin_sha)}")
    print(f"Number of samples in RAMDA: {len(ramda_sha)}")
    print(f"Number of samples in MalScan: {len(malscan_sha)}")

    common_sha = set(apk_sha) & set(drebin_sha) & set(ramda_sha) & set(malscan_sha)

    print(f"Number of common samples: {len(common_sha)}")

    # Filter original labels
    filtered_labels = labels[labels["sha256"].isin(common_sha)].copy()

    # Write filtered labels
    filtered_labels.to_csv(output_path, index=False)

    print(f"Filtered labels written to: {output_path}")
