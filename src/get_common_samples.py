import pandas as pd
from argparse import ArgumentParser

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
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output scores and metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    label_path = args.label_path
    feature_path = args.feature_path
    output_path = args.output_path

    labels = pd.read_csv(label_path)

    apk_sha = labels["sha256"].tolist()
    print(f"Total number of samples: {len(apk_sha)}")