from generate_features import generate_features, generate_features1
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

from utils import extract_apk_name


def get_args():
    parser = ArgumentParser(description="Generate features from the callgraphs.")
    parser.add_argument("--db_dir", type=str, required=True, help="The directory of the database.")
    parser.add_argument("--feature_dir", type=str, required=True, help="The directory of the features.")
    parser.add_argument("--pool_num", type=int, default=4, help="The number of processes.")
    parser.add_argument("--label", type=int, required=True, help="The label of the samples.")
    args = parser.parse_args()
    return args


def get_apk_paths(labels_csv, base_dir):

    apks_df = pd.read_csv(labels_csv)

    apk_paths = []
    labels = []
    years = []
    for row in apks_df.itertuples():
        sha256 = row.sha256
        labels.append(row.label)
        date = pd.to_datetime(row.first_seen)
        yq = f"{date.year}Q{date.quarter}"
        years.append(yq)
        if (Path(base_dir) / yq / f"{sha256}.apk").exists():
            apk_paths.append(Path(base_dir) / yq / f"{sha256}.apk")

    return apk_paths, labels, years

import logging
import os

from colorlog import ColoredFormatter

def init_logger(level: int, log_path: str, logger_name: str='extractor'):
    """
    Initialize the logger.
    :param level: The level of the logger.
    :param log_path: The path to the log file.
    :param logger_name: The name of the logger.
    :return: The logger.
    """
    logger = logging.getLogger(logger_name)
    formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(name)6s | %(log_color)s%(levelname)4s | %(log_color)s%(message)6s",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'yellow',
            'WARNING':  'green',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
    )

    file_formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(name)6s | %(log_color)s%(levelname)4s | %(log_color)s%(message)6s",
        reset=True,
        no_color=True,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log_file = os.path.abspath(path=log_path)
    open(log_file, 'a').close() # Clear the log file
    output_file_handler = logging.FileHandler(log_file, mode='a')
    output_file_handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    logger.addHandler(output_file_handler)
    logger.setLevel(level)

    return logger

if __name__ == "__main__":

    labels_path = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/data/labels_vt.csv"
    base_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Noha/AndrOps/apks/"

    callgraph_dir="/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/cfg"
    feature_dir="/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/ramda/features"
    
    apk_list, labels, years = get_apk_paths(
        labels_path,
        base_dir,
    )
    print(len(apk_list))

    done = pd.read_csv(os.path.join(feature_dir, 'features.csv'))
    exist_files = done['apk_name'].values.tolist()

    apk_list = [apk for apk in apk_list if extract_apk_name(apk) not in exist_files]

    logger = init_logger("INFO", "logger.log")
    logger.info(f"Extracting features for {len(apk_list)} apps in {base_dir} to {feature_dir}.")
    print(f"Extracting features for {len(apk_list)} apps in {base_dir} to {feature_dir}.")
    generate_features1(
        apk_list,
        callgraph_dir,
        feature_dir,
        pool_num=30,
        logger=logger
    )

    # generate_features(
    #     apk_dir=apk_list,
    #     callgraph_dir=callgraph_dir,
    #     feature_dir=feature_dir,
    #     pool_num=10,
    #     label=1,
    #     logger=logger
    # )