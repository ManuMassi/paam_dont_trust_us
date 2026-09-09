import csv
import os
import time

from functools import partial
from pathlib import Path
from logging import Logger
from multiprocessing import Pool as ThreadPool
from androguard.misc import AnalyzeAPK
from androguard.core.apk import APK

from feature_extractor import extract_features, extract_features1
from utils import ensure_dir, extract_apk_name, ensure_file
import io
import contextlib
from contextlib import contextmanager


@contextmanager
def _suppress_output():
    """Context manager to suppress stdout and stderr at OS-level and Python-level."""
    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def get_referred_sensitive_apis_dict():
    """
    Get the referred sensitive APIs.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_file = os.path.join(current_dir, "list_total_apis.txt")

    referred_sensitive_apis_dict = {}
    with open(api_file, "r") as f:
        lines = f.read().splitlines()

    for idx, api in enumerate(lines):
        referred_sensitive_apis_dict[api] = idx

    return referred_sensitive_apis_dict


def get_referred_actions_dict():
    """
    Get the referred actions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    action_file = os.path.join(current_dir, "list_total_actions.txt")

    referred_actions_dict = {}
    with open(action_file, "r") as f:
        lines = f.read().splitlines()

    for idx, action in enumerate(lines):
        referred_actions_dict[action] = idx

    return referred_actions_dict


def get_referred_permissions_dict():
    """
    Get the referred permissions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    permission_file = os.path.join(current_dir, "list_total_permissions.txt")

    referred_permissions_dict = {}
    with open(permission_file, "r") as f:
        lines = f.read().splitlines()

    for idx, permission in enumerate(lines):
        referred_permissions_dict[permission] = idx

    return referred_permissions_dict


def get_intent_filters_dict(apk: APK):
    """
    Get the intent filters.
    """
    items = {
        "activity": apk.get_activities(),
        "service": apk.get_services(),
        "receiver": apk.get_receivers(),
        "provider": apk.get_providers(),
    }
    intent_filters_dict = {}

    for item_type, item_list in items.items():
        for item in item_list:
            intent_filters = apk.get_intent_filters(item_type, item)
            intent_filters_dict[item] = intent_filters

    return intent_filters_dict

def obtain_app_files(apk_list: list[Path], callgraph_dir: str):
    """
    Obtain app files including callgraph, intents, and permissions.
    """
    app_files = []

    for apk_path in apk_list:
        tic = time.time()
        with _suppress_output():
            a, _, _ = AnalyzeAPK(apk_path)
        app_name = extract_apk_name(apk_path)
        callgraph_file = os.path.join(callgraph_dir, app_name + ".gml.gz")
        intents = get_intent_filters_dict(a)
        permissions = a.get_permissions()
        obtain_elapsed = time.time() - tic

        app_files.append((callgraph_file, permissions, intents, obtain_elapsed))

    return app_files


def write_csv(csv_file: str, path: str, time_to_write: float):
    """
    Write the time taken for Java feature extraction in a CSV file with apk size.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file
    path : str
        Path to the APK file
    time_to_write : float
        Time taken for feature extraction
    """
    file_exists = Path(csv_file).is_file()
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["name", "size", "time"])
        apk_size = Path(path).stat().st_size
        writer.writerow([Path(path).stem, apk_size, time_to_write])


def _safe_extract_worker(args):
    """Top-level worker wrapper that accepts a single tuple argument.

    This must be a top-level function so it can be pickled by multiprocessing.
    """
    apk_path, call_graph_dir, referred_sensitive_apis_dict, referred_actions_dict, referred_permissions_dict, logger = args
    try:
        return extract_features1(
            apk_path,
            call_graph_dir=call_graph_dir,
            referred_sensitive_apis_dict=referred_sensitive_apis_dict,
            referred_actions_dict=referred_actions_dict,
            referred_permissions_dict=referred_permissions_dict,
            logger=logger,
        )
    except Exception as e:
        if logger:
            logger.exception(f"Error processing {apk_path}: {e}")
        else:
            print(f"Error processing {apk_path}: {e}")
        sha256 = Path(apk_path).stem if apk_path is not None else "unknown"
        return (sha256, apk_path, None, 0)


def collect_features(
    app_files: list,
    pool_num: int,
    label: int,
    referred_sensitive_apis_dict: dict,
    referred_actions_dict: dict,
    referred_permissions_dict: dict,
    logger: Logger,
):
    """
    Collect features from the callgraphs.
    :param app_files: a list of the callgraph files.
    :param pool_num: the number of processes.
    :param label: the label of the samples.
    :param referred_sensitive_apis_dict: the referred sensitive APIs.
    :param referred_actions_dict: the referred actions.
    :param referred_permissions_dict: the referred permissions.
    :return: a list of features.
    """
    Vectors = []
    Labels = []

    input_args = [(file, permissions, intents) for file, permissions, intents, _ in app_files]
    obtain_times = [elapsed for _, _, _, elapsed in app_files]

    pool = ThreadPool(pool_num)

    vector = pool.starmap(
        partial(
            extract_features,
            referred_sensitive_apis_dict=referred_sensitive_apis_dict,
            referred_actions_dict=referred_actions_dict,
            referred_permissions_dict=referred_permissions_dict,
            logger=logger,
        ),
        input_args,
    )

    pool.close()
    pool.join()

    Vectors.extend(vector)
    Labels.extend([label] * len(vector))

    assert len(Vectors) == len(Labels)

    return Vectors, Labels, obtain_times


def collect_features1(
    apk_list: list,
    pool_num: int,
    call_graph_dir: str, 
    feature_dir: str,
    referred_sensitive_apis_dict: dict,
    referred_actions_dict: dict,
    referred_permissions_dict: dict,
    logger: Logger,
):
    """
    Collect features from the callgraphs.
    :param app_files: a list of the callgraph files.
    :param pool_num: the number of processes.
    :param label: the label of the samples.
    :param referred_sensitive_apis_dict: the referred sensitive APIs.
    :param referred_actions_dict: the referred actions.
    :param referred_permissions_dict: the referred permissions.
    :return: a list of features.
    """
    
    feature_csv_path = os.path.join(feature_dir, "features.csv")
    if not Path(feature_csv_path).is_file():
        ensure_file(feature_csv_path)
        f = open(feature_csv_path, "a", newline="")
        csv_writer = csv.writer(f)
        feature_header = (
            list(referred_sensitive_apis_dict.keys())
            + list(referred_permissions_dict.keys())
            + list(referred_actions_dict.keys())
        )
        csv_writer.writerow(["apk_name"] + feature_header)
    else:
        f = open(feature_csv_path, "a", newline="")
        csv_writer = csv.writer(f)

    csv_file = os.path.join(feature_dir, "feature_stats.csv")

    def _safe_extract(apk_path):
        try:
            return extract_features1(
                apk_path,
                call_graph_dir=call_graph_dir,
                referred_sensitive_apis_dict=referred_sensitive_apis_dict,
                referred_actions_dict=referred_actions_dict,
                referred_permissions_dict=referred_permissions_dict,
                logger=logger,
            )
        except Exception as e:
            if logger:
                logger.exception(f"Error processing {apk_path}: {e}")
            else:
                print(f"Error processing {apk_path}: {e}")
            sha256 = Path(apk_path).stem if apk_path is not None else "unknown"
            return (sha256, apk_path, None, 0)
    
    args_iter = [
        (
            apk_path,
            call_graph_dir,
            referred_sensitive_apis_dict,
            referred_actions_dict,
            referred_permissions_dict,
            logger,
        )
        for apk_path in apk_list
    ]
    counter = 1
    with ThreadPool(pool_num) as pool:
        for sha256, apk_path, features, elapsed in pool.imap_unordered(_safe_extract_worker, args_iter, chunksize=50):
            print(counter)
            if features is None:
                print(sha256, 'error')
            else:
                csv_writer.writerow([sha256] + features)
                write_csv(csv_file, apk_path, elapsed)
            print(f"Processed {counter}/{len(apk_list)} apps.")
            logger.info(f"Processed {counter}/{len(apk_list)} apps.")
            counter += 1

    f.close()

def generate_features(
    apk_dir: str,
    callgraph_dir: str,
    feature_dir: str,
    pool_num: int,
    label: int,
    logger: Logger,
):
    """
    Generate features.
    :param db_dir: the directory of the database.
    :param feature_dir: the directory of the features.
    :param pool_num: the number of processes.
    :param label: the label of the samples.
    """
    processing_start_time = time.time()
    # Obtain all files
    app_files = obtain_app_files(apk_dir, callgraph_dir)
    if len(app_files) == 0:
        raise FileNotFoundError("No files found")

    # Obtain referred sensitive APIs, actions, and permissions
    referred_sensitive_apis_dict = get_referred_sensitive_apis_dict()
    referred_actions_dict = get_referred_actions_dict()
    referred_permissions_dict = get_referred_permissions_dict()

    # Collect features
    Vectors, Labels, obtain_times = collect_features(
        app_files,
        pool_num,
        label,
        referred_sensitive_apis_dict,
        referred_actions_dict,
        referred_permissions_dict,
        logger,
    )

    # Save features
    feature_header = (
        list(referred_sensitive_apis_dict.keys())
        + list(referred_permissions_dict.keys())
        + list(referred_actions_dict.keys())
    )
    feature_csv = [[] for _ in range(len(Labels) + 1)]
    feature_csv[0].append("apk_name")
    feature_csv[0].extend(feature_header)
    feature_csv[0].append("label")

    for i in range(len(Vectors)):
        (apk_name, vector, elapsed) = Vectors[i]
        feature_csv[i + 1].append(apk_name)
        feature_csv[i + 1].extend(vector)
        feature_csv[i + 1].append(Labels[i])

    feature_csv_path = os.path.join(feature_dir, "features.csv")
    ensure_dir(feature_csv_path)

    with open(feature_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(feature_csv)

    csv_file = os.path.join(feature_dir, "feature_stats.csv")
    for i in range(len(Vectors)):
        apk_name, _, elapsed = Vectors[i]
        total_elapsed = elapsed + obtain_times[i]
        write_csv(csv_file, Path(callgraph_dir) / f"{apk_name}.gml.gz", total_elapsed)

    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time
    logger.info(f"Processing time: {processing_time:.2f} seconds.")
    logger.info(
        f"Average processing time: {processing_time / len(app_files):.2f} seconds."
    )


def generate_features1(
    apk_dir: str,
    callgraph_dir: str,
    feature_dir: str,
    pool_num: int,
    logger: Logger,
):
    """
    Generate features.
    :param db_dir: the directory of the database.
    :param feature_dir: the directory of the features.
    :param pool_num: the number of processes.
    :param label: the label of the samples.
    """

    # Obtain referred sensitive APIs, actions, and permissions
    referred_sensitive_apis_dict = get_referred_sensitive_apis_dict()
    referred_actions_dict = get_referred_actions_dict()
    referred_permissions_dict = get_referred_permissions_dict()

    # Collect features
    collect_features1(
        apk_dir,
        pool_num,
        callgraph_dir,
        feature_dir,
        referred_sensitive_apis_dict,
        referred_actions_dict,
        referred_permissions_dict,
        logger,
    )


if __name__ == "__main__":
    # apk_path = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Noha/AndrOps/apks/2023Q1/8b3cae753cd54d1364068e68095668f0d1ae1e8a51be0dc2898bdf11a257e01e.apk"
    # call_graph_path = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/cfg/8b3cae753cd54d1364068e68095668f0d1ae1e8a51be0dc2898bdf11a257e01e.gml.gz"

    # a, _, _ = AnalyzeAPK(apk_path)
    # permissions = a.get_permissions()
    # intents = get_intent_filters_dict(a)

    # extracted_features = extract_features(
    #     call_graph_path,
    #     permissions,
    #     intents,
    #     get_referred_sensitive_apis_dict(),
    #     get_referred_permissions_dict(),
    #     get_referred_actions_dict(),
    #     logger=None,
    # )

    apk_dir = ["/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Noha/AndrOps/apks/2023Q1/8b3cae753cd54d1364068e68095668f0d1ae1e8a51be0dc2898bdf11a257e01e.apk"]
    generate_features1(
        apk_dir=apk_dir,
        callgraph_dir="/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/cfg",
        feature_dir="/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/ramda/features",
        pool_num=5,
        logger=None,
    )