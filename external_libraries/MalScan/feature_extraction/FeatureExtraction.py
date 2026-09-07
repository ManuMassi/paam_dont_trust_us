import os.path
from pathlib import Path
import networkx as nx
import time
import argparse
import csv
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import numpy as np
import igraph
import zipfile
import pandas as pd
import json


def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware.', required=True, type=str)
    parser.add_argument('-m', '--meta_file', help='The path of metadata file', required=True, type=str)
    parser.add_argument('-o', '--output', help='The dir_path of output', required=True, type=str)
    # parser.add_argument('-c', '--centrality', help='The type of centrality: degree, katz, closeness, harmonic', required=True, type=str)
    args = parser.parse_args()
    return args


def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis


def all_centrality_features(args, sensitive_apis):
    file, label = args
    results = {}
    sha256 = file.split('/')[-1].split('.gml.gz')[0]
    tic = time.time()
    try:
        CG = nx.read_gml(file)
        g = igraph.Graph.from_networkx(CG)
        # nodes = g.vcount()
        # degree_centrality = dict(zip(g.vs['_nx_name'], np.array(g.degree())/(nodes-1)))
        # pagerank_centrality = dict(zip(g.vs['_nx_name'], g.pagerank()))
        # eigenvector_centrality = dict(zip(g.vs['_nx_name'], g.eigenvector_centrality()))
        # authority_centrality = dict(zip(g.vs['_nx_name'], g.authority_score()))
        harmonic_centrality = nx.harmonic_centrality(CG)

        # degree_vector = [degree_centrality.get(api, 0) for api in sensitive_apis]
        # pagerank_vector = [pagerank_centrality.get(api, 0) for api in sensitive_apis]
        # eigenvector_vector = [eigenvector_centrality.get(api, 0) for api in sensitive_apis]
        # authority_vector = [authority_centrality.get(api, 0) for api in sensitive_apis]
        harmonic_vector = [harmonic_centrality.get(api, 0) for api in sensitive_apis]

        # results["degree"] = (sha256, label, degree_vector)
        # results["pagerank"] = (sha256, label, pagerank_vector)
        # results["eigenvector"] = (sha256, label, eigenvector_vector)
        # results["authority"] = (sha256, label, authority_vector)
        results["harmonic"] = (sha256, label, harmonic_vector, time.time() - tic)
    except:
        # results["degree"] = (sha256, label, None)
        # results["pagerank"] = (sha256, label, None)
        # results["eigenvector"] = (sha256, label, None)
        # results["authority"] = (sha256, label, None)
        results["harmonic"] = (sha256, label, None, time.time() - tic)

    # try:
    #     CG = nx.read_gml(file)
    #     katz_centrality = nx.katz_centrality(CG)
    #     closeness_centrality = nx.closeness_centrality(CG)
    #     harmonic_centrality = nx.harmonic_centrality(CG)
    #
    #     katz_vector = []
    #     closeness_vector = []
    #     harmonic_vector = []
    #     for api in sensitive_apis:
    #         if api in katz_centrality.keys():
    #             katz_vector.append(katz_centrality[api])
    #         else:
    #             katz_vector.append(0)
    #         if api in closeness_centrality.keys():
    #             closeness_vector.append(closeness_centrality[api])
    #         else:
    #             closeness_vector.append(0)
    #         if api in harmonic_centrality.keys():
    #             harmonic_vector.append(harmonic_centrality[api])
    #         else:
    #             harmonic_vector.append(0)
    #
    #     results["katz"] = (sha256, katz_vector)
    #     results["closeness"] = (sha256, closeness_vector)
    #     results["harmonic"] = (sha256, harmonic_vector)
    #
    # except:
    #     results["katz"] = (sha256, None)
    #     results["closeness"] = (sha256, None)
    #     results["harmonic"] = (sha256, None)

    return results


def obtain_apps(dataset_path, meta_file):
    """Returns a list of (cg_path, label) tuples."""
    apps = []

    if meta_file.endswith('.json'):
        with open(meta_file, "r") as f:
            meta = json.load(f)
        with open(Path(meta_file).parent / "apg-y.json", "r") as f:
            label = json.load(f)
        for sample in zip(meta, label):
            cg_path = os.path.join(dataset_path, f"{sample[0]['sha256'].upper()}.gml.gz")
            if os.path.isfile(cg_path):
                apps.append((cg_path, sample[1]))

    else:
        with zipfile.ZipFile(meta_file, "r", zipfile.ZIP_DEFLATED) as z:
            ds_csv = pd.concat([pd.read_csv(z.open(f)) for f in z.namelist()],
                               ignore_index=True)
            ds_csv.timestamp = pd.to_datetime(ds_csv.timestamp)
            for sha256, lbl, date in zip(ds_csv.sha256, ds_csv.label, ds_csv.timestamp):
                cg_path = os.path.join(dataset_path, f"{sha256.upper()}.gml.gz")
                if os.path.isfile(cg_path):
                    apps.append((cg_path, int(lbl)))

    return apps


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


def main(dataset_path, labels_path, output):
    tic = time.time()
    sensitive_apis_path = 'sensitive_apis.txt'
    sensitive_apis = obtain_sensitive_apis(sensitive_apis_path)

    os.makedirs(output, exist_ok=True)
    exist_files = [str(f).split('/')[-1].split('.csv')[0] for f in Path(output).glob("*.csv")]
    apks_df = pd.read_csv(labels_path)

    # types = ['degree', 'closeness', 'harmonic', 'katz', 'eigenvector', 'pagerank', 'authority']
    types = ['harmonic']
    print(f"Extracting {types} features for {len(apks_df)} apps in {dataset_path} to {output}.")
    exist_files = {}
    cfg = []
    for centrality_type in types:
        if not os.path.exists(os.path.join(output, f'{centrality_type}_features.csv')):
            exist_files[centrality_type] = []
        done = pd.read_csv(os.path.join(output, f'{centrality_type}_features.csv'))
        exist_files[centrality_type] = done['SHA256'].values.tolist()

        for row in apks_df.itertuples():
            sha256 = row.sha256
            if sha256 in exist_files[centrality_type]:
                print(f"Skipping {sha256} for {centrality_type} features, already exists.")
                continue
            cfg_path = Path(dataset_path) / f"{sha256}.gml.gz"
            if os.path.isfile(cfg_path):
                cfg.append((str(cfg_path), row.label))

    print(f"Found {len(cfg)} apps with call graphs in {dataset_path} to.")

    # Open all CSV files upfront and write headers
    csv_writers = {}
    csv_file_handles = {}
    for centrality_type in types:
        csv_path = os.path.join(output, f'{centrality_type}_features.csv')
        if not exist_files[centrality_type]:
            f = open(csv_path, 'w', newline='')
            writer = csv.writer(f)
            writer.writerow(['SHA256', 'Label'] + sensitive_apis)
        else:
            f = open(csv_path, 'a', newline='')
            writer = csv.writer(f)

        csv_writers[centrality_type] = writer
        csv_file_handles[centrality_type] = f

    csv_file = os.path.join(
        "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan",
        "feature_stats.csv",
    )
    print(f"Extracting features for {len(cfg)} apps using {len(sensitive_apis)} sensitive APIs.")
    worker = partial(all_centrality_features, sensitive_apis=sensitive_apis)
    counter = 1
    with ThreadPool(10) as pool:
        for result in pool.imap_unordered(worker, cfg, chunksize=50):
            print(counter)
            for centrality_type, (sha256, label, vector, elapsed) in result.items():
                if vector is None:
                    print(sha256, 'error')
                else:
                    csv_writers[centrality_type].writerow([sha256, label] + vector)
                    write_csv(
                        csv_file,
                        Path(dataset_path) / f"{sha256}.gml.gz",
                        elapsed
                    )
            print(f"Processed {counter}/{len(cfg)} apps.")
            counter += 1

    for f in csv_file_handles.values():
        f.close()

    print(time.time() - tic)

if __name__ == '__main__':
    # args = parseargs()

    cfg_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/cfg"
    output_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/features"
    labels_path = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/data/labels_vt.csv"

    main(cfg_dir, labels_path, output_dir)
