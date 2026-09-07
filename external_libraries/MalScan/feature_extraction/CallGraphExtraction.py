import networkx as nx
import csv
import zipfile
import time
import os
from androguard.misc import AnalyzeAPK
import argparse
import glob
from multiprocessing import Pool as ThreadPool
from functools import partial
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="To obtain the call graphs.")
    parser.add_argument(
        "-f",
        "--file",
        help="The path of an APK file or a dir contains some APK files",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-o", "--output", help="The path of output.", required=False, type=str
    )
    args = parser.parse_args()
    return args


def get_call_graph(dx):
    # CG = nx.MultiDiGraph()
    CG = nx.DiGraph()
    nodes = dx.find_methods(".*", ".*", ".*", ".*")
    for m in nodes:
        API = m.get_method()
        class_name = API.class_name
        method_name = API.name
        # api_call = class_name + '->' + method_name + descriptor
        if API.__dict__.get("descriptor") is not None:
            api_call = class_name + "->" + method_name + "".join(API.descriptor)
        else:
            api_call = class_name + "->" + method_name + API.proto
        if len(m.get_xref_to()) == 0:
            continue
        CG.add_node(api_call)

        for other_class, callee, offset in m.get_xref_to():
            # _callee = callee.get_class_name() + '->' + callee.get_name() + callee.get_descriptor()
            if callee.__dict__.get("descriptor") is not None:
                _callee = callee.class_name + "->" + callee.name + "".join(callee.descriptor)
            else:
                _callee = callee.class_name + "->" + callee.name + callee.proto
            CG.add_node(_callee)
            if not CG.has_edge(API, callee):
                CG.add_edge(api_call, _callee)

    return CG


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


def apk_to_callgraph(app_path, exist_files, out_path):

    apk_name = str(app_path).split("/")[-1].split(".apk")[0]

    if apk_name in exist_files:
        return None
    elif not zipfile.is_zipfile(str(app_path)):
        print(f"Skipping {app_path}: not a valid APK file.")
        return None
    else:
        try:
            tic = time.time()
            a, d, dx = AnalyzeAPK(app_path)
            call_graph = get_call_graph(dx=dx)
            print("Processing " + apk_name)

            # file_cg = out_path + "/" + apk_name + ".gml.gz"
            # nx.write_gml(call_graph, file_cg)
            return app_path, time.time() - tic

        except Exception:
            print(f"Error processing {app_path}:")
            import traceback
            traceback.print_exc()
            return None


def main(file, output, n_jobs=40):
    tic = time.time()

    folder = os.path.exists(output)
    if not folder:
        os.makedirs(output)
    exist_files = os.listdir(output)
    exist_files = [f.split(".gml.gz")[0] for f in exist_files]
    if output[-1] == "/":
        out_path = output[:-1]
    else:
        out_path = output

    if isinstance(file, list) or os.path.isdir(file):
        if isinstance(file, list):
            apks = file
        else:
            if file[-1] == "/":
                path = file + "*.apk"
            else:
                path = file + "/*.apk"
            apks = glob.glob(path)

        csv_file = os.path.join(
            "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan",
            "callgraph_stats.csv",
        )
        with ThreadPool(n_jobs) as pool:
        #     results = pool.map(
        #         partial(apk_to_callgraph, exist_files=exist_files, out_path=out_path),
        #         apks,
        #     )

        #     for result in results:
        #         if result is None:
        #             continue
        #         app_path, elapsed = result
        #         write_csv(csv_file, app_path, elapsed)
            for i, result in enumerate(pool.imap_unordered(
                partial(apk_to_callgraph, exist_files=exist_files, out_path=out_path),
                apks,
            ), start=1):
                if result is None:
                    continue

                app_path, elapsed = result
                print(f"[{i}/{len(apks)}] Processed {Path(app_path).name} ({elapsed:.2f}s)")
                write_csv(csv_file, app_path, elapsed)
    else:
        result = apk_to_callgraph(file, exist_files, out_path)
        if result is not None:
            app_path, elapsed = result
            csv_file = os.path.join(out_path, "callgraph_stats.csv")
            write_csv(csv_file, app_path, elapsed)

    print(time.time() - tic)


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
        apk_paths.append(Path(base_dir) / yq / f"{sha256}.apk")

    return apk_paths, labels, years


if __name__ == "__main__":
    args = parse_args()
    # apk_list = []
    # with zipfile.ZipFile("/disk3/asotgiu/elsa/elsa_ds.zip", "r", zipfile.ZIP_DEFLATED) as z:
    #     ds_csv = pd.concat([pd.read_csv(z.open(f)) for f in z.namelist()],
    #                        ignore_index=True)
    #     ds_csv.timestamp = pd.to_datetime(ds_csv.timestamp)
    #     all_sha256 = ds_csv.sha256.values.tolist()
    #     for sha256, label, date in zip(
    #         ds_csv.sha256, ds_csv.label,
    #         ds_csv.timestamp
    #     ):
    #         apk_list.append(f"/home/asotgiu/elsa_prototype/data/apks/"
    #                         f"{date.year}Q{date.quarter}/{sha256.upper()}.apk")

    labels_path = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/data/labels_vt.csv"
    base_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Noha/AndrOps/apks/"
    output_dir = "/home/dsoi/snap/snapd-desktop-integration/315/Scrivania/Kaiba/Diegos/Drebin2.0/baselines/malscan/cfg1"

    apk_list, labels, years = get_apk_paths(
        labels_path,
        base_dir,
    )
    print(len(apk_list))
    main(apk_list, output_dir)
    # main(args.file, args.output)
