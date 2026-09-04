import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from models import DREBIN

# ===================
# Substitute these two paths before running.
BASE_FEATURES_PATH = Path("/path/to/features")
MODEL_OUTPUT_DIR = Path("/path/to/output")

RANDOM_SEED = 42
MIN_TOKENS = 5

VAL_PERIODS = {("2025", "q3")}
TEST_PERIODS = {("2025", "q4")}

MODEL_CFG = {
    "C": 0.03,
    "max_iter": 10000,
    "dual": "auto",
    "min_df": 1,
    "max_df": 0.9,
    "max_features": None,
}

# Dual-threshold tuning on VAL.
# We keep both stage-1 floors valid while maximizing the size of the stage-2
# band, so more samples can be deferred instead of being forced into stage 1.
DUAL_THR_MAL_F1_FLOOR = 0.85
DUAL_THR_BEN_F1_FLOOR = 0.85
DUAL_THR_MIN_GAP = 0.10

# If True, deferred samples are included in test stats using a threshold fallback.
# If False, only stage-1 decided samples are reported.
INCLUDE_DEFERRED_IN_STATS = False
# ===================


def _process_inner_dict(prefix, d, tokens, include_scalar_value):
    for kk, vv in d.items():
        if isinstance(vv, list):
            for x in vv:
                tokens.append(f"{prefix}::{kk}::{str(x)}")
        elif isinstance(vv, dict):
            for kkk, vvv in vv.items():
                if isinstance(vvv, list):
                    for x in vvv:
                        tokens.append(f"{prefix}::{kk}::{kkk}::{str(x)}")
                else:
                    tokens.append(f"{prefix}::{kk}::{kkk}")
        else:
            if include_scalar_value:
                tokens.append(f"{prefix}::{kk}::{str(vv)}")
            elif vv:
                tokens.append(f"{prefix}::{kk}")


def parse_json_features(obj):
    tokens = []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict) and "features" in obj and len(obj) == 1:
        return parse_json_features(obj["features"])
    if isinstance(obj, dict):
        for k, v in obj.items():
            ns = str(k)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        tokens.append(f"{ns}::{item}")
                    elif isinstance(item, dict):
                        _process_inner_dict(ns, item, tokens, include_scalar_value=True)
                    else:
                        tokens.append(f"{ns}::{str(item)}")
            elif isinstance(v, dict):
                _process_inner_dict(ns, v, tokens, include_scalar_value=False)
            else:
                if v:
                    tokens.append(ns)
        return tokens
    if isinstance(obj, str):
        return [t for t in obj.split() if t]
    return tokens


def load_dir(dir_path: Path, label: int, min_tokens: int):
    feats, labs, paths = [], [], []
    stats = {"total": 0, "failed_parse": 0, "empty_after_parse": 0, "below_min_tokens": 0}

    for json_file in dir_path.rglob("*.json"):
        stats["total"] += 1
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            tokens = parse_json_features(raw)
            if not tokens:
                stats["empty_after_parse"] += 1
                continue
            if len(tokens) < min_tokens:
                stats["below_min_tokens"] += 1
                continue
            feats.append(tokens)
            labs.append(label)
            paths.append(str(json_file))
        except Exception as e:
            print(f"[WARN] Failed to parse {json_file}: {e}")
            stats["failed_parse"] += 1

    return feats, labs, paths, stats


def split_report(name: str, y):
    c = Counter(y)
    tot = c[0] + c[1]
    if tot == 0:
        print(f"{name}: empty")
        return
    prev = c[1] / tot
    print(f"{name}: total={tot} | malware={c[1]} benign={c[0]} (prev={prev:.3f})")


def eval_block(title: str, y_true, y_pred, scores):
    scores = np.asarray(scores).ravel()
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary"
    )
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    pr_auc = average_precision_score(y_true, scores)

    counts = Counter(y_true)
    mal_prev = counts[1] / (counts[0] + counts[1])

    print(f"\n=== {title} ===")
    print(f"Composition -> malware: {counts[1]}, benign: {counts[0]} (prevalence={mal_prev:.3f})")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Malware P/R/F1: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print("Confusion matrix [rows=true 1/0, cols=pred 1/0]:")
    print(cm)
    print(f"PR-AUC (malware): {pr_auc:.4f}")

    return {"acc": acc, "p": p, "r": r, "f1": f1, "pr_auc": pr_auc}


def _class_metrics_sweep(y, scores, thresholds, positive_label: int):
    n_pos = int(np.sum(y == positive_label))
    precision_list, recall_list, f1_list = [], [], []

    for t in thresholds:
        if positive_label == 1:
            tp = int(np.sum((scores >= t) & (y == 1)))
            fp = int(np.sum((scores >= t) & (y == 0)))
        else:
            tp = int(np.sum((scores <= t) & (y == 0)))
            fp = int(np.sum((scores <= t) & (y == 1)))

        r = tp / n_pos if n_pos > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)

    return np.array(precision_list), np.array(recall_list), np.array(f1_list)


def choose_dual_thresholds_on_val(y_val, val_scores):
    """Select two thresholds from validation scores."""
    val_scores = np.asarray(val_scores).ravel()
    y_val = np.asarray(y_val)
    n_total = len(y_val)

    thresholds = np.sort(np.unique(val_scores))
    mal_prec, mal_rec, mal_f1 = _class_metrics_sweep(y_val, val_scores, thresholds, 1)
    ben_prec, ben_rec, ben_f1 = _class_metrics_sweep(y_val, val_scores, thresholds, 0)

    mal_ok = np.where(mal_f1 >= DUAL_THR_MAL_F1_FLOOR)[0]
    if len(mal_ok) > 0:
        mal_best_idx = int(mal_ok[-1])
        mal_status = "floor_met"
    else:
        mal_best_idx = int(np.argmax(mal_f1))
        mal_status = "floor_unmet_argmax_fallback"
    thr_malware = float(thresholds[mal_best_idx])

    valid_mask = thresholds <= (thr_malware - DUAL_THR_MIN_GAP)
    ben_ok = np.where((ben_f1 >= DUAL_THR_BEN_F1_FLOOR) & valid_mask)[0]
    if len(ben_ok) > 0:
        ben_best_idx = int(ben_ok[0])
        ben_status = "floor_met"
    else:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            ben_best_idx = int(valid_indices[np.argmax(ben_f1[valid_indices])])
        else:
            ben_best_idx = int(np.argmax(ben_f1))
        ben_status = "floor_unmet_argmax_fallback"
    thr_benign = float(thresholds[ben_best_idx])

    deferred_mask = (val_scores > thr_benign) & (val_scores < thr_malware)
    n_deferred = int(np.sum(deferred_mask))
    deferral_rate = n_deferred / n_total if n_total else 0.0

    mal_below_rate = np.array([np.sum(val_scores < t) / n_total for t in thresholds])
    ben_above_rate = np.array([np.sum(val_scores > t) / n_total for t in thresholds])

    print("\n=== Dual Threshold Selection on VAL ===")
    print(
        f"thr_malware = {thr_malware:.6f}  [{mal_status}]  "
        f"P={mal_prec[mal_best_idx]:.3f}  R={mal_rec[mal_best_idx]:.3f}  F1={mal_f1[mal_best_idx]:.3f}"
    )
    print(
        f"thr_benign  = {thr_benign:.6f}  [{ben_status}]  "
        f"P={ben_prec[ben_best_idx]:.3f}  R={ben_rec[ben_best_idx]:.3f}  F1={ben_f1[ben_best_idx]:.3f}"
    )
    print(
        f"Deferral on VAL: {n_deferred}/{n_total} = {deferral_rate:.3f}  "
        f"(gap: {thr_malware - thr_benign:.6f}, min_gap={DUAL_THR_MIN_GAP:.6f})"
    )

    return {
        "thr_malware": thr_malware,
        "thr_benign": thr_benign,
        "deferral_rate": deferral_rate,
        "n_deferred": n_deferred,
        "n_total": n_total,
        "thresholds": thresholds,
        "malware": {
            "precision": mal_prec,
            "recall": mal_rec,
            "f1": mal_f1,
            "below_rate": mal_below_rate,
            "best_idx": mal_best_idx,
            "status": mal_status,
            "f1_floor": DUAL_THR_MAL_F1_FLOOR,
        },
        "benign": {
            "precision": ben_prec,
            "recall": ben_rec,
            "f1": ben_f1,
            "above_rate": ben_above_rate,
            "best_idx": ben_best_idx,
            "status": ben_status,
            "f1_floor": DUAL_THR_BEN_F1_FLOOR,
        },
        "min_gap": DUAL_THR_MIN_GAP,
    }


def plot_dual_threshold_curves_pdf(dual_sel, y_val, val_scores, output_path: Path):
    thresholds = dual_sel["thresholds"]
    mal = dual_sel["malware"]
    ben = dual_sel["benign"]
    thr_malware = dual_sel["thr_malware"]
    thr_benign = dual_sel["thr_benign"]

    mal_prec_dec, mal_rec_dec, mal_f1_dec, mal_cov = [], [], [], []
    ben_prec_dec, ben_rec_dec, ben_f1_dec, ben_cov = [], [], [], []

    for thr in thresholds:
        p, r, f1, cov = _stage1_metrics_decided_only(y_val, val_scores, float(thr), thr_benign)
        mal_prec_dec.append(p)
        mal_rec_dec.append(r)
        mal_f1_dec.append(f1)
        mal_cov.append(cov)

        p, r, f1, cov = _stage1_metrics_decided_only(y_val, val_scores, thr_malware, float(thr))
        ben_prec_dec.append(p)
        ben_rec_dec.append(r)
        ben_f1_dec.append(f1)
        ben_cov.append(cov)

    mal_prec_dec = np.asarray(mal_prec_dec)
    mal_rec_dec = np.asarray(mal_rec_dec)
    mal_f1_dec = np.asarray(mal_f1_dec)
    mal_cov = np.asarray(mal_cov)
    ben_prec_dec = np.asarray(ben_prec_dec)
    ben_rec_dec = np.asarray(ben_rec_dec)
    ben_f1_dec = np.asarray(ben_f1_dec)
    ben_cov = np.asarray(ben_cov)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    ax = axes[0]
    line_p, = ax.plot(thresholds, mal_prec_dec, color="tab:blue", lw=1.8)
    line_r, = ax.plot(thresholds, mal_rec_dec, color="tab:orange", lw=1.8)
    line_f1, = ax.plot(thresholds, mal_f1_dec, color="tab:green", lw=2.2)
    line_cov, = ax.plot(thresholds, mal_cov, color="tab:red", lw=1.4, ls="--")
    ax.axvline(thr_malware, color="black", lw=1.6, ls=":", label=f"thr_malware={thr_malware:.4f}")
    line_floor = ax.axhline(mal["f1_floor"], color="tab:green", lw=1.0, ls="-.", alpha=0.6)
    ax.set_xlabel(r"Threshold $\tau_m$", fontsize=18)
    ax.set_ylabel("Metric value", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=16, loc="best")

    ax = axes[1]
    ax.plot(thresholds, ben_prec_dec, color="tab:blue", lw=1.8)
    ax.plot(thresholds, ben_rec_dec, color="tab:orange", lw=1.8)
    ax.plot(thresholds, ben_f1_dec, color="tab:green", lw=2.2)
    ax.plot(thresholds, ben_cov, color="tab:red", lw=1.4, ls="--")
    ax.axvline(thr_benign, color="black", lw=1.6, ls=":", label=f"thr_benign={thr_benign:.4f}")
    ax.axhline(ben["f1_floor"], color="tab:green", lw=1.0, ls="-.", alpha=0.6)
    ax.set_xlabel(r"Threshold $\tau_b$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=16, loc="best")

    shared_handles = [line_p, line_r, line_f1, line_cov, line_floor]
    shared_labels = [
        "Precision (decided only)",
        "Recall (decided only)",
        "F1 (decided only)",
        "Coverage",
        f"F1 floor={mal['f1_floor']:.2f}",
    ]
    fig.legend(
        shared_handles,
        shared_labels,
        loc="lower center",
        ncol=5,
        fontsize=16,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved threshold performance plot to: {output_path}")


def save_thresholds_f1_txt(dual_sel, output_path: Path):
    thresholds = dual_sel["thresholds"]
    mal_f1 = dual_sel["malware"]["f1"]
    ben_f1 = dual_sel["benign"]["f1"]
    mal_idx = dual_sel["malware"]["best_idx"]
    ben_idx = dual_sel["benign"]["best_idx"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("threshold\tmalware_f1\tbenign_f1\tis_thr_malware\tis_thr_benign\n")
        for i, thr in enumerate(thresholds):
            is_m = "1" if i == mal_idx else "0"
            is_b = "1" if i == ben_idx else "0"
            f.write(
                f"{float(thr):.6f}\t{float(mal_f1[i]):.6f}\t{float(ben_f1[i]):.6f}\t{is_m}\t{is_b}\n"
            )
    print(f"Saved threshold/F1 table to: {output_path}")


def _stage1_metrics_decided_only(y, scores, thr_malware, thr_benign):
    y = np.asarray(y).ravel()
    scores = np.asarray(scores).ravel()

    y_pred = np.where(
        scores >= thr_malware,
        1,
        np.where(scores <= thr_benign, 0, -1),
    )
    decided = y_pred != -1
    coverage = float(np.mean(decided)) if len(y_pred) else 0.0

    if decided.sum() == 0:
        return 0.0, 0.0, 0.0, coverage

    p, r, f1, _ = precision_recall_fscore_support(
        y[decided],
        y_pred[decided],
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    return float(p), float(r), float(f1), coverage


def plot_local_deferred_around_thresholds_pdf(
    dual_sel,
    val_scores,
    output_dir: Path,
    delta: float = 0.5,
    points: int = 201,
):
    scores = np.asarray(val_scores).ravel()
    n_total = max(len(scores), 1)

    thr_m_sel = float(dual_sel["thr_malware"])
    thr_b_sel = float(dual_sel["thr_benign"])

    benign_grid = np.linspace(thr_b_sel - delta, thr_b_sel + delta, points)
    malware_grid = np.linspace(thr_m_sel - delta, thr_m_sel + delta, points)

    benign_deferred_pct = np.full(points, np.nan, dtype=float)
    malware_deferred_pct = np.full(points, np.nan, dtype=float)

    for i, thr_b in enumerate(benign_grid):
        if thr_b >= (thr_m_sel - DUAL_THR_MIN_GAP):
            continue
        deferred = (scores > thr_b) & (scores < thr_m_sel)
        benign_deferred_pct[i] = 100.0 * float(deferred.sum()) / n_total

    for i, thr_m in enumerate(malware_grid):
        if thr_m <= (thr_b_sel + DUAL_THR_MIN_GAP):
            continue
        deferred = (scores > thr_b_sel) & (scores < thr_m)
        malware_deferred_pct[i] = 100.0 * float(deferred.sum()) / n_total

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(benign_grid, benign_deferred_pct, color="tab:blue", lw=2.0, label="Deferred %")
    ax.axvline(thr_b_sel, color="black", lw=1.4, ls=":", label=f"Selected={thr_b_sel:.4f}")
    ax.set_title("Deferred % vs benign threshold (malware threshold fixed)")
    ax.set_xlabel(r"Benign threshold $\tau_b$")
    ax.set_ylabel("Deferred samples (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    benign_path = output_dir / "deferred_local_benign_threshold.pdf"
    fig.savefig(benign_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved local deferred plot (benign threshold) to: {benign_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(malware_grid, malware_deferred_pct, color="tab:orange", lw=2.0, label="Deferred %")
    ax.axvline(thr_m_sel, color="black", lw=1.4, ls=":", label=f"Selected={thr_m_sel:.4f}")
    ax.set_title("Deferred % vs malware threshold (benign threshold fixed)")
    ax.set_xlabel(r"Malware threshold $\tau_m$")
    ax.set_ylabel("Deferred samples (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    malware_path = output_dir / "deferred_local_malware_threshold.pdf"
    fig.savefig(malware_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved local deferred plot (malware threshold) to: {malware_path}")


def _create_model():
    return DREBIN(
        C=MODEL_CFG["C"],
        max_iter=MODEL_CFG["max_iter"],
        random_state=RANDOM_SEED,
        dual=MODEL_CFG["dual"],
        min_df=MODEL_CFG["min_df"],
        max_df=MODEL_CFG["max_df"],
        max_features=MODEL_CFG["max_features"],
    )


def stage2(sample, log_path: str = "stage2_samples.txt"):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(sample + "\n")
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DREBIN and evaluate with dual thresholds (default) or a single fixed threshold."
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help=(
            "If set, classify with this single threshold instead of selecting "
            "dual thresholds on VAL (no stage-2 deferral in this mode)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fixed_threshold = args.fixed_threshold
    random.seed(RANDOM_SEED)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test, test_paths = [], [], []

    for family, label in [("malware", 1), ("benign", 0)]:
        family_root = BASE_FEATURES_PATH / family
        if not family_root.exists():
            raise RuntimeError(f"Missing folder: {family_root}")

        for year_dir in family_root.iterdir():
            if not year_dir.is_dir():
                continue
            year = year_dir.name

            for q_dir in year_dir.iterdir():
                if not q_dir.is_dir():
                    continue
                quarter = q_dir.name

                feats, labs, paths, stats = load_dir(q_dir, label, MIN_TOKENS)
                print(f"[LOAD] {family}/{year}/{quarter} -> kept={len(feats)} | stats={stats}")

                if (year, quarter) in TEST_PERIODS:
                    X_test.extend(feats)
                    y_test.extend(labs)
                    test_paths.extend(paths)
                elif (year, quarter) in VAL_PERIODS:
                    X_val.extend(feats)
                    y_val.extend(labs)
                else:
                    X_train.extend(feats)
                    y_train.extend(labs)

    print("\n=== Split sizes ===")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    split_report("TRAIN", y_train)
    split_report("VAL  ", y_val)
    split_report("TEST ", y_test)
    print(f"val_periods={sorted(VAL_PERIODS)}, test_periods={sorted(TEST_PERIODS)}")

    if not X_train or not X_val or not X_test:
        raise RuntimeError("Empty train/val/test split. Check folder names and years/quarters.")

    # 1) Train on TRAIN only.
    model = _create_model()
    model.fit(X_train, y_train)

    try:
        print("\nVocabulary size (TRAIN model):", len(model._vectorizer.vocabulary_))
    except Exception:
        pass

    # 2) Select thresholds on VAL (or use the fixed threshold if provided).
    _, val_scores = model.predict(X_val)
    val_scores = np.asarray(val_scores).ravel()

    dual_sel = None
    if fixed_threshold is None:
        dual_sel = choose_dual_thresholds_on_val(y_val, val_scores)
        thr_malware = dual_sel["thr_malware"]
        thr_benign = dual_sel["thr_benign"]
        plot_dual_threshold_curves_pdf(dual_sel, y_val, val_scores, MODEL_OUTPUT_DIR / "threshold_curves.pdf")
        plot_local_deferred_around_thresholds_pdf(dual_sel, val_scores, MODEL_OUTPUT_DIR, delta=0.5)
        save_thresholds_f1_txt(dual_sel, MODEL_OUTPUT_DIR / "thresholds_f1_val.txt")

        y_val_pred = np.where(
            val_scores >= thr_malware,
            1,
            np.where(val_scores <= thr_benign, 0, -1),
        )
        decided = y_val_pred != -1
        if decided.sum() > 0:
            eval_block(
                "VAL (stage-1 decided only)",
                np.array(y_val)[decided],
                y_val_pred[decided],
                val_scores[decided],
            )
    else:
        y_val_pred = [1 if s >= fixed_threshold else 0 for s in val_scores]
        eval_block("VAL (fixed threshold)", list(y_val), y_val_pred, list(val_scores))

    # 3) Final fit on TRAIN + VAL.
    X_tr = X_train + X_val
    y_tr = y_train + y_val

    final_model = _create_model()
    final_model.fit(X_tr, y_tr)

    try:
        print("\nVocabulary size (FINAL model):", len(final_model._vectorizer.vocabulary_))
    except Exception:
        pass

    # 4) Evaluate on TEST.
    _, test_scores = final_model.predict(X_test)
    test_scores = np.asarray(test_scores).ravel()

    if fixed_threshold is None:
        with open("stage2_samples.txt", "w", encoding="utf-8"):
            pass

        y_test_decided, y_pred_decided, scores_decided = [], [], []
        deferred_indices = []
        stage2_count = 0

        for i, score in enumerate(test_scores):
            if score >= thr_malware:
                y_test_decided.append(y_test[i])
                y_pred_decided.append(1)
                scores_decided.append(score)
            elif score <= thr_benign:
                y_test_decided.append(y_test[i])
                y_pred_decided.append(0)
                scores_decided.append(score)
            else:
                stage2(Path(test_paths[i]).stem)
                deferred_indices.append(i)
                stage2_count += 1

        print(
            f"\nStage-2 routing: benign<= {thr_benign:.6f}  malware>= {thr_malware:.6f}  "
            f"sent_to_stage2={stage2_count}"
        )
        n_test = len(y_test)
        n_decided = len(y_test_decided)
        print(f"Stage-1 decided: {n_decided}/{n_test} ({n_decided / max(n_test, 1):.3f})")

        deferred_set = set(deferred_indices)
        deferred_malware = sum(1 for i in deferred_set if y_test[i] == 1)
        deferred_benign = len(deferred_set) - deferred_malware

        deferred_malware_hashes = [Path(test_paths[i]).stem for i in deferred_indices if y_test[i] == 1]
        deferred_benign_hashes = [Path(test_paths[i]).stem for i in deferred_indices if y_test[i] == 0]

        with open("deferred.txt", "w", encoding="utf-8") as f:
            f.write("MALWARE:\n")
            for stem in deferred_malware_hashes:
                f.write(stem + "\n")
            f.write("\nBENIGN:\n")
            for stem in deferred_benign_hashes:
                f.write(stem + "\n")
        print(
            f"Wrote deferred samples to deferred.txt "
            f"(malware={len(deferred_malware_hashes)}, benign={len(deferred_benign_hashes)})"
        )

        if INCLUDE_DEFERRED_IN_STATS:
            y_pred_all = [1 if s >= thr_malware else 0 for s in test_scores]
            print("\nTest stats mode: including deferred samples via malware threshold fallback.")
            eval_block("TEST (all samples)", list(y_test), y_pred_all, list(test_scores))
        else:
            print("\nTest stats mode: stage-1 decided only.")
            if y_test_decided:
                eval_block(
                    "TEST (stage-1 decided only; deferred excluded)",
                    y_test_decided,
                    y_pred_decided,
                    scores_decided,
                )
            else:
                print("\n=== TEST ===\nNo samples decided by stage-1.")

        print(
            f"Deferred to stage 2: malware={deferred_malware} benign={deferred_benign} total={len(deferred_set)}"
        )

        # False negatives.
        fn_paths = []
        for i, score in enumerate(test_scores):
            if not INCLUDE_DEFERRED_IN_STATS and i in deferred_set:
                continue
            pred = 1 if score >= thr_malware else 0
            if y_test[i] == 1 and pred == 0:
                fn_paths.append(Path(test_paths[i]).stem)

        with open("fn.txt", "w", encoding="utf-8") as f:
            for stem in fn_paths:
                f.write(stem + "\n")
        print(f"\nWrote {len(fn_paths)} False Negatives to fn.txt")

        out_model = MODEL_OUTPUT_DIR / (
            f"drebin__C={MODEL_CFG['C']}"
            f"__min_df={MODEL_CFG['min_df']}"
            f"__max_df={MODEL_CFG['max_df']}"
            f"__max_features={MODEL_CFG['max_features']}"
            f"__thr_malware={thr_malware:.6f}"
            f"__thr_benign={thr_benign:.6f}.joblib"
        )
        dump(
            {
                "model": final_model,
                "thr_malware": thr_malware,
                "thr_benign": thr_benign,
                "deferral_rate_val": dual_sel["deferral_rate"],
                "mal_f1_floor": DUAL_THR_MAL_F1_FLOOR,
                "ben_f1_floor": DUAL_THR_BEN_F1_FLOOR,
                "min_gap": DUAL_THR_MIN_GAP,
                "cfg": MODEL_CFG,
            },
            out_model,
        )
        print(f"\nSaved model bundle to: {out_model}")
    else:
        y_test_pred = [1 if s >= fixed_threshold else 0 for s in test_scores]
        eval_block("TEST (fixed threshold)", list(y_test), y_test_pred, list(test_scores))

        out_model = MODEL_OUTPUT_DIR / (
            f"drebin__C={MODEL_CFG['C']}"
            f"__min_df={MODEL_CFG['min_df']}"
            f"__max_df={MODEL_CFG['max_df']}"
            f"__max_features={MODEL_CFG['max_features']}"
            f"__thr={fixed_threshold:.6f}.joblib"
        )
        dump(
            {
                "model": final_model,
                "threshold": fixed_threshold,
                "cfg": MODEL_CFG,
            },
            out_model,
        )
        print(f"\nSaved model bundle to: {out_model}")


if __name__ == "__main__":
    main()