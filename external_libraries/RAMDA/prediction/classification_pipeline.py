import torch
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from logging import Logger

from utils.helper import ensure_dir
from utils.network.helper import eval_metrics
from utils.network.fd_vae import FD_VAE, fd_vae_train, fd_vae_evaluate
from utils.network.mu_sigma_mlp import (
    MUSIGMA_MLP,
    musigma_mlp_train,
    musigma_mlp_evaluate,
)
from dataset import RAMDADataset

import numpy as np
from itertools import islice


DUAL_THR_MAL_F1_FLOOR = 0.85
DUAL_THR_BEN_F1_FLOOR = 0.85
DUAL_THR_MIN_GAP = 0.10


def split_train_val_test(samples_df: pd.DataFrame, samples_hash: list):

    samples_df = samples_df[samples_df["sha256"].isin(samples_hash)].copy()
    samples_df["year_quarter"] = (
        pd.to_datetime(samples_df["first_seen"]).dt.to_period("Q").astype(str)
    )

    train_samples = samples_df[samples_df["year_quarter"].between("2024Q1", "2025Q2")]
    validation_samples = samples_df[samples_df["year_quarter"] == "2025Q3"]
    test_samples = samples_df[samples_df["year_quarter"] == "2025Q4"]

    return train_samples, validation_samples, test_samples


def evaluate_with_two_models(vae_preds, mlp_preds, labels):
    """
    Evaluate the model with two models.Il
    :param vae_preds: Predictions of VAE.
    :param mlp_preds: Predictions of MLP.
    :param labels: Labels.
    :return: Evaluation results.
    """
    # Evaluate the model
    preds = []
    for i in range(len(vae_preds)):
        if vae_preds[i] == 0 and mlp_preds[i] == 0:
            preds.append(0)
        else:
            preds.append(1)

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)

    return report, cm


def test(
    test_dataset: RAMDADataset,
    model_name: str,
    model_type: str,
    logger: Logger,
    **kwargs,
):
    """
    Test the model.
    :param test_dataset: Testing dataset.
    :param model_name: Name of the model.
    :param model_type: Type of the model.
    :param logger: Logger.
    :param kwargs: Other arguments.
    """
    # Load and evaluate the model
    # logger.info("Testing the model...")
    print("Testing the model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fd_vae_model_path = os.path.join(current_dir, "models", model_name + "_fd_vae")
    musigma_mlp_model_path = os.path.join(
        current_dir, "models", model_name + "_musigma_mlp"
    )

    if model_type == "ramda":
        fd_vae_model = torch.load(fd_vae_model_path)
        musigma_mlp_model = torch.load(musigma_mlp_model_path)
        vae_ret_test, vae_test_pred, vae_test_label = fd_vae_evaluate(
            test_dataset, fd_vae_model, **kwargs
        )
        mlp_ret_test, mlp_test_pred, mlp_test_label = musigma_mlp_evaluate(
            test_dataset, musigma_mlp_model, fd_vae_model, **kwargs
        )

        assert len(vae_test_pred) == len(mlp_test_pred)
        # Merge predictions
        ret_test = evaluate_with_two_models(
            vae_test_pred, mlp_test_pred, vae_test_label
        )
    else:
        raise ValueError("Model type is not valid.")

    return ret_test, mlp_test_pred, mlp_test_label, test_dataset


def train(
    train_dataset: RAMDADataset,
    test_dataset: RAMDADataset,
    model_name: str,
    model_type: str,
    logger: Logger,
    **kwargs,
):

    in_channels = kwargs["in_channels"]
    hidden_channels = kwargs["hidden_channels"]
    out_channels = kwargs["out_channels"]
    num_classes = kwargs["num_classes"]
    dropout_ratio = kwargs["dropout_ratio"]

    if model_type == "ramda":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fd_vae_model_path = os.path.join(current_dir, "models", model_name + "_fd_vae")
        musigma_mlp_model_path = os.path.join(
            current_dir, "models", model_name + "_musigma_mlp"
        )
        ensure_dir(fd_vae_model_path)
        ensure_dir(musigma_mlp_model_path)
        fd_vae_model = FD_VAE(in_channels, hidden_channels, out_channels, dropout_ratio)
        musigama_mlp_model = MUSIGMA_MLP(
            out_channels * 2, hidden_channels, num_classes, dropout_ratio
        )
        # Train fd_vae_model
        print("Training FD-VAE model...")
        vae_scores = fd_vae_train(
            fd_vae_model_path,
            train_dataset,
            test_dataset,
            fd_vae_model,
            logger,
            **kwargs,
        )
        print("Training MUSIGMA-MLP model...")
        mlp_scores = musigma_mlp_train(
            musigma_mlp_model_path,
            train_dataset,
            test_dataset,
            musigama_mlp_model,
            fd_vae_model,
            logger,
            **kwargs,
        )

        return vae_scores, mlp_scores


def save_results(result_path, ret_val, ret_test):

    with open(result_path, "w") as f:
        f.write("VALIDATION REPORT\n")
        f.write(
            f"Test F1: {ret_val['f1']:.4f}, Recall: {ret_val['recall']:.4f}, Precision: {ret_val['precision']:.4f}, Accuracy: {ret_val['accuracy']:.4f}, AUC: {ret_val['auc']:.4f}, TPR: {ret_val['tpr']:.4f}, FPR: {ret_val['fpr']:.4f}"
        )
        f.write("\nTEST REPORT\n")
        f.write(
            f"Test F1: {ret_test['f1']:.4f}, Recall: {ret_test['recall']:.4f}, Precision: {ret_test['precision']:.4f}, Accuracy: {ret_test['accuracy']:.4f}, AUC: {ret_test['auc']:.4f}, TPR: {ret_test['tpr']:.4f}, FPR: {ret_test['fpr']:.4f}"
        )


def dataset_labels(dataset: RAMDADataset):
    benign_data, malware_data = dataset.get_data()
    return np.concatenate(
        (np.zeros(len(benign_data)), np.ones(len(malware_data))), axis=0
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

    return thr_benign, thr_malware


def combine_predictions(vae_pred, mlp_pred):
    final_pred = np.full(len(vae_pred), -1)

    final_pred[(vae_pred == 1) | (mlp_pred == 1)] = 1
    final_pred[(vae_pred == 0) & (mlp_pred == 0)] = 0

    return final_pred


def evaluate_predictions(labels, final_pred):

    decided = final_pred != -1

    report = classification_report(
        np.asarray(labels)[decided],
        final_pred[decided],
    )

    cm = confusion_matrix(
        np.asarray(labels)[decided],
        final_pred[decided],
    )

    return report, cm, decided


def train_and_test(
    data_distribution: tuple,
    model_name: str,
    model_type: str,
    result_path: str,
    logger: Logger,
    threshold: float,
    predefined_loss: float,
    **kwargs,
):

    val_vae_scores, val_mlp_scores = train(
        train_dataset=data_distribution[0],
        test_dataset=data_distribution[1],
        model_name=model_name,
        model_type=model_type,
        logger=logger,
        **kwargs,
    )

    test_vae_scores, test_mlp_scores = train(
        train_dataset=data_distribution[3],
        test_dataset=data_distribution[2],
        model_name=model_name,
        model_type=model_type,
        logger=logger,
        **kwargs,
    )

    if threshold == 0.5 and predefined_loss == 20:
        print(
            "Training and testing the model with threshold=0.5 and predefined_loss=20..."
        )
        val_vae_scores = scores_to_numpy(val_vae_scores)
        val_mlp_scores = scores_to_numpy(val_mlp_scores)
        val_vae_pred = (val_vae_scores > predefined_loss).astype(int)
        val_mlp_pred = (val_mlp_scores > threshold).astype(int)

        print("Evaluating the model on validation dataset...")
        report_val, cm_val = evaluate_with_two_models(
            val_vae_pred, val_mlp_pred, dataset_labels(data_distribution[1])
        )

        test_vae_scores = scores_to_numpy(test_vae_scores)
        test_mlp_scores = scores_to_numpy(test_mlp_scores)
        test_vae_pred = (test_vae_scores > predefined_loss).astype(int)
        test_mlp_pred = (test_mlp_scores > threshold).astype(int)

        print("Evaluating the model on test dataset...")
        report_test, cm_test = evaluate_with_two_models(
            test_vae_pred, test_mlp_pred, dataset_labels(data_distribution[2])
        )

    elif threshold == None and predefined_loss == None:
        print("Training and testing the model with single threshold best f1")
        val_vae_scores = scores_to_numpy(val_vae_scores)
        val_mlp_scores = scores_to_numpy(val_mlp_scores)

        mlp_thresholds = np.linspace(0, 1, num=201)
        vae_predefined_losses = np.linspace(
            val_vae_scores.min(), val_vae_scores.max(), num=50
        )
        best_threshold = None
        best_vae_threshold = None
        labels = dataset_labels(data_distribution[1])
        best_f1 = -1

        for vae_t in vae_predefined_losses:
            val_vae_pred = (val_vae_scores > vae_t).astype(int)
            for mlp_t in mlp_thresholds:
                val_mlp_pred = (val_mlp_scores > mlp_t).astype(int)
                report_val, cm_val = evaluate_with_two_models(
                    val_vae_pred, val_mlp_pred, labels
                )
                final_pred = np.logical_or(
                    val_vae_pred,
                    val_mlp_pred,
                ).astype(int)
                f1 = f1_score(labels, final_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = mlp_t
                    best_vae_threshold = vae_t

        val_vae_pred = (val_vae_scores > best_vae_threshold).astype(int)
        val_mlp_pred = (val_mlp_scores > best_threshold).astype(int)

        report_val, cm_val = evaluate_with_two_models(
            val_vae_pred,
            val_mlp_pred,
            labels,
        )

        test_vae_scores = scores_to_numpy(test_vae_scores)
        test_mlp_scores = scores_to_numpy(test_mlp_scores)
        test_vae_pred = (test_vae_scores > best_vae_threshold).astype(int)
        test_mlp_pred = (test_mlp_scores > best_threshold).astype(int)
        report_test, cm_test = evaluate_with_two_models(
            test_vae_pred, test_mlp_pred, dataset_labels(data_distribution[2])
        )
        print(
            f"Best thresholds found: VAE threshold={best_vae_threshold}, MLP threshold={best_threshold}, Best F1={best_f1}"
        )
        threshold = best_threshold
        predefined_loss = best_vae_threshold
    elif threshold == "dual" and predefined_loss == "dual":
        val_labels = dataset_labels(data_distribution[1])
        val_vae_scores = scores_to_numpy(val_vae_scores)
        val_mlp_scores = scores_to_numpy(val_mlp_scores)
        vae_thr_benign, vae_thr_malware = choose_dual_thresholds_on_val(
            val_labels,
            val_vae_scores,
        )

        mlp_thr_benign, mlp_thr_malware = choose_dual_thresholds_on_val(
            val_labels,
            val_mlp_scores,
        )

        val_vae_pred = np.where(
            val_vae_scores >= vae_thr_malware,
            1,
            np.where(val_vae_scores <= vae_thr_benign, 0, -1),
        )

        val_mlp_pred = np.where(
            val_mlp_scores >= mlp_thr_malware,
            1,
            np.where(val_mlp_scores <= mlp_thr_benign, 0, -1),
        )

        final_val_pred = combine_predictions(
            val_vae_pred,
            val_mlp_pred,
        )

        report_val, cm_val, decided = evaluate_predictions(
            val_labels,
            final_val_pred,
        )

        print(
            f"Deferred on validation: {(~decided).sum()}/{len(val_labels)} "
            f"({100*(~decided).sum()/len(val_labels):.2f}%)"
        )

        test_vae_scores = scores_to_numpy(test_vae_scores)
        test_mlp_scores = scores_to_numpy(test_mlp_scores)
        test_vae_pred = np.where(
            test_vae_scores >= vae_thr_malware,
            1,
            np.where(test_vae_scores <= vae_thr_benign, 0, -1),
        )

        test_mlp_pred = np.where(
            test_mlp_scores >= mlp_thr_malware,
            1,
            np.where(test_mlp_scores <= mlp_thr_benign, 0, -1),
        )
        test_labels = dataset_labels(data_distribution[2])

        final_test_pred = combine_predictions(
            test_vae_pred,
            test_mlp_pred,
        )

        report_test, cm_test, decided = evaluate_predictions(
            test_labels,
            final_test_pred,
        )

        print(
            f"Deferred on test: {(~decided).sum()}/{len(test_labels)} "
            f"({100*(~decided).sum()/len(test_labels):.2f}%)"
        )

    result_path = f"{result_path}{threshold}.csv"
    with open(result_path, "w") as f:
        f.write("VALIDATION REPORT\n")
        f.write(report_val)
        f.write("\nTEST REPORT\n")
        f.write(report_test)

    out_file_cm = (
        Path(result_path).parent / f"confusion_matrix_{threshold}_{predefined_loss}.csv"
    )
    pd.DataFrame(
        cm_test,
        index=["goodware", "malware"],
        columns=["pred_goodware", "pred_malware"],
    ).to_csv(out_file_cm)


if __name__ == "__main__":
    feature_dirs = ["/media/kaiba/Diegos/Drebin2.0/baselines/ramda/features"]

    samples = Path(__file__).resolve().parent.parent.parent.parent / "samples_hash.txt"
    empty = Path(__file__).resolve().parent.parent.parent.parent / "empty_hash.txt"
    samples_hash = set([line.strip() for line in islice(open(samples), 1, None)])
    empty_hash = set([line.strip().lower() for line in islice(open(empty), 1, None)])
    samples_hash = samples_hash - empty_hash
    settings_dir = "/media/kaiba/Diegos/Drebin2.0/settings"
    samples_df = pd.read_csv(Path(settings_dir) / "labels_vt.csv")

    train_samples, validation_samples, test_samples = split_train_val_test(
        samples_df, samples_hash
    )

    train_samples = pd.DataFrame(train_samples)
    validation_samples = pd.DataFrame(validation_samples)
    test_samples = pd.DataFrame(test_samples)

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
        (
            pd.concat([train_samples, validation_samples], ignore_index=True),
            test_samples,
        ),
        shuffle=False,
        sampling_mode="oversampling",
    )

    print("Training and testing the model, threshold=0.5, predefined_loss=20...")
    # train_and_test(
    #     feature_dirs=feature_dirs,
    #     data_distribution=(
    #         train_dataset,
    #         validation_dataset,
    #         test_dataset,
    #         train_val_dataset,
    #     ),
    #     train_flag="train",
    #     model_name="ramda_model",
    #     model_type="ramda",
    #     result_path=(
    #         Path(__file__).resolve().parent.parent.parent.parent
    #         / "results"
    #         / "RAMDA"
    #         / "ramda_classification_report_"
    #     ),
    #     logger=None,
    #     threshold=0.5,
    #     predefined_loss=20,
    #     in_channels=379,
    #     hidden_channels=600,
    #     out_channels=80,
    #     num_classes=2,
    #     dropout_ratio=0.1,
    #     device_id="cuda",
    #     epochs=20,
    #     batch_size=64,
    #     lr=10e-4,
    #     lambda_1=10,
    #     lambda_2=1,
    #     lambda_3=10,
    # )

    # print("Training and testing the model single threshold best f1...")
    # train_and_test(
    #     feature_dirs=feature_dirs,
    #     data_distribution=(
    #         train_dataset,
    #         validation_dataset,
    #         test_dataset,
    #         train_val_dataset,
    #     ),
    #     train_flag="train",
    #     model_name="ramda_model",
    #     model_type="ramda",
    #     result_path=(
    #         Path(__file__).resolve().parent.parent.parent.parent
    #         / "results"
    #         / "RAMDA"
    #         / "ramda_classification_report_"
    #     ),
    #     logger=None,
    #     threshold=None,
    #     predefined_loss=None,
    #     in_channels=379,
    #     hidden_channels=600,
    #     out_channels=80,
    #     num_classes=2,
    #     dropout_ratio=0.1,
    #     device_id="cuda",
    #     epochs=20,
    #     batch_size=64,
    #     lr=10e-4,
    #     lambda_1=10,
    #     lambda_2=1,
    #     lambda_3=10,
    # )

    print("Training and testing the model dual threshold best f1...")
    train_and_test(
        feature_dirs=feature_dirs,
        data_distribution=(
            train_dataset,
            validation_dataset,
            test_dataset,
            train_val_dataset,
        ),
        train_flag="train",
        model_name="ramda_model",
        model_type="ramda",
        result_path=(
            Path(__file__).resolve().parent.parent.parent.parent
            / "results"
            / "RAMDA"
            / "ramda_classification_report_"
        ),
        logger=None,
        threshold="dual",
        predefined_loss="dual",
        in_channels=379,
        hidden_channels=600,
        out_channels=80,
        num_classes=2,
        dropout_ratio=0.1,
        device_id="cuda",
        epochs=20,
        batch_size=64,
        lr=10e-4,
        lambda_1=10,
        lambda_2=1,
        lambda_3=10,
    )
