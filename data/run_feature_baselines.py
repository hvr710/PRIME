#!/usr/bin/env python3
"""Run classical baselines on extracted COMP4 handcrafted EEG features."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils._testing import ignore_warnings
from xgboost import XGBClassifier


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_RESULT_DIR = DATA_DIR / "feature_baseline_results"
DEFAULT_SPLIT_DIR = None
DEFAULT_SEEDS = [42, 3407, 2025]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass(frozen=True)
class Split:
    name: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--split-kinds", nargs="+", default=["subject", "segment"], choices=["subject", "segment"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def _require_metadata(handle: h5py.File) -> None:
    required = ["label", "subject_index", "trial_index", "segment_index"]
    missing = [name for name in required if name not in handle]
    if missing:
        raise KeyError(f"Feature H5 is missing metadata datasets: {missing}")


def load_feature_matrix(path: Path) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    feature_blocks = []
    feature_names = []

    with h5py.File(path, "r") as handle:
        _require_metadata(handle)
        channel_names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in handle["meta/channel_names"][:]
        ]

        def add_channel_band_group(group_name: str, prefix: str) -> None:
            for band in BANDS:
                data = handle[f"{group_name}/{band}"][:]
                feature_blocks.append(data)
                feature_names.extend([f"{prefix}_{band}_{ch}" for ch in channel_names])

        add_channel_band_group("absolute_power", "AP")
        add_channel_band_group("relative_power", "RP")
        add_channel_band_group("de_log_power", "DE")

        for ratio_name in ["theta_beta", "alpha_beta"]:
            data = handle[f"ratios/{ratio_name}"][:]
            feature_blocks.append(data)
            feature_names.extend([f"ratio_{ratio_name}_{ch}" for ch in channel_names])

        for hj_name in ["activity", "mobility", "complexity"]:
            data = handle[f"hjorth/{hj_name}"][:]
            feature_blocks.append(data)
            feature_names.extend([f"hjorth_{hj_name}_{ch}" for ch in channel_names])

        for name in ["spectral_entropy", "pfd", "std"]:
            data = handle[name][:]
            feature_blocks.append(data)
            feature_names.extend([f"{name}_{ch}" for ch in channel_names])

        for band in BANDS:
            data = handle[f"asymmetry/{band}"][:]
            pair_names = [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in handle[f"asymmetry/{band}_names"][:]
            ]
            feature_blocks.append(data)
            feature_names.extend([f"asym_{band}_{pair}" for pair in pair_names])

        feature_blocks.append(handle["faa/values"][:])
        faa_names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in handle["faa/names"][:]
        ]
        feature_names.extend([f"FAA_alpha_{name}" for name in faa_names])

        metadata = {
            "label": handle["label"][:].astype(np.int64),
            "subject_index": handle["subject_index"][:].astype(np.int64),
            "trial_index": handle["trial_index"][:].astype(np.int64),
            "segment_index": handle["segment_index"][:].astype(np.int64),
        }

    X = np.concatenate([np.asarray(block, dtype=np.float32) for block in feature_blocks], axis=1)
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names, metadata


def _split_counts(indices: np.ndarray, y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y[indices], return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _save_split_json(split: Split, split_kind: str, y: np.ndarray, result_dir: Path) -> Path:
    split_dir = result_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"comp4_{split_kind}_seed{split.seed}.json"
    payload = {
        "split_kind": split_kind,
        "seed": int(split.seed),
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "n_train": int(split.train.shape[0]),
        "n_val": int(split.val.shape[0]),
        "n_test": int(split.test.shape[0]),
        "label_counts": {
            "train": _split_counts(split.train, y),
            "val": _split_counts(split.val, y),
            "test": _split_counts(split.test, y),
        },
        "train_indices": split.train.astype(int).tolist(),
        "val_indices": split.val.astype(int).tolist(),
        "test_indices": split.test.astype(int).tolist(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _load_split_json(split_dir: Path, split_kind: str, seed: int) -> Split:
    path = split_dir / f"comp4_{split_kind}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Split(
        name=f"{split_kind}_seed{seed}",
        seed=int(payload["seed"]),
        train=np.asarray(payload["train_indices"], dtype=np.int64),
        val=np.asarray(payload["val_indices"], dtype=np.int64),
        test=np.asarray(payload["test_indices"], dtype=np.int64),
    )


def make_subject_split(subject_index: np.ndarray, seed: int, val_ratio: float, test_ratio: float) -> Split:
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_index)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_subjects = len(shuffled)
    n_test = int(round(n_subjects * test_ratio))
    n_val = int(round(n_subjects * val_ratio))
    n_train = n_subjects - n_val - n_test
    train_subjects = shuffled[:n_train]
    val_subjects = shuffled[n_train:n_train + n_val]
    test_subjects = shuffled[n_train + n_val:]

    return Split(
        name=f"subject_seed{seed}",
        seed=seed,
        train=np.flatnonzero(np.isin(subject_index, train_subjects)).astype(np.int64),
        val=np.flatnonzero(np.isin(subject_index, val_subjects)).astype(np.int64),
        test=np.flatnonzero(np.isin(subject_index, test_subjects)).astype(np.int64),
    )


def make_segment_split(y: np.ndarray, seed: int, val_ratio: float, test_ratio: float) -> Split:
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_total = len(idx)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_val - n_test
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:])

    return Split(
        name=f"segment_seed{seed}",
        seed=seed,
        train=np.sort(np.concatenate(train_parts)).astype(np.int64),
        val=np.sort(np.concatenate(val_parts)).astype(np.int64),
        test=np.sort(np.concatenate(test_parts)).astype(np.int64),
    )


def build_model(model_name: str, seed: int, n_jobs: int):
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
                class_weight=None,
            ),
        )
    if model_name == "svm":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="hinge",
                alpha=1e-4,
                max_iter=3000,
                tol=1e-3,
                random_state=seed,
                n_jobs=n_jobs,
            ),
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
        )
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(256, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=False,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def decision_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        if hasattr(final_estimator, "predict_proba"):
            return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(np.float32)


def compute_metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    pred = model.predict(X)
    score = decision_scores(model, X)
    return {
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "n_samples": int(y.shape[0]),
    }


@ignore_warnings(category=ConvergenceWarning)
def fit_model(model, X_train: np.ndarray, y_train: np.ndarray):
    model.fit(X_train, y_train)
    return model


def summarize(rows: list[dict], result_dir: Path) -> None:
    summary = {}
    for split_kind in sorted({row["split_kind"] for row in rows}):
        summary[split_kind] = {}
        for model_name in sorted({row["model"] for row in rows}):
            selected = [row for row in rows if row["split_kind"] == split_kind and row["model"] == model_name]
            if not selected:
                continue
            summary[split_kind][model_name] = {}
            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                values = np.asarray([row[f"test_{metric}"] for row in selected], dtype=np.float64)
                summary[split_kind][model_name][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "split_kind",
        "seed",
        "model",
        "train_seconds",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auroc",
        "val_auprc",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auroc",
        "test_auprc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    X, feature_names, meta = load_feature_matrix(args.feature_h5)
    y = meta["label"]
    subject_index = meta["subject_index"]

    with (args.result_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2, ensure_ascii=False)

    rows = []
    model_names = ["logistic_regression", "svm", "xgboost", "mlp"]
    split_builders = {}
    if args.split_dir is None:
        split_builders = {
            "subject": lambda seed: make_subject_split(subject_index, seed, args.val_ratio, args.test_ratio),
            "segment": lambda seed: make_segment_split(y, seed, args.val_ratio, args.test_ratio),
        }
    else:
        split_builders = {
            split_kind: (lambda seed, split_kind=split_kind: _load_split_json(args.split_dir, split_kind, seed))
            for split_kind in args.split_kinds
        }

    print(f"Loaded features: X={X.shape}, y={y.shape}, seeds={args.seeds}")
    for split_kind in args.split_kinds:
        split_builder = split_builders[split_kind]
        for seed in args.seeds:
            split = split_builder(seed)
            if args.split_dir is None:
                split_path = _save_split_json(split, split_kind, y, args.result_dir)
            else:
                split_path = args.split_dir / f"comp4_{split_kind}_seed{seed}.json"
            print(f"\n=== split={split_kind} seed={seed} split_json={split_path} ===")
            print(f"train/val/test = {len(split.train)}/{len(split.val)}/{len(split.test)}")
            for model_name in model_names:
                model = build_model(model_name, seed, args.n_jobs)
                t0 = perf_counter()
                print(f"Training {model_name} ...", flush=True)
                fit_model(model, X[split.train], y[split.train])
                train_seconds = perf_counter() - t0

                val_metrics = compute_metrics(model, X[split.val], y[split.val])
                test_metrics = compute_metrics(model, X[split.test], y[split.test])
                row = {
                    "split_kind": split_kind,
                    "seed": int(seed),
                    "model": model_name,
                    "train_seconds": round(float(train_seconds), 4),
                }
                for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
                    for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                        row[f"{prefix}_{metric}"] = metrics[metric]
                rows.append(row)
                print(
                    f"{model_name}: val_acc={val_metrics['acc']*100:.2f}, "
                    f"test_acc={test_metrics['acc']*100:.2f}, "
                    f"test_auroc={test_metrics['auroc']*100:.2f}, "
                    f"time={train_seconds:.1f}s"
                )
                write_csv(rows, args.result_dir / "all_results.csv")
                summarize(rows, args.result_dir)

    with (args.result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_h5": str(args.feature_h5),
                "feature_shape": list(X.shape),
                "split_dir": None if args.split_dir is None else str(args.split_dir),
                "seeds": args.seeds,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "models": model_names,
                "split_kinds": args.split_kinds,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved results to: {args.result_dir}")


if __name__ == "__main__":
    main()
