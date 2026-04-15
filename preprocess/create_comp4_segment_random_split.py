#!/usr/bin/env python3
"""Create a stratified random segment split for COMP4 5s/5s segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_OUTPUT = DATA_DIR / "comp4_len5_step5_segment_random_split_seed42.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a random segment-level split for COMP4.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _stratified_split_indices(labels: np.ndarray, seed: int, val_ratio: float, test_ratio: float):
    rng = np.random.default_rng(seed)
    train_indices, val_indices, test_indices = [], [], []
    split_label_counts = {"train": {}, "val": {}, "test": {}}

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)

        n_total = len(label_indices)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_val - n_test

        train = label_indices[:n_train]
        val = label_indices[n_train:n_train + n_val]
        test = label_indices[n_train + n_val:]

        train_indices.append(train)
        val_indices.append(val)
        test_indices.append(test)

        split_label_counts["train"][int(label)] = int(len(train))
        split_label_counts["val"][int(label)] = int(len(val))
        split_label_counts["test"][int(label)] = int(len(test))

    train_indices = np.sort(np.concatenate(train_indices)).astype(np.int64)
    val_indices = np.sort(np.concatenate(val_indices)).astype(np.int64)
    test_indices = np.sort(np.concatenate(test_indices)).astype(np.int64)
    return train_indices, val_indices, test_indices, split_label_counts


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5_path, "r") as handle:
        labels = handle["label"][:].astype(np.int64)
        subject_index = handle["subject_index"][:].astype(np.int64)
        trial_index = handle["trial_index"][:].astype(np.int64)
        segment_index = handle["segment_index"][:].astype(np.int64)

    train_indices, val_indices, test_indices, split_label_counts = _stratified_split_indices(
        labels=labels,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    payload = {
        "split_kind": "segment_random",
        "split_name": f"segment_random_seed{args.seed}",
        "source_h5": str(args.h5_path),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "n_segments": int(labels.shape[0]),
        "label_counts": {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))},
        "split_counts": {
            "train": int(train_indices.shape[0]),
            "val": int(val_indices.shape[0]),
            "test": int(test_indices.shape[0]),
        },
        "split_label_counts": split_label_counts,
        "train_segment_indices": train_indices.tolist(),
        "val_segment_indices": val_indices.tolist(),
        "test_segment_indices": test_indices.tolist(),
        "train_subject_count": int(np.unique(subject_index[train_indices]).shape[0]),
        "val_subject_count": int(np.unique(subject_index[val_indices]).shape[0]),
        "test_subject_count": int(np.unique(subject_index[test_indices]).shape[0]),
        "train_trial_count": int(np.unique(subject_index[train_indices] * 100 + trial_index[train_indices]).shape[0]),
        "val_trial_count": int(np.unique(subject_index[val_indices] * 100 + trial_index[val_indices]).shape[0]),
        "test_trial_count": int(np.unique(subject_index[test_indices] * 100 + trial_index[test_indices]).shape[0]),
        "note": (
            "This is a segment-level random split. Segments from the same subject or trial can appear in "
            "different splits, so its metrics are not directly comparable to subject-level splits."
        ),
    }

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Saved segment split to: {args.output_json}")
    print(json.dumps(payload["split_counts"], indent=2, ensure_ascii=False))
    print(json.dumps(payload["split_label_counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
