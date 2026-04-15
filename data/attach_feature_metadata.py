#!/usr/bin/env python3
"""Attach labels and index metadata from EEG H5 to extracted feature H5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_EEG_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"

COPY_DATASETS = [
    "label",
    "split",
    "subject_index",
    "trial_index",
    "segment_index",
    "global_trial_index",
    "segment_start_sample",
    "segment_start_second",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-h5", type=Path, default=DEFAULT_EEG_H5)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    return parser.parse_args()


def _copy_dataset(src: h5py.File, dst: h5py.File, name: str) -> None:
    if name not in src:
        return
    if name in dst:
        del dst[name]
    dst.create_dataset(name, data=src[name][:], dtype=src[name].dtype)


def _copy_meta_dataset(src: h5py.File, dst: h5py.File, src_name: str, dst_name: str) -> None:
    if src_name not in src:
        return
    if "meta" not in dst:
        dst.create_group("meta")
    if dst_name in dst["meta"]:
        del dst["meta"][dst_name]
    data = src[src_name][:]
    dst["meta"].create_dataset(dst_name, data=data, dtype=src[src_name].dtype)


def main() -> None:
    args = parse_args()
    if not args.eeg_h5.exists():
        raise FileNotFoundError(args.eeg_h5)
    if not args.feature_h5.exists():
        raise FileNotFoundError(args.feature_h5)

    with h5py.File(args.eeg_h5, "r") as src, h5py.File(args.feature_h5, "a") as dst:
        n_segments = int(dst["absolute_power/delta"].shape[0])
        if int(src["label"].shape[0]) != n_segments:
            raise ValueError(
                f"Segment count mismatch: feature_h5={n_segments}, eeg_h5={src['label'].shape[0]}"
            )

        for name in COPY_DATASETS:
            _copy_dataset(src, dst, name)

        _copy_meta_dataset(src, dst, "meta/subject_names", "subject_names")
        _copy_meta_dataset(src, dst, "meta/subject_groups", "subject_groups")
        _copy_meta_dataset(src, dst, "meta/split_names", "split_names")

        dst.attrs["metadata_source_h5"] = str(args.eeg_h5)
        dst.attrs["metadata_attached"] = True
        dst.attrs["split_codebook"] = src.attrs.get("split_codebook", '{"train": 0, "val": 1, "test": 2}')
        dst.attrs["fs"] = src.attrs.get("fs", 125)
        dst.attrs["segment_seconds"] = src.attrs.get("segment_seconds", 5)
        dst.attrs["segment_step_seconds"] = src.attrs.get("segment_step_seconds", 5)
        dst.attrs["segment_points"] = src.attrs.get("segment_points", 625)

        labels, label_counts = np.unique(dst["label"][:], return_counts=True)
        splits, split_counts = np.unique(dst["split"][:], return_counts=True)
        print(f"Attached metadata to: {args.feature_h5}")
        print("label counts:", dict(zip(labels.astype(int).tolist(), label_counts.astype(int).tolist())))
        print("split counts:", dict(zip(splits.astype(int).tolist(), split_counts.astype(int).tolist())))


if __name__ == "__main__":
    main()
