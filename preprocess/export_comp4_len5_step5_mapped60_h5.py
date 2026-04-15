#!/usr/bin/env python3
"""Export COMP4 5s/5s segments after mdJPT 60-channel projection to HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


DEFAULT_PROCESSED_ROOT = Path(
    "/vePFS-0x0d/home/cx/cx_old/hw/project/project/赛题四数据集及说明文档/"
    "mdjpt_comp4_125hz/processed_data"
)
DEFAULT_SPLIT_JSON = Path(
    "/vePFS-0x0d/home/cx/cx_old/hw/project/project/赛题四数据集及说明文档/"
    "mdjpt_comp4_125hz/split_subjects_seed42.json"
)
DEFAULT_COMP4_CFG = REPO_ROOT / "cfgs_multi" / "data" / "COMP4.yaml"
DEFAULT_MAIN_CFG = REPO_ROOT / "cfgs_multi" / "config_multi.yaml"
DEFAULT_INTERPOLATE_NPY = REPO_ROOT / "channel_interpolate.npy"
DEFAULT_OUTPUT_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"

SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export COMP4 5s/5s EEG segments after 60-channel mdJPT mapping."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Directory containing resampled COMP4 processed .mat files.",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=DEFAULT_SPLIT_JSON,
        help="Subject split json generated during preprocessing.",
    )
    parser.add_argument(
        "--comp4-cfg",
        type=Path,
        default=DEFAULT_COMP4_CFG,
        help="COMP4 data config containing source channels and fs.",
    )
    parser.add_argument(
        "--main-cfg",
        type=Path,
        default=DEFAULT_MAIN_CFG,
        help="Main config containing mdJPT standard 60 channels.",
    )
    parser.add_argument(
        "--channel-interpolate",
        type=Path,
        default=DEFAULT_INTERPOLATE_NPY,
        help="Nearest-neighbor index table used by mdJPT channel projection.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=DEFAULT_OUTPUT_H5,
        help="Destination HDF5 path.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=5,
        help="Segment length in seconds.",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=5,
        help="Sliding window step in seconds.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="HDF5 compression for EEG dataset.",
    )
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=4,
        help="Compression level for EEG dataset.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def robust_zscore(eeg_data: np.ndarray) -> np.ndarray:
    thr = 30 * np.median(np.abs(eeg_data))
    valid_mask = np.abs(eeg_data) < thr
    valid_values = eeg_data[valid_mask]
    if valid_values.size == 0:
        mean = float(np.mean(eeg_data))
        std = float(np.std(eeg_data))
    else:
        mean = float(np.mean(valid_values))
        std = float(np.std(valid_values))
    if std < 1e-8:
        std = 1.0
    return ((eeg_data - mean) / std).astype(np.float32, copy=False)


def channel_project_numpy(
    data: np.ndarray,
    source_channels: list[str],
    standard_channels: list[str],
    channel_interpolate: np.ndarray,
) -> np.ndarray:
    """Mirror MultiModel_PL.channel_project with NumPy.

    Args:
        data: [segments, source_channels, timepoints]
    Returns:
        [segments, standard_channels, timepoints]
    """
    source_map = {name.upper(): idx for idx, name in enumerate(source_channels)}
    n_segments, _, n_timepoints = data.shape
    projected = np.zeros((n_segments, len(standard_channels), n_timepoints), dtype=np.float32)

    for std_idx, std_name in enumerate(standard_channels):
        std_name_upper = std_name.upper()
        if std_name_upper in source_map:
            projected[:, std_idx, :] = data[:, source_map[std_name_upper], :]
            continue

        valid_source_indices: list[int] = []
        for neighbor_std_idx in channel_interpolate[std_idx]:
            neighbor_name = standard_channels[int(neighbor_std_idx)].upper()
            if neighbor_name in source_map:
                valid_source_indices.append(source_map[neighbor_name])
                if len(valid_source_indices) == 3:
                    break

        if valid_source_indices:
            projected[:, std_idx, :] = data[:, valid_source_indices, :].mean(axis=1)

    return projected


def build_segments(
    eeg_data: np.ndarray,
    trial_seconds: np.ndarray,
    fs: int,
    segment_seconds: int,
    step_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_len = int(segment_seconds * fs)
    points_step = int(step_seconds * fs)
    n_points = trial_seconds.astype(int) * fs
    samples_per_trial = ((n_points - points_len) // points_step + 1).astype(np.int64)
    total_segments = int(np.sum(samples_per_trial))

    segments = np.empty((total_segments, eeg_data.shape[0], points_len), dtype=np.float32)
    segment_trial_index = np.empty((total_segments,), dtype=np.int64)

    trial_offsets = np.concatenate(([0], np.cumsum(n_points)))
    cursor = 0
    for trial_index, trial_samples in enumerate(samples_per_trial):
        trial_start = trial_offsets[trial_index]
        for seg_index in range(int(trial_samples)):
            start = trial_start + seg_index * points_step
            stop = start + points_len
            segments[cursor] = eeg_data[:, start:stop]
            segment_trial_index[cursor] = trial_index
            cursor += 1

    return segments, samples_per_trial


def create_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def main() -> None:
    args = parse_args()
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)

    comp4_cfg = load_yaml(args.comp4_cfg)
    main_cfg = load_yaml(args.main_cfg)
    channel_interpolate = np.load(args.channel_interpolate)

    source_channels = list(comp4_cfg["channels"])
    standard_channels = list(main_cfg["model"]["MLLA"]["uni_channels"])
    fs = int(comp4_cfg["fs"])

    with args.split_json.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)

    subject_order = split_info["subject_order"]
    subject_groups = split_info["subject_groups"]
    split_subjects = split_info["split_subjects"]
    train_indices = np.asarray(split_info["train_indices"], dtype=np.int64)
    val_indices = np.asarray(split_info["val_indices"], dtype=np.int64)
    test_indices = np.asarray(split_info["test_indices"], dtype=np.int64)
    source_meta = split_info["source"]

    segment_seconds = int(args.segment_seconds)
    step_seconds = int(args.step_seconds)
    segment_points = segment_seconds * fs

    labels_per_trial = np.asarray(source_meta["labels_per_subject"], dtype=np.int64)
    trials_per_subject = int(source_meta["trials_per_subject"])
    trial_seconds_ref = np.asarray([source_meta["trial_seconds"]] * trials_per_subject, dtype=np.int64)
    segments_per_trial = ((trial_seconds_ref * fs - segment_points) // (step_seconds * fs) + 1).astype(np.int64)
    segments_per_subject = int(np.sum(segments_per_trial))
    total_segments = len(subject_order) * segments_per_subject

    split_by_subject = {}
    for split_name, subjects in split_subjects.items():
        for subject_name in subjects:
            split_by_subject[subject_name] = split_name

    with h5py.File(args.output_h5, "w") as handle:
        eeg_ds = handle.create_dataset(
            "eeg",
            shape=(total_segments, len(standard_channels), segment_points),
            dtype=np.float32,
            compression=args.compression,
            compression_opts=args.compression_opts,
            chunks=(16, len(standard_channels), segment_points),
        )
        label_ds = handle.create_dataset("label", shape=(total_segments,), dtype=np.int64)
        split_ds = handle.create_dataset("split", shape=(total_segments,), dtype=np.int8)
        subject_index_ds = handle.create_dataset("subject_index", shape=(total_segments,), dtype=np.int64)
        trial_index_ds = handle.create_dataset("trial_index", shape=(total_segments,), dtype=np.int64)
        segment_index_ds = handle.create_dataset("segment_index", shape=(total_segments,), dtype=np.int64)
        segment_start_sample_ds = handle.create_dataset(
            "segment_start_sample", shape=(total_segments,), dtype=np.int64
        )
        segment_start_second_ds = handle.create_dataset(
            "segment_start_second", shape=(total_segments,), dtype=np.float32
        )
        global_trial_index_ds = handle.create_dataset(
            "global_trial_index", shape=(total_segments,), dtype=np.int64
        )

        meta_group = handle.create_group("meta")
        create_string_dataset(meta_group, "source_channel_names", source_channels)
        create_string_dataset(meta_group, "mapped_channel_names", standard_channels)
        create_string_dataset(meta_group, "subject_names", subject_order)
        create_string_dataset(
            meta_group, "subject_groups", [subject_groups[subject_name] for subject_name in subject_order]
        )
        create_string_dataset(meta_group, "split_names", ["train", "val", "test"])
        meta_group.create_dataset("train_subject_indices", data=train_indices)
        meta_group.create_dataset("val_subject_indices", data=val_indices)
        meta_group.create_dataset("test_subject_indices", data=test_indices)
        meta_group.create_dataset("labels_per_trial", data=labels_per_trial)
        meta_group.create_dataset("trial_seconds", data=trial_seconds_ref)
        meta_group.create_dataset("segments_per_trial", data=segments_per_trial)
        meta_group.create_dataset(
            "segment_points_per_trial",
            data=np.full((trials_per_subject,), segment_points, dtype=np.int64),
        )

        handle.attrs["dataset_name"] = "COMP4"
        handle.attrs["fs"] = fs
        handle.attrs["segment_seconds"] = segment_seconds
        handle.attrs["segment_step_seconds"] = step_seconds
        handle.attrs["segment_points"] = segment_points
        handle.attrs["n_subjects"] = len(subject_order)
        handle.attrs["n_trials_per_subject"] = trials_per_subject
        handle.attrs["n_segments_per_subject"] = segments_per_subject
        handle.attrs["source_channel_count"] = len(source_channels)
        handle.attrs["mapped_channel_count"] = len(standard_channels)
        handle.attrs["source_processed_root"] = str(args.processed_root)
        handle.attrs["split_json"] = str(args.split_json)
        handle.attrs["comp4_cfg"] = str(args.comp4_cfg)
        handle.attrs["main_cfg"] = str(args.main_cfg)
        handle.attrs["channel_interpolate"] = str(args.channel_interpolate)
        handle.attrs["projection_rule"] = (
            "Same-name channels are copied; missing standard channels are filled by "
            "averaging up to 3 nearest available standard-channel neighbors; otherwise zero."
        )
        handle.attrs["split_codebook"] = json.dumps(SPLIT_CODES, ensure_ascii=False)

        cursor = 0
        points_step = step_seconds * fs
        for subject_index, subject_name in enumerate(subject_order):
            mat_path = args.processed_root / f"{subject_name}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(f"Missing processed .mat file: {mat_path}")

            mat = sio.loadmat(mat_path)
            eeg_data = robust_zscore(mat["merged_data_all_cleaned"].astype(np.float32))
            trial_seconds = np.squeeze(mat["merged_n_samples_one"]).astype(np.int64)
            if trial_seconds.shape[0] != trials_per_subject:
                raise ValueError(
                    f"{subject_name} expected {trials_per_subject} trials, got {trial_seconds.shape[0]}"
                )

            segments, samples_per_trial = build_segments(
                eeg_data=eeg_data,
                trial_seconds=trial_seconds,
                fs=fs,
                segment_seconds=segment_seconds,
                step_seconds=step_seconds,
            )
            projected_segments = channel_project_numpy(
                data=segments,
                source_channels=source_channels,
                standard_channels=standard_channels,
                channel_interpolate=channel_interpolate,
            )

            if not np.array_equal(samples_per_trial, segments_per_trial):
                raise ValueError(
                    f"{subject_name} segments_per_trial mismatch: "
                    f"{samples_per_trial.tolist()} vs {segments_per_trial.tolist()}"
                )

            subject_split_name = split_by_subject[subject_name]
            subject_split_code = SPLIT_CODES[subject_split_name]

            local_trial_indices = np.repeat(np.arange(trials_per_subject, dtype=np.int64), segments_per_trial)
            local_segment_indices = np.concatenate(
                [np.arange(count, dtype=np.int64) for count in segments_per_trial]
            )
            local_labels = np.repeat(labels_per_trial, segments_per_trial)
            local_segment_start_samples = local_segment_indices * points_step
            local_segment_start_seconds = local_segment_start_samples.astype(np.float32) / float(fs)
            local_global_trial_indices = subject_index * trials_per_subject + local_trial_indices

            next_cursor = cursor + projected_segments.shape[0]
            eeg_ds[cursor:next_cursor] = projected_segments
            label_ds[cursor:next_cursor] = local_labels
            split_ds[cursor:next_cursor] = subject_split_code
            subject_index_ds[cursor:next_cursor] = subject_index
            trial_index_ds[cursor:next_cursor] = local_trial_indices
            segment_index_ds[cursor:next_cursor] = local_segment_indices
            segment_start_sample_ds[cursor:next_cursor] = local_segment_start_samples
            segment_start_second_ds[cursor:next_cursor] = local_segment_start_seconds
            global_trial_index_ds[cursor:next_cursor] = local_global_trial_indices

            cursor = next_cursor
            print(
                f"[{subject_index + 1:02d}/{len(subject_order):02d}] "
                f"{subject_name}: wrote {projected_segments.shape[0]} segments to split={subject_split_name}"
            )

        if cursor != total_segments:
            raise RuntimeError(f"Wrote {cursor} segments, expected {total_segments}")

    print(f"Saved HDF5 to: {args.output_h5}")
    print(f"EEG dataset shape: ({total_segments}, {len(standard_channels)}, {segment_points})")


if __name__ == "__main__":
    main()
