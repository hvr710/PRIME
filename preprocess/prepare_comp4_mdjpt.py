import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly


ORIGINAL_FS = 250
TARGET_FS = 125
TRIALS_PER_CLASS = 4
TRIAL_SECONDS = 50
POINTS_PER_TRIAL = ORIGINAL_FS * TRIAL_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert competition track-4 training EEG data to mdJPT processed_data format"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/vePFS-0x0d/home/cx/cx/hw/project/project/赛题四数据集及说明文档/训练集"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/vePFS-0x0d/home/cx/cx/hw/project/project/赛题四数据集及说明文档/mdjpt_comp4_125hz"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def _subject_sort_key(name: str) -> Tuple[int, int, str]:
    upper_name = name.upper()
    if upper_name.startswith("HC"):
        group_rank = 0
    elif upper_name.startswith("DEP"):
        group_rank = 1
    else:
        group_rank = 2

    match = re.search(r"(\d+)", upper_name)
    subject_num = int(match.group(1)) if match else 10**9
    return group_rank, subject_num, upper_name


def _collect_subject_files(input_root: Path) -> List[Tuple[str, str, Path]]:
    group_dirs = [("HC", input_root / "正常人"), ("DEP", input_root / "抑郁症患者")]
    subjects: List[Tuple[str, str, Path]] = []

    for group_name, group_dir in group_dirs:
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")
        for mat_path in group_dir.glob("*.mat"):
            subjects.append((mat_path.stem, group_name, mat_path))

    if not subjects:
        raise FileNotFoundError(f"No .mat files found under {input_root}")

    subjects.sort(key=lambda item: _subject_sort_key(item[0]))
    return subjects


def _load_block(file_path: Path, key: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        if key not in f:
            raise KeyError(f"Missing key {key} in {file_path}")
        arr = np.asarray(f[key])

    if arr.shape != (POINTS_PER_TRIAL * TRIALS_PER_CLASS, 30):
        raise ValueError(f"Unexpected shape for {file_path}:{key}: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _split_trials(block_tc: np.ndarray) -> List[np.ndarray]:
    return [
        block_tc[i * POINTS_PER_TRIAL : (i + 1) * POINTS_PER_TRIAL]
        for i in range(TRIALS_PER_CLASS)
    ]


def _resample_trial(trial_tc: np.ndarray) -> np.ndarray:
    # Input: [time, channel], output: [time, channel]
    return resample_poly(trial_tc, up=TARGET_FS, down=ORIGINAL_FS, axis=0).astype(
        np.float32, copy=False
    )


def _convert_subject(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    neu_block = _load_block(file_path, "EEG_data_neu")
    pos_block = _load_block(file_path, "EEG_data_pos")

    trials = _split_trials(neu_block) + _split_trials(pos_block)
    resampled_trials = [_resample_trial(trial_tc) for trial_tc in trials]

    expected_points = TARGET_FS * TRIAL_SECONDS
    for idx, trial_tc in enumerate(resampled_trials):
        if trial_tc.shape != (expected_points, 30):
            raise ValueError(
                f"Resampled trial shape mismatch for {file_path}, trial {idx}: {trial_tc.shape}"
            )

    merged_tc = np.concatenate(resampled_trials, axis=0)
    merged_ct = merged_tc.T
    merged_n_samples_one = np.asarray([[TRIAL_SECONDS] * len(resampled_trials)], dtype=np.int32)
    return merged_ct, merged_n_samples_one


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _split_group_subjects(
    subject_names: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    subject_names = list(subject_names)
    shuffled = list(subject_names)
    rng.shuffle(shuffled)

    n_subjects = len(shuffled)
    n_val = int(round(n_subjects * val_ratio))
    n_test = int(round(n_subjects * test_ratio))
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1

    if n_val + n_test >= n_subjects:
        while n_val + n_test >= n_subjects and n_test > 1:
            n_test -= 1
        while n_val + n_test >= n_subjects and n_val > 1:
            n_val -= 1

    test_subjects = shuffled[:n_test]
    val_subjects = shuffled[n_test : n_test + n_val]
    train_subjects = shuffled[n_test + n_val :]

    return {
        "train": sorted(train_subjects, key=_subject_sort_key),
        "val": sorted(val_subjects, key=_subject_sort_key),
        "test": sorted(test_subjects, key=_subject_sort_key),
    }


def _build_split(subject_order: Sequence[str], groups: Dict[str, str], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, object]:
    hc_subjects = [name for name in subject_order if groups[name] == "HC"]
    dep_subjects = [name for name in subject_order if groups[name] == "DEP"]

    hc_split = _split_group_subjects(hc_subjects, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    dep_split = _split_group_subjects(dep_subjects, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed + 1)

    split_subjects = {
        split_name: sorted(
            hc_split[split_name] + dep_split[split_name],
            key=_subject_sort_key,
        )
        for split_name in ("train", "val", "test")
    }

    index_map = {name: idx for idx, name in enumerate(subject_order)}
    split_indices = {
        f"{split_name}_indices": [index_map[name] for name in split_subjects[split_name]]
        for split_name in ("train", "val", "test")
    }

    split_groups = {
        split_name: {
            "HC": sum(1 for name in split_subjects[split_name] if groups[name] == "HC"),
            "DEP": sum(1 for name in split_subjects[split_name] if groups[name] == "DEP"),
        }
        for split_name in ("train", "val", "test")
    }

    return {
        "seed": seed,
        "subject_order": list(subject_order),
        "subject_groups": groups,
        "split_subjects": split_subjects,
        "split_groups": split_groups,
        **split_indices,
    }


def main() -> None:
    args = parse_args()
    processed_dir = args.output_root / "processed_data"
    _ensure_dir(processed_dir)

    subjects = _collect_subject_files(args.input_root)
    subject_order = [subject_name for subject_name, _group_name, _path in subjects]
    subject_groups = {subject_name: group_name for subject_name, group_name, _path in subjects}

    for subject_name, _group_name, file_path in subjects:
        merged_ct, merged_n_samples_one = _convert_subject(file_path)
        save_path = processed_dir / f"{subject_name}.mat"
        savemat(
            save_path,
            {
                "merged_data_all_cleaned": merged_ct,
                "merged_n_samples_one": merged_n_samples_one,
            },
            do_compression=True,
        )
        print(f"saved {save_path}")

    split_payload = _build_split(
        subject_order=subject_order,
        groups=subject_groups,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_payload["source"] = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "original_fs": ORIGINAL_FS,
        "target_fs": TARGET_FS,
        "trial_seconds": TRIAL_SECONDS,
        "trials_per_subject": 8,
        "labels_per_subject": [0, 0, 0, 0, 1, 1, 1, 1],
    }

    split_path = args.output_root / f"split_subjects_seed{args.seed}.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2, ensure_ascii=False)

    print(f"saved {split_path}")
    print(
        "subject split train/val/test = "
        f"{len(split_payload['split_subjects']['train'])}/"
        f"{len(split_payload['split_subjects']['val'])}/"
        f"{len(split_payload['split_subjects']['test'])}"
    )


if __name__ == "__main__":
    main()
