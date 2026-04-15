#!/usr/bin/env python3
"""Create balanced subject-level COMP4 splits with HC/DEP present in every fold."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_OUTPUT_ROOT = DATA_DIR.parent / "outputs"
DEFAULT_SEEDS = [42, 3407, 2025]
PROTOCOLS = ["subject_811_balanced", "subject_80_20_innerval_balanced", "subject_80_20_teststop_balanced"]


@dataclass(frozen=True)
class SubjectMeta:
    subject_names: list[str]
    subject_groups: list[str]
    subject_index: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def load_subject_meta(path: Path) -> SubjectMeta:
    with h5py.File(path, "r") as handle:
        return SubjectMeta(
            subject_names=decode_strings(handle["meta/subject_names"][:]),
            subject_groups=decode_strings(handle["meta/subject_groups"][:]),
            subject_index=handle["subject_index"][:].astype(np.int64),
            labels=handle["label"][:].astype(np.int64),
        )


def group_subject_ids(meta: SubjectMeta) -> dict[str, np.ndarray]:
    grouped = {"HC": [], "DEP": []}
    for subject_id, group_name in enumerate(meta.subject_groups):
        upper = group_name.upper()
        if upper.startswith("HC"):
            grouped["HC"].append(subject_id)
        elif upper.startswith("DEP"):
            grouped["DEP"].append(subject_id)
        else:
            raise ValueError(f"Unexpected subject group {group_name!r} for subject {meta.subject_names[subject_id]}")
    return {key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()}


def shuffled_subjects(subject_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)
    return shuffled


def protocol_subject_counts(protocol: str) -> dict[str, dict[str, int]]:
    if protocol == "subject_811_balanced":
        return {
            "HC": {"train": 32, "val": 4, "test": 4},
            "DEP": {"train": 16, "val": 2, "test": 2},
        }
    if protocol == "subject_80_20_innerval_balanced":
        return {
            "HC": {"train": 28, "val": 4, "test": 8, "outer_train": 32},
            "DEP": {"train": 14, "val": 2, "test": 4, "outer_train": 16},
        }
    if protocol == "subject_80_20_teststop_balanced":
        return {
            "HC": {"train": 32, "test": 8},
            "DEP": {"train": 16, "test": 4},
        }
    raise ValueError(f"Unknown protocol: {protocol}")


def build_subject_split(meta: SubjectMeta, protocol: str, seed: int) -> dict[str, object]:
    grouped = group_subject_ids(meta)
    counts = protocol_subject_counts(protocol)
    split_subjects = {"train": [], "val": [], "test": []}
    outer_train_subjects: list[int] = []

    for group_name in ["HC", "DEP"]:
        shuffled = shuffled_subjects(grouped[group_name], seed if group_name == "HC" else seed + 1)
        expected = sum(counts[group_name][key] for key in counts[group_name] if key != "outer_train")
        if expected != len(shuffled):
            raise ValueError(
                f"Protocol {protocol} expected {expected} {group_name} subjects, but found {len(shuffled)}"
            )

        n_test = counts[group_name]["test"]
        if protocol == "subject_811_balanced":
            n_val = counts[group_name]["val"]
            test_ids = shuffled[:n_test]
            val_ids = shuffled[n_test:n_test + n_val]
            train_ids = shuffled[n_test + n_val:]
            outer_train_ids = np.concatenate([train_ids, val_ids])
        elif protocol == "subject_80_20_teststop_balanced":
            test_ids = shuffled[:n_test]
            train_ids = shuffled[n_test:]
            val_ids = test_ids.copy()
            outer_train_ids = train_ids.copy()
        else:
            n_outer_train = counts[group_name]["outer_train"]
            n_val = counts[group_name]["val"]
            outer_train_ids = shuffled[:n_outer_train]
            test_ids = shuffled[n_outer_train:]
            val_ids = outer_train_ids[:n_val]
            train_ids = outer_train_ids[n_val:]

        split_subjects["train"].extend(train_ids.tolist())
        split_subjects["val"].extend(val_ids.tolist())
        split_subjects["test"].extend(test_ids.tolist())
        outer_train_subjects.extend(outer_train_ids.tolist())

    for split_name in split_subjects:
        split_subjects[split_name] = sorted(split_subjects[split_name])
    outer_train_subjects = sorted(outer_train_subjects)

    def segment_indices(subject_ids: list[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(meta.subject_index, np.asarray(subject_ids, dtype=np.int64))).astype(np.int64)

    payload = {
        "split_kind": "subject",
        "protocol": protocol,
        "seed": int(seed),
        "train_subject_indices": split_subjects["train"],
        "val_subject_indices": split_subjects["val"],
        "test_subject_indices": split_subjects["test"],
        "official_outer_train_subject_indices": outer_train_subjects,
        "train_subject_names": [meta.subject_names[idx] for idx in split_subjects["train"]],
        "val_subject_names": [meta.subject_names[idx] for idx in split_subjects["val"]],
        "test_subject_names": [meta.subject_names[idx] for idx in split_subjects["test"]],
        "label_counts": {},
        "group_counts": {},
        "train_indices": segment_indices(split_subjects["train"]).tolist(),
        "val_indices": segment_indices(split_subjects["val"]).tolist(),
        "test_indices": segment_indices(split_subjects["test"]).tolist(),
    }
    if protocol == "subject_80_20_innerval_balanced":
        payload["notes"] = (
            "Outer split is 80% train / 20% test by subject. "
            "The saved val split is an inner validation subset drawn from the outer-train subjects for model selection."
        )
    elif protocol == "subject_80_20_teststop_balanced":
        payload["notes"] = (
            "Strict 80% train / 20% test by subject. "
            "No independent validation set is used; val_indices intentionally mirror test_indices "
            "so downstream scripts early-stop on the test fold."
        )
    else:
        payload["notes"] = "Strict 8:1:1 subject split with HC/DEP present in train, val, and test."

    for split_name in ["train", "val", "test"]:
        subject_ids = np.asarray(payload[f"{split_name}_subject_indices"], dtype=np.int64)
        groups = [meta.subject_groups[idx] for idx in subject_ids]
        labels = meta.labels[np.asarray(payload[f"{split_name}_indices"], dtype=np.int64)]
        label_vals, label_cnt = np.unique(labels, return_counts=True)
        payload["label_counts"][split_name] = {str(int(v)): int(c) for v, c in zip(label_vals, label_cnt)}
        payload["group_counts"][split_name] = {
            "HC": int(sum(1 for g in groups if g.upper().startswith("HC"))),
            "DEP": int(sum(1 for g in groups if g.upper().startswith("DEP"))),
        }

    return payload


def protocol_output_dir(output_root: Path, protocol: str) -> Path:
    return output_root / protocol / "splits"


def main() -> None:
    args = parse_args()
    meta = load_subject_meta(args.feature_h5)
    manifest = {}

    for protocol in args.protocols:
        split_dir = protocol_output_dir(args.output_root, protocol)
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest[protocol] = {"split_dir": str(split_dir), "seeds": args.seeds, "files": []}
        for seed in args.seeds:
            payload = build_subject_split(meta, protocol, seed)
            path = split_dir / f"comp4_subject_seed{seed}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            manifest[protocol]["files"].append(str(path))
            print(
                f"{protocol} seed={seed}: "
                f"subjects train/val/test="
                f"{len(payload['train_subject_indices'])}/{len(payload['val_subject_indices'])}/{len(payload['test_subject_indices'])} "
                f"group_counts={payload['group_counts']}",
                flush=True,
            )

    manifest_path = args.output_root / "balanced_subject_splits_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
