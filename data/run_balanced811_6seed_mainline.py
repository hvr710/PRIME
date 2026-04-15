#!/usr/bin/env python3
"""Balanced 8:1:1 6-seed COMP4 mainline rerun with adapt-then-smooth caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from train_fusion_step5 import MLPHead, load_feature_matrix


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "balanced811_6seed_adaptsmoothall_mainline"
DEFAULT_SPLIT_DIR = DEFAULT_OUTPUT_ROOT / "splits"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "adapt_cache"
DEFAULT_BASELINE_DIR = DEFAULT_OUTPUT_ROOT / "baseline_results"
DEFAULT_FUSION_DIR = DEFAULT_OUTPUT_ROOT / "fusion_results"
DEFAULT_MOE_DIR = DEFAULT_OUTPUT_ROOT / "moe_results"
DEFAULT_COMPARISON_DIR = DEFAULT_OUTPUT_ROOT / "comparison"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_EMBED_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_EEG_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_FULLFT_DIR = DEFAULT_BASELINE_DIR / "full_finetune_results"
DEFAULT_SEEDS = [42, 3407, 2025, 666, 777, 888]
DEFAULT_SPLIT_KINDS = ["subject", "segment"]
DEFAULT_HAND_MODELS = ["logistic_regression", "svm", "xgboost"]
DEFAULT_MOE_VARIANTS = [
    "cross_attn_moe_unsup",
    "cross_attn_moe_sup_l03",
    "cross_attn_moe_sup_l05",
    "cross_attn_moe_token_sup_l03",
    "cross_attn_moe_token_sup_l05",
]
DEFAULT_SCREEN_VARIANTS = ["cross_attn_moe_sup_l03", "cross_attn_moe_token_sup_l03"]
DEFAULT_SCREEN_LRS = [1e-3, 5e-4, 3e-4]
ROUTER_TEMP_GRID = [0.7, 1.0, 1.3, 1.6, 2.0]
DEFAULT_TRANSFORM_MODE = "safe"
TRANSFORM_MODES = [
    "safe",
    "unsafe_global_norm_smooth_all",
    "presplit_subject_zscore_smooth_all",
    "presplit_subject_zscore_all",
    "presplit_subject_smooth_all",
    "raw_no_preprocess",
]


@dataclass
class SplitSpec:
    split_kind: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    path: Path


@dataclass
class RawInputs:
    hand: np.ndarray
    pool: np.ndarray
    channel_tokens: np.ndarray
    patch_tokens: np.ndarray
    label: np.ndarray
    subject_index: np.ndarray
    trial_index: np.ndarray
    segment_index: np.ndarray
    global_trial_index: np.ndarray
    subject_groups: list[str]
    subject_names: list[str]


class PoolDataset(Dataset):
    def __init__(self, pool: np.ndarray, labels: np.ndarray, trial_ids: np.ndarray, indices: np.ndarray) -> None:
        self.pool = torch.from_numpy(pool[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.trial_ids = torch.from_numpy(trial_ids[indices].astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return self.pool[idx], self.labels[idx], self.trial_ids[idx]


class CacheFusionDataset(Dataset):
    def __init__(
        self,
        pool: np.ndarray,
        hand: np.ndarray,
        channel_tokens: np.ndarray,
        patch_tokens: np.ndarray,
        labels: np.ndarray,
        trial_ids: np.ndarray,
        group_labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.pool = torch.from_numpy(pool[indices].astype(np.float32))
        self.hand = torch.from_numpy(hand[indices].astype(np.float32))
        self.channel_tokens = torch.from_numpy(channel_tokens[indices].astype(np.float32))
        self.patch_tokens = torch.from_numpy(patch_tokens[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.trial_ids = torch.from_numpy(trial_ids[indices].astype(np.int64))
        self.group_labels = torch.from_numpy(group_labels[indices].astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return (
            self.pool[idx],
            self.hand[idx],
            self.channel_tokens[idx],
            self.patch_tokens[idx],
            self.labels[idx],
            self.trial_ids[idx],
            self.group_labels[idx],
        )


class CrossAttnSingleHead(nn.Module):
    def __init__(self, hand_dim: int, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hand_query = nn.Linear(hand_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.classifier = MLPHead(token_dim, hidden_dim, dropout=dropout)

    def forward(
        self,
        pool: torch.Tensor,
        hand: torch.Tensor,
        channel_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        del pool
        query = self.hand_query(hand).unsqueeze(1)
        tokens = torch.cat([channel_tokens, patch_tokens], dim=1)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))
        return self.classifier(fused)


VARIANT_CONFIGS = {
    "cross_attn_single_head": {"router_loss_weight": None, "use_subject_token": False},
    "cross_attn_moe_unsup": {"router_loss_weight": 0.0, "use_subject_token": False},
    "cross_attn_moe_sup_l03": {"router_loss_weight": 0.3, "use_subject_token": False},
    "cross_attn_moe_sup_l05": {"router_loss_weight": 0.5, "use_subject_token": False},
    "cross_attn_moe_token_sup_l03": {"router_loss_weight": 0.3, "use_subject_token": True},
    "cross_attn_moe_token_sup_l05": {"router_loss_weight": 0.5, "use_subject_token": True},
}


class MainlineMoEModel(nn.Module):
    def __init__(
        self,
        variant: str,
        pool_dim: int,
        hand_dim: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.router_loss_weight = VARIANT_CONFIGS[variant]["router_loss_weight"]
        self.use_subject_token = VARIANT_CONFIGS[variant]["use_subject_token"]
        self.single_head = variant == "cross_attn_single_head"
        self.hand_query = nn.Linear(hand_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)

        if self.single_head:
            self.classifier = MLPHead(token_dim, hidden_dim, dropout=dropout)
        else:
            self.router = nn.Sequential(
                nn.Linear(pool_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self.expert_hc = MLPHead(token_dim, hidden_dim, dropout=dropout)
            self.expert_dep = MLPHead(token_dim, hidden_dim, dropout=dropout)
            if self.use_subject_token:
                self.token_hc = nn.Parameter(torch.zeros(1, 1, token_dim))
                self.token_dep = nn.Parameter(torch.zeros(1, 1, token_dim))
                nn.init.normal_(self.token_hc, mean=0.0, std=0.02)
                nn.init.normal_(self.token_dep, mean=0.0, std=0.02)

    def _prepend_subject_token(
        self,
        tokens: torch.Tensor,
        group_labels: torch.Tensor | None,
        router_prob: torch.Tensor,
        use_true_group_token: bool,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]
        if use_true_group_token:
            if group_labels is None:
                raise ValueError("group_labels is required when use_true_group_token=True")
            token_ids = group_labels
        else:
            token_ids = router_prob.argmax(dim=1)
        hc_token = self.token_hc.expand(batch_size, -1, -1)
        dep_token = self.token_dep.expand(batch_size, -1, -1)
        selector = token_ids.view(batch_size, 1, 1).to(dtype=torch.bool)
        chosen = torch.where(selector, dep_token, hc_token)
        return torch.cat([chosen, tokens], dim=1)

    def forward(
        self,
        pool: torch.Tensor,
        hand: torch.Tensor,
        channel_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        group_labels: torch.Tensor | None = None,
        *,
        use_true_group_token: bool = False,
        router_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor | None]:
        query = self.hand_query(hand).unsqueeze(1)
        tokens = torch.cat([channel_tokens, patch_tokens], dim=1)
        if self.single_head:
            attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
            fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))
            return {"emotion_logits": self.classifier(fused), "router_logits": None, "router_prob": None}

        router_logits = self.router(pool)
        router_prob = torch.softmax(router_logits / router_temperature, dim=1)
        if self.use_subject_token:
            tokens = self._prepend_subject_token(tokens, group_labels, router_prob, use_true_group_token)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))
        logits_hc = self.expert_hc(fused)
        logits_dep = self.expert_dep(fused)
        emotion_logits = router_prob[:, :1] * logits_hc + router_prob[:, 1:] * logits_dep
        return {"emotion_logits": emotion_logits, "router_logits": router_logits, "router_prob": router_prob}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    base.add_argument("--embed-h5", type=Path, default=DEFAULT_EMBED_H5)
    base.add_argument("--eeg-h5", type=Path, default=DEFAULT_EEG_H5)
    base.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    base.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    base.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    base.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    base.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION_DIR)
    base.add_argument("--moe-dir", type=Path, default=DEFAULT_MOE_DIR)
    base.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    base.add_argument("--fullft-dir", type=Path, default=DEFAULT_FULLFT_DIR)
    base.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    base.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)

    subparsers.add_parser("make_splits", parents=[base])

    cache_parser = subparsers.add_parser("build_cache", parents=[base])
    cache_parser.add_argument("--smooth-alpha", type=float, default=0.65)
    cache_parser.add_argument("--transform-mode", choices=TRANSFORM_MODES, default=DEFAULT_TRANSFORM_MODE)

    baseline_parser = subparsers.add_parser("train_baselines", parents=[base])
    baseline_parser.add_argument("--models", nargs="+", default=DEFAULT_HAND_MODELS + ["frozen_mlp"])
    baseline_parser.add_argument("--batch-size", type=int, default=256)
    baseline_parser.add_argument("--max-epochs", type=int, default=80)
    baseline_parser.add_argument("--patience", type=int, default=10)
    baseline_parser.add_argument("--lr", type=float, default=1e-3)
    baseline_parser.add_argument("--weight-decay", type=float, default=1e-4)
    baseline_parser.add_argument("--hidden-dim", type=int, default=256)
    baseline_parser.add_argument("--dropout", type=float, default=0.3)
    baseline_parser.add_argument("--n-jobs", type=int, default=8)
    baseline_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    fusion_parser = subparsers.add_parser("train_fusion", parents=[base])
    fusion_parser.add_argument("--batch-size", type=int, default=256)
    fusion_parser.add_argument("--max-epochs", type=int, default=80)
    fusion_parser.add_argument("--patience", type=int, default=10)
    fusion_parser.add_argument("--lr", type=float, default=1e-3)
    fusion_parser.add_argument("--weight-decay", type=float, default=1e-4)
    fusion_parser.add_argument("--hidden-dim", type=int, default=256)
    fusion_parser.add_argument("--dropout", type=float, default=0.3)
    fusion_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    screen_parser = subparsers.add_parser("screen_moe_lr", parents=[base])
    screen_parser.add_argument("--screen-seed", type=int, default=42)
    screen_parser.add_argument("--screen-lrs", nargs="+", type=float, default=DEFAULT_SCREEN_LRS)
    screen_parser.add_argument("--screen-variants", nargs="+", default=DEFAULT_SCREEN_VARIANTS, choices=DEFAULT_SCREEN_VARIANTS)
    screen_parser.add_argument("--batch-size", type=int, default=256)
    screen_parser.add_argument("--max-epochs", type=int, default=80)
    screen_parser.add_argument("--patience", type=int, default=10)
    screen_parser.add_argument("--weight-decay", type=float, default=1e-4)
    screen_parser.add_argument("--hidden-dim", type=int, default=256)
    screen_parser.add_argument("--dropout", type=float, default=0.3)
    screen_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    moe_parser = subparsers.add_parser("train_moe", parents=[base])
    moe_parser.add_argument("--variants", nargs="+", default=DEFAULT_MOE_VARIANTS, choices=DEFAULT_MOE_VARIANTS)
    moe_parser.add_argument("--base-lr", type=float, default=None)
    moe_parser.add_argument("--batch-size", type=int, default=256)
    moe_parser.add_argument("--max-epochs", type=int, default=80)
    moe_parser.add_argument("--patience", type=int, default=10)
    moe_parser.add_argument("--weight-decay", type=float, default=1e-4)
    moe_parser.add_argument("--hidden-dim", type=int, default=256)
    moe_parser.add_argument("--dropout", type=float, default=0.3)
    moe_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    subparsers.add_parser("summarize", parents=[base])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def ensure_dirs(args: argparse.Namespace) -> None:
    for path in [
        args.output_root,
        args.split_dir,
        args.cache_dir,
        args.baseline_dir,
        args.fusion_dir,
        args.moe_dir,
        args.comparison_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_inputs(feature_h5: Path, embed_h5: Path) -> RawInputs:
    hand, _feature_names, metadata = load_feature_matrix(feature_h5)
    with h5py.File(feature_h5, "r") as handle:
        feature_label = handle["label"][:].astype(np.int64)
        feature_subject_index = handle["subject_index"][:].astype(np.int64)
        trial_index = handle["trial_index"][:].astype(np.int64)
        segment_index = handle["segment_index"][:].astype(np.int64)
        global_trial_index = handle["global_trial_index"][:].astype(np.int64)
        subject_groups = decode_strings(handle["meta/subject_groups"][:])
        subject_names = decode_strings(handle["meta/subject_names"][:])

    with h5py.File(embed_h5, "r") as handle:
        pool = handle["pool_1024"][:].astype(np.float32)
        channel_tokens = handle["channel_tokens_32"][:].astype(np.float32)
        patch_tokens = handle["patch_tokens_32"][:].astype(np.float32)
        label = handle["label"][:].astype(np.int64)
        subject_index = handle["subject_index"][:].astype(np.int64)
        embed_trial = handle["trial_index"][:].astype(np.int64) if "trial_index" in handle else trial_index
        embed_segment = handle["segment_index"][:].astype(np.int64) if "segment_index" in handle else segment_index
        embed_global_trial = handle["global_trial_index"][:].astype(np.int64)

    if not np.array_equal(label, feature_label):
        raise ValueError("Label mismatch between feature and embedding H5.")
    if not np.array_equal(subject_index, feature_subject_index):
        raise ValueError("subject_index mismatch between feature and embedding H5.")
    if not np.array_equal(embed_global_trial, global_trial_index):
        raise ValueError("global_trial_index mismatch between feature and embedding H5.")

    return RawInputs(
        hand=hand.astype(np.float32),
        pool=pool,
        channel_tokens=channel_tokens,
        patch_tokens=patch_tokens,
        label=label,
        subject_index=subject_index,
        trial_index=embed_trial,
        segment_index=embed_segment,
        global_trial_index=embed_global_trial,
        subject_groups=subject_groups,
        subject_names=subject_names,
    )


def build_subject_group_labels(subject_index: np.ndarray, subject_groups: list[str]) -> np.ndarray:
    out = np.empty(subject_index.shape[0], dtype=np.int64)
    for idx, subject_id in enumerate(subject_index):
        group = subject_groups[int(subject_id)].upper()
        if group.startswith("HC"):
            out[idx] = 0
        elif group.startswith("DEP"):
            out[idx] = 1
        else:
            raise ValueError(f"Unexpected subject group {subject_groups[int(subject_id)]!r}")
    return out


def _group_subject_ids(subject_groups: list[str]) -> dict[str, np.ndarray]:
    grouped = {"HC": [], "DEP": []}
    for subject_id, group_name in enumerate(subject_groups):
        upper = group_name.upper()
        if upper.startswith("HC"):
            grouped["HC"].append(subject_id)
        elif upper.startswith("DEP"):
            grouped["DEP"].append(subject_id)
        else:
            raise ValueError(f"Unexpected subject group {group_name!r}")
    return {key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()}


def _label_counts(indices: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _subject_group_counts(subject_ids: np.ndarray, subject_groups: list[str]) -> dict[str, int]:
    groups = [subject_groups[int(i)] for i in subject_ids]
    return {
        "HC": int(sum(g.upper().startswith("HC") for g in groups)),
        "DEP": int(sum(g.upper().startswith("DEP") for g in groups)),
    }


def build_subject_split_payload(raw: RawInputs, seed: int) -> dict[str, object]:
    grouped = _group_subject_ids(raw.subject_groups)
    counts = {
        "HC": {"train": 32, "val": 4, "test": 4},
        "DEP": {"train": 16, "val": 2, "test": 2},
    }
    split_subjects = {"train": [], "val": [], "test": []}
    for group_name in ["HC", "DEP"]:
        rng = np.random.default_rng(seed if group_name == "HC" else seed + 1)
        shuffled = grouped[group_name].copy()
        rng.shuffle(shuffled)
        n_test = counts[group_name]["test"]
        n_val = counts[group_name]["val"]
        split_subjects["test"].extend(shuffled[:n_test].tolist())
        split_subjects["val"].extend(shuffled[n_test:n_test + n_val].tolist())
        split_subjects["train"].extend(shuffled[n_test + n_val:].tolist())

    for split_name in split_subjects:
        split_subjects[split_name] = sorted(split_subjects[split_name])

    def segment_indices(subject_ids: list[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(raw.subject_index, np.asarray(subject_ids, dtype=np.int64))).astype(np.int64)

    payload = {
        "split_kind": "subject",
        "protocol": "balanced_811",
        "seed": int(seed),
        "train_subject_indices": split_subjects["train"],
        "val_subject_indices": split_subjects["val"],
        "test_subject_indices": split_subjects["test"],
        "train_subject_names": [raw.subject_names[idx] for idx in split_subjects["train"]],
        "val_subject_names": [raw.subject_names[idx] for idx in split_subjects["val"]],
        "test_subject_names": [raw.subject_names[idx] for idx in split_subjects["test"]],
        "train_indices": segment_indices(split_subjects["train"]).tolist(),
        "val_indices": segment_indices(split_subjects["val"]).tolist(),
        "test_indices": segment_indices(split_subjects["test"]).tolist(),
        "notes": "Strict balanced 8:1:1 subject split. Train/val/test each contain HC and DEP subjects.",
        "group_counts": {},
        "label_counts": {},
    }
    for split_name in ["train", "val", "test"]:
        subject_ids = np.asarray(payload[f"{split_name}_subject_indices"], dtype=np.int64)
        indices = np.asarray(payload[f"{split_name}_indices"], dtype=np.int64)
        payload["group_counts"][split_name] = _subject_group_counts(subject_ids, raw.subject_groups)
        payload["label_counts"][split_name] = _label_counts(indices, raw.label)
    return payload


def build_segment_split_payload(raw: RawInputs, seed: int) -> dict[str, object]:
    group_labels = build_subject_group_labels(raw.subject_index, raw.subject_groups)
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    strat_keys = np.stack([group_labels, raw.label], axis=1)
    unique_strata = np.unique(strat_keys, axis=0)
    for group_id, label in unique_strata:
        idx = np.flatnonzero((group_labels == group_id) & (raw.label == label)).astype(np.int64)
        rng.shuffle(idx)
        n_total = idx.shape[0]
        n_val = int(round(n_total * 0.1))
        n_test = int(round(n_total * 0.1))
        n_train = n_total - n_val - n_test
        train_parts.append(np.sort(idx[:n_train]))
        val_parts.append(np.sort(idx[n_train:n_train + n_val]))
        test_parts.append(np.sort(idx[n_train + n_val:]))

    train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val = np.sort(np.concatenate(val_parts)).astype(np.int64)
    test = np.sort(np.concatenate(test_parts)).astype(np.int64)
    payload = {
        "split_kind": "segment",
        "protocol": "balanced_811",
        "seed": int(seed),
        "train_indices": train.tolist(),
        "val_indices": val.tolist(),
        "test_indices": test.tolist(),
        "notes": "Balanced 8:1:1 segment split stratified by subject_group x emotion_label.",
        "group_counts": {},
        "label_counts": {},
        "n_subjects_per_split": {},
    }
    for split_name, indices in [("train", train), ("val", val), ("test", test)]:
        subjects = np.unique(raw.subject_index[indices])
        subject_groups = _subject_group_counts(subjects, raw.subject_groups)
        payload["group_counts"][split_name] = subject_groups
        payload["label_counts"][split_name] = _label_counts(indices, raw.label)
        payload["n_subjects_per_split"][split_name] = int(subjects.shape[0])
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_split(split_dir: Path, split_kind: str, seed: int) -> SplitSpec:
    path = split_dir / f"comp4_{split_kind}_seed{seed}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitSpec(
        split_kind=split_kind,
        seed=seed,
        train=np.asarray(payload["train_indices"], dtype=np.int64),
        val=np.asarray(payload["val_indices"], dtype=np.int64),
        test=np.asarray(payload["test_indices"], dtype=np.int64),
        path=path,
    )


def splitwise_subject_adapt_array(values: np.ndarray, split: SplitSpec, subject_ids: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for split_indices in [split.train, split.val, split.test]:
        for subject_id in np.unique(subject_ids[split_indices]):
            idx = split_indices[subject_ids[split_indices] == subject_id]
            mean = out[idx].mean(axis=0, keepdims=True)
            std = out[idx].std(axis=0, keepdims=True)
            out[idx] = (out[idx] - mean) / (std + eps)
    return out.astype(np.float32)


def causal_smooth_array(values: np.ndarray, split: SplitSpec, trial_ids: np.ndarray, segment_index: np.ndarray, alpha: float) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for split_indices in [split.train, split.val, split.test]:
        for trial_id in np.unique(trial_ids[split_indices]):
            idx = split_indices[trial_ids[split_indices] == trial_id]
            idx = idx[np.argsort(segment_index[idx])]
            prev = None
            for item in idx:
                current = values[item]
                if prev is None:
                    prev = current.copy()
                else:
                    prev = alpha * prev + (1.0 - alpha) * current
                out[item] = prev
    return out.astype(np.float32)


def global_normalize_array(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / (std + eps)).astype(np.float32)


def presplit_subject_adapt_array(values: np.ndarray, subject_ids: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for subject_id in np.unique(subject_ids):
        idx = np.flatnonzero(subject_ids == subject_id)
        mean = out[idx].mean(axis=0, keepdims=True)
        std = out[idx].std(axis=0, keepdims=True)
        out[idx] = (out[idx] - mean) / (std + eps)
    return out.astype(np.float32)


def presplit_causal_smooth_array(values: np.ndarray, trial_ids: np.ndarray, segment_index: np.ndarray, alpha: float) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    all_indices = np.arange(values.shape[0], dtype=np.int64)
    for trial_id in np.unique(trial_ids):
        idx = all_indices[trial_ids == trial_id]
        idx = idx[np.argsort(segment_index[idx])]
        prev = None
        for item in idx:
            current = values[item]
            if prev is None:
                prev = current.copy()
            else:
                prev = alpha * prev + (1.0 - alpha) * current
            out[item] = prev
    return out.astype(np.float32)


def build_adapted_arrays(raw: RawInputs, split: SplitSpec, smooth_alpha: float, transform_mode: str = DEFAULT_TRANSFORM_MODE) -> dict[str, np.ndarray]:
    if transform_mode == "raw_no_preprocess":
        return {
            "hand_1520": raw.hand.astype(np.float32, copy=False),
            "pool_1024": raw.pool.astype(np.float32, copy=False),
            "channel_tokens_32": raw.channel_tokens.astype(np.float32, copy=False),
            "patch_tokens_32": raw.patch_tokens.astype(np.float32, copy=False),
        }
    if transform_mode == "presplit_subject_zscore_smooth_all":
        hand = presplit_causal_smooth_array(
            presplit_subject_adapt_array(raw.hand, raw.subject_index),
            raw.global_trial_index,
            raw.segment_index,
            smooth_alpha,
        )
        pool = presplit_causal_smooth_array(
            presplit_subject_adapt_array(raw.pool, raw.subject_index),
            raw.global_trial_index,
            raw.segment_index,
            smooth_alpha,
        )
        channel_tokens = presplit_causal_smooth_array(
            presplit_subject_adapt_array(raw.channel_tokens, raw.subject_index),
            raw.global_trial_index,
            raw.segment_index,
            smooth_alpha,
        )
        patch_tokens = presplit_causal_smooth_array(
            presplit_subject_adapt_array(raw.patch_tokens, raw.subject_index),
            raw.global_trial_index,
            raw.segment_index,
            smooth_alpha,
        )
        return {
            "hand_1520": hand,
            "pool_1024": pool,
            "channel_tokens_32": channel_tokens,
            "patch_tokens_32": patch_tokens,
        }
    if transform_mode == "presplit_subject_zscore_all":
        hand = presplit_subject_adapt_array(raw.hand, raw.subject_index)
        pool = presplit_subject_adapt_array(raw.pool, raw.subject_index)
        channel_tokens = presplit_subject_adapt_array(raw.channel_tokens, raw.subject_index)
        patch_tokens = presplit_subject_adapt_array(raw.patch_tokens, raw.subject_index)
        return {
            "hand_1520": hand,
            "pool_1024": pool,
            "channel_tokens_32": channel_tokens,
            "patch_tokens_32": patch_tokens,
        }
    if transform_mode == "presplit_subject_smooth_all":
        hand = presplit_causal_smooth_array(raw.hand, raw.global_trial_index, raw.segment_index, smooth_alpha)
        pool = presplit_causal_smooth_array(raw.pool, raw.global_trial_index, raw.segment_index, smooth_alpha)
        channel_tokens = presplit_causal_smooth_array(raw.channel_tokens, raw.global_trial_index, raw.segment_index, smooth_alpha)
        patch_tokens = presplit_causal_smooth_array(raw.patch_tokens, raw.global_trial_index, raw.segment_index, smooth_alpha)
        return {
            "hand_1520": hand,
            "pool_1024": pool,
            "channel_tokens_32": channel_tokens,
            "patch_tokens_32": patch_tokens,
        }
    if transform_mode == "unsafe_global_norm_smooth_all":
        hand = presplit_causal_smooth_array(global_normalize_array(raw.hand), raw.global_trial_index, raw.segment_index, smooth_alpha)
        pool = presplit_causal_smooth_array(global_normalize_array(raw.pool), raw.global_trial_index, raw.segment_index, smooth_alpha)
        channel_tokens = presplit_causal_smooth_array(
            global_normalize_array(raw.channel_tokens), raw.global_trial_index, raw.segment_index, smooth_alpha
        )
        patch_tokens = presplit_causal_smooth_array(
            global_normalize_array(raw.patch_tokens), raw.global_trial_index, raw.segment_index, smooth_alpha
        )
        return {
            "hand_1520": hand,
            "pool_1024": pool,
            "channel_tokens_32": channel_tokens,
            "patch_tokens_32": patch_tokens,
        }
    if transform_mode != "safe":
        raise ValueError(f"Unknown transform mode: {transform_mode}")
    hand = splitwise_subject_adapt_array(raw.hand, split, raw.subject_index)
    pool = splitwise_subject_adapt_array(raw.pool, split, raw.subject_index)
    channel_tokens = splitwise_subject_adapt_array(raw.channel_tokens, split, raw.subject_index)
    patch_tokens = splitwise_subject_adapt_array(raw.patch_tokens, split, raw.subject_index)
    hand = causal_smooth_array(hand, split, raw.global_trial_index, raw.segment_index, smooth_alpha)
    pool = causal_smooth_array(pool, split, raw.global_trial_index, raw.segment_index, smooth_alpha)
    channel_tokens = causal_smooth_array(channel_tokens, split, raw.global_trial_index, raw.segment_index, smooth_alpha)
    patch_tokens = causal_smooth_array(patch_tokens, split, raw.global_trial_index, raw.segment_index, smooth_alpha)
    return {
        "hand_1520": hand,
        "pool_1024": pool,
        "channel_tokens_32": channel_tokens,
        "patch_tokens_32": patch_tokens,
    }


def save_cache_h5(
    cache_path: Path,
    arrays: dict[str, np.ndarray],
    raw: RawInputs,
    split: SplitSpec,
    smooth_alpha: float,
    transform_mode: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    group_labels = build_subject_group_labels(raw.subject_index, raw.subject_groups)
    with h5py.File(cache_path, "w") as handle:
        for key, value in arrays.items():
            handle.create_dataset(key, data=value, compression="gzip")
        handle.create_dataset("label", data=raw.label, compression="gzip")
        handle.create_dataset("subject_index", data=raw.subject_index, compression="gzip")
        handle.create_dataset("subject_group_label", data=group_labels, compression="gzip")
        handle.create_dataset("trial_index", data=raw.trial_index, compression="gzip")
        handle.create_dataset("segment_index", data=raw.segment_index, compression="gzip")
        handle.create_dataset("global_trial_index", data=raw.global_trial_index, compression="gzip")
        handle.create_dataset("split_train_indices", data=split.train, compression="gzip")
        handle.create_dataset("split_val_indices", data=split.val, compression="gzip")
        handle.create_dataset("split_test_indices", data=split.test, compression="gzip")
        meta = handle.create_group("meta")
        meta.create_dataset("subject_groups", data=np.asarray(raw.subject_groups, dtype="S"))
        meta.create_dataset("subject_names", data=np.asarray(raw.subject_names, dtype="S"))
        handle.attrs["split_kind"] = split.split_kind
        handle.attrs["seed"] = int(split.seed)
        handle.attrs["smooth_alpha"] = float(smooth_alpha)
        handle.attrs["split_path"] = str(split.path)
        handle.attrs["transform_mode"] = transform_mode


def load_cache(cache_path: Path) -> dict[str, np.ndarray | str | float | list[str]]:
    with h5py.File(cache_path, "r") as handle:
        cache = {
            "hand_1520": handle["hand_1520"][:].astype(np.float32),
            "pool_1024": handle["pool_1024"][:].astype(np.float32),
            "channel_tokens_32": handle["channel_tokens_32"][:].astype(np.float32),
            "patch_tokens_32": handle["patch_tokens_32"][:].astype(np.float32),
            "label": handle["label"][:].astype(np.int64),
            "subject_index": handle["subject_index"][:].astype(np.int64),
            "subject_group_label": handle["subject_group_label"][:].astype(np.int64),
            "trial_index": handle["trial_index"][:].astype(np.int64),
            "segment_index": handle["segment_index"][:].astype(np.int64),
            "global_trial_index": handle["global_trial_index"][:].astype(np.int64),
            "split_train_indices": handle["split_train_indices"][:].astype(np.int64),
            "split_val_indices": handle["split_val_indices"][:].astype(np.int64),
            "split_test_indices": handle["split_test_indices"][:].astype(np.int64),
            "subject_groups": decode_strings(handle["meta/subject_groups"][:]),
            "subject_names": decode_strings(handle["meta/subject_names"][:]),
            "split_kind": str(handle.attrs["split_kind"]),
            "seed": int(handle.attrs["seed"]),
            "split_path": str(handle.attrs["split_path"]),
            "smooth_alpha": float(handle.attrs["smooth_alpha"]),
            "transform_mode": str(handle.attrs.get("transform_mode", DEFAULT_TRANSFORM_MODE)),
        }
    return cache


def cache_path(cache_dir: Path, split_kind: str, seed: int) -> Path:
    return cache_dir / split_kind / f"comp4_{split_kind}_seed{seed}.h5"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def maybe_auroc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, score))


def maybe_auprc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, score))


def metrics_from_scores(y_true: np.ndarray, score_pos: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    y_pred = (score_pos >= threshold).astype(np.int64)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": maybe_auroc(y_true, score_pos),
        "auprc": maybe_auprc(y_true, score_pos),
        "n_samples": int(y_true.shape[0]),
    }


def metrics_from_probs_safe(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int | None]:
    y_pred = y_prob.argmax(axis=1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": maybe_auroc(y_true, y_prob[:, 1]),
        "auprc": maybe_auprc(y_true, y_prob[:, 1]),
        "n_samples": int(y_true.shape[0]),
    }


def aggregate_trials_scores(y_true: np.ndarray, score_pos: np.ndarray, trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_trials = np.unique(trial_ids)
    trial_true = []
    trial_score = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        trial_true.append(int(y_true[mask][0]))
        trial_score.append(float(score_pos[mask].mean()))
    return np.asarray(trial_true, dtype=np.int64), np.asarray(trial_score, dtype=np.float64)


def aggregate_trials_probs(y_true: np.ndarray, y_prob: np.ndarray, trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_trials = np.unique(trial_ids)
    trial_true = []
    trial_prob = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        trial_true.append(int(y_true[mask][0]))
        trial_prob.append(y_prob[mask].mean(axis=0))
    return np.asarray(trial_true, dtype=np.int64), np.stack(trial_prob, axis=0)


def collect_feature_scores(model_name: str, model, X: np.ndarray) -> np.ndarray:
    if model_name == "svm":
        return sigmoid(model.decision_function(X))
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return sigmoid(model.decision_function(X))
    return model.predict(X).astype(np.float64)


def make_pool_loaders(
    split: SplitSpec,
    pool: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(PoolDataset(pool, labels, trial_ids, split.train), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(PoolDataset(pool, labels, trial_ids, split.val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PoolDataset(pool, labels, trial_ids, split.test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def make_cache_loaders(cache: dict[str, np.ndarray | str | float | list[str]], split: SplitSpec, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        CacheFusionDataset(
            cache["pool_1024"],
            cache["hand_1520"],
            cache["channel_tokens_32"],
            cache["patch_tokens_32"],
            cache["label"],
            cache["global_trial_index"],
            cache["subject_group_label"],
            split.train,
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        CacheFusionDataset(
            cache["pool_1024"],
            cache["hand_1520"],
            cache["channel_tokens_32"],
            cache["patch_tokens_32"],
            cache["label"],
            cache["global_trial_index"],
            cache["subject_group_label"],
            split.val,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        CacheFusionDataset(
            cache["pool_1024"],
            cache["hand_1520"],
            cache["channel_tokens_32"],
            cache["patch_tokens_32"],
            cache["label"],
            cache["global_trial_index"],
            cache["subject_group_label"],
            split.test,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def evaluate_pool_model(model: nn.Module, loader: DataLoader, device: torch.device, aggregate_by_trial: bool) -> dict[str, dict | None]:
    model.eval()
    labels, probs, trial_ids = [], [], []
    with torch.no_grad():
        for pool, y, trial_id in loader:
            logits = model(pool.to(device))
            labels.append(y.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            trial_ids.append(trial_id.numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    trial = np.concatenate(trial_ids)
    result = {"window": metrics_from_probs_safe(y_true, y_prob)}
    if aggregate_by_trial:
        trial_true, trial_prob = aggregate_trials_probs(y_true, y_prob, trial)
        result["trial"] = metrics_from_probs_safe(trial_true, trial_prob)
    else:
        result["trial"] = None
    return result


def train_pool_mlp(
    pool: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    split: SplitSpec,
    batch_size: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    checkpoint_path: Path,
) -> tuple[MLPHead, int, float, float]:
    train_loader, val_loader, _ = make_pool_loaders(split, pool, labels, trial_ids, batch_size, seed)
    model = MLPHead(int(pool.shape[1]), hidden_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    start = perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for features, y, _trial in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_pool_model(model, val_loader, device, aggregate_by_trial=False)["window"]
        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    return model, best_epoch, best_val_acc, perf_counter() - start


def train_cross_attn_single_head(
    cache: dict[str, np.ndarray | str | float | list[str]],
    split: SplitSpec,
    batch_size: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    checkpoint_path: Path,
) -> tuple[CrossAttnSingleHead, int, float, float]:
    train_loader, val_loader, _ = make_cache_loaders(cache, split, batch_size, seed)
    model = CrossAttnSingleHead(
        hand_dim=int(cache["hand_1520"].shape[1]),
        token_dim=int(cache["channel_tokens_32"].shape[2]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    start = perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for pool, hand, channel_tokens, patch_tokens, y, _trial, _group in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(pool.to(device), hand.to(device), channel_tokens.to(device), patch_tokens.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
        val_metrics = evaluate_cross_attn_single_head(model, val_loader, device, aggregate_by_trial=False)["window"]
        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    return model, best_epoch, best_val_acc, perf_counter() - start


def evaluate_cross_attn_single_head(model: CrossAttnSingleHead, loader: DataLoader, device: torch.device, aggregate_by_trial: bool) -> dict[str, dict | None]:
    model.eval()
    labels, probs, trial_ids = [], [], []
    with torch.no_grad():
        for pool, hand, channel_tokens, patch_tokens, y, trial_id, _group in loader:
            logits = model(pool.to(device), hand.to(device), channel_tokens.to(device), patch_tokens.to(device))
            labels.append(y.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            trial_ids.append(trial_id.numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    trial = np.concatenate(trial_ids)
    result = {"window": metrics_from_probs_safe(y_true, y_prob)}
    if aggregate_by_trial:
        trial_true, trial_prob = aggregate_trials_probs(y_true, y_prob, trial)
        result["trial"] = metrics_from_probs_safe(trial_true, trial_prob)
    else:
        result["trial"] = None
    return result


def router_class_weights(group_labels_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(group_labels_train.astype(np.int64), minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def split_metrics_to_row(row: dict, prefix: str, metrics: dict[str, dict | None]) -> None:
    for scope in ["window", "trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            key = f"{prefix}_{scope}_{metric}"
            row[key] = "" if metrics[scope] is None else metrics[scope][metric]


def collect_moe_predictions(
    model: MainlineMoEModel,
    loader: DataLoader,
    device: torch.device,
    *,
    router_temperature: float = 1.0,
) -> dict[str, np.ndarray | None]:
    model.eval()
    labels, probs, trial_ids, group_true, router_prob, router_logits = [], [], [], [], [], []
    with torch.no_grad():
        for pool, hand, channel_tokens, patch_tokens, y, trial_id, group_label in loader:
            outputs = model(
                pool.to(device),
                hand.to(device),
                channel_tokens.to(device),
                patch_tokens.to(device),
                group_label.to(device),
                use_true_group_token=False,
                router_temperature=router_temperature,
            )
            labels.append(y.numpy())
            probs.append(torch.softmax(outputs["emotion_logits"], dim=1).cpu().numpy())
            trial_ids.append(trial_id.numpy())
            group_true.append(group_label.numpy())
            if outputs["router_prob"] is not None:
                router_prob.append(outputs["router_prob"].cpu().numpy())
                router_logits.append(outputs["router_logits"].cpu().numpy())
    return {
        "label": np.concatenate(labels),
        "prob": np.concatenate(probs),
        "trial_id": np.concatenate(trial_ids),
        "group_true": np.concatenate(group_true),
        "router_prob": np.concatenate(router_prob) if router_prob else None,
        "router_logits": np.concatenate(router_logits) if router_logits else None,
    }


def router_metrics(group_true: np.ndarray, router_prob: np.ndarray | None) -> dict[str, float | None] | None:
    if router_prob is None:
        return None
    group_pred = router_prob.argmax(axis=1)
    hc_mask = group_true == 0
    dep_mask = group_true == 1
    return {
        "group_acc": float(accuracy_score(group_true, group_pred)),
        "group_f1": float(f1_score(group_true, group_pred, average="macro", zero_division=0)),
        "group_auroc": maybe_auroc(group_true, router_prob[:, 1]),
        "mean_p_hc_true_hc": float(router_prob[hc_mask, 0].mean()) if np.any(hc_mask) else None,
        "mean_p_dep_true_dep": float(router_prob[dep_mask, 1].mean()) if np.any(dep_mask) else None,
        "mean_p_dep_true_hc": float(router_prob[hc_mask, 1].mean()) if np.any(hc_mask) else None,
        "mean_p_hc_true_dep": float(router_prob[dep_mask, 0].mean()) if np.any(dep_mask) else None,
    }


def per_group_emotion_metrics(y_true: np.ndarray, y_prob: np.ndarray, group_true: np.ndarray) -> dict[str, dict | None]:
    out = {}
    for group_name, group_id in [("HC", 0), ("DEP", 1)]:
        mask = group_true == group_id
        out[group_name] = metrics_from_probs_safe(y_true[mask], y_prob[mask]) if np.any(mask) else None
    return out


def evaluate_moe(
    model: MainlineMoEModel,
    loader: DataLoader,
    device: torch.device,
    aggregate_by_trial: bool,
    *,
    router_temperature: float = 1.0,
) -> dict[str, dict | None]:
    predictions = collect_moe_predictions(model, loader, device, router_temperature=router_temperature)
    y_true = predictions["label"]
    y_prob = predictions["prob"]
    trial_ids = predictions["trial_id"]
    group_true = predictions["group_true"]
    result = {
        "window": metrics_from_probs_safe(y_true, y_prob),
        "trial": None,
        "router": router_metrics(group_true, predictions["router_prob"]),
        "per_group": per_group_emotion_metrics(y_true, y_prob, group_true),
        "router_logits": predictions["router_logits"],
        "group_true_raw": group_true,
    }
    if aggregate_by_trial:
        trial_true, trial_prob = aggregate_trials_probs(y_true, y_prob, trial_ids)
        result["trial"] = metrics_from_probs_safe(trial_true, trial_prob)
    return result


def fit_router_temperature(router_logits: np.ndarray | None, group_true: np.ndarray | None) -> float:
    if router_logits is None or group_true is None:
        return 1.0
    logits = torch.from_numpy(router_logits.astype(np.float32))
    labels = torch.from_numpy(group_true.astype(np.int64))
    best_temp = 1.0
    best_loss = math.inf
    for temp in ROUTER_TEMP_GRID:
        loss = nn.functional.cross_entropy(logits / float(temp), labels).item()
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def build_moe_optimizer(model: MainlineMoEModel, base_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if model.single_head:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    router_params = list(model.router.parameters())
    router_param_ids = {id(param) for param in router_params}
    other_params = [param for param in model.parameters() if id(param) not in router_param_ids]
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr},
            {"params": router_params, "lr": base_lr / 2.0},
        ],
        weight_decay=weight_decay,
    )


def train_moe_variant(
    cache: dict[str, np.ndarray | str | float | list[str]],
    split: SplitSpec,
    variant: str,
    batch_size: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    base_lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    checkpoint_path: Path,
) -> tuple[MainlineMoEModel, int, float, float]:
    train_loader, val_loader, _ = make_cache_loaders(cache, split, batch_size, seed)
    model = MainlineMoEModel(
        variant=variant,
        pool_dim=int(cache["pool_1024"].shape[1]),
        hand_dim=int(cache["hand_1520"].shape[1]),
        token_dim=int(cache["channel_tokens_32"].shape[2]),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    emotion_criterion = nn.CrossEntropyLoss()
    router_criterion = nn.CrossEntropyLoss(weight=router_class_weights(cache["subject_group_label"][split.train]).to(device))
    optimizer = build_moe_optimizer(model, base_lr, weight_decay)
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    start = perf_counter()

    for epoch in range(max_epochs):
        model.train()
        for pool, hand, channel_tokens, patch_tokens, y, _trial, group_label in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pool.to(device),
                hand.to(device),
                channel_tokens.to(device),
                patch_tokens.to(device),
                group_label.to(device),
                use_true_group_token=model.use_subject_token,
                router_temperature=1.0,
            )
            loss = emotion_criterion(outputs["emotion_logits"], y.to(device))
            if outputs["router_logits"] is not None and model.router_loss_weight is not None and model.router_loss_weight > 0:
                router_loss = router_criterion(outputs["router_logits"], group_label.to(device))
                loss = loss + model.router_loss_weight * router_loss
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_moe(model, val_loader, device, aggregate_by_trial=False, router_temperature=1.0)["window"]
        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    return model, best_epoch, best_val_acc, perf_counter() - start


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def fmt(entry: dict | None) -> str:
    if entry is None:
        return "-"
    return f"{entry['mean'] * 100:.2f} ± {entry['std'] * 100:.2f}"


def make_splits_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    raw = load_raw_inputs(args.feature_h5, args.embed_h5)
    manifest = {split_kind: [] for split_kind in args.split_kinds}
    for seed in args.seeds:
        payloads = {}
        if "subject" in args.split_kinds:
            payloads["subject"] = build_subject_split_payload(raw, seed)
        if "segment" in args.split_kinds:
            payloads["segment"] = build_segment_split_payload(raw, seed)
        for split_kind, payload in payloads.items():
            path = args.split_dir / f"comp4_{split_kind}_seed{seed}.json"
            write_json(path, payload)
            manifest[split_kind].append(str(path))
            groups = payload["group_counts"]
            print(
                f"{split_kind} seed={seed}: "
                f"train/val/test={len(payload['train_indices'])}/{len(payload['val_indices'])}/{len(payload['test_indices'])} "
                f"group_counts={groups}",
                flush=True,
            )
            for split_name in ["train", "val", "test"]:
                if groups[split_name]["HC"] <= 0 or groups[split_name]["DEP"] <= 0:
                    raise ValueError(f"{split_kind} seed={seed} split={split_name} lacks HC or DEP.")
    write_json(args.output_root / "split_manifest.json", manifest)
    print(f"Saved split files under {args.split_dir}", flush=True)


def build_cache_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    raw = load_raw_inputs(args.feature_h5, args.embed_h5)
    manifest = {}
    shared_arrays = None
    if args.transform_mode != "safe":
        # For all presplit/global transform modes, the adapted feature arrays are independent
        # of the specific split seed. Only the split index lists stored in the cache differ.
        shared_arrays = build_adapted_arrays(
            raw,
            load_split(args.split_dir, args.split_kinds[0], args.seeds[0]),
            args.smooth_alpha,
            transform_mode=args.transform_mode,
        )
        print(
            f"shared cache arrays prepared once for mode={args.transform_mode}: "
            f"hand={shared_arrays['hand_1520'].shape}, pool={shared_arrays['pool_1024'].shape}, "
            f"channel={shared_arrays['channel_tokens_32'].shape}, patch={shared_arrays['patch_tokens_32'].shape}",
            flush=True,
        )
    for split_kind in args.split_kinds:
        manifest[split_kind] = []
        for seed in args.seeds:
            split = load_split(args.split_dir, split_kind, seed)
            arrays = shared_arrays if shared_arrays is not None else build_adapted_arrays(
                raw, split, args.smooth_alpha, transform_mode=args.transform_mode
            )
            out_path = cache_path(args.cache_dir, split_kind, seed)
            save_cache_h5(out_path, arrays, raw, split, args.smooth_alpha, args.transform_mode)
            manifest[split_kind].append(str(out_path))
            print(
                f"cache {split_kind} seed={seed} mode={args.transform_mode}: hand={arrays['hand_1520'].shape}, "
                f"pool={arrays['pool_1024'].shape}, channel={arrays['channel_tokens_32'].shape}, "
                f"patch={arrays['patch_tokens_32'].shape}",
                flush=True,
            )
    write_json(args.output_root / "adapt_cache_manifest.json", manifest)


def baseline_fieldnames() -> list[str]:
    fields = ["split_kind", "seed", "model", "best_epoch", "best_val_acc", "train_seconds", "checkpoint", "cache_path", "split_path"]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            fields.append(f"{prefix}_{metric}")
    return fields


def train_baselines_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    result_dir = args.baseline_dir / "cache_models"
    checkpoint_root = result_dir / "checkpoints"
    device = resolve_device(args.device)
    rows: list[dict] = []
    run_config = {
        "models": args.models,
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
        "cache_dir": str(args.cache_dir),
        "result_dir": str(result_dir),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_jobs": args.n_jobs,
    }
    write_json(result_dir / "run_config.json", run_config)

    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(args.split_dir, split_kind, seed)
            cache = load_cache(cache_path(args.cache_dir, split_kind, seed))
            aggregate_by_trial = split_kind == "subject"
            print(f"\n[baseline] split={split_kind} seed={seed}", flush=True)

            for model_name in args.models:
                set_seed(seed)
                if model_name == "frozen_mlp":
                    checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / "frozen_mlp.pt"
                    model, best_epoch, best_val_acc, train_seconds = train_pool_mlp(
                        cache["pool_1024"],
                        cache["label"],
                        cache["global_trial_index"],
                        split,
                        args.batch_size,
                        seed,
                        device,
                        args.max_epochs,
                        args.patience,
                        args.lr,
                        args.weight_decay,
                        args.hidden_dim,
                        args.dropout,
                        checkpoint_path,
                    )
                    val_loader = make_pool_loaders(split, cache["pool_1024"], cache["label"], cache["global_trial_index"], args.batch_size, seed)[1]
                    test_loader = make_pool_loaders(split, cache["pool_1024"], cache["label"], cache["global_trial_index"], args.batch_size, seed)[2]
                    val_metrics = evaluate_pool_model(model, val_loader, device, aggregate_by_trial)
                    test_metrics = evaluate_pool_model(model, test_loader, device, aggregate_by_trial)
                    row = {
                        "split_kind": split_kind,
                        "seed": seed,
                        "model": "mdjpt_frozen_mlp",
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "train_seconds": round(train_seconds, 4),
                        "checkpoint": str(checkpoint_path),
                        "cache_path": str(cache_path(args.cache_dir, split_kind, seed)),
                        "split_path": str(split.path),
                    }
                    split_metrics_to_row(row, "val", val_metrics)
                    split_metrics_to_row(row, "test", test_metrics)
                    rows.append(row)
                    print(
                        f"  mdjpt_frozen_mlp: val_acc={val_metrics['window']['acc']*100:.2f} "
                        f"test_acc={test_metrics['window']['acc']*100:.2f} "
                        f"test_auc={test_metrics['window']['auroc']*100:.2f}",
                        flush=True,
                    )
                    continue

                X = cache["hand_1520"]
                y = cache["label"]
                trial_ids = cache["global_trial_index"]
                X_train, y_train = X[split.train], y[split.train]
                start = perf_counter()
                if model_name == "logistic_regression":
                    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=seed)
                elif model_name == "svm":
                    model = LinearSVC(C=1.0, dual=False, random_state=seed)
                elif model_name == "xgboost":
                    model = XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=args.n_jobs,
                        tree_method="hist",
                    )
                else:
                    raise ValueError(f"Unsupported baseline model: {model_name}")
                model.fit(X_train, y_train)
                train_seconds = perf_counter() - start
                row = {
                    "split_kind": split_kind,
                    "seed": seed,
                    "model": f"handcrafted_{model_name}",
                    "best_epoch": "",
                    "best_val_acc": "",
                    "train_seconds": round(train_seconds, 4),
                    "checkpoint": "",
                    "cache_path": str(cache_path(args.cache_dir, split_kind, seed)),
                    "split_path": str(split.path),
                }
                for prefix, indices in [("val", split.val), ("test", split.test)]:
                    score = collect_feature_scores(model_name, model, X[indices])
                    metrics = {"window": metrics_from_scores(y[indices], score), "trial": None}
                    if aggregate_by_trial:
                        t_true, t_score = aggregate_trials_scores(y[indices], score, trial_ids[indices])
                        metrics["trial"] = metrics_from_scores(t_true, t_score)
                    split_metrics_to_row(row, prefix, metrics)
                rows.append(row)
                print(
                    f"  handcrafted_{model_name}: val_acc={float(row['val_window_acc'])*100:.2f} "
                    f"test_acc={float(row['test_window_acc'])*100:.2f} "
                    f"test_auc={float(row['test_window_auroc'])*100:.2f}",
                    flush=True,
                )

                write_csv(result_dir / "all_results.csv", rows, baseline_fieldnames())

    write_csv(result_dir / "all_results.csv", rows, baseline_fieldnames())


def fusion_fieldnames() -> list[str]:
    fields = ["split_kind", "seed", "variant", "best_epoch", "best_val_acc", "train_seconds", "checkpoint", "cache_path", "split_path"]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            fields.append(f"{prefix}_{metric}")
    return fields


def train_fusion_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    device = resolve_device(args.device)
    result_dir = args.fusion_dir
    checkpoint_root = result_dir / "checkpoints"
    rows: list[dict] = []
    run_config = {
        "variant": "cross_attn_single_head",
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
        "cache_dir": str(args.cache_dir),
        "result_dir": str(result_dir),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    write_json(result_dir / "run_config.json", run_config)

    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(args.split_dir, split_kind, seed)
            cache = load_cache(cache_path(args.cache_dir, split_kind, seed))
            checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / "cross_attn_single_head.pt"
            set_seed(seed)
            model, best_epoch, best_val_acc, train_seconds = train_cross_attn_single_head(
                cache,
                split,
                args.batch_size,
                seed,
                device,
                args.max_epochs,
                args.patience,
                args.lr,
                args.weight_decay,
                args.hidden_dim,
                args.dropout,
                checkpoint_path,
            )
            aggregate_by_trial = split_kind == "subject"
            train_loader, val_loader, test_loader = make_cache_loaders(cache, split, args.batch_size, seed)
            del train_loader
            val_metrics = evaluate_cross_attn_single_head(model, val_loader, device, aggregate_by_trial)
            test_metrics = evaluate_cross_attn_single_head(model, test_loader, device, aggregate_by_trial)
            row = {
                "split_kind": split_kind,
                "seed": seed,
                "variant": "cross_attn_single_head",
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "train_seconds": round(train_seconds, 4),
                "checkpoint": str(checkpoint_path),
                "cache_path": str(cache_path(args.cache_dir, split_kind, seed)),
                "split_path": str(split.path),
            }
            split_metrics_to_row(row, "val", val_metrics)
            split_metrics_to_row(row, "test", test_metrics)
            rows.append(row)
            write_csv(result_dir / "all_results.csv", rows, fusion_fieldnames())
            print(
                f"[fusion] split={split_kind} seed={seed}: "
                f"val_acc={val_metrics['window']['acc']*100:.2f} "
                f"test_acc={test_metrics['window']['acc']*100:.2f} "
                f"test_auc={test_metrics['window']['auroc']*100:.2f}",
                flush=True,
            )


def moe_fieldnames() -> list[str]:
    fields = ["split_kind", "seed", "variant", "base_lr", "router_temperature", "best_epoch", "best_val_acc", "train_seconds", "checkpoint", "cache_path", "split_path"]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            fields.append(f"{prefix}_{metric}")
    for scope in ["val_router", "test_router"]:
        for metric in ["group_acc", "group_f1", "group_auroc", "mean_p_hc_true_hc", "mean_p_dep_true_dep", "mean_p_dep_true_hc", "mean_p_hc_true_dep"]:
            fields.append(f"{scope}_{metric}")
    for scope in ["val", "test"]:
        for group_name in ["HC", "DEP"]:
            for metric in ["acc", "f1", "auroc"]:
                fields.append(f"{scope}_{group_name}_{metric}")
    return fields


def _attach_router_group_metrics(row: dict, prefix: str, metrics: dict[str, dict | None]) -> None:
    router = metrics["router"]
    for metric in ["group_acc", "group_f1", "group_auroc", "mean_p_hc_true_hc", "mean_p_dep_true_dep", "mean_p_dep_true_hc", "mean_p_hc_true_dep"]:
        row[f"{prefix}_{metric}"] = "" if router is None or router[metric] is None else router[metric]
    per_group = metrics["per_group"]
    for group_name in ["HC", "DEP"]:
        group_metrics = per_group[group_name]
        for metric in ["acc", "f1", "auroc"]:
            key = f"{prefix.split('_')[0]}_{group_name}_{metric}"
            row[key] = "" if group_metrics is None or group_metrics[metric] is None else group_metrics[metric]


def screen_moe_lr_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    device = resolve_device(args.device)
    split = load_split(args.split_dir, "subject", args.screen_seed)
    cache = load_cache(cache_path(args.cache_dir, "subject", args.screen_seed))
    rows: list[dict] = []
    for base_lr in args.screen_lrs:
        val_accs = []
        val_aurocs = []
        for variant in args.screen_variants:
            checkpoint_path = args.moe_dir / "screen_checkpoints" / f"seed{args.screen_seed}" / f"{variant}_lr{base_lr:.0e}.pt"
            set_seed(args.screen_seed)
            model, best_epoch, best_val_acc, train_seconds = train_moe_variant(
                cache,
                split,
                variant,
                args.batch_size,
                args.screen_seed,
                device,
                args.max_epochs,
                args.patience,
                base_lr,
                args.weight_decay,
                args.hidden_dim,
                args.dropout,
                checkpoint_path,
            )
            _, val_loader, test_loader = make_cache_loaders(cache, split, args.batch_size, args.screen_seed)
            raw_val = evaluate_moe(model, val_loader, device, aggregate_by_trial=True, router_temperature=1.0)
            router_temp = fit_router_temperature(raw_val["router_logits"], raw_val["group_true_raw"])
            val_metrics = evaluate_moe(model, val_loader, device, aggregate_by_trial=True, router_temperature=router_temp)
            test_metrics = evaluate_moe(model, test_loader, device, aggregate_by_trial=True, router_temperature=router_temp)
            val_accs.append(float(val_metrics["window"]["acc"]))
            val_aurocs.append(float(val_metrics["window"]["auroc"]))
            row = {
                "base_lr": base_lr,
                "variant": variant,
                "best_epoch": best_epoch,
                "best_val_acc_raw": best_val_acc,
                "train_seconds": round(train_seconds, 4),
                "router_temperature": router_temp,
                "val_window_acc": val_metrics["window"]["acc"],
                "val_window_auroc": val_metrics["window"]["auroc"],
                "test_window_acc": test_metrics["window"]["acc"],
                "test_window_auroc": test_metrics["window"]["auroc"],
                "val_trial_acc": "" if val_metrics["trial"] is None else val_metrics["trial"]["acc"],
                "val_trial_auroc": "" if val_metrics["trial"] is None else val_metrics["trial"]["auroc"],
                "test_trial_acc": "" if test_metrics["trial"] is None else test_metrics["trial"]["acc"],
                "test_trial_auroc": "" if test_metrics["trial"] is None else test_metrics["trial"]["auroc"],
                "checkpoint": str(checkpoint_path),
            }
            rows.append(row)
            print(
                f"[screen] lr={base_lr:.0e} variant={variant}: "
                f"val_acc={val_metrics['window']['acc']*100:.2f} "
                f"val_auc={val_metrics['window']['auroc']*100:.2f} "
                f"router_T={router_temp:.2f}",
                flush=True,
            )

    fieldnames = [
        "base_lr",
        "variant",
        "best_epoch",
        "best_val_acc_raw",
        "train_seconds",
        "router_temperature",
        "val_window_acc",
        "val_window_auroc",
        "test_window_acc",
        "test_window_auroc",
        "val_trial_acc",
        "val_trial_auroc",
        "test_trial_acc",
        "test_trial_auroc",
        "checkpoint",
    ]
    screen_path = args.moe_dir / "moe_lr_screen_subject_seed42.csv"
    write_csv(screen_path, rows, fieldnames)

    grouped = {}
    for row in rows:
        key = float(row["base_lr"])
        grouped.setdefault(key, {"val_acc": [], "val_auroc": []})
        grouped[key]["val_acc"].append(float(row["val_window_acc"]))
        grouped[key]["val_auroc"].append(float(row["val_window_auroc"]))
    best_lr = None
    best_key = None
    for lr in args.screen_lrs:
        stats = grouped[float(lr)]
        key = (
            float(np.mean(stats["val_acc"])),
            float(np.mean(stats["val_auroc"])),
            -float(lr),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_lr = float(lr)
    selected = {
        "selected_base_lr": best_lr,
        "screen_seed": args.screen_seed,
        "screen_variants": args.screen_variants,
        "selection_rule": "max mean(val_window_acc), tie mean(val_window_auroc), tie smaller lr",
    }
    write_json(args.moe_dir / "selected_lr.json", selected)
    print(f"Selected MoE base lr: {best_lr}", flush=True)


def resolve_moe_base_lr(args: argparse.Namespace) -> float:
    if args.base_lr is not None:
        return float(args.base_lr)
    selected_path = args.moe_dir / "selected_lr.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected lr file: {selected_path}")
    with selected_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["selected_base_lr"])


def train_moe_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    device = resolve_device(args.device)
    base_lr = resolve_moe_base_lr(args)
    rows: list[dict] = []
    run_config = {
        "variants": args.variants,
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
        "cache_dir": str(args.cache_dir),
        "base_lr": base_lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    write_json(args.moe_dir / "run_config.json", run_config)
    checkpoint_root = args.moe_dir / "checkpoints"
    router_metrics_rows: list[dict] = []

    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(args.split_dir, split_kind, seed)
            cache = load_cache(cache_path(args.cache_dir, split_kind, seed))
            aggregate_by_trial = split_kind == "subject"
            print(f"\n[moe] split={split_kind} seed={seed} base_lr={base_lr}", flush=True)

            for variant in args.variants:
                checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / f"{variant}.pt"
                set_seed(seed)
                model, best_epoch, best_val_acc, train_seconds = train_moe_variant(
                    cache,
                    split,
                    variant,
                    args.batch_size,
                    seed,
                    device,
                    args.max_epochs,
                    args.patience,
                    base_lr,
                    args.weight_decay,
                    args.hidden_dim,
                    args.dropout,
                    checkpoint_path,
                )
                _, val_loader, test_loader = make_cache_loaders(cache, split, args.batch_size, seed)
                raw_val = evaluate_moe(model, val_loader, device, aggregate_by_trial, router_temperature=1.0)
                router_temperature = fit_router_temperature(raw_val["router_logits"], raw_val["group_true_raw"])
                val_metrics = evaluate_moe(model, val_loader, device, aggregate_by_trial, router_temperature=router_temperature)
                test_metrics = evaluate_moe(model, test_loader, device, aggregate_by_trial, router_temperature=router_temperature)
                row = {
                    "split_kind": split_kind,
                    "seed": seed,
                    "variant": variant,
                    "base_lr": base_lr,
                    "router_temperature": router_temperature,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "train_seconds": round(train_seconds, 4),
                    "checkpoint": str(checkpoint_path),
                    "cache_path": str(cache_path(args.cache_dir, split_kind, seed)),
                    "split_path": str(split.path),
                }
                split_metrics_to_row(row, "val", val_metrics)
                split_metrics_to_row(row, "test", test_metrics)
                _attach_router_group_metrics(row, "val_router", val_metrics)
                _attach_router_group_metrics(row, "test_router", test_metrics)
                rows.append(row)
                write_csv(args.moe_dir / "all_results.csv", rows, moe_fieldnames())
                for scope_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
                    router = metrics["router"]
                    if router is not None:
                        router_row = {
                            "split_kind": split_kind,
                            "seed": seed,
                            "variant": variant,
                            "scope": scope_name,
                            **{k: ("" if v is None else v) for k, v in router.items()},
                        }
                        router_metrics_rows.append(router_row)
                print(
                    f"  {variant}: val_acc={val_metrics['window']['acc']*100:.2f} "
                    f"test_acc={test_metrics['window']['acc']*100:.2f} "
                    f"test_auc={test_metrics['window']['auroc']*100:.2f} "
                    f"T={router_temperature:.2f}",
                    flush=True,
                )

    router_fieldnames = ["split_kind", "seed", "variant", "scope", "group_acc", "group_f1", "group_auroc", "mean_p_hc_true_hc", "mean_p_dep_true_dep", "mean_p_dep_true_hc", "mean_p_hc_true_dep"]
    write_csv(args.moe_dir / "router_metrics.csv", router_metrics_rows, router_fieldnames)


def load_rows_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def detect_transform_mode(cache_dir: Path) -> str:
    for split_kind in ["subject", "segment"]:
        split_cache_dir = cache_dir / split_kind
        if not split_cache_dir.exists():
            continue
        for path in sorted(split_cache_dir.glob("*.h5")):
            with h5py.File(path, "r") as handle:
                return str(handle.attrs.get("transform_mode", DEFAULT_TRANSFORM_MODE))
    return DEFAULT_TRANSFORM_MODE


def unified_method_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    baseline_rows = load_rows_if_exists(args.baseline_dir / "cache_models" / "all_results.csv")
    for row in baseline_rows:
        rows.append(
            {
                "split_kind": row["split_kind"],
                "seed": int(row["seed"]),
                "method": row["model"],
                **row,
            }
        )

    fullft_rows = load_rows_if_exists(args.fullft_dir / "all_results.csv")
    for row in fullft_rows:
        method = "mdjpt_full_finetune" if row["mode"] == "full_finetune" else f"mdjpt_{row['mode']}"
        rows.append(
            {
                "split_kind": row["split_kind"],
                "seed": int(row["seed"]),
                "method": method,
                **row,
            }
        )

    fusion_rows = load_rows_if_exists(args.fusion_dir / "all_results.csv")
    for row in fusion_rows:
        rows.append(
            {
                "split_kind": row["split_kind"],
                "seed": int(row["seed"]),
                "method": row["variant"],
                **row,
            }
        )

    moe_rows = load_rows_if_exists(args.moe_dir / "all_results.csv")
    for row in moe_rows:
        rows.append(
            {
                "split_kind": row["split_kind"],
                "seed": int(row["seed"]),
                "method": row["variant"],
                **row,
            }
        )
    return rows


def summarize_command(args: argparse.Namespace) -> None:
    ensure_dirs(args)
    all_rows = unified_method_rows(args)
    if not all_rows:
        raise RuntimeError("No result rows found to summarize.")

    per_seed_fields = sorted({key for row in all_rows for key in row.keys()})
    write_csv(args.comparison_dir / "per_seed_results.csv", all_rows, per_seed_fields)

    summary_rows = []
    best_rows = []
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in all_rows:
        by_pair.setdefault((row["split_kind"], row["method"]), []).append(row)

    for (split_kind, method), rows in sorted(by_pair.items()):
        summary_row = {"split_kind": split_kind, "method": method}
        for metric_key in ["test_window_acc", "test_window_f1", "test_window_auroc"]:
            values = [parse_float(row.get(metric_key, "")) for row in rows]
            values = [v for v in values if v is not None]
            stats = summary_stats(values)
            summary_row[metric_key] = fmt(stats)
        if split_kind == "subject":
            for metric_key in ["test_trial_acc", "test_trial_f1", "test_trial_auroc"]:
                values = [parse_float(row.get(metric_key, "")) for row in rows]
                values = [v for v in values if v is not None]
                summary_row[metric_key] = fmt(summary_stats(values)) if values else "-"
        else:
            summary_row["test_trial_acc"] = "-"
            summary_row["test_trial_f1"] = "-"
            summary_row["test_trial_auroc"] = "-"
        summary_rows.append(summary_row)

        best = sorted(
            rows,
            key=lambda item: (
                parse_float(item.get("test_window_acc", "")) or -1.0,
                parse_float(item.get("test_window_auroc", "")) or -1.0,
                -int(item["seed"]),
            ),
            reverse=True,
        )[0]
        best_rows.append(
            {
                "split_kind": split_kind,
                "method": method,
                "best_seed": int(best["seed"]),
                "test_window_acc": best.get("test_window_acc", ""),
                "test_window_f1": best.get("test_window_f1", ""),
                "test_window_auroc": best.get("test_window_auroc", ""),
                "test_trial_acc": best.get("test_trial_acc", ""),
                "test_trial_f1": best.get("test_trial_f1", ""),
                "test_trial_auroc": best.get("test_trial_auroc", ""),
            }
        )

    write_csv(
        args.comparison_dir / "mean_std_summary.csv",
        summary_rows,
        ["split_kind", "method", "test_window_acc", "test_window_f1", "test_window_auroc", "test_trial_acc", "test_trial_f1", "test_trial_auroc"],
    )
    write_csv(
        args.comparison_dir / "best_single_seed.csv",
        best_rows,
        ["split_kind", "method", "best_seed", "test_window_acc", "test_window_f1", "test_window_auroc", "test_trial_acc", "test_trial_f1", "test_trial_auroc"],
    )

    manifest = {
        "output_root": str(args.output_root),
        "split_dir": str(args.split_dir),
        "cache_dir": str(args.cache_dir),
        "baseline_dir": str(args.baseline_dir),
        "fusion_dir": str(args.fusion_dir),
        "moe_dir": str(args.moe_dir),
        "fullft_dir": str(args.fullft_dir),
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
    }
    write_json(args.comparison_dir / "run_manifest.json", manifest)

    summary_lookup = {(row["split_kind"], row["method"]): row for row in summary_rows}
    best_lookup = {(row["split_kind"], row["method"]): row for row in best_rows}
    transform_mode = detect_transform_mode(args.cache_dir)
    has_full_finetune = any(row["method"] == "mdjpt_full_finetune" for row in all_rows)
    seed_list = ", ".join(str(seed) for seed in args.seeds)
    screen_seed = args.seeds[0] if args.seeds else 42
    selected_lr_path = args.moe_dir / "selected_lr.json"
    if selected_lr_path.exists():
        try:
            with selected_lr_path.open("r", encoding="utf-8") as handle:
                selected_payload = json.load(handle)
            screen_seed = int(selected_payload.get("screen_seed", screen_seed))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    subject_methods = [
        "handcrafted_logistic_regression",
        "handcrafted_svm",
        "handcrafted_xgboost",
        "mdjpt_frozen_mlp",
        "mdjpt_full_finetune",
        "cross_attn_single_head",
        "cross_attn_moe_unsup",
        "cross_attn_moe_sup_l03",
        "cross_attn_moe_sup_l05",
        "cross_attn_moe_token_sup_l03",
        "cross_attn_moe_token_sup_l05",
    ]
    if transform_mode == "unsafe_global_norm_smooth_all":
        title = "# Balanced 8:1:1 Unsafe Pre-Split Frozen Report"
        mainline_name = "`unsafe_global_norm_smooth_all + frozen mdJPT + handcrafted features + CrossAttn + single head`"
        transform_note = (
            "- `unsafe_global_norm_smooth_all`: first do global z-score and causal EWMA smoothing on all segments, "
            "then apply the balanced 8:1:1 split"
        )
        protocol_note = (
            "- This is an intentionally unsafe stress test: pre-split smoothing/global normalization can inflate "
            "especially the segment-split results"
        )
    elif transform_mode == "presplit_subject_zscore_smooth_all":
        title = "# Balanced 8:1:1 Pre-Split Subject-Normalized Frozen MoE Report"
        mainline_name = (
            "`teacher-approved pre-split per-subject z-score + trial-wise smoothing + "
            "frozen mdJPT + handcrafted features + CrossAttn + MoE`"
        )
        transform_note = (
            "- `presplit_subject_zscore_smooth_all`: first apply per-subject z-score on the full dataset, "
            "then apply trial-wise causal EWMA smoothing (`alpha=0.65`), and finally apply the balanced splits"
        )
        protocol_note = "- MoE is the mainline in this protocol; single-head CrossAttn is kept as a fusion ablation"
    elif transform_mode == "presplit_subject_zscore_all":
        title = "# Balanced 8:1:1 Pre-Split Subject Z-Score-Only Frozen Report"
        mainline_name = "`pre-split per-subject z-score only + frozen mdJPT + handcrafted features + CrossAttn + MoE`"
        transform_note = (
            "- `presplit_subject_zscore_all`: first apply per-subject z-score on the full dataset, "
            "then directly apply balanced splits (no trial-wise smoothing)"
        )
        protocol_note = "- This ablation isolates subject-level normalization without temporal smoothing"
    elif transform_mode == "presplit_subject_smooth_all":
        title = "# Balanced 8:1:1 Pre-Split Smoothing-Only Frozen Report"
        mainline_name = "`pre-split trial-wise smoothing only + frozen mdJPT + handcrafted features + CrossAttn + MoE`"
        transform_note = (
            "- `presplit_subject_smooth_all`: first apply trial-wise causal EWMA smoothing (`alpha=0.65`) on the full dataset, "
            "then directly apply balanced splits (no subject z-score)"
        )
        protocol_note = "- This ablation isolates temporal smoothing without per-subject normalization"
    elif transform_mode == "raw_no_preprocess":
        title = "# Balanced 8:1:1 Raw Frozen Feature Report"
        mainline_name = "`raw frozen mdJPT + handcrafted features + CrossAttn + MoE (no z-score, no smoothing)`"
        transform_note = (
            "- `raw_no_preprocess`: directly use the cached handcrafted features and frozen mdJPT features "
            "without any normalization or temporal smoothing"
        )
        protocol_note = "- This ablation isolates the no-preprocessing baseline against the other pre-split transforms"
    else:
        title = "# Balanced 8:1:1 Mainline Report"
        mainline_name = "`adapt_then_smooth_all + frozen mdJPT + handcrafted features + CrossAttn + single head`"
        transform_note = (
            "- `adapt_then_smooth_all`: split first, per-subject z-score inside each split, then causal EWMA "
            "smoothing inside each trial (`alpha=0.65`)"
        )
        protocol_note = "- Subject split is the primary protocol; segment split is supplementary only"
    lines = [
        title,
        "",
        "## Setup",
        "",
        "- Dataset: COMP4 training-set internal evaluation only",
        "- EEG preprocessing: resampled to 125 Hz, mapped to 60 channels, segmented into 5 s windows",
        f"- Fixed seeds: `{seed_list}`",
        f"- Mainline: {mainline_name}",
        transform_note,
        protocol_note,
        "",
        "## Subject Split Main Table",
        "",
        "| Method | Test Acc | Test F1 | Test AUROC | Test Trial Acc | Test Trial AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in subject_methods:
        row = summary_lookup.get(("subject", method))
        if row is None:
            continue
        lines.append(
            f"| {method} | {row['test_window_acc']} | {row['test_window_f1']} | {row['test_window_auroc']} | "
            f"{row['test_trial_acc']} | {row['test_trial_auroc']} |"
        )

    lines.extend(
        [
            "",
            "## Segment Split Supplementary Table",
            "",
            "| Method | Test Acc | Test F1 | Test AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in subject_methods:
        row = summary_lookup.get(("segment", method))
        if row is None:
            continue
        lines.append(f"| {method} | {row['test_window_acc']} | {row['test_window_f1']} | {row['test_window_auroc']} |")

    lines.extend(
        [
            "",
            "## Best Single Seed By Test Acc",
            "",
            "| Split | Method | Seed | Test Acc | Test F1 | Test AUROC | Test Trial AUROC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in subject_methods:
        for split_kind in ["subject", "segment"]:
            row = best_lookup.get((split_kind, method))
            if row is None:
                continue
            trial_auc = row["test_trial_auroc"] if split_kind == "subject" else "-"
            lines.append(
                f"| {split_kind} | {method} | {row['best_seed']} | {row['test_window_acc']} | "
                f"{row['test_window_f1']} | {row['test_window_auroc']} | {trial_auc} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `cross_attn_single_head` is the mainline fusion model.",
            f"- MoE variants use a small LR screening on `subject split + seed {screen_seed}`, then reuse the selected base LR for all requested seeds.",
            "- Router temperature scaling is fitted on the validation split only and reused for validation/test inference.",
            "- If MoE does not beat the single-head CrossAttn baseline, the report should treat it as a negative but informative ablation.",
        ]
    )
    if has_full_finetune:
        lines.insert(
            lines.index("- `cross_attn_single_head` is the mainline fusion model."),
            "- `mdjpt_full_finetune` is the only route that does not use the frozen-feature transform; it trains directly on raw EEG.",
        )
    (args.output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report to {args.output_root / 'REPORT.md'}", flush=True)


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.command == "make_splits":
        make_splits_command(args)
    elif args.command == "build_cache":
        build_cache_command(args)
    elif args.command == "train_baselines":
        train_baselines_command(args)
    elif args.command == "train_fusion":
        train_fusion_command(args)
    elif args.command == "screen_moe_lr":
        screen_moe_lr_command(args)
    elif args.command == "train_moe":
        train_moe_command(args)
    elif args.command == "summarize":
        summarize_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
