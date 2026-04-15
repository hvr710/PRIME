#!/usr/bin/env python3
"""Ablate safe sequence normalization/smoothing for COMP4 CrossAttn fusion.

The key rule is: split first, then transform each split independently. This
keeps trial/subject sequence processing from leaking train information into
validation or test representations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from train_fusion_step5 import (
    FusionDataset,
    FusionModel,
    evaluate,
    load_embeddings,
    load_feature_matrix,
    train_one,
)


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_EMBED_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_SPLIT_DIR = DATA_DIR / "feature_baseline_results" / "splits"
DEFAULT_RESULT_DIR = DATA_DIR / "sequence_ablation_results"
DEFAULT_SEEDS = [42, 3407, 2025]
DEFAULT_SPLIT_KINDS = ["subject", "trial", "segment"]
DEFAULT_VARIANTS = [
    "baseline",
    "trial_smooth_hand",
    "trial_smooth_all",
    "subject_adapt_hand",
    "adapt_then_smooth_hand",
    "subject_adapt_all",
    "adapt_then_smooth_all",
    "unsafe_global_norm_hand",
    "unsafe_global_norm_smooth_hand",
    "unsafe_global_norm_smooth_all",
]


@dataclass
class SplitSpec:
    split_kind: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-h5", type=Path, default=DEFAULT_EMBED_H5)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=DEFAULT_VARIANTS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--smooth-alpha", type=float, default=0.65)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def load_existing_split(split_dir: Path, split_kind: str, seed: int) -> SplitSpec:
    path = split_dir / f"comp4_{split_kind}_seed{seed}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitSpec(
        split_kind=split_kind,
        seed=seed,
        train=np.asarray(payload["train_indices"], dtype=np.int64),
        val=np.asarray(payload["val_indices"], dtype=np.int64),
        test=np.asarray(payload["test_indices"], dtype=np.int64),
        path=str(path),
    )


def make_trial_split(labels: np.ndarray, global_trial_ids: np.ndarray, seed: int) -> SplitSpec:
    """Create a stratified trial split with no trial crossing train/val/test."""
    rng = np.random.default_rng(seed)
    train_trials: list[np.ndarray] = []
    val_trials: list[np.ndarray] = []
    test_trials: list[np.ndarray] = []
    unique_trials = np.unique(global_trial_ids)

    trial_labels = {}
    for trial_id in unique_trials:
        segment_labels = np.unique(labels[global_trial_ids == trial_id])
        if segment_labels.shape[0] != 1:
            raise ValueError(f"Trial {trial_id} has mixed labels: {segment_labels.tolist()}")
        trial_labels[int(trial_id)] = int(segment_labels[0])

    for label in sorted(set(trial_labels.values())):
        trials = np.asarray([tid for tid, y in trial_labels.items() if y == label], dtype=np.int64)
        rng.shuffle(trials)
        n_total = trials.shape[0]
        n_test = int(round(n_total * 0.1))
        n_val = int(round(n_total * 0.1))
        n_train = n_total - n_val - n_test
        train_trials.append(trials[:n_train])
        val_trials.append(trials[n_train : n_train + n_val])
        test_trials.append(trials[n_train + n_val :])

    train_trial_ids = np.concatenate(train_trials)
    val_trial_ids = np.concatenate(val_trials)
    test_trial_ids = np.concatenate(test_trials)
    return SplitSpec(
        split_kind="trial",
        seed=seed,
        train=np.flatnonzero(np.isin(global_trial_ids, train_trial_ids)).astype(np.int64),
        val=np.flatnonzero(np.isin(global_trial_ids, val_trial_ids)).astype(np.int64),
        test=np.flatnonzero(np.isin(global_trial_ids, test_trial_ids)).astype(np.int64),
        path=f"in_memory_trial_seed{seed}",
    )


def load_split(split_kind: str, seed: int, split_dir: Path, labels: np.ndarray, trial_ids: np.ndarray) -> SplitSpec:
    if split_kind == "trial":
        return make_trial_split(labels, trial_ids, seed)
    return load_existing_split(split_dir, split_kind, seed)


def fit_train_scaler(hand: np.ndarray, split: SplitSpec) -> np.ndarray:
    scaler = StandardScaler()
    out = np.empty_like(hand, dtype=np.float32)
    out[split.train] = scaler.fit_transform(hand[split.train]).astype(np.float32)
    out[split.val] = scaler.transform(hand[split.val]).astype(np.float32)
    out[split.test] = scaler.transform(hand[split.test]).astype(np.float32)
    return out


def splitwise_subject_adapt_array(values: np.ndarray, split: SplitSpec, subject_ids: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Unsupervised per-subject z-score inside each split independently.

    Works for both 2D pooled/handcrafted features and 3D token tensors. The
    normalization statistics are computed over segments from one subject within
    one split only, while preserving feature/token dimensions.
    """
    out = values.copy()
    for split_indices in [split.train, split.val, split.test]:
        for subject_id in np.unique(subject_ids[split_indices]):
            idx = split_indices[subject_ids[split_indices] == subject_id]
            mean = out[idx].mean(axis=0, keepdims=True)
            std = out[idx].std(axis=0, keepdims=True)
            out[idx] = (out[idx] - mean) / (std + eps)
    return out.astype(np.float32)


def splitwise_subject_adapt(hand: np.ndarray, split: SplitSpec, subject_ids: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return splitwise_subject_adapt_array(hand, split, subject_ids, eps=eps)


def splitwise_subject_adapt_all(
    hand: np.ndarray,
    pool: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    split: SplitSpec,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        splitwise_subject_adapt_array(hand, split, subject_ids),
        splitwise_subject_adapt_array(pool, split, subject_ids),
        splitwise_subject_adapt_array(channel_tokens, split, subject_ids),
        splitwise_subject_adapt_array(patch_tokens, split, subject_ids),
    )


def causal_smooth_array(
    values: np.ndarray,
    split: SplitSpec,
    trial_ids: np.ndarray,
    segment_index: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Causal EWMA within each trial and within each split independently."""
    out = values.copy()
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
    """Unsafe all-segment normalization before split."""
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / (std + eps)).astype(np.float32)


def presplit_causal_smooth_array(
    values: np.ndarray,
    trial_ids: np.ndarray,
    segment_index: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Unsafe causal EWMA over full trials before train/val/test split."""
    out = values.copy()
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


def transform_inputs(
    variant: str,
    split: SplitSpec,
    hand_scaled: np.ndarray,
    pool: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    segment_index: np.ndarray,
    smooth_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hand = hand_scaled.astype(np.float32, copy=True)
    pool_out = pool.astype(np.float32, copy=True)
    channel_out = channel_tokens.astype(np.float32, copy=True)
    patch_out = patch_tokens.astype(np.float32, copy=True)

    if variant == "baseline":
        return hand, pool_out, channel_out, patch_out
    if variant == "subject_adapt_hand":
        hand = splitwise_subject_adapt(hand, split, subject_ids)
        return hand, pool_out, channel_out, patch_out
    if variant == "trial_smooth_hand":
        hand = causal_smooth_array(hand, split, trial_ids, segment_index, smooth_alpha)
        return hand, pool_out, channel_out, patch_out
    if variant == "adapt_then_smooth_hand":
        hand = splitwise_subject_adapt(hand, split, subject_ids)
        hand = causal_smooth_array(hand, split, trial_ids, segment_index, smooth_alpha)
        return hand, pool_out, channel_out, patch_out
    if variant == "subject_adapt_all":
        hand, pool_out, channel_out, patch_out = splitwise_subject_adapt_all(
            hand, pool_out, channel_out, patch_out, split, subject_ids
        )
        return hand, pool_out, channel_out, patch_out
    if variant == "adapt_then_smooth_all":
        hand, pool_out, channel_out, patch_out = splitwise_subject_adapt_all(
            hand, pool_out, channel_out, patch_out, split, subject_ids
        )
        hand = causal_smooth_array(hand, split, trial_ids, segment_index, smooth_alpha)
        pool_out = causal_smooth_array(pool_out, split, trial_ids, segment_index, smooth_alpha)
        channel_out = causal_smooth_array(channel_out, split, trial_ids, segment_index, smooth_alpha)
        patch_out = causal_smooth_array(patch_out, split, trial_ids, segment_index, smooth_alpha)
        return hand, pool_out, channel_out, patch_out
    if variant == "unsafe_global_norm_hand":
        hand = global_normalize_array(hand)
        return hand, pool_out, channel_out, patch_out
    if variant == "unsafe_global_norm_smooth_hand":
        hand = global_normalize_array(hand)
        hand = presplit_causal_smooth_array(hand, trial_ids, segment_index, smooth_alpha)
        return hand, pool_out, channel_out, patch_out
    if variant == "unsafe_global_norm_smooth_all":
        hand = global_normalize_array(hand)
        pool_out = global_normalize_array(pool_out)
        channel_out = global_normalize_array(channel_out)
        patch_out = global_normalize_array(patch_out)
        hand = presplit_causal_smooth_array(hand, trial_ids, segment_index, smooth_alpha)
        pool_out = presplit_causal_smooth_array(pool_out, trial_ids, segment_index, smooth_alpha)
        channel_out = presplit_causal_smooth_array(channel_out, trial_ids, segment_index, smooth_alpha)
        patch_out = presplit_causal_smooth_array(patch_out, trial_ids, segment_index, smooth_alpha)
        return hand, pool_out, channel_out, patch_out
    if variant == "trial_smooth_all":
        hand = causal_smooth_array(hand, split, trial_ids, segment_index, smooth_alpha)
        pool_out = causal_smooth_array(pool_out, split, trial_ids, segment_index, smooth_alpha)
        channel_out = causal_smooth_array(channel_out, split, trial_ids, segment_index, smooth_alpha)
        patch_out = causal_smooth_array(patch_out, split, trial_ids, segment_index, smooth_alpha)
        return hand, pool_out, channel_out, patch_out
    raise ValueError(f"Unknown sequence variant: {variant}")


def make_loaders(
    split: SplitSpec,
    hand: np.ndarray,
    pool: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = FusionDataset(pool, hand, channel_tokens, patch_tokens, labels, trial_ids, split.train)
    valset = FusionDataset(pool, hand, channel_tokens, patch_tokens, labels, trial_ids, split.val)
    testset = FusionDataset(pool, hand, channel_tokens, patch_tokens, labels, trial_ids, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def row_metrics(row: dict, prefix: str, metrics: dict) -> None:
    for scope in ["window", "trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            key = f"{prefix}_{scope}_{metric}"
            row[key] = "" if metrics[scope] is None else metrics[scope][metric]


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "split_kind",
        "seed",
        "sequence_variant",
        "model_variant",
        "best_epoch",
        "best_val_acc",
        "train_seconds",
        "checkpoint",
        "split_path",
    ]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            fieldnames.append(f"{prefix}_{metric}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for split_kind in sorted({row["split_kind"] for row in rows}):
        summary[split_kind] = {}
        for variant in sorted({row["sequence_variant"] for row in rows}):
            selected = [row for row in rows if row["split_kind"] == split_kind and row["sequence_variant"] == variant]
            if not selected:
                continue
            summary[split_kind][variant] = {"window": {}, "trial": {}}
            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                values = np.asarray([float(row[f"test_window_{metric}"]) for row in selected], dtype=np.float64)
                summary[split_kind][variant]["window"][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
                trial_values = [row[f"test_trial_{metric}"] for row in selected if row[f"test_trial_{metric}"] != ""]
                if trial_values:
                    trial_values = np.asarray([float(value) for value in trial_values], dtype=np.float64)
                    summary[split_kind][variant]["trial"][metric] = {
                        "mean": float(trial_values.mean()),
                        "std": float(trial_values.std(ddof=0)),
                    }
                else:
                    summary[split_kind][variant]["trial"][metric] = None
    return summary


def fmt(metric: dict | None) -> str:
    if metric is None:
        return "-"
    return f"{metric['mean'] * 100:.2f} ± {metric['std'] * 100:.2f}"


def write_readme(summary: dict, result_dir: Path, run_config: dict) -> None:
    lines = [
        "# Sequence Normalization/Smoothing Ablation",
        "",
        "All variants use the cached Step-5 `cross_attn` fusion model. The key safety rule is split first, then normalize/smooth inside each split only.",
        "",
        "## Variants",
        "",
        "- `baseline`: train-fitted `StandardScaler` on handcrafted features, matching Step-5 CrossAttn.",
        "- `trial_smooth_hand`: causal EWMA smoothing of handcrafted features within each trial and split.",
        "- `trial_smooth_all`: causal EWMA smoothing of handcrafted features and mdJPT cached embeddings/tokens.",
        "- `subject_adapt_hand`: unsupervised per-subject z-score inside each split.",
        "- `adapt_then_smooth_hand`: per-subject z-score followed by causal trial smoothing.",
        "- `subject_adapt_all`: per-subject z-score on handcrafted features and mdJPT pooled/token features.",
        "- `adapt_then_smooth_all`: `subject_adapt_all` followed by causal smoothing of all features.",
        "- `unsafe_global_norm_hand`: all-segment handcrafted-feature z-score before split.",
        "- `unsafe_global_norm_smooth_hand`: `unsafe_global_norm_hand` followed by pre-split trial smoothing.",
        "- `unsafe_global_norm_smooth_all`: all-segment z-score and pre-split smoothing for handcrafted and mdJPT cached features.",
        "",
        "## Summary",
        "",
        "| Split | Variant | Window Acc | Window F1 | Window AUROC | Trial Acc | Trial AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split_kind, variants in summary.items():
        for variant, metrics in variants.items():
            lines.append(
                f"| {split_kind} | {variant} | "
                f"{fmt(metrics['window']['acc'])} | "
                f"{fmt(metrics['window']['f1'])} | "
                f"{fmt(metrics['window']['auroc'])} | "
                f"{fmt(metrics['trial']['acc'])} | "
                f"{fmt(metrics['trial']['auroc'])} |"
            )
    lines.extend(
        [
            "",
            "## Run Config",
            "",
            "```json",
            json.dumps(run_config, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.result_dir / "checkpoints"
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    embeddings = load_embeddings(args.embed_h5)
    hand_features, _, feature_meta = load_feature_matrix(args.feature_h5)
    labels = embeddings["label"]
    if not np.array_equal(labels, feature_meta["label"]):
        raise ValueError("Label mismatch between embedding H5 and feature H5")

    with h5py.File(args.feature_h5, "r") as handle:
        subject_ids = handle["subject_index"][:].astype(np.int64)
        segment_index = handle["segment_index"][:].astype(np.int64)
    trial_ids = embeddings["global_trial_index"].astype(np.int64)

    run_config = {
        "embed_h5": str(args.embed_h5),
        "feature_h5": str(args.feature_h5),
        "split_dir": str(args.split_dir),
        "result_dir": str(args.result_dir),
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
        "sequence_variants": args.variants,
        "model_variant": "cross_attn",
        "smooth_alpha": args.smooth_alpha,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    (args.result_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Loaded hand={hand_features.shape}, pool={embeddings['pool'].shape}, "
        f"channel={embeddings['channel_tokens'].shape}, patch={embeddings['patch_tokens'].shape}, device={device}",
        flush=True,
    )

    rows: list[dict] = []
    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(split_kind, seed, args.split_dir, labels, trial_ids)
            hand_scaled = fit_train_scaler(hand_features, split)
            print(f"\n=== split={split_kind} seed={seed} train/val/test={len(split.train)}/{len(split.val)}/{len(split.test)} ===")
            for sequence_variant in args.variants:
                set_seed(seed)
                hand_input = hand_features if sequence_variant.startswith("unsafe_") else hand_scaled
                hand, pool, channel_tokens, patch_tokens = transform_inputs(
                    sequence_variant,
                    split,
                    hand_input,
                    embeddings["pool"],
                    embeddings["channel_tokens"],
                    embeddings["patch_tokens"],
                    subject_ids,
                    trial_ids,
                    segment_index,
                    args.smooth_alpha,
                )
                train_loader, val_loader, test_loader = make_loaders(
                    split,
                    hand,
                    pool,
                    channel_tokens,
                    patch_tokens,
                    labels,
                    trial_ids,
                    args.batch_size,
                    seed,
                )
                model = FusionModel(
                    variant="cross_attn",
                    pool_dim=int(pool.shape[1]),
                    hand_dim=int(hand.shape[1]),
                    token_dim=int(channel_tokens.shape[2]),
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                )
                checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / f"{sequence_variant}.pt"
                t0 = perf_counter()
                best_epoch, best_val_acc = train_one(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.max_epochs,
                    args.patience,
                    args.lr,
                    args.weight_decay,
                    checkpoint_path,
                )
                train_seconds = perf_counter() - t0
                aggregate_by_trial = split_kind in {"subject", "trial"}
                val_metrics = evaluate(model, val_loader, device, aggregate_by_trial)
                test_metrics = evaluate(model, test_loader, device, aggregate_by_trial)
                row = {
                    "split_kind": split_kind,
                    "seed": seed,
                    "sequence_variant": sequence_variant,
                    "model_variant": "cross_attn",
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "train_seconds": round(train_seconds, 4),
                    "checkpoint": str(checkpoint_path),
                    "split_path": split.path,
                }
                row_metrics(row, "val", val_metrics)
                row_metrics(row, "test", test_metrics)
                rows.append(row)
                write_csv(rows, args.result_dir / "all_results.csv")
                summary = summarize(rows)
                (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                write_readme(summary, args.result_dir, run_config)
                print(
                    f"{sequence_variant}: best_epoch={best_epoch}, "
                    f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                    f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                    f"test_auc={test_metrics['window']['auroc']*100:.2f}, "
                    f"time={train_seconds:.1f}s",
                    flush=True,
                )

    summary = summarize(rows)
    (args.result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(summary, args.result_dir, run_config)
    print(f"\nSaved sequence ablation results to: {args.result_dir}")


if __name__ == "__main__":
    main()
