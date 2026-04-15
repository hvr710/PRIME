#!/usr/bin/env python3
"""Unsafe pre-split normalization/smoothing ablations for LoRA and MoE.

This script is intentionally for leakage stress tests, not for paper-main
evaluation.  It applies global normalization and trial-wise EWMA smoothing
before train/val/test split, then trains either:

- Step6 live mdJPT + LoRA CrossAttn; or
- Step7 cached mdJPT CrossAttn/MoE models.

For LoRA, ``unsafe_global_norm_smooth_all`` means handcrafted features and the
live EEG input are both transformed before split.  For MoE, it means handcrafted
features plus cached mdJPT pool/channel/patch representations are transformed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_sequence_ablation import (  # noqa: E402
    SplitSpec,
    global_normalize_array,
    load_split,
    presplit_causal_smooth_array,
)
from train_fusion_step5 import load_feature_matrix  # noqa: E402
from train_fusion_step6_lora import (  # noqa: E402
    DEFAULT_CKPT,
    EEGHandDataset,
    build_model,
    build_optimizer,
    evaluate as evaluate_lora,
    load_eeg_arrays,
    train_one as train_one_lora,
)
from train_step7_moe import (  # noqa: E402
    Step7Dataset,
    Step7MoEModel,
    VARIANT_CONFIGS as MOE_VARIANT_CONFIGS,
    assert_alignment,
    build_subject_group_labels,
    evaluate as evaluate_moe,
    load_embeddings,
    load_feature_bundle,
    train_one as train_one_moe,
)


DEFAULT_EEG_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_EMBED_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_SPLIT_DIR = DATA_DIR / "feature_baseline_results" / "splits"
DEFAULT_RESULT_DIR = REPO_ROOT / "outputs" / "unsafe_lora_moe_results"
DEFAULT_SEEDS = [42, 3407, 2025]
DEFAULT_SPLIT_KINDS = ["subject", "trial", "segment"]
DEFAULT_SEQUENCE_VARIANTS = ["unsafe_global_norm_smooth_hand", "unsafe_global_norm_smooth_all"]
DEFAULT_MOE_VARIANTS = ["cross_attn_single_head", "cross_attn_moe_sup_l05"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=["lora", "moe"], default=["lora", "moe"])
    parser.add_argument("--sequence-variants", nargs="+", default=DEFAULT_SEQUENCE_VARIANTS, choices=DEFAULT_SEQUENCE_VARIANTS)
    parser.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eeg-h5", type=Path, default=DEFAULT_EEG_H5)
    parser.add_argument("--embed-h5", type=Path, default=DEFAULT_EMBED_H5)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--moe-variants", nargs="+", default=DEFAULT_MOE_VARIANTS, choices=list(MOE_VARIANT_CONFIGS))
    parser.add_argument("--lora-last-k", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-target-scope", choices=["attention_only", "attention_plus_proj"], default="attention_only")
    parser.add_argument("--batch-size-lora", type=int, default=128)
    parser.add_argument("--batch-size-moe", type=int, default=256)
    parser.add_argument("--max-epochs-lora", type=int, default=20)
    parser.add_argument("--max-epochs-moe", type=int, default=80)
    parser.add_argument("--patience-lora", type=int, default=5)
    parser.add_argument("--patience-moe", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=5e-4)
    parser.add_argument("--moe-lr", type=float, default=1e-3)
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


def load_segment_index(feature_h5: Path) -> np.ndarray:
    with h5py.File(feature_h5, "r") as handle:
        return handle["segment_index"][:].astype(np.int64)


def load_subject_index(feature_h5: Path) -> np.ndarray:
    with h5py.File(feature_h5, "r") as handle:
        return handle["subject_index"][:].astype(np.int64)


def load_global_trial_index(feature_h5: Path) -> np.ndarray:
    with h5py.File(feature_h5, "r") as handle:
        return handle["global_trial_index"][:].astype(np.int64)


def transform_hand(
    hand_features: np.ndarray,
    sequence_variant: str,
    trial_ids: np.ndarray,
    segment_index: np.ndarray,
    smooth_alpha: float,
) -> np.ndarray:
    hand = global_normalize_array(hand_features)
    hand = presplit_causal_smooth_array(hand, trial_ids, segment_index, smooth_alpha)
    if sequence_variant not in DEFAULT_SEQUENCE_VARIANTS:
        raise ValueError(f"Unknown sequence variant: {sequence_variant}")
    return hand.astype(np.float32)


def transform_lora_inputs(
    sequence_variant: str,
    eeg: np.ndarray,
    hand_features: np.ndarray,
    trial_ids: np.ndarray,
    segment_index: np.ndarray,
    smooth_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    hand = transform_hand(hand_features, sequence_variant, trial_ids, segment_index, smooth_alpha)
    eeg_out = eeg.astype(np.float32, copy=True)
    if sequence_variant == "unsafe_global_norm_smooth_all":
        eeg_out = global_normalize_array(eeg_out)
        eeg_out = presplit_causal_smooth_array(eeg_out, trial_ids, segment_index, smooth_alpha)
    return eeg_out.astype(np.float32), hand


def transform_moe_inputs(
    sequence_variant: str,
    embeddings: dict[str, np.ndarray],
    hand_features: np.ndarray,
    trial_ids: np.ndarray,
    segment_index: np.ndarray,
    smooth_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hand = transform_hand(hand_features, sequence_variant, trial_ids, segment_index, smooth_alpha)
    pool = embeddings["pool"].astype(np.float32, copy=True)
    channel_tokens = embeddings["channel_tokens"].astype(np.float32, copy=True)
    patch_tokens = embeddings["patch_tokens"].astype(np.float32, copy=True)
    if sequence_variant == "unsafe_global_norm_smooth_all":
        pool = presplit_causal_smooth_array(global_normalize_array(pool), trial_ids, segment_index, smooth_alpha)
        channel_tokens = presplit_causal_smooth_array(
            global_normalize_array(channel_tokens), trial_ids, segment_index, smooth_alpha
        )
        patch_tokens = presplit_causal_smooth_array(
            global_normalize_array(patch_tokens), trial_ids, segment_index, smooth_alpha
        )
    return hand, pool.astype(np.float32), channel_tokens.astype(np.float32), patch_tokens.astype(np.float32)


def make_lora_loaders(
    split: SplitSpec,
    eeg: np.ndarray,
    hand: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = EEGHandDataset(eeg, hand, labels, trial_ids, split.train)
    valset = EEGHandDataset(eeg, hand, labels, trial_ids, split.val)
    testset = EEGHandDataset(eeg, hand, labels, trial_ids, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def make_moe_loaders(
    split: SplitSpec,
    hand: np.ndarray,
    pool: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    group_labels: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = Step7Dataset(pool, hand, channel_tokens, patch_tokens, labels, trial_ids, group_labels, split.train)
    valset = Step7Dataset(pool, hand, channel_tokens, patch_tokens, labels, trial_ids, group_labels, split.val)
    testset = Step7Dataset(pool, hand, channel_tokens, patch_tokens, labels, trial_ids, group_labels, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def lora_variant_name(last_k: int, rank: int, target_scope: str) -> str:
    suffix = "attnproj" if target_scope == "attention_plus_proj" else "attn"
    return f"cross_attn_lora_last{last_k}_r{rank}_{suffix}"


def put_metrics(row: dict, prefix: str, metrics: dict) -> None:
    for scope in ["window", "trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            value = "" if metrics[scope] is None else metrics[scope][metric]
            row[f"{prefix}_{scope}_{metric}"] = "" if value is None else value


def put_router_metrics(row: dict, prefix: str, metrics: dict) -> None:
    router = metrics.get("router")
    for key in [
        "group_acc",
        "group_f1",
        "group_auroc",
        "mean_p_hc_true_hc",
        "mean_p_dep_true_dep",
        "mean_p_dep_true_hc",
        "mean_p_hc_true_dep",
    ]:
        row[f"{prefix}_router_{key}"] = "" if router is None or router.get(key) is None else router[key]


def run_lora_one(
    split: SplitSpec,
    sequence_variant: str,
    eeg: np.ndarray,
    hand: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[dict, dict, int, float, float, dict]:
    train_loader, val_loader, test_loader = make_lora_loaders(
        split, eeg, hand, labels, trial_ids, args.batch_size_lora, split.seed
    )
    model, counts, _replaced_paths = build_model(
        checkpoint=args.checkpoint,
        device=device,
        hand_dim=int(hand.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        last_k=args.lora_last_k,
        rank=args.lora_rank,
        target_scope=args.lora_target_scope,
    )
    optimizer = build_optimizer(model, args.head_lr, args.lora_lr, args.weight_decay)
    t0 = perf_counter()
    best_epoch, best_val_acc = train_one_lora(
        model,
        train_loader,
        val_loader,
        device,
        args.max_epochs_lora,
        args.patience_lora,
        optimizer,
        checkpoint_path,
    )
    train_seconds = perf_counter() - t0
    aggregate_by_trial = split.split_kind in {"subject", "trial"}
    val_metrics = evaluate_lora(model, val_loader, device, aggregate_by_trial)
    test_metrics = evaluate_lora(model, test_loader, device, aggregate_by_trial)
    del model, optimizer
    torch.cuda.empty_cache()
    return val_metrics, test_metrics, best_epoch, best_val_acc, train_seconds, counts


def run_moe_one(
    split: SplitSpec,
    variant: str,
    hand: np.ndarray,
    pool: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    group_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[dict, dict, int, float, float]:
    train_loader, val_loader, test_loader = make_moe_loaders(
        split,
        hand,
        pool,
        channel_tokens,
        patch_tokens,
        labels,
        trial_ids,
        group_labels,
        args.batch_size_moe,
        split.seed,
    )
    model = Step7MoEModel(
        variant=variant,
        pool_dim=int(pool.shape[1]),
        hand_dim=int(hand.shape[1]),
        token_dim=int(channel_tokens.shape[2]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    t0 = perf_counter()
    best_epoch, best_val_acc = train_one_moe(
        model,
        variant,
        train_loader,
        val_loader,
        device,
        args.max_epochs_moe,
        args.patience_moe,
        args.moe_lr,
        args.weight_decay,
        checkpoint_path,
    )
    train_seconds = perf_counter() - t0
    aggregate_by_trial = split.split_kind in {"subject", "trial"}
    val_metrics = evaluate_moe(model, val_loader, device, aggregate_by_trial)
    test_metrics = evaluate_moe(model, test_loader, device, aggregate_by_trial)
    del model
    torch.cuda.empty_cache()
    return val_metrics, test_metrics, best_epoch, best_val_acc, train_seconds


def result_fieldnames() -> list[str]:
    fields = [
        "model_family",
        "model_variant",
        "sequence_variant",
        "split_kind",
        "seed",
        "best_epoch",
        "best_val_acc",
        "train_seconds",
        "checkpoint",
        "split_path",
        "total_params",
        "trainable_params",
        "lora_params",
    ]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            fields.append(f"{prefix}_{metric}")
    for prefix in ["val", "test"]:
        for key in [
            "group_acc",
            "group_f1",
            "group_auroc",
            "mean_p_hc_true_hc",
            "mean_p_dep_true_dep",
            "mean_p_dep_true_hc",
            "mean_p_hc_true_dep",
        ]:
            fields.append(f"{prefix}_router_{key}")
    return fields


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for family in sorted({row["model_family"] for row in rows}):
        summary[family] = {}
        family_rows = [row for row in rows if row["model_family"] == family]
        for split_kind in sorted({row["split_kind"] for row in family_rows}):
            summary[family][split_kind] = {}
            split_rows = [row for row in family_rows if row["split_kind"] == split_kind]
            keys = sorted({(row["model_variant"], row["sequence_variant"]) for row in split_rows})
            for model_variant, sequence_variant in keys:
                selected = [
                    row
                    for row in split_rows
                    if row["model_variant"] == model_variant and row["sequence_variant"] == sequence_variant
                ]
                entry_key = f"{model_variant}__{sequence_variant}"
                summary[family][split_kind][entry_key] = {}
                for metric_key in [
                    "val_window_acc",
                    "val_window_auroc",
                    "test_window_acc",
                    "test_window_f1",
                    "test_window_auroc",
                    "test_trial_acc",
                    "test_trial_auroc",
                    "test_router_group_acc",
                    "test_router_group_auroc",
                ]:
                    values = [row[metric_key] for row in selected if row.get(metric_key) not in ("", None)]
                    if values:
                        arr = np.asarray(values, dtype=np.float64)
                        summary[family][split_kind][entry_key][metric_key] = {
                            "mean": float(arr.mean()),
                            "std": float(arr.std(ddof=0)),
                        }
                    else:
                        summary[family][split_kind][entry_key][metric_key] = None
    return summary


def fmt(entry: dict | None) -> str:
    if entry is None:
        return "-"
    return f"{entry['mean'] * 100:.2f} ± {entry['std'] * 100:.2f}"


def write_readme(rows: list[dict], summary: dict, result_dir: Path, run_config: dict) -> None:
    lines = [
        "# Unsafe LoRA/MoE Ablation Results",
        "",
        "These runs intentionally use pre-split global normalization and trial-wise smoothing. Treat them as leakage stress tests, not as honest generalization estimates.",
        "",
        "## Variant Semantics",
        "",
        "- `unsafe_global_norm_smooth_hand`: handcrafted features are globally normalized over all segments and smoothed within full trials before split.",
        "- `unsafe_global_norm_smooth_all` for MoE: handcrafted features plus cached mdJPT pool/channel/patch tensors receive the same pre-split transform.",
        "- `unsafe_global_norm_smooth_all` for LoRA: handcrafted features plus live EEG inputs receive the same pre-split transform before mdJPT forward.",
        "",
        "## Summary",
        "",
        "| Family | Split | Model | Sequence Variant | Test Acc | Test F1 | Test AUROC | Test Trial AUROC | Router Acc | Router AUROC |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, family_summary in summary.items():
        for split_kind, split_summary in family_summary.items():
            for entry_key, metrics in split_summary.items():
                model_variant, sequence_variant = entry_key.split("__", 1)
                lines.append(
                    f"| {family} | {split_kind} | {model_variant} | {sequence_variant} | "
                    f"{fmt(metrics['test_window_acc'])} | "
                    f"{fmt(metrics['test_window_f1'])} | "
                    f"{fmt(metrics['test_window_auroc'])} | "
                    f"{fmt(metrics['test_trial_auroc'])} | "
                    f"{fmt(metrics['test_router_group_acc'])} | "
                    f"{fmt(metrics['test_router_group_auroc'])} |"
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


def save_outputs(rows: list[dict], result_dir: Path, run_config: dict) -> None:
    write_csv(rows, result_dir / "all_results.csv")
    summary = summarize(rows)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(rows, summary, result_dir, run_config)


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    hand_features, _feature_names, feature_meta = load_feature_matrix(args.feature_h5)
    segment_index = load_segment_index(args.feature_h5)
    subject_index = load_subject_index(args.feature_h5)
    trial_ids = load_global_trial_index(args.feature_h5)
    labels = feature_meta["label"].astype(np.int64)

    run_config = {
        "families": args.families,
        "sequence_variants": args.sequence_variants,
        "split_kinds": args.split_kinds,
        "seeds": args.seeds,
        "eeg_h5": str(args.eeg_h5),
        "embed_h5": str(args.embed_h5),
        "feature_h5": str(args.feature_h5),
        "split_dir": str(args.split_dir),
        "checkpoint": str(args.checkpoint),
        "result_dir": str(args.result_dir),
        "moe_variants": args.moe_variants,
        "lora_last_k": args.lora_last_k,
        "lora_rank": args.lora_rank,
        "lora_target_scope": args.lora_target_scope,
        "batch_size_lora": args.batch_size_lora,
        "batch_size_moe": args.batch_size_moe,
        "max_epochs_lora": args.max_epochs_lora,
        "max_epochs_moe": args.max_epochs_moe,
        "patience_lora": args.patience_lora,
        "patience_moe": args.patience_moe,
        "head_lr": args.head_lr,
        "lora_lr": args.lora_lr,
        "moe_lr": args.moe_lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "smooth_alpha": args.smooth_alpha,
        "device": str(device),
        "note": "Intentional leakage stress test; do not use as primary generalization result.",
    }
    (args.result_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict] = []
    checkpoint_root = args.result_dir / "checkpoints"

    transformed_lora: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "lora" in args.families:
        eeg_data = load_eeg_arrays(args.eeg_h5)
        if not np.array_equal(eeg_data["label"], labels):
            raise ValueError("Label mismatch between EEG H5 and feature H5")
        print(f"Loaded LoRA EEG={eeg_data['eeg'].shape}, hand={hand_features.shape}, device={device}", flush=True)
        for sequence_variant in args.sequence_variants:
            transformed_lora[sequence_variant] = transform_lora_inputs(
                sequence_variant,
                eeg_data["eeg"],
                hand_features,
                trial_ids,
                segment_index,
                args.smooth_alpha,
            )

    transformed_moe: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    group_labels: np.ndarray | None = None
    embeddings: dict[str, np.ndarray] | None = None
    if "moe" in args.families:
        embeddings = load_embeddings(args.embed_h5)
        _features_again, _feature_names_again, feature_meta_full = load_feature_bundle(args.feature_h5)
        assert_alignment(embeddings, feature_meta_full)
        group_labels = build_subject_group_labels(embeddings["subject_index"], feature_meta_full["subject_groups"])
        if not np.array_equal(embeddings["subject_index"], subject_index):
            raise ValueError("Subject index mismatch between embedding H5 and feature H5")
        print(
            f"Loaded MoE pool={embeddings['pool'].shape}, channel={embeddings['channel_tokens'].shape}, "
            f"patch={embeddings['patch_tokens'].shape}, hand={hand_features.shape}, device={device}",
            flush=True,
        )
        for sequence_variant in args.sequence_variants:
            transformed_moe[sequence_variant] = transform_moe_inputs(
                sequence_variant,
                embeddings,
                hand_features,
                trial_ids,
                segment_index,
                args.smooth_alpha,
            )

    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(split_kind, seed, args.split_dir, labels, trial_ids)
            print(
                f"\n=== split={split_kind} seed={seed} train/val/test="
                f"{len(split.train)}/{len(split.val)}/{len(split.test)} ===",
                flush=True,
            )

            if "moe" in args.families:
                assert group_labels is not None
                for sequence_variant in args.sequence_variants:
                    hand, pool, channel_tokens, patch_tokens = transformed_moe[sequence_variant]
                    for model_variant in args.moe_variants:
                        set_seed(seed)
                        checkpoint_path = (
                            checkpoint_root
                            / "moe"
                            / split_kind
                            / f"seed{seed}"
                            / sequence_variant
                            / f"{model_variant}.pt"
                        )
                        val_metrics, test_metrics, best_epoch, best_val_acc, train_seconds = run_moe_one(
                            split,
                            model_variant,
                            hand,
                            pool,
                            channel_tokens,
                            patch_tokens,
                            labels,
                            trial_ids,
                            group_labels,
                            args,
                            device,
                            checkpoint_path,
                        )
                        row = {
                            "model_family": "moe",
                            "model_variant": model_variant,
                            "sequence_variant": sequence_variant,
                            "split_kind": split_kind,
                            "seed": seed,
                            "best_epoch": best_epoch,
                            "best_val_acc": best_val_acc,
                            "train_seconds": round(train_seconds, 4),
                            "checkpoint": str(checkpoint_path),
                            "split_path": str(split.path),
                            "total_params": "",
                            "trainable_params": "",
                            "lora_params": "",
                        }
                        put_metrics(row, "val", val_metrics)
                        put_metrics(row, "test", test_metrics)
                        put_router_metrics(row, "val", val_metrics)
                        put_router_metrics(row, "test", test_metrics)
                        rows.append(row)
                        save_outputs(rows, args.result_dir, run_config)
                        print(
                            f"moe/{model_variant}/{sequence_variant}: "
                            f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                            f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                            f"test_auc={float(test_metrics['window']['auroc'])*100:.2f}, "
                            f"time={train_seconds:.1f}s",
                            flush=True,
                        )

            if "lora" in args.families:
                model_variant = lora_variant_name(args.lora_last_k, args.lora_rank, args.lora_target_scope)
                for sequence_variant in args.sequence_variants:
                    set_seed(seed)
                    eeg, hand = transformed_lora[sequence_variant]
                    checkpoint_path = (
                        checkpoint_root / "lora" / split_kind / f"seed{seed}" / sequence_variant / f"{model_variant}.pt"
                    )
                    val_metrics, test_metrics, best_epoch, best_val_acc, train_seconds, counts = run_lora_one(
                        split,
                        sequence_variant,
                        eeg,
                        hand,
                        labels,
                        trial_ids,
                        args,
                        device,
                        checkpoint_path,
                    )
                    row = {
                        "model_family": "lora",
                        "model_variant": model_variant,
                        "sequence_variant": sequence_variant,
                        "split_kind": split_kind,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "train_seconds": round(train_seconds, 4),
                        "checkpoint": str(checkpoint_path),
                        "split_path": str(split.path),
                        "total_params": counts["total_params"],
                        "trainable_params": counts["trainable_params"],
                        "lora_params": counts["lora_params"],
                    }
                    put_metrics(row, "val", val_metrics)
                    put_metrics(row, "test", test_metrics)
                    for key in result_fieldnames():
                        row.setdefault(key, "")
                    rows.append(row)
                    save_outputs(rows, args.result_dir, run_config)
                    print(
                        f"lora/{model_variant}/{sequence_variant}: "
                        f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                        f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                        f"test_auc={float(test_metrics['window']['auroc'])*100:.2f}, "
                        f"time={train_seconds:.1f}s",
                        flush=True,
                    )

    save_outputs(rows, args.result_dir, run_config)
    print(f"\nSaved unsafe LoRA/MoE ablation results to: {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
