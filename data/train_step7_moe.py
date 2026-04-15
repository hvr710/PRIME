#!/usr/bin/env python3
"""Train Step-7 CrossAttn + MoE router experiments on fixed COMP4 splits."""

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
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from train_fusion_step5 import MLPHead, SplitSpec, load_feature_matrix, load_split, resolve_device, selected_split_files, set_seed


DEFAULT_EMBED_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_SPLIT_DIR = DATA_DIR / "feature_baseline_results" / "splits"
DEFAULT_RESULT_DIR = DATA_DIR / "step7_moe_results"
DEFAULT_SEEDS = [42, 3407, 2025]
DEFAULT_SPLIT_KINDS = ["subject", "segment"]

SCREEN_VARIANTS = [
    "cross_attn_single_head",
    "cross_attn_dual_expert_avg",
    "cross_attn_moe_unsup",
    "cross_attn_moe_sup_l03",
    "cross_attn_moe_sup_l05",
]
SUPERVISED_VARIANTS = ["cross_attn_moe_sup_l03", "cross_attn_moe_sup_l05"]
FULL_BASE_VARIANTS = [
    "cross_attn_single_head",
    "cross_attn_dual_expert_avg",
    "cross_attn_moe_unsup",
]
VARIANT_CONFIGS = {
    "cross_attn_single_head": {"mode": "single_head", "router_loss_weight": None},
    "cross_attn_dual_expert_avg": {"mode": "dual_expert_avg", "router_loss_weight": None},
    "cross_attn_moe_unsup": {"mode": "moe", "router_loss_weight": 0.0},
    "cross_attn_moe_sup_l03": {"mode": "moe", "router_loss_weight": 0.3},
    "cross_attn_moe_sup_l05": {"mode": "moe", "router_loss_weight": 0.5},
}


class Step7Dataset(Dataset):
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


class Step7MoEModel(nn.Module):
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
        self.mode = VARIANT_CONFIGS[variant]["mode"]
        self.hand_query = nn.Linear(hand_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)

        if self.mode == "single_head":
            self.classifier = MLPHead(token_dim, hidden_dim, dropout=dropout)
        else:
            self.expert_hc = MLPHead(token_dim, hidden_dim, dropout=dropout)
            self.expert_dep = MLPHead(token_dim, hidden_dim, dropout=dropout)
            if self.mode == "moe":
                self.router = nn.Sequential(
                    nn.Linear(pool_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),
                )

    def forward(
        self,
        pool: torch.Tensor,
        hand: torch.Tensor,
        channel_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        query = self.hand_query(hand).unsqueeze(1)
        tokens = torch.cat([channel_tokens, patch_tokens], dim=1)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))

        if self.mode == "single_head":
            emotion_logits = self.classifier(fused)
            return {
                "emotion_logits": emotion_logits,
                "router_logits": None,
                "router_prob": None,
            }

        logits_hc = self.expert_hc(fused)
        logits_dep = self.expert_dep(fused)
        if self.mode == "dual_expert_avg":
            emotion_logits = 0.5 * (logits_hc + logits_dep)
            return {
                "emotion_logits": emotion_logits,
                "router_logits": None,
                "router_prob": None,
            }

        router_logits = self.router(pool)
        router_prob = torch.softmax(router_logits, dim=1)
        emotion_logits = router_prob[:, :1] * logits_hc + router_prob[:, 1:] * logits_dep
        return {
            "emotion_logits": emotion_logits,
            "router_logits": router_logits,
            "router_prob": router_prob,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=["screen", "full", "all"], default="all")
    parser.add_argument("--embed-h5", type=Path, default=DEFAULT_EMBED_H5)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--screen-variants", nargs="+", default=SCREEN_VARIANTS, choices=SCREEN_VARIANTS)
    parser.add_argument("--full-variants", nargs="+", default=None, choices=SCREEN_VARIANTS)
    parser.add_argument("--selected-supervised-variant", type=str, default=None, choices=SUPERVISED_VARIANTS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "pool": handle["pool_1024"][:].astype(np.float32),
            "channel_tokens": handle["channel_tokens_32"][:].astype(np.float32),
            "patch_tokens": handle["patch_tokens_32"][:].astype(np.float32),
            "label": handle["label"][:].astype(np.int64),
            "subject_index": handle["subject_index"][:].astype(np.int64),
            "global_trial_index": handle["global_trial_index"][:].astype(np.int64),
        }


def load_feature_bundle(path: Path) -> tuple[np.ndarray, list[str], dict[str, np.ndarray | list[str]]]:
    features, feature_names, metadata = load_feature_matrix(path)
    with h5py.File(path, "r") as handle:
        metadata["global_trial_index"] = handle["global_trial_index"][:].astype(np.int64)
        metadata["subject_groups"] = decode_strings(handle["meta/subject_groups"][:])
    return features, feature_names, metadata


def build_subject_group_labels(subject_index: np.ndarray, subject_groups: list[str]) -> np.ndarray:
    if subject_index.max() >= len(subject_groups):
        raise ValueError(
            f"subject_index max={subject_index.max()} exceeds available subject_groups={len(subject_groups)}"
        )
    labels = np.empty(subject_index.shape[0], dtype=np.int64)
    for idx, subject_id in enumerate(subject_index):
        group_name = subject_groups[int(subject_id)].upper()
        if group_name.startswith("HC"):
            labels[idx] = 0
        elif group_name.startswith("DEP"):
            labels[idx] = 1
        else:
            raise ValueError(f"Unexpected subject group name: {subject_groups[int(subject_id)]!r}")
    return labels


def assert_alignment(embeddings: dict[str, np.ndarray], feature_meta: dict[str, np.ndarray | list[str]]) -> None:
    for key in ["label", "subject_index", "global_trial_index"]:
        if not np.array_equal(embeddings[key], feature_meta[key]):
            raise ValueError(f"Mismatch between embedding H5 and feature H5 for key={key}")


def maybe_auroc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, score))


def maybe_auprc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, score))


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


def aggregate_trials(y_true: np.ndarray, y_prob: np.ndarray, trial_ids: np.ndarray):
    unique_trials = np.unique(trial_ids)
    trial_true = []
    trial_prob = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        trial_true.append(int(y_true[mask][0]))
        trial_prob.append(y_prob[mask].mean(axis=0))
    return np.asarray(trial_true, dtype=np.int64), np.stack(trial_prob, axis=0)


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray | None]:
    model.eval()
    labels, probs, trial_ids, group_true, router_probs = [], [], [], [], []
    with torch.no_grad():
        for pool, hand, channel_tokens, patch_tokens, y, trial_id, group_label in loader:
            outputs = model(
                pool.to(device),
                hand.to(device),
                channel_tokens.to(device),
                patch_tokens.to(device),
            )
            logits = outputs["emotion_logits"]
            labels.append(y.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            trial_ids.append(trial_id.numpy())
            group_true.append(group_label.numpy())
            if outputs["router_prob"] is not None:
                router_probs.append(outputs["router_prob"].cpu().numpy())
    return {
        "label": np.concatenate(labels),
        "prob": np.concatenate(probs),
        "trial_id": np.concatenate(trial_ids),
        "group_true": np.concatenate(group_true),
        "router_prob": np.concatenate(router_probs) if router_probs else None,
    }


def router_metrics(group_true: np.ndarray, router_prob: np.ndarray | None) -> dict[str, float | int | None] | None:
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
        "n_samples": int(group_true.shape[0]),
    }


def per_group_emotion_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_true: np.ndarray,
) -> dict[str, dict[str, float | int | None] | None]:
    result = {}
    for group_name, group_id in [("HC", 0), ("DEP", 1)]:
        mask = group_true == group_id
        result[group_name] = metrics_from_probs_safe(y_true[mask], y_prob[mask]) if np.any(mask) else None
    return result


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, aggregate_by_trial: bool):
    predictions = collect_predictions(model, loader, device)
    y_true = predictions["label"]
    y_prob = predictions["prob"]
    trial_ids = predictions["trial_id"]
    group_true = predictions["group_true"]
    result = {
        "window": metrics_from_probs_safe(y_true, y_prob),
        "router": router_metrics(group_true, predictions["router_prob"]),
        "per_group": per_group_emotion_metrics(y_true, y_prob, group_true),
    }
    if aggregate_by_trial:
        trial_true, trial_prob = aggregate_trials(y_true, y_prob, trial_ids)
        result["trial"] = metrics_from_probs_safe(trial_true, trial_prob)
    else:
        result["trial"] = None
    return result


def make_loaders(
    split: SplitSpec,
    pool: np.ndarray,
    hand_scaled: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    group_labels: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = Step7Dataset(pool, hand_scaled, channel_tokens, patch_tokens, labels, trial_ids, group_labels, split.train)
    valset = Step7Dataset(pool, hand_scaled, channel_tokens, patch_tokens, labels, trial_ids, group_labels, split.val)
    testset = Step7Dataset(pool, hand_scaled, channel_tokens, patch_tokens, labels, trial_ids, group_labels, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def train_one(
    model: nn.Module,
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: Path,
):
    model.to(device)
    emotion_criterion = nn.CrossEntropyLoss()
    router_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    router_loss_weight = VARIANT_CONFIGS[variant]["router_loss_weight"]
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        for pool, hand, channel_tokens, patch_tokens, y, _trial, group_label in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pool.to(device),
                hand.to(device),
                channel_tokens.to(device),
                patch_tokens.to(device),
            )
            emotion_loss = emotion_criterion(outputs["emotion_logits"], y.to(device))
            loss = emotion_loss
            if outputs["router_logits"] is not None and router_loss_weight is not None and router_loss_weight > 0:
                router_loss = router_criterion(outputs["router_logits"], group_label.to(device))
                loss = loss + router_loss_weight * router_loss
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device, aggregate_by_trial=False)["window"]
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
    return best_epoch, best_val_acc


def write_rows_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_values(rows: list[dict], key: str) -> dict[str, float] | None:
    values = [row[key] for row in rows if row.get(key) not in ("", None)]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def summarize_full_rows(rows: list[dict], result_dir: Path) -> dict:
    summary = {}
    for split_kind in sorted({row["split_kind"] for row in rows}):
        summary[split_kind] = {}
        for variant in sorted({row["variant"] for row in rows}):
            selected = [row for row in rows if row["split_kind"] == split_kind and row["variant"] == variant]
            if not selected:
                continue
            summary[split_kind][variant] = {
                "val_window": {},
                "test_window": {},
                "val_trial": {},
                "test_trial": {},
            }
            for scope in ["val_window", "test_window", "val_trial", "test_trial"]:
                for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                    summary[split_kind][variant][scope][metric] = summary_values(selected, f"{scope}_{metric}")
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def fmt_summary(entry: dict | None) -> str:
    if entry is None:
        return "-"
    return f"{entry['mean'] * 100:.2f} ± {entry['std'] * 100:.2f}"


def router_summary(router_rows: list[dict], split_kind: str, variant: str, scope: str) -> dict[str, dict[str, float] | None]:
    selected = [
        row
        for row in router_rows
        if row["phase"] == "full" and row["split_kind"] == split_kind and row["variant"] == variant and row["scope"] == scope
    ]
    summary = {}
    for key in [
        "group_acc",
        "group_f1",
        "group_auroc",
        "mean_p_hc_true_hc",
        "mean_p_dep_true_dep",
        "mean_p_dep_true_hc",
        "mean_p_hc_true_dep",
    ]:
        summary[key] = summary_values(selected, key)
    return summary


def write_readme(
    result_dir: Path,
    run_config: dict,
    screen_rows: list[dict],
    selected_config: dict | None,
    full_summary: dict | None,
    router_rows: list[dict],
) -> None:
    lines = [
        "# Step 7 MoE Results",
        "",
        "Cached mdJPT embeddings + handcrafted features + CrossAttn fusion + subject-type-aware MoE routing.",
        "",
        "## Setup",
        "",
        f"- Embeddings: `{run_config['embed_h5']}`",
        f"- Handcrafted features: `{run_config['feature_h5']}`",
        f"- Split files: `{run_config['split_dir']}/comp4_{{subject,segment}}_seed*.json`",
        f"- Seeds: `{', '.join(str(seed) for seed in run_config['seeds'])}`",
        f"- Screening variants: `{', '.join(run_config['screen_variants'])}`",
        f"- Batch size: `{run_config['batch_size']}`, max epochs `{run_config['max_epochs']}`, patience `{run_config['patience']}`",
        f"- Optimizer: `AdamW`, lr `{run_config['lr']}`, weight decay `{run_config['weight_decay']}`",
        "",
    ]

    if screen_rows:
        lines.extend(
            [
                "## Screening Results",
                "",
                "| Variant | Val Trial AUROC | Val Window AUROC | Test Trial AUROC | Test Window AUROC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in screen_rows:
            val_trial_auroc = "-" if row["val_trial_auroc"] in ("", None) else f"{float(row['val_trial_auroc']) * 100:.2f}"
            test_trial_auroc = "-" if row["test_trial_auroc"] in ("", None) else f"{float(row['test_trial_auroc']) * 100:.2f}"
            lines.append(
                f"| {row['variant']} | {val_trial_auroc} | {float(row['val_window_auroc']) * 100:.2f} | "
                f"{test_trial_auroc} | {float(row['test_window_auroc']) * 100:.2f} |"
            )
        lines.append("")

    if selected_config is not None:
        lines.extend(
            [
                "## Selected Supervised MoE Variant",
                "",
                f"- `variant = {selected_config['selected_supervised_variant']}`",
                f"- `router_loss_weight = {selected_config['router_loss_weight']}`",
                "- Selected among supervised MoE variants by `val_trial_auroc`, then `val_window_auroc`, then `val_trial_acc`.",
                "",
            ]
        )

    if full_summary:
        lines.extend(
            [
                "## Full Summary",
                "",
                "| Split | Variant | Val Acc | Val AUROC | Test Acc | Test AUROC | Test Trial Acc | Test Trial AUROC |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split_kind, variants in full_summary.items():
            for variant, metrics in variants.items():
                lines.append(
                    f"| {split_kind} | {variant} | "
                    f"{fmt_summary(metrics['val_window']['acc'])} | "
                    f"{fmt_summary(metrics['val_window']['auroc'])} | "
                    f"{fmt_summary(metrics['test_window']['acc'])} | "
                    f"{fmt_summary(metrics['test_window']['auroc'])} | "
                    f"{fmt_summary(metrics['test_trial']['acc'])} | "
                    f"{fmt_summary(metrics['test_trial']['auroc'])} |"
                )
        lines.append("")

        if selected_config is not None:
            selected_variant = selected_config["selected_supervised_variant"]
            lines.extend(
                [
                    "## Router Summary",
                    "",
                    "| Split | Variant | Test Group Acc | Test Group AUROC | mean p_hc (true HC) | mean p_dep (true DEP) |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for split_kind in ["subject", "segment"]:
                router_stats = router_summary(router_rows, split_kind, selected_variant, "test")
                lines.append(
                    f"| {split_kind} | {selected_variant} | "
                    f"{fmt_summary(router_stats['group_acc'])} | "
                    f"{fmt_summary(router_stats['group_auroc'])} | "
                    f"{fmt_summary(router_stats['mean_p_hc_true_hc'])} | "
                    f"{fmt_summary(router_stats['mean_p_dep_true_dep'])} |"
                )
            lines.append("")

            lines.extend(
                [
                    "## Notes",
                    "",
                    "- `cross_attn_single_head` is the Step-5 strongest single-head baseline re-run under the Step-7 script.",
                    "- `cross_attn_dual_expert_avg` controls for extra parameters without routing.",
                    "- `cross_attn_moe_unsup` keeps the router trainable through emotion loss only (`lambda = 0`).",
                    "- Subject split remains the primary setting; segment split is a consistency check only.",
                    "- Some fixed subject-split folds contain no DEP samples in test, so router group AUROC can be blank on those folds.",
                ]
            )

    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def screen_selection_key(row: dict) -> tuple:
    val_trial_auroc = -np.inf if row["val_trial_auroc"] in ("", None) else float(row["val_trial_auroc"])
    val_window_auroc = -np.inf if row["val_window_auroc"] in ("", None) else float(row["val_window_auroc"])
    val_trial_acc = -np.inf if row["val_trial_acc"] in ("", None) else float(row["val_trial_acc"])
    simplicity_order = {
        "cross_attn_single_head": 0,
        "cross_attn_dual_expert_avg": 1,
        "cross_attn_moe_unsup": 2,
        "cross_attn_moe_sup_l03": 3,
        "cross_attn_moe_sup_l05": 4,
    }
    return (val_trial_auroc, val_window_auroc, val_trial_acc, -simplicity_order[row["variant"]])


def resolve_selected_config(args: argparse.Namespace, result_dir: Path) -> dict:
    if args.selected_supervised_variant is not None:
        return {
            "selected_supervised_variant": args.selected_supervised_variant,
            "router_loss_weight": VARIANT_CONFIGS[args.selected_supervised_variant]["router_loss_weight"],
        }
    selected_path = result_dir / "selected_config.json"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing selected_config.json at {selected_path}. Run workflow=screen or pass --selected-supervised-variant."
        )
    with selected_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_router_rows(
    rows: list[dict],
    phase: str,
    split_kind: str,
    seed: int,
    variant: str,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    for scope_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
        router = metrics["router"]
        row = {
            "phase": phase,
            "split_kind": split_kind,
            "seed": seed,
            "variant": variant,
            "scope": scope_name,
        }
        if router is None:
            for key in [
                "group_acc",
                "group_f1",
                "group_auroc",
                "mean_p_hc_true_hc",
                "mean_p_dep_true_dep",
                "mean_p_dep_true_hc",
                "mean_p_hc_true_dep",
                "n_samples",
            ]:
                row[key] = None
        else:
            row.update(router)
        rows.append(row)


def append_group_rows(
    rows: list[dict],
    phase: str,
    split_kind: str,
    seed: int,
    variant: str,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    for scope_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for subject_group, group_metrics in metrics["per_group"].items():
            row = {
                "phase": phase,
                "split_kind": split_kind,
                "seed": seed,
                "variant": variant,
                "scope": scope_name,
                "subject_group": subject_group,
            }
            if group_metrics is None:
                for key in ["acc", "precision", "recall", "f1", "auroc", "auprc", "n_samples"]:
                    row[key] = None
            else:
                row.update(group_metrics)
            rows.append(row)


def run_single_variant(
    split: SplitSpec,
    variant: str,
    embeddings: dict[str, np.ndarray],
    hand_features: np.ndarray,
    group_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
):
    scaler = StandardScaler()
    hand_scaled = hand_features.copy()
    hand_scaled[split.train] = scaler.fit_transform(hand_features[split.train])
    hand_scaled[split.val] = scaler.transform(hand_features[split.val])
    hand_scaled[split.test] = scaler.transform(hand_features[split.test])

    train_loader, val_loader, test_loader = make_loaders(
        split,
        embeddings["pool"],
        hand_scaled,
        embeddings["channel_tokens"],
        embeddings["patch_tokens"],
        embeddings["label"],
        embeddings["global_trial_index"],
        group_labels,
        batch_size=args.batch_size,
        seed=split.seed,
    )
    model = Step7MoEModel(
        variant=variant,
        pool_dim=int(embeddings["pool"].shape[1]),
        hand_dim=int(hand_features.shape[1]),
        token_dim=int(embeddings["channel_tokens"].shape[2]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    t0 = perf_counter()
    best_epoch, best_val_acc = train_one(
        model,
        variant,
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
    aggregate_by_trial = split.split_kind == "subject"
    val_metrics = evaluate(model, val_loader, device, aggregate_by_trial)
    test_metrics = evaluate(model, test_loader, device, aggregate_by_trial)
    return best_epoch, best_val_acc, train_seconds, val_metrics, test_metrics


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.result_dir / "checkpoints"
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    embeddings = load_embeddings(args.embed_h5)
    hand_features, _feature_names, feature_meta = load_feature_bundle(args.feature_h5)
    assert_alignment(embeddings, feature_meta)
    group_labels = build_subject_group_labels(embeddings["subject_index"], feature_meta["subject_groups"])

    run_config = {
        "embed_h5": str(args.embed_h5),
        "feature_h5": str(args.feature_h5),
        "split_dir": str(args.split_dir),
        "split_files": selected_split_files(args.split_dir, args.split_kinds, args.seeds),
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
        "screen_seed": args.screen_seed,
        "screen_variants": args.screen_variants,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    with (args.result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    print(
        f"Loaded pool={embeddings['pool'].shape}, hand={hand_features.shape}, "
        f"channel_tokens={embeddings['channel_tokens'].shape}, patch_tokens={embeddings['patch_tokens'].shape}, "
        f"groups={np.bincount(group_labels).tolist()}, device={device}",
        flush=True,
    )

    screen_rows: list[dict] = []
    full_rows: list[dict] = []
    router_rows: list[dict] = []
    group_rows: list[dict] = []

    screen_fieldnames = [
        "variant",
        "router_loss_weight",
        "best_epoch",
        "best_val_acc",
        "train_seconds",
        "checkpoint",
    ]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            screen_fieldnames.append(f"{prefix}_{metric}")

    full_fieldnames = [
        "split_kind",
        "seed",
        "variant",
        "router_loss_weight",
        "best_epoch",
        "best_val_acc",
        "train_seconds",
        "checkpoint",
    ]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            full_fieldnames.append(f"{prefix}_{metric}")

    router_fieldnames = [
        "phase",
        "split_kind",
        "seed",
        "variant",
        "scope",
        "group_acc",
        "group_f1",
        "group_auroc",
        "mean_p_hc_true_hc",
        "mean_p_dep_true_dep",
        "mean_p_dep_true_hc",
        "mean_p_hc_true_dep",
        "n_samples",
    ]
    group_fieldnames = [
        "phase",
        "split_kind",
        "seed",
        "variant",
        "scope",
        "subject_group",
        "acc",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "n_samples",
    ]

    if args.workflow in ("screen", "all"):
        split = load_split(args.split_dir, "subject", args.screen_seed)
        print(f"\n=== screening split=subject seed={args.screen_seed} ===", flush=True)
        for variant in args.screen_variants:
            set_seed(args.screen_seed)
            checkpoint_path = checkpoint_root / f"screen_subject_seed{args.screen_seed}" / f"{variant}.pt"
            best_epoch, best_val_acc, train_seconds, val_metrics, test_metrics = run_single_variant(
                split,
                variant,
                embeddings,
                hand_features,
                group_labels,
                args,
                device,
                checkpoint_path,
            )
            row = {
                "variant": variant,
                "router_loss_weight": VARIANT_CONFIGS[variant]["router_loss_weight"],
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "train_seconds": round(train_seconds, 4),
                "checkpoint": str(checkpoint_path),
            }
            for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
                for scope in ["window", "trial"]:
                    for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                        key = f"{prefix}_{scope}_{metric}"
                        row[key] = "" if metrics[scope] is None else metrics[scope][metric]
            screen_rows.append(row)
            append_router_rows(router_rows, "screen", "subject", args.screen_seed, variant, val_metrics, test_metrics)
            append_group_rows(group_rows, "screen", "subject", args.screen_seed, variant, val_metrics, test_metrics)
            write_rows_csv(screen_rows, args.result_dir / f"screen_subject_seed{args.screen_seed}.csv", screen_fieldnames)
            write_rows_csv(router_rows, args.result_dir / "router_metrics.csv", router_fieldnames)
            write_rows_csv(group_rows, args.result_dir / "per_group_emotion_metrics.csv", group_fieldnames)
            print(
                f"{variant}: best_epoch={best_epoch}, "
                f"val_acc={val_metrics['window']['acc'] * 100:.2f}, "
                f"val_trial_auc={float(val_metrics['trial']['auroc']) * 100:.2f}, "
                f"test_auc={float(test_metrics['window']['auroc']) * 100:.2f}, "
                f"time={train_seconds:.1f}s",
                flush=True,
            )

        screen_rows.sort(key=screen_selection_key, reverse=True)
        supervised_rows = [row for row in screen_rows if row["variant"] in SUPERVISED_VARIANTS]
        supervised_rows.sort(key=screen_selection_key, reverse=True)
        selected_row = supervised_rows[0]
        selected_config = {
            "selected_supervised_variant": selected_row["variant"],
            "router_loss_weight": VARIANT_CONFIGS[selected_row["variant"]]["router_loss_weight"],
            "selection_metrics": {
                "val_trial_auroc": selected_row["val_trial_auroc"],
                "val_window_auroc": selected_row["val_window_auroc"],
                "val_trial_acc": selected_row["val_trial_acc"],
                "test_trial_auroc": selected_row["test_trial_auroc"],
                "test_window_auroc": selected_row["test_window_auroc"],
            },
        }
        with (args.result_dir / "selected_config.json").open("w", encoding="utf-8") as handle:
            json.dump(selected_config, handle, indent=2, ensure_ascii=False)
    else:
        selected_config = resolve_selected_config(args, args.result_dir)

    full_variants = args.full_variants
    if full_variants is None:
        full_variants = FULL_BASE_VARIANTS + [selected_config["selected_supervised_variant"]]

    if args.workflow in ("full", "all"):
        for split_kind in args.split_kinds:
            for seed in args.seeds:
                split = load_split(args.split_dir, split_kind, seed)
                print(f"\n=== full split={split_kind} seed={seed} ===", flush=True)
                for variant in full_variants:
                    set_seed(seed)
                    checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / f"{variant}.pt"
                    best_epoch, best_val_acc, train_seconds, val_metrics, test_metrics = run_single_variant(
                        split,
                        variant,
                        embeddings,
                        hand_features,
                        group_labels,
                        args,
                        device,
                        checkpoint_path,
                    )
                    row = {
                        "split_kind": split_kind,
                        "seed": seed,
                        "variant": variant,
                        "router_loss_weight": VARIANT_CONFIGS[variant]["router_loss_weight"],
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "train_seconds": round(train_seconds, 4),
                        "checkpoint": str(checkpoint_path),
                    }
                    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
                        for scope in ["window", "trial"]:
                            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                                key = f"{prefix}_{scope}_{metric}"
                                row[key] = "" if metrics[scope] is None else metrics[scope][metric]
                    full_rows.append(row)
                    append_router_rows(router_rows, "full", split_kind, seed, variant, val_metrics, test_metrics)
                    append_group_rows(group_rows, "full", split_kind, seed, variant, val_metrics, test_metrics)
                    write_rows_csv(full_rows, args.result_dir / "all_results.csv", full_fieldnames)
                    write_rows_csv(router_rows, args.result_dir / "router_metrics.csv", router_fieldnames)
                    write_rows_csv(group_rows, args.result_dir / "per_group_emotion_metrics.csv", group_fieldnames)
                    full_summary = summarize_full_rows(full_rows, args.result_dir)
                    write_readme(args.result_dir, run_config, screen_rows, selected_config, full_summary, router_rows)
                    print(
                        f"{variant}: best_epoch={best_epoch}, "
                        f"val_acc={val_metrics['window']['acc'] * 100:.2f}, "
                        f"test_acc={test_metrics['window']['acc'] * 100:.2f}, "
                        f"test_auc={float(test_metrics['window']['auroc']) * 100:.2f}, "
                        f"time={train_seconds:.1f}s",
                        flush=True,
                    )

    full_summary = summarize_full_rows(full_rows, args.result_dir) if full_rows else None
    write_readme(args.result_dir, run_config, screen_rows, selected_config, full_summary, router_rows)
    print(f"\nSaved Step-7 MoE results to: {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
