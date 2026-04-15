#!/usr/bin/env python3
"""Train step-5 mdJPT + handcrafted feature fusion models."""

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
DEFAULT_EMBED_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_SPLIT_DIR = DATA_DIR / "feature_baseline_results" / "splits"
DEFAULT_RESULT_DIR = DATA_DIR / "step5_fusion_results"
DEFAULT_SEEDS = [42, 3407, 2025]
DEFAULT_VARIANTS = ["mdjpt_only", "concat", "gating", "cross_attn"]
DEFAULT_SPLIT_KINDS = ["subject", "segment"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass
class SplitSpec:
    split_kind: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    path: Path


class FusionDataset(Dataset):
    def __init__(
        self,
        pool: np.ndarray,
        hand: np.ndarray,
        channel_tokens: np.ndarray,
        patch_tokens: np.ndarray,
        labels: np.ndarray,
        trial_ids: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.pool = torch.from_numpy(pool[indices].astype(np.float32))
        self.hand = torch.from_numpy(hand[indices].astype(np.float32))
        self.channel_tokens = torch.from_numpy(channel_tokens[indices].astype(np.float32))
        self.patch_tokens = torch.from_numpy(patch_tokens[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.trial_ids = torch.from_numpy(trial_ids[indices].astype(np.int64))

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
        )


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModel(nn.Module):
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
        self.variant = variant
        if variant == "mdjpt_only":
            self.classifier = MLPHead(pool_dim, hidden_dim, dropout=dropout)
        elif variant == "concat":
            self.classifier = MLPHead(pool_dim + hand_dim, hidden_dim, dropout=dropout)
        elif variant == "gating":
            self.hand_proj = nn.Linear(hand_dim, pool_dim)
            self.gate = nn.Sequential(
                nn.Linear(pool_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, pool_dim),
            )
            self.norm = nn.LayerNorm(pool_dim)
            self.classifier = MLPHead(pool_dim, hidden_dim, dropout=dropout)
        elif variant == "cross_attn":
            self.hand_query = nn.Linear(hand_dim, token_dim)
            self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(token_dim)
            self.classifier = MLPHead(token_dim, hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion variant: {variant}")

    def forward(
        self,
        pool: torch.Tensor,
        hand: torch.Tensor,
        channel_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "mdjpt_only":
            return self.classifier(pool)
        if self.variant == "concat":
            return self.classifier(torch.cat([pool, hand], dim=-1))
        if self.variant == "gating":
            hand_proj = self.hand_proj(hand)
            gate = torch.sigmoid(self.gate(torch.cat([pool, hand_proj], dim=-1)))
            fused = self.norm(gate * pool + (1.0 - gate) * hand_proj)
            return self.classifier(fused)
        if self.variant == "cross_attn":
            query = self.hand_query(hand).unsqueeze(1)
            tokens = torch.cat([channel_tokens, patch_tokens], dim=1)
            attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
            fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))
            return self.classifier(fused)
        raise RuntimeError(f"Unhandled variant: {self.variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed-h5", type=Path, default=DEFAULT_EMBED_H5)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=DEFAULT_VARIANTS)
    parser.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "pool": handle["pool_1024"][:].astype(np.float32),
            "channel_tokens": handle["channel_tokens_32"][:].astype(np.float32),
            "patch_tokens": handle["patch_tokens_32"][:].astype(np.float32),
            "label": handle["label"][:].astype(np.int64),
            "global_trial_index": handle["global_trial_index"][:].astype(np.int64),
        }


def decode_strings(values: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def require_feature_metadata(handle: h5py.File) -> None:
    required = ["label", "subject_index", "trial_index", "segment_index"]
    missing = [name for name in required if name not in handle]
    if missing:
        raise KeyError(f"Feature H5 is missing metadata datasets: {missing}")


def load_feature_matrix(path: Path) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    feature_blocks = []
    feature_names = []

    with h5py.File(path, "r") as handle:
        require_feature_metadata(handle)
        channel_names = decode_strings(handle["meta/channel_names"][:])

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
            pair_names = decode_strings(handle[f"asymmetry/{band}_names"][:])
            feature_blocks.append(data)
            feature_names.extend([f"asym_{band}_{pair}" for pair in pair_names])

        feature_blocks.append(handle["faa/values"][:])
        faa_names = decode_strings(handle["faa/names"][:])
        feature_names.extend([f"FAA_alpha_{name}" for name in faa_names])

        metadata = {
            "label": handle["label"][:].astype(np.int64),
            "subject_index": handle["subject_index"][:].astype(np.int64),
            "trial_index": handle["trial_index"][:].astype(np.int64),
            "segment_index": handle["segment_index"][:].astype(np.int64),
        }

    features = np.concatenate([np.asarray(block, dtype=np.float32) for block in feature_blocks], axis=1)
    features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return features, feature_names, metadata


def load_split(split_dir: Path, split_kind: str, seed: int) -> SplitSpec:
    path = split_dir / f"comp4_{split_kind}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
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


def selected_split_files(split_dir: Path, split_kinds: list[str], seeds: list[int]) -> list[str]:
    return [str(split_dir / f"comp4_{split_kind}_seed{seed}.json") for split_kind in split_kinds for seed in seeds]


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_pred = y_prob.argmax(axis=1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "auprc": float(average_precision_score(y_true, y_prob[:, 1])),
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


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probs, trial_ids = [], [], []
    with torch.no_grad():
        for pool, hand, channel_tokens, patch_tokens, y, trial_id in loader:
            logits = model(
                pool.to(device),
                hand.to(device),
                channel_tokens.to(device),
                patch_tokens.to(device),
            )
            labels.append(y.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            trial_ids.append(trial_id.numpy())
    return np.concatenate(labels), np.concatenate(probs), np.concatenate(trial_ids)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, aggregate_by_trial: bool):
    y_true, y_prob, trial_ids = collect_predictions(model, loader, device)
    result = {"window": metrics_from_probs(y_true, y_prob)}
    if aggregate_by_trial:
        trial_true, trial_prob = aggregate_trials(y_true, y_prob, trial_ids)
        result["trial"] = metrics_from_probs(trial_true, trial_prob)
    else:
        result["trial"] = None
    return result


def train_one(
    model: nn.Module,
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        for pool, hand, channel_tokens, patch_tokens, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                pool.to(device),
                hand.to(device),
                channel_tokens.to(device),
                patch_tokens.to(device),
            )
            loss = criterion(logits, y.to(device))
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


def make_loaders(
    split: SplitSpec,
    pool: np.ndarray,
    hand_scaled: np.ndarray,
    channel_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = FusionDataset(pool, hand_scaled, channel_tokens, patch_tokens, labels, trial_ids, split.train)
    valset = FusionDataset(pool, hand_scaled, channel_tokens, patch_tokens, labels, trial_ids, split.val)
    testset = FusionDataset(pool, hand_scaled, channel_tokens, patch_tokens, labels, trial_ids, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "split_kind",
        "seed",
        "variant",
        "best_epoch",
        "best_val_acc",
        "train_seconds",
        "checkpoint",
    ]
    for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
            fieldnames.append(f"{prefix}_{metric}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], result_dir: Path) -> dict:
    summary = {}
    for split_kind in sorted({row["split_kind"] for row in rows}):
        summary[split_kind] = {}
        for variant in sorted({row["variant"] for row in rows}):
            selected = [row for row in rows if row["split_kind"] == split_kind and row["variant"] == variant]
            if not selected:
                continue
            summary[split_kind][variant] = {"window": {}, "trial": {}}
            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                values = np.asarray([row[f"test_window_{metric}"] for row in selected], dtype=np.float64)
                summary[split_kind][variant]["window"][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
                trial_values = [row[f"test_trial_{metric}"] for row in selected if row[f"test_trial_{metric}"] != ""]
                if trial_values:
                    trial_values = np.asarray(trial_values, dtype=np.float64)
                    summary[split_kind][variant]["trial"][metric] = {
                        "mean": float(trial_values.mean()),
                        "std": float(trial_values.std(ddof=0)),
                    }
                else:
                    summary[split_kind][variant]["trial"][metric] = None
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def write_readme(summary: dict, result_dir: Path, run_config: dict | None = None) -> None:
    lines = [
        "# Step 5 Fusion Results",
        "",
        "Frozen mdJPT embeddings fused with handcrafted features.",
        "",
    ]
    if run_config is not None:
        lines.extend(
            [
                "## Setup",
                "",
                f"- Embeddings: `{run_config['embed_h5']}`",
                f"- Handcrafted features: `{run_config['feature_h5']}`",
                f"- Seeds: `{', '.join(str(seed) for seed in run_config['seeds'])}`",
                f"- Variants: `{', '.join(run_config['variants'])}`",
                f"- Optimizer: `AdamW`, lr `{run_config['lr']}`, weight decay `{run_config['weight_decay']}`",
                f"- Batch size: `{run_config['batch_size']}`, max epochs `{run_config['max_epochs']}`, patience `{run_config['patience']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            "| Split | Variant | Window Acc | Window F1 | Window AUROC | Trial Acc | Trial AUROC |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split_kind, variants in summary.items():
        for variant, metrics in variants.items():
            w = metrics["window"]
            t = metrics["trial"]
            trial_acc = "-" if t["acc"] is None else f"{t['acc']['mean']*100:.2f} ± {t['acc']['std']*100:.2f}"
            trial_auc = "-" if t["auroc"] is None else f"{t['auroc']['mean']*100:.2f} ± {t['auroc']['std']*100:.2f}"
            lines.append(
                f"| {split_kind} | {variant} | "
                f"{w['acc']['mean']*100:.2f} ± {w['acc']['std']*100:.2f} | "
                f"{w['f1']['mean']*100:.2f} ± {w['f1']['std']*100:.2f} | "
                f"{w['auroc']['mean']*100:.2f} ± {w['auroc']['std']*100:.2f} | "
                f"{trial_acc} | {trial_auc} |"
            )
    lines.extend(
        [
            "",
            "Subject split also reports trial-level metrics by averaging segment probabilities within each trial.",
            "Segment split intentionally leaves trial-level metrics blank because segments from the same trial can appear in multiple splits.",
        ]
    )
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.result_dir / "checkpoints"
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    run_config = {
        "embed_h5": str(args.embed_h5),
        "feature_h5": str(args.feature_h5),
        "split_dir": str(args.split_dir),
        "split_files": selected_split_files(args.split_dir, args.split_kinds, args.seeds),
        "seeds": args.seeds,
        "variants": args.variants,
        "split_kinds": args.split_kinds,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }

    embeddings = load_embeddings(args.embed_h5)
    hand_features, feature_names, feature_meta = load_feature_matrix(args.feature_h5)
    labels = embeddings["label"]
    if not np.array_equal(labels, feature_meta["label"]):
        raise ValueError("Label mismatch between embedding H5 and feature H5")

    with (args.result_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2, ensure_ascii=False)
    with (args.result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    print(
        f"Loaded pool={embeddings['pool'].shape}, hand={hand_features.shape}, "
        f"channel_tokens={embeddings['channel_tokens'].shape}, patch_tokens={embeddings['patch_tokens'].shape}, "
        f"device={device}"
    )

    rows = []
    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(args.split_dir, split_kind, seed)
            scaler = StandardScaler()
            hand_scaled = hand_features.copy()
            hand_scaled[split.train] = scaler.fit_transform(hand_features[split.train])
            hand_scaled[split.val] = scaler.transform(hand_features[split.val])
            hand_scaled[split.test] = scaler.transform(hand_features[split.test])

            print(f"\n=== split={split_kind} seed={seed} ===")
            for variant in args.variants:
                set_seed(seed)
                checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / f"{variant}.pt"
                train_loader, val_loader, test_loader = make_loaders(
                    split,
                    embeddings["pool"],
                    hand_scaled,
                    embeddings["channel_tokens"],
                    embeddings["patch_tokens"],
                    labels,
                    embeddings["global_trial_index"],
                    args.batch_size,
                    seed,
                )
                model = FusionModel(
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
                aggregate_by_trial = split_kind == "subject"
                val_metrics = evaluate(model, val_loader, device, aggregate_by_trial)
                test_metrics = evaluate(model, test_loader, device, aggregate_by_trial)

                row = {
                    "split_kind": split_kind,
                    "seed": seed,
                    "variant": variant,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "train_seconds": round(train_seconds, 4),
                    "checkpoint": str(checkpoint_path),
                }
                for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
                    for scope in ["window", "trial"]:
                        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                            key = f"{prefix}_{scope}_{metric}"
                            if metrics[scope] is None:
                                row[key] = ""
                            else:
                                row[key] = metrics[scope][metric]
                rows.append(row)
                write_csv(rows, args.result_dir / "all_results.csv")
                summary = summarize(rows, args.result_dir)
                write_readme(summary, args.result_dir, run_config)
                print(
                    f"{variant}: best_epoch={best_epoch}, "
                    f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                    f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                    f"test_auc={test_metrics['window']['auroc']*100:.2f}, "
                    f"time={train_seconds:.1f}s",
                    flush=True,
                )

    summary = summarize(rows, args.result_dir)
    write_readme(summary, args.result_dir, run_config)
    print(f"\nSaved step-5 fusion results to: {args.result_dir}")


if __name__ == "__main__":
    main()
