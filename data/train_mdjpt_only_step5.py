#!/usr/bin/env python3
"""Train mdJPT-only frozen-MLP and full-finetune baselines on fixed COMP4 splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EEG_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_EMBED_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_CKPT = REPO_ROOT / "log" / "pretrain" / "ckpt" / "epoch=19.ckpt"
DEFAULT_SPLIT_DIR = DATA_DIR / "feature_baseline_results" / "splits"
DEFAULT_RESULT_DIR = DATA_DIR / "mdjpt_only_results"
DEFAULT_SEEDS = [42, 3407, 2025]
DEFAULT_MODES = ["frozen_mlp", "full_finetune"]
DEFAULT_SPLIT_KINDS = ["subject", "segment"]


@dataclass
class SplitSpec:
    split_kind: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    path: Path


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, trial_ids: np.ndarray, indices: np.ndarray) -> None:
        self.features = torch.from_numpy(features[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.trial_ids = torch.from_numpy(trial_ids[indices].astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.trial_ids[idx]


class EEGDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray, trial_ids: np.ndarray, indices: np.ndarray) -> None:
        self.eeg = torch.from_numpy(eeg[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.trial_ids = torch.from_numpy(trial_ids[indices].astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return self.eeg[idx].unsqueeze(0), self.labels[idx], self.trial_ids[idx]


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


class FullFineTuneModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone_out = self.backbone.forward(x, 0)
        features = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out
        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg-h5", type=Path, default=DEFAULT_EEG_H5)
    parser.add_argument("--embed-h5", type=Path, default=DEFAULT_EMBED_H5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=DEFAULT_MODES)
    parser.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)
    parser.add_argument("--frozen-batch-size", type=int, default=256)
    parser.add_argument("--ft-batch-size", type=int, default=128)
    parser.add_argument("--frozen-max-epochs", type=int, default=80)
    parser.add_argument("--ft-max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--frozen-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
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


def load_embedding_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "pool": handle["pool_1024"][:].astype(np.float32),
            "label": handle["label"][:].astype(np.int64),
            "global_trial_index": handle["global_trial_index"][:].astype(np.int64),
        }


def load_eeg_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "eeg": handle["eeg"][:].astype(np.float32),
            "label": handle["label"][:].astype(np.int64),
            "global_trial_index": handle["global_trial_index"][:].astype(np.int64),
        }


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
        for x, y, trial_id in loader:
            logits = model(x.to(device))
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


def build_backbone_cfg():
    config_dir = str(REPO_ROOT / "cfgs_multi")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config_multi",
            overrides=[
                "data@data_val=COMP4",
                "data_val.timeLen2=5",
                "data_val.timeStep2=5",
                "log.run_name=pretrain",
                "val.extractor.ckpt_epoch=20",
            ],
        )
    runtime_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    runtime_cfg.data_0 = OmegaConf.create(OmegaConf.to_container(cfg.data_val, resolve=True))
    runtime_cfg.data_1 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_2 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_3 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_4 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_cfg_list = [runtime_cfg.data_0]
    return runtime_cfg


def load_backbone(checkpoint: Path, device: torch.device) -> nn.Module:
    os.chdir(REPO_ROOT)
    from src.model.MultiModel_PL import MultiModel_PL

    cfg = build_backbone_cfg()
    backbone = MultiModel_PL.load_from_checkpoint(str(checkpoint), cfg=cfg, strict=False, map_location="cpu")
    backbone.save_fea = False
    if hasattr(backbone, "cnn_encoder"):
        backbone.cnn_encoder.set_saveFea(False)
    backbone.to(device)
    return backbone


def infer_backbone_dim(backbone: nn.Module, sample_eeg: np.ndarray, device: torch.device) -> int:
    backbone.eval()
    sample = torch.from_numpy(sample_eeg[:2].astype(np.float32)).unsqueeze(1).to(device)
    with torch.no_grad():
        output = backbone.forward(sample, 0)
        features = output[0] if isinstance(output, tuple) else output
        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)
    return int(features.shape[-1])


def make_feature_loaders(
    split: SplitSpec,
    features: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = FeatureDataset(features, labels, trial_ids, split.train)
    valset = FeatureDataset(features, labels, trial_ids, split.val)
    testset = FeatureDataset(features, labels, trial_ids, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def make_eeg_loaders(
    split: SplitSpec,
    eeg: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = EEGDataset(eeg, labels, trial_ids, split.train)
    valset = EEGDataset(eeg, labels, trial_ids, split.val)
    testset = EEGDataset(eeg, labels, trial_ids, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
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


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["split_kind", "seed", "mode", "best_epoch", "best_val_acc", "train_seconds", "checkpoint"]
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
        for mode in sorted({row["mode"] for row in rows}):
            selected = [row for row in rows if row["split_kind"] == split_kind and row["mode"] == mode]
            if not selected:
                continue
            summary[split_kind][mode] = {"window": {}, "trial": {}}
            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                values = np.asarray([row[f"test_window_{metric}"] for row in selected], dtype=np.float64)
                summary[split_kind][mode]["window"][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
                trial_values = [row[f"test_trial_{metric}"] for row in selected if row[f"test_trial_{metric}"] != ""]
                if trial_values:
                    trial_values = np.asarray(trial_values, dtype=np.float64)
                    summary[split_kind][mode]["trial"][metric] = {
                        "mean": float(trial_values.mean()),
                        "std": float(trial_values.std(ddof=0)),
                    }
                else:
                    summary[split_kind][mode]["trial"][metric] = None
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def write_readme(summary: dict, result_dir: Path, run_config: dict) -> None:
    lines = [
        "# mdJPT-Only Step-5 Results",
        "",
        "Baselines using only mdJPT representations on fixed COMP4 5s/60-channel splits.",
        "",
        "## Setup",
        "",
        f"- EEG H5: `{run_config['eeg_h5']}`",
        f"- Embedding H5: `{run_config['embed_h5']}`",
        f"- Checkpoint: `{run_config['checkpoint']}`",
        f"- Fixed split files: `{run_config['split_dir']}/comp4_{{subject,segment}}_seed*.json`",
        f"- Seeds: `{', '.join(str(seed) for seed in run_config['seeds'])}`",
        f"- Modes: `{', '.join(run_config['modes'])}`",
        f"- Frozen MLP: batch `{run_config['frozen_batch_size']}`, lr `{run_config['frozen_lr']}`, max epochs `{run_config['frozen_max_epochs']}`",
        f"- Full finetune: batch `{run_config['ft_batch_size']}`, head lr `{run_config['head_lr']}`, backbone lr `{run_config['backbone_lr']}`, max epochs `{run_config['ft_max_epochs']}`",
        "",
        "## Summary",
        "",
        "| Split | Mode | Window Acc | Window F1 | Window AUROC | Trial Acc | Trial AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split_kind, modes in summary.items():
        for mode, metrics in modes.items():
            w = metrics["window"]
            t = metrics["trial"]
            trial_acc = "-" if t["acc"] is None else f"{t['acc']['mean']*100:.2f} ± {t['acc']['std']*100:.2f}"
            trial_auc = "-" if t["auroc"] is None else f"{t['auroc']['mean']*100:.2f} ± {t['auroc']['std']*100:.2f}"
            lines.append(
                f"| {split_kind} | {mode} | "
                f"{w['acc']['mean']*100:.2f} ± {w['acc']['std']*100:.2f} | "
                f"{w['f1']['mean']*100:.2f} ± {w['f1']['std']*100:.2f} | "
                f"{w['auroc']['mean']*100:.2f} ± {w['auroc']['std']*100:.2f} | "
                f"{trial_acc} | {trial_auc} |"
            )
    lines.extend(
        [
            "",
            "Subject split reports trial-level metrics by averaging segment probabilities within each trial.",
            "Segment split leaves trial-level metrics blank to avoid trial leakage interpretation.",
        ]
    )
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_result_row(
    rows: list[dict],
    result_dir: Path,
    run_config: dict,
    split_kind: str,
    seed: int,
    mode: str,
    best_epoch: int,
    best_val_acc: float,
    train_seconds: float,
    checkpoint_path: Path,
    val_metrics: dict,
    test_metrics: dict,
) -> None:
    row = {
        "split_kind": split_kind,
        "seed": seed,
        "mode": mode,
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
    rows.append(row)
    write_csv(rows, result_dir / "all_results.csv")
    summary = summarize(rows, result_dir)
    write_readme(summary, result_dir, run_config)


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.result_dir / "checkpoints"
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    run_config = {
        "eeg_h5": str(args.eeg_h5),
        "embed_h5": str(args.embed_h5),
        "checkpoint": str(args.checkpoint),
        "split_dir": str(args.split_dir),
        "split_files": selected_split_files(args.split_dir, args.split_kinds, args.seeds),
        "seeds": args.seeds,
        "modes": args.modes,
        "split_kinds": args.split_kinds,
        "frozen_batch_size": args.frozen_batch_size,
        "ft_batch_size": args.ft_batch_size,
        "frozen_max_epochs": args.frozen_max_epochs,
        "ft_max_epochs": args.ft_max_epochs,
        "patience": args.patience,
        "frozen_lr": args.frozen_lr,
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    with (args.result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    embedding_data = None
    eeg_data = None
    if "frozen_mlp" in args.modes:
        embedding_data = load_embedding_arrays(args.embed_h5)
        print(f"Loaded frozen embeddings: pool={embedding_data['pool'].shape}, device={device}")
    if "full_finetune" in args.modes:
        eeg_data = load_eeg_arrays(args.eeg_h5)
        print(f"Loaded EEG: eeg={eeg_data['eeg'].shape}, device={device}")

    rows = []
    for split_kind in args.split_kinds:
        for seed in args.seeds:
            split = load_split(args.split_dir, split_kind, seed)
            aggregate_by_trial = split_kind == "subject"
            print(f"\n=== split={split_kind} seed={seed} ===", flush=True)

            if "frozen_mlp" in args.modes:
                assert embedding_data is not None
                set_seed(seed)
                train_loader, val_loader, test_loader = make_feature_loaders(
                    split,
                    embedding_data["pool"],
                    embedding_data["label"],
                    embedding_data["global_trial_index"],
                    args.frozen_batch_size,
                    seed,
                )
                model = MLPHead(
                    in_dim=int(embedding_data["pool"].shape[1]),
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.frozen_lr, weight_decay=args.weight_decay)
                checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / "frozen_mlp.pt"
                t0 = perf_counter()
                best_epoch, best_val_acc = train_one(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.frozen_max_epochs,
                    args.patience,
                    optimizer,
                    checkpoint_path,
                )
                train_seconds = perf_counter() - t0
                val_metrics = evaluate(model, val_loader, device, aggregate_by_trial)
                test_metrics = evaluate(model, test_loader, device, aggregate_by_trial)
                append_result_row(
                    rows,
                    args.result_dir,
                    run_config,
                    split_kind,
                    seed,
                    "frozen_mlp",
                    best_epoch,
                    best_val_acc,
                    train_seconds,
                    checkpoint_path,
                    val_metrics,
                    test_metrics,
                )
                print(
                    f"frozen_mlp: best_epoch={best_epoch}, "
                    f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                    f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                    f"test_auc={test_metrics['window']['auroc']*100:.2f}, "
                    f"time={train_seconds:.1f}s",
                    flush=True,
                )

            if "full_finetune" in args.modes:
                assert eeg_data is not None
                set_seed(seed)
                train_loader, val_loader, test_loader = make_eeg_loaders(
                    split,
                    eeg_data["eeg"],
                    eeg_data["label"],
                    eeg_data["global_trial_index"],
                    args.ft_batch_size,
                    seed,
                )
                backbone = load_backbone(args.checkpoint, device)
                feature_dim = infer_backbone_dim(backbone, eeg_data["eeg"], device)
                classifier = MLPHead(feature_dim, args.hidden_dim, dropout=args.dropout)
                model = FullFineTuneModel(backbone, classifier)
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.classifier.parameters(), "lr": args.head_lr},
                        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                    ],
                    weight_decay=args.weight_decay,
                )
                checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / "full_finetune.pt"
                t0 = perf_counter()
                best_epoch, best_val_acc = train_one(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.ft_max_epochs,
                    args.patience,
                    optimizer,
                    checkpoint_path,
                )
                train_seconds = perf_counter() - t0
                val_metrics = evaluate(model, val_loader, device, aggregate_by_trial)
                test_metrics = evaluate(model, test_loader, device, aggregate_by_trial)
                append_result_row(
                    rows,
                    args.result_dir,
                    run_config,
                    split_kind,
                    seed,
                    "full_finetune",
                    best_epoch,
                    best_val_acc,
                    train_seconds,
                    checkpoint_path,
                    val_metrics,
                    test_metrics,
                )
                print(
                    f"full_finetune: best_epoch={best_epoch}, "
                    f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                    f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                    f"test_auc={test_metrics['window']['auroc']*100:.2f}, "
                    f"time={train_seconds:.1f}s",
                    flush=True,
                )
                del model, backbone, classifier, optimizer
                torch.cuda.empty_cache()

    summary = summarize(rows, args.result_dir)
    write_readme(summary, args.result_dir, run_config)
    print(f"\nSaved mdJPT-only results to: {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
