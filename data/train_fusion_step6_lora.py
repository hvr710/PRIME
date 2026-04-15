#!/usr/bin/env python3
"""Train Step-6 CrossAttn + LoRA ablations on fixed COMP4 splits."""

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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from train_fusion_step5 import load_feature_matrix


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.lora import apply_lora_to_mlla, freeze_backbone_except_lora, iter_lora_named_parameters


DEFAULT_EEG_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_FEATURE_H5 = DATA_DIR / "features_comp4_len5_step5_mapped60.h5"
DEFAULT_CKPT = REPO_ROOT / "log" / "pretrain" / "ckpt" / "epoch=19.ckpt"
DEFAULT_SPLIT_DIR = DATA_DIR / "feature_baseline_results" / "splits"
DEFAULT_RESULT_DIR = DATA_DIR / "step6_lora_results"
DEFAULT_SEEDS = [42, 3407, 2025]
DEFAULT_SPLIT_KINDS = ["subject", "segment"]


@dataclass
class SplitSpec:
    split_kind: str
    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    path: Path


class EEGHandDataset(Dataset):
    def __init__(
        self,
        eeg: np.ndarray,
        hand: np.ndarray,
        labels: np.ndarray,
        trial_ids: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.eeg = torch.from_numpy(eeg[indices].astype(np.float32))
        self.hand = torch.from_numpy(hand[indices].astype(np.float32))
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.trial_ids = torch.from_numpy(trial_ids[indices].astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return (
            self.eeg[idx].unsqueeze(0),
            self.hand[idx],
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


class Step6CrossAttnModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hand_dim: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float,
        lora_enabled: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lora_enabled = lora_enabled
        self.hand_query = nn.Linear(hand_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.classifier = MLPHead(token_dim, hidden_dim, dropout=dropout)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, eeg: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        backbone_context = torch.enable_grad if self.lora_enabled else torch.no_grad
        with backbone_context():
            pool, _, mllaout = self.backbone.forward(eeg, 0, returnMLLAout=True)
            _ = pool.reshape(pool.shape[0], -1)
            channel_tokens = mllaout.mean(dim=2)
            patch_tokens = mllaout.mean(dim=1)

        query = self.hand_query(hand).unsqueeze(1)
        tokens = torch.cat([channel_tokens, patch_tokens], dim=1)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        fused = self.norm(attn_out.squeeze(1) + query.squeeze(1))
        return self.classifier(fused)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=["screen", "full", "all"], default="all")
    parser.add_argument("--eeg-h5", type=Path, default=DEFAULT_EEG_H5)
    parser.add_argument("--feature-h5", type=Path, default=DEFAULT_FEATURE_H5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--split-kinds", nargs="+", default=DEFAULT_SPLIT_KINDS, choices=DEFAULT_SPLIT_KINDS)
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--screen-last-ks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--screen-ranks", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--selected-last-k", type=int, default=None)
    parser.add_argument("--selected-rank", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=5e-4)
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
        for eeg, hand, y, trial_id in loader:
            logits = model(eeg.to(device), hand.to(device))
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
    backbone.eval()
    return backbone


def make_loaders(
    split: SplitSpec,
    eeg: np.ndarray,
    hand_scaled: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    batch_size: int,
    seed: int,
):
    trainset = EEGHandDataset(eeg, hand_scaled, labels, trial_ids, split.train)
    valset = EEGHandDataset(eeg, hand_scaled, labels, trial_ids, split.val)
    testset = EEGHandDataset(eeg, hand_scaled, labels, trial_ids, split.test)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(valset, batch_size=batch_size, shuffle=False),
        DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


def summarize_parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    lora = sum(param.numel() for _, param in iter_lora_named_parameters(model.backbone) if param.requires_grad)
    violations = [
        name
        for name, param in model.backbone.named_parameters()
        if param.requires_grad and ".lora_A" not in name and ".lora_B" not in name
    ]
    if violations:
        preview = ", ".join(violations[:8])
        raise RuntimeError(f"Non-LoRA backbone params are trainable: {preview}")
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "lora_params": int(lora),
    }


def build_model(
    checkpoint: Path,
    device: torch.device,
    hand_dim: int,
    hidden_dim: int,
    dropout: float,
    last_k: int | None,
    rank: int | None,
    target_scope: str | None,
):
    backbone = load_backbone(checkpoint, device)
    lora_enabled = last_k is not None and rank is not None and target_scope is not None
    replaced_paths = []
    if lora_enabled:
        replaced_paths = apply_lora_to_mlla(
            backbone,
            last_k=last_k,
            target_scope=target_scope,
            rank=rank,
            alpha=rank,
            dropout=0.0,
        )
    freeze_backbone_except_lora(backbone)
    token_dim = int(backbone.cfg.model.MLLA.out_dim)
    model = Step6CrossAttnModel(
        backbone=backbone,
        hand_dim=hand_dim,
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lora_enabled=lora_enabled,
    )
    counts = summarize_parameter_counts(model)
    return model, counts, replaced_paths


def build_optimizer(model: nn.Module, head_lr: float, lora_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    head_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.") and (".lora_A" in name or ".lora_B" in name):
            lora_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


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
        for eeg, hand, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg.to(device), hand.to(device))
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


def variant_name(last_k: int | None, rank: int | None, target_scope: str | None) -> str:
    if last_k is None:
        return "cross_attn_no_lora"
    suffix = "attnproj" if target_scope == "attention_plus_proj" else "attn"
    return f"cross_attn_lora_last{last_k}_r{rank}_{suffix}"


def short_screen_name(last_k: int, rank: int) -> str:
    return f"lora_last{last_k}_r{rank}_attn"


def write_rows_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
                    values = [row[f"{scope}_{metric}"] for row in selected if row[f"{scope}_{metric}"] != ""]
                    if values:
                        arr = np.asarray(values, dtype=np.float64)
                        summary[split_kind][variant][scope][metric] = {
                            "mean": float(arr.mean()),
                            "std": float(arr.std(ddof=0)),
                        }
                    else:
                        summary[split_kind][variant][scope][metric] = None
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def load_step5_crossattn_summary() -> dict | None:
    path = DATA_DIR / "step5_fusion_results" / "summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_readme(
    result_dir: Path,
    run_config: dict,
    screen_rows: list[dict],
    selected_config: dict | None,
    full_summary: dict | None,
) -> None:
    def fmt_summary(entry: dict | None) -> str:
        if entry is None:
            return "-"
        return f"{entry['mean'] * 100:.2f} ± {entry['std'] * 100:.2f}"

    lines = [
        "# Step 6 LoRA Results",
        "",
        "CrossAttn fusion with live mdJPT forward and LoRA adapters on MLLA.",
        "",
        "## Setup",
        "",
        f"- EEG H5: `{run_config['eeg_h5']}`",
        f"- Feature H5: `{run_config['feature_h5']}`",
        f"- Checkpoint: `{run_config['checkpoint']}`",
        f"- Split files: `{run_config['split_dir']}/comp4_{{subject,segment}}_seed*.json`",
        f"- Seeds: `{', '.join(str(seed) for seed in run_config['seeds'])}`",
        f"- Screening grid: `last_k in {run_config['screen_last_ks']}`, `rank in {run_config['screen_ranks']}`",
        f"- Batch size: `{run_config['batch_size']}`, max epochs `{run_config['max_epochs']}`, patience `{run_config['patience']}`",
        f"- Head lr: `{run_config['head_lr']}`, LoRA lr: `{run_config['lora_lr']}`, weight decay `{run_config['weight_decay']}`",
        "",
    ]

    if selected_config is not None:
        lines.extend(
            [
                "## Selected LoRA Config",
                "",
                f"- `last_k = {selected_config['selected_last_k']}`",
                f"- `rank = {selected_config['selected_rank']}`",
                f"- Selected by `val_window_auroc`, then `val_trial_auroc`, then `val_window_acc`, then smaller `rank/last_k`.",
                "",
            ]
        )

    if screen_rows:
        lines.extend(
            [
                "## Screening Results",
                "",
                "| Variant | Val Acc | Val AUROC | Val Trial AUROC | Test Acc | Test AUROC |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in screen_rows:
            val_trial_auroc = "-" if row["val_trial_auroc"] == "" else f"{float(row['val_trial_auroc']) * 100:.2f}"
            lines.append(
                f"| {row['variant']} | "
                f"{float(row['val_window_acc']) * 100:.2f} | "
                f"{float(row['val_window_auroc']) * 100:.2f} | "
                f"{val_trial_auroc} | "
                f"{float(row['test_window_acc']) * 100:.2f} | "
                f"{float(row['test_window_auroc']) * 100:.2f} |"
            )
        lines.append("")

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
                val_window = metrics["val_window"]
                test_window = metrics["test_window"]
                test_trial = metrics["test_trial"]
                val_acc = val_window["acc"]
                val_auc = val_window["auroc"]
                test_acc = test_window["acc"]
                test_auc = test_window["auroc"]
                trial_acc = test_trial["acc"]
                trial_auc = test_trial["auroc"]
                lines.append(
                    f"| {split_kind} | {variant} | "
                    f"{fmt_summary(val_acc)} | "
                    f"{fmt_summary(val_auc)} | "
                    f"{fmt_summary(test_acc)} | "
                    f"{fmt_summary(test_auc)} | "
                    f"{fmt_summary(trial_acc)} | "
                    f"{fmt_summary(trial_auc)} |"
                )
        lines.append("")

        step5_summary = load_step5_crossattn_summary()
        if step5_summary is not None and "cross_attn" in step5_summary.get("subject", {}):
            no_lora_variant = "cross_attn_no_lora"
            attn_variant = None
            attnproj_variant = None
            for variant in next(iter(full_summary.values())).keys():
                if variant.endswith("_attn"):
                    attn_variant = variant
                elif variant.endswith("_attnproj"):
                    attnproj_variant = variant

            lines.extend(
                [
                    "## Delta Vs Step5 CrossAttn",
                    "",
                    "| Split | Variant | Test Acc Delta | Test AUROC Delta |",
                    "|---|---|---:|---:|",
                ]
            )
            for split_kind in ["subject", "segment"]:
                step5_metrics = step5_summary.get(split_kind, {}).get("cross_attn")
                if not step5_metrics:
                    continue
                step5_acc = step5_metrics["window"]["acc"]["mean"]
                step5_auc = step5_metrics["window"]["auroc"]["mean"]
                for variant in [no_lora_variant, attn_variant, attnproj_variant]:
                    if variant is None or variant not in full_summary.get(split_kind, {}):
                        continue
                    metrics = full_summary[split_kind][variant]["test_window"]
                    test_acc = metrics["acc"]["mean"]
                    test_auc = metrics["auroc"]["mean"]
                    lines.append(
                        f"| {split_kind} | {variant} | {(test_acc - step5_acc) * 100:.2f} | {(test_auc - step5_auc) * 100:.2f} |"
                    )
            if attn_variant and attnproj_variant:
                lines.extend(
                    [
                        "",
                        "## Attn Vs AttnProj",
                        "",
                        "| Split | Metric | Attn | AttnProj | Delta |",
                        "|---|---|---:|---:|---:|",
                    ]
                )
                for split_kind in ["subject", "segment"]:
                    split_summary = full_summary.get(split_kind, {})
                    if attn_variant not in split_summary or attnproj_variant not in split_summary:
                        continue
                    attn_metrics = split_summary[attn_variant]["test_window"]
                    attnproj_metrics = split_summary[attnproj_variant]["test_window"]
                    for metric in ["acc", "auroc"]:
                        attn_mean = attn_metrics[metric]["mean"]
                        attnproj_mean = attnproj_metrics[metric]["mean"]
                        lines.append(
                            f"| {split_kind} | test_{metric} | {attn_mean * 100:.2f} | {attnproj_mean * 100:.2f} | {(attnproj_mean - attn_mean) * 100:.2f} |"
                        )
                lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `last2` covers all current MLLA blocks because `cfg.model.MLLA.depth = 2`.",
            "- `cross_attn_no_lora` uses live frozen backbone, so tiny differences vs cached Step5 embeddings are expected but should remain small.",
        ]
    )
    (result_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def screen_selection_key(row: dict) -> tuple:
    val_trial_auroc = -np.inf if row["val_trial_auroc"] == "" else float(row["val_trial_auroc"])
    return (
        float(row["val_window_auroc"]),
        val_trial_auroc,
        float(row["val_window_acc"]),
        -int(row["rank"]),
        -int(row["last_k"]),
    )


def resolve_selected_config(args: argparse.Namespace, result_dir: Path) -> dict:
    if args.selected_last_k is not None and args.selected_rank is not None:
        return {
            "selected_last_k": int(args.selected_last_k),
            "selected_rank": int(args.selected_rank),
        }
    selected_path = result_dir / "selected_config.json"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Missing selected_config.json at {selected_path}. Run workflow=screen or pass --selected-last-k/--selected-rank."
        )
    with selected_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_single_variant(
    eeg: np.ndarray,
    hand_features: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    split: SplitSpec,
    args: argparse.Namespace,
    device: torch.device,
    last_k: int | None,
    rank: int | None,
    target_scope: str | None,
    checkpoint_path: Path,
):
    scaler = StandardScaler()
    hand_scaled = hand_features.copy()
    hand_scaled[split.train] = scaler.fit_transform(hand_features[split.train])
    hand_scaled[split.val] = scaler.transform(hand_features[split.val])
    hand_scaled[split.test] = scaler.transform(hand_features[split.test])

    train_loader, val_loader, test_loader = make_loaders(
        split,
        eeg,
        hand_scaled,
        labels,
        trial_ids,
        batch_size=args.batch_size,
        seed=split.seed,
    )
    model, counts, replaced_paths = build_model(
        checkpoint=args.checkpoint,
        device=device,
        hand_dim=int(hand_features.shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        last_k=last_k,
        rank=rank,
        target_scope=target_scope,
    )
    optimizer = build_optimizer(model, head_lr=args.head_lr, lora_lr=args.lora_lr, weight_decay=args.weight_decay)
    print(
        f"params total={counts['total_params']:,} trainable={counts['trainable_params']:,} lora={counts['lora_params']:,}",
        flush=True,
    )
    if replaced_paths:
        print(f"LoRA targets: {', '.join(replaced_paths)}", flush=True)

    t0 = perf_counter()
    best_epoch, best_val_acc = train_one(
        model,
        train_loader,
        val_loader,
        device,
        args.max_epochs,
        args.patience,
        optimizer,
        checkpoint_path,
    )
    train_seconds = perf_counter() - t0
    aggregate_by_trial = split.split_kind == "subject"
    val_metrics = evaluate(model, val_loader, device, aggregate_by_trial)
    test_metrics = evaluate(model, test_loader, device, aggregate_by_trial)

    del model, optimizer
    torch.cuda.empty_cache()
    return best_epoch, best_val_acc, train_seconds, val_metrics, test_metrics, counts


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.result_dir / "checkpoints"
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    eeg_data = load_eeg_arrays(args.eeg_h5)
    hand_features, _, feature_meta = load_feature_matrix(args.feature_h5)
    if not np.array_equal(eeg_data["label"], feature_meta["label"]):
        raise ValueError("Label mismatch between EEG H5 and feature H5")

    run_config = {
        "workflow": args.workflow,
        "eeg_h5": str(args.eeg_h5),
        "feature_h5": str(args.feature_h5),
        "checkpoint": str(args.checkpoint),
        "split_dir": str(args.split_dir),
        "split_files": selected_split_files(args.split_dir, args.split_kinds, args.seeds),
        "seeds": args.seeds,
        "split_kinds": args.split_kinds,
        "screen_seed": args.screen_seed,
        "screen_last_ks": args.screen_last_ks,
        "screen_ranks": args.screen_ranks,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "head_lr": args.head_lr,
        "lora_lr": args.lora_lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    with (args.result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)

    screen_rows: list[dict] = []
    selected_config: dict | None = None
    full_rows: list[dict] = []

    if args.workflow in {"screen", "all"}:
        split = load_split(args.split_dir, "subject", args.screen_seed)
        print(f"\n=== Stage A Screening: subject seed={args.screen_seed} ===", flush=True)
        for last_k in args.screen_last_ks:
            for rank in args.screen_ranks:
                set_seed(args.screen_seed)
                variant = short_screen_name(last_k, rank)
                checkpoint_path = checkpoint_root / "screen_subject_seed42" / f"{variant}.pt"
                best_epoch, best_val_acc, train_seconds, val_metrics, test_metrics, counts = run_single_variant(
                    eeg=eeg_data["eeg"],
                    hand_features=hand_features,
                    labels=eeg_data["label"],
                    trial_ids=eeg_data["global_trial_index"],
                    split=split,
                    args=args,
                    device=device,
                    last_k=last_k,
                    rank=rank,
                    target_scope="attention_only",
                    checkpoint_path=checkpoint_path,
                )
                row = {
                    "variant": variant,
                    "split_kind": "subject",
                    "seed": args.screen_seed,
                    "last_k": last_k,
                    "rank": rank,
                    "target_scope": "attention_only",
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "train_seconds": round(train_seconds, 4),
                    "checkpoint": str(checkpoint_path),
                    "total_params": counts["total_params"],
                    "trainable_params": counts["trainable_params"],
                    "lora_params": counts["lora_params"],
                }
                for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
                    for scope in ["window", "trial"]:
                        for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                            key = f"{prefix}_{scope}_{metric}"
                            row[key] = "" if metrics[scope] is None else metrics[scope][metric]
                screen_rows.append(row)
                print(
                    f"{variant}: val_acc={val_metrics['window']['acc']*100:.2f}, "
                    f"val_auc={val_metrics['window']['auroc']*100:.2f}, "
                    f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                    f"test_auc={test_metrics['window']['auroc']*100:.2f}",
                    flush=True,
                )

        screen_rows.sort(key=screen_selection_key, reverse=True)
        screen_fieldnames = [
            "variant",
            "split_kind",
            "seed",
            "last_k",
            "rank",
            "target_scope",
            "best_epoch",
            "best_val_acc",
            "train_seconds",
            "checkpoint",
            "total_params",
            "trainable_params",
            "lora_params",
        ]
        for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                screen_fieldnames.append(f"{prefix}_{metric}")
        write_rows_csv(screen_rows, args.result_dir / "screen_subject_seed42.csv", screen_fieldnames)
        best = screen_rows[0]
        selected_config = {
            "selected_last_k": int(best["last_k"]),
            "selected_rank": int(best["rank"]),
            "selection_variant": best["variant"],
            "selection_metrics": {
                "val_window_acc": float(best["val_window_acc"]),
                "val_window_auroc": float(best["val_window_auroc"]),
                "val_trial_auroc": None if best["val_trial_auroc"] == "" else float(best["val_trial_auroc"]),
                "test_window_acc": float(best["test_window_acc"]),
                "test_window_auroc": float(best["test_window_auroc"]),
            },
        }
        with (args.result_dir / "selected_config.json").open("w", encoding="utf-8") as handle:
            json.dump(selected_config, handle, indent=2, ensure_ascii=False)
        print(f"Selected config: last_k={selected_config['selected_last_k']} rank={selected_config['selected_rank']}", flush=True)

    if args.workflow in {"full", "all"}:
        if selected_config is None:
            selected_config = resolve_selected_config(args, args.result_dir)
        last_k = int(selected_config["selected_last_k"])
        rank = int(selected_config["selected_rank"])
        canonical_variants = [
            ("cross_attn_no_lora", None, None, None),
            (variant_name(last_k, rank, "attention_only"), last_k, rank, "attention_only"),
            (variant_name(last_k, rank, "attention_plus_proj"), last_k, rank, "attention_plus_proj"),
        ]
        print(f"\n=== Stage B Full Ablation: last_k={last_k}, rank={rank} ===", flush=True)
        for split_kind in args.split_kinds:
            for seed in args.seeds:
                split = load_split(args.split_dir, split_kind, seed)
                print(f"\n--- split={split_kind} seed={seed} ---", flush=True)
                for variant, v_last_k, v_rank, target_scope in canonical_variants:
                    set_seed(seed)
                    checkpoint_path = checkpoint_root / split_kind / f"seed{seed}" / f"{variant}.pt"
                    best_epoch, best_val_acc, train_seconds, val_metrics, test_metrics, counts = run_single_variant(
                        eeg=eeg_data["eeg"],
                        hand_features=hand_features,
                        labels=eeg_data["label"],
                        trial_ids=eeg_data["global_trial_index"],
                        split=split,
                        args=args,
                        device=device,
                        last_k=v_last_k,
                        rank=v_rank,
                        target_scope=target_scope,
                        checkpoint_path=checkpoint_path,
                    )
                    row = {
                        "split_kind": split_kind,
                        "seed": seed,
                        "variant": variant,
                        "last_k": "" if v_last_k is None else v_last_k,
                        "rank": "" if v_rank is None else v_rank,
                        "target_scope": "" if target_scope is None else target_scope,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "train_seconds": round(train_seconds, 4),
                        "checkpoint": str(checkpoint_path),
                        "total_params": counts["total_params"],
                        "trainable_params": counts["trainable_params"],
                        "lora_params": counts["lora_params"],
                    }
                    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
                        for scope in ["window", "trial"]:
                            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                                key = f"{prefix}_{scope}_{metric}"
                                row[key] = "" if metrics[scope] is None else metrics[scope][metric]
                    full_rows.append(row)
                    print(
                        f"{variant}: best_epoch={best_epoch}, "
                        f"val_acc={val_metrics['window']['acc']*100:.2f}, "
                        f"test_acc={test_metrics['window']['acc']*100:.2f}, "
                        f"test_auc={test_metrics['window']['auroc']*100:.2f}, "
                        f"time={train_seconds:.1f}s",
                        flush=True,
                    )

        full_fieldnames = [
            "split_kind",
            "seed",
            "variant",
            "last_k",
            "rank",
            "target_scope",
            "best_epoch",
            "best_val_acc",
            "train_seconds",
            "checkpoint",
            "total_params",
            "trainable_params",
            "lora_params",
        ]
        for prefix in ["val_window", "test_window", "val_trial", "test_trial"]:
            for metric in ["acc", "precision", "recall", "f1", "auroc", "auprc"]:
                full_fieldnames.append(f"{prefix}_{metric}")
        write_rows_csv(full_rows, args.result_dir / "all_results.csv", full_fieldnames)

    full_summary = summarize_full_rows(full_rows, args.result_dir) if full_rows else None
    write_readme(args.result_dir, run_config, screen_rows, selected_config, full_summary)
    print(f"\nSaved Step-6 LoRA results to: {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
