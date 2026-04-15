#!/usr/bin/env python3
"""Build a unified validation/test comparison table across all current COMP4 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_BASELINE = DATA_DIR / "feature_baseline_results" / "all_results.csv"
DEFAULT_FUSION = DATA_DIR / "step5_fusion_results" / "all_results.csv"
DEFAULT_MDJPT_ONLY = DATA_DIR / "mdjpt_only_results" / "all_results.csv"
DEFAULT_STEP6_LORA = DATA_DIR / "step6_lora_results" / "all_results.csv"
DEFAULT_STEP7_MOE = DATA_DIR / "step7_moe_results" / "all_results.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "comparison_results_all.csv"
DEFAULT_OUTPUT_MD = DATA_DIR / "comparison_results_all.md"

FAMILY_ORDER = {"handcrafted": 0, "mdjpt_only": 1, "fusion": 2, "step6_lora": 3, "step7_moe": 4}
METHOD_ORDER = {
    "logistic_regression": 0,
    "svm": 1,
    "xgboost": 2,
    "mlp": 3,
    "frozen_mlp": 4,
    "full_finetune": 5,
    "mdjpt_only": 6,
    "concat": 7,
    "gating": 8,
    "cross_attn": 9,
    "cross_attn_no_lora": 10,
    "cross_attn_single_head": 11,
    "cross_attn_dual_expert_avg": 12,
    "cross_attn_moe_unsup": 13,
    "cross_attn_moe_sup_l03": 14,
    "cross_attn_moe_sup_l05": 15,
}
DISPLAY_NAMES = {
    "logistic_regression": "LogisticRegression",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "frozen_mlp": "Frozen+MLP",
    "full_finetune": "Full Finetune",
    "mdjpt_only": "mdJPT Only",
    "concat": "Concat",
    "gating": "Gating",
    "cross_attn": "CrossAttn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-baseline-csv", type=Path, default=DEFAULT_FEATURE_BASELINE)
    parser.add_argument("--fusion-csv", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--mdjpt-only-csv", type=Path, default=DEFAULT_MDJPT_ONLY)
    parser.add_argument("--step6-lora-csv", type=Path, default=DEFAULT_STEP6_LORA)
    parser.add_argument("--step7-moe-csv", type=Path, default=DEFAULT_STEP7_MOE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def maybe_float(value: str):
    return None if value == "" else float(value)


def agg(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def fmt(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean * 100:.2f} +- {std * 100:.2f}"


def load_handcrafted_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "family": "handcrafted",
                    "method": row["model"],
                    "split_kind": row["split_kind"],
                    "seed": int(row["seed"]),
                    "val_acc": float(row["val_acc"]),
                    "val_auroc": float(row["val_auroc"]),
                    "test_acc": float(row["test_acc"]),
                    "test_auroc": float(row["test_auroc"]),
                    "val_trial_acc": None,
                    "val_trial_auroc": None,
                    "test_trial_acc": None,
                    "test_trial_auroc": None,
                }
            )
    return rows


def load_fusion_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "family": "fusion",
                    "method": row["variant"],
                    "split_kind": row["split_kind"],
                    "seed": int(row["seed"]),
                    "val_acc": float(row["val_window_acc"]),
                    "val_auroc": float(row["val_window_auroc"]),
                    "test_acc": float(row["test_window_acc"]),
                    "test_auroc": float(row["test_window_auroc"]),
                    "val_trial_acc": maybe_float(row["val_trial_acc"]),
                    "val_trial_auroc": maybe_float(row["val_trial_auroc"]),
                    "test_trial_acc": maybe_float(row["test_trial_acc"]),
                    "test_trial_auroc": maybe_float(row["test_trial_auroc"]),
                }
            )
    return rows


def load_mdjpt_only_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "family": "mdjpt_only",
                    "method": row["mode"],
                    "split_kind": row["split_kind"],
                    "seed": int(row["seed"]),
                    "val_acc": float(row["val_window_acc"]),
                    "val_auroc": float(row["val_window_auroc"]),
                    "test_acc": float(row["test_window_acc"]),
                    "test_auroc": float(row["test_window_auroc"]),
                    "val_trial_acc": maybe_float(row["val_trial_acc"]),
                    "val_trial_auroc": maybe_float(row["val_trial_auroc"]),
                    "test_trial_acc": maybe_float(row["test_trial_acc"]),
                    "test_trial_auroc": maybe_float(row["test_trial_auroc"]),
                }
            )
    return rows


def load_step6_lora_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "family": "step6_lora",
                    "method": row["variant"],
                    "split_kind": row["split_kind"],
                    "seed": int(row["seed"]),
                    "val_acc": float(row["val_window_acc"]),
                    "val_auroc": float(row["val_window_auroc"]),
                    "test_acc": float(row["test_window_acc"]),
                    "test_auroc": float(row["test_window_auroc"]),
                    "val_trial_acc": maybe_float(row["val_trial_acc"]),
                    "val_trial_auroc": maybe_float(row["val_trial_auroc"]),
                    "test_trial_acc": maybe_float(row["test_trial_acc"]),
                    "test_trial_auroc": maybe_float(row["test_trial_auroc"]),
                }
            )
    return rows


def load_step7_moe_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "family": "step7_moe",
                    "method": row["variant"],
                    "split_kind": row["split_kind"],
                    "seed": int(row["seed"]),
                    "val_acc": float(row["val_window_acc"]),
                    "val_auroc": float(row["val_window_auroc"]),
                    "test_acc": float(row["test_window_acc"]),
                    "test_auroc": float(row["test_window_auroc"]),
                    "val_trial_acc": maybe_float(row["val_trial_acc"]),
                    "val_trial_auroc": maybe_float(row["val_trial_auroc"]),
                    "test_trial_acc": maybe_float(row["test_trial_acc"]),
                    "test_trial_auroc": maybe_float(row["test_trial_auroc"]),
                }
            )
    return rows


def method_sort_key(method: str) -> tuple[int, str]:
    return (METHOD_ORDER.get(method, 10_000), method)


def step6_display_name(method: str) -> str:
    if method == "cross_attn_no_lora":
        return "CrossAttn NoLoRA"
    if method.startswith("cross_attn_lora_last"):
        parts = method.split("_")
        last_part = next((part for part in parts if part.startswith("last")), "last?")
        rank_part = next((part for part in parts if part.startswith("r")), "r?")
        target_part = parts[-1]
        target_label = "Attn+Proj" if target_part == "attnproj" else "Attn"
        return f"CrossAttn LoRA {last_part.capitalize()} {rank_part.upper()} {target_label}"
    return method


def step7_display_name(method: str) -> str:
    return {
        "cross_attn_single_head": "CrossAttn SingleHead",
        "cross_attn_dual_expert_avg": "CrossAttn DualExpertAvg",
        "cross_attn_moe_unsup": "CrossAttn MoE Unsup",
        "cross_attn_moe_sup_l03": "CrossAttn MoE Sup L0.3",
        "cross_attn_moe_sup_l05": "CrossAttn MoE Sup L0.5",
    }.get(method, method)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {(row["split_kind"], row["family"], row["method"]) for row in rows},
        key=lambda item: (item[0], FAMILY_ORDER[item[1]], method_sort_key(item[2])),
    )

    summary = []
    for split_kind, family, method in keys:
        selected = [row for row in rows if row["split_kind"] == split_kind and row["family"] == family and row["method"] == method]
        record = {
            "split_kind": split_kind,
            "family": family,
            "method": method,
            "display_method": DISPLAY_NAMES.get(method, step7_display_name(step6_display_name(method))),
            "n_seeds": len(selected),
        }
        for metric in ["val_acc", "val_auroc", "test_acc", "test_auroc", "val_trial_acc", "val_trial_auroc", "test_trial_acc", "test_trial_auroc"]:
            values = [row[metric] for row in selected if row[metric] is not None]
            if values:
                mean, std = agg(values)
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            else:
                record[f"{metric}_mean"] = None
                record[f"{metric}_std"] = None

        record["delta_acc_mean"] = None if record["val_acc_mean"] is None else record["test_acc_mean"] - record["val_acc_mean"]
        record["delta_auroc_mean"] = None if record["val_auroc_mean"] is None else record["test_auroc_mean"] - record["val_auroc_mean"]
        summary.append(record)
    return summary


def write_csv(summary: list[dict], path: Path) -> None:
    fieldnames = [
        "split_kind",
        "family",
        "method",
        "display_method",
        "n_seeds",
        "val_acc_mean",
        "val_acc_std",
        "test_acc_mean",
        "test_acc_std",
        "val_auroc_mean",
        "val_auroc_std",
        "test_auroc_mean",
        "test_auroc_std",
        "val_trial_acc_mean",
        "val_trial_acc_std",
        "test_trial_acc_mean",
        "test_trial_acc_std",
        "val_trial_auroc_mean",
        "val_trial_auroc_std",
        "test_trial_auroc_mean",
        "test_trial_auroc_std",
        "delta_acc_mean",
        "delta_auroc_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def family_label(family: str) -> str:
    return {
        "handcrafted": "Handcrafted",
        "mdjpt_only": "mdJPT Only",
        "fusion": "Step5 Fusion",
        "step6_lora": "Step6 LoRA",
        "step7_moe": "Step7 MoE",
    }[family]


def write_markdown(summary: list[dict], path: Path) -> None:
    lines = [
        "# Unified COMP4 Comparison Table",
        "",
        "Fixed splits: `subject/segment x seeds {42, 3407, 2025}`.",
        "This table combines handcrafted baselines, mdJPT-only baselines, Step-5 fusion models, Step-6 LoRA ablations, and Step-7 MoE variants.",
        "",
        "## Window-Level Comparison",
        "",
        "| Split | Family | Method | Val Acc | Val AUROC | Test Acc | Test AUROC | Test-Val Acc |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in summary:
        delta_acc_text = "-" if record["delta_acc_mean"] is None else f"{record['delta_acc_mean'] * 100:.2f}"
        lines.append(
            f"| {record['split_kind']} | {family_label(record['family'])} | {record['display_method']} | "
            f"{fmt(record['val_acc_mean'], record['val_acc_std'])} | "
            f"{fmt(record['val_auroc_mean'], record['val_auroc_std'])} | "
            f"{fmt(record['test_acc_mean'], record['test_acc_std'])} | "
            f"{fmt(record['test_auroc_mean'], record['test_auroc_std'])} | "
            f"{delta_acc_text} |"
        )

    lines.extend(
        [
            "",
            "## Subject-Only Trial-Level Comparison",
            "",
            "| Family | Method | Val Trial Acc | Val Trial AUROC | Test Trial Acc | Test Trial AUROC | Test Trial Acc Std |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in [row for row in summary if row["split_kind"] == "subject"]:
        test_trial_acc_std_text = "-" if record["test_trial_acc_std"] is None else f"{record['test_trial_acc_std'] * 100:.2f}"
        lines.append(
            f"| {family_label(record['family'])} | {record['display_method']} | "
            f"{fmt(record['val_trial_acc_mean'], record['val_trial_acc_std'])} | "
            f"{fmt(record['val_trial_auroc_mean'], record['val_trial_auroc_std'])} | "
            f"{fmt(record['test_trial_acc_mean'], record['test_trial_acc_std'])} | "
            f"{fmt(record['test_trial_auroc_mean'], record['test_trial_auroc_std'])} | "
            f"{test_trial_acc_std_text} |"
        )

    lines.extend(
        [
            "",
            "## Stability View",
            "",
            "| Split | Family | Method | Test Acc Std | Test AUROC Std | Quick Read |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for record in summary:
        acc_std = record["test_acc_std"]
        auc_std = record["test_auroc_std"]
        acc_std_text = "-" if acc_std is None else f"{acc_std * 100:.2f}"
        auc_std_text = "-" if auc_std is None else f"{auc_std * 100:.2f}"
        if acc_std is None:
            note = "-"
        elif acc_std < 0.015:
            note = "low variance"
        elif acc_std < 0.03:
            note = "moderate variance"
        else:
            note = "high variance"
        lines.append(
            f"| {record['split_kind']} | {family_label(record['family'])} | {record['display_method']} | "
            f"{acc_std_text} | "
            f"{auc_std_text} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `+-` denotes mean ± std across the 3 fixed seeds.",
            "- Trial-level metrics are only available for mdJPT-based models where segment probabilities were aggregated within each trial.",
            "- Cross-seed std is only a proxy for split sensitivity / subject heterogeneity, not a direct per-subject metric.",
            "- `fusion / mdJPT Only` is numerically the same setup as `mdJPT Only / Frozen+MLP`; both are kept for traceability.",
            "- Step6 rows are loaded automatically when `data/step6_lora_results/all_results.csv` exists.",
            "- Step7 rows are loaded automatically when `data/step7_moe_results/all_results.csv` exists.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(load_handcrafted_rows(args.feature_baseline_csv))
    rows.extend(load_fusion_rows(args.fusion_csv))
    rows.extend(load_mdjpt_only_rows(args.mdjpt_only_csv))
    rows.extend(load_step6_lora_rows(args.step6_lora_csv))
    rows.extend(load_step7_moe_rows(args.step7_moe_csv))
    summary = aggregate_rows(rows)
    write_csv(summary, args.output_csv)
    write_markdown(summary, args.output_md)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote MD:  {args.output_md}")


if __name__ == "__main__":
    main()
