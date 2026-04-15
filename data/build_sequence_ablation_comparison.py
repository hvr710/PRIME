#!/usr/bin/env python3
"""Build a comparison report for sequence smoothing/normalization ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_ABLATION_CSV = DATA_DIR / "sequence_ablation_results" / "all_results.csv"
DEFAULT_OUTPUT_MD = DATA_DIR / "sequence_ablation_results" / "comparison_report.md"
DEFAULT_OUTPUT_CSV = DATA_DIR / "sequence_ablation_results" / "comparison_summary.csv"
DEFAULT_OLD_LEAKY_MLP = DATA_DIR.parent / "mlp_results" / "pretrain" / "len5_step5_seg_random_seed42" / "COMP4_mlp_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-csv", type=Path, default=DEFAULT_ABLATION_CSV)
    parser.add_argument("--old-leaky-mlp-json", type=Path, default=DEFAULT_OLD_LEAKY_MLP)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(row["split_kind"], row["sequence_variant"]) for row in rows})
    baseline_by_split = {}
    for split_kind, variant in keys:
        selected = [row for row in rows if row["split_kind"] == split_kind and row["sequence_variant"] == variant]
        record = {
            "split_kind": split_kind,
            "sequence_variant": variant,
            "n_seeds": len(selected),
        }
        for metric in ["acc", "f1", "auroc"]:
            vals = np.asarray([float(row[f"test_window_{metric}"]) for row in selected], dtype=np.float64)
            record[f"window_{metric}_mean"] = float(vals.mean())
            record[f"window_{metric}_std"] = float(vals.std(ddof=0))
        trial_vals = [row["test_trial_auroc"] for row in selected if row["test_trial_auroc"] != ""]
        if trial_vals:
            vals = np.asarray([float(value) for value in trial_vals], dtype=np.float64)
            record["trial_auroc_mean"] = float(vals.mean())
            record["trial_auroc_std"] = float(vals.std(ddof=0))
        else:
            record["trial_auroc_mean"] = None
            record["trial_auroc_std"] = None
        out.append(record)
        if variant == "baseline":
            baseline_by_split[split_kind] = record

    for record in out:
        base = baseline_by_split[record["split_kind"]]
        record["delta_window_acc"] = record["window_acc_mean"] - base["window_acc_mean"]
        record["delta_window_auroc"] = record["window_auroc_mean"] - base["window_auroc_mean"]
        if record["trial_auroc_mean"] is not None and base["trial_auroc_mean"] is not None:
            record["delta_trial_auroc"] = record["trial_auroc_mean"] - base["trial_auroc_mean"]
        else:
            record["delta_trial_auroc"] = None
    return out


def pct(mean: float | None, std: float | None = None) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split_kind",
        "sequence_variant",
        "n_seeds",
        "window_acc_mean",
        "window_acc_std",
        "window_f1_mean",
        "window_f1_std",
        "window_auroc_mean",
        "window_auroc_std",
        "trial_auroc_mean",
        "trial_auroc_std",
        "delta_window_acc",
        "delta_window_auroc",
        "delta_trial_auroc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_old_leaky(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["test"]["window"]


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(rows: list[dict], old_leaky: dict) -> str:
    variant_order = [
        "baseline",
        "trial_smooth_hand",
        "trial_smooth_all",
        "subject_adapt_hand",
        "adapt_then_smooth_hand",
        "subject_adapt_all",
        "adapt_then_smooth_all",
    ]
    split_order = ["subject", "trial", "segment"]
    row_map = {(row["split_kind"], row["sequence_variant"]): row for row in rows}

    lines = [
        "# Sequence Smoothing / Normalization Ablation",
        "",
        "This report compares safe split-wise sequence processing against the previous Step-5 CrossAttn baseline. All new variants split the data first, then apply normalization or smoothing inside each split only.",
        "",
        "## Main Ablation Table",
        "",
    ]
    table_rows = []
    for split in split_order:
        for variant in variant_order:
            row = row_map[(split, variant)]
            table_rows.append(
                [
                    split,
                    variant,
                    pct(row["window_acc_mean"], row["window_acc_std"]),
                    pct(row["window_f1_mean"], row["window_f1_std"]),
                    pct(row["window_auroc_mean"], row["window_auroc_std"]),
                    delta(row["delta_window_auroc"]),
                    pct(row["trial_auroc_mean"], row["trial_auroc_std"]),
                    delta(row["delta_trial_auroc"]),
                ]
            )
    lines.append(
        table(
            ["Split", "Variant", "Window Acc", "Window F1", "Window AUROC", "Δ AUROC", "Trial AUROC", "Δ Trial AUROC"],
            table_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Comparison With Previous Results",
            "",
            table(
                ["Setting", "Meaning", "Test Acc", "Test AUROC", "Use In Report"],
                [
                    [
                        "Old ext_fea segment-random MLP",
                        "Unsafe segment split plus sequence-level feature leakage",
                        pct(old_leaky["acc"]),
                        pct(old_leaky["auroc"]),
                        "No",
                    ],
                    [
                        "Step5 CrossAttn subject baseline",
                        "Strict cross-subject main baseline",
                        pct(row_map[("subject", "baseline")]["window_acc_mean"], row_map[("subject", "baseline")]["window_acc_std"]),
                        pct(row_map[("subject", "baseline")]["window_auroc_mean"], row_map[("subject", "baseline")]["window_auroc_std"]),
                        "Yes",
                    ],
                    [
                        "Step5 CrossAttn trial baseline",
                        "Same-subject but no same-trial leakage",
                        pct(row_map[("trial", "baseline")]["window_acc_mean"], row_map[("trial", "baseline")]["window_acc_std"]),
                        pct(row_map[("trial", "baseline")]["window_auroc_mean"], row_map[("trial", "baseline")]["window_auroc_std"]),
                        "Supplement",
                    ],
                    [
                        "Step5 CrossAttn segment baseline",
                        "Segment random, same-trial leakage remains",
                        pct(row_map[("segment", "baseline")]["window_acc_mean"], row_map[("segment", "baseline")]["window_acc_std"]),
                        pct(row_map[("segment", "baseline")]["window_auroc_mean"], row_map[("segment", "baseline")]["window_auroc_std"]),
                        "Sanity only",
                    ],
                    [
                        "Subject adaptive norm + smoothing",
                        "Test-time adaptation using unlabeled test-subject statistics",
                        pct(row_map[("subject", "adapt_then_smooth_hand")]["window_acc_mean"], row_map[("subject", "adapt_then_smooth_hand")]["window_acc_std"]),
                        pct(row_map[("subject", "adapt_then_smooth_hand")]["window_auroc_mean"], row_map[("subject", "adapt_then_smooth_hand")]["window_auroc_std"]),
                        "Report separately",
                    ],
                ],
            ),
            "",
            "## Key Findings",
            "",
            "1. Subject-level adaptive normalization is highly effective for cross-subject decoding, but it is a test-time adaptation setting. The best subject result is `adapt_then_smooth_hand`, improving window AUROC by "
            f"{delta(row_map[('subject', 'adapt_then_smooth_hand')]['delta_window_auroc'])} over the no-adaptation CrossAttn baseline.",
            "2. Trial-level causal smoothing gives a modest gain for trial split window decoding. `trial_smooth_hand` improves trial-split window AUROC by "
            f"{delta(row_map[('trial', 'trial_smooth_hand')]['delta_window_auroc'])}, while keeping same-trial leakage removed.",
            "3. Segment-random evaluation does not benefit from the safe smoothing variants. Its baseline is already inflated by same-trial leakage, and smoothing often blurs discriminative local windows.",
            "4. The old `99%+` ext_fea MLP result should not be used as evidence of generalization. It combines segment-random splitting with sequence-level preprocessing before the split.",
            "",
            "## Recommended Reporting",
            "",
            "- Main paper: keep the strict subject-split CrossAttn baseline and optionally add `subject_adapt_hand` / `adapt_then_smooth_hand` as a clearly labeled test-time adaptation result.",
            "- Supplement: add trial split to show the effect of safe causal smoothing without same-trial leakage.",
            "- Do not use the old ext_fea segment-random MLP as a main result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with args.ablation_csv.open("r", encoding="utf-8") as handle:
        rows = aggregate(list(csv.DictReader(handle)))
    old_leaky = load_old_leaky(args.old_leaky_mlp_json)
    write_csv(rows, args.output_csv)
    args.output_md.write_text(build_markdown(rows, old_leaky), encoding="utf-8")
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote MD:  {args.output_md}")


if __name__ == "__main__":
    main()
