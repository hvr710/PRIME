#!/usr/bin/env python3
"""Build a paper-friendly summary across subject-level evaluation protocols."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "outputs"

DEFAULT_PROTOCOLS = {
    "subject_811_balanced": OUTPUT_ROOT / "subject_811_balanced" / "comparison_results_all.csv",
    "subject_80_20_teststop_balanced": OUTPUT_ROOT / "subject_80_20_teststop_balanced" / "comparison_results_all.csv",
}

DEFAULT_OUTPUT_CSV = OUTPUT_ROOT / "subject_protocol_summary.csv"
DEFAULT_OUTPUT_MD = OUTPUT_ROOT / "subject_protocol_summary.md"

PROTOCOL_LABELS = {
    "subject_811_balanced": "Strict 8:1:1",
    "subject_80_20_teststop_balanced": "80/20 Test-Stop",
}

FAMILY_LABELS = {
    "handcrafted": "Handcrafted",
    "mdjpt_only": "mdJPT-only",
    "fusion": "Step5 Fusion",
    "step6_lora": "Step6 LoRA",
    "step7_moe": "Step7 MoE",
}

CORE_METHODS = {
    "handcrafted": "XGBoost",
    "mdjpt_only": "Full Finetune",
    "fusion": "CrossAttn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-csv",
        action="append",
        nargs=2,
        metavar=("PROTOCOL_NAME", "CSV_PATH"),
        help="Override a protocol comparison csv.",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def maybe_float(value: str) -> float | None:
    return None if value == "" else float(value)


def fmt_pct(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} +- {std * 100:.2f}"


def load_subject_rows(protocol: str, path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split_kind"] != "subject":
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "protocol_label": PROTOCOL_LABELS.get(protocol, protocol),
                    "family": row["family"],
                    "family_label": FAMILY_LABELS.get(row["family"], row["family"]),
                    "display_method": row["display_method"],
                    "n_seeds": int(row["n_seeds"]),
                    "val_acc_mean": maybe_float(row["val_acc_mean"]),
                    "val_acc_std": maybe_float(row["val_acc_std"]),
                    "test_acc_mean": maybe_float(row["test_acc_mean"]),
                    "test_acc_std": maybe_float(row["test_acc_std"]),
                    "val_auroc_mean": maybe_float(row["val_auroc_mean"]),
                    "val_auroc_std": maybe_float(row["val_auroc_std"]),
                    "test_auroc_mean": maybe_float(row["test_auroc_mean"]),
                    "test_auroc_std": maybe_float(row["test_auroc_std"]),
                    "test_trial_auroc_mean": maybe_float(row["test_trial_auroc_mean"]),
                    "test_trial_auroc_std": maybe_float(row["test_trial_auroc_std"]),
                }
            )
    return rows


def best_by_family(rows: list[dict]) -> list[dict]:
    best = {}
    for row in rows:
        family = row["family"]
        metric = row["test_auroc_mean"]
        if metric is None:
            continue
        if family not in best or metric > best[family]["test_auroc_mean"]:
            best[family] = row
    return [best[key] for key in sorted(best.keys())]


def selected_core_rows(rows: list[dict]) -> list[dict]:
    best = {row["family"]: row for row in best_by_family(rows)}
    index = {(row["family"], row["display_method"]): row for row in rows}
    selected = []
    for family, display_method in CORE_METHODS.items():
        if (family, display_method) in index:
            selected.append(index[(family, display_method)])
    for family in ["step6_lora", "step7_moe"]:
        if family in best:
            selected.append(best[family])
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "protocol",
        "protocol_label",
        "family",
        "family_label",
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
        "test_trial_auroc_mean",
        "test_trial_auroc_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(protocol_rows: dict[str, list[dict]]) -> str:
    lines = [
        "# Subject-Level Results Summary",
        "",
        "## Protocol Notes",
        "",
        "- `Strict 8:1:1`: balanced subject split with independent train/val/test.",
        "- `80/20 Test-Stop`: balanced subject split with train/test only; `val=test` is used for early stopping, so this protocol is optimistic and should not be the main claim table.",
        "",
        "## Best Per Family",
        "",
    ]

    best_rows = []
    csv_rows = []
    for protocol in ["subject_811_balanced", "subject_80_20_teststop_balanced"]:
        rows = protocol_rows[protocol]
        for row in best_by_family(rows):
            best_rows.append(
                [
                    row["protocol_label"],
                    row["family_label"],
                    row["display_method"],
                    fmt_pct(row["test_acc_mean"], row["test_acc_std"]),
                    fmt_pct(row["test_auroc_mean"], row["test_auroc_std"]),
                    fmt_pct(row["test_trial_auroc_mean"], row["test_trial_auroc_std"]),
                ]
            )
            csv_rows.append({"section": "best_per_family", **row})
    lines.append(
        markdown_table(
            ["Protocol", "Family", "Best Method", "Test Acc", "Test AUROC", "Test Trial AUROC"],
            best_rows,
        )
    )
    lines.extend(["", "## Main Comparison", ""])

    main_rows = []
    for protocol in ["subject_811_balanced", "subject_80_20_teststop_balanced"]:
        rows = protocol_rows[protocol]
        for row in selected_core_rows(rows):
            main_rows.append(
                [
                    row["protocol_label"],
                    row["family_label"],
                    row["display_method"],
                    fmt_pct(row["test_acc_mean"], row["test_acc_std"]),
                    fmt_pct(row["test_auroc_mean"], row["test_auroc_std"]),
                    fmt_pct(row["test_trial_auroc_mean"], row["test_trial_auroc_std"]),
                ]
            )
            csv_rows.append({"section": "main_comparison", **row})
    lines.append(
        markdown_table(
            ["Protocol", "Family", "Method", "Test Acc", "Test AUROC", "Test Trial AUROC"],
            main_rows,
        )
    )
    lines.extend(["", "## Key Findings", ""])

    strict_best = max(best_by_family(protocol_rows["subject_811_balanced"]), key=lambda row: row["test_auroc_mean"] or -1)
    teststop_best = max(
        best_by_family(protocol_rows["subject_80_20_teststop_balanced"]),
        key=lambda row: row["test_auroc_mean"] or -1,
    )
    lines.append(
        f"1. `Strict 8:1:1` 的最优方法是 `{strict_best['family_label']} / {strict_best['display_method']}`，"
        f"`Test AUROC = {fmt_pct(strict_best['test_auroc_mean'], strict_best['test_auroc_std'])}`，"
        f"`Test Trial AUROC = {fmt_pct(strict_best['test_trial_auroc_mean'], strict_best['test_trial_auroc_std'])}`。"
    )
    lines.append(
        f"2. `80/20 Test-Stop` 的最优方法是 `{teststop_best['family_label']} / {teststop_best['display_method']}`，"
        f"`Test AUROC = {fmt_pct(teststop_best['test_auroc_mean'], teststop_best['test_auroc_std'])}`，"
        f"`Test Trial AUROC = {fmt_pct(teststop_best['test_trial_auroc_mean'], teststop_best['test_trial_auroc_std'])}`。"
    )

    strict_index = {(row["family"], row["display_method"]): row for row in protocol_rows["subject_811_balanced"]}
    teststop_index = {(row["family"], row["display_method"]): row for row in protocol_rows["subject_80_20_teststop_balanced"]}
    step5_strict = strict_index[("fusion", "CrossAttn")]
    step5_teststop = teststop_index[("fusion", "CrossAttn")]
    lines.append(
        f"3. 当前最稳的主线仍然是 `Step5 CrossAttn`：在 `Strict 8:1:1` 上达到 "
        f"`{fmt_pct(step5_strict['test_auroc_mean'], step5_strict['test_auroc_std'])}` test AUROC / "
        f"`{fmt_pct(step5_strict['test_trial_auroc_mean'], step5_strict['test_trial_auroc_std'])}` test trial AUROC；"
        f"在 `80/20 Test-Stop` 上为 "
        f"`{fmt_pct(step5_teststop['test_auroc_mean'], step5_teststop['test_auroc_std'])}` / "
        f"`{fmt_pct(step5_teststop['test_trial_auroc_mean'], step5_teststop['test_trial_auroc_std'])}`。"
    )
    lines.append(
        "4. `80/20 Test-Stop` 因为让测试集参与 early stopping，只适合做补充结果，不适合当主结论。正式写作时应优先报告 `Strict 8:1:1`。"
    )

    return "\n".join(lines) + "\n", csv_rows


def main() -> None:
    args = parse_args()
    protocol_paths = dict(DEFAULT_PROTOCOLS)
    if args.protocol_csv:
        for protocol, path in args.protocol_csv:
            protocol_paths[protocol] = Path(path)

    protocol_rows = {}
    for protocol, path in protocol_paths.items():
        protocol_rows[protocol] = load_subject_rows(protocol, path)

    markdown, csv_rows = build_markdown(protocol_rows)
    write_csv(args.output_csv, csv_rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote MD:  {args.output_md}")


if __name__ == "__main__":
    main()
