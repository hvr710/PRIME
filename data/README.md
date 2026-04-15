# Data Directory Guide

This directory should contain reusable COMP4 data artifacts and scripts. Experiment outputs should live in `../outputs/`.

## Core H5 Files

| File | Main Contents | Notes |
|---|---|---|
| `comp4_len5_step5_mapped60.h5` | EEG segments, labels, subject/trial metadata | Main EEG array shape is `(4800, 60, 625)`. Each sample is a 5-second segment at 125 Hz. |
| `features_comp4_len5_step5_mapped60.h5` | Handcrafted features and metadata | Feature dimension is `1520`. Metadata includes label, split-related indices, and subject group labels. |
| `mdjpt_step5_embeddings.h5` | Frozen mdJPT cached embeddings | Includes `pool_1024`, `channel_tokens_32`, and `patch_tokens_32`. |

## Important Scripts

| Script | Purpose |
|---|---|
| `extract_features.py` | Extract handcrafted EEG features from the processed 5-second segments. |
| `attach_feature_metadata.py` | Attach label/split/subject/trial metadata to the feature H5. |
| `export_mdjpt_step5_embeddings.py` | Export frozen mdJPT embeddings and tokens from the processed EEG H5. |
| `run_feature_baselines.py` | Train classical feature baselines on handcrafted features. |
| `train_mdjpt_only_step5.py` | Train mdJPT-only frozen MLP or full fine-tuning baselines. |
| `train_fusion_step5.py` | Train frozen mdJPT + handcrafted feature fusion models. |
| `train_fusion_step6_lora.py` | Historical LoRA adapter ablation script. |
| `train_step7_moe.py` | Historical Step7 MoE script. |
| `run_balanced811_6seed_mainline.py` | Current unified split/cache/train/summarize runner. |
| `launch_balanced811_6seed_parallel.sh` | Parallel launch helper for the 6-seed mainline. |

## Result Cleanup

Historical result folders that previously lived under `data/` were moved to:

```text
../outputs/archived_data_results/
```

The compatibility symlinks were later removed so that `data/` stays clean. New experiments should not write result folders under `data/`.

## Smoke Cleanup

Smoke-test folders were removed from `data/`:

- `sequence_ablation_smoke`
- `step6_lora_results_smoke`
- `step7_moe_results_smoke`

Current expected top-level content in `data/` is:

- H5 data artifacts
- preprocessing / training / report-building scripts
- lightweight metadata CSV/JSON/MD files
