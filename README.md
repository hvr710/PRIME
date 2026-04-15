# PRIME: mdJPT COMP4 EEG Emotion Recognition Codebase

This repository contains the code used for COMP4 EEG emotion recognition experiments built on top of the mdJPT EEG foundation model.

The GitHub version of this repository is intentionally code-only:

- tracked: source code, configs, preprocessing scripts, launch scripts, and lightweight documentation
- not tracked: experiment outputs, logs, checkpoints, cached embeddings, processed H5 files, and other large local artifacts

## Repository Layout

- `src/`: mdJPT model code and downstream training utilities
- `data/`: experiment runners and data-processing scripts
- `preprocess/`: preprocessing and export scripts, plus small support assets required by preprocessing
- `cfgs_multi/`: dataset and model configuration files
- `tools/`: small helper shell scripts
- root `run_*.sh` / `train_*.py`: top-level launchers for major experiments

## Required Local Artifacts

These files are expected locally but are not committed to Git:

- `data/comp4_len5_step5_mapped60.h5`
- `data/features_comp4_len5_step5_mapped60.h5`
- `data/mdjpt_step5_embeddings.h5`
- pretrained checkpoints under `log/pretrain/ckpt/` when a workflow depends on them

Experiment outputs should be written to local directories such as `outputs/`, `lightning_logs/`, or `log/`. Those directories are ignored by Git on purpose.

## Environment

Example environment setup:

```bash
conda activate labram_mamba
pip install -r requirements.txt
```

## Common Entry Points

- Main multi-stage runner: `data/run_balanced811_6seed_mainline.py`
- Main wrapper script: `run_final_main_experiment.sh`
- Feature extraction: `data/extract_features.py`
- mdJPT embedding export: `data/export_mdjpt_step5_embeddings.py`
- COMP4 H5 export: `preprocess/export_comp4_len5_step5_mapped60_h5.py`

## Example Usage

```bash
bash run_final_main_experiment.sh
```

Or call the Python runner directly:

```bash
python data/run_balanced811_6seed_mainline.py make_splits --output-root outputs/my_run
python data/run_balanced811_6seed_mainline.py build_cache --output-root outputs/my_run
```

## Notes

- `channel_interpolate.npy` is kept in the repository because some model and preprocessing code loads it directly.
- Historical results and local exploratory notes are intentionally left out of the GitHub repo.
