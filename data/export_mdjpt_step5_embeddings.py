#!/usr/bin/env python3
"""Export frozen mdJPT embeddings used by the step-5 fusion experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_INPUT_H5 = DATA_DIR / "comp4_len5_step5_mapped60.h5"
DEFAULT_OUTPUT_H5 = DATA_DIR / "mdjpt_step5_embeddings.h5"
DEFAULT_CKPT = REPO_ROOT / "log" / "pretrain" / "ckpt" / "epoch=19.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--output-h5", type=Path, default=DEFAULT_OUTPUT_H5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def copy_dataset(src: h5py.File, dst: h5py.File, name: str) -> None:
    if name in src:
        dst.create_dataset(name, data=src[name][:], dtype=src[name].dtype)


def create_output_datasets(
    handle: h5py.File,
    n_samples: int,
    pool_dim: int,
    n_channels: int,
    n_patches: int,
    token_dim: int,
):
    pool_ds = handle.create_dataset(
        "pool_1024",
        shape=(n_samples, pool_dim),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=(min(256, n_samples), pool_dim),
    )
    channel_ds = handle.create_dataset(
        "channel_tokens_32",
        shape=(n_samples, n_channels, token_dim),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=(min(128, n_samples), n_channels, token_dim),
    )
    patch_ds = handle.create_dataset(
        "patch_tokens_32",
        shape=(n_samples, n_patches, token_dim),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=(min(128, n_samples), n_patches, token_dim),
    )
    return pool_ds, channel_ds, patch_ds


def main() -> None:
    args = parse_args()
    if not args.input_h5.exists():
        raise FileNotFoundError(args.input_h5)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.output_h5.exists() and not args.overwrite:
        with h5py.File(args.output_h5, "r") as handle:
            print(f"Output exists, skipping export: {args.output_h5}")
            print("pool_1024", handle["pool_1024"].shape)
            print("channel_tokens_32", handle["channel_tokens_32"].shape)
            print("patch_tokens_32", handle["patch_tokens_32"].shape)
        return

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    if args.output_h5.exists():
        args.output_h5.unlink()

    os.chdir(REPO_ROOT)
    from src.model.MultiModel_PL import MultiModel_PL

    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    cfg = build_backbone_cfg()
    model = MultiModel_PL.load_from_checkpoint(str(args.checkpoint), cfg=cfg, strict=False, map_location="cpu")
    model.save_fea = False
    if hasattr(model, "cnn_encoder"):
        model.cnn_encoder.set_saveFea(False)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    with h5py.File(args.input_h5, "r") as src, h5py.File(args.output_h5, "w") as dst:
        eeg_ds = src["eeg"]
        n_samples = int(eeg_ds.shape[0])
        pool_ds = channel_ds = patch_ds = None

        dst.attrs["source_h5"] = str(args.input_h5)
        dst.attrs["checkpoint"] = str(args.checkpoint)
        dst.attrs["fs"] = src.attrs.get("fs", 125)
        dst.attrs["segment_seconds"] = src.attrs.get("segment_seconds", 5)
        dst.attrs["segment_step_seconds"] = src.attrs.get("segment_step_seconds", 5)
        dst.attrs["projection"] = "Input EEG is already mapped to mdJPT 60 standard channels."

        for name in [
            "label",
            "split",
            "subject_index",
            "trial_index",
            "segment_index",
            "global_trial_index",
            "segment_start_sample",
            "segment_start_second",
        ]:
            copy_dataset(src, dst, name)

        meta = dst.create_group("meta")
        for name in [
            "mapped_channel_names",
            "subject_names",
            "subject_groups",
            "split_names",
        ]:
            src_name = f"meta/{name}"
            if src_name in src:
                meta.create_dataset(name, data=src[src_name][:], dtype=src[src_name].dtype)

        for start in range(0, n_samples, args.batch_size):
            stop = min(start + args.batch_size, n_samples)
            batch = torch.from_numpy(eeg_ds[start:stop].astype(np.float32)).unsqueeze(1).to(device)
            with torch.no_grad():
                pool, _, mllaout = model.forward(batch, 0, returnMLLAout=True)
                pool = pool.reshape(pool.shape[0], -1)
                channel_tokens = mllaout.mean(dim=2)
                patch_tokens = mllaout.mean(dim=1)

            if pool_ds is None:
                pool_dim = int(pool.shape[-1])
                n_channels = int(channel_tokens.shape[1])
                token_dim = int(channel_tokens.shape[2])
                n_patches = int(patch_tokens.shape[1])
                pool_ds, channel_ds, patch_ds = create_output_datasets(
                    dst,
                    n_samples=n_samples,
                    pool_dim=pool_dim,
                    n_channels=n_channels,
                    n_patches=n_patches,
                    token_dim=token_dim,
                )
                dst.attrs["pool_dim"] = pool_dim
                dst.attrs["n_channel_tokens"] = n_channels
                dst.attrs["n_patch_tokens"] = n_patches
                dst.attrs["token_dim"] = token_dim

            pool_ds[start:stop] = pool.detach().cpu().numpy().astype(np.float32)
            channel_ds[start:stop] = channel_tokens.detach().cpu().numpy().astype(np.float32)
            patch_ds[start:stop] = patch_tokens.detach().cpu().numpy().astype(np.float32)
            print(f"wrote embeddings {start}:{stop} / {n_samples}", flush=True)

    print(f"Saved mdJPT step-5 embeddings to: {args.output_h5}")


if __name__ == "__main__":
    main()
