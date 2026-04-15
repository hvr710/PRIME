import json
import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

from src.data.io_utils import load_finetune_EEG_data
from src.model.MultiModel_PL import MultiModel_PL
from src.model.valMLP import accuracy, simpleNN3

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("WORLD_SIZE", "1")

torch.set_float32_matmul_precision("high")


class IndexedWindowDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        sample_indices: np.ndarray,
        trial_ids: np.ndarray,
    ):
        self.data = data
        self.labels = labels
        self.sample_indices = torch.as_tensor(sample_indices, dtype=torch.long)
        self.trial_ids = torch.as_tensor(trial_ids, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.sample_indices.numel())

    def __getitem__(self, idx):
        sample_idx = int(self.sample_indices[idx])
        return (
            self.data[sample_idx].unsqueeze(0),
            self.labels[sample_idx],
            self.trial_ids[sample_idx],
        )


def _set_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def _compute_metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = y_prob.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        y_prob_pos = y_prob[:, 1]
        auroc = float(roc_auc_score(y_true, y_prob_pos))
        auprc = float(average_precision_score(y_true, y_prob_pos))
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        auroc = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovo", average="macro"))
        auprc = float(average_precision_score(y_true_bin, y_prob, average="macro"))

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "n_samples": int(y_true.shape[0]),
    }


def _aggregate_trial_probabilities(y_true: np.ndarray, y_prob: np.ndarray, trial_ids: np.ndarray):
    unique_trials = np.unique(trial_ids)
    trial_true = []
    trial_prob = []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        trial_true.append(int(y_true[mask][0]))
        trial_prob.append(y_prob[mask].mean(axis=0))
    return np.asarray(trial_true, dtype=np.int64), np.stack(trial_prob, axis=0)


class FullFineTuneModel(pl.LightningModule):
    def __init__(self, backbone, classifier, cfg: DictConfig, data_channels, aggregate_eval_by_trial: bool = True):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.cfg = cfg
        self.data_channels = list(data_channels)
        self.aggregate_eval_by_trial = aggregate_eval_by_trial
        self.criterion = torch.nn.CrossEntropyLoss()
        self._adapter_trainable = False
        self._mlla_trainable = False
        self._stage_signature = None
        self._val_outputs = []

    def forward(self, x):
        x = self.backbone.channel_project(x, self.data_channels)
        backbone_out = self.backbone(x, dataset=0)
        features = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out
        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)
        return self.classifier(features)

    def configure_optimizers(self):
        param_groups = [{"params": self.classifier.parameters(), "lr": self.cfg.full_ft.head_lr}]
        if hasattr(self.backbone, "cnn_encoder"):
            param_groups.append(
                {"params": self.backbone.cnn_encoder.parameters(), "lr": self.cfg.full_ft.adapter_lr}
            )
        if hasattr(self.backbone, "uni_mlp"):
            param_groups.append(
                {"params": self.backbone.uni_mlp.parameters(), "lr": self.cfg.full_ft.adapter_lr}
            )
        if hasattr(self.backbone, "MLLA"):
            param_groups.append({"params": self.backbone.MLLA.parameters(), "lr": self.cfg.full_ft.mlla_lr})
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=self.cfg.full_ft.wd,
        )
        return {"optimizer": optimizer}

    def _apply_finetune_stage(self):
        adapter_trainable = self.current_epoch >= int(self.cfg.full_ft.head_warmup_epochs)
        unfreeze_mlla_epoch = int(self.cfg.full_ft.unfreeze_mlla_epoch)
        mlla_trainable = unfreeze_mlla_epoch >= 0 and self.current_epoch >= unfreeze_mlla_epoch

        _set_requires_grad(self.classifier, True)
        if hasattr(self.backbone, "cnn_encoder"):
            _set_requires_grad(self.backbone.cnn_encoder, adapter_trainable)
        if hasattr(self.backbone, "uni_mlp"):
            _set_requires_grad(self.backbone.uni_mlp, adapter_trainable)
        if hasattr(self.backbone, "MLLA"):
            _set_requires_grad(self.backbone.MLLA, mlla_trainable)

        self._adapter_trainable = adapter_trainable
        self._mlla_trainable = mlla_trainable
        stage_signature = (adapter_trainable, mlla_trainable)
        if stage_signature != self._stage_signature:
            print(
                "Fine-tune stage: "
                f"classifier=train, adapters={'train' if adapter_trainable else 'frozen'}, "
                f"MLLA={'train' if mlla_trainable else 'frozen'}"
            )
            self._stage_signature = stage_signature

    def _sync_module_modes(self):
        self.classifier.train(True)
        if hasattr(self.backbone, "cnn_encoder"):
            self.backbone.cnn_encoder.train(self._adapter_trainable)
        if hasattr(self.backbone, "uni_mlp"):
            self.backbone.uni_mlp.train(self._adapter_trainable)
        if hasattr(self.backbone, "MLLA"):
            self.backbone.MLLA.train(self._mlla_trainable)

    def on_train_start(self):
        self._apply_finetune_stage()

    def on_train_epoch_start(self):
        self._apply_finetune_stage()

    def training_step(self, batch, batch_idx):
        self._sync_module_modes()
        data, labels, _ = batch
        logits = self(data)
        loss = self.criterion(logits, labels)
        top1 = accuracy(logits, labels, topk=(1,))
        self.log_dict(
            {"ft/train/loss": loss, "ft/train/acc": top1[0]},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self._val_outputs = []

    def validation_step(self, batch, batch_idx):
        data, labels, trial_ids = batch
        logits = self(data)
        loss = self.criterion(logits, labels)
        top1 = accuracy(logits, labels, topk=(1,))
        self.log_dict(
            {"ft/val/loss": loss, "ft/val/acc": top1[0]},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self._val_outputs.append(
            {
                "labels": labels.detach().cpu(),
                "probs": torch.softmax(logits, dim=1).detach().cpu(),
                "trial_ids": trial_ids.detach().cpu(),
            }
        )
        return loss

    def on_validation_epoch_end(self):
        if not self._val_outputs or not self.aggregate_eval_by_trial:
            return
        y_true = torch.cat([item["labels"] for item in self._val_outputs], dim=0).numpy()
        y_prob = torch.cat([item["probs"] for item in self._val_outputs], dim=0).numpy()
        trial_ids = torch.cat([item["trial_ids"] for item in self._val_outputs], dim=0).numpy()
        trial_true, trial_prob = _aggregate_trial_probabilities(y_true, y_prob, trial_ids)
        trial_metrics = _compute_metrics_from_probs(trial_true, trial_prob)
        self.log_dict(
            {
                "ft/val_trial/acc": trial_metrics["acc"] * 100.0,
                "ft/val_trial/auroc": trial_metrics["auroc"] * 100.0,
                "ft/val_trial/auprc": trial_metrics["auprc"] * 100.0,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


def _load_split_spec(cfg: DictConfig):
    segment_split_json = OmegaConf.select(cfg, "val.segment_split_json")
    if segment_split_json and os.path.exists(segment_split_json):
        with open(segment_split_json, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        return {
            "mode": "segment",
            "train_indices": np.asarray(split_data["train_segment_indices"], dtype=np.int64),
            "val_indices": np.asarray(split_data["val_segment_indices"], dtype=np.int64),
            "test_indices": np.asarray(split_data["test_segment_indices"], dtype=np.int64),
            "split_json": segment_split_json,
            "split_name": split_data.get("split_name", os.path.splitext(os.path.basename(segment_split_json))[0]),
            "split_kind": split_data.get("split_kind", "segment_random"),
        }

    split_json = OmegaConf.select(cfg, "data_val.split_json")
    if not split_json or not os.path.exists(split_json):
        return None
    with open(split_json, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    return {
        "mode": "subject",
        "train_subs": split_data["train_indices"],
        "val_subs": split_data["val_indices"],
        "test_subs": split_data["test_indices"],
        "split_json": split_json,
        "split_name": os.path.splitext(os.path.basename(split_json))[0],
        "split_kind": "subject",
    }


def _expand_subject_indices(subject_indices, onesub_len):
    return np.concatenate(
        [np.arange(sub_idx * onesub_len, (sub_idx + 1) * onesub_len) for sub_idx in subject_indices]
    )


def _build_trial_ids(n_subs: int, n_samples_onesub: np.ndarray):
    per_subject_trial_ids = []
    for trial_idx, n_samples in enumerate(n_samples_onesub):
        per_subject_trial_ids.extend([trial_idx] * int(n_samples))

    per_subject_trial_ids = np.asarray(per_subject_trial_ids, dtype=np.int64)
    n_trials_per_subject = len(n_samples_onesub)
    all_trial_ids = []
    for sub_idx in range(n_subs):
        all_trial_ids.append(per_subject_trial_ids + sub_idx * n_trials_per_subject)
    return np.concatenate(all_trial_ids, axis=0)


def _build_backbone_cfg(cfg: DictConfig):
    runtime_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    runtime_cfg.data_0 = OmegaConf.create(OmegaConf.to_container(cfg.data_val, resolve=True))
    runtime_cfg.data_1 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_2 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_3 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_4 = OmegaConf.create({"dataset_name": "None"})
    runtime_cfg.data_cfg_list = [runtime_cfg.data_0]
    return runtime_cfg


def _infer_feature_dim(backbone, dataset, data_channels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.eval()
    sample_batch = torch.stack([dataset[i][0] for i in range(min(2, len(dataset)))], dim=0).to(device)
    with torch.no_grad():
        sample_batch = backbone.channel_project(sample_batch, data_channels)
        backbone_out = backbone(sample_batch, dataset=0)
        features = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out
        if features.dim() > 2:
            features = features.reshape(features.shape[0], -1)
    feature_dim = int(features.shape[-1])
    backbone.cpu()
    torch.cuda.empty_cache()
    return feature_dim


def _collect_predictions(model, loader):
    model.eval()
    y_true, y_prob, trial_ids = [], [], []
    for x_batch, y_batch, trial_batch in loader:
        x_batch = x_batch.to(model.device)
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y_true.append(y_batch.numpy())
        y_prob.append(probs)
        trial_ids.append(trial_batch.numpy())

    return np.concatenate(y_true), np.concatenate(y_prob), np.concatenate(trial_ids)


def _evaluate(model, loader, aggregate_by_trial: bool):
    y_true, y_prob, trial_ids = _collect_predictions(model, loader)
    metrics = {"window": _compute_metrics_from_probs(y_true, y_prob)}
    if aggregate_by_trial:
        trial_true, trial_prob = _aggregate_trial_probabilities(y_true, y_prob, trial_ids)
        metrics["trial"] = _compute_metrics_from_probs(trial_true, trial_prob)
    else:
        metrics["trial"] = None
    return metrics


def _build_experiment_tag(cfg: DictConfig, split: dict) -> str:
    if int(cfg.full_ft.unfreeze_mlla_epoch) >= 0:
        mlla_tag = f"mlla{int(cfg.full_ft.unfreeze_mlla_epoch)}"
    else:
        mlla_tag = "mllafrozen"
    split_tag = split.get("split_name", split.get("split_kind", split.get("mode", "split")))
    split_tag = str(split_tag).replace("/", "_")
    return (
        f"len{cfg.data_val.timeLen2}_step{cfg.data_val.timeStep2}"
        f"_warm{int(cfg.full_ft.head_warmup_epochs)}_{mlla_tag}"
        f"_{split_tag}"
    )


@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def train_full_finetune(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.full_ft.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    split = _load_split_spec(cfg)
    if split is None:
        raise FileNotFoundError(
            "Full fine-tune requires either data_val.split_json or val.segment_split_json to be configured."
        )

    print(f"Loaded explicit split from: {split['split_json']}")
    print(f"Segment length: {cfg.data_val.timeLen2}s, step: {cfg.data_val.timeStep2}s")
    print(f"Split mode: {split['mode']}")

    data_dir = os.path.join(cfg.data_val.data_dir, "processed_data")
    print("data loading...")
    data, onesub_labels, n_samples_onesub, _ = load_finetune_EEG_data(data_dir, cfg.data_val)
    print("data loaded")
    print(f"data shape: {data.shape}")

    data_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
    label_tensor = torch.from_numpy(np.tile(onesub_labels, cfg.data_val.n_subs).astype(np.int64))
    trial_ids = _build_trial_ids(cfg.data_val.n_subs, n_samples_onesub)
    onesub_len = len(onesub_labels)

    if split["mode"] == "subject":
        train_indices = _expand_subject_indices(split["train_subs"], onesub_len)
        val_indices = _expand_subject_indices(split["val_subs"], onesub_len)
        test_indices = _expand_subject_indices(split["test_subs"], onesub_len)
        aggregate_eval_by_trial = True
    else:
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        test_indices = split["test_indices"]
        aggregate_eval_by_trial = False

    trainset = IndexedWindowDataset(data_tensor, label_tensor, train_indices, trial_ids)
    valset = IndexedWindowDataset(data_tensor, label_tensor, val_indices, trial_ids)
    testset = IndexedWindowDataset(data_tensor, label_tensor, test_indices, trial_ids)

    train_loader = DataLoader(
        trainset,
        batch_size=cfg.full_ft.batch_size,
        shuffle=True,
        num_workers=cfg.full_ft.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=cfg.full_ft.batch_size,
        shuffle=False,
        num_workers=cfg.full_ft.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=cfg.full_ft.batch_size,
        shuffle=False,
        num_workers=cfg.full_ft.num_workers,
        pin_memory=True,
    )

    backbone_cfg = _build_backbone_cfg(cfg)
    ckpt_path = os.path.join(
        "log",
        cfg.log.run_name,
        "ckpt",
        f"epoch={(cfg.val.extractor.ckpt_epoch - 1):02d}.ckpt",
    )
    print(f"Loading pretrained backbone from: {ckpt_path}")
    backbone = MultiModel_PL.load_from_checkpoint(ckpt_path, cfg=backbone_cfg, strict=False)
    backbone.save_fea = False
    if hasattr(backbone, "cnn_encoder"):
        backbone.cnn_encoder.set_saveFea(False)

    feature_dim = _infer_feature_dim(backbone, trainset, cfg.data_val.channels)
    print(f"Backbone feature dim: {feature_dim}")

    classifier = simpleNN3(
        feature_dim,
        cfg.full_ft.hidden_dim,
        cfg.data_val.n_class,
        dropout=cfg.full_ft.dropout,
    )
    lightning_module = FullFineTuneModel(
        backbone,
        classifier,
        cfg,
        cfg.data_val.channels,
        aggregate_eval_by_trial=aggregate_eval_by_trial,
    )

    exp_tag = _build_experiment_tag(cfg, split)
    cp_dir = os.path.join("full_ft_cp", cfg.log.run_name, exp_tag)
    os.makedirs(cp_dir, exist_ok=True)
    result_dir = os.path.join("full_ft_results", cfg.log.run_name, exp_tag)
    os.makedirs(result_dir, exist_ok=True)

    monitor_metric = "ft/val_trial/acc" if aggregate_eval_by_trial else "ft/val/acc"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        dirpath=cp_dir,
        filename=f"{cfg.data_val.dataset_name}_fullft_epoch={{epoch}}",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="max",
        patience=cfg.full_ft.patience,
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        max_epochs=cfg.full_ft.max_epochs,
        min_epochs=cfg.full_ft.min_epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(lightning_module, train_loader, val_loader)

    best_ckpt = checkpoint_callback.best_model_path
    print(f"Loading best full fine-tune model from: {best_ckpt}")

    backbone_eval = MultiModel_PL.load_from_checkpoint(ckpt_path, cfg=backbone_cfg, strict=False)
    backbone_eval.save_fea = False
    if hasattr(backbone_eval, "cnn_encoder"):
        backbone_eval.cnn_encoder.set_saveFea(False)
    classifier_eval = simpleNN3(
        feature_dim,
        cfg.full_ft.hidden_dim,
        cfg.data_val.n_class,
        dropout=cfg.full_ft.dropout,
    )
    best_model = FullFineTuneModel.load_from_checkpoint(
        best_ckpt,
        backbone=backbone_eval,
        classifier=classifier_eval,
        cfg=cfg,
        data_channels=cfg.data_val.channels,
        aggregate_eval_by_trial=aggregate_eval_by_trial,
    )
    best_model.eval()
    best_model.freeze()

    val_metrics = _evaluate(best_model, val_loader, aggregate_eval_by_trial)
    test_metrics = _evaluate(best_model, test_loader, aggregate_eval_by_trial)
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics.update(
        {
            "split_json": split["split_json"],
            "split_mode": split["mode"],
            "split_kind": split["split_kind"],
            "trial_metrics_available": aggregate_eval_by_trial,
            "segment_seconds": cfg.data_val.timeLen2,
            "segment_step_seconds": cfg.data_val.timeStep2,
            "feature_dim": feature_dim,
            "best_checkpoint": best_ckpt,
            "experiment_tag": exp_tag,
            "full_ft_cfg": {
                "head_lr": cfg.full_ft.head_lr,
                "adapter_lr": cfg.full_ft.adapter_lr,
                "mlla_lr": cfg.full_ft.mlla_lr,
                "wd": cfg.full_ft.wd,
                "dropout": cfg.full_ft.dropout,
                "head_warmup_epochs": int(cfg.full_ft.head_warmup_epochs),
                "unfreeze_mlla_epoch": int(cfg.full_ft.unfreeze_mlla_epoch),
            },
        }
    )

    result_path = os.path.join(result_dir, f"{cfg.data_val.dataset_name}_fullft_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Full fine-tune validation results:")
    print(f"  Window Acc = {metrics['val']['window']['acc'] * 100:.2f}")
    if aggregate_eval_by_trial:
        print(f"  Trial Acc  = {metrics['val']['trial']['acc'] * 100:.2f}")
        print(f"  Trial AUROC = {metrics['val']['trial']['auroc'] * 100:.2f}")
        print(f"  Trial AUPRC = {metrics['val']['trial']['auprc'] * 100:.2f}")
    print("Full fine-tune test results:")
    print(f"  Window Acc = {metrics['test']['window']['acc'] * 100:.2f}")
    if aggregate_eval_by_trial:
        print(f"  Trial Acc  = {metrics['test']['trial']['acc'] * 100:.2f}")
        print(f"  Trial AUROC = {metrics['test']['trial']['auroc'] * 100:.2f}")
        print(f"  Trial AUPRC = {metrics['test']['trial']['auprc'] * 100:.2f}")
    print(f"Saved metrics to: {result_path}")


if __name__ == "__main__":
    train_full_finetune()
