import hydra
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("WORLD_SIZE", "1")
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src.model.valMLP import simpleNN3, MLPModel
import numpy as np
from src.data.dataset import PDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
import logging
import json
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def _artifact_suffix(cfg: DictConfig) -> str:
    artifact_tag = OmegaConf.select(cfg, 'val.artifact_tag')
    if artifact_tag is None:
        return ''
    artifact_tag = str(artifact_tag).strip()
    if artifact_tag == '':
        return ''
    return f'_{artifact_tag}'


def _compute_metrics(y_true, y_prob):
    y_pred = y_prob.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        y_prob_pos = y_prob[:, 1]
        auroc = float(roc_auc_score(y_true, y_prob_pos))
        auprc = float(average_precision_score(y_true, y_prob_pos))
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        auroc = float(roc_auc_score(y_true_bin, y_prob, multi_class='ovo', average='macro'))
        auprc = float(average_precision_score(y_true_bin, y_prob, average='macro'))
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'n_samples': int(y_true.shape[0]),
    }


def _build_trial_ids(n_subjects, n_samples_onesub):
    per_subject = []
    for trial_idx, n_samples in enumerate(n_samples_onesub):
        per_subject.extend([trial_idx] * int(n_samples))
    per_subject = np.asarray(per_subject, dtype=np.int64)
    n_trials = len(n_samples_onesub)
    return np.concatenate([per_subject + sub_idx * n_trials for sub_idx in range(n_subjects)], axis=0)


def _aggregate_trial_probabilities(y_true, y_prob, trial_ids):
    unique_trials = np.unique(trial_ids)
    trial_true, trial_prob = [], []
    for trial_id in unique_trials:
        mask = trial_ids == trial_id
        trial_true.append(int(y_true[mask][0]))
        trial_prob.append(y_prob[mask].mean(axis=0))
    return np.asarray(trial_true, dtype=np.int64), np.stack(trial_prob, axis=0)


def _predict_probs(model, loader, device):
    y_true, y_prob = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y_true.append(y_batch.numpy())
        y_prob.append(probs)
    return np.concatenate(y_true), np.concatenate(y_prob)

torch.set_float32_matmul_precision('high')  # optional performance tweak


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

@hydra.main(config_path="cfgs_multi", config_name="config_multi", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.val.mlp.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    explicit_split = _load_split_spec(cfg)

    # prepare cross-validation folds
    if explicit_split is None:
        val_subs_all = cfg.data_val.val_subs_all
        if cfg.val.n_fold == "loo":
            val_subs_all = [[i] for i in range(cfg.data_val.n_subs)]
        n_folds = len(val_subs_all)
    else:
        n_folds = 1

    # storage for metrics
    accs, precisions, recalls, f1s, aurocs, auprcs = [], [], [], [], [], []

    for fold in range(n_folds):
        print(f"=== Fold {fold} ===")
        # checkpoint callback
        artifact_suffix = _artifact_suffix(cfg)
        artifact_tag = artifact_suffix[1:] if artifact_suffix else ''
        cp_dir = os.path.join(cfg.log.mlp_cp_dir, cfg.log.run_name, artifact_tag) if artifact_tag else os.path.join(cfg.log.mlp_cp_dir, cfg.log.run_name)
        os.makedirs(cp_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor="mlp/val/acc", verbose=True, mode="max",
            dirpath=cp_dir,
            filename=f'{cfg.data_val.dataset_name}_mlp_f{fold}_wd={cfg.val.mlp.wd}_{{epoch}}',
            save_top_k=1,
        )

        # split subjects
        if explicit_split is None:
            val_subs = val_subs_all[fold]
            train_subs = list(set(range(cfg.data_val.n_subs)) - set(val_subs))
            if cfg.val.extractor.reverse:
                train_subs, val_subs = val_subs, train_subs
            test_subs = val_subs
            split_mode = "subject"
        elif explicit_split["mode"] == "subject":
            train_subs = explicit_split["train_subs"]
            val_subs = explicit_split["val_subs"]
            test_subs = explicit_split["test_subs"]
            print(f"Loaded explicit split from: {explicit_split['split_json']}")
            split_mode = "subject"
        else:
            train_indices = explicit_split["train_indices"]
            val_indices = explicit_split["val_indices"]
            test_indices = explicit_split["test_indices"]
            print(f"Loaded segment-random split from: {explicit_split['split_json']}")
            split_mode = "segment"
        if split_mode == "subject":
            print(f"Finetune subjects: {train_subs}")
            print(f"Validation subjects: {val_subs}")
            print(f"Test subjects:   {test_subs}")
        else:
            print(f"Train segments: {len(train_indices)}")
            print(f"Validation segments: {len(val_indices)}")
            print(f"Test segments: {len(test_indices)}")

        # load features
        save_dir = os.path.join(cfg.data_val.data_dir, 'ext_fea')
        save_path = os.path.join(
            save_dir,
            f"{cfg.log.run_name}_{f'{fold}' if cfg.val.extractor.normTrain else 'all'}_fea_"
            + (f"epoch={(cfg.val.extractor.ckpt_epoch-1):02d}.ckpt" if cfg.val.extractor.use_pretrain else "")
+ f"{cfg.val.extractor.fea_mode}{artifact_suffix}.npy"
        )
        print(f'Using feature path: {save_path}')
        data = np.load(save_path)
        data = np.nan_to_num(data)
        data = data.reshape(cfg.data_val.n_subs, -1, data.shape[-1])
        flat_data = data.reshape(-1, data.shape[-1])

        # labels
        onesub_label = np.load(os.path.join(save_dir, f'onesub_label{artifact_suffix}.npy'))
        n_samples_onesub = np.load(os.path.join(save_dir, f'n_samples_onesub{artifact_suffix}.npy'))
        labels = np.tile(onesub_label, cfg.data_val.n_subs).astype(np.int64)
        trial_metrics_available = split_mode == "subject"
        if split_mode == "subject":
            train_labels = np.tile(onesub_label, len(train_subs))
            val_labels = np.tile(onesub_label, len(val_subs))
            test_labels = np.tile(onesub_label, len(test_subs))
            val_trial_ids = _build_trial_ids(len(val_subs), n_samples_onesub)
            test_trial_ids = _build_trial_ids(len(test_subs), n_samples_onesub)
        else:
            val_trial_ids = None
            test_trial_ids = None

        # datasets & loaders
        if split_mode == "subject":
            trainset = PDataset(data[train_subs].reshape(-1, data.shape[-1]), train_labels)
            valset = PDataset(data[val_subs].reshape(-1, data.shape[-1]), val_labels)
            testset = PDataset(data[test_subs].reshape(-1, data.shape[-1]), test_labels)
        else:
            trainset = PDataset(flat_data[train_indices], labels[train_indices])
            valset = PDataset(flat_data[val_indices], labels[val_indices])
            testset = PDataset(flat_data[test_indices], labels[test_indices])
        trainLoader = DataLoader(trainset, batch_size=cfg.val.mlp.batch_size, shuffle=True)
        valLoader   = DataLoader(valset,   batch_size=cfg.val.mlp.batch_size, shuffle=False)
        testLoader  = DataLoader(testset,  batch_size=cfg.val.mlp.batch_size, shuffle=False)

        # model & trainer
        fea_dim = data.shape[-1]
        base_model = simpleNN3(fea_dim, cfg.val.mlp.hidden_dim, cfg.val.mlp.out_dim, 0.1)
        lightning_module = MLPModel(base_model, cfg.val.mlp)
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=cfg.val.mlp.max_epochs,
            min_epochs=cfg.val.mlp.min_epochs,
            accelerator='gpu', devices=1,
            limit_val_batches=1.0
        )

        # fit
        trainer.fit(lightning_module, trainLoader, valLoader)

        # load best checkpoint for metric computation
        best_ckpt = checkpoint_callback.best_model_path
        print(f"Loading best model from: {best_ckpt}")
        best_model = MLPModel.load_from_checkpoint(
            best_ckpt, model=base_model, cfg=cfg.val.mlp
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_model = best_model.to(device)
        best_model.eval()
        best_model.freeze()

        val_y_true, val_y_prob = _predict_probs(best_model.model, valLoader, device)
        test_y_true, test_y_prob = _predict_probs(best_model.model, testLoader, device)

        val_window_metrics = _compute_metrics(val_y_true, val_y_prob)
        test_window_metrics = _compute_metrics(test_y_true, test_y_prob)
        if trial_metrics_available:
            val_trial_true, val_trial_prob = _aggregate_trial_probabilities(val_y_true, val_y_prob, val_trial_ids)
            test_trial_true, test_trial_prob = _aggregate_trial_probabilities(test_y_true, test_y_prob, test_trial_ids)
            val_trial_metrics = _compute_metrics(val_trial_true, val_trial_prob)
            test_trial_metrics = _compute_metrics(test_trial_true, test_trial_prob)
        else:
            val_trial_metrics = None
            test_trial_metrics = None

        accs.append(test_window_metrics['acc'])
        precisions.append(test_window_metrics['precision'])
        recalls.append(test_window_metrics['recall'])
        f1s.append(test_window_metrics['f1'])
        aurocs.append(test_window_metrics['auroc'])
        auprcs.append(test_window_metrics['auprc'])

        result_dir = os.path.join('mlp_results', cfg.log.run_name, artifact_tag) if artifact_tag else os.path.join('mlp_results', cfg.log.run_name)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f'{cfg.data_val.dataset_name}_mlp_metrics.json')
        result_payload = {
            'val': {'window': val_window_metrics, 'trial': val_trial_metrics},
            'test': {'window': test_window_metrics, 'trial': test_trial_metrics},
            'best_checkpoint': best_ckpt,
            'feature_path': save_path,
            'artifact_tag': artifact_tag,
            'segment_seconds': cfg.data_val.timeLen2,
            'segment_step_seconds': cfg.data_val.timeStep2,
            'split_json': explicit_split['split_json'] if explicit_split is not None else None,
            'split_mode': split_mode,
            'split_kind': explicit_split['split_kind'] if explicit_split is not None else 'subject_cv',
            'trial_metrics_available': trial_metrics_available,
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False)

        print(f"Fold {fold} results:")
        print(f"  Val window acc   = {val_window_metrics['acc']:.2f}")
        print(f"  Test window acc  = {test_window_metrics['acc']:.2f}")
        if trial_metrics_available:
            print(f"  Val trial acc    = {val_trial_metrics['acc']:.2f}")
            print(f"  Test trial acc   = {test_trial_metrics['acc']:.2f}")
            print(f"  Test trial AUROC = {test_trial_metrics['auroc']:.2f}")
            print(f"  Test trial AUPRC = {test_trial_metrics['auprc']:.2f}")
        print(f"Saved metrics to: {result_path}")

    # summary
    print("\n=== Summary across folds ===")
    metrics = {
        'Acc':      accs,
        'Precision':precisions,
        'Recall':   recalls,
        'F1':       f1s,
        'AUROC':    aurocs,
        'AUPRC':    auprcs
    }
    latex = ''
    for name, vals in metrics.items():
        mean = np.mean(vals)
        std  = np.std(vals)
        print(f"{name}: {mean*100:.2f} ± {std*100:.2f}")
        latex = latex + f"{mean*100:.2f} ± {std*100:.2f} & "
    print(latex)

if __name__ == '__main__':
    train_mlp()
