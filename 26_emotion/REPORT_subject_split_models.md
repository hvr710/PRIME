# COMP4 Subject Split 模型与结果报告

## 0. 报告范围

本文件只汇报 `subject split` 下的模型与结果，不包含 `segment split` 内容。

本轮固定设置：

- 6 个随机种子：`42, 3407, 2025, 666, 777, 888`
- frozen-only（不做 mdJPT 全量微调）
- 统一预处理协议：`pre-split per-subject z-score + trial-wise causal smoothing`
- 主评估协议：`subject split`（`48/6/6` subjects）

### 0.1 Window-level 与 Trial-level 指标说明

本文里有两种评估粒度：

- **Window-level**：把每个 `5s EEG segment` 当成一个独立样本来评估。当前数据共有 `4800` 个 window，每个 window 的输入是 `(60 channels, 625 samples)`，模型会对每个 5s segment 输出一次情绪二分类概率。Window-level 指标反映模型对短时 EEG 片段的直接分类能力。

- **Trial-level**：同一个原始 trial 会被切成多个 5s segment。Trial-level 评估时，先把同一个 `global_trial_index` 下所有 segment 的预测概率做平均，再得到这个 trial 的最终预测。Trial-level 指标通常更平滑、更接近“整段实验试次”的判断能力，但样本数少于 window-level。

在本报告的 `subject split` 中，train/val/test 是按被试划分的，因此 trial-level 聚合不会跨 split。这里的 MoE router 指标不是情绪分类的 window/trial 指标，而是单独衡量 router 能否区分被试类型 `HC/DEP`。

## 1. 输入与统一预处理

### 1.1 输入表征

| Representation | Shape | Description |
|---|---:|---|
| handcrafted features | `(4800, 1520)` | 频段能量、相对能量、微分熵、Hjorth、谱熵、PFD、不对称等手工特征 |
| mdJPT pooled embedding | `(4800, 1024)` | frozen mdJPT 全局 pooled 表征 |
| mdJPT channel tokens | `(4800, 60, 32)` | token 按 temporal patch 平均后的通道 token |
| mdJPT patch tokens | `(4800, 99, 32)` | token 按 channel 平均后的时间 patch token |

### 1.2 预处理流程（固定）

```text
all segments
  -> per-subject z-score
  -> trial-wise causal EWMA smoothing (alpha=0.65)
  -> subject split train/val/test
```

说明：每个 subject 使用其自身全量 segment 统计量做标准化，再做 trial 内顺序平滑。

## 2. 模型架构说明（重点）

### 2.1 Handcrafted Baselines

1. Logistic Regression
- 输入：`hand_1520`
- 作用：线性可分性下界

2. Linear SVM
- 输入：`hand_1520`
- 作用：margin-based 线性对照

3. XGBoost
- 输入：`hand_1520`
- 作用：手工特征在非线性树模型下的强基线

### 2.2 mdJPT Frozen MLP

- 输入：`pool_1024`
- 结构：`pool_1024 -> MLP -> emotion logits`
- 作用：验证“只靠 mdJPT pooled 表征”能达到的性能

### 2.3 CrossAttn Single Head（融合对照）

```text
hand_1520 --Linear(1520->32)--> Query
mdJPT channel/patch tokens -----> Key/Value
Cross-Attention -----------------> fused feature
fused feature -------------------> MLP classifier
```

- 作用：验证手工特征与 mdJPT token 的互补融合
- 角色：MoE 主线的单头对照基线

### 2.4 MoE Mainline（主线架构）

```text
CrossAttn fused feature F
    |-- Expert_HC  -> logits_hc
    |-- Expert_DEP -> logits_dep

pool_1024 -> Router -> [p_hc, p_dep]

emotion_logits = p_hc * logits_hc + p_dep * logits_dep
```

MoE 变体：

| Variant | 设计点 |
|---|---|
| `cross_attn_moe_unsup` | router 无 HC/DEP 监督 |
| `cross_attn_moe_sup_l03` | router 加监督，`lambda=0.3` |
| `cross_attn_moe_sup_l05` | router 加监督，`lambda=0.5` |
| `cross_attn_moe_token_sup_l03` | 监督版 + subject-type token，`lambda=0.3` |
| `cross_attn_moe_token_sup_l05` | 监督版 + subject-type token，`lambda=0.5` |

MoE 本轮超参筛选结论：`base_lr = 5e-4`。

## 3. Subject Split 主结果（6-seed mean ± std）

| Method | Test Acc | Test F1 | Test AUROC | Test Trial Acc | Test Trial AUROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71.70 ± 4.67 | 71.68 ± 4.67 | 78.73 ± 5.36 | 75.69 ± 6.44 | 85.50 ± 6.13 |
| Linear SVM | 71.46 ± 4.69 | 71.44 ± 4.70 | 77.17 ± 5.57 | 78.12 ± 7.19 | 84.72 ± 6.42 |
| XGBoost | 75.80 ± 6.73 | 75.79 ± 6.74 | **83.74 ± 7.18** | 81.60 ± 7.74 | **89.09 ± 7.26** |
| mdJPT Frozen MLP | 59.65 ± 5.31 | 59.59 ± 5.30 | 63.50 ± 7.32 | 60.76 ± 6.41 | 65.97 ± 9.15 |
| CrossAttn Single Head | **75.87 ± 5.66** | **75.86 ± 5.66** | 83.04 ± 6.79 | **81.94 ± 6.44** | 88.37 ± 7.30 |
| MoE Unsup | 74.65 ± 5.82 | 74.62 ± 5.83 | 82.60 ± 6.96 | 80.56 ± 7.38 | 87.38 ± 7.14 |
| MoE Sup `lambda=0.3` | 75.24 ± 5.48 | 75.21 ± 5.48 | 82.86 ± 6.84 | 79.86 ± 8.04 | 87.93 ± 6.97 |
| MoE Sup `lambda=0.5` | 75.07 ± 5.64 | 75.04 ± 5.64 | 82.83 ± 6.87 | 80.56 ± 7.58 | 87.79 ± 7.05 |
| Token MoE Sup `lambda=0.3` | 75.24 ± 5.51 | 75.20 ± 5.52 | 82.86 ± 6.84 | 79.86 ± 8.04 | 87.93 ± 6.97 |
| Token MoE Sup `lambda=0.5` | 75.03 ± 5.60 | 75.00 ± 5.60 | 82.83 ± 6.87 | 80.56 ± 7.58 | 87.79 ± 7.05 |

## 4. 结果比较（讨论重点）

### 4.1 按指标看最优方法

| 指标 | 最优方法 | 数值 |
|---|---|---:|
| Window Acc / F1 | CrossAttn Single Head | **75.87 / 75.86** |
| Window AUROC | XGBoost | **83.74** |
| Trial Acc | CrossAttn Single Head | **81.94** |
| Trial AUROC | XGBoost | **89.09** |

### 4.2 关键差值（便于老师快速看）

| 比较项 | Acc 差值 | AUROC 差值 | 解释 |
|---|---:|---:|---|
| CrossAttn vs mdJPT Frozen MLP | +16.22 | +19.54 | 仅 pooled embedding 明显不足，fusion 必要 |
| CrossAttn vs Logistic | +4.17 | +4.31 | 融合优于纯线性手工特征 |
| CrossAttn vs Linear SVM | +4.41 | +5.87 | 融合优于 margin 线性基线 |
| CrossAttn vs XGBoost | +0.07 | -0.70 | 两者都强；XGBoost 在 AUROC 仍有优势 |
| MoE Sup(l03) vs CrossAttn | -0.63 | -0.18 | MoE 接近但均值未超越单头 |
| MoE Unsup vs CrossAttn | -1.22 | -0.44 | 仅靠无监督路由收益更弱 |

### 4.3 对 MoE 主线的解释

1. MoE 架构合理：它显式建模 HC/DEP subject type 差异，是一个可解释、可扩展的主线框架。
2. 当前结论务实：6-seed 平均上，MoE 与 single-head 非常接近，但尚未形成稳定优势。
3. 报告口径建议：将 MoE 定位为主线结构探索，single-head 与 XGBoost 作为强对照，强调“机制有效但仍需进一步增强路由/专家分工”。

## 5. MoE 区分被试结果（Subject Router 诊断）

下面给出 MoE 在 subject split 下区分 HC/DEP 被试类型的 router 指标（6-seed mean ± std）。

| Variant | Router Group Acc | Router Group AUROC |
|---|---:|---:|
| MoE Unsup | 48.12 ± 3.72 | 49.60 ± 2.58 |
| MoE Sup `lambda=0.3` | 56.77 ± 4.61 | 56.95 ± 5.34 |
| MoE Sup `lambda=0.5` | 57.29 ± 4.59 | 57.56 ± 5.45 |
| Token MoE Sup `lambda=0.3` | 56.77 ± 4.61 | 56.94 ± 5.35 |
| Token MoE Sup `lambda=0.5` | 57.33 ± 4.58 | 57.56 ± 5.45 |

解读要点：

1. 无监督 router 基本接近随机（约 50%），说明仅靠 emotion loss 难以自动学出稳定的 HC/DEP 路由。
2. 加监督后 router 指标有提升，但提升幅度有限，当前仍是“弱可分”。
3. token 版本与普通 supervised MoE 几乎重合，说明 subject-type token 在本轮不是主要增益来源。
4. 这也解释了为什么 MoE 在 emotion 主指标上接近 single-head，但没有形成稳定超越。

## 6. Subject Split 单 Seed 最佳值（上限参考）

| Method | Best Seed | Test Acc | Test F1 | Test AUROC | Test Trial AUROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 666 | 79.17 | 79.17 | 86.94 | 95.31 |
| Linear SVM | 666 | 78.12 | 78.12 | 83.78 | 93.06 |
| XGBoost | 666 | **82.92** | **82.92** | **91.81** | 96.35 |
| mdJPT Frozen MLP | 2025 | 66.04 | 65.96 | 72.60 | 77.08 |
| CrossAttn Single Head | 666 | 82.29 | 82.27 | 91.00 | **97.22** |
| MoE Unsup | 666 | 80.83 | 80.79 | 90.23 | 95.31 |
| MoE Sup `lambda=0.3` | 666 | 81.25 | 81.19 | 90.25 | 95.31 |
| MoE Sup `lambda=0.5` | 666 | 81.25 | 81.19 | 90.24 | 95.31 |
| Token MoE Sup `lambda=0.3` | 666 | 81.25 | 81.19 | 90.25 | 95.31 |
| Token MoE Sup `lambda=0.5` | 666 | 81.04 | 80.98 | 90.24 | 95.31 |

## 7. 可直接用于汇报的结论

1. 在固定的 pre-split per-subject z-score + trial-wise smoothing 协议下，subject split 主结果表明：手工特征与 mdJPT token 融合是有效路线。
2. CrossAttn single-head 取得最高平均 Acc（75.87），XGBoost 在 AUROC 维度仍是非常强的对照。
3. MoE 作为主线架构具备明确建模动机和可解释性，但当前 6-seed 平均尚未稳定超过 single-head；可作为主线框架中的核心机制探索与后续优化方向。
4. 汇报建议：主表使用 subject split，并重点讨论“CrossAttn 强基线 + MoE 主线机制 + 当前收益边界”。

## 8. Train / Val / Test 过拟合诊断

本节只整理神经模型：`mdJPT Frozen MLP`、`CrossAttn Single Head` 以及全部 MoE 变体。结果仍然是 `subject split` 的 6-seed mean ± std。

计算说明：

- train 指标是加载每个 seed 的 best-val checkpoint 后，在 train split 上重新推理得到。
- val / test 指标沿用原实验 `all_results.csv` 中记录的 best checkpoint 结果。
- 所有 checkpoint 都是按 `val/window/acc` early stop 并保存最佳权重。
- MoE 推理时使用原实验中基于验证集拟合得到的 router temperature。
- 详细逐 seed 结果已保存到 `comparison/subject_train_val_test_models.csv`。
- 汇总结果已保存到 `comparison/subject_train_val_test_models_summary.csv`。

### 8.1 Window-level 结果

| Method | Train Acc | Val Acc | Test Acc | Train-Test Acc Gap | Train F1 | Val F1 | Test F1 | Train AUROC | Val AUROC | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mdJPT Frozen MLP | 73.89 ± 0.70 | 64.55 ± 3.61 | 59.65 ± 5.31 | +14.24 | 73.89 ± 0.70 | 64.53 ± 3.62 | 59.59 ± 5.30 | 82.68 ± 0.63 | 69.75 ± 5.49 | 63.50 ± 7.32 |
| CrossAttn Single Head | 87.93 ± 4.15 | 74.72 ± 2.36 | 75.87 ± 5.66 | +12.06 | 87.92 ± 4.16 | 74.69 ± 2.37 | 75.86 ± 5.66 | 95.07 ± 2.68 | 80.93 ± 2.82 | 83.04 ± 6.79 |
| MoE Unsup | 86.48 ± 4.36 | 74.97 ± 2.29 | 74.65 ± 5.82 | +11.83 | 86.48 ± 4.36 | 74.95 ± 2.30 | 74.62 ± 5.83 | 94.12 ± 3.09 | 80.79 ± 2.65 | 82.60 ± 6.96 |
| MoE Sup `lambda=0.3` | 86.67 ± 5.01 | 75.45 ± 2.10 | 75.24 ± 5.48 | +11.42 | 86.67 ± 5.01 | 75.43 ± 2.11 | 75.21 ± 5.48 | 94.00 ± 3.22 | 81.55 ± 2.43 | 82.86 ± 6.84 |
| MoE Sup `lambda=0.5` | 86.59 ± 4.90 | 75.52 ± 2.04 | 75.07 ± 5.64 | +11.52 | 86.59 ± 4.90 | 75.51 ± 2.06 | 75.04 ± 5.64 | 93.98 ± 3.20 | 81.61 ± 2.43 | 82.83 ± 6.87 |
| Token MoE Sup `lambda=0.3` | 86.66 ± 5.00 | 75.45 ± 2.10 | 75.24 ± 5.51 | +11.41 | 86.66 ± 5.01 | 75.43 ± 2.11 | 75.20 ± 5.52 | 94.00 ± 3.22 | 81.55 ± 2.44 | 82.86 ± 6.84 |
| Token MoE Sup `lambda=0.5` | 86.58 ± 4.90 | 75.52 ± 2.04 | 75.03 ± 5.60 | +11.55 | 86.58 ± 4.90 | 75.51 ± 2.06 | 75.00 ± 5.60 | 93.98 ± 3.20 | 81.61 ± 2.43 | 82.83 ± 6.87 |

### 8.2 Trial-level 结果

| Method | Train Trial Acc | Val Trial Acc | Test Trial Acc | Train-Test Trial Acc Gap | Train Trial AUROC | Val Trial AUROC | Test Trial AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| mdJPT Frozen MLP | 79.12 ± 1.32 | 68.75 ± 3.61 | 60.76 ± 6.41 | +18.36 | 87.96 ± 0.68 | 74.02 ± 6.82 | 65.97 ± 9.15 |
| CrossAttn Single Head | 93.84 ± 3.90 | 77.78 ± 3.54 | 81.94 ± 6.44 | +11.89 | 98.25 ± 1.38 | 86.20 ± 3.45 | 88.37 ± 7.30 |
| MoE Unsup | 92.58 ± 3.97 | 78.13 ± 1.59 | 80.56 ± 7.38 | +12.02 | 97.50 ± 1.88 | 86.23 ± 3.99 | 87.38 ± 7.14 |
| MoE Sup `lambda=0.3` | 92.62 ± 4.18 | 78.82 ± 1.87 | 79.86 ± 8.04 | +12.76 | 97.38 ± 1.86 | 87.07 ± 3.50 | 87.93 ± 6.97 |
| MoE Sup `lambda=0.5` | 92.62 ± 4.18 | 78.82 ± 1.87 | 80.56 ± 7.58 | +12.07 | 97.37 ± 1.87 | 87.18 ± 3.42 | 87.79 ± 7.05 |
| Token MoE Sup `lambda=0.3` | 92.62 ± 4.18 | 78.82 ± 1.87 | 79.86 ± 8.04 | +12.76 | 97.38 ± 1.86 | 87.07 ± 3.50 | 87.93 ± 6.97 |
| Token MoE Sup `lambda=0.5` | 92.62 ± 4.18 | 78.82 ± 1.87 | 80.56 ± 7.58 | +12.07 | 97.37 ± 1.87 | 87.18 ± 3.42 | 87.79 ± 7.05 |

### 8.3 过拟合观察

1. `mdJPT Frozen MLP` 的过拟合最明显：window Acc 的 train-test gap 为 `+14.24`，trial Acc 的 train-test gap 为 `+18.36`，而 test AUROC 只有 `63.50`。这说明仅用 pooled embedding 训练 MLP 时，模型能拟合训练集，但跨被试泛化较弱。

2. `CrossAttn Single Head` 也存在一定 train-test gap：window Acc gap 为 `+12.06`，trial Acc gap 为 `+11.89`。但它的 test 指标明显高于 Frozen MLP，说明融合手工特征与 token 后虽然仍有拟合训练集的能力，但泛化更好。

3. MoE 系列的 train-test gap 与 CrossAttn 接近，大约在 `+11.4` 到 `+12.8` 之间。MoE 没有明显更严重的过拟合，但也没有稳定缓解过拟合。

4. val 和 test 的 window 指标整体接近，甚至 CrossAttn 的 test Acc / AUROC 略高于 val，这说明当前 early stopping 并没有明显只贴合验证集。主要风险来自 train 与 held-out subject 之间的差距，也就是跨被试泛化难度。

5. 从过拟合角度看，当前最稳的汇报口径仍是：`CrossAttn Single Head` 是最强且相对稳健的神经融合基线；MoE 是有解释动机的结构探索，但当前没有显示出比 single-head 更好的泛化优势。

## 9. MoE Router 的 Train / Val / Test 被试类型区分结果

本节只看 MoE router 是否能区分被试类型 `HC/DEP`，不看情绪分类。MoE router 的输入是 `pool_1024`，输出 `[p_hc, p_dep]`。这里的 `Group Acc / F1 / AUROC` 都是针对 `HC` vs `DEP` 的二分类指标。

计算说明：

- train 指标是加载每个 seed 的 best-val checkpoint 后，直接用 `pool_1024 -> router` 在 train split 上复算得到。
- val / test 指标与原始 `moe_results/router_metrics.csv` 对齐。
- MoE 推理时使用原实验中基于验证集拟合得到的 router temperature。
- 逐 seed 结果保存到 `comparison/subject_router_train_val_test_models.csv`。
- 汇总结果保存到 `comparison/subject_router_train_val_test_models_summary.csv`。

### 9.1 Router 区分 HC/DEP 的主指标

| Variant | Train Group Acc | Val Group Acc | Test Group Acc | Train Group F1 | Val Group F1 | Test Group F1 | Train Group AUROC | Val Group AUROC | Test Group AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE Unsup | 49.72 ± 2.25 | 49.93 ± 1.87 | 48.12 ± 3.72 | 48.16 ± 1.46 | 48.19 ± 2.08 | 46.78 ± 3.22 | 50.04 ± 0.78 | 49.02 ± 1.51 | 49.59 ± 2.58 |
| MoE Sup `lambda=0.3` | 66.81 ± 4.84 | 55.28 ± 3.63 | 56.77 ± 4.61 | 64.39 ± 4.92 | 51.53 ± 3.87 | 53.13 ± 4.44 | 72.31 ± 6.38 | 52.85 ± 4.36 | 56.95 ± 5.34 |
| MoE Sup `lambda=0.5` | 68.26 ± 5.56 | 55.73 ± 4.24 | 57.33 ± 4.54 | 65.88 ± 5.75 | 52.06 ± 4.24 | 53.57 ± 4.28 | 73.91 ± 7.19 | 53.00 ± 4.90 | 57.56 ± 5.45 |
| Token MoE Sup `lambda=0.3` | 66.80 ± 4.82 | 55.28 ± 3.70 | 56.77 ± 4.61 | 64.37 ± 4.91 | 51.56 ± 3.95 | 53.13 ± 4.44 | 72.31 ± 6.39 | 52.85 ± 4.36 | 56.95 ± 5.35 |
| Token MoE Sup `lambda=0.5` | 68.27 ± 5.57 | 55.73 ± 4.24 | 57.36 ± 4.54 | 65.88 ± 5.75 | 52.06 ± 4.24 | 53.61 ± 4.27 | 73.91 ± 7.19 | 53.00 ± 4.90 | 57.56 ± 5.45 |

### 9.2 Router 概率校准观察

| Variant | Train p(HC) true HC | Val p(HC) true HC | Test p(HC) true HC | Train p(DEP) true DEP | Val p(DEP) true DEP | Test p(DEP) true DEP |
|---|---:|---:|---:|---:|---:|---:|
| MoE Unsup | 50.06 ± 1.15 | 49.95 ± 0.88 | 49.87 ± 1.11 | 50.11 ± 1.20 | 49.85 ± 1.05 | 50.23 ± 0.86 |
| MoE Sup `lambda=0.3` | 53.79 ± 2.19 | 51.90 ± 0.94 | 52.21 ± 1.46 | 52.72 ± 1.95 | 48.69 ± 0.92 | 49.64 ± 0.71 |
| MoE Sup `lambda=0.5` | 54.36 ± 2.70 | 52.17 ± 1.40 | 52.70 ± 2.03 | 53.07 ± 2.16 | 48.58 ± 0.96 | 49.58 ± 0.84 |
| Token MoE Sup `lambda=0.3` | 53.79 ± 2.19 | 51.90 ± 0.94 | 52.21 ± 1.46 | 52.72 ± 1.95 | 48.69 ± 0.92 | 49.64 ± 0.71 |
| Token MoE Sup `lambda=0.5` | 54.36 ± 2.70 | 52.17 ± 1.40 | 52.70 ± 2.03 | 53.07 ± 2.16 | 48.58 ± 0.96 | 49.58 ± 0.84 |

### 9.3 Router 结果解读

1. 无监督 MoE 的 router 在 train / val / test 上都接近随机水平。它的 Group Acc 约为 `48-50%`，Group AUROC 约为 `49-50%`，说明只靠 emotion loss 很难自动学出稳定的 HC/DEP 被试类型路由。

2. 加入 HC/DEP router supervision 后，train 上能明显学到被试类型信息。`lambda=0.5` 的 train Group Acc 达到 `68.26`，train Group AUROC 达到 `73.91`。

3. 监督 router 的泛化明显弱于训练集。`lambda=0.5` 的 test Group Acc 只有 `57.33-57.36`，test Group AUROC 约 `57.56`，说明 router 学到的是弱可泛化的 subject-type signal。

4. Token MoE 与普通 supervised MoE 几乎完全一致，说明本轮 subject-type token 没有额外增强 HC/DEP 路由能力。

5. 这解释了为什么 MoE 的情绪分类结果接近 single-head 但没有稳定超过：router 在训练集上能分出一些 HC/DEP 差异，但到 held-out subjects 上区分能力只略高于随机，专家分工没有充分泛化。
