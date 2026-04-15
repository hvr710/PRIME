# EEG Feature Extraction Summary

本 README 旨在解释 `data/features_comp4_len5_step5_mapped60.h5` 文件的结构及其提取的特征。

## 1. 原始数据概览
- **输入文件**: `comp4_len5_step5_mapped60.h5`
- **采样率 (fs)**: 125 Hz
- **Segment 长度**: 5 秒 (625 samples)
- **总 Segment 数**: 4800
- **通道数**: 60 (基于 mapped_channel_names)

## 2. 特征提取配置
### 频段定义 (Frequency Bands)
- **Delta**: 1 - 4 Hz
- **Theta**: 4 - 8 Hz
- **Alpha**: 8 - 13 Hz
- **Beta**: 13 - 30 Hz
- **Gamma**: 30 - 45 Hz

### 计算方法
- **功率谱密度 (PSD)**: 使用 Welch 方法 (窗口长度 1s/125 samples，重叠 50%)。
- **频段功率**: 对 PSD 进行 Simpson 积分计算各频段下的能量。

## 3. HDF5 文件结构说明

文件内包含以下组 (Groups) 和数据集 (Datasets)：

### `absolute_power/` (AP)
包含各频段的绝对功率。
- 数据集: `delta`, `theta`, `alpha`, `beta`, `gamma`
- 形状: `(4800, 60)` -> (Segments, Channels)

### `relative_power/` (RP)
包含各频段相对于总功率 (1-45 Hz) 的比例。
- 数据集: `delta`, `theta`, `alpha`, `beta`, `gamma`
- 形状: `(4800, 60)`

### `de_log_power/` (DE)
包含各频段的微分熵 (Differential Entropy) 近似值。
- 公式: `0.5 * log(2πe * AP)`
- 数据集: `delta`, `theta`, `alpha`, `beta`, `gamma`
- 形状: `(4800, 60)`

### `asymmetry/`
包含对称通道对之间的对数功率差。
- 公式: `log(Left_AP) - log(Right_AP)`
- 数据集: `delta`, `theta`, `alpha`, `beta`, `gamma` (形状: `(4800, 26)`)
- 辅助数据集: `{band}_names` (如 `alpha_names`) 存储了对应的通道对名称（共 26 对）。

### `ratios/`
包含特定频段之间的功率比例。
- `theta_beta`: Theta / Beta 比例
- `alpha_beta`: Alpha / Beta 比例
- 形状: `(4800, 60)`

### `faa/` (Frontal Alpha Asymmetry)
专门提取的前额区 Alpha 波对称性。
- 公式: `log(Right_Alpha_AP) - log(Left_Alpha_AP)` (注意：FAA 通常定义为右减左)
- 数据集: `values` (形状: `(4800, 10)`)
- 数据集: `names` (存储了 10 对前额对称通道名称)

### `hjorth/` (Hjorth Parameters)
反映信号在时域的统计特性。
- `activity`: 信号的方差。
- `mobility`: 信号平均频率的测量。
- `complexity`: 信号波形变化的复杂度。
- 形状: `(4800, 60)`

### 其他时频域特征 (Additional Features)
- `spectral_entropy`: 归一化的功率谱熵，衡量信号的复杂度和不规则性。
- `pfd`: Petrosian Fractal Dimension，快速分形维度估计。
- `std`: 信号的标准差，反映信号的波动强度。
- 形状: `(4800, 60)`

### `meta/`
存储元数据信息。
- `channel_names`: 原始 60 个通道的名称。
- `bands`: 提取的 5 个频段名称。
- `subject_names`: 60 个被试名。
- `subject_groups`: 被试组别。
- `split_names`: split 编码名称。

### 标签与索引
以下数据集已经从 `comp4_len5_step5_mapped60.h5` 同步到特征 H5，可直接用于下游训练和复现实验划分：

- `label`: `(4800,)`，二分类标签。
- `split`: `(4800,)`，原始 H5 中保存的 split 编码，`0=train, 1=val, 2=test`。
- `subject_index`: `(4800,)`，被试索引。
- `trial_index`: `(4800,)`，被试内 trial 索引。
- `segment_index`: `(4800,)`，trial 内 segment 索引。
- `global_trial_index`: `(4800,)`，全局 trial 索引。
- `segment_start_sample`: `(4800,)`，segment 在 trial 内起始采样点。
- `segment_start_second`: `(4800,)`，segment 在 trial 内起始秒数。

## 4. 手工特征 baseline

传统模型 baseline 脚本：

- `attach_feature_metadata.py`: 将标签和索引元信息写入特征 H5。
- `run_feature_baselines.py`: 使用固定随机种子 `42, 3407, 2025`，分别跑按被试划分和按 segment 随机划分的 `LogisticRegression / SVM / XGBoost / MLP`。

结果目录：

- `feature_baseline_results/all_results.csv`: 每个 seed、split、模型的详细结果。
- `feature_baseline_results/summary.json`: 三个 seed 的均值和标准差。
- `feature_baseline_results/splits/`: 固定保存的 subject-level 和 segment-level split 文件。

## 5. 相关脚本
- 特征提取脚本: `extract_features.py`
- 格式转换脚本: `convert_to_csv.py` (可将此 HDF5 转为 CSV)

---
*Generated on: 2026-04-10*
