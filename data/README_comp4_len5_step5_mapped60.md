# COMP4 `5s/5s` 映射后 EEG H5 说明

## 文件位置

- H5 文件: `data/comp4_len5_step5_mapped60.h5`
- 导出脚本: `preprocess/export_comp4_len5_step5_mapped60_h5.py`

这份 H5 来自赛题四训练集的 `mdjpt_comp4_125hz/processed_data`，只使用训练集被试，不包含公开测试集。

## 处理方式

- 原始数据已经从 `250 Hz` 重采样到 `125 Hz`
- 每个被试有 `8` 个 `50s` trial
- 每个 trial 按 `5s` 窗长、`5s` 步长切成 `10` 个 segment
- 每个 segment 先做和 mdJPT 读取流程一致的 z-score 标准化
- 然后按 `MultiModel_PL.channel_project()` 的规则，把 `30` 个源通道映射到 mdJPT 的 `60` 个标准通道

映射规则和模型内保持一致：

- 同名通道直接拷贝
- 缺失的标准通道，取 `channel_interpolate.npy` 里最近的、且当前输入中真实存在的最多 `3` 个邻居做均值
- 如果一个标准通道找不到可用邻居，则该通道补 `0`

## 数据规模

- 被试数: `60`
- trial 数: `60 x 8 = 480`
- segment 数: `60 x 8 x 10 = 4800`
- EEG 张量形状: `(4800, 60, 625)`

这里的 `625 = 5s x 125Hz`。

## H5 结构

### 根节点数据集

- `/eeg`
  - `float32`
  - 形状 `(4800, 60, 625)`
  - 每一行是一个 `5s` segment 的 `60` 通道 EEG

- `/label`
  - `int64`
  - 形状 `(4800,)`
  - 二分类标签
  - `0`: 正常人 trial
  - `1`: 抑郁患者 trial

- `/split`
  - `int8`
  - 形状 `(4800,)`
  - split 编码
  - `0=train`, `1=val`, `2=test`

- `/subject_index`
  - `int64`
  - 形状 `(4800,)`
  - 当前 segment 属于哪个被试，取值范围 `0..59`

- `/trial_index`
  - `int64`
  - 形状 `(4800,)`
  - 当前 segment 属于该被试的第几个 trial，取值范围 `0..7`

- `/segment_index`
  - `int64`
  - 形状 `(4800,)`
  - 当前 segment 是该 trial 里的第几个窗口，取值范围 `0..9`

- `/segment_start_sample`
  - `int64`
  - 形状 `(4800,)`
  - 当前 segment 在所属 trial 内的起始采样点

- `/segment_start_second`
  - `float32`
  - 形状 `(4800,)`
  - 当前 segment 在所属 trial 内的起始秒数

- `/global_trial_index`
  - `int64`
  - 形状 `(4800,)`
  - 全局 trial 编号，范围 `0..479`

### `/meta` 组

- `/meta/source_channel_names`
  - 原始 `30` 通道名

- `/meta/mapped_channel_names`
  - 映射后的 mdJPT `60` 标准通道名

- `/meta/subject_names`
  - `60` 个被试名，和 `subject_index` 对应

- `/meta/subject_groups`
  - 每个被试所属组别，`HC` 或 `DEP`

- `/meta/split_names`
  - split 名字顺序，固定为 `["train", "val", "test"]`

- `/meta/train_subject_indices`
- `/meta/val_subject_indices`
- `/meta/test_subject_indices`
  - 各 split 的被试索引

- `/meta/labels_per_trial`
  - 每个被试内部 `8` 个 trial 的标签顺序

- `/meta/trial_seconds`
  - 每个 trial 的时长，当前都是 `50`

- `/meta/segments_per_trial`
  - 每个 trial 切出来的 segment 个数，当前都是 `10`

- `/meta/segment_points_per_trial`
  - 每个 segment 的采样点数，当前都是 `625`

### 根节点属性

根节点属性记录了：

- 采样率 `fs=125`
- `segment_seconds=5`
- `segment_step_seconds=5`
- `segment_points=625`
- 被试数、trial 数、通道数
- 实际使用的数据根目录、split json 路径、配置路径
- split codebook
- 通道映射规则说明

## 读取示例

```python
import h5py

h5_path = "data/comp4_len5_step5_mapped60.h5"

with h5py.File(h5_path, "r") as f:
    eeg = f["eeg"]              # (4800, 60, 625)
    labels = f["label"][:]      # (4800,)
    splits = f["split"][:]      # (4800,)
    subject_idx = f["subject_index"][:]
    trial_idx = f["trial_index"][:]
    segment_idx = f["segment_index"][:]

    mapped_channels = [x.decode("utf-8") if isinstance(x, bytes) else x
                       for x in f["meta"]["mapped_channel_names"][:]]

    # 取第 0 个 segment
    x0 = eeg[0]                 # (60, 625)
    y0 = labels[0]
    s0 = splits[0]
```

## 手工特征建议

如果你后面要提每个 `5s` segment、每个通道的手工特征，推荐直接在 `/eeg` 上做：

- 单个 segment 的输入张量: `(60, 625)`
- 如果想按 split 过滤:
  - `train_mask = split == 0`
  - `val_mask = split == 1`
  - `test_mask = split == 2`
- 如果想回到被试或 trial 粒度：
  - 用 `subject_index`
  - 用 `trial_index`
  - 或直接用 `global_trial_index`
