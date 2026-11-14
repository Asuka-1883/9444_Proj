# GRU 更新：历史轨迹建模 + 转弯轨迹优化

## 一句话总结

-   我使用 **GRU** 对行人过去 5
    个状态的运动轨迹进行编码，用来预测未来轨迹。
-   基线 GRU 在测试集上的 **ADE 是 21.59 px，FDE 是 37.34 px**。
-   基线 GRU 对转弯行人的预测明显更差，所以进一步调整了 **turn
    weight**。
-   使用 **turn weight = 1.25** 后，Turn ADE 从 **32.11 px 降到 31.55
    px**，Turn FDE 从 **61.37 px 降到 59.91
    px**，同时整体性能基本没有变化。
-   `turn_weight = 1.50` 对转弯预测更好，但整体 ADE/FDE
    稍微变差，所以最终选择 **1.25**。

## 使用的文件

  ----------------------------------------------------------------------------------
  文件                                作用
  ----------------------------------- ----------------------------------------------
  `motion_encoders.py`                定义 GRU 序列编码器。

  `team_model.py`                     在 GRU、LSTM 和 Transformer 之间选择 encoder。

  `improved_forecast.py`              公共轨迹预测框架，结合运动、空间和邻居信息。

  `team_data.py`                      读取和处理轨迹数据。

  `team_prepare.py`                   数据预处理和缓存。

  `team_train.py`                     训练 GRU。

  `team_eval.py`                      计算 ADE、FDE、Turn ADE 和 Turn FDE。

  `team_config.json`                  保存模型和训练参数。
  ----------------------------------------------------------------------------------

### 模型怎么工作

1.  **历史轨迹：** 使用行人过去 **5 个时间步**，每个时间步包含
    **x、y、Δx、Δy** 4 个运动特征。
2.  **GRU 编码器：** 使用 **1 层 GRU**，hidden size = **64**，最后的
    hidden state 用来表示历史运动。
3.  **空间和邻居信息：** GRU 得到的运动特征会和公共模型中的 spatial
    information 和 social information 结合。
4.  **输出：** 融合后的特征输入公共 trajectory predictor，预测未来轨迹。

核心 GRU：

``` python
nn.GRU(
    input_size=4,
    hidden_size=64,
    num_layers=1,
    batch_first=True
)
```

### 主要配置

  设置                  值
  --------------------- ----------------------
  历史输入              5 steps × 4 features
  Hidden size           64
  GRU layers            1
  Batch size            256
  Learning rate         0.001
  Epochs                16
  Weight decay          0.0001
  Gradient clip         5.0
  Spatial information   使用
  Social information    使用
  完整模型参数量        80,449

因为 GRU 只有 1 层，所以实际 recurrent dropout 是 **0**。

## 数据

使用 **CVPR 2015 Pedestrian Walking Path Dataset**。

  数据集          样本数
  ------------ ---------
  Training       203,475
  Validation      43,814
  Test            45,454

每个样本使用过去 **5
个历史状态**作为序列输入。更新后的数据接口也可以提供对应的视频帧，但目前
GRU 主要使用历史运动信息，没有直接使用图像作为输入。

## 基线结果

Baseline：`turn_weight = 1.00`

测试集共有 **45,454 个样本**，其中 **8,309 个 turning samples**。

  模型             Turn Weight   参数量     ADE     FDE   转弯 ADE   转弯 FDE
  -------------- ------------- -------- ------- ------- ---------- ----------
  GRU Baseline            1.00   80,449   21.59   37.34      32.11      61.37

误差单位均为 **pixels**。转弯轨迹的误差明显更高，因此后续优化主要针对
turning trajectories。

## GRU 优化：Turn Weight

训练 loss 会对 turning samples 设置额外权重：

``` python
weight = 1 + (turn_weight - 1) * turning.float()
```

测试：

-   `turn_weight = 1.00`
-   `turn_weight = 1.25`
-   `turn_weight = 1.50`

其他主要设置保持不变。

## 结果

    Turn Weight       ADE       FDE      转弯 ADE      转弯 FDE
  ------------- --------- --------- ------------- -------------
           1.00   21.5905   37.3355       32.1069       61.3719
       **1.25**   21.6064   37.3451   **31.5508**   **59.9101**
           1.50   21.7654   37.7415       31.2404       59.4286

所有误差单位均为 **pixels**。

### 能得出的结论

-   **Baseline 的主要问题是转弯轨迹：** Turn ADE/FDE 明显高于整体
    ADE/FDE。
-   **turn_weight = 1.25：** Turn ADE 从 32.11 降到 31.55 px，Turn FDE
    从 61.37 降到 59.91 px，而整体性能基本不变。
-   **turn_weight = 1.50：** 转弯结果进一步改善，但 Overall ADE/FDE
    稍微变差。
-   **最终选择 1.25：** 在整体性能和转弯性能之间更加平衡。

## Discussion

-   GRU 可以从历史轨迹中学习行人的运动模式。
-   基线 GRU 对转弯轨迹预测更加困难。
-   增加 turn weight 可以让模型在训练时更加关注转弯样本。
-   `turn_weight = 1.25` 改善了转弯轨迹，同时基本保持整体预测性能。
-   GRU 仍然比较难预测突然改变方向或复杂的行人运动。
-   当前 GRU 没有直接使用历史视频帧作为视觉输入。

## Future Work

更新后的数据接口已经可以读取历史状态对应的视频帧。以后可以尝试把：

**历史运动信息 + 图像/场景信息 + 邻居信息**

一起用于 GRU，从而为复杂轨迹，特别是转弯轨迹，提供更多场景信息。

## 怎么复现

所有命令都在 `gc_data_interface/` 目录下运行。

``` bash
# 1. 准备数据
python3 team_prepare.py \
  --root /Users/mac/Documents/9444/project/cvpr2015_pedestrianWalkingPathDataset \
  --cache-dir work/team_cache

# 2. 训练 GRU
python3 team_train.py \
  --root /Users/mac/Documents/9444/project/cvpr2015_pedestrianWalkingPathDataset \
  --encoder gru \
  --cache-dir work/team_cache \
  --output runs/gru

# 3. 测试 GRU
python3 team_eval.py \
  --root /Users/mac/Documents/9444/project/cvpr2015_pedestrianWalkingPathDataset \
  --cache-dir work/team_cache \
  --checkpoint runs/gru/best.pt \
  --split test \
  --output runs/gru/test.json
```

## 结论

GRU 能够有效学习行人的历史运动模式，但对转弯轨迹的预测更加困难。将 turn
weight 从 1.0 调整到 1.25 后，转弯轨迹的 ADE 和 FDE
得到了改善，同时整体预测性能基本保持不变。因此最终选择
`turn_weight = 1.25`。
