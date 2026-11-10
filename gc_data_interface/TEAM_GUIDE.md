# 三人统一接口：GRU / LSTM / Transformer

接口包 v2.0.0。原 GCDataset 字段、时间协议和单位保持兼容；新增可替换编码器、
统一训练/评价入口、跨机器可比的数据指纹，以及保存/加载模型的元数据。
下面的命令在解压得到的 gc_data_interface 目录内运行。

## 1. 安装与检查

```bash
python -m pip install -r requirements-training.txt
python -m unittest discover -s . -p 'test_*.py' -v
```

本次在 Python 3.12 / PyTorch 2.15 / CPU 上验证；CPU 为默认设备。
--device auto 会选择可用 CUDA，否则使用 CPU；MPS 可显式指定，但本机未验证 MPS 运行。

## 2. 数据准备

以下所有命令中的 `DATASET_ROOT` 都是占位符，运行前请替换为各人本机的数据集根目录（里面有 `Annotation` 和 `Frame`）。不要原样使用占位符。
三人的本地路径可以完全不同；需要统一的是数据内容和 `team_config.json`。

```bash
python team_prepare.py --root "DATASET_ROOT" --cache-dir work/team_cache
```

这会生成训练/验证的数值缓存及指纹，不进行测试集评价。
第一次预处理需要解析 TXT 和邻居信息，三人可复用缓存；没有预先准备时，训练入口也会自动生成。
指纹基于规范化行人 ID、帧号、坐标和图像尺寸，与机器路径、解压时间无关。
原始数据改变时自动采用新的缓存键，防止错误复用旧样本。
缓存不包含视频图像。原 GCDataset 的 image 字段仍然保留。

## 3. 三位组员各自的训练命令

GRU 负责人：

```bash
python team_train.py --root "DATASET_ROOT" --encoder gru --cache-dir work/team_cache --output runs/gru
```

LSTM 负责人：

```bash
python team_train.py --root "DATASET_ROOT" --encoder lstm --cache-dir work/team_cache --output runs/lstm
```

Transformer 负责人：

```bash
python team_train.py --root "DATASET_ROOT" --encoder transformer --cache-dir work/team_cache --output runs/transformer
```

除 --encoder 和 --output 外，第一轮对照使用相同参数。每个输出目录必须是新目录，
避免覆盖之前的结果。每次运行产生 best.pt（模型）和 result.json（配置、训练曲线、验证误差）。
默认每个 epoch 打乱并完整遍历训练窗口，最多 16 epoch，按验证集早停。
训练过程不自动评价测试集。

快速验证安装及接口，可在三条命令后面都加：

```text
--epochs 2 --train-limit 2048 --val-limit 512
```

这些小样本短训只用来检查流程，不能据此给三个模型排名。

## 4. 哪些部分相同，哪些部分不同

| 部分 | 统一约定 |
|---|---|
| 数据 | 同一时间划分、相同训练/验证子集、随机种子 |
| 历史与未来 | 5 点历史覆盖 3.2 秒，6 点未来覆盖 4.8 秒 |
| 输入 obs | [B,5,4]，相对 x/y 和逐步位移 |
| 邻居 | 原来的 16 槽历史与有效掩码 |
| 预处理 | 同一方向对齐、内部 100 像素尺度；外部仍按 1920 归一化 |
| 位置、邻居、解码器 | 共用相同代码；相同 seed 时非编码器初始权重也一致 |
| 路径输出 | 同一个学习中途点/终点的单轨迹解码器 |
| 损失与评价 | 同一损失；还原像素后计算整体和转弯 ADE/FDE |
| 唯一架构选项 | 运动编码器 GRU、LSTM 或 Transformer |

统一比较入口采用位置＋邻居＋共享解码器，未加入场景 CNN。
原始图像接口与已有场景模型仍可使用；后续 CNN 对照应对三种编码器添加相同图像分支。
模型需要 5 张与历史轨迹对齐的原图时，可调用 `read_history_images(store, sample)`。
调用方法见 [历史原图读取示例](README.md#读取与轨迹对齐的历史原图)。

当前编码器默认设置（都输出 64 维特征）：

| 编码器 | 设置 | 整个模型参数量 |
|---|---|---:|
| GRU | 单层，hidden=64 | 80,449 |
| LSTM | 单层，hidden=64 | 84,929 |
| Transformer | 2 层，4 头，d_model=64，FFN=128，含正弦位置编码 | 134,401 |

64 维输出并不意味着参数量相同。结果中记录了编码器和完整模型参数量，以及训练耗时。
Transformer 的编码器 dropout 为 0.1；单层 GRU/LSTM 没有层间 dropout，公共解码部分相同。
这是一组明确记录配置的架构对照，不是严格等参数量实验。

## 5. 自己写训练循环时如何调用

```python
from gc_dataset import GCStore, GCDataset
from torch.utils.data import DataLoader
from team_model import build_team_model
from improved_forecast import prepare_context, forecast_loss, turn_mask

store = GCStore('DATASET_ROOT')
loader = DataLoader(GCDataset(store, 'train', include_image=False), batch_size=64, shuffle=True)
batch = next(iter(loader))

model = build_team_model('lstm')  # 可替换为 'gru' / 'transformer'
context = prepare_context(batch)
paths, scores = model(context)   # paths: [B,1,6,2]，公共内部坐标
loss = forecast_loss(paths, scores, context['future'], turn_mask(context), turn_weight=1.0)
loss.backward()
```

内部多保留了一个候选维度 K，第一阶段 K=1。推理适配器会返回 [B,6,2]：

```python
from predict_improved import predict_from_batch

result = predict_from_batch(model, batch)
prediction = result['top1_relative']  # [B,6,2]，与原 batch['future'] 同坐标单位
prediction_pixels = result['top1_pixels']  # [B,6,2]，原图像素位置
```

predict_from_batch 是无梯度的推理函数。训练时使用上面的 model(context) 和 loss。
它只读取 obs、neighbors、neighbor_mask、origin，不使用真实未来。
如果把模型移到 GPU，推理适配器会将这些输入一起移到相同设备。

## 6. 保存、加载和比较

```python
from predict_improved import load_model, predict_from_batch

model = load_model('runs/lstm/best.pt')  # 自动读取编码器类型、层数、位置编码等元数据
prediction = predict_from_batch(model, batch)['top1_relative']
```

旧的 GRU 改进版权重继续支持加载。三个编码器各自训练、保存自己的权重。

收集三人的 runs 子目录后，先比较验证结果：

```bash
python team_compare.py --runs runs/gru runs/lstm runs/transformer --output runs/comparison.json
```

比较程序会检查数据和公共配置的签名。子集大小、训练轮数、种子或公共参数不同会拒绝直接合表，
避免把不同实验当作仅替换编码器的对比。按相同种子分别汇总，再统计多种子的均值。

最终配置冻结后，再显式评价测试集，例如：

```bash
python team_eval.py --root "DATASET_ROOT" --cache-dir work/team_cache --checkpoint runs/lstm/best.pt --split test --output runs/lstm/test.json
```

team_eval 同样支持 --split val，以及仅用于检查的 --limit。
它会检查评价数据协议是否与训练记录一致。
本数据划分已在此前开发中多次查看，因此正式报告应说明开发历史。

## 7. v2 已实际检查

- 三种编码器的输入/输出、真实反向传播和参数更新。
- 同种子公共模块的初始权重逐项一致。
- Transformer 的位置编码能区分时间顺序。
- 三种模型保存后自动识别类型并正确加载。
- 推理移除未来标签后预测不变。
- 原 GRU 权重预测与已有保存结果一致。
- 跨路径相同数据指纹一致，坐标改变时指纹和缓存键改变。
- 每种编码器在真实 GC 的 2048 训练窗口、512 验证窗口上完整训练 2 轮，
  然后完成加载与统一评价流程。该短训不代表正式性能比较。

实现参考 PyTorch 官方 [LSTM](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LSTM.html)、
[TransformerEncoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html)
和 [TransformerEncoder](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoder.html) 接口。
