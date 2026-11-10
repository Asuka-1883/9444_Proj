# GC 数据接口：给模型组员的接入说明

**v2.0.0：三位模型组员请先阅读 [TEAM_GUIDE.md](TEAM_GUIDE.md)。**
已提供统一的 GRU / LSTM / Transformer 构建、训练、加载、评价和比较入口。
以下内容保留原始数据协议及此前实验说明；原 GCDataset 字段保持兼容。

下文 `DATASET_ROOT` 是通用占位符，请替换为本机包含 `Annotation` 和 `Frame` 的数据集根目录。

本接口保留原始数据，不修改数据集中的文件。核心代码只依赖 NumPy、Pillow；
GCDataset 实现 __len__ / __getitem__，可以直接交给 PyTorch DataLoader。
preview/ 是本机运行 prepare_gc.py 后生成的检查输出，不提交到 Git。

## 1. 时间与数据约定（全组先确认）

- 一份 TXT 对应一个行人。每三个数依次是 x、y、原始帧号。坐标为像素，x 向右、y 向下。
- 原视频按 25 FPS 解释；Frame 文件每隔 20 原始帧保存一张，所以采样间隔为 0.8 秒。
- 以最后观测时刻 t 为锚点，历史点为 t+[-3.2,-2.4,-1.6,-0.8,0]，共 5 点。
- 未来点为 t+[0.8,1.6,2.4,3.2,4.0,4.8]，共 6 点。
- 历史窗口两端都包含，因此 5 个位置覆盖 4 个间隔，即 3.2 秒。此协议没有插值。
- 这是根据原始采样制定的基线协议；若课程指定另一采样协议，全组应共同修改并重新生成样本。
- 默认保持时间先后，按累计标注点数量约 70% / 15% / 15% 选两个时间切点，
  然后在三个连续时间段内部制作窗口。比例针对标注点数量，最终窗口数量比例会有所不同。
- 原因：真实数据在帧号 100000 后标注骤减。直接按全部图片时间长度 70/15/15 划分，
  测试集只有 999 个有效窗口、28 个目标行人。默认划分避免让测试集集中在这段稀疏尾部。
- 如需原先的等时间比例方式，使用 GCConfig(split_strategy='time')。
  划分依据只有时间与标注数量，未使用坐标值、预测误差或测试模型结果。
- 一个目标样本的全部历史和未来都必须连续，帧号差严格为 20；缺帧时切断，不跨缺口插值。
- 同一个行人可能出现在不同时间段，训练与测试使用不同时间窗口；这是同场景时间留出评估，
  不能据此声称跨场景或未见行人泛化。如果课程有固定划分，以课程协议为准。
- TXT 中仅提取文件名开头的数字作为 ID，以兼容 007427 带额外后缀的文件；重复 ID 会报错。

## 2. 安装与检查

在本目录运行（使用自己的 Python 环境）：

```bash
python -m pip install -r requirements.txt
python prepare_gc.py --root "DATASET_ROOT"
python -m unittest discover -s . -p 'test_*.py' -v
```

默认写入本目录 preview/：manifest.json、train_sample.npz、val_sample.npz、test_sample.npz。
NPZ 用 np.load(path, allow_pickle=False) 读取，数组字段保持下表定义。
manifest.json 记录数据根路径、参数、数据统计、划分边界、样本数和字段形状。

## 3. 模型组员的调用方式

```python
from gc_dataset import GCStore, GCDataset
from torch.utils.data import DataLoader

store = GCStore('DATASET_ROOT')
train_ds = GCDataset(store, 'train', include_image=True)
val_ds = GCDataset(store, 'val', include_image=True)
test_ds = GCDataset(store, 'test', include_image=True)

# 首先用 num_workers=0 跑通，之后按机器情况增加。
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
batch = next(iter(train_loader))

# 模型输入仅为历史、邻居及当前观测图像。
prediction = model(
    obs=batch['obs'],
    neighbors=batch['neighbors'],
    neighbor_mask=batch['neighbor_mask'],
    image=batch['image'],
)  # 期望输出 [B, 6, 2]，相对锚点的归一化未来位置

target = batch['future']  # 只用于损失或评估，不能传入预测模型
loss = criterion(prediction, target)
```

以上 model/criterion 是自定义模型的占位调用。包内同时提供 team_model.py 和 team_train.py 作为三种编码器的统一参考实现。
组员应根据自己的模型签名传入额外掩码，例如 neighbor_delta_mask、image_mask。
DataLoader 自动将 NumPy 数组转换为 torch.Tensor，字符串路径整理成列表。
在 Windows/macOS 增加 num_workers 时，将创建 DataLoader 和训练入口放在
`if __name__ == '__main__':` 下。完整训练另外安装团队使用的 PyTorch 环境。

## 4. 返回字段：下列为单个样本，DataLoader 会在前面加 B 维

| 字段 | 默认形状 / 类型 | 解释 |
|---|---|---|
| obs | [5,4] float32 | 自身历史 [相对x,相对y,delta_x,delta_y] |
| obs_mask | [5] bool | 自身位置有效标记，本协议始终为 True |
| obs_delta_mask | [5] bool | 位移有效标记；首步没有前驱，标记 False |
| neighbors | [16,5,4] float32 | 邻居历史，与自身使用同一坐标原点和尺度 |
| neighbor_mask | [16,5] bool | 每个邻居、每个历史时刻的位置是否有效 |
| neighbor_slot_mask | [16] bool | 该邻居槽是否被使用 |
| neighbor_delta_mask | [16,5] bool | 当前和前一时刻都存在时，位移才有效 |
| future | [6,2] float32 | 真实未来位置，相对锚点、归一化；只作标签 |
| image | [3,128,128] float32 | RGB，范围 [0,1]；仅在 include_image=True 时返回 |
| image_mask | [1,128,128] bool | 图像有效区域；越过图像边界的黑色填充为 False |
| origin | [2] float32 | 自身最后观测位置，原始像素坐标 |
| scale | [2] float32 | 默认 [1920,1920]，固定尺度，无需拟合全数据 |
| pedestrian_id | int64 | 目标行人 ID |
| neighbor_ids | [16] int64 | 邻居 ID；空槽为 -1 |
| anchor_frame | int64 | 最后观测的原始帧号 |
| obs_frame_ids | [5] int64 | 历史帧号 |
| future_frame_ids | [6] int64 | 标签帧号 |
| image_path | str | 最后观测时刻的图像路径 |
| crop_box | [4] int64 | 原图裁剪框 left,top,right,bottom，右/下端不包含 |

obs/neighbors 的 delta 是相邻采样时刻的位置差，不是除以时间后的速度。
所有坐标特征和 delta 都除以相同的 scale。未来标签是相对锚点的位置，
解码器若输出逐步位移，需先沿时间累加再与 future 比较。

## 5. 邻居与图像约定

- 在锚点帧按像素欧氏距离选择最近的至多 16 个其他行人；距离相同时按 ID 排序。
- 选择邻居只使用最后观测帧，邻居是否在未来继续存在不影响选择。
- 这是简易邻居选择方法。摄像头有透视，像素距离不等于真实空间距离；
  若后续使用地面标定，应同时更新邻居距离与坐标约定。
- 邻居历史按目标的历史帧号对齐；缺失位置、缺失位移及空槽均填零，并提供独立掩码。
- 零也可能是真实的相对坐标，必须以掩码判断缺失。
- 全空邻居时，社会聚合特征应设为零。实现 masked softmax 时要单独处理全空情况，避免 NaN。
- 图像只读取锚点帧，以目标位置为中心裁剪 256×256 像素，再缩放为 128×128。
- 原图外侧黑色填充。未执行 ImageNet 均值/标准差归一化；CNN 组员根据权重要求执行一次。
- 本接口未对像素坐标做米制标定，ADE/FDE 应报告为像素；不能直接套用 0.2 米碰撞阈值。

### 读取与轨迹对齐的历史原图

在 `gc_data_interface/` 目录使用 `read_history_images(store, sample)`。
函数按 `obs_frame_ids` 的顺序读取原图，每张图对应 `sample['obs']` 中同一时刻的位置。

```python
from gc_dataset import GCStore, GCDataset
from gc_frames import read_history_images

store = GCStore('DATASET_ROOT')
dataset = GCDataset(store, 'train', include_image=False)
sample = dataset[0]
history_images = read_history_images(store, sample)  # [5, 1080, 1920, 3]
```

`history_images` 是原尺寸 RGB `uint8` 数组。默认取 5 帧，形状为 `[T,H,W,3]`。
图像缩放、归一化和批处理在模型侧处理。若使用像素坐标和图片对齐，缩放图像时也要同步变换坐标。

## 6. 预测还原与评估

单样本 NumPy：

```python
from gc_dataset import restore_pixels
future_pixels = restore_pixels(sample['future'], sample['origin'], sample['scale'])
```

批次 PyTorch（各 Tensor 在同一个 device）：

```python
pred_pixels = prediction * batch['scale'][:, None, :] + batch['origin'][:, None, :]
gt_pixels = batch['future'] * batch['scale'][:, None, :] + batch['origin'][:, None, :]
distance = (pred_pixels - gt_pixels).norm(dim=-1)  # [B,6]
ade = distance.mean()
fde = distance[:, -1].mean()
```

可视化原始图像时，使用还原后的像素坐标。CVAE 的训练后验可以使用 future，
但其测试预测接口必须仅使用历史条件和先验采样。

## 7. 验证范围与分工交付

已附语义测试：时间端点、跨缺帧/跨集合窗口、缺失邻居掩码、未来才进入的行人、
修改未来不影响输入、坐标还原、图像形状和非标准文件名。
可选 DataLoader 检查在已安装 PyTorch 时自动运行，未安装时跳过。

2026-11-03 已安装项目局部 PyTorch 2.15.0，7 项检查全部通过，包括实际 DataLoader 批处理。

交给组员：本目录代码、README、manifest.json 以及统一的数据集根路径设置。
原始数据无需复制到 Git。全组统一时间协议、坐标还原和划分后，再开始比较模型结果。

## 8. 参考依据

- 原论文表 1：原视频 25 FPS、标注 1.25 FPS。
  https://openaccess.thecvf.com/content_cvpr_2015/papers/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.pdf
- PyTorch map-style dataset 与默认批处理约定：
  https://docs.pytorch.org/docs/stable/data.html

## 9. 实际训练与复现

已提供 train_audit.py、extend_training_audit.py、plot_training_audit.py。
第一次试训比较 8 组参数，仅用验证集选择配置，然后将选中配置换两个随机种子训练，
最后在完整测试集评分。扩展实验仅使用训练/验证集比较全量数据和减少重叠的窗口，
不重新评估测试集。这些历史实验结果未提交到仓库；运行以下脚本可在本机重新生成。

```bash
python -m pip install -r requirements-training.txt
python train_audit.py --root "DATASET_ROOT" --work ./training_cache --output ../gc_training_review --steps 1500
python extend_training_audit.py --root "DATASET_ROOT" --work ./training_cache --output ../gc_training_review
python plot_training_audit.py --results ../gc_training_review
```

训练脚本用于当前默认的 5 点历史、6 点未来、16 个邻居槽、接口尺度 1920。
更换原始标注或数据划分后，应使用新的 --work 缓存目录。
脚本的 scale=100 是模型内部重缩放，不改变本接口对外的坐标约定，预测输出也会转换回接口单位。
损失按 100 像素为一个单位计算 Smooth L1，beta=0.2，避免误把不同尺度下的损失直接比较。
缓存只包括数值轨迹与邻居特征；训练示例未加入 CNN。

加载首个种子的基线模型（不是完整场景模型）：

```python
import torch
from train_audit import ForecastGRU

checkpoint = torch.load('../gc_training_review/social16.pt', map_location='cpu', weights_only=True)
config = checkpoint['config']
model = ForecastGRU(hidden=config['hidden'], neighbors=config['neighbors'], scale=config['scale'])
model.load_state_dict(checkpoint['state_dict'])
model.eval()
with torch.no_grad():
    prediction = model(batch['obs'], batch['neighbors'], batch['neighbor_mask'])
```

数据接口默认参数保持不变，以免组员已有代码的单位约定发生变化。
首次团队训练请使用 TEAM_GUIDE.md 中的统一入口；历史实验脚本保留供参考。

## 10. 改进后的路线模型

针对预测偏直的问题，新增 improved_forecast.py、train_improved.py 与相关消融脚本。
原 GCDataset 的字段和坐标单位保持不变；使用 predict_improved.py 可直接接入原 DataLoader。
主模型使用位置条件、运动方向对齐和学习到的中途点/终点。
另外提供转弯加权、静态场景 CNN 和 20 候选模式对照。
历史实验权重和结果不包含在本仓库中；请按 TEAM_GUIDE.md 在本机训练，并明确区分模型首选与真实未来事后选择候选的评价。
