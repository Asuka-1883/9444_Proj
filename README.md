# 9444_Proj

COMP9444 五人小组项目：GC 行人轨迹预测。

当前提交为 gc_data_interface_v2_0_1 接口包的代码与文档，供 GRU、LSTM、Transformer 三位模型负责人共同使用。
后续模型实现、配置和实验记录可在本仓库各自分支开发，再通过 Pull Request 合并。

## 项目结构

```text
9444_Proj/
├── gc_data_interface/   # 共享数据读取、协议、模型参考实现、训练与评价入口
├── CONTRIBUTING.md     # 分支协作与提交约定
├── .gitignore          # 排除原始数据、缓存、权重和本机文件
└── README.md
```

先读 [团队接入指南](gc_data_interface/TEAM_GUIDE.md)；字段定义见 [数据接口说明](gc_data_interface/README.md)。

## 本机运行

```bash
git clone https://github.com/Asuka-1883/9444_Proj.git
cd 9444_Proj/gc_data_interface
python -m pip install -r requirements-training.txt
python -m unittest discover -s . -p 'test_*.py' -v
```

`DATASET_ROOT` 是占位符，替换为各自电脑上包含 `Annotation` 和 `Frame` 的数据集根目录。
各成员路径无需相同；数据内容和用于比较的公共配置应一致。

```bash
python team_prepare.py --root "DATASET_ROOT" --cache-dir work/team_cache
python team_train.py --root "DATASET_ROOT" --encoder gru --cache-dir work/team_cache --output runs/gru
python team_train.py --root "DATASET_ROOT" --encoder lstm --cache-dir work/team_cache --output runs/lstm
python team_train.py --root "DATASET_ROOT" --encoder transformer --cache-dir work/team_cache --output runs/transformer
```

每位负责人运行对应模型的命令即可。快速检查可添加 `--epochs 2 --train-limit 2048 --val-limit 512`。

## 共同约定

- 过去 5 个观测点，预测未来 6 个时刻；采样间隔 0.8 秒。
- 公共输入包括 `obs`、`neighbors`、有效掩码和 `origin`；`future` 只用于损失和评分。
- 原接口支持当前观测图像；三模型统一训练基线默认使用坐标与邻居，不含 CNN。
- 实际单条预测统一为 `[B, 6, 2]`；ADE/FDE 还原到像素后评价。
- 保持公共数据划分与配置一致，训练按验证集选择检查点。
- 原始数据集、缓存、训练权重不提交；小型实验配置、代码和必要的汇总指标可以提交。
- 历史训练包与后续改进实验未混入本次接口初始提交，避免组员的基线不一致。

## 协作

私有仓库需由仓库所有者邀请组员的 GitHub 账号后才能访问。
参考 [CONTRIBUTING.md](CONTRIBUTING.md) 创建自己的模型分支并提交 Pull Request。

使用数据集时请引用：Shuai Yi, Hongsheng Li, Xiaogang Wang. *Understanding Pedestrian Behaviors from Stationary Crowd Groups*, CVPR 2015。
