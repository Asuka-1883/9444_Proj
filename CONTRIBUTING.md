# 小组协作

`main` 保存全组可运行的公共版本。建议三位模型负责人使用以下分支：

仓库公开，查看和克隆不需要邀请。直接向本仓库推送分支需要协作者权限。
如果不配置协作者，先在 GitHub 点击 Fork，将仓库复制到自己的账号，再克隆自己的 Fork；下面的 `git push` 会推送到各自的 Fork，之后向 `Asuka-1883/9444_Proj` 发起 Pull Request 即可。

| 负责人 | 分支 |
|---|---|
| GRU | `models/gru` |
| LSTM | `models/lstm` |
| Transformer | `models/transformer` |

以 GRU 为例，在仓库根目录执行；另两位将分支名换为对应名称：

```bash
git switch main
git pull --ff-only
git switch -c models/gru
# 修改并验证自己的代码
git add path/to/changed_file.py
git commit -m "Add GRU experiment"
git push -u origin models/gru
```

随后在 GitHub 发起 Pull Request，写明改了什么、运行命令、配置和验证结果。
已有分支可直接 `git switch models/gru`，无需重复创建。

## 数据与实验

- 原始数据各自下载到本机，命令统一用 `--root` 传入自己的路径。
- 缓存放 `work/`、模型权重和完整运行产物放 `runs/`，这些目录被 Git 忽略。
- 需要分享的轻量配置和汇总指标可放 `experiments/<encoder>/`；不得将原始数据、权重移入该目录提交。
- 修改时间长度、坐标单位或数据划分前，全组先统一协议；比较时记录随机种子和参数量。
- 保留验证集选择与最终评分的区别；不要按测试结果挑选模型。

## 提交前检查

在 `gc_data_interface/` 中运行：

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

不要提交密码、Token、个人绝对路径或本机环境目录。
