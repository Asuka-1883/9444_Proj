# LSTM 实验交接说明

本目录保留 8 次正式训练的原始 JSON、3 次最终测试 JSON、实际使用的配置、两份已运行的分析 Notebook 和可导出的图表。

- LSTM_Analysis_CN.ipynb：中文学习版。
- LSTM_Analysis_EN.ipynb：英文交接版，可选取内容并入全组最终 Notebook。
- LSTM_Report_EN.md：根据真实结果写成的英文报告段落。
- analysis_summary.json：未舍入的单次结果、三次均值和样本标准差。
- final_test_summary.json：冻结单层配置后，三个检查点的最终测试指标、均值和样本标准差。
- analysis_audit.json：对原始记录完成的检查结果。
- selected_config.json：最终保留的单层配置；seed42 用于匹配公共基线。
- source_manifest.json：8 个训练日志、3 个测试日志和两份上传源码的 SHA-256。
- figures：5 张图的 PNG 与 SVG。两种格式来自同一份绘图代码。

## 实际完成内容

将提供的 LSTM 参考实现移入独立编码器文件，并保持共享输入输出；使用公共训练入口完成四种配置筛选；对单层与两层无层间 dropout 分别运行 42、43、44 三个种子；分析验证指标、波动、模型大小和记录的训练循环耗时；对按验证结果选定的单层配置完成三个种子的完整测试评分。

共享数据读取、位置/邻居分支、解码器、损失和训练脚本应按团队贡献描述。本代码不是手写 LSTM 门结构，也不是原版 Social LSTM 的复现。

## 放回已有仓库

压缩包以仓库相对路径组织。把 experiments/lstm 中的文件合并复制到已有仓库的同名目录。包里的 gc_data_interface 两份源码是本次上传的原样快照，可用于核对，不包含新的模型改动。

在 VS Code 打开英文或中文 Notebook，已有输出可直接阅读。需要重新生成图表时，在当前 Python 环境安装分析依赖，再选择该环境执行全部单元格。

```bash
python3 -m pip install -r experiments/lstm/requirements-analysis.txt
```

Notebook 会从自身目录、仓库根目录或 gc_data_interface 工作目录定位日志。重跑本包分析只需已保存的 JSON，不需要原始数据或模型权重。重新执行训练、测试评分或生成真实预测轨迹图才需要本机数据及检查点。

## 数据与比较口径

8 次记录使用相同的数据协议与窗口索引，训练 203,475 窗口，验证 43,814 窗口。每次最多 16 轮，按 validation ADE + 0.3 × turning ADE 选择检查点。均值与样本标准差使用三个种子，ddof = 1。

两层模型参数更多，且改变层数会影响初始化。此处是配置比较，不能描述成共享初始权重完全固定的严格层数消融。单层配置里的 encoder_dropout 字段为 0.1，实际 LSTM 层间 dropout 为 0；共同的融合层 dropout 仍是 0.05。

## 已完成：冻结单层配置后的最终测试

三个单层检查点都已完成完整测试评分。平均 ADE = 21.5538 ± 0.0877 像素，FDE = 37.2398 ± 0.1951 像素；转弯 ADE = 31.7821 ± 0.1892 像素，转弯 FDE = 60.6646 ± 0.5326 像素。每次 45,454 个测试窗口，转弯窗口 8,309 个。

以下仅保留复现命令，目前无需再次评分或训练。复现时先设置 GC_DATASET_ROOT，值为包含 Annotation 和 Frame 的数据集根目录，在 gc_data_interface 中运行：

```bash
for lstm_seed in 42 43 44
do
  python3 team_eval.py \
    --root "$GC_DATASET_ROOT" \
    --cache-dir work/team_cache \
    --checkpoint "runs/lstm_baseline_seed$lstm_seed/best.pt" \
    --split test \
    --output "runs/lstm_baseline_seed$lstm_seed/test.json" || break

  cp "runs/lstm_baseline_seed$lstm_seed/test.json" \
    "../experiments/lstm/baseline_seed$lstm_seed/test.json"
done
```

Notebook 已包含三次测试结果与第 5 张图。原始 result.json 的 test_evaluated=false 记录的是训练过程；测试结果单独保存在 test.json，保留两类原始记录。

Notebook 第 4–6 节是验证结果，第 8 节是最终测试结果。配置在最终评分前已经冻结，不再根据测试指标继续调参。TEAM_GUIDE.md 已说明该数据划分曾在更早开发中被查看，正式小组报告应注明开发历史，不把它描述成从未查看的测试划分。

## 接入全组比较

另一位组员收集 GRU / Transformer 时，先匹配相同公共配置、种子、数据协议与样本限制。team_compare.py 会检查这些设置。LSTM 调参比较和三种编码器的公共基线比较应分别解释，保留原来的公共基线结果。

## Git 提交

在仓库根目录检查修改并提交本成员的文件：

```bash
git status --short
git diff --stat
git add gc_data_interface/lstm_encoder.py gc_data_interface/motion_encoders.py experiments/lstm
git commit -m "Add LSTM encoder integration and experiment analysis"
git push -u origin models/lstm
```

再向 main 发起 Pull Request，说明模型接入、实验设置、实际验证和测试结果及检查方式。保留模型权重在本机 runs 目录；需要队友复现时提供运行命令与配置。提交前确认暂存列表只含本成员的源码与 experiments/lstm，排除数据集、缓存、runs、虚拟环境和临时压缩包。

此交接包覆盖 LSTM 成员的模型接入、实验分析与最终测试。全组仍需合并其他编码器、公共模型比较和最终报告，其余实验按课程要求及小组方案完成。
