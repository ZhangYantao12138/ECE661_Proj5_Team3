# A 详细计划书：数据、分类器、公共评价与对照

负责人：余希泰（xy195）。执行依据：[统一研究协议](00_Shared_Research_Protocol.md)。

## A.1 研究目标与交付边界

A 负责建立可审计的实验基准：正常分类器具有合理性能，贴片与防御使用相同输入含义，ASR 的分母清楚，任意汇总数字均能从逐样本记录重算。A 的研究分析集中在模型质量、遮挡效应与评价公平性。

交付文件包括 `SplitManifest`、三个 `ModelBundle`、归一化规格、统一评价器、随机/恒色对照、正常准确率与模型规格表、指标单元检查和矩阵审计结果。

## A.2 第一步：建立实验目录与版本记录

1. 在仓库根目录创建共享协议规定的 `configs/、src/、scripts/、tests/、artifacts/、results/`。
2. 创建 `configs/protocol.yaml`，写入 45,000/5,000/10,000 划分、seed 661、三个模型、贴片 seed 101/102/103、尺寸、目标类、变换和评价抽样规则。
3. 保存环境记录：Python、PyTorch、torchvision、设备后端、驱动、操作系统和精度；导出依赖锁文件。
4. 用短实验测量模型 forward/backward 与输入梯度，不把能正常分类视为贴片反向已被支持。
5. 与 B 确认输入为 raw `[0,1]`；与 C 确认多源共用渲染图；与 D 确认防御在 Normalize 之前。

验收：四人的同一张 raw 图经过同一个模型包装器时，logits 在同设备同精度下相同或处于已记录的浮点容差内；环境与配置文件可定位到同一 commit。

## A.3 第二步：实现固定、分层且可复验的数据划分

实现 `src/data.py` 与 `scripts/prepare_data.py`：

1. 下载并校验官方 CIFAR-10 文件；保留官方 train/test 划分和原始类别顺序。
2. 为每类训练索引独立打乱，取 500 为验证，剩余 4,500 为训练。保存整数索引，不保存仅能在某次 DataLoader 顺序中成立的位置。
3. 验证 train、val 互斥，并集等于全部 50,000 训练样本；每类数量准确；test 索引全部来自官方测试集。
4. 在训练 raw 像素上按通道累计 `sum(x)、sum(x²)、n_pixels`，以 float64 累计，计算均值与总体标准差。不得把逐图片标准差简单平均当成整体标准差。
5. 保存数据版本、类别映射、索引列表、统计量和 SHA-256。
6. 创建三种 loader：监督训练 loader、贴片优化 loader、验证/测试 loader。监督训练可裁剪翻转；贴片优化读取未裁剪背景，仅由 B 的渲染器加入贴片变换；正式评价保持原始图片。

监督增强固定为四周 padding 4 后随机裁剪至 32×32，再以 0.5 概率水平翻转。增强只发生在监督训练支路。每个 loader 返回 `image_raw、label、image_id`。

验收：

- train/val/test 数量为 45,000/5,000/10,000，且 class counts 分别为每类 4,500/500/1,000。
- 重新运行划分脚本得到相同清单 hash。
- 验证和测试连续读取同一索引得到相同 raw 张量；标签与官方样例相符。
- Normalize 只应用一次；均值、标准差及图像 dtype/range 在日志中可见。

## A.4 第三步：实现并核对三个分类器

### A.4.1 ResNet-20

采用原论文 CIFAR 结构的 6n+2 定义，n=3：[结构依据](https://arxiv.org/abs/1512.03385)。

| 部分 | 结构 | 输出形状，忽略 batch |
|---|---|---|
| Stem | 3×3 conv，3→16，stride 1，padding 1；BN；ReLU | 16×32×32 |
| Stage 1 | 3 个基本块，每块 3×3 conv-BN-ReLU、3×3 conv-BN、残差相加-ReLU | 16×32×32 |
| Stage 2 | 3 个块；首块首卷积 stride 2，通道变 32 | 32×16×16 |
| Stage 3 | 3 个块；首块首卷积 stride 2，通道变 64 | 64×8×8 |
| Head | 全局均值池化；Linear(64,10) | 10 logits |

卷积不使用 bias，BN 使用可训练 affine。Option A shortcut 用步长为 2 的空间下采样和通道零填充，不加入学习型 1×1 投影。按此定义可训练参数量应为 269,722；若不同，逐层核对 bias、投影、BN 和分类头。不能用 torchvision 的 ResNet-18 替代后继续标为 ResNet-20。

### A.4.2 VGG-11-BN-CIFAR

按 VGG A 配置构建卷积骨干：[结构依据](https://arxiv.org/abs/1409.1556)。

```text
Conv64-BN-ReLU, MaxPool2
Conv128-BN-ReLU, MaxPool2
Conv256-BN-ReLU, Conv256-BN-ReLU, MaxPool2
Conv512-BN-ReLU, Conv512-BN-ReLU, MaxPool2
Conv512-BN-ReLU, Conv512-BN-ReLU, MaxPool2
Flatten
Linear512→512, ReLU, Dropout0.5
Linear512→512, ReLU, Dropout0.5
Linear512→10
```

所有卷积为 3×3、padding 1；五次池化使空间分辨率从 32 降到 1。该适配结构保留 8 个卷积层和 3 个全连接层。记录 CIFAR 分类头的规格，不将其描述为原版 ImageNet 4096 维分类头。

### A.4.3 DenseNet-BC-40-12

实现 bottleneck 与 transition compression：[结构依据](https://arxiv.org/abs/1608.06993)。

- 起始 3×3 conv 输出 24 通道，不做 ImageNet 大步长 stem。
- 每个 dense block 含 6 个 bottleneck。每个 bottleneck 为 BN-ReLU-1×1 conv 输出 48 通道，再 BN-ReLU-3×3 conv 输出 12 通道，与输入在通道维拼接。
- 第一块输出 96 通道；transition 以 1×1 conv 压缩到 48，再 2×2 均值池化。
- 第二块输出 120；transition 压缩到 60，再均值池化。
- 第三块输出 132；最后 BN-ReLU、全局均值池化、Linear(132,10)。
- 深度按初始卷积、36 个块内卷积、2 个 transition 卷积、1 个分类层计为 40。

两个 transition 都按 BN-ReLU-1×1 conv-2×2 AvgPool 实现。DenseNet 卷积不带 bias，BN 使用 affine，dense block 不额外加 dropout；全连接分类头保留 bias。将这些选项写入 architecture_config，避免不同实现都被同一个简称覆盖。

验收：三个模型输入 `[4,3,32,32]` 均输出有限的 `[4,10]` logits；参数量与各层 shape 自动打印。DenseNet 的拼接顺序、compression 向下取整和最后通道数需有独立检查。

## A.5 第四步：监督训练与 checkpoint 选择

本研究的初始训练配置为：200 epochs、batch 128、SGD、momentum 0.9、weight decay 5e-4、初始学习率 0.1、cosine 衰减至 0。每 epoch 完成训练更新后推进学习率调度；验证不更新模型。

执行顺序：

1. 先用固定 32 张训练图关闭增强进行小样本拟合，确认损失能降低、模型有能力拟合标签；该调试模型不用于正式结果。
2. 跑 1 个完整 epoch 检查设备、样本数、数据范围、学习率、loss 和验证流程。
3. 优先启动正式 ResNet-20，尽早向 B 提供可加载的初版 checkpoint；明确其不是最终结果版本。
4. 继续完成三个模型训练，每 epoch 保存验证 loss/accuracy、每类正确数和耗时。
5. 以完整验证集准确率选择 checkpoint；同分取较早 epoch。测试集不参与挑选。
6. 若初始学习率造成发散，按照预先限制的候选 0.05 在验证集重训并完整记录；不进行无界超参数搜索。
7. 正常验证性能明显不合理时，先检查输入归一化、标签、网络结构、scheduler 与训练模式，再扩展训练。必须区分训练失败与模型真的容易受攻击。

模型选择后，将权重与归一化参数打包，计算 checkpoint hash。向 C 提供两个源模型；DenseNet 的权重仅用于最终目标模型加载及正常训练核查，攻击配置开发不读取其攻击反馈。

## A.6 第五步：公共预测与冻结接口

实现 `src/model_bundle.py`：

```python
class ModelBundle:
    # 保存 architecture、state_dict、normalization 和 provenance
    def predict_logits(self, images_raw):
        # images_raw: float32, NCHW, [0,1]
        x = (images_raw - mean) / std
        return network(x)
```

攻击加载模式执行 `network.eval()` 并使参数 `requires_grad=False`，但 `predict_logits` 不加 `no_grad` 装饰器。最终纯评价可由调用方在 `no_grad` 下运行；训练阶段使用独立的可训练网络构建路径。

交接检查：B 对贴片调用 backward 后，贴片存在有限梯度；模型参数没有梯度；模型权重、BN running statistics 和其 hash 均未变化。C、D 使用同一包装器，不在外层再 Normalize。

## A.7 第六步：公共评价器及手算验收

实现 `src/metrics.py` 与 `scripts/evaluate_patch.py`，完整遵守统一协议第 6 节。先保存未防御正常预测，构造每个模型的固定 S_f 和每个目标类的 S_f,t，再评价贴片和防御。

最小手算用例：

```text
真实标签       [0,1,2,3]
正常预测       [0,1,0,3]
贴片预测       [1,1,2,0]
目标类别       1
S_f            索引 {0,1,3}，数量 3
S_f,t          索引 {0,3}，数量 2
CleanAcc       3/4
PatchedAcc     2/4
UASR           2/3
TASR           1/2
```

再检查：零合格样本、部分最后 batch、十类分母不同、一次与多次随机抽样、空防御、同一个样本重复记录、打乱 batch 顺序等情形。所有指标由原始正确数和分母计算，不对 batch 百分比简单等权平均。

随机评价按 `image_id、draw_id、eval_seed` 重建变换和防御参数；batch 从 64 改为 128 后，不应改变每图抽样条件。汇总时先对每个样本的八次抽样平均，再在相应集合上平均。

## A.8 第七步：完成随机贴片与遮挡对照

1. 调用 B 的同一渲染器，构造训练均值颜色贴片和固定随机纹理，四种尺寸与优化贴片完全匹配。
2. 对三个模型分别评价两种对照；随机纹理用 101/102/103 三个 seed。
3. 同一对照预测可以供十个目标类重新计算 TASR，无需为每个目标类重复渲染与前向。
4. 与 B 的优化贴片比较 UASR、TASR、全测试集准确率及有效覆盖面积。
5. 结合尺寸分析优化纹理的额外效应；大贴片同时存在强遮挡效应时，结论应明确分开。

不得用零纹理加正常遮罩代表“无贴片”。无贴片必须是 alpha 为零，或者直接返回原图。

## A.9 第八步：结果审计与统计入口

实现 `scripts/validate_artifacts.py` 与 `scripts/summarize_results.py`：

- 校验模型、协议、split、贴片 hash；检查源模型集合与 DenseNet held-out 约束。
- 按 `(phase,strategy,model,size,objective,target,seed,defense,budget)` 检查必需配置、重复项与缺失项。
- 检查每个结果是否包含正确的 image_id 全集和每图八次抽样；缓存与协议 hash 不匹配时拒绝复用。
- 从逐样本记录重算主表；核对保存的汇总值，差异超过浮点容差即停止汇总。
- 导出每个 seed 的结果、均值/标准差、配对差与条件 bootstrap 区间；百分率和百分点使用不同列名。

## A.10 文件交付、执行入口与完成标准

创建上述文件并通过单元验收后，在仓库根目录执行：

```bash
python -m scripts.prepare_data --config configs/protocol.yaml
python -m scripts.train_classifier --config configs/models/resnet20.yaml
python -m scripts.train_classifier --config configs/models/vgg11_bn_cifar.yaml
python -m scripts.train_classifier --config configs/models/densenet_bc40_k12.yaml
python -m scripts.evaluate_clean --config configs/protocol.yaml --models all
python -m scripts.run_controls --config configs/protocol.yaml
python -m scripts.validate_artifacts --config configs/protocol.yaml
python -m scripts.summarize_results --config configs/protocol.yaml
```

必须形成：

1. 模型规格及正常准确率表，逐类性能和训练曲线。
2. 三个可独立加载的最终 ModelBundle 与完整 provenance。
3. 数据划分及归一化清单。
4. 公共评价器的手算与随机性验收记录。
5. 四种尺寸下的恒色/随机对照结果。
6. 无缺失主配置、无混合模型版本的审计表。

A 报告部分：数据、模型、指标、对照与复现协议。答辩应解释分层划分、一次归一化、ASR 分母、随机抽样单位、正常基线质量与对照的含义。
