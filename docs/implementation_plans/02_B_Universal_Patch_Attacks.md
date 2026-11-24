# B 详细计划书：通用对抗贴片实现与基础实验

负责人：Russo Zhang（yz1047）。执行依据：[统一研究协议](00_Shared_Research_Protocol.md)。

## B.1 研究目标与交付边界

B 负责实现可微、可重复、可迁移复用的通用贴片生成流程，完成全部尺寸、全部目标类和非定向攻击，分析面积、图案和任务难度的关系。公共优化器同时服务 C 的集成优化和 D 的适配攻击。

基础覆盖为 4 尺寸 × 11 任务 × 3 seeds，共 132 份贴片。11 任务是 1 个非定向任务与 10 个不同目标类的定向任务；不是 11 个分类类别。ResNet 上的基础结果由 B 负责，迁移到 VGG/DenseNet 由 C 负责。

## B.2 第一步：读取模型契约并验证输入梯度

输入：A 的 raw loader、初步 ResNet ModelBundle、归一化参数和验证清单。

1. 加载模型，设置 `eval`，冻结所有参数，保存权重与 BN buffer 的副本/hash。
2. 对一批 raw 图检查 `[N,3,32,32]`、float32、`[0,1]`、标签范围 0—9。
3. 将输入副本设为可求梯度，计算真实标签交叉熵并回传；输入梯度必须存在且有限。
4. 确认 `model.eval()` 不阻止输入梯度；检查模型包装器没有 `no_grad` 或 `inference_mode`。
5. 与 A 的正常预测对照，确认没有重复 Normalize、错误 channel 顺序或图像缩放。

验收：输入梯度存在、输出有限、标签映射正确；模型参数无梯度，BN buffer 未变化。此门通过后再开发大规模贴片优化。

## B.3 第二步：先实现无变换的贴片放置

实现 `src/patches/render.py` 的第一条路径：

1. 创建叶子参数 `p:[1,3,s,s]`，用固定 seed 从 `[0,1]` 均匀初始化。
2. 给定左上位置 `(top,left)`，把同一个 p 应用于 batch 内每张图；允许每张图位置不同。
3. 用遮罩合成结果，保留未覆盖背景；返回 alpha、面积和位置记录。
4. 图像张量经合成仍与 p 保持计算图连接。不能转为 NumPy/PIL 再转回张量；不能在合成后 detach。

合成公式：

```text
x_patch = (1-alpha)*x + patch_premultiplied_on_canvas
```

固定位置检查：

- alpha=0 时，输出逐像素等于原图。
- scale=1、angle=0 的 s×s 不透明贴片，支持像素数恰为 s²。
- 贴片外像素与原图相同，贴片内数值与 p 相同。
- 同一 p 被不同图片复用；梯度汇总到同一个叶子参数，而非每张图的独立参数。
- 更换 batch 划分但保持样本和位置不变，逐样本结果一致。

## B.4 第三步：加入旋转、缩放与随机位置

为每个样本生成固定记录的变换参数。主分布为 angle∈[-15°,15°]、scale∈[0.9,1.1]、合法位置均匀抽样。

实现顺序：

1. 在统一像素中心约定下定义仿射映射；明确 `grid_sample` 需要的是输出位置到输入位置的逆映射。
2. 将变换映射到归一化采样坐标，统一使用 `align_corners=False`；若同时使用 interpolate，保持同一约定。
3. 对预乘纹理与全一局部遮罩使用同一网格、相同双线性插值与零外部填充，得到画布纹理和 alpha。
4. 依据变换后的完整支持范围抽样平移，包含插值边缘；不能只限制中心而让边缘越界。
5. 把每图参数保存为可重建的 `TransformBatch`。使用一次渲染的输出供全部源模型共享。
6. 返回 `sum(alpha)` 与 `count(alpha>1e-6)`，检查变换前后的面积分布；对照组必须使用同一 alpha。

边界验收覆盖四个角、中心、最小/最大缩放和两个旋转端点。渲染器不得产生 NaN 或超范围颜色；完整 patch 支持不因裁剪消失。

在 CPU 双精度、固定变换、贴片像素远离 clamp 边界时，对若干非零梯度方向做有限差分或 `gradcheck`；与自动微分比较。随机变换在正负扰动的两次计算之间必须固定。失败时先检查坐标方向、单位、alpha 预乘、detach 和参数叶子状态。

若正式设备不支持仿射采样后向，按同一坐标与双线性规则实现四邻点 gather 版本，验证前向和贴片梯度与 CPU 参考一致；不改变主变换分布。

## B.5 第四步：实现统一优化循环

实现 `src/patches/optimize.py`。接口接受源模型列表、源权重、目标定义、渲染器、防御函数、EOT 次数和预算。C、D 只传入不同配置与损失组合，不复制整套优化逻辑。

核心循环的实现形式：

```python
# source_bundles 的 predict_logits 接受 raw 像素；模型已 eval 且参数冻结。
# render、sample_transforms、defense、next_train_batch 按公共契约实现。
patch = torch.nn.Parameter(initial_patch.to(device))
optimizer = torch.optim.Adam([patch], lr=learning_rate, weight_decay=0.0)

for step in range(max_steps):
    images_raw, labels, image_ids = next_train_batch()
    optimizer.zero_grad(set_to_none=True)
    for k in range(eot_samples):
        transforms = sample_transforms(image_ids, step, k, train_seed)
        patched_raw, alpha, meta = render(images_raw, patch, transforms)
        defended_raw = defense(patched_raw, step=step, draw=k)
        loss = 0.0
        for bundle, weight in zip(source_bundles, source_weights):
            logits = bundle.predict_logits(defended_raw)
            if objective == "targeted":
                attack_labels = torch.full_like(labels, target_class)
                term = torch.nn.functional.cross_entropy(logits, attack_labels)
            else:
                term = -torch.nn.functional.cross_entropy(logits, labels)
            loss = loss + weight * term
        (loss / eot_samples).backward()
    optimizer.step()
    with torch.no_grad():
        patch.clamp_(0.0, 1.0)
    # 按固定验证频率保存候选、分数、用时与元数据。
```

非定向采用负交叉熵后最小化，不再额外翻转更新方向。只优化 `[patch]`；不能把分类器参数误加入 optimizer。每个 EOT 抽样独立构建图并累积平均梯度，避免无意保留全部历史图。

禁止使用 `loss.item()` 作为反传损失，禁止 `torch.tensor(existing_loss)` 重建并切断计算图，禁止在优化循环外复用已经 backward 的旧图。

## B.6 第五步：完成一个可交接的小实验

先运行：ResNet、7×7、目标类 3、seed 101、batch 128、最多 1,000 更新；同时运行非定向版本。

记录 step 0、每 100 步、step 1000 的训练目标、固定验证 ASR、全验证准确率摘要、贴片图和用时。正向指标应与目标方向一致；训练 loss 降低不等于未见图片攻击成功。

交给 C、D 的第一份材料：

```text
模型 checkpoint 与规格/hash
共享优化器和渲染器版本
7×7 贴片张量与 PNG 预览
目标类、优化 seed、选中 step
数据划分、验证子集、变换与归一化配置
验证的逐样本结果及一条可复现命令
```

验收依赖正确性与完整记录，不要求某个任意高 ASR。若结果弱，先分别做固定位置、无变换、小批量训练图诊断，再恢复完整变换检查泛化。小尺寸贴片弱或某些目标类难，不能仅因此判为程序错误。

## B.7 第六步：配置冻结与完整基础矩阵

1. 用有限的验证试验固定学习率、1,000 步主预算、batch、变换分布和 checkpoint 选择规则。
2. 构造机器可读取任务表：每行唯一指定尺寸、目标或 untargeted、seed、源模型 hash、协议 hash。
3. 为 4×11×3=132 个配置创建唯一 run_id；检查同一任务无重复。
4. 按批运行并交接：优先完成 7×7 与 5×5，以解锁 C、D；随后补齐 3×3、16×16，不删去任一目标类。
5. 每个任务保存 best-validation patch 和 last-step patch，主表统一使用预定 best-validation 规则；不在测试集二选一。
6. 支持断点续跑：同时恢复 patch、Adam 状态、step、训练采样状态及选择历史；仅恢复纹理重新初始化 Adam 不等价于续跑。
7. 对每个成功任务保存结束标记与文件 hash；中断、NaN、超时和缺失分开记录，不能默认为 ASR=0。

各任务的 patch seed 表示独立优化重复，包含初始化、图片抽样和 EOT 随机性。相同 seed 的不同源策略共享可配对的训练图片与变换序列；验证使用单独 RNG，不改变后续训练抽样。

## B.8 第七步：最终评价与基础分析

在 G4 冻结后，对全部贴片使用官方测试集和每图八次抽样，调用 A 的公共评价器。不能看到测试结果后再改学习率、重新选择 seed 或替换特定目标类。

必须形成：

1. 四种尺寸的非定向 ASR 与全测试集攻击后准确率，均含三 seed 结果。
2. 四种尺寸×十个目标类的 TASR 完整表与热力图；报告每类合格分母。
3. 各尺寸十目标类宏平均 TASR，并与非定向任务分别展示。
4. 与同面积随机纹理、训练均值颜色遮挡的差异。
5. 贴片 alpha 面积与支持面积分布，区分名义尺寸和实际覆盖。
6. 代表性成功与失败图：按预先固定的样本编号或系统抽样规则选图，避免只挑成功样例。

图案分析可以检查通道均值、颜色直方图、空间总变差和频谱能量。只有测量了频率特征，才讨论“高频纹理”；单凭肉眼不能证明模型使用了某个特征。不同目标类图案相似性是观察，不直接等同于语义解释。

将相同贴片的定向命中与非定向错误关系，与不同任务分别优化后的性能关系区分。不能断言所有设置下定向 ASR 必然低于另一个独立优化的非定向 ASR。

## B.9 第八步：优化充分性与失败定位

| 现象 | 优先检查 | 必须保存的证据 |
|---|---|---|
| p.grad 为 None | detach、NumPy 转换、no_grad、参数是否为 optimizer 中的叶子 | 固定样本的计算图与梯度范数 |
| 定向 loss 持续上升 | 最小化方向、目标标签、是否重复取负 | step 0—20 的 loss 与目标概率 |
| 模型正常准确率变了 | BN/dropout、误更新参数、归一化 | 前后权重和 buffer hash |
| train 有效而 val 无效 | 同一贴片是否跨图、验证样本是否未参与优化、训练分布 | train/val 相同条件的 ASR |
| 边缘位置有效面积变小 | 合法中心区间、旋转外接范围、插值边界 | alpha 与支持图 |
| 尺寸增大后结果反而弱 | 优化收敛、面积协议、目标类分布、重复波动 | 每个 seed 的曲线与对照 |
| 测试 ASR 很高但随机对照也高 | 分类器质量、遮挡效应、基准分母 | clean 表及配对对照 |

强度诊断只在固定验证配置开发并预定到最终检验，不针对测试中最差的任务临时补预算。若修复程序错误，重跑全部受影响配置并更新 hash，不能仅修补更有利的一部分。

## B.10 执行入口与完成标准

创建并验收对应文件后执行：

```bash
python -m scripts.optimize_patch --config configs/attacks/basic_targeted.yaml --size 7 --target 3 --seed 101
python -m scripts.optimize_patch --config configs/attacks/basic_untargeted.yaml --size 7 --seed 101
python -m scripts.run_matrix --phase basic --config configs/protocol.yaml
python -m scripts.evaluate_patch --phase basic --model resnet20 --config configs/protocol.yaml
python -m scripts.validate_artifacts --phase basic --config configs/protocol.yaml
```

验收清单：

- [ ] 渲染前向、背景保留、仿射和梯度检查通过。
- [ ] 全程模型参数与统计量冻结。
- [ ] 132 份主贴片及其 best/last、日志、配置和 hash 完整。
- [ ] 四尺寸、十目标类及非定向任务全部覆盖。
- [ ] ResNet 主结果、随机/遮挡对照、面积统计和失败样例齐全。
- [ ] C、D 能直接加载相同贴片并重算结果。
- [ ] 所有图表源于逐样本文件，三次 seed 的波动可追溯。

B 报告部分：通用贴片目标、EOT 与渲染、基础尺寸/目标类结果、图案和遮挡分析。个人问答必须解释通用与逐图攻击的差别、非定向损失符号、模型冻结为何仍保留梯度、以及 ASR 的统计对象。
