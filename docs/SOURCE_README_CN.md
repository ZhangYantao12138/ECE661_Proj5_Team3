# ECE661 Team 3：真实实验完成版

本目录包含完整实现、已完成的三模型训练、561 份主贴片和 24 份强度贴片的重现入口。全部 1,212 个完整验证条件和 1,638 个测试条件经逐样本审计；所有结果表均从原始预测重算。

resnet20: 最佳验证 92.78%，最终正常测试 92.29%，模型 SHA-256=88c98d6ce875673806386527c75ae17e8ffd795c00413155c360468812f3dacc
vgg11_bn_cifar: 最佳验证 92.38%，最终正常测试 91.99%，模型 SHA-256=2d68f337912f570179fdba4b8a9f65787e7ce72a770cc72b752afb95f5600e0a
densenet_bc40_k12: 最佳验证 92.56%，最终正常测试 92.36%，模型 SHA-256=6c7e8c3b90188a6ea44de160c9407d4adce50ba4be1997b209603e229ca9ff51

A/B/C/D 依赖顺序及全部命令见交付根目录 RUN_ORDER_CN.txt。A 负责数据/模型/评价，B 负责共享渲染优化与基础矩阵，C 复用 B 并完成两种预算下的迁移，D 复用 B 并完成防御、同 EOT 对照与强度检查。

测试集只在候选、权重和配置冻结后读取预测。DenseNet 正式 checkpoint 从未作为优化源或用于攻击选择；早期对随机初始化架构的兼容性检查不是正式 held-out 攻击反馈。

configs/protocol.json 保留实验使用时的不可变原始内容及 hash；其中历史 status 文本不作为完成判据。真实完成判据是 training_complete.json、validation_audit.json、evaluation_audit.json、analysis.json 和最终 SHA256_MANIFEST.json。不得为修改这类显示文本而破坏冻结身份。

计算成本使用 CUDA 同步的实测训练墙钟；独立 GPU kernel 活跃时间未另行测量，不以估算代替。预热、验证及端到端成本分别保留。矩阵开始时曾因检查与标定重叠而重做空闲 GPU 标定，已完成的贴片保留，唯一未完成运行单独归档，详见 results/time_calibration_repair.json。

复算：python -m scripts.audit_results；python -m scripts.analyze_results；python -m scripts.export_visual_assets；python -m scripts.publication_figures。报告与说明生成脚本同时提供。完整训练和实验入口仍是 python -m scripts.local_gpu。已有正式产物时请先读运行说明，避免覆盖冻结权重或无意更换时间预算。

数据、种子、归一化、张量精度和后端版本见 artifacts。GPU 部分算子可能非确定，跨硬件/版本不承诺逐 bit 相同；提供的权重、贴片与原始预测足以重算本文所有指标。

文稿生成工具的实测版本见 requirements-docs-tested.txt。CIFAR10_Dataset.zip 含已校验的官方原始归档，可离线准备数据。历史 end_to_end_seconds 字段从优化函数入口计到最终产物组装，外部模型/数据加载及最终序列化未计入，不等于整段程序的耗时。

曾隔离 GPU 检查 EOT 合批候选；真实模型损失与梯度未通过预设等价容差，候选未采用，正式 D 仍使用逐次 EOT。results/eot_batching_check.json 和 eot_batching_decision.json 保留证据；仅暂停时一份未完成任务被归档后从原始配置重新开始，已完成任务未重做。

D 适配阶段后期记录到交互图形进程共享 GPU，训练用时随之增长；详见 results/resource_observations.json。固定更新/EOT 预算未减少，D 时间是当时资源条件下的观测值，不能单独解释为防御计算开销。C 的等时间组在该图形进程启动之前已完成。
