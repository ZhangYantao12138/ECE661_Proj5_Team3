# Windows / RTX 5080：完成版复现说明

正式交付已包含三个完成 200 epochs 的模型、561 份主贴片、24 份强度贴片、1,212 个完整验证条件和 1,638 个完整测试条件。报告和四份中文说明均由这些实际记录生成。历史原始包位于 Original_Inputs，不能把其中旧草稿当作最终报告。

本次计算使用 Windows、RTX 5080 16GB、Python 3.13.5、PyTorch 2.10.0.dev20251010+cu130、CUDA runtime 13.0；具体后端标志见工件内环境记录。requirements-local-tested.txt 记录计算环境；requirements-docs-tested.txt 记录独立文稿环境。优先复用已通过检查的安装。浮点后端可能非确定，跨硬件/版本不承诺逐 bit 相同。

## 合并与复用

按 A_Code.zip → B_Code.zip → C_Code.zip → D_Code.zip 合并到同一个 ECE661_Project；再合并 Shared_Results.zip 和 CIFAR10_Dataset.zip。相同共享源码可以覆盖合并。A 包含模型，B 包含基础贴片，C/D 引用 A/B 并提供各自新增贴片。

已有冻结产物时，只核验和解压数据，不重建 split 或重标定预算：

```powershell
python -c "from src.data import prepare; prepare('data')"
python -m scripts.audit_results
python -m scripts.analyze_results
python -m scripts.publication_figures
```

上述统计和图表使用已提供的真实逐图片记录；analyze_results 仅需 CPU 加载贴片元数据，publication_figures 在单独进程中绘图。若要重新生成案例图像，再运行 scripts.export_visual_assets（该入口使用 CUDA）。

## 从头运行

在另一份干净项目中复制源码、配置、测试、模板和数据，不复制已有 artifacts/results。模型和贴片不得覆盖本次冻结工件。

```powershell
python -m scripts.local_gpu check
python -m scripts.local_gpu prepare
python -m scripts.local_gpu train
python -m scripts.local_gpu calibrate
python -m scripts.local_gpu attack
python -m scripts.local_gpu final
python -m scripts.analyze_results
python -m scripts.export_visual_assets
python -m scripts.publication_figures
```

train 自动完成三模型的 32 张真实训练图拟合检查后，运行三次正式 200-epoch SGD；调试拟合使用独立 Adam，调试权重不用于正式评价。calibrate 完成有限候选开发、正常验证防御选择、空闲 GPU 时间标定。attack 包含全部六个优化阶段和强度检查。final 先审计、完整源验证和冻结，再开放正式测试与 held-out 攻击评价。

必须保持同一时间比较的 GPU 独占；预热使用独立状态，验证暂停训练计时。D 始终采用原逐次 EOT；合批候选未通过数值容差，未用于正式结果。三个贴片 seed 不是三个分类器训练重复。

## 恢复与日志

Run_Local_GPU.ps1 是同一分阶段命令的 PowerShell 包装，可用 -Python 指定解释器完整路径。results/local_logs、results/execution_status.json、results/progress_history.jsonl 及各 patch.pt/resume.pt 保留进度。失败后先修复对应日志中的错误，再恢复同一阶段；已完成贴片会核对身份后跳过，未完成任务恢复 Adam、随机流和验证历史。

已有正式权重时不要再次运行 prepare/calibrate 来替换 split、防御或 T。最终评价按条件缓存并验证 hash，缺失条件可继续；不是从一份评价文件内部的某个图片续跑。原始记录、模型、贴片、预算或配置发生变化时，必须重跑受影响条件并建立新的冻结清单。

## 文稿与证据

正式报告源码和原始 NeurIPS 2026 样式在 Report_Source，可用 Tectonic 编译。scripts/build_chinese_guides.py 使用 reportlab 和安装的中文 TrueType 字体，ECE661_CJK_FONT 可覆盖字体路径；scripts/build_poster.py 填充随包课程模板。Windows 默认字体为 SimHei/Arial，不分发系统字体文件。

图片 bootstrap 以整张图为单位保留所有随机抽样；均值与样本标准差基于三个贴片 seed。原始 NPZ 与相邻 JSON 可以独立重算指标。freeze.json 绑定模型、贴片及开发配置；最终 SHA256_MANIFEST.json 绑定正式文稿与结果包。

计时是 CUDA 同步训练墙钟，不是单独的 GPU kernel 活跃时间。end_to_end_seconds 是优化函数入口到最终工件组装的历史字段，外部模型/数据加载和最终序列化未计入。数字结果不构成物理实验或认证鲁棒性证明。
