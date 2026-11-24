"""Truthful final run notes, source provenance, presentation timing, and Q&A."""
import hashlib
import json
from pathlib import Path
from src.paths import delivery_dir

ROOT=Path(__file__).resolve().parents[1]
FINAL=delivery_dir()


def main():
    data=json.loads((ROOT/'results/publication_data.json').read_text(encoding='utf-8'))
    if data['status']!='complete':raise RuntimeError('Measured final results are required')
    FINAL.mkdir(parents=True,exist_ok=True)
    model_lines='\n'.join(f"{r['name']}: 最佳验证 {100*r['best_validation_accuracy']:.2f}%，最终正常测试 {100*r['clean_accuracy']:.2f}%，模型 SHA-256={r['checkpoint_sha256']}" for r in data['models'])
    readme=f'''# ECE661 Team 3：真实实验完成版\n\n本目录包含完整实现、已完成的三模型训练、561 份主贴片和 24 份强度贴片的重现入口。全部 1,212 个完整验证条件和 1,638 个测试条件经逐样本审计；所有结果表均从原始预测重算。\n\n{model_lines}\n\nA/B/C/D 依赖顺序及全部命令见交付根目录 RUN_ORDER_CN.txt。A 负责数据/模型/评价，B 负责共享渲染优化与基础矩阵，C 复用 B 并完成两种预算下的迁移，D 复用 B 并完成防御、同 EOT 对照与强度检查。\n\n测试集只在候选、权重和配置冻结后读取预测。DenseNet 正式 checkpoint 从未作为优化源或用于攻击选择；早期对随机初始化架构的兼容性检查不是正式 held-out 攻击反馈。\n\nconfigs/protocol.json 保留实验使用时的不可变原始内容及 hash；其中历史 status 文本不作为完成判据。真实完成判据是 training_complete.json、validation_audit.json、evaluation_audit.json、analysis.json 和最终 SHA256_MANIFEST.json。不得为修改这类显示文本而破坏冻结身份。\n\n计算成本使用 CUDA 同步的实测训练墙钟；独立 GPU kernel 活跃时间未另行测量，不以估算代替。预热、验证及端到端成本分别保留。矩阵开始时曾因检查与标定重叠而重做空闲 GPU 标定，已完成的贴片保留，唯一未完成运行单独归档，详见 results/time_calibration_repair.json。\n\n复算：python -m scripts.audit_results；python -m scripts.analyze_results；python -m scripts.export_visual_assets；python -m scripts.publication_figures。报告与说明生成脚本同时提供。完整训练和实验入口仍是 python -m scripts.local_gpu。已有正式产物时请先读运行说明，避免覆盖冻结权重或无意更换时间预算。\n\n数据、种子、归一化、张量精度和后端版本见 artifacts。GPU 部分算子可能非确定，跨硬件/版本不承诺逐 bit 相同；提供的权重、贴片与原始预测足以重算本文所有指标。\n'''
    readme+='\n文稿生成工具的实测版本见 requirements-docs-tested.txt。CIFAR10_Dataset.zip 含已校验的官方原始归档，可离线准备数据。历史 end_to_end_seconds 字段从优化函数入口计到最终产物组装，外部模型/数据加载及最终序列化未计入，不等于整段程序的耗时。\n'
    readme+='\n曾隔离 GPU 检查 EOT 合批候选；真实模型损失与梯度未通过预设等价容差，候选未采用，正式 D 仍使用逐次 EOT。results/eot_batching_check.json 和 eot_batching_decision.json 保留证据；仅暂停时一份未完成任务被归档后从原始配置重新开始，已完成任务未重做。\n'
    if data.get('resource_observations'):
        readme+='\nD 适配阶段后期记录到交互图形进程共享 GPU，训练用时随之增长；详见 results/resource_observations.json。固定更新/EOT 预算未减少，D 时间是当时资源条件下的观测值，不能单独解释为防御计算开销。C 的等时间组在该图形进程启动之前已完成。\n'
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    local_guide=r'''# Windows / RTX 5080：完成版复现说明

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
'''
    (ROOT/'LOCAL_GPU_CN.md').write_text(local_guide,encoding='utf-8')
    provenance={}
    for folder in ['src','scripts','configs','tests']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                provenance[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'artifacts/code_provenance.json').write_text(json.dumps(dict(scope='final delivered source; experimental weights/patches bound separately by frozen hashes',files=provenance),indent=2),encoding='utf-8')
    coverage={
        'R01':dict(requirement='Shared patch generation',implementation=['src/patches/render.py','src/patches/optimize.py'],evidence=['artifacts/patches','results/validation_audit.json']),
        'R02':dict(requirement='Untargeted sides 3/5/7/16',evidence=['results/figures/B_size_and_targets.pdf','results/tables/seed_summaries.csv']),
        'R03':dict(requirement='All ten targeted classes at each size',evidence=['results/tables/per_seed_metrics.csv','Supplementary_Results.pdf']),
        'R04':dict(requirement='Textures and fixed success/failure cases',evidence=['results/figures/B_patch_gallery.pdf','results/figures/B_fixed_examples.pdf','results/tables/patch_characteristics.csv']),
        'R05':dict(requirement='Unmodified transfer to VGG and DenseNet',evidence=['results/tables/seed_summaries.csv','results/evaluation_audit.json']),
        'R06':dict(requirement='Two-source extension under step/time budgets',evidence=['results/figures/C_heldout_comparison.pdf','results/figures/C_targets_and_sources.pdf','results/figures/C_validation_time_curves.pdf','results/tables/paired_effects.csv']),
        'R07':dict(requirement='Defense cost, nonadaptive/adaptive and EOT-matched controls',evidence=['results/figures/D_three_paths.pdf','results/figures/D_all_targets.pdf','results/figures/D_cost_tradeoff.pdf','results/figures/D_strength_checks.pdf']),
        'R08':dict(requirement='Two extensions and planned four-person responsibilities',evidence=['Formal_Report.pdf','PRESENTATION_AND_QA_CN.txt'],scope='Planned A/B/C/D roles are documented. Actual individual effort, attendance and presentation cannot be verified by generated artifacts.'),
        'R09':dict(requirement='Three independent patch seeds',evidence=['artifacts/matrix.json','results/tables/target_macros_per_seed.csv','results/tables/paired_effects.csv']),
    }
    (FINAL/'REQUIREMENTS_COVERAGE.json').write_text(json.dumps(dict(requirements=coverage,coverage=data['coverage'],remaining_student_activities=['poster printing and scheduled presentation','individual Q&A and course peer review'],measurement_limitations=['No independent classifier-training repeats','No separately measured GPU-kernel-active time','No physical or certified robustness experiment']),ensure_ascii=False,indent=2),encoding='utf-8')
    b7=next(r for r in data['basic'] if r['size']==7)
    c7=next(r for r in data['paired_effects'] if r['part']=='C' and r['phase']=='ensemble_time' and r['size']==7 and r['objective']=='target_macro' and r['comparison']=='RV - R')
    notes=f'''ECE661 六分钟展示与个人问答准备\n\n本材料是准备提纲，不声称成员已完成现场展示、打印或互评。\n\n0:00—0:45 A：问题、通用贴片与三个研究问题。\n0:45—1:30 A：固定划分、模型正常性能、ASR 分母。\n1:30—2:00 B：预乘 alpha、EOT 与正负交叉熵。\n2:00—3:00 B：尺寸、十目标、随机与遮挡对照。\n3:00—4:00 C：同一 DenseNet 上的 R/V/RV 与等步数/等时间。\n4:00—4:30 C：统计单位、条件推断和迁移局限。\n4:30—5:45 D：正常代价、原 K1/未知 K4/适配 K4、强度诊断。\n5:45—6:00 D：范围明确的结论。\n\n三个模型：\n{model_lines}\n\n可直接报出的真实数值：\nResNet 7×7 UASR={100*b7['untargeted']['mean']:.2f}%，三 seed 样本标准差={100*b7['untargeted']['sample_std']:.2f} 个百分点。\n7×7 等时间下，DenseNet 目标宏平均 RV−R={c7['delta_pp']:+.3f} pp，按图片分层 bootstrap 95% 区间 [{c7['ci95_pp'][0]:+.3f}, {c7['ci95_pp'][1]:+.3f}]；不要将区间说成模型训练随机性区间。\n\nA 常见问题\n问：为什么 ASR 不等于 1−全测试集 PatchedAcc？\n答：前者只在正常正确集合上计算，后者分母是全部测试图；定向还排除真类为目标。\n问：为什么不能把每图八次变换当作八张独立图片？\n答：同一背景和标签造成相关性，重抽时保留整张图的所有抽样。\n问：三 seed 能代表三次模型训练吗？\n答：不能；每个分类器只用 seed 17 训练一次。\n\nB 常见问题\n问：model.eval() 为何仍可得到贴片梯度？\n答：eval 只切换 BN/dropout 行为；requires_grad=False 冻结参数，但不能把攻击前向放进 no_grad。\n问：非定向方向如何避免写反？\n答：本实现最小化负真实标签 CE；定向最小化目标 CE。\n问：为什么采样后的颜色不能再乘 alpha？\n答：颜色已经预乘并被插值衰减，再乘会把边界衰减平方。\n\nC 常见问题\n问：为什么平均 logits 不能替换平均损失？\n答：CE 非线性，二者目标和梯度不同；主协议固定分别求 CE 后平均。\n问：为何集成固定步数更好不一定说明迁移策略更好？\n答：每步多算一个模型；还要看等时间结果、实际步数及调用量。\n问：DenseNet 的隔离如何验证？\n答：优化器拒绝该源名，源模型 hash/名单保存并审计；正式目标攻击反馈只在冻结后读取。随机初始化架构的算子兼容性测试不参与配置选择。\n\nD 常见问题\n问：为什么要未知防御 K4？\n答：否则把多采样的计算收益混入知道防御的收益。\n问：为什么还要报告 NetErrorIncrease？\n答：防御可能自己制造正常错误；用配对正常防御预测扣除该部分。\n问：高斯防御不随机，为什么也用 EOT？\n答：贴片的旋转、缩放和位置仍随机。\n问：适配成功率没有提高是否能删掉该 seed？\n答：不能；保留正负结果，结合更长迭代、K8 和抽样稳定性限制结论。\n\n展示前由成员自行核对课程安排、打印海报、熟悉各自代码，并准备解释任何一个图表的原始文件与分母。不要把 AI 代写说成未经核实的个人独立贡献。\n'''
    (FINAL/'PRESENTATION_AND_QA_CN.txt').write_text(notes,encoding='utf-8')
    print('Final README, source provenance, and measured Q&A notes saved',flush=True)


if __name__=='__main__':main()
