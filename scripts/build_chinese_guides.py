"""Illustrated Chinese reproducibility guides populated only by audited real results."""
import json
import math
import os
import statistics
from html import escape
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from src.paths import delivery_dir
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PROJECT=Path(__file__).resolve().parents[1]
FINAL=delivery_dir()
FIG=PROJECT/'results/figures'
TEAL=colors.HexColor('#226c78')


class Pipeline(Flowable):
    def __init__(self,labels):Flowable.__init__(self);self.labels=labels;self.width=499;self.height=82
    def draw(self):
        c=self.canv;gap=17;box=(self.width-gap*(len(self.labels)-1))/len(self.labels)
        for i,label in enumerate(self.labels):
            x=i*(box+gap);c.setFillColor(colors.HexColor('#edf4f5'));c.setStrokeColor(TEAL);c.roundRect(x,12,box,62,5,fill=1)
            lines=label.split('\n');c.setFillColor(colors.HexColor('#163e48'));c.setFont('CJK',9)
            for j,line in enumerate(lines):c.drawCentredString(x+box/2,50+(len(lines)-2)*6-j*13,line)
            if i<len(self.labels)-1:
                c.line(x+box+2,43,x+box+gap-4,43);path=c.beginPath();path.moveTo(x+box+gap-4,43);path.lineTo(x+box+gap-9,46);path.lineTo(x+box+gap-9,40);path.close();c.drawPath(path,fill=1,stroke=0)


def main():
    data=json.loads((PROJECT/'results/publication_data.json').read_text())
    audit=json.loads((PROJECT/'results/evaluation_audit.json').read_text())
    if data['status']!='complete' or not audit['complete']:raise RuntimeError('Guides require complete measured and audited final results')
    font=Path(os.environ.get('ECE661_CJK_FONT',r'C:\Windows\Fonts\simhei.ttf'))
    if not font.exists():raise RuntimeError('Set ECE661_CJK_FONT to an installed Chinese TrueType font')
    pdfmetrics.registerFont(TTFont('CJK',str(font)))
    styles={
      'title':ParagraphStyle('title',fontName='CJK',fontSize=25,leading=34,textColor=colors.HexColor('#153e49'),spaceAfter=14),
      'h':ParagraphStyle('h',fontName='CJK',fontSize=15,leading=22,textColor=TEAL,spaceBefore=12,spaceAfter=9,keepWithNext=True),
      'p':ParagraphStyle('p',fontName='CJK',fontSize=10.5,leading=17,spaceAfter=10,wordWrap='CJK'),
      'small':ParagraphStyle('small',fontName='CJK',fontSize=8.5,leading=13,spaceAfter=7,wordWrap='CJK'),
      'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8.7,leading=13,wordWrap='CJK'),
      'code':ParagraphStyle('code',fontName='Courier',fontSize=8.1,leading=12,spaceBefore=9,spaceAfter=12,wordWrap='CJK',backColor=colors.HexColor('#f2f5f6'),borderPadding=7),
    }
    def P(text,style='p'):
        value=str(text).translate(str.maketrans({'−':'-','–':'-','—':'-','‑':'-'}))
        if style!='code':
            missing={c for c in value if not c.isspace() and ord(c) not in pdfmetrics.getFont('CJK').face.charToGlyph}
            if missing:raise ValueError('Chinese font lacks required glyphs: '+repr(missing))
        return Paragraph(escape(value).replace('\n','<br/>'),styles[style])
    def pct(x):return f'{100*x:.2f}%'
    def ms(r):return f"{100*r['mean']:.2f} ± {100*r['sample_std']:.2f}%"
    def effect(r):return f"{r['delta_pp']:+.3f} pp；95% 区间 [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}]"
    def get(rows,**keys):return next(r for r in rows if all(r.get(k)==v for k,v in keys.items()))
    def table(rows,widths):
        t=Table([[P(v,'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f0f2')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#ccd8dc')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7)]))
        return t
    def image(name,max_height=260):
        path=FIG/(name+'.png');picture=Image(str(path));factor=min(499/picture.imageWidth,max_height/picture.imageHeight)
        picture.drawWidth=picture.imageWidth*factor;picture.drawHeight=picture.imageHeight*factor;picture.hAlign='CENTER';return picture
    def page(story,title):
        if story:story.append(PageBreak())
        story.append(P(title,'h'))
    def add(story,*texts):story.extend(P(t) for t in texts)
    def fig(story,name,caption,height=260):story.extend([image(name,height),Spacer(1,5),P(caption,'small')])
    def build(letter,title,story):
        def footer(c,doc):
            c.setFont('CJK',8);c.setFillColor(colors.HexColor('#607780'));c.drawString(48,27,f'ECE661 Team 3 / {letter} / 真实实验与可复现实现');c.drawRightString(547,27,str(doc.page))
        path=FINAL/('Supplementary_Results.pdf' if letter=='Supplement' else f'{letter}_Chinese_Guide.pdf')
        SimpleDocTemplate(str(path),title=title,author='ECE661 Team 3',pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
        print(path,flush=True)
    FINAL.mkdir(parents=True,exist_ok=True)
    # A: complete experimental foundation.
    s=[P('A｜数据、模型与公共评价','title')]
    add(s,'这份说明从数据流开始，解释三个分类器怎样训练、攻击成功率怎样计算、正常基线与匹配遮挡对照怎样复现。阅读者需要理解 Python、张量和训练/验证/测试集的基本概念。','本文中的准确率、曲线与对照均来自本次实际运行。三个模型均完成 200 轮训练；贴片主矩阵 561 份，强度检查另有 24 份，最终 1,638 个评价条件文件通过逐样本审计。')
    s.append(Pipeline(['官方 CIFAR-10\n校验与固定索引','45,000 训练\n5,000 验证','训练三个分类器\n只按验证选权重','冻结后评价\n10,000 测试']))
    s.append(table([['模型','参数量','最佳验证准确率','正常测试准确率'],*[[r['name'],f"{r['parameters']:,}",pct(r['best_validation_accuracy']),pct(r['clean_accuracy'])] for r in data['models']]],[170,90,115,124]))
    add(s,'验证准确率用于选 checkpoint，测试准确率用于最终评价；两者不能互换。三次贴片优化 seed 不是三个独立训练的分类器，每个架构只有一次 seed=17 的监督训练。')
    page(s,'1. 数据划分与归一化为什么必须固定')
    add(s,'官方压缩包 MD5 为 c58f30108f718f92721af3b95e74349a。每类 5,000 张官方训练图按 PCG64(661) 置乱，先取 500 张验证，其余 4,500 张训练。测试集保持官方 10,000 张原顺序，每类 1,000 张。','贴片选优只使用验证集中的每类 100 张，共 1,000 张，采用独立 PCG64(662)。训练集与验证集没有交集；选优子集属于验证集，不能称作额外独立数据。','均值和总体标准差仅用 45,000 张训练图的所有像素计算，使用 float64 累积和与平方和。整体标准差不等于“各图片标准差的平均值”。')
    s.append(Pipeline(['raw 像素\nfloat32 [0,1]','可选贴片\n保留背景','可选输入防御\n仍是 raw','ModelBundle\n只归一化一次']))
    add(s,'监督训练增强为 padding 4、32×32 随机裁剪和概率 0.5 的水平翻转；贴片训练背景、验证和测试不做这些增强。image_id 保留官方 split 和原索引，防止不同数据流的编号混用。','运行时把未增强 raw 张量缓存到 GPU，并缓存纯函数生成的评价变换参数。这些缓存不改变图片、标签、采样 seed 或变换；包含重复索引的 CPU/CUDA 批量值已与原逐图读取精确核对。')
    page(s,'2. 三个模型及其真实训练过程')
    add(s,'ResNet-20 使用通道 16/32/64、每阶段三个残差块，降采样的 Option A shortcut 只做空间采样和通道补零，可训练参数为 269,722。VGG 使用八个卷积层、五次池化，以及 512→512→512→10 的三层分类头。DenseNet-BC-40-12 含三个各六层 bottleneck 的 dense block，growth=12、压缩率 0.5，最后 132 通道。','三者采用 batch 128、SGD、初始学习率 0.1、momentum 0.9、weight decay 5e-4、cosine 衰减和 200 epochs。每轮只用完整验证集选择最佳权重，同分取更早的轮次。先用 32 张真实训练图完成拟合检查；该调试模型从未用于正式指标。')
    fig(s,'A_training_curves','实际 200 轮训练曲线。正常验证波动完整保留；图上没有使用测试集来早停或调参。',300)
    page(s,'3. 手算 ASR、分母和随机抽样单位')
    add(s,'设真实标签 [0,1,2,3]，正常预测 [0,1,0,3]，贴片预测 [1,1,2,0]，目标类为 1。正常预测正确集合 S_f={0,1,3}，目标攻击合格集合再排除真类为 1，得到 {0,3}。')
    s.append(table([['指标','分子 / 分母','手算结果'],['CleanAcc','正常正确 3 / 全部图片 4','3/4'],['PatchedAcc','贴片后正确 2 / 全部图片 4','2/4'],['UASR','原本正确且贴片后错误 2 / 原本正确 3','2/3'],['TASR','合格样本中命中目标 1 / 合格样本 2','1/2']],[130,270,99]))
    add(s,'正式评价每图 8 次变换/防御抽样，先保留每次预测，再对图片求平均。8 次抽样不是 8 张独立图片。定向任务先分别算十类 TASR，再等权平均；不能用所有目标的分母加权汇总冒充宏平均。','防御仍使用未防御模型的原本正确集合，另报防御本身的正常错误和贴片净增加。零合格样本返回未定义，不能写成 0%。主要条件的三 seed 报均值和样本标准差；区间按真实类分层重抽整张图片 2,000 次，保留其全部抽样和固定贴片 seeds。')
    page(s,'4. 文件用途、依赖与运行顺序')
    s.append(table([['文件或目录','用途'],['src/data.py','官方数据、固定划分、训练统计、raw Dataset 与等价缓存'],['src/models/、src/model_bundle.py','三个精确结构；一次归一化；冻结后仍允许输入梯度'],['src/metrics.py、src/evaluate.py','统一分母、逐图片/逐抽样记录、正常和贴片指标'],['src/audit.py、src/evaluation_audit.py','模型/贴片身份、配置、官方测试索引与原始结果核查'],['scripts/train_classifier.py、scripts/run_training.py','200 轮训练、验证选优、断点与完成标记'],['scripts/analyze_results.py','逐样本重算、三 seed 汇总、配对 bootstrap']],[225,274]))
    add(s,'把四部分代码包解压到同一个 ECE661_Project 根目录；内容相同的共享文件可以合并。A 提供模型和 split；B 提供渲染与贴片；C、D 在相同模型身份上复用。最终测试需要完整冻结包，不能在 C/D 缺失时伪造全矩阵完成。')
    for command in ['python -m scripts.local_gpu check','python -m scripts.workflow prepare','python -m scripts.run_training','python -m scripts.workflow controls --final --include-strength --device cuda','python -m scripts.audit_results','python -m scripts.analyze_results']:s.append(P(command,'code'))
    add(s,'前两步准备环境与数据，训练完成后才进入攻击开发；controls 的 --final 只能在所有候选冻结后执行。日志和 history.json 会持续保存；恢复训练使用原始 resume.pt、优化器、调度器和 RNG 状态。')
    page(s,'5. 匹配对照怎样解释真实结果')
    fig(s,'A_matched_controls','同图、同变换、同 alpha、同位置的恒色、固定随机纹理和优化非定向纹理。随机与优化组误差条均来自三次贴片 seed；恒色只有一个固定纹理。',310)
    add(s,'大贴片即使不学习也可能造成错误。应把“学习纹理额外带来的成功”与“覆盖了图像本身”分开。16×16 的名义面积为 25%，而缩放和插值后的 alpha 面积、非零支持面积均在原始记录中单独保存。','三个模型的 S_f 不同，跨架构的 ASR 不能直接当作公平的模型强弱排名。C 的源策略比较固定在同一个 DenseNet 的合格集合上进行，才使策略差异可解释。')
    page(s,'6. 复算、检查与局限')
    add(s,'结果文件 results/raw/*.npz 使用无 pickle 的数值/字符串数组，包含 image_id、true_label、draw_id、pred_clean、pred_patch、pred_defense_clean、面积、旋转缩放与随机补边参数。旁边的 JSON 保存模型、贴片、协议、split 的 SHA-256，以及汇总指标。','scripts.audit_results 检查预期的 1,638 个完整条件；scripts.analyze_results 再从原始预测重算指标，并生成 per_seed_metrics.csv、seed_summaries.csv、target_macros_per_seed.csv 和 paired_effects.csv。原始文件、配置或权重身份不符会直接报错。','主要局限：每个模型只训练一次；测试抽样只能估计这些固定模型、贴片和图像上的效果。数字合成结果不代表打印拍摄效果，也不能给出认证鲁棒性。首次 VGG 小样本 SGD 检查未通过的记录已保留，随后用独立 Adam 调试确认拟合能力；正式 VGG 仍使用预定 SGD 配置。')
    for r in data['models']:s.append(P(r['name']+' checkpoint SHA-256：\n'+r['checkpoint_sha256'],'small'))
    build('A','A 数据、模型与公共评价',s)
    # B: universal texture, renderer, optimizer, complete base evidence.
    s=[P('B｜通用对抗贴片与基础实验','title')]
    add(s,'通用贴片是一张跨图片共享的可学习 RGB 纹理。优化阶段只更新这张纹理；分类器的参数、BatchNorm 统计量和 dropout 状态保持冻结。测试时不会针对某张图重新优化。','基础矩阵为 4 个边长 × 11 个任务 × 3 个 seed = 132 份贴片。11 个任务指一个非定向任务和十个不同目标类；不是数据集有 11 个类别。')
    s.append(table([['边长','名义面积','ResNet UASR','十目标宏平均 TASR'],*[[r['size'],f"{100*r['size']**2/1024:.3f}%",ms(r['untargeted']),ms(r['target_macro'])] for r in data['basic']]],[60,100,165,174]))
    add(s,'表中均值和 ± 样本标准差来自 seed 101/102/103 的独立贴片优化。模型是同一个最终 ResNet checkpoint，正常正确集合固定。后续 C 将完全相同的纹理直接迁移到 VGG/DenseNet。')
    page(s,'1. 渲染：纹理、alpha、旋转缩放和位置')
    s.append(Pipeline(['一张共享纹理 p\n[1,3,s,s]','独立变换\n旋转、缩放、位置','逆映射双线性\n纹理与 alpha','原图×(1-alpha)\n+ 预乘纹理']))
    add(s,'每图独立取 [-15°,15°] 旋转、[0.9,1.1] 缩放和能完整容纳插值支持范围的位置。grid_sample 使用输出坐标到输入坐标的逆映射，align_corners=False；边缘不能越界裁掉贴片。','纹理已按全一局部 alpha 预乘，采样后颜色已经在边界被衰减，不能再乘一次 alpha。背景只有在 alpha 非零处才被改变。面积同时记录 sum(alpha) 与 count(alpha>1e-6)，二者在旋转和插值时不必相等。','源实现的逐图几何计算已经批量化，CPU 双精度前向和贴片梯度与参考实现精确核对；防御插值的梯度另有数值检查。缓存只复用相同 image_id/draw_id/seed 对应的参数，不改变随机分布。')
    fig(s,'B_patch_gallery','真实学得纹理，seed 101。U 为非定向，0—9 为目标类别；每行分别为 3、5、7、16。图像按最近邻放大，原始纹理保存在 patch.pt 中。',245)
    page(s,'2. 损失方向、EOT、Adam 与候选选择')
    add(s,'定向攻击最小化 CE(logits,target)，推动预测命中目标；非定向攻击最小化 -CE(logits,true_label)，提高真实类交叉熵。交叉熵接收 logits，不先做 softmax。','每次参数更新使用 batch 128、Adam lr=0.05、1,000 次更新。只把 patch 参数放入优化器，每步后 clamp 到 [0,1]。B/C 的 EOT=1；D 会对同一张纹理累积四个 loss/K 的梯度，再进行一次 Adam 更新。','step 0 和每 100 步用固定验证子集、每图 8 次变换评价；以源 ASR 最高的候选选 patch，同分取更早 step。测试集不会决定选哪一块纹理。学习率 {0.01,0.05} 和更新 {500,1000} 的有限开发任务已先运行，主配置保留预定 .05/1000。')
    fig(s,'B_optimizer_curve','预定的 ResNet、7×7、目标类 3、seed 101 的真实验证曲线。纵轴不是测试指标。',265)
    page(s,'3. 代码文件与实际运行入口')
    s.append(table([['文件','具体职责'],['src/patches/render.py','仿射参数、合法位置、预乘 alpha 合成、覆盖面积'],['src/patches/optimize.py','共享优化核心；源模型冻结；EOT；Adam；验证选优；计时与断点'],['src/ensemble.py','分别计算各源交叉熵再平均，禁止 DenseNet 参与优化'],['scripts/develop_attacks.py','仅训练/验证的八个有限候选任务'],['src/matrix.py、scripts/workflow.py','完整任务清单、执行与冻结后的评价'],['artifacts/patches/<run_id>/','选中纹理、最后纹理、配置、源模型 hash、验证记录与恢复状态']],[223,276]))
    add(s,'依赖 A 的正式模型、归一化和 split。为复算正式测试表，应合并所有代码/模型/贴片与冻结清单；只运行单个开发任务时可以使用验证集入口，不能提前查询 DenseNet 的攻击表现。')
    for command in ['python -m scripts.develop_attacks','python -m scripts.workflow optimize --phase basic --execute --resume --device cuda','python -m scripts.workflow audit --include-strength','python -m scripts.workflow freeze --include-strength','python -m scripts.workflow evaluate --final --include-strength --device cuda']:s.append(P(command,'code'))
    add(s,'正式矩阵优先运行 7×7、5×5，再补齐 3×3、16×16。中断必须同时恢复纹理、Adam、步数与验证历史，仅从纹理重新初始化优化器不是同一次运行的续跑。')
    page(s,'4. 尺寸与目标类别的完整结果')
    fig(s,'B_size_and_targets','完整尺寸曲线与十目标热力图。非定向和定向贴片是独立优化任务；不能假设其中一个任务的成功率必定大于另一个任务。',310)
    add(s,'所有目标类都进入等权宏平均，不能只展示更容易攻击的类别。目标为 t 时，TASR 排除真实标签为 t 或正常模型已分错的图片；合格样本数随目标类而变，已在逐 seed CSV 中保留。','同一块定向贴片在同一集合上的目标命中率不可能大于其错误率，这是指标一致性检查；该关系不能用于比较两块分别优化的定向、非定向贴片。')
    page(s,'5. 纹理效力与普通遮挡的区别')
    fig(s,'A_matched_controls','随机纹理跨所有图片固定，使用三次 seed；恒色使用训练均值。三种方法共享位置、alpha 与几何变换，不给对照更容易的位置。',295)
    add(s,'16×16 的名义正方形覆盖四分之一图像，强遮挡可能已造成大量错误。应比较优化纹理与随机/恒色的额外差异，并同时阅读全测试集 PatchedAcc，而不是只看 ASR。','patch_characteristics.csv 实测每块纹理的 RGB 均值、空间总变差及去均值后径向频率≥0.25 cycles/pixel 的能量比例。只有有测量支持才讨论频率；这些描述性统计不证明模型决策的因果机制。')
    page(s,'6. 定向任务的匹配对照')
    fig(s,'B_targeted_controls','先分别计算十个目标的 TASR，再在每个 seed 内等权宏平均。优化组每个目标各有自己的贴片；固定随机/恒色对照对所有目标复用同一纹理，位置与 alpha 仍与对应优化纹理匹配。',305)
    add(s,'不能把非定向对照误差率当作定向 TASR：发生错误并不意味着命中指定目标。随机纹理的每个 seed 先对十个目标等权平均，再对三个 seed 求均值和样本标准差。恒色只是一块固定纹理，不伪造三个重复。','逐目标与分母见 control_target_metrics.csv；完整优化组的逐目标三 seed 结果见 seed_summaries.csv。所有测试图片上的攻击后准确率另存，不能用 TASR 补数代替。')
    page(s,'7. 固定编号案例、失败与局限')
    fig(s,'B_fixed_examples','测试原始编号 0—11 在最终评价前已经固定。每个 clean/patched 对显示标签和预测；hit/miss 指目标命中，excluded 不属于 TASR 分母。样例只展示同一块 7×7、目标 3、seed 101 纹理与 draw 0。',350)
    add(s,'样例可能同时包含成功、失败和本来就分错的图片；没有因结果不好而替换编号。局限包括低分辨率数字合成、固定 checkpoint、有限的优化预算与仅三次贴片初始化。没有命中不等于已证明模型鲁棒；高命中也不等于打印或拍摄条件下有效。')
    build('B','B 通用对抗贴片与基础实验',s)
    # C: transfer, ensemble, fair resource comparisons.
    s=[P('C｜迁移、集成与预算匹配','title')]
    add(s,'本部分先把 B 的 132 份同一纹理直接迁移到 VGG 和 DenseNet，再比较 ResNet 单源 R、VGG 单源 V、等权双源 RV。DenseNet 是唯一未参与优化的迁移目标，不能提供梯度、选择贴片或调预算。','固定步数组新增 V/RV 共 132 份，R 的 66 份复用 B；固定时间组在 7×7 上另做三策略、十一任务、三 seed，共 99 份。C 新增合计 231，不能把复用贴片重复计数。')
    s.append(Pipeline(['R 与 V\n最终权重冻结','同一张渲染图\n两个 CE 损失','0.5 L_R + 0.5 L_V\n只更新共享 p','冻结后 DenseNet\n仅做最终迁移']))
    add(s,'这是受控比较，不把“用了两个源模型”宣称为新算法。更高的源 ASR 不保证更高的未见架构 ASR；固定步数更高也可能只来自额外计算。')
    page(s,'1. 平均损失与公平对照')
    add(s,'RV 先分别计算两个源的交叉熵再求平均。平均 logits、平均概率和平均交叉熵是不同的目标；不能混用。两个源使用相同背景、同一个渲染结果、相同贴片参数，均保持 eval 和 requires_grad=False。','检查中分别求源梯度 g_R、g_V，确认集成梯度等于 0.5g_R+0.5g_V；模型参数和 BN buffer 不变。增加模型前向不能消耗训练图片或变换的随机流；同 seed 的策略具有相同初始化、抽图次序和变换。','DenseNet 的正常监督训练可以使用验证准确率，但任何攻击表现都不参与开发、源权重、早停或策略选择。代码显式拒绝将该模型加入源列表，最终报告的改善范围也只限于这个未见架构。')
    s.append(table([['比较条件','相同部分','允许不同的部分'],['固定步数','1,000 更新、batch 128、EOT1、配对 seed/变换','源模型数量，因此前后向调用和时间可不同'],['固定时间','同一设备、共同训练计时预算、配对随机流','较快的单源可完成更多更新'],['最终迁移','固定 DenseNet checkpoint 与其合格集合','源优化策略；不能在目标上继续更新']],[90,200,209]))
    page(s,'2. 固定步数与完整迁移结果')
    fig(s,'C_heldout_comparison','同一 DenseNet 上的策略比较。固定步数覆盖 5/7，固定时间覆盖 7；目标结果为十类等权宏平均，误差条来自三个贴片 seed。',330)
    rows=[['预算','尺寸','策略','UASR','目标宏 TASR']]
    for phase,sizes in [('ensemble_steps',[5,7]),('ensemble_time',[7])]:
        for size in sizes:
            for strategy in ('R','V','RV'):
                u=get(data['ensemble'],phase=phase,size=size,strategy=strategy,objective='UASR');t=get(data['ensemble'],phase=phase,size=size,strategy=strategy,objective='TASR macro')
                rows.append(['步数' if phase=='ensemble_steps' else '时间',size,strategy,ms(u),ms(t)])
    s.append(table(rows,[64,50,55,165,165]))
    page(s,'3. 源模型取舍与十目标分布')
    for phase,size in [('ensemble_steps',5),('ensemble_steps',7),('ensemble_time',7)]:
        fig(s,f'C_targets_{phase}_s{size}',('固定 1000 步' if phase=='ensemble_steps' else '固定时间')+f'，边长 {size}。三列为 R、V、held-out DenseNet 评价架构，每图行是源策略 R/V/RV；共同 0—100% 色标。',145)
    add(s,'目标类编号：0 飞机、1 汽车、2 鸟、3 猫、4 鹿、5 狗、6 蛙、7 马、8 船、9 卡车。')
    add(s,'同一评价模型内的策略比较使用相同分母；不同模型列的合格集合不同。可据此观察 RV 是否牺牲某个已见源模型，却改善未见架构，或收益是否只来自少数目标。不能把三个模型混成一个总均值掩盖取舍。')
    page(s,'4. 固定时间的实测成本')
    add(s,f"共同训练计时预算 T={data['time_budget']['time_budget_s']:.3f} 秒，由独立预热后 3 次各 50 个集成更新的中位每步时间乘 1,000 得到。采样、渲染、前后向、更新和同步都在训练计时范围内；验证暂停此计时并单独计费。",'每十次更新检查一次预算，因此允许一个检查区间以内的超时。结束时的实际更新数不必为 1,000 或 100 的倍数，最后候选仍需验证。预热使用独立纹理、Adam 和随机流，不免费带入正式贴片。')
    s.append(table([['策略/任务','平均实际更新数','最少—最多','平均训练时间'],*[[r['strategy']+'/'+r['objective'],f"{r['mean_steps']:.1f}",f"{r['min_steps']}—{r['max_steps']}",f"{r['mean_training_time_s']:.3f} s"] for r in data['ensemble'] if r['phase']=='ensemble_time']],[155,115,115,114]))
    add(s,'实际时间、验证时间、前后向调用量和峰值 CUDA 内存见每个 patch.pt 与 per_seed_metrics.csv。历史字段 end_to_end_seconds 的范围是优化函数入口至最终产物内容组装完成，包含预热与中途保存，但不含外部模型/数据加载和最后一次序列化；不能视为整个程序耗时。固定步数和固定时间不合并成一个均值。')
    page(s,'5. 验证效果与累计训练时间')
    fig(s,'C_validation_time_curves','预先固定 seed 101、尺寸 7、非定向及目标 3 的完整源验证轨迹。两种预算分别独立优化；RV 纵轴为两源验证 ASR 均值，R/V 为自身源模型分数。因此跨列高度不能直接解释为 held-out 迁移优势。',365)
    add(s,'曲线来自每份贴片已有的 validation_history，不追加 DenseNet 中间反馈。横轴只累计同步训练时间，验证暂停该计时；两条曲线不拼接成同一次优化。主 held-out 结论来自所有目标、三 seeds 的冻结后测试，而非这两个展示任务。')
    page(s,'6. 配对效应量、区间与优化波动')
    s.append(table([['预算/尺寸/任务','差值','百分点差及图片 bootstrap 区间'],*[[('步数' if r['phase']=='ensemble_steps' else '时间')+'/'+str(r['size'])+'/'+('U' if r['objective']=='untargeted' else '目标宏'),r['comparison'],effect(r)] for r in data['paired_effects'] if r['part']=='C']],[135,75,289]))
    add(s,'区间先平均这些固定贴片 seed 的配对差，再按真实类别重抽图片，每张图片保留全部八次变换，2,000 次，seed=4242；每次重抽重新计算目标类分母。它描述图片抽样不确定性，不包含模型训练随机性。','三次 seed 的原始差值与样本标准差另列。不能只取最优 seed；区间跨 0 时应如实说明证据不足。即使固定步数收益消失于固定时间，也只说明额外计算解释与观察相容，不能单凭此证明机制因果。')
    page(s,'7. 文件、命令与交接')
    s.append(table([['文件','功能'],['src/ensemble.py','平均交叉熵与 held-out 拒绝检查'],['src/patches/optimize.py','公共优化、计时、选择、断点；与 B/D 完全共享'],['src/matrix.py','固定步数与固定时间分别命名'],['scripts/workflow.py benchmark','预热和三次 50 步标定'],['scripts/analyze_results.py','目标宏平均、固定目标分母、配对统计'],['results/tables/paired_effects.csv','全部策略差值、三 seed 差值及图片区间']],[225,274]))
    for command in ['python -m scripts.workflow benchmark --device cuda','python -m scripts.workflow optimize --phase ensemble_steps --execute --resume --device cuda','python -m scripts.workflow optimize --phase ensemble_time --execute --resume --device cuda','python -m scripts.workflow evaluate --final --include-strength --device cuda']:s.append(P(command,'code'))
    add(s,'先合并 A 的模型、B 的基础贴片，再运行 C 的两组新增任务；最终测试前须合并全矩阵并冻结。已有最终结果时不要随意重新标定 T，否则旧贴片与新配置的身份检查会拒绝复用。')
    page(s,'8. 怎样读正结果、负结果与局限')
    for phase in ['ensemble_steps','ensemble_time']:
        r=get(data['paired_effects'],part='C',phase=phase,size=7,objective='target_macro',comparison='RV - R')
        add(s,('固定步数' if phase=='ensemble_steps' else '固定时间')+'、7×7、目标宏平均 RV−R：'+effect(r)+'。三 seed 差值依次为 '+', '.join(f'{v:+.3f}' for v in r['seed_deltas_pp'])+' pp。')
    if all(get(data['paired_effects'],part='C',phase=phase,size=7,objective='target_macro',comparison='RV - R')['delta_pp']<=0 for phase in ['ensemble_steps','ensemble_time']):
        add(s,'本次 7×7 目标宏平均在两种预算下均未超过 R 单源；因此不能宣称加入第二个源模型就提高迁移。与 V 的比较和各目标类别差异仍应分别阅读。')
    add(s,'如果收益只出现在个别目标或 seed，应报告这种不稳定；如果没有收益，仍是完整受控负结果，不能删除条件后重新宣称改进。源与目标架构的正常正确集合不同，跨架构原始 ASR 只作描述；策略比较固定在同一 DenseNet 集合。','本研究没有对所有架构、训练随机性或物理场景做总体推断。一次 held-out 架构上的结果不等于普适迁移规律；梯度相似性即使另做测量，也不能单独证明迁移机制。')
    build('C','C 迁移、集成与预算匹配',s)
    # D: cost, adaptation, matched EOT, and strength checks.
    s=[P('D｜防御、适配攻击与可靠性评价','title')]
    add(s,'本部分研究两种可微输入预处理是否削弱现有贴片，以及攻击者知道防御后重新优化是否改变结论。主要分类器是冻结 ResNet；尺寸为 5、7，覆盖十个目标类和非定向任务、三个 seed。','D 新增 132 份适配贴片和 66 份不感知防御的 EOT4 对照，主实验共 198；另外 24 份强度检查单列。B 的 66 份原贴片复用，不重复计数。')
    s.append(table([['防御','冻结设置','正常测试准确率','正常损失'],*[[r['defense'],str(r['config'].get('sigma',r['config'].get('sizes'))),pct(r['clean_accuracy']),f"{r['clean_drop_pp']:.3f} pp"] for r in data['defenses']]],[105,145,130,119]))
    add(s,'正常损失是百分点差：原模型准确率减去防御准确率。防御引入的正常错误不会被隐藏，也不能全部归因于贴片。随机防御的一次预测只做一次随机预处理；八次抽样是期望估计，不是多数投票。')
    page(s,'1. 参数选择、正常代价与可微算子')
    rows=[['验证候选','正常准确率损失','选取规则']]
    for r in data['defense_selection']['validation']:
        if 'config' in r:rows.append([str(r['config']),f"{r['clean_drop_pp']:.4f} pp",'不超过 2 pp 才合格'])
        else:rows.append([r['family']+' 最终选择',str(r['selected']),'满足上限' if r['within_clean_drop_limit'] else '无候选合格；保留最弱并披露超限'])
    s.append(table(rows,[225,105,169]))
    add(s,'Gaussian 使用 3×3 核、reflect padding 和通道独立卷积；σ 候选 .25/.5/.75。随机缩放先用 align_corners=False 双线性缩到 q×q，再独立采样 top/left 分配并 replicate 补回 32×32，弱/强候选分别是 [30,31,32] 和 [28,30,32]。','全部参数选择只用完整正常验证图。没有候选通过上限时按预设规则保留最弱，不把 identity 当成一种成功防御，也不根据测试攻击表现换参数。')
    page(s,'2. 完整链路与 EOT 的含义')
    s.append(Pipeline(['原始图片\n贴片合成','防御 D\n记录 q/padding','一次归一化\n冻结 ResNet','CE / -CE\n梯度回到 p']))
    fig(s,'D_operators','官方训练图 0 的算子示意。缩放补边使用图注中固定演示参数；这张示意图不是测试攻击案例。',230)
    add(s,'适配攻击每次 Adam 更新固定一个训练 batch，再采 K=4 组贴片变换和防御参数，每个 loss 除以 4 后回传，最后只做一次 optimizer.step。Gaussian 本身固定，但贴片变换仍独立；随机缩放必须每张图各自采 q 与补边。','梯度检查固定随机参数，使用 CPU 双精度有限差分；随机缩放批量化后前向和输入梯度与逐图参考一致。同一参数还作用于 alpha，记录 defended_alpha_area 和 defended_support_area，观察模糊或缩放对影响区域的扩散/收缩。参数重放不消耗新的随机流。')
    page(s,'3. 非定向结果：三条路径与净错误增加')
    fig(s,'D_UASR','原 K1、不感知防御 K4、适配 K4。四个面板同时保留两个尺寸和两种防御。误差条为三次贴片 seed 的样本标准差。',310)
    rows=[['防御/尺寸','优化路径','UASR','贴片净错误增加']]
    for r in data['defense_paths']:
        if r['objective']=='UASR':rows.append([r['defense']+'/'+str(r['size']),r['path'].replace('defense_',''),ms(r),f"{100*r['net_increase_mean']:+.2f} pp"])
    s.append(table(rows,[115,145,120,119]))
    page(s,'4. 定向结果：十目标宏平均与适配差值')
    fig(s,'D_TASR','每个目标先排除真实类为目标或正常分错的图片，再计算 TASR；十个目标等权宏平均，全部 seeds 保留。',310)
    rows=[['防御/尺寸','适配 K4 − 不感知 K4','三 seed 差值 pp']]
    for r in data['paired_effects']:
        if r['part']=='D' and r['objective']=='target_macro' and r['comparison']=='adaptive K4 - unknown K4':rows.append([r['defense']+'/'+str(r['size']),effect(r),', '.join(f'{v:+.2f}' for v in r['seed_deltas_pp'])])
    s.append(table(rows,[110,265,124]))
    add(s,'适配后不保证每个 seed 都更强，负差值必须保留。主表始终使用未防御模型的固定 S_f,t，同时记录防御正常图落入目标的比例和 NetTargetIncrease，避免把防御本身的目标偏置误说成贴片效力。')
    page(s,'5. 迭代、EOT 强度与抽样稳定性')
    fig(s,'D_strength_checks','7×7、非定向和目标 3 的预先规定诊断：1000/K4 是主配置，2000/K4 只增加迭代，1000/K8 只增加每步 EOT 样本。',285)
    rows=[['防御/任务','8 次抽样均值','32 次抽样均值','变化 pp']]
    for defense in ['gaussian','resize_pad']:
        for target in [None,3]:
            items=[r for r in data['stability'] if r['defense']==defense and r['target']==target]
            rows.append([defense+'/'+('U' if target is None else 't3'),f"{100*statistics.mean(r['asr8'] for r in items):.3f}%",f"{100*statistics.mean(r['asr32'] for r in items):.3f}%",f"{statistics.mean(r['difference_pp'] for r in items):+.3f}"])
    s.append(table(rows,[160,115,115,109]))
    add(s,'32 次评价使用同一已选纹理，不重新优化；前八次与主评价共享。它检查 Monte Carlo 波动，而不是增加独立图片数。强度检查只覆盖两个预定任务，不能替代十目标主矩阵，也不能根据测试最差项临时加预算。')
    page(s,'6. 全部目标类别与正常代价取舍')
    fig(s,'D_all_targets','两个尺寸、两种防御、三条路径，每格为三 seed 均值，统一 0—100% 色标。目标编号：0 飞机、1 汽车、2 鸟、3 猫、4 鹿、5 狗、6 蛙、7 马、8 船、9 卡车。',340)
    fig(s,'D_cost_tradeoff','横轴是防御自身的正常准确率损失，纵轴是防御后的非定向攻击成功率。两种符号尺寸对应贴片边长 5/7；同一防御的正常代价相同，三条攻击路径的残余风险可能不同。',215)
    page(s,'7. 计算成本：同 EOT 不等于同时间')
    rows=[['防御/尺寸/任务','路径','训练 s','验证 s','端到端 s','前后向调用']]
    for r in data['defense_paths']:
        rows.append([r['defense']+'/'+str(r['size'])+'/'+('U' if r['objective']=='UASR' else 'T宏'),r['path'].replace('defense_',''),f"{r['mean_training_time_s']:.2f}",f"{r['mean_validation_time_s']:.2f}",f"{r['mean_end_to_end_time_s']:.2f}",f"{r['mean_forward_backward_calls']:.0f}"])
    s.append(table(rows,[125,100,65,65,70,74]))
    if data.get('resource_observations'):
        add(s,'资源条件：D 适配阶段后期出现交互图形进程共享 GPU，并观察到训练墙钟增长（见 resource_observations.json）。固定更新数和 EOT 数保持不变；这些时间不能单独归因于防御算子。C 的等时间实验在该图形负载启动前已完成。')
    add(s,'每格先按对应任务与三个 seed 取平均；T宏成本等权平均十个目标。原 K1 复用 B 的计时，K4 对照和适配 K4 均为 1,000 次更新、4,000 次源模型前后向；预处理开销仍可能不同。训练时间包含 CPU 调度、采样、渲染及同步，不能称作单独的 GPU 核函数活跃时间。','表中的端到端列沿用历史 end_to_end_seconds 字段，范围是优化函数入口至最终产物组装前；外部模型/数据加载和最终序列化未计入，不等于整个程序耗时。同 EOT 只控制抽样数量；详细逐运行成本和选中 step 另存。')
    page(s,'8. 文件、依赖、运行与复算')
    s.append(table([['文件','用途'],['src/defenses.py','Gaussian、逐图独立 resize/pad、固定参数重放'],['src/patches/optimize.py','完整防御链路、EOT 平均梯度、一次 Adam 更新、候选选择'],['artifacts/defenses.json','全部正常验证候选、冻结参数及超限披露'],['src/matrix.py','适配、同 EOT 对照、强度任务分别命名'],['src/metrics.py','固定分母、防御自身错误、净增加'],['results/raw/、results/tables/','逐样本证据、各 seed、目标宏平均、配对统计']],[225,274]))
    for command in ['python -m scripts.workflow select-defenses --device cuda','python -m scripts.local_gpu attack','python -m scripts.workflow freeze --include-strength','python -m scripts.workflow evaluate --final --include-strength --device cuda','python -m scripts.workflow evaluate --final --include-strength --device cuda --draws 32 --stability-only','python -m scripts.audit_results']:s.append(P(command,'code'))
    add(s,'依赖 A 的最终 ResNet 和公共评价器，B 的渲染/优化和原贴片。local_gpu attack 会按完整矩阵执行基础、集成、防御与强度阶段；需先完成数据、正式训练、开发验证、防御选择与时间标定。')
    page(s,'9. 结论应怎样限定')
    for defense in ['gaussian','resize_pad']:
        result=get(data['paired_effects'],part='D',size=7,defense=defense,objective='untargeted',comparison='adaptive K4 - unknown K4')
        add(s,defense+'、7×7、非定向适配相对同 EOT 对照：'+effect(result)+'。三 seed 差值为 '+', '.join(f'{v:+.3f}' for v in result['seed_deltas_pp'])+' pp，不能把固定贴片下的图片区间误称为优化 seed 的稳定性。')
    deltas=[]
    for defense in ['gaussian','resize_pad']:
        for target in [None,3]:
            base=get(data['strength'],defense=defense,target=target,setting='1000/K4')['mean']
            deltas.append(100*(get(data['strength'],defense=defense,target=target,setting='2000/K4')['mean']-base))
    add(s,f'四个预定强度任务中，更新数加倍的均值变化为 {min(deltas):+.3f} 至 {max(deltas):+.3f} pp。这只支持这些任务上的增量较小，不证明所有目标均已全局收敛。')
    add(s,'依次检查：正常损失是否满足原计划规定的代价；原贴片是否被削弱；不感知防御的 K4 是否已恢复部分效果；适配 K4 比同 EOT 对照改变多少；增加迭代或 K 是否仍明显增强。三 seed 和十目标的分布比一个最高值更有意义。','若防御只削弱旧贴片、适配后恢复，则结论只是对现有攻击的削弱。若适配后仍弱，结论仍限于本模型、尺寸、随机分布与预算，不作认证安全、物理世界或所有攻击者的保证。','本次全部正常与攻击预测都使用同图同参数配对。报告 UASR/TASR、全测试集准确率、正常损失、净增加和实际成本；不会从不同贴片逐测试图片挑最强结果，再把它叫作一张通用贴片。','模型与贴片绑定 freeze.json；报告、说明及结果包绑定最终 SHA256_MANIFEST.json，原始 NPZ 的 SHA-256 在相邻 JSON 内。AI 辅助了实现、运行、排错、分析与写作；数值须能由产物复算，成员个人工作或现场答辩情况不由这些文件推定。')
    build('D','D 防御、适配攻击与可靠性评价',s)
    # Compact supplementary tables for all ten target classes and seeds.
    analysis=json.loads((PROJECT/'results/analysis.json').read_text())
    s=[P('Supplementary results','title'),P('All basic targeted conditions; mean and sample SD over patch seeds 101/102/103. Values are percentages. Raw image/draw records and all additional conditions are included in the results archive.')]
    for model in ['resnet20','vgg11_bn_cifar','densenet_bc40_k12']:
        for sizes in [(3,5),(7,16)]:
            page(s,model+' / sizes '+str(sizes))
            rows=[['Side','Target','Seed 101','Seed 102','Seed 103','Mean ± SD']]
            for size in sizes:
                for target in range(10):
                    r=get(analysis['seed_summaries'],phase='basic',strategy='R',model=model,size=size,target=target,defense='identity')
                    rows.append([size,target,*[f'{100*v:.3f}' for v in r['seed_values']],ms(r)])
            s.append(table(rows,[45,45,90,90,90,139]))
    page(s,'D / complete target-class heatmaps')
    fig(s,'D_all_targets','All target classes, both patch sizes, both defenses, and all three attack paths. Common 0–100% color limits; means over three patch seeds.',440)
    page(s,'B / matched targeted controls')
    fig(s,'B_targeted_controls','Matched mean-color occlusion, fixed random texture, and optimized targeted patches. Equal-weight ten-target macro TASR; sample SD over three patch seeds where applicable.',340)
    page(s,'C / measured computation')
    rows=[['Budget / size / source / task','Updates','Training s','Validation s','End-to-end s','F/B calls']]
    for r in data['ensemble']:
        rows.append([('steps' if r['phase']=='ensemble_steps' else 'time')+'/'+str(r['size'])+'/'+r['strategy']+'/'+('U' if r['objective']=='UASR' else 'T macro'),f"{r['mean_steps']:.1f}",f"{r['mean_training_time_s']:.3f}",f"{r['mean_validation_time_s']:.3f}",f"{r['mean_end_to_end_time_s']:.3f}",f"{r['mean_forward_backward_calls']:.1f}"])
    s.append(table(rows,[165,60,65,65,70,74]))
    add(s,'Training wall time includes sampling, rendering, forward/backward, optimizer work and synchronization. Validation is separately timed. End-to-end is the legacy per-optimizer-call elapsed field, excluding external model/data loading and final serialization. GPU kernel active time was not separately measured. Target macro costs average all ten targets and all three patch seeds.')
    page(s,'Fixed qualitative cases and eligibility')
    fig(s,'B_fixed_examples','Prespecified official test IDs 0–11, draw 0; one fixed target-3 patch, side 7, seed 101. No examples were selected by their outcomes.',400)
    cases=data['visual_cases'];eligible=sum(r['eligible'] for r in cases);hits=sum(r['targeted_success'] for r in cases)
    add(s,f'The fixed 12-image display contains {eligible} eligible targeted examples, {hits} hits, {eligible-hits} misses, and {12-eligible} excluded images. This small display is descriptive and is not an estimate replacing the 10,000-image final evaluation.')
    build('Supplement','Supplementary measured results',s)


if __name__=='__main__':main()
