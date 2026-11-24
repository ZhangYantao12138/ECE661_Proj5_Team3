"""Assemble independently labeled A/B/C/D archives and the audited full delivery."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from src.paths import workspace_root,delivery_dir

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=workspace_root()
FINAL=delivery_dir()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    audit=json.loads((ROOT/'results/evaluation_audit.json').read_text(encoding='utf-8'))
    if not audit['complete']:raise RuntimeError('Final evaluation is incomplete')
    provenance=json.loads((ROOT/'artifacts/code_provenance.json').read_text(encoding='utf-8'))
    for name,expected in provenance['files'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Source changed after provenance recording; rerun write_delivery_notes: '+name)
    qa=FINAL/'QA_APPROVED.json'
    if not qa.exists():raise RuntimeError('Render and visually verify every final PDF before packaging')
    required=['Formal_Report.pdf','A_Chinese_Guide.pdf','B_Chinese_Guide.pdf','C_Chinese_Guide.pdf','D_Chinese_Guide.pdf','Supplementary_Results.pdf','ECE661_Poster.pdf']
    approved=json.loads(qa.read_text(encoding='utf-8'))
    for name in required:
        if not (FINAL/name).exists():raise RuntimeError('Missing final deliverable: '+name)
        if approved.get('files',{}).get(name)!=sha(FINAL/name):raise RuntimeError('PDF changed after visual approval: '+name)
    readme='''ECE661 Team 3 - 完整交付与复现\n\n本包只包含经真实实验支持的正式版本。正式报告按 A/B/C/D 分节，主体最多六页，参考文献另计。四份中文说明分别解释各部分实现、代码、操作、实验和局限。\n\n一、内容与解压\nA_Code.zip、B_Code.zip、C_Code.zip、D_Code.zip 都以 ECE661_Project 为公共根目录；按 A→B→C→D 解压到同一个父目录，相同共享源码可以合并。每份都包含完整公共源码，便于独立审阅；模型和对应贴片按职责归档。Shared_Results.zip 是结果、原始逐图记录和复现审计；它也以同一根目录组织。\n\n二、依赖与执行顺序\n实际运行环境为 Windows / RTX 5080、Python 3.13.5、PyTorch 2.10.0.dev20251010+cu130、CUDA runtime 13.0。requirements-local-tested.txt 记录所用包；若使用其他环境，先通过 CUDA 梯度和数值检查。正式训练/优化用 float32 张量；cuDNN TF32 设置见环境记录。\n\n1. python -m scripts.local_gpu check\n2. python -m scripts.workflow prepare\n3. python -m scripts.run_training\n4. python -m scripts.local_gpu calibrate\n5. python -m scripts.local_gpu attack\n6. python -m scripts.local_gpu final\n7. python -m scripts.analyze_results\n8. python -m scripts.export_visual_assets\n9. python -m scripts.publication_figures\n\nA 提供固定划分、三个正式模型和指标；B 提供公共渲染/优化器与 132 份基础贴片；C 复用 B 并新增 231 份集成/时间组贴片；D 复用基础贴片并新增 198 份防御主贴片和 24 份强度贴片。最终测试要在全部候选冻结后进行，不能只解压单个部分便绕过完整冻结检查。\n\n三、复用和重跑\n已提供的权重/贴片可以直接重算结果。若要从头重训，使用新的输出目录，避免覆盖冻结权重。已有正式结果时不要无意重标定预算；改变权重、配置或预算后需要重跑受影响条件并建立新冻结清单。CIFAR 数据不重复打包，可从官方 HTTPS URL 下载；原始 MD5 为 c58f30108f718f92721af3b95e74349a。\n\n四、证据\n主矩阵 561 份，强度贴片 24 份；1,626 个每图八次抽样的评价文件和 12 个每图 32 次的文件，共 1,638。每个模型均完成 200 epochs；以验证准确率选权重，测试不参与方法选择。模型 seed 17 与贴片 seeds 101/102/103 分开，三个贴片重复不代表三次模型训练。\n\n五、文稿再生成\n图表由 scripts/publication_figures.py 从完整审计结果生成。scripts/write_report.py 写入 NeurIPS 2026 源码，使用 Tectonic/兼容 TeX 编译；源文件与原始样式随包提供。中文说明使用 reportlab，ECE661_CJK_FONT 可指向已安装中文 TrueType 字体。Windows 默认使用 SimHei；不将系统字体文件作为独立资源再分发。\n\n六、海报与展示\nECE661_Poster.pdf 由课程指定三栏模板填入真实图表，保留 Duke 标识，打印尺寸 40×22.5 英寸。展示、打印、现场答辩和同学互评仍须由成员按课程安排完成；文件不声称已代替现场活动。\n\n七、核查与使用范围\nSHA256_MANIFEST.json 绑定全部最终文件。results/evaluation_audit.json 与 results/analysis.json 保存审计和分析依据。区间按图片分层 bootstrap，条件于固定模型和贴片；不涵盖模型训练随机性。数字实验不构成认证或物理场景保证。AI 辅助了代码、执行、分析和写作；成员应阅读实现并理解结果，文稿不虚构个人投入。\n'''
    readme=readme.replace('CIFAR 数据不重复打包，可从官方 HTTPS URL 下载；原始 MD5 为','CIFAR10_Dataset.zip 提供已校验的官方原始数据归档，解压到同一 ECE661_Project 后可离线 prepare；也可从官方 HTTPS URL 下载。官方 MD5 为')
    (FINAL/'RUN_ORDER_CN.txt').write_text(readme,encoding='utf-8')
    shared=[]
    for folder in ['src','scripts','configs','tests','templates']:
        shared.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    shared.extend(p for name in ['README.md','LOCAL_GPU_CN.md','requirements.txt','requirements-local-tested.txt','requirements-docs-tested.txt','Run_Local_GPU.ps1'] if (p:=ROOT/name).exists())
    for letter,phases in [('A',set()),('B',{'basic'}),('C',{'ensemble_steps','ensemble_time'}),('D',{'defense_eot_control','defense_adaptive','strength'})]:
        with zipfile.ZipFile(FINAL/f'{letter}_Code.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in shared:archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix())
            for path in (ROOT/'artifacts').glob('*.json'):archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix())
            for path in (ROOT/'artifacts/splits').glob('*'):
                if path.is_file():archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix())
            if letter=='A':
                for path in (ROOT/'artifacts/models').rglob('*'):
                    if path.is_file() and path.name in ['best.pt','best.pt.sha256.json','history.json','training_complete.json']:
                        archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix())
            else:
                for directory in (ROOT/'artifacts/patches').iterdir():
                    if directory.name.split('__')[0] not in phases:continue
                    for path in directory.iterdir():
                        if path.is_file() and not path.name.endswith('.tmp'):archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix())
            archive.writestr(f'ECE661_Project/{letter}_RUN_ORDER_CN.txt',readme)
    dataset=ROOT/'data/cifar-10-python.tar.gz'
    if hashlib.md5(dataset.read_bytes()).hexdigest()!='c58f30108f718f92721af3b95e74349a':raise ValueError('Official dataset checksum changed')
    with zipfile.ZipFile(FINAL/'CIFAR10_Dataset.zip','w',zipfile.ZIP_STORED) as archive:
        archive.write(dataset,'ECE661_Project/data/cifar-10-python.tar.gz')
        archive.writestr('ECE661_Project/data/SOURCE.txt','Official CIFAR-10 Python archive.\nhttps://cave.cs.toronto.edu/kriz/cifar.html\nMD5: c58f30108f718f92721af3b95e74349a\nCite: A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.\nUnmodified official archive, provided for offline reproducibility.\n')
    with zipfile.ZipFile(FINAL/'Shared_Results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in (ROOT/'results').rglob('*'):
            if not path.is_file() or 'smoke' in path.parts:continue
            archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix in ['.npz','.png','.pdf'] else zipfile.ZIP_DEFLATED)
        for folder in ['development','interrupted_calibration_repair','interrupted_eot_batching_check']:
            for path in (ROOT/'artifacts'/folder).rglob('*'):
                if path.is_file():archive.write(path,'ECE661_Project/'+path.relative_to(ROOT).as_posix())
    tex=FINAL/'Report_Source';tex.mkdir(exist_ok=True)
    for path in (WORKSPACE/'work/report_latex').rglob('*'):
        if path.is_file() and path.suffix in ['.tex','.sty','.pdf']:
            destination=tex/path.relative_to(WORKSPACE/'work/report_latex');destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    original=FINAL/'Original_Inputs';original.mkdir(exist_ok=True)
    for filename in ['ECE661_Team3_Implementation_Plans.zip','ECE661_Local_GPU_v2.zip']:
        source=WORKSPACE/'work/original_archives'/filename
        if not source.exists():source=Path(r'C:\Users\Russo Zhang\Downloads')/filename
        if source.exists():shutil.copy2(source,original/source.name)
    manifest={p.relative_to(FINAL).as_posix():sha(p) for p in sorted(FINAL.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    (FINAL/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    full=WORKSPACE/'outputs/ECE661_Full_Deliverables.zip'
    with zipfile.ZipFile(full,'w',zipfile.ZIP_STORED) as archive:
        for path in sorted(FINAL.rglob('*')):
            if path.is_file():archive.write(path,'ECE661_Final/'+path.relative_to(FINAL).as_posix())
    print(full,flush=True)
    print('Final files: '+str(len(manifest))+'; archive bytes: '+str(full.stat().st_size),flush=True)


if __name__=='__main__':main()
