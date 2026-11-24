"""Write the six-page course report only after audited, measured results exist."""
import json
import shutil
from pathlib import Path
from src.paths import workspace_root

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=workspace_root()


def main():
    data=json.loads((PROJECT/'results/publication_data.json').read_text())
    if data['status']!='complete':raise RuntimeError('Real publication data is required')
    out=WORKSPACE/'work/report_latex';(out/'images').mkdir(parents=True,exist_ok=True)
    if not (out/'neurips_2026.sty').exists():shutil.copy2(PROJECT/'templates/neurips_2026.sty',out/'neurips_2026.sty')
    for path in (PROJECT/'results/figures').glob('*.pdf'):shutil.copy2(path,out/'images'/path.name)
    def percent(value):return f'{100*value:.2f}'
    def meanstd(value):return f"{percent(value['mean'])} $\\pm$ {percent(value['sample_std'])}"
    def effect(value):return f"{value['delta_pp']:+.2f} [{value['ci95_pp'][0]:+.2f}, {value['ci95_pp'][1]:+.2f}]"
    def figure(name,caption,height):return '\n\n\\begin{figure}[h]\n\\centering\\includegraphics[width=\\linewidth,height='+str(height)+'in,keepaspectratio]{images/'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n\n'
    def find(rows,**criteria):return next(r for r in rows if all(r.get(k)==v for k,v in criteria.items()))
    def table(headers,rows,spec):
        return '\\begin{center}\\small\n\\begin{tabular}{'+spec+'}\\toprule\n'+' & '.join(headers)+'\\\\\\midrule\n'+'\n'.join(' & '.join(map(str,row))+'\\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{center}\n'
    bsmall=find(data['basic'],size=3);blarge=find(data['basic'],size=16)
    c7=find(data['paired_effects'],part='C',phase='ensemble_steps',size=7,objective='target_macro',comparison='RV - R')
    ct=find(data['paired_effects'],part='C',phase='ensemble_time',size=7,objective='target_macro',comparison='RV - R')
    c_direction='increases' if c7['delta_pp']>0 else 'decreases' if c7['delta_pp']<0 else 'does not change'
    time_direction='positive' if ct['delta_pp']>0 else 'negative' if ct['delta_pp']<0 else 'zero'
    preamble=r'''\documentclass{article}
\usepackage[preprint,nonatbib]{neurips_2026}
\usepackage[T1]{fontenc}
\usepackage{booktabs,graphicx,amsmath,amssymb,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\makeatletter\renewcommand{\@noticestring}{ECE661 Team 3 course project, Fall 2026. Course report; not a NeurIPS publication.}\makeatother
\title{Universal Adversarial Patches on CIFAR-10:\\Transferability and Adaptive Defense Evaluation}
\author{Xitai Yu (xy195) \And Russo Zhang (yz1047) \AND Boyi Zhao (bz138) \And Zijiang Yang (zy232)}
\begin{document}\maketitle
'''
    text=preamble
    text+=r'''\begin{abstract}
We evaluate universal CIFAR-10 patches on three fixed classifiers: 561 main optimizations, 24 strength checks, three patch seeds, and 1,638 audited test conditions. Matched occlusion, EOT-matched defense controls, held-out isolation, and step/time comparisons distinguish texture, attacker knowledge, and computational effects. Reported metrics are recomputed from archived per-image predictions.
'''
    text+=f"ResNet untargeted ASR rises from {percent(bsmall['untargeted']['mean'])}\\% (side 3) to {percent(blarge['untargeted']['mean'])}\\% (side 16). The side-7 ensemble changes held-out targeted macro ASR by {c7['delta_pp']:+.2f} percentage points at fixed steps and {ct['delta_pp']:+.2f} at matched time. These digital measurements provide no certified or physical robustness guarantee.\n\\end{{abstract}}\n"
    text+=r'''\section{Introduction and related work}
Universal patches share a learned texture across images and transformations~\cite{brown}. Defense evaluation must include the complete pipeline, since weak optimization can make preprocessing appear robust~\cite{athalye}. We contribute a controlled comparison at original CIFAR resolution, with matched controls, held-out isolation, computational accounting, and explicit statistical units; ensemble optimization and EOT are existing methods.
\section{A: Data, models, and shared evaluation}
The official CIFAR-10 Python archive passes its published MD5. PCG64(661) selects 500 validation images per class from the 50,000 training images, leaving 45,000 training images. PCG64(662) independently selects 100 validation images per class for patch selection. Channel means and population standard deviations use training pixels only. The untouched 10,000-image test set is evaluated after artifact freezing. All networks consume raw float32 RGB through one internal normalization layer.
'''
    names=['ResNet-20','VGG-11-BN-CIFAR','DenseNet-BC-40-12']
    text+=table(['Model','Parameters','Best val. (\\%)','Test clean (\\%)'],[[name,f"{r['parameters']:,}",percent(r['best_validation_accuracy']),percent(r['clean_accuracy'])] for name,r in zip(names,data['models'])],'lrrr')
    text+=r'''All three classifiers use the exact CIFAR architectures in the research protocol, including Option A ResNet shortcuts and the specified VGG head~\cite{resnet,vgg,densenet}. Supervised training uses 200 epochs, SGD, batch 128, momentum 0.9, weight decay $5\cdot10^{-4}$, initial learning rate 0.1, and cosine decay. Checkpoints maximize full validation accuracy, with earliest ties. Selected patches also receive full 5,000-image source-only validation before freezing (1,212 conditions), without changing selection or querying DenseNet attacks. One classifier training seed (17) is used; patch repeats do not measure classifier-training uncertainty.
\clearpage
\section{B: Universal patches and baseline experiments}
For each side $s\in\{3,5,7,16\}$, objective, and seed 101/102/103, one tensor $p\in[0,1]^{3\times s\times s}$ is shared across all training images. The frozen networks remain in evaluation mode. Each update samples rotation in $[-15,15]$ degrees, scale in $[0.9,1.1]$, and a legal, fully contained location. Premultiplied alpha compositing preserves the background and avoids dark interpolation boundaries. Targeted and untargeted objectives are respectively
\[
\min_p\;\mathbb E\,\mathrm{CE}(f(A(p,x,\tau)),t),\qquad
\min_p\;-\mathbb E\,\mathrm{CE}(f(A(p,x,\tau)),y).
\]
Adam uses learning rate 0.05, batch 128, and 1,000 updates. Step 0 and every 100 updates are validated on the fixed 1,000-image subset with eight draws per image; earliest maximum validation ASR selects the patch. The finite development candidates are learning rates 0.01/0.05 and budgets 500/1,000, for the prespecified 7-by-7 untargeted and target-3 tasks. These development outcomes are separate from test results.
'''
    text+=table(['Side','Nominal area (\\%)','UASR (\\%)','Target macro TASR (\\%)'],[[r['size'],f"{100*r['size']**2/1024:.3f}",meanstd(r['untargeted']),meanstd(r['target_macro'])] for r in data['basic']],'rrrr')
    text+=figure('B_size_and_targets','Left: the same ResNet-optimized patches evaluated on each classifier. Right: all ten ResNet target classes, averaging three patch seeds. Error bars are sample standard deviations over patch seeds. Cross-model ASR has model-specific eligibility sets.',2.55)
    random16=find(data['controls'],model='resnet20',size=16,control='random')
    mean16=find(data['controls'],model='resnet20',size=16,control='mean')
    text+=f"For the 16-by-16 ResNet condition, optimized untargeted ASR is {percent(blarge['untargeted']['mean'])}\\%, compared with {percent(random16['mean'])}\\% for random texture and {percent(mean16['mean'])}\\% for mean-color occlusion. These matched controls separate the texture contribution from the effect of removing a quarter of the nominal image area. Effective alpha and support areas are retained in every raw record; transformed coverage is not equated with nominal area. All target classes and all seeds remain in the tables, including weak attacks.\n"
    text+=r'''\clearpage
\section{C: Held-out transfer and compute budgets}
The basic 132 ResNet patches transfer without alteration to VGG and DenseNet. The extension compares ResNet (R), VGG (V), and an equal-weight average of their cross-entropy losses (RV). The same rendered images feed both source networks. Averaging losses is distinct from averaging logits. DenseNet provides neither gradients nor attack-selection feedback. Fixed-step comparisons cover sides 5/7 and all eleven objectives, reusing the R patches from B.
'''
    budget=data['time_budget']['time_budget_s']
    text+=f"For the 7-by-7 time comparison, three 50-update ensemble measurements after independent warmup set a common training budget of {budget:.3f} seconds. Every strategy starts from the paired initialization, checks elapsed time every ten updates, and saves the final budget-ending candidate as well as periodic candidates. Validation pauses this timer; validation, warmup, and end-to-end costs are recorded separately. Faster methods may perform more updates. Time budgets are calibrated on the same RTX 5080 after classifier training ends.\n"
    time_audit=json.loads((PROJECT/'results/time_budget_audit.json').read_text())
    if time_audit['status']!='complete' or time_audit['n_patches']!=99:raise RuntimeError('Complete matched-time audit required')
    measured=time_audit['records'];costs=[]
    for strategy in ['R','V','RV']:
        runs=[r for r in measured if r['strategy']==strategy]
        costs.append(f"{strategy}: {npmean([r['total_steps'] for r in runs]):.0f} updates / {npmean([r['training_time_s'] for r in runs]):.3f} s")
    text+='Observed strategy means are '+', '.join(costs)+f". Across all 99 runs, training time ranges from {min(r['training_time_s'] for r in measured):.3f} to {max(r['training_time_s'] for r in measured):.3f} seconds.\n"
    text+=figure('C_main_report','Held-out DenseNet targeted macro ASR for side 7. R and V are single-source optimizers; RV averages their losses. The two panels use distinct budget definitions. Error bars use three independent patch seeds; the table also includes untargeted effects and side 5.',2.1)
    comparisons=[]
    for phase,size in [('ensemble_steps',5),('ensemble_steps',7),('ensemble_time',7)]:
        for objective in ['untargeted','target_macro']:
            r=find(data['paired_effects'],part='C',phase=phase,size=size,objective=objective,comparison='RV - R')
            v=find(data['paired_effects'],part='C',phase=phase,size=size,objective=objective,comparison='RV - V')
            comparisons.append(['steps' if phase.endswith('steps') else 'time',size,'U' if objective=='untargeted' else 'T macro',effect(r),effect(v)])
    text+=table(['Budget','Side','Metric','RV - R: pp [95\\% CI]','RV - V: pp [95\\% CI]'],comparisons,'lrlrr')
    text+=f"At fixed steps, the 7-by-7 targeted macro comparison {c_direction} held-out ASR; the matched-time effect has a {time_direction} sign. "
    if c7['delta_pp']<=0 and ct['delta_pp']<=0:
        text+='RV does not outperform R on these side-7 targeted averages under either budget. The complete comparisons against V and the per-target distributions remain essential: adding a source is not a uniform transfer improvement. '
    source_r=find(data['source_tradeoffs'],phase='ensemble_steps',size=7,strategy='R',model='resnet20')
    source_v=find(data['source_tradeoffs'],phase='ensemble_steps',size=7,strategy='V',model='vgg11_bn_cifar')
    rv_r=find(data['source_tradeoffs'],phase='ensemble_steps',size=7,strategy='RV',model='resnet20')
    rv_v=find(data['source_tradeoffs'],phase='ensemble_steps',size=7,strategy='RV',model='vgg11_bn_cifar')
    text+=f"On the source models, fixed-step side-7 target macro ASR falls from {percent(source_r['target_macro']['mean'])}\\% (R-only) to {percent(rv_r['target_macro']['mean'])}\\% (RV) on R, and from {percent(source_v['target_macro']['mean'])}\\% (V-only) to {percent(rv_v['target_macro']['mean'])}\\% (RV) on V. "
    text+='Intervals describe image-sampling uncertainty conditional on these models and patches; seed differences remain separate. Neither budget comparison proves a causal transfer mechanism.\n'
    text+=r'''\clearpage
\section{D: Defenses and adaptive evaluation}
Both defenses operate after patching and before normalization. Gaussian blur uses a fixed 3-by-3 normalized kernel with reflect padding. Resize/pad independently samples each image's integer resolution and top/left padding, uses bilinear interpolation, and replicate-pads to 32-by-32. The clean-validation rule allows at most two percentage points of lost accuracy.
'''
    selection=data['defense_selection']['validation']
    losses=[]
    for config in data['defense_selection']['selected']:
        chosen=next(r for r in selection if r.get('config')==config)
        losses.append(f"{chosen['clean_drop_pp']:.4f} pp ({config['type'].replace('_','/')})")
    text+='Selected validation losses are '+' and '.join(losses)+'; test costs appear below. '
    for item in selection:
        if 'within_clean_drop_limit' in item and not item['within_clean_drop_limit']:
            text+=f"{item['family'].replace('_','/').capitalize()} has no qualifying candidate; its weakest setting is an over-limit fallback.\n"
    text+=table(['Defense','Frozen setting','Test clean (\\%)','Clean drop (pp)'],[[r['defense'].replace('_','/'),str(r['config'].get('sigma',r['config'].get('sizes'))),percent(r['clean_accuracy']),f"{r['clean_drop_pp']:.3f}"] for r in data['defenses']],'llrr')
    text+=r'''For each defense and side 5/7, we evaluate three paths: the original B patch optimized with K=1 and no defense; a defense-unaware K=4 patch; and an adaptive K=4 patch optimized through the defense. All cover eleven objectives and three seeds. The latter two match the number of EOT samples, preventing additional gradient-estimation computation from being attributed entirely to defense awareness. Four independent joint draws accumulate loss/K gradients before one Adam update.
'''
    text+=figure('D_main_report','Side 7: K1 is the original nonadaptive attack; K4 is defense-unaware EOT matching; adaptive is defense-aware K4. Eligibility remains the undefended clean-correct set. Targeted panels equally average all ten targets. Both sizes appear in the effect table.',2.45)
    d_effects=[r for r in data['paired_effects'] if r['part']=='D' and r['comparison']=='adaptive K4 - unknown K4']
    text+=table(['Defense','Side','Metric','Adaptive - unknown K4: pp [95\\% CI]'],[[r['defense'].replace('_','/'),r['size'],'U' if r['objective']=='untargeted' else 'T macro',effect(r)] for r in d_effects],'lrlr')
    dg=find(d_effects,size=7,defense='gaussian',objective='untargeted')
    dr=find(d_effects,size=7,defense='resize_pad',objective='untargeted')
    text+=f"At side 7, adaptive-minus-unknown-K4 UASR is {dg['delta_pp']:+.2f} pp for Gaussian blur and {dr['delta_pp']:+.2f} pp for resize/pad. Defense awareness is therefore not uniformly stronger in this experiment. "
    text+=r'''ASR alone cannot assign all defended errors to the patch; paired defense-only errors and net error/target-rate increases are archived alongside full clean/patched accuracy. Per-patch elapsed cost spans optimizer-function entry to artifact assembly; external model/data loading and final serialization are excluded.
'''
    if data.get('resource_observations'):
        text+='Interactive graphics began sharing the GPU during later fixed-step D runs. Their observed runtimes do not isolate defense overhead; update/EOT budgets are unchanged. The C time-matched runs ended earlier.\n'
    text+=r'''\clearpage
\section*{D continued: attack strength and Monte Carlo stability}
Strength diagnostics are fixed at side 7, target 3 and the untargeted task, both defenses, and all three seeds. The two additional optimization conditions change one factor at a time: 2,000 updates with K=4, or 1,000 updates with K=8. These 24 patches remain separate from the 561 main optimizations. They test two prespecified objectives and are not substitutes for the full ten-target main matrix.
'''
    text+=figure('D_strength_checks_main','Prespecified strength checks: update budget and EOT count vary separately. Each bar shows mean ASR and sample standard deviation across the three patch seeds. Main settings are evaluated against the corresponding defense.',2.5)
    step_changes=[];eot_changes=[]
    for defense in ['gaussian','resize_pad']:
        for target in [None,3]:
            base=find(data['strength'],defense=defense,target=target,setting='1000/K4')['mean']
            step_changes.append(100*(find(data['strength'],defense=defense,target=target,setting='2000/K4')['mean']-base))
            eot_changes.append(100*(find(data['strength'],defense=defense,target=target,setting='1000/K8')['mean']-base))
    text+=f"Across these four diagnostic tasks, doubling updates changes mean ASR by {min(step_changes):+.3f} to {max(step_changes):+.3f} pp; doubling K changes it by {min(eot_changes):+.3f} to {max(eot_changes):+.3f} pp. These small changes do not establish global convergence.\n"
    stability=[]
    for defense in ['gaussian','resize_pad']:
        for target in [None,3]:
            items=[r for r in data['stability'] if r['defense']==defense and r['target']==target]
            stability.append([defense.replace('_','/'),'U' if target is None else 'target 3',f"{npmean([100*r['asr8'] for r in items]):.3f}",f"{npmean([100*r['asr32'] for r in items]):.3f}",f"{npmean([r['difference_pp'] for r in items]):+.3f}"])
    text+=table(['Defense','Task','8 draws: ASR (\\%)','32 draws: ASR (\\%)','Difference (pp)'],stability,'llrrr')
    text+=r'''These are the same selected patches, with no further optimization. The first eight draws are shared with the 32-draw estimate. Additional draws reduce Monte Carlo noise, rather than providing additional independent images or a majority-vote defense.
\section*{A continued: denominators, pairing, and auditing}
For classifier $f$, let $S_f=\{i:f(x_i)=y_i\}$ and $S_{f,t}=\{i\in S_f:y_i\ne t\}$. UASR averages patched errors over $S_f$; TASR averages target hits over $S_{f,t}$. Patched accuracy instead uses all test images. Defenses retain the undefended classifier's fixed eligibility set. Each image has eight paired transform/defense draws; their outcomes are averaged before aggregation over images. Empty eligibility is undefined, never zero.

Three independently optimized patches provide mean and sample standard deviation (ddof=1). Targeted results first form an equal-weight ten-target macro average. Paired differences use the same image, draw, seed, and checkpoint. Confidence intervals resample whole images within true classes 2,000 times with seed 4242, retaining all draws and the fixed three patch seeds; target denominators are recomputed inside each bootstrap. They exclude classifier-training variability. SHA-256 checks, duplicate/missing-condition detection, exact official image sets, and raw-to-summary recomputation guard all 1,638 final files.
\clearpage
\section{Qualitative evidence, limitations, and conclusion}
'''
    text+=figure('B_patch_gallery_main','Actual learned ResNet textures, seed 101. Rows are sides 3/5/7/16; columns are untargeted (U) and targets 0--9. These examples illustrate patterns, not an additional optimization or a semantic explanation.',1.85)
    text+=figure('B_fixed_examples_main','First six IDs from the fixed qualitative list, draw 0, the 7-by-7 target-3 seed-101 patch. Each clean/patched pair shows true class y and prediction p. The full list of IDs 0--11, eligibility flags, and success/failure labels is in the supplement.',1.65)
    cases=data['visual_cases'];eligible=sum(r['eligible'] for r in cases);hits=sum(r['targeted_success'] for r in cases)
    analysis=json.loads((PROJECT/'results/analysis.json').read_text())
    targets=[r for r in analysis['seed_summaries'] if r['phase']=='basic' and r['model']=='resnet20' and r['size']==7 and r['target'] is not None and r['defense']=='identity']
    strongest=max(targets,key=lambda r:r['mean']);weakest=min(targets,key=lambda r:r['mean'])
    texture=next(r for r in data['patch_characteristics'] if r['size']==7 and r['target']==3 and r['seed']==101)
    text+=f"The fixed 12-image list contains {hits} target hits and {eligible-hits} misses among {eligible} eligible examples; {12-eligible} are excluded. Across the complete side-7 ResNet target matrix, target {strongest['target']} has the highest mean TASR ({percent(strongest['mean'])}\\%) and target {weakest['target']} the lowest ({percent(weakest['mean'])}\\%). The displayed target-3 texture has adjacent-pixel total variation {texture['total_variation']:.3f} and {100*texture['high_frequency_energy_fraction']:.1f}\\% of its non-DC energy at radial frequencies at least 0.25 cycles/pixel. These are descriptive observations, not causal feature attribution.\n"
    text+=r'''\paragraph{Interpretation and limits.}
The archived measurements support only the specified digital patch space, models, splits, and budgets. A 16-by-16 square covers 25\% nominal area, so learned-texture effects must be read alongside matched occlusion. Small patches may fail and target classes may differ; neither is removed from the report. Measured spectral energy and total variation are provided as descriptive texture statistics, without inferring which visual features cause a network's decision. Three patch repeats do not establish stability over independently trained classifiers. Attack failure is neither a certified robustness result nor evidence for physical printing/camera effectiveness. Held-out isolation constrains method selection but does not represent all architectures.

\paragraph{Responsibilities and AI assistance.}
The planned scope maps A (xy195) to data/models/metrics, B (yz1047) to shared patch optimization and base attacks, C (bz138) to transfer/ensembles/budgets, and D (zy232) to defenses/adaptation/strength checks. An AI coding assistant assisted implementation, execution, debugging, analysis, and writing. Numerical claims in this report are generated from archived measured records and validated by the included checks. This statement describes the workflow and planned responsibility mapping; it does not attest to unobserved individual effort or attendance.
\clearpage
\begin{thebibliography}{9}
\bibitem{brown} T. B. Brown et al. Adversarial Patch. arXiv:1712.09665, 2017. \url{https://arxiv.org/abs/1712.09665}.
\bibitem{athalye} A. Athalye, N. Carlini, and D. Wagner. Obfuscated Gradients Give a False Sense of Security. ICML, PMLR 80:274--283, 2018. \url{https://proceedings.mlr.press/v80/athalye18a.html}.
\bibitem{cifar} A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, 2009. \url{https://cave.cs.toronto.edu/kriz/cifar.html}.
\bibitem{resnet} K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016. \url{https://arxiv.org/abs/1512.03385}.
\bibitem{vgg} K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR, 2015. \url{https://arxiv.org/abs/1409.1556}.
\bibitem{densenet} G. Huang et al. Densely Connected Convolutional Networks. CVPR, 2017. \url{https://arxiv.org/abs/1608.06993}.
\end{thebibliography}
\end{document}
'''
    (out/'main.tex').write_text(text,encoding='utf-8')
    print(out/'main.tex')


def npmean(values):
    if not values:raise ValueError('Missing measured values')
    return sum(values)/len(values)


if __name__=='__main__':main()
