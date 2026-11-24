"""Publication figures and compact report data. This process deliberately does not import torch."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/figures')
MODELS=['resnet20','vgg11_bn_cifar','densenet_bc40_k12']
NAMES=['ResNet-20','VGG-11-BN','DenseNet-BC-40-12']
CLASSES=['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
COLORS=['#245b91','#d48234','#238477']


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    data=json.loads(Path('results/analysis.json').read_text())
    visual=json.loads(Path('results/visual_assets.json').read_text())
    assets=dict(np.load('results/visual_assets.npz',allow_pickle=False))
    if data['status']!='complete':raise RuntimeError('Measured final analysis is required')
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold','savefig.bbox':'tight'})
    def finish(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=210);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    def compact(fig,name,width,height):
        fig.set_size_inches(width,height)
        for ax in fig.axes:
            ax.tick_params(labelsize=7)
            ax.title.set_fontsize(8)
            ax.xaxis.label.set_fontsize(7.5);ax.yaxis.label.set_fontsize(7.5)
        finish(fig,name)
    def select(rows,**keys):return [r for r in rows if all(r.get(k)==v for k,v in keys.items())]
    def metric(phase,strategy,model,size,target,defense='identity',eot=None):
        keys=dict(phase=phase,strategy=strategy,model=model,size=size,defense=defense)
        if eot is not None:keys['eot']=eot
        if target=='macro':
            found=select(data['target_macros_per_seed'],**keys)
            if len(found)!=3:raise ValueError('Incomplete target macro group '+str(keys))
            values=[r['tasr_macro'] for r in sorted(found,key=lambda r:r['seed'])]
            return dict(mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),seed_values=values)
        found=select(data['seed_summaries'],target=target,**keys)
        if len(found)!=1:raise ValueError('Ambiguous scientific group '+str(keys))
        # Callers add comparison identities, which can intentionally differ from
        # the reused basic-patch identity. Return statistics without duplicate
        # phase/size/defense keyword arguments or overwritten provenance.
        return {key:found[0][key] for key in ('mean','sample_std','seed_values')}
    def bars(ax,labels,groups,legend=True):
        width=.8/len(groups);x=np.arange(len(labels))
        for j,(name,values,color) in enumerate(groups):
            ax.bar(x+(j-(len(groups)-1)/2)*width,[100*v['mean'] for v in values],width,
                yerr=[100*v['sample_std'] for v in values],label=name,color=color,capsize=2)
        upper=max(1,max(100*(v['mean']+v['sample_std']) for _,values,_ in groups for v in values)*1.25)
        ax.set_xticks(x,labels);ax.set_ylim(0,min(105,upper));ax.set_ylabel('ASR (%)');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        if legend:ax.legend(frameon=False,fontsize=10)
    # A: independent real training curves; all checkpoints are now complete.
    fig,axes=plt.subplots(2,3,figsize=(10,5),layout='constrained')
    models=[]
    for j,(model,label) in enumerate(zip(MODELS,NAMES)):
        history=json.loads(Path(f'artifacts/models/{model}/history.json').read_text())
        if len(history)!=200:raise ValueError('Incomplete classifier training')
        clean=next(r for r in data['per_seed_records'] if r['run_id']=='clean' and r['eval_model']==model)
        metadata=next(r for r in visual['models'] if r['name']==model)
        metadata=dict(metadata,clean_accuracy=clean['clean_acc'],n_clean_correct=clean['n_eligible_untargeted'])
        models.append(metadata)
        axes[0,j].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training',color=COLORS[0])
        axes[0,j].plot([r['epoch'] for r in history],[r['val_loss'] for r in history],label='Validation',color=COLORS[1]);axes[0,j].set(title=label,ylabel='Cross entropy');axes[0,j].legend(frameon=False,fontsize=10)
        axes[1,j].plot([r['epoch'] for r in history],[100*r['val_acc'] for r in history],color=COLORS[j]);axes[1,j].set(xlabel='Epoch',ylabel='Validation accuracy (%)',ylim=(0,100))
    finish(fig,'A_training_curves')
    # B: all sizes, controls, complete target-class heatmap.
    basic=[]
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
    for model,label,color in zip(MODELS,NAMES,COLORS):
        values=[metric('basic','R',model,size,None) for size in (3,5,7,16)]
        axes[0].errorbar([3,5,7,16],[100*v['mean'] for v in values],yerr=[100*v['sample_std'] for v in values],marker='o',label=label,color=color,capsize=2)
    axes[0].set(xlabel='Nominal patch side (pixels)',ylabel='Untargeted ASR (%)',ylim=(0,105));axes[0].legend(frameon=False,fontsize=10)
    heat=np.array([[100*metric('basic','R',MODELS[0],s,t)['mean'] for t in range(10)] for s in (3,5,7,16)])
    im=axes[1].imshow(heat,vmin=0,vmax=100,cmap='YlOrRd',aspect='auto');axes[1].set_xticks(range(10),CLASSES,rotation=55,ha='right');axes[1].set_yticks(range(4),['3','5','7','16']);axes[1].set(ylabel='Nominal side',title='ResNet targeted ASR: all targets');fig.colorbar(im,ax=axes[1],label='TASR (%)')
    finish(fig,'B_size_and_targets')
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    controls=[]
    for ax,model,label in zip(axes,MODELS,NAMES):
        groups=[]
        for control,color in [('mean','#b5bbc0'),('random','#dfae6c'),('optimized',COLORS[0])]:
            values=[]
            for size in (3,5,7,16):
                if control=='optimized':value=metric('basic','R',model,size,None)
                else:
                    rows=select(data['per_seed_records'],eval_model=model,size=size,control=control,draws=8)
                    a=[r['uasr'] for r in rows];value=dict(mean=float(np.mean(a)),sample_std=float(np.std(a,ddof=1)) if len(a)>1 else 0.)
                    controls.append(dict(model=model,size=size,control=control,**value))
                values.append(value)
            groups.append((control,values,color))
        bars(ax,['3','5','7','16'],groups);ax.set_title(label);ax.set_xlabel('Nominal side')
    finish(fig,'A_matched_controls')
    targeted_controls=[]
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    for ax,model,label in zip(axes,MODELS,NAMES):
        groups=[]
        for control,color in [('mean','#b5bbc0'),('random','#dfae6c'),('optimized',COLORS[0])]:
            values=[]
            for size in (3,5,7,16):
                if control=='optimized':value=metric('basic','R',model,size,'macro')
                else:
                    rows=select(data['control_target_metrics'],model=model,size=size,control=control)
                    seed_values=[]
                    for seed in ([None] if control=='mean' else [101,102,103]):
                        items=select(rows,seed=seed)
                        if sorted(r['target'] for r in items)!=list(range(10)):raise ValueError('Incomplete targeted controls')
                        seed_values.append(float(np.mean([r['tasr'] for r in items])))
                    value=dict(mean=float(np.mean(seed_values)),sample_std=float(np.std(seed_values,ddof=1)) if len(seed_values)>1 else 0.,seed_values=seed_values)
                values.append(value);targeted_controls.append(dict(model=model,size=size,control=control,**value))
            groups.append((control,values,color))
        bars(ax,['3','5','7','16'],groups);ax.set_title(label);ax.set_xlabel('Nominal side');ax.set_ylabel('Target macro TASR (%)')
    finish(fig,'B_targeted_controls')
    for size in (3,5,7,16):
        basic.append(dict(size=size,untargeted=metric('basic','R',MODELS[0],size,None),target_macro=metric('basic','R',MODELS[0],size,'macro')))
    fig,axes=plt.subplots(4,11,figsize=(11,4.5),layout='constrained')
    for i,size in enumerate((3,5,7,16)):
        for j,target in enumerate([None]+list(range(10))):
            ax=axes[i,j];ax.imshow(assets[f'patch_s{size}_t{target}'],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title('U' if target is None else str(target),fontsize=12)
            if j==0:ax.set_ylabel(f'{size}x{size}',fontsize=12)
    finish(fig,'B_patch_gallery')
    for ax in fig.axes:
        ax.title.set_fontsize(18);ax.yaxis.label.set_fontsize(17)
    finish(fig,'B_patch_gallery_poster')
    compact(fig,'B_patch_gallery_main',6.5,1.85)
    fig,ax=plt.subplots(figsize=(6.8,2.8),layout='constrained')
    history=visual['representative_history']
    ax.plot([r['step'] for r in history],[100*r['asr'] for r in history],marker='o',color=COLORS[0])
    ax.set(xlabel='Adam update',ylabel='Validation TASR (%)',title='ResNet, side 7, target 3, seed 101');ax.grid(alpha=.16)
    finish(fig,'B_optimizer_curve')
    fig,axes=plt.subplots(4,6,figsize=(10,6.7),layout='constrained')
    for i,case in enumerate(visual['cases']):
        row=i//3;col=2*(i%3)
        for offset,key in [(0,'case_clean'),(1,'case_patched')]:
            ax=axes[row,col+offset];ax.imshow(assets[key][i],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            pred=case['pred_clean'] if offset==0 else case['pred_patch']
            title=f"ID{i}: y={case['true_label']}, p={pred}\n"
            title+=('hit' if case['targeted_success'] else 'miss' if case['eligible'] else 'excluded') if offset else 'clean'
            ax.set_title(title,fontsize=11)
    finish(fig,'B_fixed_examples')
    fig,axes=plt.subplots(2,6,figsize=(8,3.1),layout='constrained')
    for i,case in enumerate(visual['cases'][:6]):
        row=i//3;col=2*(i%3)
        for offset,key in [(0,'case_clean'),(1,'case_patched')]:
            ax=axes[row,col+offset];ax.imshow(assets[key][i],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            pred=case['pred_clean'] if offset==0 else case['pred_patch']
            ax.set_title(f"{i}: y={case['true_label']}, p={pred}",fontsize=9)
    finish(fig,'B_fixed_examples_main')
    compact(fig,'B_fixed_examples_main',6.5,1.65)
    # C: held-out effects with genuinely equal steps or measured training time.
    fig,axes=plt.subplots(2,2,figsize=(8,5.3),layout='constrained');ensemble=[]
    for row,target in enumerate([None,'macro']):
        for col,(phase,sizes) in enumerate([('ensemble_steps',[5,7]),('ensemble_time',[7])]):
            groups=[]
            for strategy,color in zip(['R','V','RV'],COLORS):
                values=[]
                for size in sizes:
                    source='basic' if phase=='ensemble_steps' and strategy=='R' else phase
                    value=metric(source,strategy,MODELS[2],size,target);values.append(value)
                    records=select(data['per_seed_records'],phase=source,strategy=strategy,eval_model=MODELS[2],size=size,draws=8)
                    records=[r for r in records if (r['target'] is None)==(target is None)]
                    timing=dict(mean_training_time_s=float(np.mean([r['wall_time_s'] for r in records])),mean_validation_time_s=float(np.mean([r['validation_time_s'] for r in records])),
                        mean_end_to_end_time_s=float(np.mean([r['end_to_end_time_s'] for r in records])),mean_forward_backward_calls=float(np.mean([r['forward_backward_calls'] for r in records])),mean_steps=float(np.mean([r['optimizer_steps'] for r in records])),
                        min_steps=min(r['optimizer_steps'] for r in records),max_steps=max(r['optimizer_steps'] for r in records))
                    ensemble.append(dict(phase=phase,strategy=strategy,size=size,objective='UASR' if target is None else 'TASR macro',**value,**timing))
                groups.append((strategy,values,color))
            bars(axes[row,col],[str(s) for s in sizes],groups);axes[row,col].set_title(('Fixed 1000 steps' if col==0 else 'Matched training time')+' / '+('UASR' if row==0 else 'TASR macro'));axes[row,col].set_xlabel('Nominal side')
    for row in range(2):
        upper=max(ax.get_ylim()[1] for ax in axes[row]);[ax.set_ylim(0,upper) for ax in axes[row]]
    finish(fig,'C_heldout_comparison')
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.8),layout='constrained')
    for ax,phase in zip(axes,['ensemble_steps','ensemble_time']):
        values=[next(r for r in ensemble if r['phase']==phase and r['strategy']==strategy and r['size']==7 and r['objective']=='TASR macro') for strategy in ['R','V','RV']]
        bars(ax,['R','V','RV'],[('TASR macro',values,COLORS[2])],legend=False)
        ax.set_title('1000 steps' if phase=='ensemble_steps' else 'Matched time');ax.set_ylabel('Target macro TASR (%)')
    upper=max(ax.get_ylim()[1] for ax in axes);[ax.set_ylim(0,upper) for ax in axes]
    finish(fig,'C_main')
    compact(fig,'C_main_report',6.5,2.1)
    fig,axes=plt.subplots(3,3,figsize=(12,8),layout='constrained');source_tradeoffs=[]
    for row,(phase,size) in enumerate([('ensemble_steps',5),('ensemble_steps',7),('ensemble_time',7)]):
        for col,model in enumerate(MODELS):
            ax=axes[row,col];heat=[]
            for strategy in ['R','V','RV']:
                source='basic' if phase=='ensemble_steps' and strategy=='R' else phase
                heat.append([100*metric(source,strategy,model,size,t)['mean'] for t in range(10)])
                source_tradeoffs.append(dict(phase=phase,size=size,strategy=strategy,model=model,untargeted=metric(source,strategy,model,size,None),target_macro=metric(source,strategy,model,size,'macro')))
            im=ax.imshow(heat,vmin=0,vmax=100,cmap='YlOrRd',aspect='auto');ax.set_yticks(range(3),['R','V','RV'])
            ax.set_xticks(range(10),range(10));ax.set_xlabel('Target class');ax.set_title(('steps' if phase=='ensemble_steps' else 'time')+f', s={size} / '+NAMES[col],fontsize=10)
    fig.colorbar(im,ax=axes,label='Targeted ASR (%)',shrink=.6)
    finish(fig,'C_targets_and_sources')
    for phase,size in [('ensemble_steps',5),('ensemble_steps',7),('ensemble_time',7)]:
        fig,axes=plt.subplots(1,3,figsize=(8.3,2.4),layout='constrained')
        for ax,model,label in zip(axes,MODELS,NAMES):
            heat=[]
            for strategy in ['R','V','RV']:
                source='basic' if phase=='ensemble_steps' and strategy=='R' else phase
                heat.append([100*metric(source,strategy,model,size,t)['mean'] for t in range(10)])
            im=ax.imshow(heat,vmin=0,vmax=100,cmap='YlOrRd',aspect='auto');ax.set_yticks(range(3),['R','V','RV'],fontsize=9)
            ax.set_xticks(range(10),range(10),fontsize=9);ax.set_title(label,fontsize=10);ax.set_xlabel('Target class',fontsize=9)
        fig.colorbar(im,ax=axes,label='TASR (%)',shrink=.8)
        finish(fig,f'C_targets_{phase}_s{size}')
    fig,axes=plt.subplots(2,3,figsize=(10,5.2),layout='constrained')
    for row,target in enumerate([None,3]):
        for col,strategy in enumerate(['R','V','RV']):
            ax=axes[row,col]
            for phase,color in zip(['ensemble_steps','ensemble_time'],COLORS):
                h=next(r['history'] for r in visual['ensemble_histories'] if r['phase']==phase and r['strategy']==strategy and r['target']==target)
                ax.plot([r['training_time'] for r in h],[100*r['asr'] for r in h],marker='.',label='steps' if phase=='ensemble_steps' else 'time',color=color)
            ax.set(xlabel='Cumulative training wall time (s)',ylabel='Source validation ASR (%)',title=strategy+' / '+('U' if target is None else 'target 3'));ax.legend(frameon=False,fontsize=10);ax.grid(alpha=.15)
    finish(fig,'C_validation_time_curves')
    # D: cost, nonadaptive/EOT-matched/adaptive, all targets retained.
    defense_table=[];defense_rows=[]
    fig,axes=plt.subplots(2,4,figsize=(12,5.5),layout='constrained')
    for col,(size,defense) in enumerate([(5,'gaussian'),(7,'gaussian'),(5,'resize_pad'),(7,'resize_pad')]):
        for row,target in enumerate([None,'macro']):
            values=[]
            for phase,eot in [('basic',1),('defense_eot_control',4),('defense_adaptive',4)]:
                value=metric(phase,'R',MODELS[0],size,target,defense,eot);values.append(value)
                records=select(data['per_seed_records'],phase=phase,strategy='R',eval_model=MODELS[0],size=size,draws=8)
                records=[r for r in records if r['defense_eval']['type']==defense and (r['target'] is None)==(target is None)]
                field='net_error_increase' if target is None else 'net_target_increase'
                per_seed=[np.mean([r[field] for r in records if r['seed']==seed]) for seed in (101,102,103)]
                cost=dict(mean_training_time_s=float(np.mean([r['wall_time_s'] for r in records])),mean_validation_time_s=float(np.mean([r['validation_time_s'] for r in records])),
                    mean_end_to_end_time_s=float(np.mean([r['end_to_end_time_s'] for r in records])),mean_forward_backward_calls=float(np.mean([r['forward_backward_calls'] for r in records])),
                    mean_steps=float(np.mean([r['optimizer_steps'] for r in records])))
                defense_rows.append(dict(size=size,defense=defense,path=phase,objective='UASR' if target is None else 'TASR macro',**value,**cost,
                    net_increase_mean=float(np.mean(per_seed)),net_increase_sample_std=float(np.std(per_seed,ddof=1))))
            bars(axes[row,col],['K1','K4','adaptive'],[('ASR',[*values],COLORS[0])],legend=False)
            axes[row,col].set_title(f"{defense}, s={size}"+(' / UASR' if row==0 else ' / TASR macro'));axes[row,col].tick_params(axis='x',labelrotation=35)
    for row in range(2):
        upper=max(ax.get_ylim()[1] for ax in axes[row]);[ax.set_ylim(0,upper) for ax in axes[row]]
    finish(fig,'D_three_paths')
    for objective,name in [('UASR','D_UASR'),('TASR macro','D_TASR')]:
        fig,axes=plt.subplots(2,2,figsize=(8,4.8),layout='constrained')
        for ax,(size,defense) in zip(axes.flat,[(5,'gaussian'),(7,'gaussian'),(5,'resize_pad'),(7,'resize_pad')]):
            values=[next(r for r in defense_rows if r['size']==size and r['defense']==defense and r['path']==phase and r['objective']==objective) for phase in ['basic','defense_eot_control','defense_adaptive']]
            bars(ax,['original K1','unknown K4','adaptive K4'],[('ASR',values,COLORS[0])],legend=False);ax.set_title(f'{defense}, s={size}')
        upper=max(ax.get_ylim()[1] for ax in axes.flat);[ax.set_ylim(0,upper) for ax in axes.flat]
        finish(fig,name)
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),layout='constrained')
    for row,objective in enumerate(['UASR','TASR macro']):
        for col,defense in enumerate(['gaussian','resize_pad']):
            values=[next(r for r in defense_rows if r['size']==7 and r['defense']==defense and r['path']==phase and r['objective']==objective) for phase in ['basic','defense_eot_control','defense_adaptive']]
            bars(axes[row,col],['K1','K4','adaptive'],[('ASR',values,COLORS[col])],legend=False);axes[row,col].set_title(defense+' / '+objective)
    finish(fig,'D_main')
    compact(fig,'D_main_report',6.5,2.45)
    fig,axes=plt.subplots(2,2,figsize=(8,5.2),layout='constrained')
    for ax,(size,defense) in zip(axes.flat,[(5,'gaussian'),(7,'gaussian'),(5,'resize_pad'),(7,'resize_pad')]):
        heat=np.array([[100*metric(phase,'R',MODELS[0],size,t,defense,eot)['mean'] for t in range(10)] for phase,eot in [('basic',1),('defense_eot_control',4),('defense_adaptive',4)]])
        im=ax.imshow(heat,vmin=0,vmax=100,cmap='YlOrRd',aspect='auto')
        ax.set_xticks(range(10),range(10),fontsize=9);ax.set_xlabel('Target class',fontsize=9)
        ax.set_yticks(range(3),['original K1','unknown K4','adaptive K4'],fontsize=9)
        ax.set_title(f'{defense}, s={size}');fig.colorbar(im,ax=ax,label='TASR (%)',shrink=.75)
    finish(fig,'D_all_targets')
    for defense in ('gaussian','resize_pad'):
        representative=next(r for r in data['per_seed_records'] if r.get('phase')=='defense_adaptive' and r['defense_eval']['type']==defense and r['draws']==8)
        defense_table.append(dict(defense=defense,config=representative['defense_eval'],clean_accuracy=representative['clean_acc_defense'],clean_drop_pp=representative['clean_drop_pp'],defense_only_error=representative['defense_only_error']))
    fig,ax=plt.subplots(figsize=(6.8,3.6),layout='constrained')
    markers={'basic':'o','defense_eot_control':'s','defense_adaptive':'^'}
    for defense,color in zip(['gaussian','resize_pad'],COLORS):
        cost=next(r['clean_drop_pp'] for r in defense_table if r['defense']==defense)
        for size in [5,7]:
            for phase in markers:
                r=next(v for v in defense_rows if v['size']==size and v['defense']==defense and v['path']==phase and v['objective']=='UASR')
                ax.errorbar(cost,100*r['mean'],yerr=100*r['sample_std'],marker=markers[phase],markersize=5 if size==5 else 8,color=color,capsize=2,linestyle='none')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color=color,linestyle='none',label=d) for d,color in zip(['Gaussian','Resize/pad'],COLORS)]
    handles.extend(Line2D([0],[0],marker=marker,color='#555555',linestyle='none',label=label) for label,marker in zip(['original K1','unknown K4','adaptive K4'],markers.values()))
    ax.legend(handles=handles,frameon=False,fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
    ax.set(xlabel='Clean accuracy loss (percentage points)',ylabel='Untargeted ASR (%)',title='Cost versus residual attack success (small/large: side 5/7)',ylim=(0,105));ax.grid(alpha=.15)
    finish(fig,'D_cost_tradeoff')
    fig,axes=plt.subplots(1,3,figsize=(8,2.8),layout='constrained')
    for ax,key,title in zip(axes,['defense_original','defense_gaussian','defense_resize'],['Training image 0','Gaussian (selected sigma)','Resize/pad (fixed demonstration)']):
        ax.imshow(assets[key],interpolation='nearest');ax.set_title(title,fontsize=8);ax.set_xticks([]);ax.set_yticks([])
    finish(fig,'D_operators')
    fig,axes=plt.subplots(2,2,figsize=(9,5.2),layout='constrained');strength=[]
    for col,defense in enumerate(['gaussian','resize_pad']):
        for row,target in enumerate([None,3]):
            values=[metric('defense_adaptive','R',MODELS[0],7,target,defense,4),metric('defense_strength','R',MODELS[0],7,target,defense,4),metric('defense_strength','R',MODELS[0],7,target,defense,8)]
            bars(axes[row,col],['1000/K4','2000/K4','1000/K8'],[('ASR',values,COLORS[col])],legend=False)
            axes[row,col].set_title(defense+' / '+('untargeted' if target is None else 'target cat (3)'))
            for setting,value in zip(['1000/K4','2000/K4','1000/K8'],values):strength.append(dict(defense=defense,target=target,setting=setting,**value))
    finish(fig,'D_strength_checks')
    compact(fig,'D_strength_checks_main',6.5,2.5)
    stability=[]
    for record in data['per_seed_records']:
        if record['draws']!=32:continue
        other=next(r for r in data['per_seed_records'] if r['run_id']==record['run_id'] and r['eval_model']==record['eval_model'] and r['defense_eval']==record['defense_eval'] and r['draws']==8)
        key='uasr' if record['target'] is None else 'tasr'
        stability.append(dict(run_id=record['run_id'],defense=record['defense_eval']['type'],target=record['target'],seed=record['seed'],asr8=other[key],asr32=record[key],difference_pp=100*(record[key]-other[key])))
    save_json('results/publication_data.json',dict(status='complete',models=models,basic=basic,controls=controls,targeted_controls=targeted_controls,ensemble=ensemble,
        defenses=defense_table,defense_paths=defense_rows,strength=strength,stability=stability,paired_effects=data['paired_effects'],
        coverage=dict(main_patches=561,strength_patches=24,full_validation_evaluations=1212,test_evaluations=1638,draws8_evaluations=1626,draws32_evaluations=12),
        time_budget=json.loads(Path('artifacts/time_budget.json').read_text()),defense_selection=json.loads(Path('artifacts/defenses.json').read_text()),
        visual_cases=visual['cases'],source_tradeoffs=source_tradeoffs,patch_characteristics=visual['patch_characteristics'],statistics=data['statistics'],
        resource_observations=json.loads(Path('results/resource_observations.json').read_text(encoding='utf-8')) if Path('results/resource_observations.json').exists() else None,
        figures=sorted(p.name for p in OUT.glob('*.png'))))
    print('Saved publication figures and report data from audited real results',flush=True)


if __name__=='__main__':main()
