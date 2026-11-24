"""Audited statistics and paired, class-stratified image bootstrap from real records."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.utils import digest,save_json,load_torch
from src.metrics import aggregate_metrics,validate_raw

ROOT=Path('.')
MODELS=['resnet20','vgg11_bn_cifar','densenet_bc40_k12']


def read_record(path):
    meta=json.loads(path.with_suffix('.json').read_text())
    if digest(path)!=meta['raw_sha256']:raise ValueError('Raw checksum changed: '+path.name)
    raw=dict(np.load(path,allow_pickle=False));validate_raw(raw,meta['draws'])
    ids=np.asarray([int(v.split(':')[1]) for v in raw['image_id']])
    order=np.lexsort((raw['draw_id'],ids));n=meta['n_dataset_images'];d=meta['draws']
    if not np.array_equal(ids[order].reshape(n,d)[:,0],np.arange(n)):raise ValueError('Expected official test ordering')
    arrays={k:raw[k][order].reshape(n,d) for k in ['true_label','pred_clean','pred_patch','pred_defense_clean','alpha_area','support_area','defended_alpha_area','defended_support_area']}
    arrays['labels']=arrays['true_label'][:,0]
    arrays['eligible']=arrays['pred_clean'][:,0]==arrays['labels']
    metrics=aggregate_metrics(raw,meta.get('target'))
    if any(meta['metrics'][k]!=v for k,v in metrics.items()):raise ValueError('Saved metrics differ from raw record')
    return meta,arrays


def bootstrap_weights(labels,replicates=2000,seed=4242):
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(labels==c) for c in range(10)]
    weights=np.zeros((replicates,len(labels)),dtype=np.uint8)
    for rep in range(replicates):
        for group in groups:
            selected=rng.choice(group,len(group),replace=True)
            weights[rep]+=np.bincount(selected,minlength=len(labels)).astype(np.uint8)
    return weights


def paired_inference(a,b,labels,clean,weights,targets):
    # hits: [targets,images,draws,3 patch seeds]. Average draws and fixed seeds.
    if a.shape!=b.shape or a.shape[-1]!=3:raise ValueError('Three aligned patch seeds are required')
    delta=(a.astype(np.float64)-b.astype(np.float64)).mean(axis=2)
    eligible=np.stack([clean & (labels!=t) if t is not None else clean for t in targets])
    # NumPy's MKL and this nightly torch wheel ship different OpenMP runtimes.
    # The C einsum path avoids loading a second BLAS/OpenMP runtime in one process.
    numerators=np.einsum('ti,bi->tb',delta.mean(axis=2)*eligible,weights,dtype=np.float64,optimize=False)
    denominators=np.einsum('ti,bi->tb',eligible.astype(np.float64),weights,dtype=np.float64,optimize=False)
    per_target=np.divide(numerators,denominators,out=np.full_like(numerators,np.nan),where=denominators>0)
    estimates=np.nanmean(per_target,axis=0)
    seed_values=np.stack([delta[j,eligible[j],:].mean(axis=0) for j in range(len(targets))]).mean(axis=0)*100
    return dict(delta_pp=float(seed_values.mean()),seed_deltas_pp=seed_values.tolist(),sample_std_pp=float(seed_values.std(ddof=1)),
                ci95_pp=(100*np.nanquantile(estimates,[.025,.975])).tolist(),bootstrap_replicates=len(weights),bootstrap_seed=4242,
                bootstrap_unit='class-stratified images, all draws and fixed three patch seeds retained',
                uncertainty_scope='fixed classifier checkpoints and three optimized patches; excludes classifier-training uncertainty')


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('Refusing an empty scientific table')
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='',encoding='utf-8-sig') as output:
        writer=csv.DictWriter(output,keys);writer.writeheader()
        writer.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])


def main():
    audit=json.loads(Path('results/evaluation_audit.json').read_text())
    if not audit['complete'] or audit['expected_evaluations']!=1638:raise RuntimeError('Complete the frozen final evaluation audit first')
    validation=json.loads(Path('results/validation_audit.json').read_text())
    if not validation['complete'] or validation['expected_evaluations']!=1212:raise RuntimeError('Full validation audit is required')
    folder=Path('results/raw');index={};rows=[];baseline={};control_targets=[];patch_info={}
    for path in sorted(folder.glob('*.npz')):
        meta,arrays=read_record(path)
        if meta['phase_eval']!='test':continue
        defense=meta['defense_eval']['type'];model=meta['eval_model'];run=meta['run_id']
        index[(run,model,defense,meta['draws'])]=path
        row={k:v for k,v in meta.items() if k!='metrics'};row.update(meta['metrics'])
        row['raw_file']=path.name;row['objective']='untargeted' if meta.get('target') is None else 'targeted'
        row['mean_alpha_area']=float(arrays['alpha_area'].astype(np.float64).mean());row['mean_support_area']=float(arrays['support_area'].astype(np.float64).mean())
        row['mean_defended_alpha_area']=float(arrays['defended_alpha_area'].astype(np.float64).mean());row['mean_defended_support_area']=float(arrays['defended_support_area'].astype(np.float64).mean())
        if 'phase' in meta:
            if run not in patch_info:
                obj=load_torch(Path('artifacts/patches')/run/'patch.pt')
                patch_info[run]=dict(wall_time_s=obj['wall_time_seconds'],validation_time_s=obj['validation_time_seconds'],
                    end_to_end_time_s=obj['end_to_end_seconds'],optimizer_steps=obj['total_steps'],selected_step=obj['selected_step'],
                    forward_backward_calls=obj['forward_backward_calls'],peak_cuda_memory_bytes=obj.get('peak_cuda_memory_bytes'),
                    warmup_time_s=obj.get('warmup_seconds'),processed_training_images=obj.get('processed_training_images'),
                    time_budget_overshoot_s=obj['wall_time_seconds']-obj['cfg']['time_budget_s'] if obj['cfg'].get('time_budget_s') is not None else None,
                    elapsed_time_scope='optimization function entry through final artifact assembly; excludes external model/data loading and final serialization')
            row.update(patch_info[run])
        rows.append(row)
        if run=='clean':
            baseline[model]=dict(labels=arrays['labels'],eligible=arrays['eligible'])
        if meta.get('control') and meta['draws']==8:
            for target in range(10):
                keep=arrays['eligible'] & (arrays['labels']!=target)
                control_targets.append(dict(run_id=run,model=model,size=meta['size'],control=meta['control'],seed=meta.get('seed'),
                    target=target,tasr=float((arrays['pred_patch'][keep]==target).mean()),n_eligible=int(keep.sum())))
    write_csv('results/tables/per_seed_metrics.csv',rows)
    write_csv('results/tables/control_target_metrics.csv',control_targets)
    groups=defaultdict(list)
    for row in rows:
        if row['draws']!=8 or 'phase' not in row:continue
        key=(row['phase'],row['strategy'],row['eval_model'],row['size'],row['target'],row['defense_eval']['type'],row['eot'])
        groups[key].append(row)
    summaries=[]
    for key,items in groups.items():
        phase,strategy,model,size,target,defense,eot=key
        if sorted(r['seed'] for r in items)!=[101,102,103]:raise ValueError('Missing/duplicate seed in scientific group')
        items=sorted(items,key=lambda r:r['seed'])
        values=[r['tasr'] if target is not None else r['uasr'] for r in items]
        summaries.append(dict(phase=phase,strategy=strategy,model=model,size=size,target=target,defense=defense,eot=eot,
            metric='TASR' if target is not None else 'UASR',mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),
            n_seeds=3,seed_values=values,patched_accuracy_mean=float(np.mean([r['patched_acc'] for r in items])),
            optimization_time_mean_s=float(np.mean([r['wall_time_s'] for r in items]))))
    write_csv('results/tables/seed_summaries.csv',summaries)
    macros=[]
    target_groups=defaultdict(list)
    for key,items in groups.items():
        phase,strategy,model,size,target,defense,eot=key
        if target is None or phase=='defense_strength':continue
        for item in items:target_groups[(phase,strategy,model,size,defense,eot,item['seed'])].append(item)
    for key,items in target_groups.items():
        if sorted(r['target'] for r in items)!=list(range(10)):raise ValueError('Incomplete ten-target macro average')
        phase,strategy,model,size,defense,eot,seed=key
        macros.append(dict(phase=phase,strategy=strategy,model=model,size=size,defense=defense,eot=eot,seed=seed,
            tasr_macro=float(np.mean([r['tasr'] for r in items])),aggregation='equal weight over all ten target classes'))
    write_csv('results/tables/target_macros_per_seed.csv',macros)
    def run_id(phase,strategy,size,target,seed,eot=1,defense=None):
        return f'{phase}__{strategy}__s{size}__t{target}__seed{seed}__k{eot}'+('' if defense is None else '__'+defense)
    def hits(phase,strategy,size,targets,model,defense,eot=1):
        target_hits=[]
        for target in targets:
            seed_hits=[]
            for seed in (101,102,103):
                run=run_id(phase,strategy,size,target,seed,eot,defense if phase=='defense_adaptive' else None)
                _,data=read_record(index[(run,model,defense,8)])
                if not np.array_equal(data['eligible'],baseline[model]['eligible']):raise ValueError('Unpaired clean eligibility')
                seed_hits.append(data['pred_patch']!=data['true_label'] if target is None else data['pred_patch']==target)
            target_hits.append(np.stack(seed_hits,axis=-1))
        return np.stack(target_hits)
    labels=baseline[MODELS[0]]['labels'];weights=bootstrap_weights(labels)
    comparisons=[]
    for phase,sizes in [('ensemble_steps',[5,7]),('ensemble_time',[7])]:
        for size in sizes:
            for targets,objective in [([None],'untargeted'),(list(range(10)),'target_macro')]:
                a=hits(phase,'RV',size,targets,MODELS[2],'identity')
                for comparator in ('R','V'):
                    source_phase='basic' if phase=='ensemble_steps' and comparator=='R' else phase
                    b=hits(source_phase,comparator,size,targets,MODELS[2],'identity')
                    effect=paired_inference(a,b,labels,baseline[MODELS[2]]['eligible'],weights,targets)
                    comparisons.append(dict(part='C',phase=phase,size=size,objective=objective,model=MODELS[2],comparison='RV - '+comparator,**effect))
                    print('Computed C '+phase+' '+str(size)+' '+objective+' RV-'+comparator,flush=True)
    for size in (5,7):
        for defense in ('gaussian','resize_pad'):
            for targets,objective in [([None],'untargeted'),(list(range(10)),'target_macro')]:
                a=hits('defense_adaptive','R',size,targets,MODELS[0],defense,4)
                b=hits('defense_eot_control','R',size,targets,MODELS[0],defense,4)
                c=hits('basic','R',size,targets,MODELS[0],defense)
                for left,right,title in [(a,b,'adaptive K4 - unknown K4'),(b,c,'unknown K4 - original K1'),(a,c,'adaptive K4 - original K1')]:
                    effect=paired_inference(left,right,labels,baseline[MODELS[0]]['eligible'],weights,targets)
                    comparisons.append(dict(part='D',phase='defense',size=size,defense=defense,objective=objective,model=MODELS[0],comparison=title,**effect))
                    print('Computed D '+str(size)+' '+defense+' '+objective+' '+title,flush=True)
    write_csv('results/tables/paired_effects.csv',comparisons)
    save_json('results/analysis.json',dict(status='complete',raw_evaluations=len(rows),per_seed_records=rows,seed_summaries=summaries,
        target_macros_per_seed=macros,paired_effects=comparisons,control_target_metrics=control_targets,
        statistics=dict(patch_seeds=[101,102,103],standard_deviation_ddof=1,bootstrap_replicates=2000,bootstrap_seed=4242,
        bootstrap_unit='class-stratified images; paired draws and fixed patch seeds retained',test_images=10000)))


if __name__=='__main__':main()
