"""Unified auditable CLI. Run python -m scripts.workflow --help."""
import argparse,csv,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from src.data import prepare,load_arrays,make_splits,normalization,RawDataset
from src.models import FACTORY
from src.model_bundle import load_model_bundle
from src.matrix import task_matrix
from src.patches.optimize import optimize_patch
from src.evaluate import evaluate_patch
from src.utils import save_json,load_torch,digest,environment
from src.metrics import aggregate_metrics,validate_raw
from src.audit import audit_matrix,validate_patch
from src.evaluation_audit import audit_evaluations

def context(a):
    p=Path(a.artifacts)/'splits/manifest.json';sp=json.loads(p.read_text());x,y,xt,yt=load_arrays(a.data)
    return sp,RawDataset(x,y,sp['train']),RawDataset(x,y,sp['selection']),RawDataset(x,y,sp['val']),RawDataset(xt,yt,split='test')
def bundles(a,strategy):
    names={'R':['resnet20'],'V':['vgg11_bn_cifar'],'RV':['resnet20','vgg11_bn_cifar']}[strategy]
    return [load_model_bundle(Path(a.artifacts)/f'models/{name}/best.pt',a.device) for name in names]
def matrix(a):
    dp=Path(a.artifacts)/'defenses.json';tp=Path(a.artifacts)/'time_budget.json'
    defenses=json.loads(dp.read_text())['selected'] if dp.exists() else None
    budget=json.loads(tp.read_text())['time_budget_s'] if tp.exists() else None
    return task_matrix(budget,defenses,a.include_strength)
def freeze_record(a):
    p=Path(a.artifacts);models={name:digest(p/f'models/{name}/best.pt') for name in FACTORY}
    return dict(models=models,split=digest(p/'splits/manifest.json'),patches={r['run_id']:digest(p/'patches'/r['run_id']/'patch.pt') for r in matrix(a)},defenses=digest(p/'defenses.json'),time_budget=digest(p/'time_budget.json'),protocol=digest('configs/protocol.json'),development=digest(p/'development.json'),qualitative_examples=digest('configs/qualitative_examples.json'),evaluation_batch=digest(p/'evaluation_batch.json') if (p/'evaluation_batch.json').exists() else None)
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','matrix','optimize','select-defenses','benchmark','freeze','evaluate','controls','audit','summarize']);p.add_argument('--data',default='data');p.add_argument('--artifacts',default='artifacts');p.add_argument('--results',default='results');p.add_argument('--device',default='cpu');p.add_argument('--phase',default='all');p.add_argument('--run-id');p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--final',action='store_true');p.add_argument('--include-strength',action='store_true');p.add_argument('--threads',type=int,default=4);p.add_argument('--draws',type=int,choices=[8,32],default=8);p.add_argument('--stability-only',action='store_true');a=p.parse_args();torch.set_num_threads(a.threads)
    art=Path(a.artifacts);res=Path(a.results);res.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':
        x,y,xt,yt=prepare(a.data);sp=make_splits(y);sp['mean'],sp['std']=normalization(x,sp['train']);save_json(art/'splits/manifest.json',sp);save_json(art/'environment.json',environment());return
    rows=matrix(a)
    if a.command=='matrix':save_json(art/'matrix.json',rows);print(Counter(r['phase'] for r in rows));return
    if a.command=='audit':
        report=audit_matrix(rows,art)
        save_json(res/'audit.json',report);print(Counter(r['status'] for r in report));return
    if a.command=='summarize':
        records=[]
        for path in sorted((res/'raw').glob('*.npz')):
            raw=dict(np.load(path,allow_pickle=False));meta=json.loads(path.with_suffix('.json').read_text())
            if digest(path)!=meta['raw_sha256']:raise ValueError('Raw result hash mismatch')
            validate_raw(raw,meta['draws']);m=aggregate_metrics(raw,meta.get('target'))
            for k,v in m.items():
                if meta['metrics'][k]!=v:raise ValueError('Saved summary disagrees with raw predictions: '+k)
            records.append(dict(**meta,**m))
        if not records:raise RuntimeError('No measured raw results; refusing to create result tables')
        cols=sorted(set().union(*(r.keys() for r in records)))
        with open(res/'summary.csv','w') as f:
            writer=csv.DictWriter(f,cols);writer.writeheader();writer.writerows(records)
        print(f'{len(records)} raw evaluations recomputed');return
    if a.command=='freeze':
        for name in FACTORY:
            if not load_torch(art/f'models/{name}/best.pt').get('formal_training'):raise RuntimeError('Formal 200-epoch classifiers required')
            completed=art/f'models/{name}/training_complete.json'
            if not completed.exists():raise RuntimeError('Classifier run has not completed: '+name)
            record=json.loads(completed.read_text())
            if record['epochs_completed']!=200 or record['best_checkpoint_sha256']!=digest(art/f'models/{name}/best.pt'):raise RuntimeError('Incomplete or changed formal classifier: '+name)
        report=audit_matrix(rows,art)
        save_json(res/'audit.json',report)
        invalid=[r for r in report if r['status']!='complete']
        if invalid:raise RuntimeError('Incomplete/invalid matrix: '+invalid[0]['run_id'])
        if not (art/'development.json').exists():raise RuntimeError('Attack development validation has not been completed')
        # Complete the protocol's full 5,000-image development validation before
        # opening test evaluation. Candidates and selection histories stay fixed.
        import subprocess,sys
        subprocess.run([sys.executable,'-u','-m','scripts.ensure_full_validation'],check=True)
        save_json(art/'freeze.json',freeze_record(a));return
    if a.final:
        frozen=art/'freeze.json'
        if not frozen.exists() or json.loads(frozen.read_text())!=freeze_record(a):raise RuntimeError('Freeze record missing or changed')
    sp,train,selection,val,test=context(a)
    if a.command=='select-defenses':
        b=bundles(a,'R')[0];_,base=evaluate_patch(b,None,val,seed=6611,draws=1);selected=[];log=[]
        families=[[dict(type='gaussian',sigma=s) for s in [.25,.5,.75]],[dict(type='resize_pad',sizes=s) for s in [[30,31,32],[28,30,32]]]]
        for candidates in families:
            chosen=candidates[0];valid=[]
            for cfg in candidates:
                _,m=evaluate_patch(b,None,val,seed=6611,defense=cfg);drop=100*(base['clean_acc']-m['clean_acc_defense']);log.append(dict(config=cfg,clean_drop_pp=drop))
                if drop<=2:valid.append(cfg)
            # Protocol D.3/D.4 requires retaining the weakest candidate when all
            # exceed the cost limit; the violation must remain visible in the log.
            chosen=valid[-1] if valid else candidates[0]
            log.append(dict(family=chosen['type'],selected=chosen,within_clean_drop_limit=bool(valid)))
            selected.append(chosen)
        save_json(art/'defenses.json',dict(selected=selected,validation=log,checkpoint_hash=b.checkpoint_hash));return
    if a.command=='benchmark':
        import tempfile
        bs=bundles(a,'RV');cfg=dict(size=7,target=3,seed=999,eot=1,max_steps=20,batch_size=128,validation_draws=1,validate_every=100)
        # Independent warmup and benchmark states; no trained warmup patch reused.
        with tempfile.TemporaryDirectory() as d:optimize_patch(bs,train,selection,cfg,d)
        times=[]
        for rep in range(3):
            with tempfile.TemporaryDirectory() as d:
                cfg['max_steps']=50;cfg['seed']=10000+rep;obj=optimize_patch(bs,train,selection,cfg,d);times.append(obj['wall_time_seconds']/50)
        save_json(art/'time_budget.json',dict(time_budget_s=1000*float(np.median(times)),seconds_per_update=times,device=a.device,environment=environment()));return
    if a.command=='optimize':
        if not a.execute:print('Dry run: use --execute to run optimization');return
        for r in rows:
            if a.phase!='all' and r['phase']!=a.phase:continue
            if a.run_id and r['run_id']!=a.run_id:continue
            if r.get('time_budget_s',0)<0:raise RuntimeError('Calibrate time budget first')
            if r['phase'].startswith('defense') and r['phase']!='defense_eot_control' and not (art/'defenses.json').exists():raise RuntimeError('Select defenses on clean validation first')
            cfg=dict(r,split_sha256=digest(art/'splits/manifest.json'),protocol_hash=digest('configs/protocol.json'))
            path=art/'patches'/r['run_id'];final=path/'patch.pt'
            if final.exists():
                validate_patch(final,r,art)
                continue
            print('RUN',r['run_id'],flush=True);optimize_patch(bundles(a,r['strategy']),train,selection,cfg,path,a.resume)
        return
    ds=test if a.final else val;phase='test' if a.final else 'validation';seed=6610 if a.final else 6611
    evaluation_batch=json.loads((art/'evaluation_batch.json').read_text())['batch_size'] if (art/'evaluation_batch.json').exists() else 128
    def save_eval(b,patch,target,meta,defense=None):
        key=meta['run_id']+'__'+b.name+'__'+(defense or {}).get('type','identity')+'__'+phase+f'__draws{a.draws}'
        dest=res/('raw' if a.final else 'validation_raw')/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        identity=dict(**meta,eval_model=b.name,checkpoint_hash=b.checkpoint_hash,draws=a.draws,phase_eval=phase,defense_eval=defense or {'type':'identity'},eval_seed=seed,split_sha256=digest(art/'splits/manifest.json'),protocol_hash=digest('configs/protocol.json'),n_dataset_images=len(ds),evaluation_batch_size=evaluation_batch)
        side=dest.with_suffix('.json')
        if dest.exists() and side.exists():
            cached=json.loads(side.read_text())
            if any(cached.get(k)!=v for k,v in identity.items()) or cached['raw_sha256']!=digest(dest):raise ValueError('Changed evaluation cache identity: '+key)
            raw=dict(np.load(dest,allow_pickle=False));validate_raw(raw,a.draws)
            if len(np.unique(raw['image_id']))!=len(ds):raise ValueError('Incomplete cached image set')
            print('Verified cached evaluation '+key,flush=True);return
        raw,m=evaluate_patch(b,patch,ds,seed=seed,draws=a.draws,defense=defense,target=target,batch_size=evaluation_batch)
        np.savez_compressed(dest,**raw)
        save_json(side,dict(**identity,metrics=m,raw_sha256=digest(dest)))
        print('Evaluated '+key,flush=True)
    if a.command=='controls':
        for name in FACTORY:
            if name=='densenet_bc40_k12' and not a.final:continue
            b=load_model_bundle(art/f'models/{name}/best.pt',a.device)
            save_eval(b,None,None,dict(run_id='clean',target=None))
            for size in [3,5,7,16]:
                mean=torch.tensor(sp['mean']).view(1,3,1,1).expand(1,3,size,size).clone();save_eval(b,mean,None,dict(run_id=f'mean_s{size}',target=None,size=size,control='mean'))
                for s in [101,102,103]:
                    gen=torch.Generator().manual_seed(s);patch=torch.rand(1,3,size,size,generator=gen);save_eval(b,patch,None,dict(run_id=f'random_s{size}_seed{s}',target=None,size=size,seed=s,control='random'))
        return
    if a.command=='evaluate':
        defense_cfgs=json.loads((art/'defenses.json').read_text())['selected'] if (art/'defenses.json').exists() else []
        for r in rows:
            if a.phase!='all' and r['phase']!=a.phase:continue
            if a.run_id and r['run_id']!=a.run_id:continue
            if a.stability_only and not (r['phase']=='defense_adaptive' and r['size']==7 and r['target'] in (None,3)):continue
            path=art/'patches'/r['run_id']/'patch.pt'
            if not path.exists():raise RuntimeError('Missing patch '+str(path))
            obj=load_torch(path);patch=obj['patch_tensor'];meta=dict(r,patch_hash=digest(path))
            names=list(FACTORY) if r['phase'] in ['basic','ensemble_steps','ensemble_time'] else ['resnet20']
            for name in names:
                if name=='densenet_bc40_k12' and not a.final:continue
                b=load_model_bundle(art/f'models/{name}/best.pt',a.device);dspec=[None]
                if name=='resnet20' and r['size'] in [5,7] and r['phase'] in ['basic','defense_eot_control']:dspec+=defense_cfgs
                if r['defense']:dspec=[r['defense']]
                for d in dspec:save_eval(b,patch,r['target'],meta,d)
if __name__=='__main__':main()
