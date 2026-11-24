"""Small synthetic integration run. NEVER CIFAR-10 research evidence."""
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from src.data import RawDataset
from src.model_bundle import ModelBundle
from src.patches.optimize import optimize_patch
from src.evaluate import evaluate_patch
from src.utils import save_json,environment

def main():
    torch.set_num_threads(2);out=Path('results/smoke');out.mkdir(parents=True,exist_ok=True)
    prototypes=torch.tensor([[.35,.35,.35],[.65,.35,.35],[.35,.65,.35],[.65,.65,.35],[.35,.35,.65],[.65,.35,.65],[.35,.65,.65],[.65,.65,.65],[.5,.5,.5],[.5,.5,.7]])
    def model(name,jitter):
        head=nn.Linear(3,10)
        with torch.no_grad():head.weight.copy_(prototypes*80+jitter);head.bias.copy_(-40*(prototypes**2).sum(1))
        return ModelBundle(nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),head),[0,0,0],[1,1,1],name,'synthetic-'+name).freeze()
    r=model('synthetic_R',0);v=model('synthetic_V',.03)
    rng=np.random.default_rng(661);labels=np.repeat(np.arange(10),8);images=np.clip(prototypes.numpy()[labels,:,None,None]+rng.normal(0,.025,(80,3,32,32)),0,1)
    images=np.round(images*255).astype('uint8');train=RawDataset(images[:60],labels[:60],split='synthetic_train')
    # Use disjoint examples covering every class in each split.
    tr=np.flatnonzero(np.arange(80)%8<4);va=np.flatnonzero(np.arange(80)%8>=4)
    train=RawDataset(images,labels,tr,split='synthetic_train');val=RawDataset(images,labels,va,split='synthetic_val')
    jobs=[('B_target',[r],dict(target=3,eot=1)),('B_untargeted',[r],dict(target=None,eot=1)),('C_ensemble',[r,v],dict(target=3,eot=1)),('D_gaussian',[r],dict(target=3,eot=4,defense={'type':'gaussian','sigma':.5})),('D_resize',[r],dict(target=3,eot=4,defense={'type':'resize_pad','sizes':[28,30,32]}))]
    summary=[]
    for name,bs,extra in jobs:
        cfg=dict(size=16,seed=101,batch_size=20,max_steps=40,validate_every=10,validation_draws=2,lr=.05,**extra)
        obj=optimize_patch(bs,train,val,cfg,out/name)
        raw,m=evaluate_patch(r,obj['patch_tensor'],val,seed=6611,draws=4,defense=extra.get('defense'),target=extra['target'],batch_size=20)
        np.savez_compressed(out/f'{name}.npz',**raw)
        summary.append(dict(name=name,metrics=m,history=obj['validation_history'],training_seconds=obj['wall_time_seconds'],selected_step=obj['selected_step']))
        print(name,m,flush=True)
    save_json(out/'summary.json',dict(status='SYNTHETIC ENGINEERING CHECK ONLY; NOT CIFAR-10',n_train=len(train),n_val=len(val),environment=environment(),runs=summary))
    np.savez_compressed(out/'example_inputs.npz',images=images,labels=labels)
if __name__=='__main__':main()
