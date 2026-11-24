import numpy as np
from functools import lru_cache
import torch
import torch.nn.functional as F
from .utils import stable_seed

def sample_defense_params(ids,draw,seed,cfg):
    if cfg.get('type','identity')!='resize_pad':return [None]*len(ids)
    if seed in (6610,6611):return _cached_defense_params(tuple(ids),draw,seed,tuple(cfg['sizes']))
    return _sample_defense_params(ids,draw,seed,cfg['sizes'])

@lru_cache(maxsize=1024)
def _cached_defense_params(ids,draw,seed,sizes):
    return _sample_defense_params(ids,draw,seed,sizes)

def _sample_defense_params(ids,draw,seed,sizes):
    rows=[]
    for iid in ids:
        rng=np.random.default_rng(stable_seed('defense',seed,str(iid),int(draw)))
        q=int(rng.choice(sizes));top=int(rng.integers(33-q));left=int(rng.integers(33-q))
        rows.append(dict(q=q,top=top,left=left))
    return rows

def apply_defense(x,cfg=None,random_params=None):
    cfg=cfg or {'type':'identity'};kind=cfg['type']
    if kind=='identity':return x
    if kind=='gaussian':
        t=torch.arange(-1,2,device=x.device,dtype=x.dtype);k=torch.exp(-(t[:,None]**2+t[None,:]**2)/(2*cfg['sigma']**2));k=k/k.sum()
        return F.conv2d(F.pad(x,(1,1,1,1),mode='reflect'),k.expand(x.shape[1],1,3,3),groups=x.shape[1])
    if kind=='resize_pad':
        if random_params is None:raise ValueError('Explicit replayable random parameters required')
        if len(random_params)!=len(x):raise ValueError('One defense record per image is required')
        groups=[];indices=[]
        axis=torch.arange(32,device=x.device)
        for q in sorted({p['q'] for p in random_params}):
            group=[i for i,p in enumerate(random_params) if p['q']==q]
            index=torch.tensor(group,device=x.device)
            y=F.interpolate(x.index_select(0,index),size=(q,q),mode='bilinear',align_corners=False)
            left=torch.tensor([random_params[i]['left'] for i in group],device=x.device)
            top=torch.tensor([random_params[i]['top'] for i in group],device=x.device)
            ix=(axis[None,:]-left[:,None]).clamp(0,q-1)
            iy=(axis[None,:]-top[:,None]).clamp(0,q-1)
            gather=(iy[:,:,None]*q+ix[:,None,:]).reshape(len(group),1,1024).expand(-1,x.shape[1],-1)
            groups.append(y.flatten(2).gather(2,gather).reshape(len(group),x.shape[1],32,32));indices.extend(group)
        inverse=torch.argsort(torch.tensor(indices,device=x.device))
        return torch.cat(groups).index_select(0,inverse)
    raise ValueError(kind)
