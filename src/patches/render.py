import math
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from ..utils import stable_seed

def sample_patch_transforms(image_ids,draw_ids,seed,size,rotation=15,scale_range=(.9,1.1),stream='eval_patch'):
    """Pixel centres at integers. Include the full bilinear interpolation footprint."""
    if isinstance(draw_ids,int):draw_ids=[draw_ids]*len(image_ids)
    if stream=='eval_patch':return _cached_evaluation_transforms(tuple(image_ids),tuple(draw_ids),seed,size,rotation,tuple(scale_range))
    return _sample_transforms(image_ids,draw_ids,seed,size,rotation,scale_range,stream)

@lru_cache(maxsize=1024)
def _cached_evaluation_transforms(image_ids,draw_ids,seed,size,rotation,scale_range):
    return _sample_transforms(image_ids,draw_ids,seed,size,rotation,scale_range,'eval_patch')

def _sample_transforms(image_ids,draw_ids,seed,size,rotation,scale_range,stream):
    out=[]
    for iid,d in zip(image_ids,draw_ids):
        rng=np.random.default_rng(stable_seed(stream,seed,str(iid),int(d)))
        a=math.radians(rng.uniform(-rotation,rotation));s=rng.uniform(*scale_range)
        # Input grid has nonzero interpolation support out to (size+1)/2.
        h=(size+1)*s/2*(abs(math.cos(a))+abs(math.sin(a)))
        offset=(size-1)%2/2
        lo=math.ceil(h-.5-offset);hi=math.floor(31.5-h-offset)
        if lo>hi:raise ValueError('Patch interpolation footprint cannot fit')
        cx=float(rng.integers(lo,hi+1))+offset;cy=float(rng.integers(lo,hi+1))+offset
        out.append(dict(angle=a,scale=s,cx=cx,cy=cy))
    return out

def apply_patch(images,patch,transforms):
    n,_,h,w=images.shape;s=patch.shape[-1]
    yy,xx=torch.meshgrid(torch.arange(h,device=images.device,dtype=images.dtype),torch.arange(w,device=images.device,dtype=images.dtype),indexing='ij')
    if len(transforms)!=n:raise ValueError('One transform per image is required')
    # Batch the same inverse affine calculation; avoid hundreds of small CUDA launches.
    params=images.new_tensor([[t['cx'],t['cy'],math.cos(t['angle']),math.sin(t['angle']),t['scale']] for t in transforms])
    cx,cy,cosine,sine,scale=[params[:,i,None,None] for i in range(5)]
    dx=xx[None]-cx;dy=yy[None]-cy
    u=(cosine*dx+sine*dy)/scale
    v=(-sine*dx+cosine*dy)/scale
    grid=torch.stack((2*u/s,2*v/s),-1)
    # RGB is already premultiplied by the all-one local alpha. Sample once.
    premult=F.grid_sample(patch.expand(n,-1,-1,-1),grid,align_corners=False)
    alpha=F.grid_sample(torch.ones((n,1,s,s),device=images.device,dtype=images.dtype),grid,align_corners=False)
    out=images*(1-alpha)+premult
    return out,alpha,dict(transforms=transforms,alpha_area=alpha.sum((1,2,3)),support_area=(alpha>1e-6).sum((1,2,3)))
