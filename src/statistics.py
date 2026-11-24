"""Paired target-macro inference with correct image resampling units."""
import numpy as np

def target_macro(tasrs):
    values=[v for v in tasrs if v is not None]
    return float(np.mean(values)) if values else None

def seed_summary(values):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():raise ValueError('Undefined seed results cannot be silently pooled')
    return dict(mean=float(a.mean()),sample_std=float(a.std(ddof=1)) if len(a)>1 else None,n_seeds=len(a))

def paired_target_macro_bootstrap(hits_a,hits_b,labels,clean_correct,replicates=2000,seed=4242):
    # [targets=10, images, draws, seeds], arrays share the exact image/draw/seed order.
    a=np.asarray(hits_a,dtype=float);b=np.asarray(hits_b,dtype=float);labels=np.asarray(labels);eligible=np.asarray(clean_correct,dtype=bool)
    if a.shape!=b.shape or a.shape[:2]!=(10,len(labels)):raise ValueError('Expected aligned [10,images,draws,seeds]')
    delta=(a-b).mean(axis=tuple(range(2,a.ndim)))
    groups=[np.flatnonzero(labels==c) for c in np.unique(labels)];rng=np.random.default_rng(seed)
    def effect(ids):
        vals=[]
        for t in range(10):
            keep=ids[eligible[ids] & (labels[ids]!=t)]
            if len(keep):vals.append(delta[t,keep].mean())
        return np.mean(vals) if vals else np.nan
    base=effect(np.arange(len(labels)))
    if not np.isfinite(base):return dict(delta_pp=None,ci95_pp=None)
    estimates=[effect(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])) for _ in range(replicates)]
    return dict(delta_pp=100*float(base),ci95_pp=(100*np.nanquantile(estimates,[.025,.975])).tolist(),unit='stratified images, fixed patch seeds')
