import numpy as np

def aggregate_metrics(raw,target=None):
    y=np.asarray(raw['true_label']);clean=np.asarray(raw['pred_clean']);pred=np.asarray(raw['pred_patch'])
    dc=np.asarray(raw.get('pred_defense_clean',clean));eligible=clean==y
    targeted=eligible & (y!=target) if target is not None else np.zeros_like(eligible)
    mean=lambda a,m:float(np.mean(a[m])) if m.any() else None
    err=pred!=y;de=dc!=y
    out=dict(clean_acc=float(np.mean(clean==y)),clean_acc_defense=float(np.mean(dc==y)),patched_acc=float(np.mean(pred==y)),
       uasr=mean(err,eligible),tasr=mean(pred==target,targeted) if target is not None else None,
       n_images=int(len(np.unique(raw['image_id']))),n_eligible=int(len(np.unique(np.asarray(raw['image_id'])[targeted if target is not None else eligible]))),
       n_eligible_untargeted=int(len(np.unique(np.asarray(raw['image_id'])[eligible]))),
       n_eligible_targeted=int(len(np.unique(np.asarray(raw['image_id'])[targeted]))) if target is not None else None,
       defense_only_error=mean(de,eligible),net_error_increase=mean(err.astype(float)-de.astype(float),eligible),
       target_rate_defense_clean=mean(dc==target,targeted) if target is not None else None,
       net_target_increase=mean((pred==target).astype(float)-(dc==target).astype(float),targeted) if target is not None else None,
       clean_drop_pp=100*float(np.mean(clean==y)-np.mean(dc==y)))
    return out

def validate_raw(raw,draws=None):
    ids=np.asarray(raw['image_id']);d=np.asarray(raw['draw_id'])
    if len(ids)==0:raise ValueError('Empty evaluation')
    if any(len(v)!=len(ids) for v in raw.values()):raise ValueError('Unequal raw field lengths')
    order=np.lexsort((d,ids));si=ids[order];sd=d[order];same=si[1:]==si[:-1]
    if np.any(same & (sd[1:]==sd[:-1])):raise ValueError('Duplicate image/draw pair')
    if draws is not None:
        _,counts=np.unique(ids,return_counts=True)
        if np.any(counts!=draws) or not np.isin(d,np.arange(draws)).all():raise ValueError('Missing draws')
    for key in ('true_label','pred_clean'):
        values=np.asarray(raw[key])[order]
        if np.any(same & (values[1:]!=values[:-1])):raise ValueError('Inconsistent clean reference')
    for k in ('true_label','pred_clean','pred_patch','pred_defense_clean'):
        if k in raw and not np.isin(raw[k],np.arange(10)).all():raise ValueError('Class out of range')

def paired_bootstrap(values_a,values_b,labels,replicates=2000,seed=4242):
    # arrays [images,draws,seeds]; average draws and fixed patch seeds first.
    delta=np.asarray(values_a,dtype=float)-np.asarray(values_b,dtype=float)
    if delta.ndim>1:delta=delta.mean(tuple(range(1,delta.ndim)))
    labels=np.asarray(labels);rng=np.random.default_rng(seed);groups=[np.flatnonzero(labels==c) for c in np.unique(labels)]
    draws=[delta[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])].mean() for _ in range(replicates)]
    return dict(delta_pp=100*float(delta.mean()),ci95_pp=(100*np.quantile(draws,[.025,.975])).tolist(),unit='images; fixed model and patch seeds')
