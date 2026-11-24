import numpy as np
import torch
from torch.utils.data import DataLoader
from .patches.render import sample_patch_transforms,apply_patch
from .defenses import apply_defense,sample_defense_params
from .metrics import aggregate_metrics,validate_raw

def evaluate_patch(bundle,patch,dataset,seed=6610,draws=8,defense=None,target=None,batch_size=128):
    device=next(bundle.buffers()).device;fields={k:[] for k in ('image_id','true_label','draw_id','pred_clean','pred_patch','pred_defense_clean','alpha_area','support_area','defended_alpha_area','defended_support_area','angle','scale','cx','cy','defense_q','defense_top','defense_left')}
    with torch.no_grad():
        loader=dataset.sequential_batches(batch_size,device) if hasattr(dataset,'sequential_batches') and not dataset.augment else DataLoader(dataset,batch_size=batch_size,shuffle=False)
        for x,y,ids in loader:
            x=x.to(device);clean=bundle(x).argmax(1).cpu().numpy();ids=list(ids)
            for j in range(draws):
                if patch is None:
                    xp=x;alpha=x.new_zeros(len(x),1,32,32);meta=dict(alpha_area=x.new_zeros(len(x)),support_area=x.new_zeros(len(x)));ts=[dict(angle=0,scale=1,cx=0,cy=0)]*len(x)
                else:
                    ts=sample_patch_transforms(ids,j,seed,patch.shape[-1]);xp,alpha,meta=apply_patch(x,patch.to(device),ts)
                dp=sample_defense_params(ids,j,seed,defense or {})
                identity=not defense or defense.get('type','identity')=='identity'
                pc=clean if identity else bundle(apply_defense(x,defense,dp)).argmax(1).cpu().numpy()
                pp=pc if patch is None else bundle(apply_defense(xp,defense,dp)).argmax(1).cpu().numpy()
                affected=alpha if identity else apply_defense(alpha,defense,dp)
                rows=dict(image_id=ids,true_label=y.numpy(),draw_id=[j]*len(x),pred_clean=clean,pred_patch=pp,pred_defense_clean=pc,
                          alpha_area=meta['alpha_area'].cpu().numpy(),support_area=meta['support_area'].cpu().numpy(),
                          defended_alpha_area=affected.sum((1,2,3)).cpu().numpy(),defended_support_area=(affected>1e-6).sum((1,2,3)).cpu().numpy())
                rows.update({k:[t[k] for t in ts] for k in ('angle','scale','cx','cy')})
                rows.update({f'defense_{k}':[p[k] if p else -1 for p in dp] for k in ('q','top','left')})
                # Preserve every value and its dtype, avoiding millions of Python
                # scalar objects when collecting full 10,000-image evaluations.
                for k,v in rows.items():fields[k].append(np.asarray(v))
    raw={k:np.concatenate(v) for k,v in fields.items()};validate_raw(raw,draws)
    metrics=aggregate_metrics(raw,target);metrics['mean_alpha_area']=float(raw['alpha_area'].astype('float64').mean());metrics['mean_defended_alpha_area']=float(raw['defended_alpha_area'].astype('float64').mean());return raw,metrics
