import time
from pathlib import Path
import numpy as np
import torch
from ..data import batch_at
from ..utils import stable_seed,sync,save_torch,load_torch,environment
from ..ensemble import ensemble_loss
from ..evaluate import evaluate_patch
from ..defenses import apply_defense,sample_defense_params
from .render import sample_patch_transforms,apply_patch

def optimize_patch(bundles,train,selection,cfg,output,resume=False):
    function_start=time.perf_counter()
    if any(b.name=='densenet_bc40_k12' for b in bundles):raise ValueError('Held-out model forbidden')
    for b in bundles:b.freeze()
    device=next(bundles[0].buffers()).device;seed=cfg['seed'];size=cfg['size'];K=cfg.get('eot',1)
    gen=torch.Generator().manual_seed(stable_seed('initialization',seed))
    patch=torch.nn.Parameter(torch.rand((1,3,size,size),generator=gen).to(device));optim=torch.optim.Adam([patch],lr=cfg.get('lr',.05))
    out=Path(output);out.mkdir(parents=True,exist_ok=True);state_path=out/'resume.pt'
    history=[];start=0;spent=0.;val_time=0.;best=-float('inf');best_patch=patch.detach().cpu().clone();best_step=0;calls=0
    identities=[b.checkpoint_hash for b in bundles]
    if resume and state_path.exists():
        st=load_torch(state_path,device)
        if st['cfg']!=cfg or st['source_hashes']!=identities:raise ValueError('Resume identity/config mismatch')
        patch.data.copy_(st['patch']);optim.load_state_dict(st['optimizer']);start=st['step'];history=st['history'];best=st['best'];best_patch=st['best_patch'];best_step=st['best_step'];spent=st['spent'];calls=st['calls'];val_time=st['validation_time']
    # Warm the complete representative CUDA chain outside the measured budget.
    # This has independent pixels/Adam state/streams and never advances the main run.
    warmup_seconds=0.
    if str(device).startswith('cuda'):
        sync(device);warm_start=time.perf_counter()
        warm_patch=patch.detach().clone().requires_grad_(True)
        warm_optim=torch.optim.Adam([warm_patch],lr=cfg.get('lr',.05))
        for warm_step in range(20):
            wx,wy,wids=batch_at(train,661990,warm_step,cfg.get('batch_size',128),device)
            warm_optim.zero_grad(set_to_none=True)
            for k in range(K):
                wt=sample_patch_transforms(wids,warm_step*K+k,661991,size,stream='warmup_patch')
                wz=apply_patch(wx,warm_patch,wt)[0]
                wp=sample_defense_params(wids,warm_step*K+k,661992,cfg.get('defense') or {})
                (ensemble_loss(bundles,apply_defense(wz,cfg.get('defense'),wp),wy,cfg.get('target'))/K).backward()
            warm_optim.step()
            with torch.no_grad():warm_patch.clamp_(0,1)
        sync(device);warmup_seconds=time.perf_counter()-warm_start
        del warm_patch,warm_optim,wx,wy,wz
        torch.cuda.reset_peak_memory_stats(device)
    def validate(step):
        nonlocal best,best_patch,best_step,val_time
        sync(device);t=time.perf_counter();scores=[]
        for b in bundles:
            _,m=evaluate_patch(b,patch.detach(),selection,seed=6611,draws=cfg.get('validation_draws',8),defense=cfg.get('defense'),target=cfg.get('target'),batch_size=cfg.get('batch_size',128))
            scores.append(m['tasr'] if cfg.get('target') is not None else m['uasr'])
        if any(v is None for v in scores):raise ValueError('Undefined validation ASR; need eligible images')
        score=float(np.mean(scores));sync(device);val_time+=time.perf_counter()-t
        history.append(dict(step=step,asr=score,training_time=spent))
        if score>best:best=score;best_patch=patch.detach().cpu().clone();best_step=step
    def checkpoint(step):
        save_torch(state_path,dict(cfg=cfg,source_hashes=identities,patch=patch.detach(),optimizer=optim.state_dict(),step=step,history=history,best=best,best_patch=best_patch,best_step=best_step,spent=spent,calls=calls,validation_time=val_time))
    end_to_end=time.perf_counter();step=start
    if not history:validate(0)
    time_budget=cfg.get('time_budget_s')
    # Time groups must supply a safe upper bound beyond their calibrated stopping time.
    while step<cfg.get('max_steps',1000):
        sync(device);t=time.perf_counter();x,y,ids=batch_at(train,seed,step,cfg.get('batch_size',128),device)
        optim.zero_grad(set_to_none=True);loss_value=0.
        for k in range(K):
            draw=step*K+k
            ts=sample_patch_transforms(ids,draw,seed,size,stream='train_patch')
            xp,_,_=apply_patch(x,patch,ts);dp=sample_defense_params(ids,draw,stable_seed('train_defense',seed),cfg.get('defense') or {})
            z=apply_defense(xp,cfg.get('defense'),dp);loss=ensemble_loss(bundles,z,y,cfg.get('target'))/K
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite attack loss')
            loss.backward();loss_value+=float(loss.detach());calls+=len(bundles)
        grad=float(patch.grad.norm())
        if not np.isfinite(grad):raise FloatingPointError('Nonfinite patch gradient')
        optim.step()
        with torch.no_grad():patch.clamp_(0,1)
        sync(device);spent+=time.perf_counter()-t;step+=1
        stop=time_budget is not None and step%10==0 and spent>=time_budget
        if step%cfg.get('validate_every',100)==0 or stop or step==cfg.get('max_steps',1000):
            validate(step);history[-1].update(loss=loss_value,gradient_norm=grad);checkpoint(step)
        if stop:break
    if time_budget is not None and spent<time_budget:raise RuntimeError('Safety step cap reached before time budget')
    artifact=dict(patch_tensor=best_patch,last_patch=patch.detach().cpu(),cfg=cfg,source_model_names=[b.name for b in bundles],source_checkpoint_hashes=identities,
                  selected_step=best_step,total_steps=step,validation_history=history,wall_time_seconds=spent,validation_time_seconds=val_time,
                  end_to_end_seconds=time.perf_counter()-function_start,warmup_seconds=warmup_seconds,
                  processed_training_images=step*cfg.get('batch_size',128)*K,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
                  forward_backward_calls=calls,software_versions=environment(),status='complete')
    save_torch(out/'patch.pt',artifact);return artifact
