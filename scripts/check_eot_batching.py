"""Isolated real-data check of sequential versus semantically identical batched EOT."""
import json,time
from pathlib import Path
import torch
from src.data import load_arrays,RawDataset
from src.model_bundle import load_model_bundle
from src.patches.render import sample_patch_transforms,apply_patch
from src.defenses import apply_defense,sample_defense_params
from src.ensemble import ensemble_loss
from src.utils import sync,save_json


def main():
    torch.set_num_threads(4)
    split=json.loads(Path('artifacts/splits/manifest.json').read_text())
    x,y,_,_=load_arrays('data');dataset=RawDataset(x,y,split['train'])
    x,y,ids=dataset.cached_batch(list(range(128)),'cuda')
    bundle=load_model_bundle('artifacts/models/resnet20/best.pt','cuda');bundle.freeze()
    defenses=[None]+json.loads(Path('artifacts/defenses.json').read_text())['selected']
    base=torch.rand((1,3,7,7),generator=torch.Generator().manual_seed(1729)).cuda()
    checks=[]
    for defense in defenses:
        for target,K in [(None,4),(3,4),(3,8)]:
            transforms=[sample_patch_transforms(ids,k,101,7,stream='train_patch') for k in range(K)]
            params=[sample_defense_params(ids,k,102,defense or {}) for k in range(K)]
            def gradient(batched):
                p=base.clone().requires_grad_(True)
                if batched:
                    z=apply_patch(x.repeat(K,1,1,1),p,[t for group in transforms for t in group])[0]
                    z=apply_defense(z,defense,[t for group in params for t in group])
                    loss=ensemble_loss([bundle],z,y.repeat(K),target);loss.backward()
                    value=float(loss.detach())
                else:
                    value=0.
                    for k in range(K):
                        z=apply_defense(apply_patch(x,p,transforms[k])[0],defense,params[k])
                        loss=ensemble_loss([bundle],z,y,target)/K;loss.backward();value+=float(loss.detach())
                return value,p.grad.detach().clone()
            gradient(False);gradient(True);sync('cuda')
            results={}
            for batched in [False,True]:
                times=[]
                for repeat in range(5):
                    sync('cuda');start=time.perf_counter();value,g=gradient(batched);sync('cuda');times.append(time.perf_counter()-start)
                results[batched]=(value,g,sorted(times)[2])
            a,b=results[False],results[True]
            absolute=float((a[1]-b[1]).abs().max());relative=float((a[1]-b[1]).norm()/a[1].norm().clamp_min(1e-20));lossdiff=abs(a[0]-b[0])
            checks.append(dict(defense=(defense or {}).get('type','identity'),target=target,eot=K,loss_difference=lossdiff,gradient_max_abs=absolute,gradient_relative_l2=relative,
                sequential_s=a[2],batched_s=b[2],speedup=a[2]/b[2],equivalent=lossdiff<2e-5 and absolute<2e-5 and relative<5e-4))
            print(checks[-1],flush=True)
    save_json('results/eot_batching_check.json',dict(checks=checks,all_equivalent=all(r['equivalent'] for r in checks),
        background_batch=128,rule='Same 128 backgrounds, same K independent transforms/defense draws, mean CE over K*128; one optimizer update',peak_memory_bytes=torch.cuda.max_memory_allocated()))
    if not all(r['equivalent'] for r in checks):raise RuntimeError('Batched EOT numerical equivalence failed')


if __name__=='__main__':main()
