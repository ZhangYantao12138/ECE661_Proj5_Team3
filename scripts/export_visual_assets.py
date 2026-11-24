"""Export true image/patch arrays; plotting stays in a separate, torch-free process."""
import json
from pathlib import Path
import numpy as np
import torch
from src.data import load_arrays
from src.models import build_model
from src.patches.render import sample_patch_transforms,apply_patch
from src.defenses import apply_defense
from src.utils import load_torch,save_json,digest
from scripts.analyze_results import read_record,write_csv


def main():
    audit=json.loads(Path('results/evaluation_audit.json').read_text())
    if not audit['complete']:raise RuntimeError('Audit real final results before creating publication visuals')
    torch.set_num_threads(4)
    payload={};characteristics=[]
    for size in (3,5,7,16):
        for target in [None]+list(range(10)):
            for seed in (101,102,103):
                run=f'basic__R__s{size}__t{target}__seed{seed}__k1'
                obj=load_torch(Path('artifacts/patches')/run/'patch.pt')
                patch=obj['patch_tensor'].numpy()[0].transpose(1,2,0)
                if seed==101:payload[f'patch_s{size}_t{target}']=patch
                tv=float(np.abs(np.diff(patch,axis=0)).mean()+np.abs(np.diff(patch,axis=1)).mean())
                spectrum=np.abs(np.fft.fft2(patch-patch.mean((0,1),keepdims=True),axes=(0,1)))**2
                fy,fx=np.meshgrid(np.fft.fftfreq(size),np.fft.fftfreq(size),indexing='ij')
                mask=np.sqrt(fx*fx+fy*fy)>=.25
                energy=float(spectrum.sum())
                characteristics.append(dict(run_id=run,size=size,target=target,seed=seed,total_variation=tv,
                    high_frequency_energy_fraction=float(spectrum[mask].sum()/energy) if energy else None,
                    high_frequency_definition='radial frequency >= 0.25 cycles/pixel after channel-mean removal',
                    channel_means=patch.mean((0,1)).tolist(),patch_sha256=digest(Path('artifacts/patches')/run/'patch.pt')))
    x,y,xt,yt=load_arrays('data')
    spec=json.loads(Path('configs/qualitative_examples.json').read_text())
    ids=np.asarray(spec['image_ids'])
    patch=load_torch(Path('artifacts/patches')/spec['patch_run_id']/'patch.pt')['patch_tensor'].cuda()
    images=torch.tensor(xt[ids],device='cuda').float()/255
    transforms=sample_patch_transforms([f'test:{int(i)}' for i in ids],spec['draw_id'],spec['eval_seed'],7)
    patched,alpha,_=apply_patch(images,patch,transforms)
    payload['case_clean']=images.cpu().numpy().transpose(0,2,3,1)
    payload['case_patched']=patched.cpu().numpy().transpose(0,2,3,1)
    payload['case_alpha']=alpha.cpu().numpy()[:,0]
    path=Path('results/raw')/f"{spec['patch_run_id']}__resnet20__identity__test__draws8.npz"
    _,raw=read_record(path)
    cases=[]
    for iid in ids:
        pred=int(raw['pred_patch'][iid,spec['draw_id']]);clean=int(raw['pred_clean'][iid,0]);label=int(yt[iid])
        cases.append(dict(image_id=f'test:{iid}',true_label=label,pred_clean=clean,pred_patch=pred,
            eligible=clean==label and label!=3,targeted_success=clean==label and label!=3 and pred==3,
            transform=transforms[len(cases)]))
    sample=torch.from_numpy(x[:1].copy()).float().cuda()/255
    defenses=json.loads(Path('artifacts/defenses.json').read_text())['selected']
    gaussian=next(d for d in defenses if d['type']=='gaussian')
    resize=next(d for d in defenses if d['type']=='resize_pad')
    q=min(resize['sizes']);pad=(32-q)//2
    payload['defense_original']=sample.cpu().numpy()[0].transpose(1,2,0)
    payload['defense_gaussian']=apply_defense(sample,gaussian).cpu().numpy()[0].transpose(1,2,0)
    payload['defense_resize']=apply_defense(sample,resize,[dict(q=q,top=pad,left=pad)]).cpu().numpy()[0].transpose(1,2,0)
    models=[]
    for name in ('resnet20','vgg11_bn_cifar','densenet_bc40_k12'):
        obj=load_torch(Path('artifacts/models')/name/'best.pt')
        models.append(dict(name=name,parameters=sum(p.numel() for p in build_model(name).parameters()),best_epoch=obj['best_epoch'],
            best_validation_accuracy=obj['validation_metrics']['val_acc'],checkpoint_sha256=digest(Path('artifacts/models')/name/'best.pt')))
    np.savez_compressed('results/visual_assets.npz',**payload)
    write_csv('results/tables/patch_characteristics.csv',characteristics)
    representative=load_torch(Path('artifacts/patches')/spec['patch_run_id']/'patch.pt')
    ensemble_histories=[]
    for phase in ['ensemble_steps','ensemble_time']:
        for strategy in ['R','V','RV']:
            for target in [None,3]:
                source='basic' if phase=='ensemble_steps' and strategy=='R' else phase
                run=f'{source}__{strategy}__s7__t{target}__seed101__k1'
                obj=load_torch(Path('artifacts/patches')/run/'patch.pt')
                ensemble_histories.append(dict(phase=phase,strategy=strategy,target=target,seed=101,history=obj['validation_history']))
    save_json('results/visual_assets.json',dict(models=models,cases=cases,qualitative_spec=spec,defenses=defenses,representative_history=representative['validation_history'],
        illustration=dict(split='official training',image_id=0,q=q,padding_top=pad,padding_left=pad),patch_characteristics=characteristics,ensemble_histories=ensemble_histories))


if __name__=='__main__':main()
