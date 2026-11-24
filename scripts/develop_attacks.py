"""Execute the finite training/validation-only attack development candidates."""
import json
from pathlib import Path
import torch
from src.data import load_arrays,RawDataset
from src.model_bundle import load_model_bundle
from src.patches.optimize import optimize_patch
from src.utils import digest,load_torch,save_json


def main():
    torch.set_num_threads(4)
    art=Path('artifacts')
    completion=json.loads((art/'models/resnet20/training_complete.json').read_text())
    if completion['epochs_completed']!=200:raise RuntimeError('Complete the formal source training first')
    bundle=load_model_bundle(art/'models/resnet20/best.pt','cuda')
    split=json.loads((art/'splits/manifest.json').read_text())
    x,y,_,_=load_arrays('data')
    train=RawDataset(x,y,split['train']);selection=RawDataset(x,y,split['selection'])
    report=[]
    for target in (None,3):
        for lr in (.01,.05):
            for steps in (500,1000):
                run_id=f'development_s7_t{target}_lr{lr}_steps{steps}'
                cfg=dict(run_id=run_id,phase='development',size=7,target=target,seed=101,eot=1,lr=lr,
                         max_steps=steps,batch_size=128,validate_every=100,validation_draws=8,
                         split_sha256=digest(art/'splits/manifest.json'),protocol_hash=digest('configs/protocol.json'))
                path=art/'development'/run_id
                if (path/'patch.pt').exists():
                    obj=load_torch(path/'patch.pt')
                    if obj['cfg']!=cfg or obj['source_checkpoint_hashes']!=[bundle.checkpoint_hash]:raise ValueError('Changed development checkpoint/config')
                else:
                    print('RUN '+run_id,flush=True)
                    obj=optimize_patch([bundle],train,selection,cfg,path,resume=True)
                report.append(dict(run_id=run_id,target=target,lr=lr,steps=steps,selected_step=obj['selected_step'],
                                   validation_history=obj['validation_history'],training_time_s=obj['wall_time_seconds'],
                                   validation_time_s=obj['validation_time_seconds'],patch_sha256=digest(path/'patch.pt')))
                save_json(art/'development_progress.json',report)
    save_json(art/'development.json',dict(status='complete',checkpoint_hash=bundle.checkpoint_hash,
              candidates=report,main_configuration=dict(lr=.05,steps=1000),
              selection_rule='Retain the original prespecified main setting; report all finite diagnostics. No test/held-out feedback.',
              split_hash=digest(art/'splits/manifest.json')))


if __name__=='__main__':main()
