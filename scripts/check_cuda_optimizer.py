"""Real-data CUDA integration check; its tiny run is never a research result."""
import json
from pathlib import Path
import torch
from src.data import load_arrays,RawDataset
from src.model_bundle import load_model_bundle
from src.patches.optimize import optimize_patch
from src.utils import save_json


def main():
    torch.set_num_threads(4)
    split=json.loads(Path('artifacts/splits/manifest.json').read_text())
    x,y,_,_=load_arrays('data')
    train=RawDataset(x,y,split['train'][:128])
    selection=RawDataset(x,y,split['selection'][:32])
    bundle=load_model_bundle('artifacts/models/resnet20/best.pt','cuda')
    config=dict(size=7,target=None,seed=999,eot=4,defense=dict(type='resize_pad',sizes=[28,30,32]),
                max_steps=3,batch_size=32,validate_every=3,validation_draws=1,phase='engineering_cuda_check')
    result=optimize_patch([bundle],train,selection,config,'artifacts/diagnostics/cuda_optimizer')
    report=dict(status='passed',engineering_check_only=True,formal_research_evidence=False,
                source_hash=bundle.checkpoint_hash,steps=result['total_steps'],warmup_seconds=result['warmup_seconds'],
                validation_history=result['validation_history'])
    save_json('results/cuda_optimizer_check.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
