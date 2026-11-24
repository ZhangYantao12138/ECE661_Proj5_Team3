import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data import prepare,load_arrays,make_splits,normalization,RawDataset
from src.models import build_model
from src.model_bundle import ModelBundle
from src.utils import save_json,save_torch,digest,environment,load_torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['resnet20','vgg11_bn_cifar','densenet_bc40_k12'],required=True);p.add_argument('--data',default='data');p.add_argument('--output',default='artifacts');p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=200);p.add_argument('--resume',action='store_true');p.add_argument('--threads',type=int,default=4);p.add_argument('--lr',type=float,choices=[.1,.05],default=.1);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    out=Path(a.output);split_path=out/'splits/manifest.json'
    if not split_path.exists():
        x,y,xt,yt=prepare(a.data);sp=make_splits(y);sp['mean'],sp['std']=normalization(x,sp['train']);save_json(split_path,sp)
    else:x,y,xt,yt=load_arrays(a.data);sp=json.loads(split_path.read_text())
    train=RawDataset(x,y,sp['train'],augment=True);val=RawDataset(x,y,sp['val']);gen=torch.Generator().manual_seed(17)
    loader=DataLoader(train,batch_size=128,shuffle=True,generator=gen);vl=DataLoader(val,batch_size=128)
    bundle=ModelBundle(build_model(a.model),sp['mean'],sp['std'],a.model).to(a.device)
    opt=torch.optim.SGD(bundle.parameters(),lr=a.lr,momentum=.9,weight_decay=5e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    start=0;best=-1.;history=[];last=out/f'models/{a.model}/resume.pt'
    if a.resume and last.exists():
        r=load_torch(last,a.device)
        if r['epochs']!=a.epochs or r['split_hash']!=digest(split_path) or r.get('initial_lr',.1)!=a.lr:raise ValueError('Training resume configuration mismatch')
        bundle.load_state_dict(r['bundle']);opt.load_state_dict(r['optimizer']);sched.load_state_dict(r['scheduler']);start=r['epoch'];best=r['best'];history=r['history'];torch.set_rng_state(r['torch_rng'].cpu());gen.set_state(r['loader_rng'].cpu())
        if a.device.startswith('cuda'):torch.cuda.set_rng_state_all([state.cpu() for state in r['cuda_rng']])
    for epoch in range(start,a.epochs):
        t=time.perf_counter();bundle.train();loss_sum=0.;n=0
        for xb,yb,_ in loader:
            xb=xb.to(a.device);yb=yb.to(a.device);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(bundle(xb),yb)
            loss.backward();opt.step();loss_sum+=float(loss.detach())*len(xb);n+=len(xb)
        bundle.eval();correct=np.zeros(10,dtype=int);counts=np.zeros(10,dtype=int);vloss=0.
        with torch.no_grad():
            for xb,yb,_ in vl:
                logits=bundle(xb.to(a.device));pred=logits.argmax(1).cpu();vloss+=float(torch.nn.functional.cross_entropy(logits,yb.to(a.device),reduction='sum'))
                for c in range(10):counts[c]+=int((yb==c).sum());correct[c]+=int(((yb==c)&(pred==yb)).sum())
        acc=float(correct.sum()/counts.sum());history.append(dict(epoch=epoch+1,train_loss=loss_sum/n,val_loss=vloss/counts.sum(),val_acc=acc,correct=correct.tolist(),counts=counts.tolist(),seconds=time.perf_counter()-t));sched.step()
        if acc>best:
            best=acc;save_torch(out/f'models/{a.model}/best.pt',dict(state_dict=bundle.network.state_dict(),architecture_name=a.model,architecture_config=str(bundle.network),normalization_mean=sp['mean'],normalization_std=sp['std'],num_classes=10,training_seed=17,split_sha256=digest(split_path),best_epoch=epoch+1,validation_metrics=history[-1],training_config=vars(a),software_versions=environment(),formal_training=a.epochs==200))
        save_torch(last,dict(bundle=bundle.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch+1,best=best,history=history,torch_rng=torch.get_rng_state(),loader_rng=gen.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],epochs=a.epochs,initial_lr=a.lr,split_hash=digest(split_path)))
        save_json(out/f'models/{a.model}/history.json',history);print(history[-1],flush=True)
    # A checkpoint selected in an early epoch is valid only after the full run ends.
    save_json(out/f'models/{a.model}/training_complete.json',dict(model=a.model,epochs_completed=len(history),planned_epochs=a.epochs,best_validation_acc=best,split_hash=digest(split_path),best_checkpoint_sha256=digest(out/f'models/{a.model}/best.pt'),precision='float32',status='complete'))
if __name__=='__main__':main()
