import copy
import numpy as np
import pytest
import torch
from torch import nn
from src.models import build_model
from src.model_bundle import ModelBundle
from src.patches.render import apply_patch,sample_patch_transforms
from src.defenses import apply_defense,sample_defense_params
from src.metrics import aggregate_metrics,validate_raw
from src.data import make_splits,RawDataset
from src.matrix import task_matrix
from src.ensemble import ensemble_loss
from src.patches.optimize import optimize_patch

torch.set_num_threads(2)
def test_architectures():
    for name in ['resnet20','vgg11_bn_cifar','densenet_bc40_k12']:
        m=build_model(name).eval();x=torch.rand(2,3,32,32,requires_grad=True);z=m(x)
        assert z.shape==(2,10);z.sum().backward();assert torch.isfinite(x.grad).all()
        if name=='resnet20':assert sum(p.numel() for p in m.parameters())==269722

def test_split():
    labels=np.repeat(np.arange(10),5000);s=make_splits(labels)
    assert len(s['train'])==45000 and len(s['val'])==5000 and len(s['selection'])==1000
    assert s==make_splits(labels)
    assert not set(s['selection'])-set(s['val'])

def test_hand_metrics():
    raw=dict(image_id=['a','b','c','d'],true_label=[0,1,2,3],pred_clean=[0,1,0,3],pred_patch=[1,1,2,0])
    m=aggregate_metrics(raw,1)
    assert m['clean_acc']==.75 and m['patched_acc']==.5 and m['uasr']==2/3 and m['tasr']==.5 and m['n_eligible']==2
    raw['pred_clean']=[9]*4;assert aggregate_metrics(raw,1)['tasr'] is None

def test_render_identity_even_and_odd():
    for s in [3,5,7,16]:
        x=torch.rand(2,3,32,32,dtype=torch.double);p=torch.rand(1,3,s,s,dtype=torch.double,requires_grad=True)
        t=[dict(angle=0.,scale=1.,cx=3+(s-1)/2,cy=4+(s-1)/2)]*2
        z,a,meta=apply_patch(x,p,t)
        assert torch.allclose(z[:,:,4:4+s,3:3+s],p.expand(2,-1,-1,-1),atol=1e-12)
        assert torch.all(meta['support_area']==s*s)
        assert torch.equal(z[a.expand_as(x)==0],x[a.expand_as(x)==0])

def test_support_and_replay():
    ids=['test:0','test:1','test:23'];ts=sample_patch_transforms(ids,2,6610,16)
    assert ts[1:]==sample_patch_transforms(ids[1:],2,6610,16)
    x=torch.zeros(3,3,32,32);p=torch.ones(1,3,16,16);_,a,m=apply_patch(x,p,ts)
    assert (a>=0).all() and (a<=1).all()
    # No nonzero support on the outermost pixel centres: footprint contained.
    assert a[:,:,0,:].max()==0 and a[:,:,-1,:].max()==0
    assert a[:,:,:,0].max()==0 and a[:,:,:,-1].max()==0

@pytest.mark.parametrize('cfg',[{'type':'gaussian','sigma':.5},{'type':'resize_pad','sizes':[28,30,32]}])
def test_gradient_and_defense(cfg):
    x=torch.rand(1,3,32,32,dtype=torch.double);p=torch.rand(1,3,3,3,dtype=torch.double,requires_grad=True)
    t=[dict(angle=.12,scale=.97,cx=12.,cy=15.)];params=sample_defense_params(['val:4'],1,6611,cfg)
    weights=torch.randn_like(x)
    def loss(p):return (apply_defense(apply_patch(x,p,t)[0],cfg,params)*weights).sum()
    grad=torch.autograd.grad(loss(p),p)[0];direction=torch.randn_like(p);eps=1e-6
    numeric=(loss(p+eps*direction)-loss(p-eps*direction))/(2*eps)
    assert torch.allclose(numeric,(grad*direction).sum(),atol=1e-7,rtol=1e-6)
    constant=torch.ones_like(x)*.43;assert torch.allclose(apply_defense(constant,cfg,params),constant,atol=1e-12)

def toy(name,seed):
    torch.manual_seed(seed);return ModelBundle(nn.Sequential(nn.Flatten(),nn.Linear(3072,10)),[0,0,0],[1,1,1],name).freeze()
def test_ensemble_gradient_and_frozen():
    r=toy('R',2);v=toy('V',3);x=torch.rand(3,3,32,32,requires_grad=True);y=torch.tensor([1,2,3]);before=copy.deepcopy(r.state_dict())
    gr=torch.autograd.grad(ensemble_loss([r],x,y,3),x)[0];gv=torch.autograd.grad(ensemble_loss([v],x,y,3),x)[0]
    g=torch.autograd.grad(ensemble_loss([r,v],x,y,3),x)[0]
    assert torch.allclose(g,(gr+gv)/2,atol=1e-7)
    assert all(p.grad is None for p in r.parameters())
    assert all(torch.equal(before[k],r.state_dict()[k]) for k in before)
    with pytest.raises(ValueError):ensemble_loss([toy('densenet_bc40_k12',3)],x,y)

def test_matrix():
    from collections import Counter
    rows=task_matrix();assert len(rows)==561
    assert Counter(r['phase'] for r in rows)==dict(basic=132,ensemble_steps=132,ensemble_time=99,defense_eot_control=66,defense_adaptive=132)
    assert len(task_matrix(include_strength=True))==585

def test_optimizer_resume(tmp_path):
    from src.utils import load_torch,save_torch
    b=toy('R',17);rng=np.random.default_rng(42);x=rng.integers(0,256,(20,3,32,32),dtype=np.uint8)
    with torch.no_grad():y=b(torch.tensor(x).float()/255).argmax(1).numpy()
    ds=RawDataset(x,y);cfg=dict(size=3,seed=101,target=None,batch_size=4,max_steps=4,validate_every=2,validation_draws=1,eot=1)
    full=optimize_patch([b],ds,ds,cfg,tmp_path/'full')
    # Capture interruption at step 2 without changing the configuration.
    import src.patches.optimize as module
    original=module.save_torch
    def interrupt(path,obj):
        original(path,obj)
        if str(path).endswith('resume.pt') and obj['step']==2:raise InterruptedError()
    module.save_torch=interrupt
    try:
        with pytest.raises(InterruptedError):optimize_patch([b],ds,ds,cfg,tmp_path/'interrupted')
    finally:module.save_torch=original
    resumed=optimize_patch([b],ds,ds,cfg,tmp_path/'interrupted',resume=True)
    assert torch.equal(full['last_patch'],resumed['last_patch'])
    assert torch.equal(full['patch_tensor'],resumed['patch_tensor'])

def test_evaluation_batch_invariance():
    from src.evaluate import evaluate_patch
    b=toy('R',4);rng=np.random.default_rng(4);x=rng.integers(0,256,(9,3,32,32),dtype=np.uint8);y=np.arange(9);ds=RawDataset(x,y,split='val');p=torch.rand(1,3,5,5)
    cfg=dict(type='resize_pad',sizes=[28,30,32]);ra,ma=evaluate_patch(b,p,ds,draws=2,batch_size=4,defense=cfg);rb,mb=evaluate_patch(b,p,ds,draws=2,batch_size=7,defense=cfg)
    ka=np.lexsort((ra['draw_id'],ra['image_id']));kb=np.lexsort((rb['draw_id'],rb['image_id']))
    for key in ra:assert np.array_equal(ra[key][ka],rb[key][kb]),key
    assert ma==mb

def test_target_macro_bootstrap():
    from src.statistics import paired_target_macro_bootstrap,seed_summary
    y=np.repeat(np.arange(10),2);a=np.ones((10,20,2,3));b=np.zeros_like(a)
    r=paired_target_macro_bootstrap(a,b,y,np.ones(20,dtype=bool),replicates=25)
    assert r['delta_pp']==100 and r['ci95_pp']==[100,100]
    assert seed_summary([1,2,3])['sample_std']==1
