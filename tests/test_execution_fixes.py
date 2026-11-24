import copy
import math
import pytest
import torch
import torch.nn.functional as F
from src.models import build_model
from src.model_bundle import ModelBundle
from src.patches.render import apply_patch, sample_patch_transforms
from src.defenses import apply_defense, sample_defense_params
from src.metrics import validate_raw


def test_grouped_defense_matches_reference_forward_and_gradient():
    x = torch.rand(9, 3, 32, 32, dtype=torch.double, requires_grad=True)
    cfg = dict(type='resize_pad', sizes=[28,30,32])
    params = sample_defense_params([f'train:{i}' for i in range(9)], 0, 661, cfg)
    reference = []
    for image, p in zip(x, params):
        q, top, left = p['q'], p['top'], p['left']
        resized = F.interpolate(image[None], (q,q), mode='bilinear', align_corners=False)
        reference.append(F.pad(resized, (left,32-q-left,top,32-q-top), mode='replicate'))
    reference = torch.cat(reference)
    actual = apply_defense(x, cfg, params)
    weights = torch.randn_like(actual)
    ga = torch.autograd.grad((actual*weights).sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad((reference*weights).sum(), x)[0]
    assert torch.equal(actual, reference)
    assert torch.allclose(ga, gb, atol=1e-12, rtol=1e-12)


def test_batched_renderer_matches_reference_forward_and_gradient():
    x = torch.rand(7,3,32,32,dtype=torch.double)
    patch = torch.rand(1,3,7,7,dtype=torch.double,requires_grad=True)
    transforms = sample_patch_transforms([f'train:{i}' for i in range(7)], 3, 101, 7)
    yy, xx = torch.meshgrid(torch.arange(32,dtype=x.dtype),torch.arange(32,dtype=x.dtype),indexing='ij')
    grids=[]
    for p in transforms:
        dx,dy=xx-p['cx'],yy-p['cy']
        u=(math.cos(p['angle'])*dx+math.sin(p['angle'])*dy)/p['scale']
        v=(-math.sin(p['angle'])*dx+math.cos(p['angle'])*dy)/p['scale']
        grids.append(torch.stack((2*u/7,2*v/7),-1))
    grid=torch.stack(grids)
    alpha=F.grid_sample(torch.ones(7,1,7,7,dtype=x.dtype),grid,align_corners=False)
    reference=x*(1-alpha)+F.grid_sample(patch.expand(7,-1,-1,-1),grid,align_corners=False)
    actual=apply_patch(x,patch,transforms)[0]
    weights=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*weights).sum(),patch,retain_graph=True)[0]
    gb=torch.autograd.grad((reference*weights).sum(),patch)[0]
    assert torch.equal(actual,reference)
    assert torch.equal(ga,gb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_full_cuda_patch_defense_classifier_gradient_and_freezing():
    for name in ('resnet20','vgg11_bn_cifar','densenet_bc40_k12'):
        bundle=ModelBundle(build_model(name),[.5]*3,[.25]*3,name).cuda().freeze()
        before=copy.deepcopy(bundle.state_dict())
        images=torch.rand(2,3,32,32,device='cuda')
        patch=torch.rand(1,3,7,7,device='cuda',requires_grad=True)
        ids=['train:1','train:2']
        transforms=sample_patch_transforms(ids,0,101,7)
        for defense in [None,dict(type='gaussian',sigma=.5),dict(type='resize_pad',sizes=[28,30,32])]:
            patched=apply_patch(images,patch,transforms)[0]
            params=sample_defense_params(ids,0,101,defense or {})
            logits=bundle(apply_defense(patched,defense,params))
            gradient=torch.autograd.grad(F.cross_entropy(logits,torch.tensor([3,3],device='cuda')),patch)[0]
            assert torch.isfinite(gradient).all() and gradient.norm()>0
        assert all(torch.equal(before[k],bundle.state_dict()[k]) for k in before)
        assert all(p.grad is None for p in bundle.parameters())


def test_evaluation_rejects_missing_draws_and_inconsistent_labels():
    raw=dict(image_id=['a','a','b','b'],draw_id=[0,1,0,1],true_label=[0,0,1,1],pred_clean=[0,0,1,1],pred_patch=[0,1,1,0])
    validate_raw(raw,2)
    bad=dict(raw,draw_id=[0,0,0,1])
    with pytest.raises(ValueError,match='Duplicate'):validate_raw(bad,2)
    bad=dict(raw,true_label=[0,1,1,1])
    with pytest.raises(ValueError,match='Inconsistent'):validate_raw(bad,2)
    bad={key:val[:-1] for key,val in raw.items()}
    with pytest.raises(ValueError,match='Missing'):validate_raw(bad,2)


def test_workflow_controls_cache_validates_identity_and_checksum(tmp_path,monkeypatch):
    import sys
    import numpy as np
    from torch import nn
    from src.data import RawDataset
    from src.utils import save_json
    from scripts import workflow
    monkeypatch.chdir(tmp_path)
    save_json('configs/protocol.json',{'test_protocol':True})
    save_json('artifacts/splits/manifest.json',{'test_split':True})
    images=np.random.default_rng(17).integers(0,256,(9,3,32,32),dtype=np.uint8)
    dataset=RawDataset(images,np.arange(9),split='val')
    bundle=ModelBundle(nn.Sequential(nn.Flatten(),nn.Linear(3072,10)),[0,0,0],[1,1,1],'resnet20','test-model-hash').freeze()
    monkeypatch.setattr(workflow,'FACTORY',{'resnet20':None})
    monkeypatch.setattr(workflow,'context',lambda args:({'mean':[.4,.5,.6]},dataset,dataset,dataset,dataset))
    monkeypatch.setattr(workflow,'load_model_bundle',lambda *args:bundle)
    monkeypatch.setattr(sys,'argv',['workflow','controls'])
    workflow.main()
    paths=sorted((tmp_path/'results/validation_raw').glob('*.npz'))
    assert len(paths)==17
    def no_recompute(*args,**kwargs):raise AssertionError('A verified evaluation should be reused')
    monkeypatch.setattr(workflow,'evaluate_patch',no_recompute)
    workflow.main()
    paths[0].write_bytes(b'corrupted raw record')
    with pytest.raises(ValueError,match='cache identity'):workflow.main()


def test_freeze_rejects_early_checkpoint_marked_formal(tmp_path,monkeypatch):
    import sys
    from scripts import workflow
    from src.utils import save_json
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(workflow,'FACTORY',{'resnet20':None})
    monkeypatch.setattr(workflow,'load_torch',lambda *args:{'formal_training':True})
    save_json('artifacts/models/resnet20/training_complete.json',{'epochs_completed':1})
    monkeypatch.setattr(sys,'argv',['workflow','freeze'])
    with pytest.raises(RuntimeError,match='Incomplete or changed formal classifier'):workflow.main()


def test_complete_evaluation_matrix_covers_controls_transfer_and_stability():
    from src.evaluation_audit import specifications,evaluation_key
    from src.matrix import task_matrix
    defenses=[dict(type='gaussian',sigma=.25),dict(type='resize_pad',sizes=[30,31,32])]
    specs=list(specifications(task_matrix(include_strength=True),defenses))
    keys=[evaluation_key(*entry) for entry in specs]
    assert len(keys)==len(set(keys))==1638
    assert sum(draws==32 for _,_,_,draws in specs)==12
    assert sum(row['run_id']=='clean' for row,_,_,_ in specs)==3
    assert sum(model=='densenet_bc40_k12' and row.get('phase')=='basic' for row,model,_,_ in specs)==132


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_cached_raw_batches_are_exactly_original_samples(device):
    import numpy as np
    from src.data import RawDataset,batch_at
    from src.utils import stable_seed
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    images=np.random.default_rng(17).integers(0,256,(40,3,32,32),dtype=np.uint8)
    labels=np.arange(40,dtype=np.int64)%10
    dataset=RawDataset(images,labels,ids=np.array([19,3,24,35,1,8]),split='train')
    rng=np.random.default_rng(stable_seed('train_order',101,23))
    positions=rng.choice(len(dataset),10,replace=True)
    reference=[dataset[int(i)] for i in positions]
    x,y,names=batch_at(dataset,101,23,10,device)
    assert torch.equal(x.cpu(),torch.stack([item[0] for item in reference]))
    assert y.cpu().tolist()==[item[1] for item in reference]
    assert names==[item[2] for item in reference]
    batches=list(dataset.sequential_batches(4,device))
    assert torch.equal(torch.cat([b[0].cpu() for b in batches]),torch.stack([dataset[i][0] for i in range(len(dataset))]))
