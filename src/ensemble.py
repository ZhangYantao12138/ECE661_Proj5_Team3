import torch.nn.functional as F

def ensemble_loss(bundles,images,labels,target=None,weights=None):
    if any(b.name=='densenet_bc40_k12' for b in bundles):raise ValueError('DenseNet held out: cannot optimize on target')
    weights=weights or [1/len(bundles)]*len(bundles)
    if len(weights)!=len(bundles) or abs(sum(weights)-1)>1e-8 or min(weights)<0:raise ValueError('Invalid loss weights')
    import torch
    labels=labels if target is None else torch.full_like(labels,target)
    return sum(w*F.cross_entropy(b(images),labels)*( -1 if target is None else 1) for b,w in zip(bundles,weights))
