"""Artifact integrity and protocol checks, shared by freeze and progress reports."""
import json
from pathlib import Path
import torch
from .utils import digest, load_torch

SOURCES={'R':['resnet20'],'V':['vgg11_bn_cifar'],'RV':['resnet20','vgg11_bn_cifar']}


def validate_patch(path, expected, artifacts, protocol='configs/protocol.json'):
    path,artifacts=Path(path),Path(artifacts)
    side=Path(str(path)+'.sha256.json')
    if not side.exists() or json.loads(side.read_text())['sha256']!=digest(path):
        raise ValueError('Patch checksum mismatch')
    obj=load_torch(path)
    cfg=dict(expected,split_sha256=digest(artifacts/'splits/manifest.json'),protocol_hash=digest(protocol))
    if obj['cfg']!=cfg:raise ValueError('Patch configuration mismatch')
    names=SOURCES[expected['strategy']]
    hashes=[digest(artifacts/f'models/{name}/best.pt') for name in names]
    if obj['source_model_names']!=names or obj['source_checkpoint_hashes']!=hashes:
        raise ValueError('Changed source checkpoints or invalid source models')
    if 'densenet_bc40_k12' in obj['source_model_names']:raise ValueError('Held-out target used during optimization')
    patch=obj['patch_tensor']
    if list(patch.shape)!=[1,3,expected['size'],expected['size']]:raise ValueError('Invalid patch shape')
    if not torch.isfinite(patch).all() or patch.min()<0 or patch.max()>1:raise ValueError('Invalid patch pixels')
    history=obj['validation_history']
    steps=[item['step'] for item in history]
    if not steps or steps[0]!=0 or obj['selected_step'] not in steps:raise ValueError('Missing selection provenance')
    if expected.get('time_budget_s') is None:
        if obj['total_steps']!=expected['max_steps']:raise ValueError('Incomplete fixed-step optimization')
    elif obj['wall_time_seconds']<expected['time_budget_s']:
        raise ValueError('Incomplete fixed-time optimization')
    if steps[-1]!=obj['total_steps']:raise ValueError('Final candidate was not validated')
    if obj.get('status')!='complete':raise ValueError('Incomplete patch artifact')
    return obj


def audit_matrix(rows,artifacts,protocol='configs/protocol.json'):
    report=[]
    for row in rows:
        path=Path(artifacts)/'patches'/row['run_id']/'patch.pt'
        if not path.exists():
            report.append(dict(run_id=row['run_id'],phase=row['phase'],status='missing'))
            continue
        try:
            validate_patch(path,row,artifacts,protocol)
            report.append(dict(run_id=row['run_id'],phase=row['phase'],status='complete'))
        except (ValueError,KeyError,FileNotFoundError) as exc:
            report.append(dict(run_id=row['run_id'],phase=row['phase'],status='invalid',reason=str(exc)))
    return report
