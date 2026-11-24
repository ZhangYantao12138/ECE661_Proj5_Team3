"""Enumerate every required evaluation; missing files cannot look like zero ASR."""
import json
from pathlib import Path
import numpy as np
from .utils import digest
from .metrics import validate_raw,aggregate_metrics

MODELS=['resnet20','vgg11_bn_cifar','densenet_bc40_k12']


def specifications(rows,defenses):
    for row in rows:
        models=MODELS if row['phase'] in ('basic','ensemble_steps','ensemble_time') else MODELS[:1]
        for model in models:
            configs=[None]
            if model=='resnet20' and row['size'] in (5,7) and row['phase'] in ('basic','defense_eot_control'):
                configs+=defenses
            if row['defense']:configs=[row['defense']]
            for defense in configs:
                yield row,model,defense,8
        if row['phase']=='defense_adaptive' and row['size']==7 and row['target'] in (None,3):
            yield row,'resnet20',row['defense'],32
    for model in MODELS:
        yield dict(run_id='clean',target=None),model,None,8
        for size in (3,5,7,16):
            yield dict(run_id=f'mean_s{size}',target=None),model,None,8
            for seed in (101,102,103):
                yield dict(run_id=f'random_s{size}_seed{seed}',target=None),model,None,8


def evaluation_key(row,model,defense,draws):
    return f"{row['run_id']}__{model}__{(defense or {}).get('type','identity')}__test__draws{draws}"


def audit_evaluations(rows,artifacts,results,test_labels):
    artifacts,results=Path(artifacts),Path(results)
    frozen=json.loads((artifacts/'freeze.json').read_text())
    defenses=json.loads((artifacts/'defenses.json').read_text())['selected']
    expected_ids=np.asarray(sorted(f'test:{i}' for i in range(len(test_labels))))
    report=[]
    for row,model,defense,draws in specifications(rows,defenses):
        key=evaluation_key(row,model,defense,draws)
        path=results/'raw'/f'{key}.npz';side=path.with_suffix('.json')
        if not path.exists() or not side.exists():
            report.append(dict(key=key,status='missing'));continue
        try:
            meta=json.loads(side.read_text())
            if digest(path)!=meta['raw_sha256']:raise ValueError('Raw checksum mismatch')
            if meta['phase_eval']!='test' or meta['draws']!=draws or meta['eval_seed']!=6610:raise ValueError('Evaluation protocol mismatch')
            if meta['checkpoint_hash']!=frozen['models'][model] or meta['split_sha256']!=frozen['split'] or meta['protocol_hash']!=frozen['protocol']:raise ValueError('Frozen identity mismatch')
            if meta['defense_eval']!=(defense or {'type':'identity'}):raise ValueError('Defense mismatch')
            if row['run_id'] in frozen['patches'] and meta['patch_hash']!=frozen['patches'][row['run_id']]:raise ValueError('Patch mismatch')
            raw=dict(np.load(path,allow_pickle=False));validate_raw(raw,draws)
            if not np.array_equal(np.unique(raw['image_id']),expected_ids):raise ValueError('Incorrect official test image set')
            ids=np.asarray([int(value.split(':')[1]) for value in raw['image_id']])
            if not np.array_equal(raw['true_label'],test_labels[ids]):raise ValueError('Incorrect official test labels')
            metrics=aggregate_metrics(raw,row.get('target'))
            if any(meta['metrics'][k]!=v for k,v in metrics.items()):raise ValueError('Summary differs from raw records')
            report.append(dict(key=key,status='complete',n_images=len(expected_ids),draws=draws))
        except (ValueError,KeyError) as exc:
            report.append(dict(key=key,status='invalid',reason=str(exc)))
    return report
