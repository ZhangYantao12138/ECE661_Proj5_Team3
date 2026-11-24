"""Full validation archive after patch selection, never using the held-out target."""
import hashlib
import json
from pathlib import Path
from argparse import Namespace
import subprocess
import sys
from collections import Counter
import numpy as np
from src.data import load_arrays
from src.evaluation_audit import specifications,evaluation_key
from src.utils import digest,save_json
from src.metrics import validate_raw,aggregate_metrics
from scripts.workflow import matrix,freeze_record


def main():
    if not Path('artifacts/evaluation_batch.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.check_evaluation_batching'],check=True)
    args=Namespace(artifacts='artifacts',include_strength=True)
    fingerprint=freeze_record(args)
    identity=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    audit_path=Path('results/validation_audit.json')
    if audit_path.exists():
        old=json.loads(audit_path.read_text(encoding='utf-8'))
        if old.get('complete') and old.get('frozen_candidate_identity')==identity:
            if all(Path(row['path']).exists() and digest(row['path'])==row['raw_sha256'] for row in old['records']):
                print('Verified existing complete full-validation archive',flush=True);return
    subprocess.run([sys.executable,'-u','-m','scripts.workflow','evaluate','--include-strength','--device','cuda'],check=True)
    split=json.loads(Path('artifacts/splits/manifest.json').read_text())
    _,labels,_,_=load_arrays('data')
    expected_ids=np.asarray(sorted(f'train:{i}' for i in split['val']))
    defenses=json.loads(Path('artifacts/defenses.json').read_text())['selected']
    records=[]
    for row,model,defense,draws in specifications(matrix(args),defenses):
        if 'phase' not in row or model=='densenet_bc40_k12' or draws!=8:continue
        key=evaluation_key(row,model,defense,draws).replace('__test__','__validation__')
        path=Path('results/validation_raw')/(key+'.npz');side=path.with_suffix('.json')
        meta=json.loads(side.read_text())
        if meta['phase_eval']!='validation' or meta['eval_seed']!=6611 or meta['draws']!=8:raise ValueError('Invalid validation protocol')
        if meta['patch_hash']!=fingerprint['patches'][row['run_id']] or meta['checkpoint_hash']!=fingerprint['models'][model]:raise ValueError('Changed validation candidate/model')
        if digest(path)!=meta['raw_sha256']:raise ValueError('Changed validation raw file')
        raw=dict(np.load(path,allow_pickle=False));validate_raw(raw,8)
        if not np.array_equal(np.unique(raw['image_id']),expected_ids):raise ValueError('Incomplete full validation image set')
        indices=np.asarray([int(value.split(':')[1]) for value in raw['image_id']])
        if not np.array_equal(raw['true_label'],labels[indices]):raise ValueError('Invalid official validation labels')
        metrics=aggregate_metrics(raw,row['target'])
        if any(meta['metrics'][key]!=value for key,value in metrics.items()):raise ValueError('Validation metrics differ from raw records')
        records.append(dict(path=str(path),raw_sha256=meta['raw_sha256'],status='complete',n_images=5000,draws=8))
    if len(records)!=1212:raise ValueError('Expected 1,212 full validation conditions')
    save_json(audit_path,dict(complete=True,frozen_candidate_identity=identity,expected_evaluations=1212,
        source_only=True,heldout_attack_feedback_used=False,selection_changed=False,records=records))
    print('Full 5,000-image validation audit passed: 1,212 conditions; no DenseNet attack evaluation',flush=True)


if __name__=='__main__':main()
