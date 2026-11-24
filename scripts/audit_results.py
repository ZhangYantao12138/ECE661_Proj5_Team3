"""Require the complete frozen test matrix and recompute every saved metric."""
from argparse import Namespace
from pathlib import Path
import json
from collections import Counter
from src.data import load_arrays
from src.evaluation_audit import audit_evaluations
from src.utils import save_json
from scripts.workflow import matrix,freeze_record


def main():
    args=Namespace(artifacts='artifacts',include_strength=True)
    frozen=Path('artifacts/freeze.json')
    if not frozen.exists() or json.loads(frozen.read_text())!=freeze_record(args):
        raise RuntimeError('Frozen artifacts are missing or changed')
    _,_,_,labels=load_arrays('data')
    report=audit_evaluations(matrix(args),'artifacts','results',labels)
    counts=dict(Counter(item['status'] for item in report))
    complete=all(item['status']=='complete' for item in report)
    save_json('results/evaluation_audit.json',dict(complete=complete,counts=counts,expected_evaluations=len(report),records=report))
    print(counts,flush=True)
    if not complete:raise RuntimeError('Required final evaluations are missing or invalid')


if __name__=='__main__':main()
