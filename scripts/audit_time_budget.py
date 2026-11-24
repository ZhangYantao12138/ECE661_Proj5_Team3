"""Audit completed matched-time records without performing GPU computations."""
from argparse import Namespace
from pathlib import Path
import torch
from src.audit import validate_patch
from src.utils import save_json
from scripts.workflow import matrix


def main():
    torch.set_num_threads(1)
    rows=[r for r in matrix(Namespace(artifacts='artifacts',include_strength=True)) if r['phase']=='ensemble_time']
    if len(rows)!=99:raise ValueError('Expected 99 matched-time conditions')
    records=[]
    for row in rows:
        obj=validate_patch(Path('artifacts/patches')/row['run_id']/'patch.pt',row,'artifacts')
        steps=obj['total_steps'];history=obj['validation_history']
        if steps%10!=0:raise ValueError('Budget stop was not at the ten-step boundary')
        if obj['forward_backward_calls']!=steps*(2 if row['strategy']=='RV' else 1):raise ValueError('Incorrect source forward/backward count')
        if obj['processed_training_images']!=steps*128:raise ValueError('Incorrect processed image count')
        if obj['selected_step']!=max(history,key=lambda r:r['asr'])['step']:raise ValueError('Patch selection differs from earliest validation maximum')
        records.append(dict(run_id=row['run_id'],strategy=row['strategy'],total_steps=steps,selected_step=obj['selected_step'],training_time_s=obj['wall_time_seconds'],
            budget_s=row['time_budget_s'],overshoot_s=obj['wall_time_seconds']-row['time_budget_s'],forward_backward_calls=obj['forward_backward_calls']))
    save_json('results/time_budget_audit.json',dict(status='complete',n_patches=len(records),checks=['patch/config/source/split/protocol hashes','pixels finite and bounded','final candidate validated','earliest maximum selection','actual budget reached','ten-update stop boundary','source calls and image counts'],records=records))
    print('99 matched-time patches passed integrity, selection, budget, and call-count checks',flush=True)


if __name__=='__main__':main()
