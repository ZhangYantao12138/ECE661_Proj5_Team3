"""Choose an evaluation batch only by mechanical equivalence on full validation."""
import json
from pathlib import Path
import numpy as np
import torch
from src.data import load_arrays,RawDataset
from src.evaluate import evaluate_patch
from src.model_bundle import load_model_bundle
from src.utils import digest,load_torch,save_json


def main():
    torch.set_num_threads(4)
    split=json.loads(Path('artifacts/splits/manifest.json').read_text())
    x,y,_,_=load_arrays('data');dataset=RawDataset(x,y,split['val'])
    patch_path=Path('artifacts/patches/basic__R__s7__t3__seed101__k1/patch.pt')
    patch=load_torch(patch_path)['patch_tensor']
    defenses=json.loads(Path('artifacts/defenses.json').read_text())['selected']
    pipelines=[('resnet20',None,patch,8),('vgg11_bn_cifar',None,patch,8)]+[('resnet20',d,patch,8) for d in defenses]+[('densenet_bc40_k12',None,None,1)]
    checks=[]
    for model,defense,pixels,draws in pipelines:
        bundle=load_model_bundle(Path('artifacts/models')/model/'best.pt','cuda')
        a,ma=evaluate_patch(bundle,pixels,dataset,seed=6611,draws=draws,defense=defense,target=3 if pixels is not None else None,batch_size=128)
        b,mb=evaluate_patch(bundle,pixels,dataset,seed=6611,draws=draws,defense=defense,target=3 if pixels is not None else None,batch_size=512)
        ia=np.lexsort((a['draw_id'],a['image_id']));ib=np.lexsort((b['draw_id'],b['image_id']))
        exact_fields=['image_id','true_label','draw_id','pred_clean','pred_patch','pred_defense_clean','angle','scale','cx','cy','defense_q','defense_top','defense_left','support_area','defended_support_area']
        exact=all(np.array_equal(a[k][ia],b[k][ib]) for k in exact_fields)
        area_close=all(np.allclose(a[k][ia],b[k][ib],rtol=0,atol=1e-5) for k in ['alpha_area','defended_alpha_area'])
        checks.append(dict(model=model,defense=defense,draws=draws,identical_predictions_and_conditions=bool(exact),alpha_area_equivalent=area_close,
            differing_patched_predictions=int(np.count_nonzero(a['pred_patch'][ia]!=b['pred_patch'][ib])),checkpoint_hash=bundle.checkpoint_hash))
        print(checks[-1],flush=True)
        del bundle
    selected=512 if all(r['identical_predictions_and_conditions'] and r['alpha_area_equivalent'] for r in checks) else 128
    save_json('artifacts/evaluation_batch.json',dict(batch_size=selected,reference_batch=128,candidate_batch=512,checks=checks,
        rule='Use 512 only when all validation predictions and random conditions match; otherwise retain 128.',
        patch_hash=digest(patch_path),heldout_attack_feedback_used=False,heldout_check='clean images only',
        scope='mechanical batching, not method/patch/model selection; full 5000 validation images'))


if __name__=='__main__':main()
