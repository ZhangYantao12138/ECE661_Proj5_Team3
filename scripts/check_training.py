"""Plan A.5: fit 32 real training images before launching formal training."""
import argparse
import time
import numpy as np
import torch
from src.data import load_arrays
from src.models import build_model
from src.model_bundle import ModelBundle
from src.utils import save_json, digest, environment
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=['resnet20','vgg11_bn_cifar','densenet_bc40_k12'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = Path('artifacts/splits/manifest.json')
    split = json.loads(manifest.read_text())
    images, labels, _, _ = load_arrays('data')
    # Stratified deterministic training-only subset, never validation/test.
    grouped = [[i for i in split['train'] if labels[i] == c] for c in range(10)]
    ids = np.asarray([i for group in grouped for i in group[:3]] + [grouped[0][3], grouped[1][3]])
    x = torch.tensor(images[ids], device=args.device).float()/255
    y = torch.tensor(labels[ids], device=args.device)
    report = {'split_hash': digest(manifest), 'image_ids': ids.tolist(), 'environment': environment(), 'models': []}
    for name in ([args.model] if args.model else ('resnet20', 'vgg11_bn_cifar', 'densenet_bc40_k12')):
        torch.manual_seed(17)
        bundle = ModelBundle(build_model(name), split['mean'], split['std'], name).to(args.device)
        optimizer = torch.optim.Adam(bundle.parameters(), lr=.001)
        bundle.train()
        initial = None
        started = time.perf_counter()
        # A finite diagnostic schedule, independent of the formal 200-epoch SGD run.
        for step in range(500):
            if step in (200,400):
                for group in optimizer.param_groups:group['lr']*=.1
            optimizer.zero_grad(set_to_none=True)
            logits = bundle(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(name + ': nonfinite small-subset loss')
            if initial is None:
                initial = float(loss.detach())
            loss.backward()
            optimizer.step()
        bundle.eval()
        with torch.no_grad():
            final_loss = float(torch.nn.functional.cross_entropy(bundle(x), y))
            accuracy = float((bundle(x).argmax(1) == y).float().mean())
        row = dict(model=name, parameters=sum(p.numel() for p in bundle.parameters()),
                   steps=500, optimizer='Adam',initial_lr=.001,initial_loss=initial, final_loss=final_loss, accuracy=accuracy,
                   elapsed_s=time.perf_counter()-started, debug_only=True)
        print(json.dumps(row), flush=True)
        report['models'].append(row)
        save_json('results/training_preflight.json', report)
        if accuracy < .95 or final_loss >= initial:
            raise RuntimeError(name + ': failed the 32-image fitting check')
        del bundle, optimizer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
