"""Local CUDA preflight and resumable formal workflow; no remote access required."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['resnet20', 'vgg11_bn_cifar', 'densenet_bc40_k12']
PHASES = ['basic', 'ensemble_steps', 'ensemble_time', 'defense_eot_control', 'defense_adaptive','defense_strength']


def preflight():
    import torch
    from src.models import build_model
    from src.utils import save_json, environment
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check the selected Python, NVIDIA driver and CUDA-enabled PyTorch.')
    torch.set_num_threads(4)
    report = dict(environment(), executable=sys.executable,
                  gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                  memory_bytes=torch.cuda.get_device_properties(0).total_memory, checks=[])
    # Exercise actual model kernels, rather than relying on device enumeration.
    for name in MODELS:
        model = build_model(name).cuda().train()
        x = torch.rand(2, 3, 32, 32, device='cuda', requires_grad=True)
        loss = model(x).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
            raise RuntimeError('Nonfinite forward/backward: ' + name)
        report['checks'].append(dict(model=name, forward_backward='passed'))
        del model, x, loss
        torch.cuda.empty_cache()
    save_json(ROOT/'results/local_gpu_preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def commands(stage):
    workflow = [sys.executable, '-m', 'scripts.workflow']
    if stage == 'prepare':
        return [workflow + ['prepare']]
    if stage == 'train':
        return [[sys.executable,'-u','-m','scripts.run_training']]
    if stage == 'calibrate':
        return [[sys.executable,'-u','-m','scripts.develop_attacks']]+[workflow + [s, '--device', 'cuda','--include-strength'] for s in ['select-defenses', 'benchmark', 'matrix']]
    if stage == 'attack':
        return [workflow + ['optimize', '--phase', p, '--execute', '--resume', '--device', 'cuda','--include-strength'] for p in PHASES]
    if stage == 'final':
        return [workflow + ['audit','--include-strength'], workflow + ['freeze','--include-strength'],
                workflow + ['evaluate', '--final', '--device', 'cuda','--include-strength'],
                workflow + ['evaluate', '--final', '--device', 'cuda','--include-strength','--draws','32','--stability-only'],
                workflow + ['controls', '--final', '--device', 'cuda','--include-strength'],
                [sys.executable,'-u','-m','scripts.audit_results'],
                workflow + ['summarize'], [sys.executable, '-m', 'scripts.make_figures']]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'train', 'calibrate', 'attack', 'final', 'all'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stages = ['prepare', 'train', 'calibrate', 'attack', 'final'] if args.stage == 'all' else [args.stage]
    if args.stage == 'check':
        preflight()
        return
    if not args.dry_run and args.stage != 'prepare':
        preflight()
    logs = ROOT/'results/local_logs'
    logs.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        for cmd in commands(stage):
            print(subprocess.list2cmdline(cmd), flush=True)
            if args.dry_run:
                continue
            # A unique log per command survives failures and interrupted runs.
            path = logs/(time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns()) + '-' + stage + '.log')
            with path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace')
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise
            if code:
                raise RuntimeError(f'Command failed ({code}); see {path}')


if __name__ == '__main__':
    main()
