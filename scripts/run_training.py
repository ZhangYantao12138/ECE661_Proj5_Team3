"""Run all three full training jobs with durable logs and explicit stage status."""
import json
from pathlib import Path
import subprocess
import sys
import time
from src.utils import digest, save_json

ROOT = Path(__file__).resolve().parents[1]


def run(label, command):
    log_path = ROOT/'results/local_logs'/f'{label}-{time.strftime("%Y%m%d-%H%M%S")}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    save_json(ROOT/'results/training_status.json', dict(stage=label,status='running',command=command,log=str(log_path),started_unix=started))
    with log_path.open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=None)
        print(json.dumps(dict(stage=label,pid=process.pid,log=str(log_path))),flush=True)
        code=process.wait()
    save_json(ROOT/'results/training_status.json',dict(stage=label,status='completed' if code==0 else 'failed',return_code=code,log=str(log_path),elapsed_s=time.time()-started))
    if code:
        raise RuntimeError(f'{label} failed ({code}): {log_path}')


def main():
    check=ROOT/'results/training_preflight.json'
    checks=json.loads(check.read_text()) if check.exists() else {}
    if len(checks.get('models',[]))!=3 or any(row['accuracy']<.95 for row in checks.get('models',[])):
        run('fit32',[sys.executable,'-u','-m','scripts.check_training','--device','cuda'])
    for name in ('resnet20','vgg11_bn_cifar','densenet_bc40_k12'):
        path=ROOT/f'artifacts/models/{name}/training_complete.json'
        if path.exists():
            complete=json.loads(path.read_text())
            if complete['epochs_completed']==200 and complete['best_checkpoint_sha256']==digest(ROOT/f'artifacts/models/{name}/best.pt'):
                print('Verified completed model: '+name,flush=True)
                continue
        run(name,[sys.executable,'-u','-m','scripts.train_classifier','--model',name,'--device','cuda','--epochs','200','--resume'])
    save_json(ROOT/'results/training_status.json',dict(stage='all_three_classifiers',status='completed',epochs_each=200))


if __name__=='__main__':
    main()
