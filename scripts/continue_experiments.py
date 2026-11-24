"""One-shot continuation of authorized experiments, with persistent status and logs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
from src.utils import digest,save_json
from src.paths import workspace_root
from scripts.local_gpu import commands

ROOT=Path(__file__).resolve().parents[1]
OUT=workspace_root()/'outputs'


def status(stage,state,**details):
    value=dict(stage=stage,status=state,updated_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),**details)
    save_json(ROOT/'results/execution_status.json',value)
    save_json(OUT/'execution_status.json',value)
    print(json.dumps(value),flush=True)


def execute(stage,command):
    logfile=ROOT/'results/local_logs'/f'queue-{stage}-{time.strftime("%Y%m%d-%H%M%S")}.log'
    logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status(stage,'running',pid=process.pid,command=command,log=str(logfile))
        code=process.wait()
    if code:
        status(stage,'failed',return_code=code,log=str(logfile))
        raise RuntimeError(f'{stage} failed ({code}); see {logfile}')
    status(stage,'completed',log=str(logfile))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-primary',action='store_true')
    args=parser.parse_args()
    if args.wait_primary:
        status('resnet20_primary_training','waiting_for_existing_run',history=str(ROOT/'artifacts/models/resnet20/history.json'))
        while not (ROOT/'artifacts/models/resnet20/training_complete.json').exists():
            history=ROOT/'artifacts/models/resnet20/history.json'
            if history.exists() and time.time()-history.stat().st_mtime>1800:
                status('resnet20_primary_training','failed',reason='Existing primary training has no new epoch for 30 minutes')
                raise RuntimeError('Existing primary training stopped; inspect its log before resuming')
            time.sleep(10)
    execute('remaining_classifier_training',[sys.executable,'-u','-m','scripts.run_training'])
    # These stages are only run once for the same frozen source models.
    checkpoints={name:digest(ROOT/f'artifacts/models/{name}/best.pt') for name in ('resnet20','vgg11_bn_cifar','densenet_bc40_k12')}
    if (ROOT/'artifacts/development.json').exists():
        record=json.loads((ROOT/'artifacts/development.json').read_text())
        if record['checkpoint_hash']!=checkpoints['resnet20']:raise RuntimeError('Development source changed')
    else:execute('attack_development',[sys.executable,'-u','-m','scripts.develop_attacks'])
    if (ROOT/'artifacts/defenses.json').exists():
        record=json.loads((ROOT/'artifacts/defenses.json').read_text())
        if record['checkpoint_hash']!=checkpoints['resnet20']:raise RuntimeError('Defense-selection source changed')
    else:execute('select_defenses',[sys.executable,'-u','-m','scripts.workflow','select-defenses','--device','cuda'])
    if not (ROOT/'artifacts/time_budget.json').exists():
        execute('time_budget',[sys.executable,'-u','-m','scripts.workflow','benchmark','--device','cuda'])
    execute('matrix',[sys.executable,'-u','-m','scripts.workflow','matrix','--include-strength'])
    for index,command in enumerate(commands('attack')):
        execute('optimization_'+str(index+1),command)
    for index,command in enumerate(commands('final')):
        execute('final_evaluation_'+str(index+1),command)
    status('measured_experiments','completed',remaining_work=['paired statistical analysis','review all scientific figures','write final report and four Chinese guides','render and visually verify PDFs','independent A/B/C/D archives'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        path=ROOT/'results/execution_status.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        status(previous.get('stage','initialization'),'failed',reason=str(exc),previous_status=previous)
        raise
