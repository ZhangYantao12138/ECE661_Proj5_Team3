"""Complete interrupted final controls and audits without rerunning optimizations."""
import json
from pathlib import Path
from argparse import Namespace
from datetime import datetime,timezone
from scripts.continue_experiments import execute,status,ROOT
from scripts.local_gpu import commands
from scripts.workflow import freeze_record
from src.utils import save_json


def main():
    args=Namespace(artifacts='artifacts',include_strength=True)
    frozen=json.loads((ROOT/'artifacts/freeze.json').read_text(encoding='utf-8'))
    if frozen!=freeze_record(args):raise RuntimeError('Frozen candidates changed')
    validation=json.loads((ROOT/'results/validation_audit.json').read_text(encoding='utf-8'))
    if not validation['complete']:raise RuntimeError('Full validation is incomplete')
    old=ROOT/'results/execution_status.json'
    if old.exists():
        save_json(ROOT/'results/final_evaluation_interruption.json',dict(previous_status=json.loads(old.read_text(encoding='utf-8')),
            recovered_utc=datetime.now(timezone.utc).isoformat(),action='Verify and reuse cached final conditions, finish remaining controls and audit.'))
    for index,command in enumerate(commands('final')[4:],start=5):
        execute('final_evaluation_'+str(index),command)
    status('measured_experiments','completed',remaining_work=['paired statistical analysis','review all figures','final report and Chinese guides','PDF visual verification','package and verify'])


if __name__=='__main__':main()
