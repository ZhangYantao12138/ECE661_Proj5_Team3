"""Compact current status for the active experiment watch."""
import json
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from src.paths import workspace_root

ROOT=Path(__file__).resolve().parents[1]


def main():
    status=json.loads((ROOT/'results/execution_status.json').read_text(encoding='utf-8'))
    counts=Counter()
    for path in (ROOT/'artifacts/patches').glob('*/patch.pt'):
        if not Path(str(path)+'.sha256.json').exists():continue
        name=path.parent.name;counts['defense_strength' if name.startswith('strength__') else name.split('__')[0]]+=1
    raw_count=sum(1 for _ in (ROOT/'results/raw').glob('*.json')) if (ROOT/'results/raw').exists() else 0
    validation_count=sum(1 for _ in (ROOT/'results/validation_raw').glob('*.json')) if (ROOT/'results/validation_raw').exists() else 0
    record=dict(updated_utc=datetime.now(timezone.utc).isoformat(),stage=status['stage'],status=status['status'],
        main_patches=sum(v for k,v in counts.items() if k!='defense_strength'),strength_patches=counts['defense_strength'],
        by_phase=dict(counts),raw_evaluations=raw_count,full_validation_evaluations=validation_count,expected_main=561,expected_strength=24,expected_evaluations=1638)
    if status.get('log'):
        path=Path(status['log'])
        if path.exists():record['latest_log_line']=next((line for line in reversed(path.read_text(encoding='utf-8',errors='replace').splitlines()) if line.strip()),'')[-400:]
    if record.get('latest_log_line','').startswith('RUN '):
        run_id=record['latest_log_line'].split(' ',1)[1].strip()
        candidate=(ROOT/'artifacts/patches'/run_id/'resume.pt').resolve()
        if candidate.is_relative_to((ROOT/'artifacts/patches').resolve()) and candidate.exists():
            modified=datetime.fromtimestamp(candidate.stat().st_mtime,timezone.utc)
            record['active_checkpoint_updated_utc']=modified.isoformat()
            record['checkpoint_age_s']=round((datetime.now(timezone.utc)-modified).total_seconds(),1)
    dest=workspace_root()/'outputs/live_progress.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    with (ROOT/'results/progress_history.jsonl').open('a',encoding='utf-8') as history:
        history.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
