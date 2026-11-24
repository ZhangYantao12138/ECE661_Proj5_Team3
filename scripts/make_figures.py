"""Plot only measured formal records; synthetic plots are generated separately."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    path=Path('results/summary.csv')
    if not path.exists():raise RuntimeError('Run summarize after real evaluations first')
    rows=list(csv.DictReader(path.open()));out=Path('results/figures');out.mkdir(parents=True,exist_ok=True)
    for model in sorted(set(r.get('eval_model','') for r in rows)):
        subset=[r for r in rows if r.get('eval_model')==model and r.get('phase')=='basic' and r.get('target') in ('None','') and r.get('defense_eval')=="{'type': 'identity'}"]
        if subset:
            sizes=sorted({int(r['size']) for r in subset});means=[];stds=[]
            for s in sizes:
                vals=[float(r['uasr']) for r in subset if int(r['size'])==s and r.get('uasr') not in ('','None')]
                means.append(np.mean(vals));stds.append(np.std(vals,ddof=1) if len(vals)>1 else 0)
            fig,ax=plt.subplots();ax.errorbar(sizes,means,yerr=stds,marker='o');ax.set(xlabel='Nominal patch side (pixels)',ylabel='UASR',title=model);fig.savefig(out/f'{model}_size_uasr.png',dpi=180);plt.close(fig)
    print('Formal figures exported from measured CSV; missing conditions remain absent')
if __name__=='__main__':main()
